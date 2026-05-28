"""Train an independent GT Singer technique grader."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .constants import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_N_MELS,
    FAMILY_NAMES,
    TECHNIQUE_KEYS,
)
from .data import GTSingerTechniqueDataset, scan_gt_singer, split_records, summarize_records, write_manifest
from .model import TechniqueGraderModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GT Singer grading model")
    parser.add_argument("--dataset-root", required=True, help="Path to the downloaded GT Singer English tree")
    parser.add_argument("--language", default="English")
    parser.add_argument("--output-dir", default="./runs/default")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to continue training from")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--split-group",
        choices=("song", "speaker"),
        default="song",
        help="Validation grouping: song keeps paired GT Singer takes together; speaker holds out whole singers.",
    )
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--conv-size", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip-loss-weight", type=float, default=1.0)
    parser.add_argument("--vad-loss-weight", type=float, default=0.3)
    parser.add_argument("--tech-loss-weight", type=float, default=0.7)
    parser.add_argument(
        "--user-audio-augmentation",
        action="store_true",
        help="Apply lightweight noise/gain/room/mic augmentation to GT Singer audio during training.",
    )
    parser.add_argument("--include-speech", action="store_true", help="Include Paired_Speech_Group in the scan")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def technique_macro_f1(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    scores = []
    predictions = probs >= 0.5
    truth = targets > 0.5
    active = mask > 0.5
    for index in range(probs.size(-1)):
        pred = predictions[..., index] & active
        gold = truth[..., index] & active
        tp = torch.sum(pred & gold).item()
        fp = torch.sum(pred & ~gold).item()
        fn = torch.sum(~pred & gold).item()
        denom = 2 * tp + fp + fn
        if denom > 0:
            scores.append((2 * tp) / denom)
    return float(sum(scores) / len(scores)) if scores else 0.0


def run_epoch(
    model: TechniqueGraderModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    clip_ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    totals: dict[str, float] = defaultdict(float)
    clip_predictions = []
    clip_targets = []
    technique_probs = []
    technique_targets = []
    technique_masks = []

    iterator = tqdm(loader, disable=False, leave=False)
    for batch in iterator:
        mel = batch["mel"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        vad_target = batch["vad_target"].to(device)
        technique_target = batch["technique_target"].to(device)
        clip_label = batch["clip_label"].to(device)

        outputs = model(
            mel,
            frame_mask=frame_mask,
            voice_activity_mask=vad_target if is_training else None,
        )

        clip_loss = clip_ce(outputs["clip_logits"], clip_label)

        vad_loss_raw = bce(outputs["vad_logits"], vad_target)
        vad_loss = (vad_loss_raw * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)

        technique_mask = frame_mask * vad_target
        tech_loss_raw = bce(outputs["technique_logits"], technique_target)
        tech_loss = (
            tech_loss_raw * technique_mask.unsqueeze(-1)
        ).sum() / (technique_mask.sum().clamp_min(1.0) * tech_loss_raw.size(-1))

        loss = (
            args.clip_loss_weight * clip_loss
            + args.vad_loss_weight * vad_loss
            + args.tech_loss_weight * tech_loss
        )

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        batch_size = mel.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["clip_loss"] += clip_loss.item() * batch_size
        totals["vad_loss"] += vad_loss.item() * batch_size
        totals["tech_loss"] += tech_loss.item() * batch_size
        totals["examples"] += batch_size

        clip_predictions.append(outputs["clip_logits"].argmax(dim=-1).detach().cpu())
        clip_targets.append(clip_label.detach().cpu())
        technique_probs.append(torch.sigmoid(outputs["technique_logits"]).detach().cpu())
        technique_targets.append(technique_target.detach().cpu())
        technique_masks.append((technique_mask > 0.5).detach().cpu())

        iterator.set_postfix(loss=f"{loss.item():.4f}")

    if totals["examples"] == 0:
        return {"loss": math.nan}

    clip_pred_tensor = torch.cat(clip_predictions)
    clip_target_tensor = torch.cat(clip_targets)
    technique_prob_tensor = torch.cat(technique_probs)
    technique_target_tensor = torch.cat(technique_targets)
    technique_mask_tensor = torch.cat(technique_masks)

    metrics = {
        "loss": totals["loss"] / totals["examples"],
        "clip_loss": totals["clip_loss"] / totals["examples"],
        "vad_loss": totals["vad_loss"] / totals["examples"],
        "tech_loss": totals["tech_loss"] / totals["examples"],
        "clip_acc": float((clip_pred_tensor == clip_target_tensor).float().mean().item()),
        "tech_macro_f1": technique_macro_f1(technique_prob_tensor, technique_target_tensor, technique_mask_tensor),
    }
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    records = scan_gt_singer(args.dataset_root, language=args.language, include_speech=args.include_speech)
    train_records, val_records = split_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_by=args.split_group,
    )

    write_manifest(os.path.join(output_dir, "train_manifest.jsonl"), train_records)
    write_manifest(os.path.join(output_dir, "val_manifest.jsonl"), val_records)

    print("Train split:", json.dumps(summarize_records(train_records), indent=2))
    print("Val split:", json.dumps(summarize_records(val_records), indent=2))

    train_dataset = GTSingerTechniqueDataset(
        train_records,
        n_mels=args.n_mels,
        max_seconds=args.max_seconds,
        training=True,
        audio_augmentation=args.user_audio_augmentation,
    )
    val_dataset = GTSingerTechniqueDataset(
        val_records,
        n_mels=args.n_mels,
        max_seconds=args.max_seconds,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")

    model_kwargs = {
        "n_mels": args.n_mels,
        "conv_size": args.conv_size,
        "hidden_size": args.hidden_size,
        "gru_layers": args.gru_layers,
        "dropout": args.dropout,
    }
    if resume_checkpoint is not None:
        model_kwargs = resume_checkpoint.get("model_kwargs", model_kwargs)

    model = TechniqueGraderModel.from_config(model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    start_epoch = 1
    best_val = -float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        previous_metrics = resume_checkpoint.get("val_metrics") or {}
        best_val = float(previous_metrics.get("clip_acc", 0.0)) + float(previous_metrics.get("tech_macro_f1", 0.0))
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        print(f"Resuming from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_loader, None, device, args)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_clip_acc={val_metrics['clip_acc']:.4f} "
            f"val_tech_f1={val_metrics['tech_macro_f1']:.4f}"
        )

        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                writer.add_scalar(f"{split_name}/{key}", value, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_kwargs": model_kwargs,
            "train_args": vars(args),
            "family_names": FAMILY_NAMES,
            "technique_keys": TECHNIQUE_KEYS,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, os.path.join(output_dir, "checkpoints", f"epoch_{epoch:03d}.pth"))

        score = val_metrics["clip_acc"] + val_metrics["tech_macro_f1"]
        if score > best_val:
            best_val = score
            torch.save(checkpoint, os.path.join(output_dir, "checkpoints", "best.pth"))

    writer.close()


if __name__ == "__main__":
    main()
