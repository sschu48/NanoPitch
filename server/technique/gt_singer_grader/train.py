"""Train an independent GT Singer technique grader."""

from __future__ import annotations

import argparse
import atexit
import errno
import json
import math
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback when tensorboard is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def add_scalar(self, *args: object, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            pass

from .constants import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_N_MELS,
    FAMILY_NAMES,
    TECHNIQUE_KEYS,
)
from .data import (
    GTSingerTechniqueDataset,
    ManifestTechniqueDataset,
    read_training_manifest,
    scan_gt_singer,
    split_records,
    summarize_manifest_records,
    summarize_records,
    write_manifest,
)
from .manifest import require_non_empty_records, trainability_reason, write_jsonl
from .model import TechniqueGraderModel
from .plan_training import plan_match_errors
from .run_metadata import collect_run_metadata, file_metadata
from .split_health import require_split_coverage, require_split_family_compatibility


def write_json(path: str, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str, payload: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def require_trainable_manifest(records: list[dict[str, object]], *, source: str) -> None:
    bad_records: list[str] = []
    for index, record in enumerate(records, start=1):
        reason = trainability_reason(record)
        if reason == "missing_family":
            bad_records.append(f"line {index}: manifest record has no family label")
            continue
        if reason != "trainable":
            record_id = record.get("recording_id") or record.get("stem") or f"line {index}"
            bad_records.append(f"{record_id}: {reason}")

    if bad_records:
        preview = "\n".join(f"  - {item}" for item in bad_records[:10])
        extra = "" if len(bad_records) <= 10 else f"\n  ... and {len(bad_records) - 10} more"
        raise SystemExit(
            f"{source} contains records that cannot be used for supervised training yet.\n"
            "Move `none`/`unclear` clips to an evaluation manifest, or relabel them as a trainable family.\n"
            f"{preview}{extra}"
        )


def require_training_records(records: list[object], *, source: str, purpose: str) -> None:
    try:
        require_non_empty_records(records, source=source, purpose=purpose)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def require_primary_split_coverage(records: list[object], *, source: str, purpose: str) -> None:
    try:
        require_split_coverage(records, source=source, purpose=purpose)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def require_primary_split_compatibility(train_records: list[object], val_records: list[object], *, source: str) -> None:
    try:
        require_split_family_compatibility(train_records, val_records, source=source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GT Singer grading model")
    parser.add_argument("--dataset-root", default=None, help="Path to the downloaded GT Singer English tree")
    parser.add_argument("--train-manifest", default=None, help="Explicit training JSONL manifest")
    parser.add_argument("--val-manifest", default=None, help="Explicit validation JSONL manifest")
    parser.add_argument(
        "--extra-train-manifest",
        action="append",
        default=[],
        help="Additional weak/supplemental training manifest. May be passed more than once.",
    )
    parser.add_argument("--language", default="English")
    parser.add_argument("--output-dir", default="./runs/default")
    parser.add_argument("--training-plan", default=None, help="Path to a gt_singer_grader.plan_training JSON report")
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-batch progress bars; epoch summaries and JSON artifacts are still written.",
    )
    return parser.parse_args()


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def acquire_output_lock(output_dir: str) -> str:
    """Prevent concurrent trainers from appending to the same run artifacts."""
    lock_path = os.path.join(output_dir, ".train.lock")
    payload = {"pid": os.getpid()}
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:
            existing = {}
        existing_pid = existing.get("pid")
        if isinstance(existing_pid, int) and not process_is_running(existing_pid):
            os.remove(lock_path)
            return acquire_output_lock(output_dir)
        raise SystemExit(
            f"another training process appears to be using {output_dir}; "
            f"remove {lock_path} only after confirming no trainer is running"
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")

    def cleanup() -> None:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    atexit.register(cleanup)
    return lock_path


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


def require_matching_training_plan(path: str, args: argparse.Namespace) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    errors = plan_match_errors(plan, args)
    if errors:
        preview = "\n".join(f"  - {error}" for error in errors)
        raise SystemExit(f"training plan does not match this train command:\n{preview}")


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

    disable_progress = bool(getattr(args, "quiet", False)) or not sys.stderr.isatty()
    iterator = tqdm(loader, disable=disable_progress, leave=False)
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


def reset_fresh_run_artifacts(output_dir: str) -> None:
    """Remove generated state that would make a fresh non-resume run ambiguous."""
    for filename in ("metrics_history.jsonl", "best_metrics.json"):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tensorboard_dir = os.path.join(output_dir, "tb")
    if os.path.isdir(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)


def prune_metrics_history_for_resume(path: str, *, max_epoch: int) -> None:
    """Keep one metrics record per completed epoch before appending resumed epochs."""
    if not os.path.isfile(path):
        return

    by_epoch: dict[int, dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = record.get("epoch")
            if not isinstance(epoch, int) or epoch < 1 or epoch > max_epoch:
                continue
            by_epoch[epoch] = record

    with open(path, "w", encoding="utf-8") as handle:
        for epoch in sorted(by_epoch):
            handle.write(json.dumps(by_epoch[epoch], sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.training_plan:
        require_matching_training_plan(args.training_plan, args)
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    acquire_output_lock(output_dir)
    if args.resume:
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    else:
        reset_fresh_run_artifacts(output_dir)
    train_manifest_output = os.path.join(output_dir, "train_manifest.jsonl")
    val_manifest_output = os.path.join(output_dir, "val_manifest.jsonl")
    run_artifacts: dict[str, object] = {
        "train_manifest": file_metadata(train_manifest_output),
        "val_manifest": file_metadata(val_manifest_output),
    }
    if args.training_plan:
        run_artifacts["training_plan"] = file_metadata(args.training_plan)

    if args.train_manifest or args.val_manifest:
        if not args.train_manifest or not args.val_manifest:
            raise SystemExit("--train-manifest and --val-manifest must be provided together")
        train_manifest_records = read_training_manifest(args.train_manifest)
        val_manifest_records = read_training_manifest(args.val_manifest)
        require_training_records(train_manifest_records, source=args.train_manifest, purpose="training")
        require_training_records(val_manifest_records, source=args.val_manifest, purpose="validation")
        require_trainable_manifest(train_manifest_records, source=args.train_manifest)
        require_trainable_manifest(val_manifest_records, source=args.val_manifest)
        require_primary_split_coverage(train_manifest_records, source=args.train_manifest, purpose="training")
        require_primary_split_coverage(val_manifest_records, source=args.val_manifest, purpose="validation")
        require_primary_split_compatibility(train_manifest_records, val_manifest_records, source="manifest")
        train_summary = summarize_manifest_records(train_manifest_records)
        val_summary = summarize_manifest_records(val_manifest_records)
        train_dataset = ManifestTechniqueDataset(
            train_manifest_records,
            n_mels=args.n_mels,
            max_seconds=args.max_seconds,
            training=True,
            audio_augmentation=args.user_audio_augmentation,
        )
        val_dataset = ManifestTechniqueDataset(
            val_manifest_records,
            n_mels=args.n_mels,
            max_seconds=args.max_seconds,
            training=False,
        )
        write_jsonl(train_manifest_output, train_manifest_records)
        write_jsonl(val_manifest_output, val_manifest_records)
        train_source_output = os.path.join(output_dir, "train_manifest_source.json")
        val_source_output = os.path.join(output_dir, "val_manifest_source.json")
        write_json(train_source_output, {"path": args.train_manifest})
        write_json(val_source_output, {"path": args.val_manifest})
        run_artifacts["train_manifest_source"] = file_metadata(train_source_output)
        run_artifacts["val_manifest_source"] = file_metadata(val_source_output)
    else:
        if not args.dataset_root:
            raise SystemExit("provide --dataset-root or both --train-manifest and --val-manifest")
        records = scan_gt_singer(args.dataset_root, language=args.language, include_speech=args.include_speech)
        train_records, val_records = split_records(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed,
            group_by=args.split_group,
        )
        require_training_records(train_records, source=args.dataset_root, purpose="training")
        require_training_records(val_records, source=args.dataset_root, purpose="validation")
        require_primary_split_coverage(train_records, source=args.dataset_root, purpose="training")
        require_primary_split_coverage(val_records, source=args.dataset_root, purpose="validation")
        require_primary_split_compatibility(train_records, val_records, source="GT Singer")

        write_manifest(train_manifest_output, train_records)
        write_manifest(val_manifest_output, val_records)

        train_summary = summarize_records(train_records)
        val_summary = summarize_records(val_records)
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

    extra_summaries = {}
    extra_manifest_artifacts = {}
    if args.extra_train_manifest:
        datasets = [train_dataset]
        for manifest_path in args.extra_train_manifest:
            extra_records = read_training_manifest(manifest_path)
            require_training_records(extra_records, source=manifest_path, purpose="extra training")
            require_trainable_manifest(extra_records, source=manifest_path)
            extra_summaries[manifest_path] = summarize_manifest_records(extra_records)
            safe_name = f"extra_train_manifest_{len(extra_summaries):02d}.jsonl"
            extra_manifest_output = os.path.join(output_dir, safe_name)
            write_jsonl(extra_manifest_output, extra_records)
            extra_manifest_artifacts[manifest_path] = file_metadata(extra_manifest_output)
            datasets.append(
                ManifestTechniqueDataset(
                    extra_records,
                    n_mels=args.n_mels,
                    max_seconds=args.max_seconds,
                    training=True,
                    audio_augmentation=args.user_audio_augmentation,
                )
            )
        train_dataset = ConcatDataset(datasets)

    run_artifacts["train_manifest"] = file_metadata(train_manifest_output)
    run_artifacts["val_manifest"] = file_metadata(val_manifest_output)
    if extra_manifest_artifacts:
        run_artifacts["extra_train_manifests"] = extra_manifest_artifacts

    print("Train split:", json.dumps(train_summary, indent=2))
    print("Val split:", json.dumps(val_summary, indent=2))
    if extra_summaries:
        print("Extra train manifests:", json.dumps(extra_summaries, indent=2))

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

    run_config = {
        "train_args": vars(args),
        "model_kwargs": model_kwargs,
        "split": {
            "group_by": args.split_group,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "train_summary": train_summary,
            "val_summary": val_summary,
            "extra_train_summaries": extra_summaries,
        },
        "families": FAMILY_NAMES,
        "techniques": TECHNIQUE_KEYS,
        "environment": collect_run_metadata(os.getcwd()),
        "artifacts": run_artifacts,
    }
    write_json(os.path.join(output_dir, "run_config.json"), run_config)
    metrics_history_path = os.path.join(output_dir, "metrics_history.jsonl")

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
        prune_metrics_history_for_resume(metrics_history_path, max_epoch=start_epoch - 1)
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

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": val_metrics["clip_acc"] + val_metrics["tech_macro_f1"],
        }
        append_jsonl(metrics_history_path, epoch_record)

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
            write_json(
                os.path.join(output_dir, "best_metrics.json"),
                {
                    "epoch": epoch,
                    "selection_score": best_val,
                    "val": val_metrics,
                    "train": train_metrics,
                    "checkpoint": "checkpoints/best.pth",
                },
            )

    writer.close()


if __name__ == "__main__":
    main()
