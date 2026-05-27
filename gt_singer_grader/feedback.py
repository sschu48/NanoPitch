"""Inference-time smoothing and feedback generation."""

from __future__ import annotations

import json

import numpy as np
import torch

from .constants import DEFAULT_ONSET_PENALTY, FAMILY_NAMES, PRIMARY_FAMILY_TO_TECHNIQUES, TECHNIQUE_KEYS


def smooth_binary_predictions(logits: torch.Tensor, onset_penalty: float = DEFAULT_ONSET_PENALTY) -> np.ndarray:
    """Binary Viterbi smoothing for voiced/unvoiced decisions."""
    probs = torch.sigmoid(logits.detach()).cpu().numpy().astype(np.float64)
    if probs.ndim != 1:
        raise ValueError("expected a 1D logits tensor for VAD smoothing")
    if probs.size == 0:
        return np.zeros(0, dtype=np.float32)

    obs = np.stack(
        [
            np.log1p(-np.clip(probs, 1e-5, 1.0 - 1e-5)),
            np.log(np.clip(probs, 1e-5, 1.0 - 1e-5)),
        ],
        axis=1,
    )

    scores = np.zeros((probs.size, 2), dtype=np.float64)
    backpointers = np.zeros((probs.size, 2), dtype=np.int64)
    scores[0] = obs[0]

    for frame in range(1, probs.size):
        for state in range(2):
            stay = scores[frame - 1, state]
            switch = scores[frame - 1, 1 - state] - onset_penalty
            if stay >= switch:
                scores[frame, state] = stay + obs[frame, state]
                backpointers[frame, state] = state
            else:
                scores[frame, state] = switch + obs[frame, state]
                backpointers[frame, state] = 1 - state

    path = np.zeros(probs.size, dtype=np.int64)
    path[-1] = int(np.argmax(scores[-1]))
    for frame in range(probs.size - 2, -1, -1):
        path[frame] = backpointers[frame + 1, path[frame + 1]]
    return path.astype(np.float32)


def _family_phrase(family: str) -> str:
    return family.replace("_", " ")


def _format_feedback(target_family: str, target_strength: float, off_target_strength: float, voiced_ratio: float) -> str:
    if voiced_ratio < 0.15:
        return "Too little voiced singing was detected to grade this take reliably."

    if target_family == "control":
        if off_target_strength > 0.55:
            return "The take carries a strong stylistic color. It does not sound neutral enough for a control sample yet."
        if off_target_strength > 0.35:
            return "The take is mostly controlled, but some technique emphasis is still leaking through."
        return "The take stays fairly neutral and controlled, which matches a control-style delivery."

    if target_strength < 0.30:
        return f"The {_family_phrase(target_family)} cue is weak right now. Lean further into that technique."
    if off_target_strength > target_strength:
        return f"Another technique is reading stronger than {_family_phrase(target_family)}. The take needs a cleaner emphasis."
    if target_strength < 0.55:
        return f"The {_family_phrase(target_family)} profile is present, but it still has room to become more obvious."
    return f"The {_family_phrase(target_family)} profile is coming through clearly."


def summarize_prediction(
    outputs: dict[str, torch.Tensor],
    *,
    target_family: str | None = None,
    onset_penalty: float = DEFAULT_ONSET_PENALTY,
) -> dict[str, object]:
    """Turn raw model outputs into a grading summary."""
    family_logits = outputs["clip_logits"].detach().cpu().squeeze(0)
    vad_logits = outputs["vad_logits"].detach().cpu().squeeze(0)
    technique_logits = outputs["technique_logits"].detach().cpu().squeeze(0)

    family_probs = torch.softmax(family_logits, dim=-1).numpy()
    technique_probs = torch.sigmoid(technique_logits).numpy()
    voiced_mask = smooth_binary_predictions(vad_logits, onset_penalty=onset_penalty)
    if voiced_mask.sum() < 3:
        voiced_mask = (torch.sigmoid(vad_logits) > 0.5).numpy().astype(np.float32)
    if voiced_mask.sum() < 1:
        voiced_mask = np.ones_like(voiced_mask, dtype=np.float32)

    weighted_scores = (technique_probs * voiced_mask[:, None]).sum(axis=0) / np.clip(voiced_mask.sum(), 1.0, None)
    technique_scores = {name: float(weighted_scores[index]) for index, name in enumerate(TECHNIQUE_KEYS)}
    family_score_map = {name: float(family_probs[index]) for index, name in enumerate(FAMILY_NAMES)}
    ranked_families = sorted(family_score_map.items(), key=lambda item: item[1], reverse=True)
    detected_family, detected_confidence = ranked_families[0]
    runner_up_family, runner_up_confidence = ranked_families[1] if len(ranked_families) > 1 else ranked_families[0]
    voiced_ratio = float(np.mean(voiced_mask))

    summary: dict[str, object] = {
        "detected_family": detected_family,
        "detected_confidence": float(detected_confidence),
        "runner_up_family": runner_up_family,
        "runner_up_confidence": float(runner_up_confidence),
        "family_margin": float(detected_confidence - runner_up_confidence),
        "family_probabilities": family_score_map,
        "technique_scores": technique_scores,
        "voiced_ratio": voiced_ratio,
    }

    if target_family is not None:
        if target_family not in PRIMARY_FAMILY_TO_TECHNIQUES:
            raise ValueError(f"unknown target family: {target_family}")

        target_keys = PRIMARY_FAMILY_TO_TECHNIQUES[target_family]
        if target_family == "control":
            target_strength = 1.0 - max(technique_scores.values())
            off_target_strength = max(technique_scores.values())
        else:
            target_strength = float(np.mean([technique_scores[key] for key in target_keys]))
            off_target_strength = float(
                np.mean([score for key, score in technique_scores.items() if key not in set(target_keys)])
            )

        family_support = family_score_map.get(target_family, 0.0)
        grade = 100.0 * np.clip(0.45 * family_support + 0.55 * target_strength - 0.25 * off_target_strength, 0.0, 1.0)
        feedback = _format_feedback(target_family, target_strength, off_target_strength, voiced_ratio)
        summary.update(
            {
                "target_family": target_family,
                "grade": float(grade),
                "target_strength": float(target_strength),
                "off_target_strength": float(off_target_strength),
                "feedback": feedback,
            }
        )

    return summary


def build_demo_assessment(summary: dict[str, object]) -> dict[str, object]:
    """Turn a prediction summary into a demo-friendly verdict."""
    target_family = summary.get("target_family")
    detected_family = str(summary["detected_family"])
    detected_confidence = float(summary.get("detected_confidence", 0.0))
    runner_up_family = str(summary.get("runner_up_family", detected_family))
    family_margin = float(summary.get("family_margin", 0.0))
    voiced_ratio = float(summary.get("voiced_ratio", 0.0))

    assessment: dict[str, object] = {
        "status": "detected_only",
        "headline": f"Detected Technique: {_family_phrase(detected_family).title()}",
        "badge": "Technique Detected",
        "detected_family_display": _family_phrase(detected_family),
        "runner_up_family_display": _family_phrase(runner_up_family),
        "confidence": detected_confidence,
        "confidence_percent": float(100.0 * detected_confidence),
        "family_margin": family_margin,
        "voiced_ratio_percent": float(100.0 * voiced_ratio),
        "feedback": str(summary.get("feedback", "")),
        "target_family_display": _family_phrase(str(target_family)) if target_family else None,
    }

    if voiced_ratio < 0.15:
        assessment.update(
            {
                "status": "not_enough_voice",
                "badge": "Not Enough Singing",
                "headline": "Not enough voiced singing to judge the take",
                "feedback": "Sing a longer, clearer phrase so the model has enough voiced audio to judge technique.",
            }
        )
        return assessment

    if detected_confidence < 0.35 and family_margin < 0.08:
        assessment.update(
            {
                "status": "uncertain",
                "badge": "Uncertain",
                "headline": "The model is not confident about the dominant technique",
                "feedback": "Try a cleaner, more focused take or choose the intended technique below for a targeted verdict.",
            }
        )
        return assessment

    if target_family is None:
        assessment["feedback"] = (
            f"The take reads most strongly as {_family_phrase(detected_family)}. "
            "Choose an intended technique in the demo to get a well-done or needs-work verdict."
        )
        return assessment

    grade = float(summary.get("grade", 0.0))
    target_strength = float(summary.get("target_strength", 0.0))
    off_target_strength = float(summary.get("off_target_strength", 0.0))
    target_family = str(target_family)
    matches_target = detected_family == target_family

    assessment.update(
        {
            "grade": grade,
            "target_strength": target_strength,
            "off_target_strength": off_target_strength,
            "matches_target": matches_target,
        }
    )

    if detected_confidence < 0.45 and family_margin < 0.12:
        assessment.update(
            {
                "status": "uncertain",
                "badge": "Uncertain",
                "headline": f"The model is unsure whether this is strong {_family_phrase(target_family)}",
            }
        )
        return assessment

    if grade >= 72.0 and target_strength >= 0.5 and (matches_target or target_strength >= off_target_strength):
        assessment.update(
            {
                "status": "well_done",
                "badge": "Well Done",
                "headline": f"{_family_phrase(target_family).title()} came through clearly",
            }
        )
        return assessment

    if grade >= 55.0 and target_strength >= 0.35:
        assessment.update(
            {
                "status": "developing",
                "badge": "Developing",
                "headline": f"{_family_phrase(target_family).title()} is present but still developing",
            }
        )
        return assessment

    assessment.update(
        {
            "status": "needs_work",
            "badge": "Needs Work",
            "headline": f"{_family_phrase(target_family).title()} is not coming through strongly enough yet",
        }
    )
    return assessment


def summary_to_json(summary: dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
