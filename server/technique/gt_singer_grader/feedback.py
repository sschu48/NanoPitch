"""Inference-time smoothing and feedback generation."""

from __future__ import annotations

import json

import numpy as np
import torch

from .constants import DEFAULT_ONSET_PENALTY, FAMILY_NAMES, FRAME_HOP_SECONDS, PRIMARY_FAMILY_TO_TECHNIQUES, TECHNIQUE_KEYS


MIN_VOICED_RATIO = 0.15
MIN_TECHNIQUE_SCORE = 0.30


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


def _technique_phrase(technique: str) -> str:
    if technique == "mix":
        return "mixed voice"
    return _family_phrase(technique)


def _rank_scores(scores: dict[str, float], *, label_key: str) -> list[dict[str, object]]:
    return [
        {label_key: name, "score": float(score)}
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _dominant_techniques(technique_scores: dict[str, float]) -> list[dict[str, object]]:
    ranked = _rank_scores(technique_scores, label_key="technique")
    return [item for item in ranked if float(item["score"]) >= MIN_TECHNIQUE_SCORE][:3]


def _technique_timeline(
    technique_probs: np.ndarray,
    voiced_mask: np.ndarray,
    *,
    hop_seconds: float = FRAME_HOP_SECONDS,
    window_seconds: float = 1.0,
    max_windows: int = 120,
) -> list[dict[str, object]]:
    """Aggregate frame probabilities into coarse UI-friendly windows."""
    if technique_probs.size == 0:
        return []

    window_frames = max(1, int(round(window_seconds / hop_seconds)))
    timeline: list[dict[str, object]] = []
    for left in range(0, technique_probs.shape[0], window_frames):
        right = min(technique_probs.shape[0], left + window_frames)
        window_probs = technique_probs[left:right]
        window_voiced = voiced_mask[left:right]
        voiced_count = float(window_voiced.sum())
        voiced_ratio = float(np.mean(window_voiced)) if window_voiced.size else 0.0

        if voiced_count >= 1.0:
            scores_np = (window_probs * window_voiced[:, None]).sum(axis=0) / voiced_count
        else:
            scores_np = window_probs.mean(axis=0)

        scores = {name: float(scores_np[index]) for index, name in enumerate(TECHNIQUE_KEYS)}
        top = max(scores.items(), key=lambda item: item[1])
        technique = top[0] if voiced_ratio >= MIN_VOICED_RATIO and top[1] >= MIN_TECHNIQUE_SCORE else "none"

        timeline.append(
            {
                "start_s": round(left * hop_seconds, 3),
                "end_s": round(right * hop_seconds, 3),
                "technique": technique,
                "score": float(top[1]),
                "voiced_ratio": voiced_ratio,
                "scores": scores,
            }
        )

    if len(timeline) <= max_windows:
        return timeline

    stride = int(np.ceil(len(timeline) / max_windows))
    return timeline[::stride]


def _detection_status(
    *,
    detected_confidence: float,
    family_margin: float,
    primary_score: float,
    voiced_ratio: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if voiced_ratio < MIN_VOICED_RATIO:
        reasons.append("not_enough_voiced_singing")
        return "not_enough_voice", reasons

    if primary_score < MIN_TECHNIQUE_SCORE and detected_confidence < 0.35:
        reasons.append("no_technique_score_cleared_threshold")
        return "no_clear_technique", reasons

    if detected_confidence < 0.35 and family_margin < 0.08:
        reasons.append("low_family_confidence")
    if primary_score < MIN_TECHNIQUE_SCORE:
        reasons.append("low_frame_level_technique_strength")

    if reasons:
        return "uncertain", reasons
    return "detected", reasons


def _format_feedback(target_family: str, target_strength: float, off_target_strength: float, voiced_ratio: float) -> str:
    if voiced_ratio < MIN_VOICED_RATIO:
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
    ranked_techniques = _rank_scores(technique_scores, label_key="technique")
    primary_technique = str(ranked_techniques[0]["technique"]) if ranked_techniques else None
    primary_technique_score = float(ranked_techniques[0]["score"]) if ranked_techniques else 0.0
    dominant_techniques = _dominant_techniques(technique_scores)
    family_margin = float(detected_confidence - runner_up_confidence)
    detection_status, detection_reasons = _detection_status(
        detected_confidence=float(detected_confidence),
        family_margin=family_margin,
        primary_score=primary_technique_score,
        voiced_ratio=voiced_ratio,
    )

    summary: dict[str, object] = {
        "detected_family": detected_family,
        "detected_confidence": float(detected_confidence),
        "runner_up_family": runner_up_family,
        "runner_up_confidence": float(runner_up_confidence),
        "family_margin": family_margin,
        "family_probabilities": family_score_map,
        "technique_scores": technique_scores,
        "ranked_techniques": ranked_techniques,
        "dominant_techniques": dominant_techniques,
        "primary_technique": primary_technique,
        "primary_technique_score": primary_technique_score,
        "detection_status": detection_status,
        "detection_reasons": detection_reasons,
        "technique_timeline": _technique_timeline(technique_probs, voiced_mask),
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
    primary_technique = summary.get("primary_technique")
    detected_confidence = float(summary.get("detected_confidence", 0.0))
    runner_up_family = str(summary.get("runner_up_family", detected_family))
    family_margin = float(summary.get("family_margin", 0.0))
    voiced_ratio = float(summary.get("voiced_ratio", 0.0))
    detection_status = str(summary.get("detection_status", "detected"))
    headline_name = _technique_phrase(str(primary_technique)) if primary_technique else _family_phrase(detected_family)

    assessment: dict[str, object] = {
        "status": "detected_only",
        "headline": f"Detected {headline_name}",
        "badge": "Technique Detected",
        "detected_family_display": _family_phrase(detected_family),
        "primary_technique_display": _technique_phrase(str(primary_technique)) if primary_technique else None,
        "runner_up_family_display": _family_phrase(runner_up_family),
        "confidence": detected_confidence,
        "confidence_percent": float(100.0 * detected_confidence),
        "family_margin": family_margin,
        "voiced_ratio_percent": float(100.0 * voiced_ratio),
        "feedback": str(summary.get("feedback", "")),
        "target_family_display": _family_phrase(str(target_family)) if target_family else None,
    }

    if voiced_ratio < MIN_VOICED_RATIO:
        assessment.update(
            {
                "status": "not_enough_voice",
                "badge": "Not Enough Singing",
                "headline": "Not enough voiced singing to judge the take",
                "feedback": "Sing a longer, clearer phrase so the model has enough voiced audio to judge technique.",
            }
        )
        return assessment

    if detection_status == "no_clear_technique":
        assessment.update(
            {
                "status": "no_clear_technique",
                "badge": "No Clear Technique",
                "headline": "No clear technique was detected",
                "feedback": "The take has voiced singing, but no technique score is strong enough to call yet.",
            }
        )
        return assessment

    if detection_status == "uncertain":
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
        detected_text = _technique_phrase(str(primary_technique)) if primary_technique else _family_phrase(detected_family)
        assessment["feedback"] = (
            f"The take reads most strongly as {detected_text}. "
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
