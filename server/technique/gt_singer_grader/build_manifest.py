"""Build normalized training manifests from supported technique datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .constants import PRIMARY_FAMILY_TO_TECHNIQUES, TECHNIQUE_KEYS
from .manifest import normalize_label_list, validate_record, write_jsonl


VOCALSET_TECHNIQUE_TO_FAMILY = {
    "belt": "mixed_voice",
    "belted": "mixed_voice",
    "breathy": "breathy",
    "glissando": "glissando",
    "messa": "control",
    "messa_di_voce": "control",
    "normal": "control",
    "spoken": "control",
    "straight": "control",
    "trill": "vibrato",
    "trillo": "vibrato",
    "vibrato": "vibrato",
    "vocal_fry": "pharyngeal",
}

TECHNIQUE_TO_PRIMARY_FAMILY = {
    technique: family
    for family, techniques in PRIMARY_FAMILY_TO_TECHNIQUES.items()
    for technique in techniques
}
TECHNIQUE_STRENGTH_VALUES = {"absent", "weak", "present", "strong"}
PRESENT_STRENGTH_VALUES = {"present", "strong"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized technique-model manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gt = subparsers.add_parser("gtsinger", help="Build a manifest from a GT Singer directory")
    gt.add_argument("--root", required=True, help="Path to downloaded GT Singer data")
    gt.add_argument("--language", default="English")
    gt.add_argument("--output", required=True)
    gt.add_argument("--include-speech", action="store_true")

    app = subparsers.add_parser("app-recordings", help="Build a manifest from a labeled CSV of app recordings")
    app.add_argument("--csv", required=True, help="CSV with audio_path plus family/technique labels")
    app.add_argument("--output", required=True)
    app.add_argument("--dataset-name", default="app_recordings")
    app.add_argument("--recording-domain", default="app_user")

    vocalset = subparsers.add_parser("vocalset", help="Build a manifest from an extracted VocalSet directory")
    vocalset.add_argument("--root", required=True, help="Path to extracted VocalSet data")
    vocalset.add_argument("--output", required=True)
    vocalset.add_argument("--dataset-name", default="vocalset")
    vocalset.add_argument("--recording-domain", default="studio_exercises")
    return parser.parse_args()


def _validate_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, line_number=index))
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


def build_gtsinger_manifest(root: str, language: str, include_speech: bool) -> list[dict[str, Any]]:
    from .data import scan_gt_singer

    records = scan_gt_singer(root, language=language, include_speech=include_speech)
    output: list[dict[str, Any]] = []
    for record in records:
        family = record.clip_label
        family_name = "control" if record.role in {"control", "speech"} else record.family
        techniques = list(PRIMARY_FAMILY_TO_TECHNIQUES.get(family_name, ()))
        output.append(
            {
                "recording_id": f"gtsinger:{record.speaker}:{record.technique_folder}:{record.song}:{record.stem}",
                "dataset": "gtsinger",
                "audio_path": record.wav_path,
                "json_path": record.json_path,
                "recording_domain": "studio",
                "label_source": "gtsinger_folder_and_alignment_json",
                "split_group": record.speaker,
                "speaker_id": record.speaker,
                "song_id": record.song,
                "labels": {
                    "families": [family_name],
                    "techniques": techniques,
                    "clip_role": record.role,
                },
                "metadata": {
                    "language": language,
                    "technique_folder": record.technique_folder,
                    "split_group_song": record.split_group,
                    "clip_label_index": int(family),
                },
            }
        )
    return output


def build_app_recordings_manifest(
    csv_path: str,
    dataset_name: str,
    recording_domain: str,
    *,
    allow_duplicate_review_rows: bool = False,
) -> list[dict[str, Any]]:
    """Build a manifest from simple user-label CSV exports.

    Required CSV columns:
      audio_path

    Label columns:
      families, techniques
      or technique strength columns: mix, falsetto, breathy, pharyngeal,
      glissando, vibrato with absent/weak/present/strong values

    Optional columns:
      recording_id, singer_id, song_id, split_group, label_source, reviewer_id,
      intended_family, notes
    """
    output: list[dict[str, Any]] = []
    seen_audio_paths: dict[str, int] = {}
    seen_recording_ids: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"audio_path"} - fields
        if missing:
            raise SystemExit(f"CSV missing required column(s): {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            audio_path = (row.get("audio_path") or "").strip()
            recording_id = (row.get("recording_id") or "").strip() or f"{dataset_name}:{row_number - 1}:{Path(audio_path).stem}"
            if not allow_duplicate_review_rows:
                _require_unique_app_review_value(
                    field="audio_path",
                    value=audio_path,
                    row_number=row_number,
                    seen=seen_audio_paths,
                )
                _require_unique_app_review_value(
                    field="recording_id",
                    value=recording_id,
                    row_number=row_number,
                    seen=seen_recording_ids,
                )
            singer_id = (row.get("singer_id") or "").strip()
            song_id = (row.get("song_id") or "").strip()
            split_group = (row.get("split_group") or "").strip() or singer_id or recording_id
            technique_strengths = _parse_technique_strengths(row, row_number=row_number)
            explicit_techniques = normalize_label_list(row.get("techniques"))
            explicit_families = normalize_label_list(row.get("families"))
            techniques = explicit_techniques or [
                technique
                for technique, strength in technique_strengths.items()
                if strength in PRESENT_STRENGTH_VALUES
            ]
            intended_family = (row.get("intended_family") or "").strip()
            families = explicit_families or _derive_app_families(
                techniques,
                technique_strengths,
                intended_family,
            )
            _require_consistent_app_review_labels(
                row_number=row_number,
                explicit_families=explicit_families,
                explicit_techniques=explicit_techniques,
                derived_families=families,
                derived_techniques=techniques,
                technique_strengths=technique_strengths,
            )
            output.append(
                {
                    "recording_id": recording_id,
                    "dataset": dataset_name,
                    "audio_path": audio_path,
                    "recording_domain": recording_domain,
                    "label_source": (row.get("label_source") or "human_labeled_app_recording").strip(),
                    "split_group": split_group,
                    "speaker_id": singer_id,
                    "song_id": song_id,
                    "labels": {
                        "families": families,
                        "techniques": techniques,
                        "intended_family": intended_family,
                        "technique_strengths": technique_strengths,
                    },
                    "metadata": {
                        "reviewer_id": (row.get("reviewer_id") or "").strip(),
                        "notes": (row.get("notes") or "").strip(),
                    },
                }
            )
    return output


def _require_unique_app_review_value(
    *,
    field: str,
    value: str,
    row_number: int,
    seen: dict[str, int],
) -> None:
    if not value:
        return
    previous = seen.get(value)
    if previous is not None:
        raise SystemExit(f"row {row_number}: duplicate {field} also appears on row {previous}: {value}")
    seen[value] = row_number


def _require_consistent_app_review_labels(
    *,
    row_number: int,
    explicit_families: list[str],
    explicit_techniques: list[str],
    derived_families: list[str],
    derived_techniques: list[str],
    technique_strengths: dict[str, str],
) -> None:
    present_techniques = sorted(
        technique
        for technique, strength in technique_strengths.items()
        if strength in PRESENT_STRENGTH_VALUES
    )
    weak_techniques = sorted(
        technique
        for technique, strength in technique_strengths.items()
        if strength == "weak"
    )
    errors: list[str] = []

    if explicit_techniques and present_techniques and sorted(explicit_techniques) != present_techniques:
        errors.append(
            "explicit techniques do not match present/strong technique strength columns "
            f"(explicit={sorted(explicit_techniques)}, present_or_strong={present_techniques})"
        )
    weak_explicit = sorted(set(explicit_techniques).intersection(weak_techniques))
    if weak_explicit:
        errors.append(
            "weak-only technique(s) cannot be listed as supervised techniques: "
            + ", ".join(weak_explicit)
        )

    implied_families = sorted(
        {
            TECHNIQUE_TO_PRIMARY_FAMILY[technique]
            for technique in derived_techniques
            if technique in TECHNIQUE_TO_PRIMARY_FAMILY
        }
    )
    evaluation_only_families = {"control", "none", "unclear"}
    if explicit_families and implied_families:
        explicit_train_families = sorted(set(explicit_families) - evaluation_only_families)
        if explicit_train_families != implied_families:
            errors.append(
                "explicit families do not match supervised technique family columns "
                f"(explicit={explicit_train_families}, implied={implied_families})"
            )
    if set(derived_families).intersection(evaluation_only_families) and present_techniques:
        errors.append(
            "control/none/unclear family labels cannot have present/strong technique columns"
        )

    if errors:
        preview = "\n".join(f"  - row {row_number}: {error}" for error in errors)
        raise SystemExit(preview)


def _derive_app_families(
    techniques: list[str],
    technique_strengths: dict[str, str],
    intended_family: str,
) -> list[str]:
    families = sorted(
        {
            TECHNIQUE_TO_PRIMARY_FAMILY[technique]
            for technique in techniques
            if technique in TECHNIQUE_TO_PRIMARY_FAMILY
        }
    )
    if families:
        return families

    if intended_family in {"control", "none", "unclear"}:
        return [intended_family]

    if technique_strengths and all(strength == "absent" for strength in technique_strengths.values()):
        return ["control"]

    if any(strength == "weak" for strength in technique_strengths.values()):
        return ["unclear"]

    return []


def _parse_technique_strengths(row: dict[str, str], *, row_number: int) -> dict[str, str]:
    strengths: dict[str, str] = {}
    for technique in TECHNIQUE_KEYS:
        value = (row.get(technique) or "").strip().lower()
        if not value:
            continue
        if value not in TECHNIQUE_STRENGTH_VALUES:
            raise SystemExit(
                f"row {row_number}: {technique} must be one of "
                f"{', '.join(sorted(TECHNIQUE_STRENGTH_VALUES))}"
            )
        strengths[technique] = value
    return strengths


def _normalize_token(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _vocalset_family_from_parts(parts: list[str]) -> tuple[str | None, str | None]:
    normalized = [_normalize_token(part) for part in parts]
    candidates: list[str] = []
    for index, token in enumerate(normalized):
        candidates.append(token)
        if index + 1 < len(normalized):
            candidates.append(f"{token}_{normalized[index + 1]}")
        if index + 2 < len(normalized):
            candidates.append(f"{token}_{normalized[index + 1]}_{normalized[index + 2]}")

    for candidate in candidates:
        if candidate in VOCALSET_TECHNIQUE_TO_FAMILY:
            return VOCALSET_TECHNIQUE_TO_FAMILY[candidate], candidate
    return None, None


def _vocalset_singer_id(parts: list[str]) -> str:
    for part in parts:
        normalized = _normalize_token(part)
        if normalized[:1] in {"f", "m"} and any(char.isdigit() for char in normalized):
            return part
    return ""


def _vocalset_context(stem_tokens: list[str]) -> str:
    known_contexts = {"arpeggios", "arpeggio", "scales", "scale", "long", "long_tones", "excerpts", "excerpt"}
    for token in stem_tokens:
        normalized = _normalize_token(token)
        if normalized in known_contexts:
            return normalized
    return ""


def _vocalset_vowel(stem_tokens: list[str]) -> str:
    for token in stem_tokens:
        normalized = _normalize_token(token)
        if normalized in {"a", "e", "i", "o", "u"}:
            return normalized
    return ""


def _vocalset_wav_roots(root_path: Path) -> list[Path]:
    technique_root = root_path / "data_by_technique"
    if technique_root.exists():
        return [technique_root]
    return [root_path]


def _is_vocalset_sidecar(relative: Path) -> bool:
    return "__MACOSX" in relative.parts or relative.name.startswith("._")


def build_vocalset_manifest(root: str, dataset_name: str, recording_domain: str) -> list[dict[str, Any]]:
    """Build a conservative technique manifest from extracted VocalSet WAV files."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"VocalSet root not found: {root_path}")

    output: list[dict[str, Any]] = []
    wav_paths = sorted(path for wav_root in _vocalset_wav_roots(root_path) for path in wav_root.rglob("*.wav"))
    for wav_path in wav_paths:
        relative = wav_path.relative_to(root_path)
        if _is_vocalset_sidecar(relative):
            continue
        parts = list(relative.parts)
        stem_tokens = wav_path.stem.split("_")
        family, source_technique = _vocalset_family_from_parts(parts + stem_tokens)
        if family is None or source_technique is None:
            continue

        singer_id = _vocalset_singer_id(parts)
        split_group = singer_id or parts[0] if parts else wav_path.parent.name
        techniques = list(PRIMARY_FAMILY_TO_TECHNIQUES.get(family, ()))
        output.append(
            {
                "recording_id": f"{dataset_name}:{relative.with_suffix('').as_posix()}",
                "dataset": dataset_name,
                "audio_path": str(wav_path),
                "recording_domain": recording_domain,
                "label_source": "vocalset_path_technique_label",
                "split_group": split_group,
                "speaker_id": singer_id,
                "song_id": _vocalset_context(stem_tokens),
                "labels": {
                    "families": [family],
                    "techniques": techniques,
                    "source_technique": source_technique,
                },
                "metadata": {
                    "relative_path": relative.as_posix(),
                    "vowel": _vocalset_vowel(stem_tokens),
                    "context": _vocalset_context(stem_tokens),
                },
            }
        )

    if not output:
        raise RuntimeError(f"No VocalSet WAV files with known technique labels found under: {root_path}")
    return output


def main() -> None:
    args = parse_args()
    if args.command == "gtsinger":
        records = build_gtsinger_manifest(args.root, args.language, args.include_speech)
    elif args.command == "app-recordings":
        records = build_app_recordings_manifest(args.csv, args.dataset_name, args.recording_domain)
    elif args.command == "vocalset":
        records = build_vocalset_manifest(args.root, args.dataset_name, args.recording_domain)
    else:
        raise SystemExit(f"unknown command: {args.command}")

    _validate_records(records)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
