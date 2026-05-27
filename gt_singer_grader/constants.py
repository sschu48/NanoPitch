"""Shared constants for the GT Singer grader."""

SAMPLE_RATE = 16_000
FRAME_HOP_SECONDS = 0.010
FRAME_WINDOW_SECONDS = 0.025
DEFAULT_N_MELS = 64
DEFAULT_MAX_SECONDS = 6.0

# Mirrored from NanoPitch's tuned realtime decoder so VAD smoothing uses the
# same transition cost assumptions when we summarize clip-level technique use.
DEFAULT_ONSET_PENALTY = 0.75

TECHNIQUE_KEYS = (
    "mix",
    "falsetto",
    "breathy",
    "pharyngeal",
    "glissando",
    "vibrato",
)
TECHNIQUE_INDEX = {name: index for index, name in enumerate(TECHNIQUE_KEYS)}

FAMILY_NAMES = (
    "control",
    "breathy",
    "glissando",
    "mixed_voice",
    "falsetto",
    "pharyngeal",
    "vibrato",
)
FAMILY_TO_INDEX = {name: index for index, name in enumerate(FAMILY_NAMES)}

TECHNIQUE_FOLDER_TO_FAMILY = {
    "Breathy": "breathy",
    "Glissando": "glissando",
    "Mixed_Voice_and_Falsetto": "mixed_voice",
    "Pharyngeal": "pharyngeal",
    "Vibrato": "vibrato",
}

GROUP_NAME_TO_FAMILY = {
    "Breathy_Group": "breathy",
    "Glissando_Group": "glissando",
    "Mixed_Voice_Group": "mixed_voice",
    "Falsetto_Group": "falsetto",
    "Pharyngeal_Group": "pharyngeal",
    "Vibrato_Group": "vibrato",
}

PRIMARY_FAMILY_TO_TECHNIQUES = {
    "control": (),
    "breathy": ("breathy",),
    "glissando": ("glissando",),
    "mixed_voice": ("mix",),
    "falsetto": ("falsetto",),
    "pharyngeal": ("pharyngeal",),
    "vibrato": ("vibrato",),
}

SILENCE_TOKENS = {"<SP>", "SP", "sil", "SIL"}
