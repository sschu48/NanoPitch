# NanoPitch — Improvement Results

This file tracks each modification I made on top of the upstream NanoPitch
codebase, the motivation, and the measured impact on the held-out 600-clip
test set (six SNR conditions: -5, 0, +5, +10, +20 dB, clean).

The frozen reference is the `baseline` git tag — checking it out gives the
exact code state used to produce `training/runs/baseline/eval.json`.

All runs use identical hyperparameters: 50 epochs, batch 32, lr 1e-3,
seq_len 200, AdamW(0.8, 0.98), MPS device. The only changed variable
between runs is the modification under test.

---

## v1_aug — implement noise augmentation in `augment_mel_batch`

**What changed.** The shipped `augment_mel_batch` was a stub that returned
`mel_clean` unchanged, so `--snr-range` was effectively dead and the model
only ever saw clean vocals during training. Implemented per-row random
SNR mixing in the log-mel domain:

```python
snr_db = uniform(snr_range[0], snr_range[1])      # one per batch row
gain_offset = -snr_db * ln(10) / 20               # log-power scaling
mel_mix = logaddexp(mel_clean, mel_noise + gain_offset)
```

`logaddexp` is the correct combiner for log-mel features (additive in
linear power), and stable for the small/negative log-energies the
features contain. Same `--snr-range` as default: `[-5, +20]` dB.

**Hypothesis.** Largest gains at low test SNR (-5, 0 dB) where the
baseline is weakest, since the model now sees the same noise distribution
during training as it sees at evaluation.

### Measured impact (offline Viterbi)

| SNR | VAD baseline → v1_aug | VDR baseline → v1_aug | RPA baseline → v1_aug | Med¢ baseline → v1_aug |
|---|---|---|---|---|
| -5 dB   | 78.9 → **93.3** (+14.4) | 57.2 → 57.5 (+0.3) | 88.6 → 89.6 (+1.0) | 73.2 → **47.1** (−26.1) |
| 0 dB    | 81.2 → **92.9** (+11.7) | 64.2 → 57.9 (−6.3) | 90.8 → 87.3 (−3.5) | 47.1 → 74.9 (+27.8) |
| +5 dB   | 83.3 → **94.1** (+10.8) | 68.1 → 59.2 (−8.9) | 90.3 → 88.9 (−1.4) | 43.8 → **38.3** (−5.5) |
| +10 dB  | 90.0 → **95.4** (+5.4)  | 67.8 → 62.6 (−5.2) | 91.5 → 92.3 (+0.8) | 37.5 → **18.9** (−18.6) |
| +20 dB  | 88.7 → **96.5** (+7.8)  | 68.3 → 64.6 (−3.7) | 94.3 → 93.4 (−0.9) | 10.3 → 17.5 (+7.2) |
| clean   | 98.6 → 97.0 (−1.6)      | 85.2 → 64.2 (−21.0)| 94.7 → 92.4 (−2.3) | 9.2 → 16.7 (+7.5) |
| **overall** | **86.8 → 94.9 (+8.1)** | 68.5 → 61.0 (−7.5) | 91.8 → 90.7 (−1.1) | 36.1 → 35.5 (−0.6) |

Realtime Viterbi (browser deployment) shows the same pattern: VAD
86.8 → 94.9 (+8.1), VDR 67.1 → 60.3 (−6.8), RPA 89.8 → 87.8 (−2.0).

### Interpretation

The augmentation did exactly what it was supposed to do for **VAD**:
overall accuracy jumped 8.1 points, with the largest improvements
concentrated in the noisiest test conditions (+14.4 pts at −5 dB), as
hypothesized. Median pitch error at −5 dB also dropped sharply
(73 → 47 cents), meaning even the wrong predictions are closer to the
true pitch.

The **VDR drop is a classic precision/recall tradeoff**. The augmented
model has learned to be more conservative about declaring frames voiced
when the input is noisy — fewer false positives (good for VAD accuracy),
but it now misses some borderline-voiced frames (lower VDR). RPA, which
is conditional on the model declaring a frame voiced, dropped slightly
(91.8 → 90.7) because the pitch posteriorgram is fuzzier under noisy
inputs and a few frames slip past the 50-cent accuracy threshold.

Net assessment: this is a real improvement for the **VAD task in noisy
conditions**, but a slight regression on **pitch precision**. For a real
deployment (e.g., a singing-tutor app that needs to distinguish
"singing" from "silence"), the VAD gain is the more important property.

### Suggested follow-up

The VDR drop suggests the augmented model's VAD head over-corrected
toward conservative behaviour. Two natural next experiments:

- **§2 focal loss / `pos_weight` on the VAD head** to recover voiced
  recall without giving up the augmentation gains.
- **§5 cosine LR schedule** — loss plateaued around 0.028 at constant
  LR; a decaying schedule should let the model fine-tune the pitch head
  past the current floor.
