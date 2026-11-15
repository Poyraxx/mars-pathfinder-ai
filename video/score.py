"""Quiet ambient bed for the demo (no hits, no melody): slow detuned pad + soft air noise -> public/score.wav."""
import wave
from pathlib import Path

import numpy as np

SR, T = 44100, 30.0
t = np.arange(int(SR * T)) / SR
rng = np.random.default_rng(7)
chords = [(110.0, 164.8, 220.0, 329.6), (98.0, 146.8, 196.0, 293.7), (110.0, 164.8, 220.0, 329.6), (87.3, 130.8, 174.6, 261.6)]
seg = T / len(chords)
pad = np.zeros_like(t)
for k, chord in enumerate(chords):
    w = np.clip(1 - np.abs((t - (k + .5) * seg) / (seg * .85)), 0, 1) ** 1.5
    for i, f in enumerate(chord):
        for d in (-.6, .6):
            pad += w * np.sin(2 * np.pi * (f + d) * t + i) / (i + 1.5)
noise = np.convolve(rng.standard_normal(t.size), np.ones(400) / 400, "same") * 2.5
mix = pad * .55 + noise * .25
mix *= np.clip(t / 2.5, 0, 1) * np.clip((T - t) / 2.0, 0, 1)
mix = mix / np.abs(mix).max() * .5
out = Path(__file__).parent / "public/score.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print("wrote", out)
