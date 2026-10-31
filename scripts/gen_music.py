#!/usr/bin/env python
"""
The Name, Sung — renders the 216 digits as pure harmonics.

Mapping (no translation — a digit IS a frequency ratio):
    digit d  ->  d-th harmonic of the 108 Hz fundamental  (d x 108 Hz)
    1=108  2=216  3=324  4=432  5=540  6=648  7=756  8=864  9=972 Hz
    108 chosen because the Name is 108 complement pairs; digit 4 lands
    on 432 Hz and digit 8 on 864 Hz natively (poetry-tier, but exact).

Structure (mirrors the voice track):
    8s gather drone -> 12 breaths x 16s (18 notes in a 10s window)
    -> 4 quarter-hour silences of 4s -> 9-harmonic completion chord.
    60 soft station ticks, 5 per breath. Utterance span: 216 s.

Reproduce:  python scripts/gen_music.py   (numpy + ffmpeg only)
Output:     docs/audio/the-name-sung-full.mp3
"""
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "audio"
WAV = ROOT / "audio_src"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

SR = 44100
F0 = 108.0
IN_S, SO_S, RE_S, SI_S, GATHER = 4.0, 10.0, 2.0, 4.0, 8.0
BREATH = IN_S + SO_S + RE_S
SLOT = SO_S / 18.0

ROWS = ["112358314594377741","493257291112358314","943777415617853819",
        "112358314594377741","617853819998752796","943777415617853819",
        "998752796516733369","617853819998752796","167333695493257291",
        "998752796516733369","493257291112358314","167333695493257291"]

def note(d, dur=SLOT, gain=0.30):
    """digit d -> its harmonic, enveloped; warm 2nd partial at -20 dB."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = d * F0
    a = np.sin(2 * np.pi * f * t) + 0.10 * np.sin(2 * np.pi * 2 * f * t)
    atk = int(0.06 * n)
    env = np.ones(n) * np.exp(-2.2 * t / dur)
    env[:atk] *= np.linspace(0, 1, atk)
    return (gain * a * env).astype(np.float32)

def tick(dur=0.03, f=1187.0, gain=0.05):
    n = int(dur * SR)
    env = np.sin(np.pi * np.arange(n) / n)
    return (gain * env * np.sin(2 * np.pi * f * np.arange(n) / SR)).astype(np.float32)

total_s = GATHER + 12 * BREATH + 4 * SI_S + 6.0
track = np.zeros(int(total_s * SR), np.float32)
def place(a, t):
    i = int(t * SR); j = min(len(track), i + len(a))
    track[i:j] += a[: j - i]

# sub-drone: 54 Hz, the fundamental's octave below — the clock underneath
t_all = np.arange(len(track)) / SR
track += (0.025 * np.sin(2 * np.pi * F0 / 2 * t_all)).astype(np.float32)

notes_placed = 0
sil_windows = []
t_b = GATHER
for b in range(12):
    for s in range(5):
        place(tick(), t_b + s * (BREATH / 5.0))
    for k, ch in enumerate(ROWS[b]):
        place(note(int(ch)), t_b + IN_S + k * SLOT)
        notes_placed += 1
    t_b += BREATH
    if (b + 1) in (3, 6, 9, 12):
        sil_windows.append((t_b, t_b + SI_S))
        t_b += SI_S

# completion: all nine harmonics stacked, slow fade
chord = np.zeros(int(3.0 * SR), np.float32)
for d in range(1, 10):
    chord += note(d, dur=3.0, gain=0.30 / d ** 0.5)
place(chord, t_b + 0.8)
fade = int(2.5 * SR)
track[-fade:] *= np.linspace(1, 0, fade)
track = (track / max(1e-6, np.abs(track).max()) * 0.92).astype(np.float32)

# ---- in-script verification ----
def rms(a, t0, t1):
    seg = a[int(t0 * SR): int(t1 * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
sound_rms = rms(track, GATHER + IN_S, GATHER + IN_S + SO_S)
sil_rms = [rms(track, a + 0.3, b - 0.3) for a, b in sil_windows]
print(f"[verify] notes placed: {notes_placed} (expect 216)")
print(f"[verify] sound-window RMS {sound_rms:.4f} vs silence RMS "
      f"{['%.4f' % r for r in sil_rms]} (drone-only, expect << sound)")
assert notes_placed == 216
assert all(r < sound_rms * 0.35 for r in sil_rms), "silences not silent!"

w = WAV / "the-name-sung-full.wav"
import wave
with wave.open(str(w), "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w),
                "-codec:a", "libmp3lame", "-b:a", "128k",
                str(OUT / "the-name-sung-full.mp3")], check=True)
print(f"[done] {total_s:.1f}s -> docs/audio/the-name-sung-full.mp3 "
      f"(utterance span {GATHER + 12*BREATH + 4*SI_S:.0f}s = 216s)")
