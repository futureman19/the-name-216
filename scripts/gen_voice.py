#!/usr/bin/env python
"""
The Voice of the Name — generates all voice assets for breathe.html
and the full guided-cycle track (the-name-breathed-full.mp3).

Engine: supertonic (starred repo supertone-oss-archive/supertonic) —
lightning-fast on-device TTS via ONNX. No GPU, no network calls after
the first model download.

Reproduce:
    uv venv --python 3.12 voxenv
    uv pip install --python voxenv/Scripts/python.exe supertonic
    voxenv/Scripts/python.exe scripts/gen_voice.py        (Windows)
    voxenv/bin/python scripts/gen_voice.py                (posix)

Outputs:
    docs/audio/letters/1.mp3 .. 9.mp3   Hebrew letter names (Aleph..Tet)
    docs/audio/cues/{inhale,rest,silence,begin,complete}.mp3
    docs/audio/the-name-breathed-full.mp3   full guided cycle (~3.7 min)
"""
import subprocess, sys
from pathlib import Path

import numpy as np
from supertonic import TTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "audio"
WAV = ROOT / "audio_src"           # working dir for raw wav (gitignored)
for d in (OUT / "letters", OUT / "cues", WAV):
    d.mkdir(parents=True, exist_ok=True)

VOICE = "M1"
SPEED = 1.0
STEPS = 8

# display name -> synthesis spelling (forces correct English-TTS reading)
LETTERS = {
    1: ("Aleph", "Ah-lef"), 2: ("Bet", "Bet"), 3: ("Gimel", "Ghee-mel"),
    4: ("Dalet", "Dah-let"), 5: ("He", "Heh"), 6: ("Vav", "Vahv"),
    7: ("Zayin", "Zah-yin"), 8: ("Chet", "Khet"), 9: ("Tet", "Tet"),
}
CUES = {
    "inhale":  "Breathe in.",
    "rest":    "Rest.",
    "silence": "Silence.",
    "begin":   "The Name. Two hundred and sixteen letters. Twelve breaths. Breathe with me.",
    "complete":"The Name has been breathed.",
}

ROWS = ["112358314594377741","493257291112358314","943777415617853819",
        "112358314594377741","617853819998752796","943777415617853819",
        "998752796516733369","617853819998752796","167333695493257291",
        "998752796516733369","493257291112358314","167333695493257291"]

# calm pace (ms) — must match breathe.html PACE
IN_S, SO_S, RE_S, SI_S = 4.0, 10.0, 2.0, 4.0
INTRO_S = 8.0                       # begin-cue runway
LETTER_SLOT = SO_S / 18.0           # 18 letters per breath

tts = TTS(auto_download=True)
style = tts.get_voice_style(VOICE)

def synth(text):
    a, _ = tts.synthesize(text, style, total_steps=STEPS, speed=SPEED,
                          silence_duration=0.05, lang="en")
    return np.asarray(a, dtype=np.float32).reshape(-1)

def probe_sr():
    p = WAV / "_probe.wav"
    tts.save_audio(synth("probe"), str(p))
    import wave
    with wave.open(str(p)) as w:
        sr = w.getframerate()
    p.unlink()
    return sr

SR = probe_sr()
print(f"[voice] sample rate {SR}")

def trim(a, thresh=0.008, pad=int(0.03 * SR)):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - pad): min(len(a), idx[-1] + pad)]

def to_mp3(a, path, kbps="96k"):
    w = WAV / (path.stem + ".wav")
    import wave
    with wave.open(str(w), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(w),
                    "-codec:a", "libmp3lame", "-b:a", kbps, str(path)], check=True)

def fit(a, seconds):
    """time-stretch to exactly `seconds` if longer (resample); pad if shorter."""
    n = int(seconds * SR)
    if len(a) > n:
        x = np.linspace(0, len(a) - 1, n)
        return np.interp(x, np.arange(len(a)), a).astype(np.float32)
    return np.concatenate([a, np.zeros(n - len(a), np.float32)])

# ---------- 1) letter + cue clips ----------
letters, cues = {}, {}
for d, (disp, say) in LETTERS.items():
    a = trim(synth(say))
    letters[d] = a
    to_mp3(a, OUT / "letters" / f"{d}.mp3")
    print(f"[letter] {d} {disp:6s} {len(a)/SR:.2f}s")
for name, text in CUES.items():
    a = trim(synth(text))
    cues[name] = a
    to_mp3(a, OUT / "cues" / f"{name}.mp3")
    print(f"[cue] {name:8s} {len(a)/SR:.2f}s")

# ---------- 2) full guided cycle ----------
BREATH = IN_S + SO_S + RE_S                       # 16 s
quarters = {3, 6, 9, 12}
total_s = INTRO_S + 12 * BREATH + 4 * SI_S + 6.0  # + outro runway
track = np.zeros(int(total_s * SR), np.float32)

def place(a, t, gain=1.0):
    i = int(t * SR)
    j = min(len(track), i + len(a))
    track[i:j] += gain * a[: j - i]

def tick(t, f=1187.0, dur=0.03, gain=0.05):       # soft station tick
    n = int(dur * SR)
    env = np.sin(np.pi * np.arange(n) / n)
    place((env * np.sin(2 * np.pi * f * np.arange(n) / SR)).astype(np.float32), t, gain)

place(cues["begin"], 0.6)
t_breath = INTRO_S
for b in range(12):
    place(cues["inhale"], t_breath + 0.25)
    # five station ticks across the breath (60 total)
    for s in range(5):
        tick(t_breath + s * (BREATH / 5.0))
    row = ROWS[b]
    for k, dch in enumerate(row):
        a = fit(letters[int(dch)], LETTER_SLOT)
        place(a, t_breath + IN_S + k * LETTER_SLOT)
    place(cues["rest"], t_breath + IN_S + SO_S + 0.25)
    t_breath += BREATH
    if (b + 1) in quarters:
        place(cues["silence"], t_breath + 0.3)
        t_breath += SI_S

place(cues["complete"], t_breath + 0.8)
tick(t_breath + 3.2, f=594.0, dur=2.4, gain=0.08)  # low closing bell
fade = int(2.0 * SR)
track[-fade:] *= np.linspace(1, 0, fade)
track = (track / max(1e-6, np.abs(track).max()) * 0.92).astype(np.float32)

to_mp3(track, OUT / "the-name-breathed-full.mp3", kbps="128k")
print(f"[track] {total_s:.1f}s total — utterance span "
      f"{INTRO_S + 12*BREATH + 4*SI_S:.0f}s = 216s by design")
print("[done] assets in docs/audio/")
