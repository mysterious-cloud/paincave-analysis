#!/usr/bin/env python3
"""
A song with the coach's voice taken out where the coach talks, for Studio's Process (paincave docs/studio.md, class
breakdown: "Without the coach").

The vocals are split out with Demucs `htdemucs_ft` (its vocals model: `stems.separate_vocals`) and subtracted from the
song only inside the spans given (the coach's subtitle lines), fading in and out over FADE seconds at each edge.
Outside the spans the song is the input, sample for sample. Singing inside a span goes with the coach: Demucs can't tell
a singer from the coach.

    remove_voice.py IN.wav -o OUT.wav --spans SPANS.json     # SPANS.json: [[start, end], ...] in seconds

Progress goes to stdout as `PROGRESS {"stage": "voice", "fraction"}` lines. OUT.wav (16-bit PCM, the input's rate and
channels) is written whole or not at all.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import stems

FADE = 0.1  # seconds over which the voice fades out of the mix and back in at each span's edges


def progress(fraction: float) -> None:
    print("PROGRESS " + json.dumps({"stage": "voice", "fraction": round(fraction, 3)}), flush=True)


def resample(audio: np.ndarray, rate: int, to: int) -> np.ndarray:
    """(channels, n) float32 from `rate` to `to` by ffmpeg (Demucs works at 44.1 kHz; the song keeps its own rate)."""
    if rate == to:
        return audio
    channels = audio.shape[0]
    result = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
                             "-ar", str(to), "-f", "f32le", "-"], input=audio.T.astype(np.float32).tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, channels).T


def gain(length: int, rate: int, spans: list[list[float]]) -> np.ndarray:
    """How much of the voice comes out at each sample: 1 inside a span, 0 outside, ramps of FADE seconds between."""
    g = np.zeros(length, dtype=np.float32)
    ramp = max(1, round(FADE * rate))
    for start, end in spans:
        a, b = max(0, round(start * rate)), min(length, round(end * rate))
        if b <= a:
            continue
        g[a:b] = 1
        fade_in = np.linspace(0, 1, ramp, endpoint=False, dtype=np.float32)
        lo = max(0, a - ramp)
        g[lo:a] = np.maximum(g[lo:a], fade_in[ramp - (a - lo):])
        hi = min(length, b + ramp)
        g[b:hi] = np.maximum(g[b:hi], fade_in[::-1][:hi - b])
    return g


def remove_voice(path: Path, spans: list[list[float]]) -> tuple[np.ndarray, int]:
    """The song at `path` with the voice taken out in `spans`: (channels, n) float32 at its own rate."""
    for span in spans:
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, (int, float)) for x in span) and 0 <= span[0] < span[1]):
            raise ValueError(f"a span is [start, end] in seconds, start before end: {span}")
    try:
        song, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:  # soundfile raises its own LibsndfileError for anything it can't read
        raise ValueError(f"could not read {path}: {error}") from error
    song = song.T
    progress(0)
    if not spans:
        progress(1)
        return song, rate
    mix = stems.load_stereo(path)
    vocals = stems.separate_vocals(mix, progress, stereo=True)
    # Back to the song's own rate and channels (a mono song: the vocals' mean)
    vocals = resample(vocals, stems.RATE, rate)
    if song.shape[0] == 1:
        vocals = vocals.mean(0, keepdims=True)
    n = min(song.shape[1], vocals.shape[1])
    out = song.copy()
    out[:, :n] -= gain(song.shape[1], rate, spans)[:n] * vocals[:, :n]
    progress(1)
    return np.clip(out, -1, 1), rate


def main():
    parser = argparse.ArgumentParser(description="A song with the coach's voice taken out where the coach talks")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Where the WAV goes (written whole or not at all)")
    parser.add_argument("--spans", type=Path, required=True, help="JSON: [[start, end], ...] in seconds")
    args = parser.parse_args()
    try:
        spans = json.loads(args.spans.read_text())
        if not isinstance(spans, list):
            raise ValueError("the spans file holds a list of [start, end]")
        out, rate = remove_voice(args.audio, spans)
    except (ValueError, RuntimeError, json.JSONDecodeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    temporary = args.output.with_name(f".{args.output.name}.tmp.wav")
    sf.write(temporary, out.T, rate, subtype="PCM_16")
    os.replace(temporary, args.output)
    print(f"voice taken out in {len(spans)} spans", file=sys.stderr)


if __name__ == "__main__":
    main()
