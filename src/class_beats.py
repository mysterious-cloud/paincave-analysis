#!/usr/bin/env python3
"""
Every beat and downbeat of a long recording, for Studio's cut editor (paincave docs/studio.md, phase 2b).

The Beat This! detector, as the import runs it, over the whole recording: it assumes no tempo, so a class's songs
and their tempo changes all come through. No grid is fitted and no tempo is bounded here; the cut editor fits a grid
per song itself. The recording is decoded by ffmpeg straight to the detector's 22.05 kHz mono, so a 45-minute class
stays small in memory.

    class_beats.py AUDIO -o OUT.json

OUT.json: { "checkpoint": { "name", "sha256" }, "beats": [seconds...], "downbeats": [seconds...] }
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import analyze
import models

RATE = 22050  # what Beat This! listens at; any other rate it resamples itself


def decode(path: Path) -> np.ndarray:
    """The recording as mono float32 samples at RATE."""
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg could not decode {path}: {result.stderr.decode(errors='replace').strip()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if not len(audio):
        raise ValueError(f"no audio in {path}")
    return audio


def detect(path: Path) -> dict:
    """The recording's beats and downbeats in seconds, and the checkpoint that heard them."""
    audio = decode(path)
    beats, downbeats = analyze.load_detector()(audio, RATE)
    return {
        "checkpoint": models.checkpoint_info(),
        "beats": [round(float(t), 3) for t in beats],
        "downbeats": [round(float(t), 3) for t in downbeats],
    }


def main():
    parser = argparse.ArgumentParser(description="Every beat and downbeat of a long recording (Beat This!, no grid)")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Where the JSON goes (written whole or not at all)")
    args = parser.parse_args()
    try:
        found = detect(args.audio)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(found))
    os.replace(temporary, args.output)
    print(f"{len(found['beats'])} beats, {len(found['downbeats'])} downbeats", file=sys.stderr)


if __name__ == "__main__":
    main()
