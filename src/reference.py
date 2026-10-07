#!/usr/bin/env python3
"""
Reference-song analysis for Songlab.

Decodes one audio file, fits the Beat This! bar grid with `analyze.fit_beat_grid` and writes a single
`analysis.json`: tempo, numbered beats with confidence, and the bars every later layer (stems, sections,
chords) hangs on. Contract: docs/reference-format.md.

None of the Pain Cave library machinery runs here: no content hash, staging, PCAV, AAC, work zones or
metadata. A refused input exits 1 with one stderr line and writes nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import analyze

FORMAT_VERSION = 1
FORMAT_KIND = "songlab-reference-analysis"
DEFAULT_MIN_BPM = 60  # reference songs run slower than the Pain Cave default of 100
DEFAULT_MAX_BPM = 200
METER = (4, 4)  # assumed, not detected: the grid, like the detector's bars, counts four beats to a bar
TIME_DECIMALS = 3  # milliseconds, as `analyze.build_beats_array` rounds beat times


def build_bars(beats: list[dict], duration: float) -> list[dict]:
    """Bars 1..N from the numbered beats: bar b runs from its first beat to the first beat of bar b+1.

    Only complete bars (four beats on the grid) count; beats after the last one are left out. The last
    bar has no next downbeat inside the audio, so it ends one beat interval after its fourth beat, never
    past `duration`. Starts and ends are the beats' own rounded times, so adjacent bars share a boundary.
    """
    per_bar = analyze.BEATS_PER_BAR
    first = next((i for i, beat in enumerate(beats) if beat["bar"] == 1 and beat["beat"] == 1), None)
    if first is None:
        return []
    bars = []
    for number, i in enumerate(range(first, len(beats) - per_bar + 1, per_bar), start=1):
        if i + per_bar < len(beats):
            end = beats[i + per_bar]["time"]
        else:
            last, previous = beats[i + per_bar - 1]["time"], beats[i + per_bar - 2]["time"]
            end = round(min(duration, last + (last - previous)), TIME_DECIMALS)
        bars.append({"bar": number, "start": beats[i]["time"], "end": end})
    return bars


def analyze_reference(
        audio_path: Path,
        min_bpm: int = DEFAULT_MIN_BPM,
        max_bpm: int = DEFAULT_MAX_BPM,
        first_downbeat: float | None = None,
) -> dict:
    """Decode `audio_path` and return the analysis document. Raises `analyze.NoBeatGrid` on a refused input."""
    if not 0 < min_bpm < max_bpm:
        raise ValueError("BPM bounds must satisfy 0 < min_bpm < max_bpm")
    audio, sr = analyze.load_audio(audio_path)
    duration = len(audio) / sr
    analyze.require_sound(audio, sr)

    det_beats, det_downbeats = analyze.detect_beats(audio, sr, min_bpm, max_bpm)
    onsets = analyze.onset_strength(analyze.band_pass(audio, sr), sr)
    fit = analyze.fit_beat_grid(det_beats, det_downbeats, duration, min_bpm, max_bpm, onsets)
    inferred = int((~fit.heard).sum())
    print(f"Grid: {fit.bpm:.1f} BPM, {len(fit.beat_times)} beats ({inferred} inferred), "
          f"residual {fit.residual_ms:.1f} ms")

    first_downbeat_t = analyze.choose_first_bar(fit, first_downbeat, duration)
    beats = [
        {"time": b["t"], "bar": b["bar"], "beat": b["beat"], "confidence": b["confidence"]}
        for b in analyze.build_beats_array(fit.beat_times, first_downbeat_t, analyze.grid_confidences(fit))
    ]
    duration_seconds = round(duration, TIME_DECIMALS)
    bars = build_bars(beats, duration_seconds)
    if not bars:
        raise analyze.NoBeatGrid("no beat grid: no complete bar follows the first downbeat")

    return {
        "version": FORMAT_VERSION,
        "kind": FORMAT_KIND,
        "source": {"file": audio_path.name, "durationSeconds": duration_seconds, "sampleRate": int(sr)},
        "config": {"minBpm": min_bpm, "maxBpm": max_bpm, "firstDownbeat": first_downbeat},
        "tempo": {"bpm": round(fit.bpm, 1), "rangeBpm": list(analyze.tempo_range(fit))},
        "meter": list(METER),
        "grid": {
            "beats": beats,
            "firstDownbeatIndex": next(i for i, b in enumerate(beats) if b["bar"] == 1 and b["beat"] == 1),
            "bars": bars,
            "residualMs": round(fit.residual_ms, 2),
            "inferredBeatCount": inferred,
        },
    }


def write_json(path: Path, document: dict) -> None:
    """Write the document whole or not at all; a failure leaves any earlier file at `path` as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reference-song analysis for Songlab: bar grid to analysis.json")
    parser.add_argument("audio_file", type=Path, help="Path to audio file (wav, flac, m4a, mp3, ...)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to write analysis.json")
    parser.add_argument("--min-bpm", type=int, default=DEFAULT_MIN_BPM,
                        help=f"Minimum BPM for the fitted beat grid (default: {DEFAULT_MIN_BPM})")
    parser.add_argument("--max-bpm", type=int, default=DEFAULT_MAX_BPM,
                        help=f"Maximum BPM for the fitted beat grid (default: {DEFAULT_MAX_BPM})")
    parser.add_argument("--first-downbeat", type=float,
                        help="Time in seconds for bar 1 (snapped to the nearest grid beat; overrides detection)")
    args = parser.parse_args(argv)

    if not 0 < args.min_bpm < args.max_bpm:
        parser.error("BPM bounds must satisfy 0 < min-bpm < max-bpm")
    if not args.audio_file.is_file():
        raise SystemExit(f"Error: file not found: {args.audio_file}")
    if args.output.resolve() == args.audio_file.resolve():
        raise SystemExit(f"Error: refused {args.audio_file.name}: output path is the input file; nothing written")

    try:
        document = analyze_reference(args.audio_file, args.min_bpm, args.max_bpm, args.first_downbeat)
    except analyze.NoBeatGrid as e:
        raise SystemExit(f"Error: refused {args.audio_file.name}: {e}; nothing written")
    except (ValueError, RuntimeError, OSError) as e:  # an undecodable file or a bad flag value
        raise SystemExit(f"Error: {args.audio_file.name}: {' '.join(str(e).split())}; nothing written")

    write_json(args.output, document)
    bars = document["grid"]["bars"]
    print(f"Wrote {args.output}: {document['tempo']['bpm']} BPM, {len(bars)} bars")


if __name__ == "__main__":
    main()
