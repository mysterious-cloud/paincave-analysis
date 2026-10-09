#!/usr/bin/env python3
"""
Reference-song analysis for Songlab.

Decodes one audio file, fits the Beat This! bar grid with `analyze.fit_beat_grid` and writes a single
`analysis.json`: tempo, numbered beats with confidence, and the bars every later layer (stems, sections,
chords) hangs on. Contract: docs/reference-format.md.

`--stages` adds layers on top of the grid. `stems` splits the song with Demucs (cached under `--cache-dir`)
and measures each stem's level per bar; `sections` finds boundaries and repetition labels (`sections.py`);
`allin1` adds All-In-One's named sections (`allin1_stage.py`, its own environment, on the cached stems);
`peaks` writes the waveform peaks the Lab draws beside the analysis.

None of the Pain Cave library machinery runs here: no content hash, staging, PCAV, AAC, work zones or
metadata. A refused input exits 1 with one stderr line and writes nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import allin1_stage
import analyze
import features
import sections

FORMAT_VERSION = 1
FORMAT_KIND = "songlab-reference-analysis"
DEFAULT_MIN_BPM = 60  # reference songs run slower than the Pain Cave default of 100
DEFAULT_MAX_BPM = 200
METER = (4, 4)  # assumed, not detected: the grid, like the detector's bars, counts four beats to a bar
TIME_DECIMALS = 3  # milliseconds, as `analyze.build_beats_array` rounds beat times
STAGES = ("grid", "stems", "sections", "allin1", "peaks")  # grid always runs; the others are opt-in


def report(stage: str, fraction: float, enabled: bool) -> None:
    """One machine-readable progress line on stdout, for Songlab's job runner."""
    if enabled:
        print("PROGRESS " + json.dumps({"stage": stage, "fraction": round(fraction, 3)}), flush=True)


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
        progress: bool = False,
) -> dict:
    """Decode `audio_path` and return the analysis document. Raises `analyze.NoBeatGrid` on a refused input."""
    if not 0 < min_bpm < max_bpm:
        raise ValueError("BPM bounds must satisfy 0 < min_bpm < max_bpm")
    report("grid", 0, progress)
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
    report("grid", 1, progress)

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


def add_layers(document: dict, audio_path: Path, stages: set[str], cache_dir: Path, progress: bool) -> dict | None:
    """Measure the opt-in layers onto `document` (in place) and return the peaks document, if asked for.

    Everything is measured in seconds on the 44.1 kHz stereo decode, then read out per grid bar.
    """
    import stems as stem_split

    bars = document["grid"]["bars"]
    mix = stem_split.load_stereo(audio_path)
    mono = mix.mean(axis=0)
    for bar, loudness in zip(bars, features.bar_loudness(mono, stem_split.RATE, bars)):
        bar["loudness"] = loudness

    separated, levels, files = None, None, {}
    if "stems" in stages:
        report("stems", 0, progress)
        files = stem_split.stems_for(audio_path, mix, cache_dir, lambda f: report("stems", f, progress))
        separated = stem_split.load_stems(files)
        levels, peak_db = features.stem_levels(separated, stem_split.RATE, bars)
        for i, bar in enumerate(bars):
            bar["stems"] = {name: levels[name][i] for name in stem_split.STEMS}
        document["stems"] = {
            "model": stem_split.MODEL,
            "files": {name: os.path.relpath(path, cache_dir) for name, path in files.items()},
            "peakDb": peak_db,
            "relDb": features.REL_DB,
            "floorDb": features.FLOOR_DB,
        }

    if "sections" in stages:
        report("sections", 0, progress)
        block, novelty = sections.analyze_sections(mono, stem_split.RATE, document["grid"]["beats"], bars, levels)
        for bar, value in zip(bars, novelty):
            bar["novelty"] = value
        document["sections"] = {"methods": {sections.METHOD: block}}
        report("sections", 1, progress)

    if "allin1" in stages:
        report("allin1", 0, progress)
        stems_dir = Path(next(iter(files.values()))).parent
        found = allin1_stage.block(allin1_stage.run(audio_path, stems_dir, cache_dir), bars)
        document.setdefault("sections", {"methods": {}})["methods"][allin1_stage.METHOD] = found
        report("allin1", 1, progress)

    if "peaks" not in stages:
        return None
    report("peaks", 0, progress)
    result = {"peaksPerSecond": features.PEAKS_PER_SECOND,
              "mix": features.peaks(mono, stem_split.RATE),
              "stems": {name: features.peaks(samples, stem_split.RATE) for name, samples in (separated or {}).items()}}
    report("peaks", 1, progress)
    return result


def write_json(path: Path, document: dict, indent: int | None = 2) -> None:
    """Write the document whole or not at all; a failure leaves any earlier file at `path` as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        separators = None if indent else (",", ":")
        temporary.write_text(json.dumps(document, indent=indent, separators=separators, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_stages(value: str) -> set[str]:
    stages = {s.strip() for s in value.split(",") if s.strip()}
    unknown = stages - set(STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stage {', '.join(sorted(unknown))}; choose from {', '.join(STAGES)}")
    # allin1 reads the Demucs stems
    return stages | {"grid"} | ({"stems"} if "allin1" in stages else set())


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
    parser.add_argument("--stages", type=parse_stages, default={"grid"},
                        help=f"Comma-separated layers to measure: {', '.join(STAGES)} (default: grid)")
    parser.add_argument("--cache-dir", type=Path,
                        help="Where stems are cached between runs (default: a 'cache' folder beside the output)")
    parser.add_argument("--peaks", type=Path,
                        help="Where the peaks stage writes its file (default: peaks.json beside the output)")
    parser.add_argument("--progress", action="store_true",
                        help='Print PROGRESS {"stage", "fraction"} lines on stdout')
    parser.add_argument("--print-activity", action="store_true",
                        help="Print which stems play in each bar (needs the stems stage)")
    args = parser.parse_args(argv)

    if not 0 < args.min_bpm < args.max_bpm:
        parser.error("BPM bounds must satisfy 0 < min-bpm < max-bpm")
    if not args.audio_file.is_file():
        raise SystemExit(f"Error: file not found: {args.audio_file}")
    if args.output.resolve() == args.audio_file.resolve():
        raise SystemExit(f"Error: refused {args.audio_file.name}: output path is the input file; nothing written")

    cache_dir = args.cache_dir or args.output.parent / "cache"
    peaks_path = args.peaks or args.output.with_name("peaks.json")
    try:
        document = analyze_reference(args.audio_file, args.min_bpm, args.max_bpm, args.first_downbeat, args.progress)
        peaks = add_layers(document, args.audio_file, args.stages, cache_dir, args.progress) \
            if args.stages - {"grid"} else None
    except analyze.NoBeatGrid as e:
        raise SystemExit(f"Error: refused {args.audio_file.name}: {e}; nothing written")
    except (ValueError, RuntimeError, OSError) as e:  # an undecodable file or a bad flag value
        raise SystemExit(f"Error: {args.audio_file.name}: {' '.join(str(e).split())}; nothing written")

    if peaks is not None:
        write_json(peaks_path, peaks, indent=None)  # before the analysis: its presence means the run finished
    write_json(args.output, document)
    if args.print_activity and "stems" in document:
        levels = {name: [bar["stems"][name] for bar in document["grid"]["bars"]] for name in document["stems"]["peakDb"]}
        print(features.activity_matrix(document["grid"]["bars"], levels, document["stems"]["peakDb"]))
    bars = document["grid"]["bars"]
    print(f"Wrote {args.output}: {document['tempo']['bpm']} BPM, {len(bars)} bars")


if __name__ == "__main__":
    main()
