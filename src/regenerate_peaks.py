#!/usr/bin/env python3
"""
Regenerate PCAV peaks without re-running beat detection.

Reads existing analysis.pcav files, preserves analysis + beats sections,
recomputes peaks at 2 resolutions (editor window=256, overview window=8192)
from original.wav, and writes the updated PCAV file. Idempotent — skips
tracks already at the correct format. Also detects and removes stale
intermediate mipmap sections (0x04-0x07) from older formats.

Usage:
    uv run src/regenerate_peaks.py [--library PATH] [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

from analyze import load_audio
from pcav import MIPMAP_WINDOWS, build_peak_mipmaps, has_stale_peak_sections, read_pcav_file, write_pcav_file


def process_track(track_dir: Path, dry_run: bool, force: bool = False) -> bool:
    """Regenerate mipmap peaks for a single track. Returns True if updated."""
    pcav_path = track_dir / "analysis.pcav"
    wav_path = track_dir / "original.wav"

    if not pcav_path.exists():
        print(f"  skip: no analysis.pcav")
        return False
    if not wav_path.exists():
        print(f"  skip: no original.wav")
        return False

    pcav = read_pcav_file(pcav_path)
    old_size = pcav_path.stat().st_size
    old_count = len(pcav.peaks.peaks) if pcav.peaks else 0
    has_overview = pcav.overview_peaks is not None and len(pcav.overview_peaks.peaks) > 0
    stale = has_stale_peak_sections(pcav_path)

    # Idempotency: check if level 0 peak count matches window=256
    # Expected: total_samples / 256. Use duration × 48000 / 256 as estimate.
    duration = pcav.analysis.duration_seconds if pcav.analysis else 0
    expected_l0 = duration * 48000 // MIPMAP_WINDOWS[0]
    # Allow ±1s tolerance (48000/256 ≈ 187 peaks per second)
    tolerance = 48000 // MIPMAP_WINDOWS[0]

    if not force and has_overview and not stale and abs(old_count - expected_l0) <= tolerance and old_count > 0:
        print(f"  skip: up to date ({old_count} L0 peaks, {old_size:,} bytes)")
        return False

    print(f"  current: {old_count} peaks, overview={'yes' if has_overview else 'no'}, stale={'yes' if stale else 'no'} ({old_size:,} bytes)")

    if dry_run:
        print(f"  would regenerate → ~{expected_l0} editor peaks + overview")
        return False

    # Load audio and compute all mipmap levels
    samples, sr = load_audio(wav_path)
    levels = build_peak_mipmaps(samples, sr)

    # Level 0 → peaks (section 0x03, editor resolution)
    pcav.peaks = levels[0]
    # Level 1 → overview_peaks (section 0x08, coarse for thumbnails)
    pcav.overview_peaks = levels[1]

    write_pcav_file(pcav_path, pcav)
    new_size = pcav_path.stat().st_size

    print(f"  updated: editor={len(levels[0].peaks)}, overview={len(levels[1].peaks)}")
    print(f"  size: {new_size:,} bytes (was {old_size:,})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate PCAV peaks (editor + overview resolutions)"
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(os.environ.get("PAINCAVE_TRACKS_DIR", "../paincave-tracks")),
        help="Path to track library (default: $PAINCAVE_TRACKS_DIR or ../paincave-tracks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all tracks even if peak counts match",
    )
    args = parser.parse_args()

    library = args.library.resolve()
    if not library.is_dir():
        print(f"Error: library not found: {library}", file=sys.stderr)
        sys.exit(1)

    print(f"Library: {library}")
    print(f"Mipmap windows: {MIPMAP_WINDOWS}")
    if args.dry_run:
        print("DRY RUN — no files will be modified\n")

    updated = 0
    total = 0

    for track_dir in sorted(library.iterdir()):
        if not track_dir.is_dir() or not (track_dir / "analysis.pcav").exists():
            continue

        total += 1
        print(f"\n{track_dir.name}:")
        if process_track(track_dir, args.dry_run, args.force):
            updated += 1

    print(f"\nDone: {updated}/{total} tracks updated")


if __name__ == "__main__":
    main()
