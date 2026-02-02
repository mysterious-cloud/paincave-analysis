#!/usr/bin/env python3
"""
Re-run audio analysis on existing songs, preserving alignment settings.

Usage:
    # Re-analyze a specific song by hash
    python reanalyze.py b877ece5cfa6

    # Re-analyze all songs
    python reanalyze.py --all

    # Dry run (show what would be done)
    python reanalyze.py --all --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyze import analyze_audio


SONGS_DIR = Path(__file__).parent.parent.parent / "workouts" / "songs"


def get_song_dirs() -> list[Path]:
    """Get all song directories (excluding drafts)."""
    if not SONGS_DIR.exists():
        return []
    return [
        d for d in SONGS_DIR.iterdir()
        if d.is_dir() and d.name != "drafts" and not d.name.startswith(".")
    ]


def find_audio_file(song_dir: Path) -> Path | None:
    """Find the audio file in a song directory."""
    for ext in [".m4a", ".mp3", ".wav", ".aac"]:
        audio = song_dir / f"audio{ext}"
        if audio.exists():
            return audio
    return None


def reanalyze_song(song_dir: Path, dry_run: bool = False) -> bool:
    """Re-analyze a single song, preserving first_downbeat if set."""
    audio_file = find_audio_file(song_dir)
    if not audio_file:
        print(f"  No audio file found in {song_dir.name}")
        return False

    analysis_file = song_dir / "analysis.json"

    # Read existing analysis to preserve first_downbeat
    first_downbeat = None
    reference_bpm = None
    if analysis_file.exists():
        try:
            existing = json.loads(analysis_file.read_text())
            first_downbeat = existing.get("first_downbeat")
            # Could also preserve reference_bpm if stored
        except Exception as e:
            print(f"  Warning: Could not read existing analysis: {e}")

    if dry_run:
        print(f"  Would re-analyze {song_dir.name}")
        print(f"    Audio: {audio_file.name}")
        print(f"    First downbeat: {first_downbeat}")
        return True

    print(f"  Analyzing {song_dir.name}...")

    try:
        analysis = analyze_audio(
            audio_file,
            first_downbeat=first_downbeat,
            reference_bpm=reference_bpm,
        )

        # Write updated analysis
        analysis_file.write_text(json.dumps(analysis, indent=2))
        print(f"    Done - {len(analysis.get('kick_zones', []))} kick zones")
        return True

    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Re-run audio analysis on existing songs"
    )
    parser.add_argument(
        "hash",
        nargs="?",
        help="Song hash to re-analyze (first 12 chars of directory name)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-analyze all songs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    if not args.hash and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        song_dirs = get_song_dirs()
        if not song_dirs:
            print("No songs found")
            sys.exit(1)

        print(f"Re-analyzing {len(song_dirs)} song(s)...")
        success = 0
        for song_dir in song_dirs:
            if reanalyze_song(song_dir, args.dry_run):
                success += 1
        print(f"\nCompleted: {success}/{len(song_dirs)} songs")
    else:
        # Find song by hash prefix
        song_dirs = get_song_dirs()
        matches = [d for d in song_dirs if d.name.startswith(args.hash)]

        if not matches:
            print(f"No song found matching '{args.hash}'")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple songs match '{args.hash}':")
            for d in matches:
                print(f"  {d.name}")
            sys.exit(1)

        reanalyze_song(matches[0], args.dry_run)


if __name__ == "__main__":
    main()
