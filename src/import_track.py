#!/usr/bin/env python3
"""
Import songs into the Pain Cave track library.

Pipeline:
1. Compute content hash
2. Create track directory
3. Copy/convert to original.wav
4. Run beat/intensity analysis
5. Write analysis.pcav (beats + peaks)
6. Write plan_workzones.json
7. Write metadata.json
8. Convert to M4A for streaming
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from analyze import analyze_audio, load_audio
from content_hash import compute_content_hash
from pcav import (
    PCAVData, AnalysisSection, BeatsSection,
    write_pcav_file, peaks_from_samples,
)


DEFAULT_TRACKS_DIR = Path(os.environ.get(
    "PAINCAVE_TRACKS_DIR",
    Path(__file__).resolve().parent.parent.parent / "paincave-tracks"
))

SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "paincave" / "shared" / "schemas"


def load_schema(name: str) -> dict:
    """Load a JSON schema from the shared schemas directory."""
    schema_path = SCHEMA_DIR / f"{name}.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    return json.loads(schema_path.read_text())


def validate_json(data: dict, schema_name: str) -> None:
    """Validate data against a JSON schema. Raises on failure."""
    schema = load_schema(schema_name)
    jsonschema.validate(data, schema)


def extract_suno_metadata(audio_path: Path) -> dict:
    """Extract SUNO metadata from audio file comment field.

    Returns dict with 'id' and 'created' if found.
    """
    result = {}
    try:
        proc = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_path)],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            return result

        data = json.loads(proc.stdout)
        comment = data.get('format', {}).get('tags', {}).get('comment', '')

        if 'suno' not in comment.lower():
            return result

        # Extract ID
        id_match = re.search(r'id=([a-f0-9-]+)', comment)
        if id_match:
            result['id'] = id_match.group(1).strip()

        # Extract created timestamp
        created_match = re.search(r'created=([0-9T:Z-]+)', comment)
        if created_match:
            result['created'] = created_match.group(1).strip()

    except Exception:
        pass

    return result


def get_file_created_time(path: Path) -> str:
    """Get file creation/modification time as ISO string."""
    mtime = path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def convert_to_wav(audio_path: Path, output_path: Path) -> None:
    """Convert any audio to WAV for archival."""
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(audio_path), '-ar', '44100', str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion to WAV failed: {result.stderr}")


def convert_to_m4a(audio_path: Path, output_path: Path) -> None:
    """Convert audio to M4A (AAC) for streaming."""
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(audio_path),
         '-c:a', 'aac', '-b:a', '256k', str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion to M4A failed: {result.stderr}")


def import_song(
    audio_path: Path,
    library_dir: Path,
    title: str | None = None,
    artist: str | None = None,
    source: str = "SUNO",
    source_id: str | None = None,
    genres: list[str] | None = None,
    min_bpm: int = 100,
    max_bpm: int = 180,
    force: bool = False,
) -> str:
    """Import a song into the library. Returns the content hash."""
    print(f"Importing: {audio_path}")

    # Step 1: Compute content hash
    print("Computing content hash...")
    content_hash = compute_content_hash(audio_path)
    print(f"  Hash: {content_hash}")

    # Step 2: Check if track exists
    song_dir = library_dir / content_hash
    if song_dir.exists():
        if force:
            print("  Track exists, --force specified, overwriting...")
            shutil.rmtree(song_dir)
        else:
            print(f"  Track already exists: {song_dir}")
            print("  Use --force to overwrite")
            return content_hash

    song_dir.mkdir(parents=True)

    # Step 3: Copy/convert to original.wav
    original_wav = song_dir / "original.wav"
    if audio_path.suffix.lower() == '.wav':
        print("Copying WAV...")
        shutil.copy2(audio_path, original_wav)
    else:
        print("Converting to WAV...")
        convert_to_wav(audio_path, original_wav)

    # Step 4: Run analysis
    print("Running analysis...")
    analysis = analyze_audio(original_wav, min_bpm, max_bpm)

    # Step 5: Write analysis.pcav
    print("Writing analysis.pcav...")
    samples, sample_rate = load_audio(original_wav)
    beat_times = analysis.get("raw", {}).get("beatTimes", [])

    pcav_data = PCAVData(
        analysis=AnalysisSection(
            duration_seconds=int(round(analysis["duration"])),
            bpm=analysis["bpm"],
            key=analysis.get("key"),
        ),
        beats=BeatsSection(beat_times=beat_times),
        peaks=peaks_from_samples(samples, sample_rate, peaks_per_second=10),
    )
    write_pcav_file(song_dir / "analysis.pcav", pcav_data)
    print(f"  {len(beat_times)} beats, {len(pcav_data.peaks.peaks)} peaks")

    # Step 6: Write plan_workzones.json
    print("Writing plan_workzones.json...")
    workzones = {
        "version": 1,
        "zones": analysis.get("workZones", []),
    }
    with open(song_dir / "plan_workzones.json", "w") as f:
        json.dump(workzones, f, indent=2)
    print(f"  {len(workzones['zones'])} zones")

    # Step 7: Write metadata.json
    print("Writing metadata.json...")
    suno_meta = extract_suno_metadata(audio_path)
    detected_source_id = source_id or suno_meta.get('id')

    # Use SUNO created time, fall back to file time
    created_at = suno_meta.get('created') or get_file_created_time(audio_path)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    metadata = {
        "hash": content_hash,
        "title": title or audio_path.stem,
        "artist": artist or "Pain Cave",
        "source": source,
        "sourceId": detected_source_id or "",
        "genres": genres or [],
        "bpm": analysis["bpm"],
        "durationSeconds": int(round(analysis["duration"])),
        "key": analysis.get("key"),
        "createdAt": created_at,
        "updatedAt": now,
    }

    try:
        validate_json(metadata, "metadata")
    except jsonschema.ValidationError as e:
        print(f"  WARNING: metadata validation failed: {e.message}")

    with open(song_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 8: Convert to M4A
    print("Converting to M4A...")
    convert_to_m4a(original_wav, song_dir / "audio.m4a")

    print(f"\nDone: {song_dir}")
    return content_hash


def main():
    parser = argparse.ArgumentParser(
        description="Import songs into the Pain Cave track library"
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument("--library", type=Path, default=DEFAULT_TRACKS_DIR,
                        help=f"Track library directory (default: $PAINCAVE_TRACKS_DIR)")
    parser.add_argument("-t", "--title", type=str, help="Track title (default: filename)")
    parser.add_argument("-a", "--artist", type=str, help="Artist name (default: Pain Cave)")
    parser.add_argument("--source", type=str, default="SUNO", choices=["SUNO"],
                        help="Audio source platform")
    parser.add_argument("--source-id", type=str, help="ID from source platform")
    parser.add_argument("-g", "--genres", type=str, help="Comma-separated genre slugs")
    parser.add_argument("--min-bpm", type=int, default=100, help="Minimum BPM (default: 100)")
    parser.add_argument("--max-bpm", type=int, default=180, help="Maximum BPM (default: 180)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing track")

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    genres = [g.strip() for g in args.genres.split(",") if g.strip()] if args.genres else None

    content_hash = import_song(
        audio_path=args.audio_file.resolve(),
        library_dir=args.library.resolve(),
        title=args.title,
        artist=args.artist,
        source=args.source,
        source_id=args.source_id,
        genres=genres,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        force=args.force,
    )

    print(f"\nHash: {content_hash}")


if __name__ == "__main__":
    main()
