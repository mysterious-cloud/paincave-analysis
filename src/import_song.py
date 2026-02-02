#!/usr/bin/env python3
"""
Import songs into the Pain Cave track library.

Usage:
    # Import all songs from import folder:
    python import_song.py

    # Import specific file:
    python import_song.py /path/to/song.wav --title "Track Name" --artist "Artist"

This will:
1. Compute content hash (MD5 of audio samples)
2. Create tracks/{hash}/ directory
3. Copy original file as-is (original.wav, original.mp3, etc.)
4. Convert to audio.m4a (streaming-friendly)
5. Run beat/intensity analysis
6. Generate waveform peaks
7. Save analysis.json (raw data for client-side use)
8. Create database record via API with editable metadata
9. Move source file out of import folder (if using import folder workflow)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests

from content_hash import compute_content_hash
from metadata_parser import SongMetadata, parse_metadata_file
from metadata_writer import write_metadata_file

# API URL for track creation
API_URL = os.environ.get("API_URL", "http://127.0.0.1:3000")


def extract_suno_metadata(audio_path: Path) -> dict | None:
    """Extract SUNO metadata embedded in audio file using ffprobe.

    Returns dict with 'suno_url' and 'suno_generated' if found, else None.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(audio_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        tags = data.get('format', {}).get('tags', {})
        comment = tags.get('comment', '')

        # Parse SUNO comment format: "made by suno; created=2026-01-14T17:30:05Z; id=uuid"
        if 'made by suno' not in comment.lower():
            return None

        suno_data = {}

        # Extract creation date
        created_match = re.search(r'created=([^;]+)', comment)
        if created_match:
            suno_data['suno_generated'] = created_match.group(1).strip()

        # Extract song ID and construct URL
        id_match = re.search(r'id=([a-f0-9-]+)', comment)
        if id_match:
            song_id = id_match.group(1).strip()
            suno_data['suno_url'] = f'https://suno.com/song/{song_id}'

        return suno_data if suno_data else None

    except Exception:
        return None


def convert_to_m4a(audio_path: Path, output_path: Path) -> None:
    """Convert any audio to m4a (AAC) for streaming."""
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(audio_path),
         '-c:a', 'aac', '-b:a', '256k', str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


def create_track_record(
    hash: str,
    title: str,
    artist: str | None,
    bpm: float,
    duration: float,
    suno_url: str | None = None,
) -> dict:
    """Create or update a track record via the API.

    Returns the API response.
    """
    record_data = {
        "hash": hash,
        "title": title,
        "artist": artist or "Pain Cave",
        "bpm": bpm,
        "duration": duration,
    }

    if suno_url:
        record_data["sunoUrl"] = suno_url

    response = requests.post(
        f"{API_URL}/api/tracks",
        json=record_data
    )
    response.raise_for_status()
    return response.json()


def ensure_wav_for_analysis(audio_path: Path) -> Path:
    """Ensure we have a WAV file for analysis. Returns path to WAV file."""
    if audio_path.suffix.lower() == '.wav':
        return audio_path
    # Convert to temp WAV for analysis
    temp_wav = audio_path.parent / f"{audio_path.stem}_temp.wav"
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', str(audio_path), '-ar', '44100', str(temp_wav)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    return temp_wav


def run_analysis(audio_path: Path, min_bpm: int = 100, max_bpm: int = 180, first_downbeat: float | None = None) -> dict:
    """Run the full analysis pipeline on a song."""
    from analyze import analyze_audio
    return analyze_audio(audio_path, min_bpm, max_bpm, first_downbeat=first_downbeat)


def run_peaks(audio_path: Path, output_dir: Path) -> None:
    """Generate waveform peaks at multiple resolutions."""
    from analyze import load_audio
    from peaks import generate_peaks_json, generate_tiny_peaks

    samples, sample_rate = load_audio(audio_path)

    resolutions = [
        ("overview.json", 16384),
        ("medium.json", 4096),
        ("detail.json", 1024),
    ]

    for filename, samples_per_peak in resolutions:
        peaks_data = generate_peaks_json(samples, sample_rate, samples_per_peak)
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(peaks_data, f)
        print(f"  Generated {filename}")

    # Tiny peaks for track list thumbnails
    tiny_data = generate_tiny_peaks(samples)
    with open(output_dir / "tiny.json", "w") as f:
        json.dump(tiny_data, f)
    print(f"  Generated tiny.json")


def find_metadata_file(audio_path: Path) -> Path | None:
    """Find a matching .md metadata file for an audio file."""
    md_path = audio_path.with_suffix('.md')
    return md_path if md_path.exists() else None


def import_song(
    audio_path: Path,
    library_dir: Path,
    title: str | None = None,
    artist: str | None = None,
    min_bpm: int = 100,
    max_bpm: int = 180,
    metadata_path: Path | None = None,
    skip_db: bool = False,
) -> str:
    """Import a song into the library.

    Returns:
        The content hash (directory name) for the imported song.
    """
    print(f"Importing: {audio_path}")

    # Load metadata from markdown file if available
    metadata: SongMetadata | None = None
    md_file = metadata_path or find_metadata_file(audio_path)
    if md_file:
        print(f"Found metadata file: {md_file}")
        metadata = parse_metadata_file(md_file)
        # Use metadata title if no explicit title provided
        if not title and metadata.title:
            title = metadata.title

    # Extract SUNO metadata from audio file if embedded
    suno_audio_meta = extract_suno_metadata(audio_path)
    if suno_audio_meta:
        print(f"Found SUNO metadata in audio file")
        from datetime import datetime
        # Create metadata if it doesn't exist
        if metadata is None:
            metadata = SongMetadata()
        # Only use audio metadata if not already provided by .md file
        if not metadata.suno.url and suno_audio_meta.get('suno_url'):
            metadata.suno.url = suno_audio_meta['suno_url']
        if not metadata.suno.generated and suno_audio_meta.get('suno_generated'):
            try:
                metadata.suno.generated = datetime.fromisoformat(
                    suno_audio_meta['suno_generated'].replace('Z', '+00:00')
                ).date()
            except ValueError:
                pass

    # Compute content hash
    print("Computing content hash...")
    content_hash = compute_content_hash(audio_path)
    print(f"Content hash: {content_hash}")

    # Create song directory
    song_dir = library_dir / content_hash
    if song_dir.exists():
        print(f"Song already exists at: {song_dir}")
        print("Updating analysis...")
    else:
        song_dir.mkdir(parents=True)
        print(f"Created: {song_dir}")

    # Copy original file as-is (preserve exact upload)
    original_ext = audio_path.suffix.lower()
    original_path = song_dir / f"original{original_ext}"
    if not original_path.exists():
        print(f"Copying original file...")
        shutil.copy2(audio_path, original_path)
        print(f"  Saved: original{original_ext}")

    # Copy/move metadata file to song directory
    if md_file:
        dest_md = song_dir / "metadata.md"
        shutil.copy2(md_file, dest_md)
        print(f"  Saved: metadata.md")

    # Convert to m4a for streaming
    audio_m4a_path = song_dir / "audio.m4a"
    if not audio_m4a_path.exists():
        print("Converting to m4a for streaming...")
        convert_to_m4a(audio_path, audio_m4a_path)
        print(f"  Saved: audio.m4a")

    # Ensure WAV for analysis (some analysis functions need WAV)
    wav_path = ensure_wav_for_analysis(audio_path)
    temp_wav = wav_path != audio_path

    # Run analysis
    print("Running analysis...")
    analysis = run_analysis(wav_path, min_bpm, max_bpm)

    # Clean up temp WAV if created
    if temp_wav and wav_path.exists():
        wav_path.unlink()
    analysis["hash"] = content_hash

    analysis_path = song_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved: analysis.json")

    # Generate peaks from the streaming audio
    peaks_dir = song_dir / "peaks"
    peaks_dir.mkdir(exist_ok=True)
    print("Generating waveform peaks...")
    run_peaks(audio_m4a_path, peaks_dir)

    # Create database record via API (skip if --skip-db)
    if skip_db:
        print("Skipping database record (--skip-db)")
    else:
        print("Creating track record via API...")
        song_title = title or audio_path.stem
        suno_url = metadata.suno.url if metadata and metadata.suno.url else None
        try:
            result = create_track_record(
                hash=content_hash,
                title=song_title,
                artist=artist,
                bpm=analysis["bpm"],
                duration=analysis["duration"],
                suno_url=suno_url,
            )
            print(f"  {result.get('action', 'created')} record for {content_hash}")
        except requests.RequestException as e:
            print(f"  Warning: Failed to create track record: {e}")
            print("  Track files saved, but metadata not in database.")

    print(f"\nImport complete: {song_dir}")
    return content_hash


def process_import_folder(
    library_dir: Path,
    import_dir: Path,
    min_bpm: int = 100,
    max_bpm: int = 180,
) -> list[str]:
    """Process all audio files in the import folder.

    Returns:
        List of content hashes for successfully imported songs.
    """
    if not import_dir.exists():
        print(f"Import folder not found: {import_dir}")
        return []

    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}

    # Find all audio files
    audio_files = [
        f for f in import_dir.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        print(f"No audio files found in: {import_dir}")
        return []

    print(f"Found {len(audio_files)} audio file(s) to import\n")

    imported = []
    for audio_file in audio_files:
        try:
            # Check for matching metadata file
            md_file = find_metadata_file(audio_file)

            content_hash = import_song(
                audio_path=audio_file,
                library_dir=library_dir,
                title=None,  # Let metadata or filename be used
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
            imported.append(content_hash)

            # Move processed files to "processed" subfolder
            processed_dir = import_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            dest = processed_dir / audio_file.name
            audio_file.rename(dest)
            print(f"  Moved to: {dest}")

            # Also move the metadata file if it exists
            if md_file and md_file.exists():
                md_dest = processed_dir / md_file.name
                md_file.rename(md_dest)
                print(f"  Moved to: {md_dest}")
            print()

        except Exception as e:
            print(f"Error importing {audio_file.name}: {e}\n")

    return imported


def main():
    parser = argparse.ArgumentParser(
        description="Import songs into the Pain Cave track library"
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        nargs="?",  # Optional - if not provided, process import folder
        help="Path to audio file (wav, mp3, m4a). If omitted, processes import folder."
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("tracks"),
        help="Path to track library directory (default: tracks)"
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        default=Path("tracks/import"),
        help="Path to import folder (default: tracks/import)"
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        help="Track title (default: filename or metadata)"
    )
    parser.add_argument(
        "--artist", "-a",
        type=str,
        help="Artist name"
    )
    parser.add_argument(
        "--min-bpm",
        type=int,
        default=100,
        help="Minimum BPM for beat tracking (default: 100)"
    )
    parser.add_argument(
        "--max-bpm",
        type=int,
        default=180,
        help="Maximum BPM for beat tracking (default: 180)"
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        default=False,
        help="Skip creating database record via API (for importer tool)"
    )

    args = parser.parse_args()

    # If no audio file specified, process import folder
    if args.audio_file is None:
        imported = process_import_folder(
            library_dir=args.library.resolve(),
            import_dir=args.import_dir.resolve(),
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
        )
        if imported:
            print(f"\nImported {len(imported)} track(s):")
            for h in imported:
                print(f"  {h}")
        return

    # Import specific file
    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    content_hash = import_song(
        audio_path=args.audio_file.resolve(),
        library_dir=args.library.resolve(),
        title=args.title,
        artist=args.artist,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        skip_db=args.skip_db,
    )

    print(f"\nHash: {content_hash}")


if __name__ == "__main__":
    main()
