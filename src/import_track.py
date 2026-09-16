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
import contextlib
import fcntl
import functools
import json
import hashlib
import math
import tempfile
import importlib.metadata
import numpy as np
import soundfile as sf
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

import models
from analyze import NoBeatGrid, analyze_audio, load_audio, require_sound
from content_hash import compute_content_hash, fingerprint
from pcav import (
    PCAVData, AnalysisSection, BeatsSection,
    write_pcav_file, build_peak_mipmaps,
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
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(data)
    json.dumps(data, allow_nan=False)
    # jsonschema's date-time validator is optional; enforce it even without its extras.
    if schema_name == "metadata":
        for name in ("createdAt", "updatedAt"):
            value = data[name]
            if not re.fullmatch(
                    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:[Zz]|[+-][0-9]{2}:[0-9]{2})",
                    value):
                raise ValueError(f"Invalid {name}: expected RFC 3339 date-time")
            datetime.fromisoformat(value.upper())


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
        ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio_path), '-map', '0:a:0', '-c:a', 'pcm_f32le',
         str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion to WAV failed: {result.stderr}")


def start_m4a_encode(audio_path: Path, output_path: Path) -> subprocess.Popen:
    """Start the M4A (AAC) streaming encode; the same decode feeds an EBU R128 loudness measurement.

    Runs alongside analysis. Collect with `finish_m4a_encode`; kill on unwind.
    """
    return subprocess.Popen(
        ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-y', '-i', str(audio_path),
         '-map', '0:a:0', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', str(output_path),
         '-map', '0:a:0', '-af', 'ebur128=peak=true:framelog=quiet', '-f', 'null', '-'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )


def finish_m4a_encode(encode: subprocess.Popen) -> dict:
    """Wait for `start_m4a_encode` and return the loudness measurement."""
    _, stderr = encode.communicate()
    if encode.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion to M4A failed: {stderr}")
    summary = stderr[stderr.rindex("Summary:"):]

    def measured(label, unit):
        value = float(re.search(rf"{label}:\s+(\S+) {unit}", summary).group(1))
        return value if math.isfinite(value) else None

    return {"integratedLufs": measured("I", "LUFS"), "truePeakDbtp": measured("Peak", "dBFS"),
            "loudnessRangeLu": measured("LRA", "LU"), "normalizationApplied": False}


def convert_to_m4a(audio_path: Path, output_path: Path) -> dict:
    """Encode to M4A and measure loudness, waiting for the result."""
    return finish_m4a_encode(start_m4a_encode(audio_path, output_path))


@contextlib.contextmanager
def background_encode(audio_path: Path, output_path: Path):
    """An encode running alongside analysis; killed if the block unwinds before it is collected."""
    encode = start_m4a_encode(audio_path, output_path)
    try:
        yield encode
    finally:
        if encode.poll() is None:
            encode.kill()
            encode.wait()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def probe_audio(path: Path) -> dict:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_streams', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("No audio stream in playback rendition")
    stream = streams[0]
    duration = float(stream.get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid playback duration")
    return {key: stream.get(key) for key in
            ("codec_name", "sample_rate", "channels", "duration", "start_time", "time_base", "bit_rate")}


@contextlib.contextmanager
def timed_stage(name: str):
    """Print `timing: <name> <seconds>s` to stderr for a block of work."""
    start = time.monotonic()
    try:
        yield
    finally:
        print(f"timing: {name} {time.monotonic() - start:.1f}s", file=sys.stderr)


STAGING_NAME = re.compile(r"\.staging-[0-9a-f]{12}-[a-z0-9_]{8}")


def sweep_stale_staging(library_dir: Path, min_age: float = 60) -> None:
    """Remove staging directories whose importer died (SIGKILL, power loss).

    A live import holds an flock on its staging directory; the kernel drops it when the process dies.
    The age check covers the instant between creating a directory and locking it.
    """
    for entry in library_dir.iterdir():
        if not STAGING_NAME.fullmatch(entry.name):
            continue
        try:
            fd = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:  # a symlink, a file, or already removed
            continue
        try:
            stat = os.fstat(fd)
            if stat.st_uid != os.getuid() or time.time() - stat.st_mtime < min_age:
                continue
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        else:
            print(f"Removing stale staging directory: {entry}")
            shutil.rmtree(entry, ignore_errors=True)
        finally:
            os.close(fd)


@contextlib.contextmanager
def staging_directory(library_dir: Path, content_hash: str):
    """A private directory in the library, locked while in use and removed on exit."""
    sweep_stale_staging(library_dir)
    tmp = tempfile.TemporaryDirectory(prefix=f".staging-{content_hash}-", dir=library_dir)
    lock = os.open(tmp.name, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tmp:
            yield Path(tmp.name)
    finally:
        os.close(lock)  # after removal, so a sweep never sees this directory unlocked


def decode_input(audio_path: Path) -> tuple[np.ndarray, int, str | None]:
    """Decode the input once. A WAV is decoded from a single read whose SHA-256 later proves its staged copy holds these samples."""
    if audio_path.suffix.lower() != '.wav':
        return *load_audio(audio_path), None
    data = audio_path.read_bytes()
    return *load_audio(audio_path, data), hashlib.sha256(data).hexdigest()


# Files in a track folder that analysis does not produce and a replacement must carry over.
KEPT_ON_REPLACE = re.compile(r"thumb(_\d+)?\.(avif|webp|png|jpg)|thumbnail\.json|plans|analyses|\.thumbnail\.lock")


def replace_active(song_dir: Path, stage: Path, tmp: Path) -> None:
    """Swap the staged analysis in as the active track folder.

    Everything the stage does not produce (thumbnails, authored plans, candidate revisions) moves into the
    stage first, then two renames swap the folders; the old analysis is removed last. Same filesystem.
    """
    retired = tmp / "retired"
    moved = []
    try:
        for entry in list(song_dir.iterdir()):
            if KEPT_ON_REPLACE.fullmatch(entry.name) and not (stage / entry.name).exists():
                entry.rename(stage / entry.name)
                moved.append(entry.name)
        song_dir.rename(retired)
        try:
            stage.rename(song_dir)
        except OSError:
            retired.rename(song_dir)
            raise
    except OSError:  # nothing lost: the old folder and its kept files come back under their names
        for name in moved:
            (stage / name).rename(song_dir / name)
        raise
    shutil.rmtree(retired)


def enclosing_track_dir(path: Path, library_dir: Path) -> str | None:
    """Key of the library track folder containing path; directories match by identity, names case-insensitively."""
    if not library_dir.is_dir():
        return None
    for folder, child in zip(path.parents, [path, *path.parents]):
        if folder.samefile(library_dir):
            key = child.name.lower()
            return key if child != path and re.fullmatch(r"[0-9a-f]{12}", key) else None
    return None


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
        replace: bool = False,
        source_file: Path | None = None,
        processing_record: Path | None = None,
        first_downbeat: float | None = None,
) -> str:
    """Atomically import a new track, retain a new analysis revision with --force, or replace the active
    analysis with --replace.

    Existing active artifacts and authored data are never replaced. Revisions are
    candidates for review, not automatic changes to the grid used by saved plans.
    """
    total_start = time.monotonic()
    audio_path = audio_path.resolve(strict=True)
    explicit_source = source_file is not None
    source_file = (source_file or audio_path).resolve(strict=True)
    library_dir = library_dir.resolve()
    if not 0 < min_bpm < max_bpm:
        raise ValueError("BPM bounds must satisfy 0 < min_bpm < max_bpm")
    with timed_stage("decode"):
        samples, sample_rate, input_sha256 = decode_input(audio_path)
    require_sound(samples, sample_rate)  # refused before identity checks and staging
    input_hash = fingerprint(samples, truncate=32)
    full_hash = input_hash
    # Audio anywhere inside a track folder (master, candidate, rendition, copy) must prove that track's
    # identity; it never mints a second active track. Other library folders (e.g. import/) hold plain inputs.
    track_dir = enclosing_track_dir(audio_path, library_dir)
    if track_dir:
        previous_manifest = library_dir / track_dir / "manifest.json"
        if previous_manifest.is_file():
            previous = json.loads(previous_manifest.read_text())
            full_hash = previous["audioFingerprint"]["md5"]
            if full_hash[:12] != track_dir:
                raise ValueError("Manifest identity does not match track directory")
            if previous["master"]["sha256"] != (input_sha256 or sha256_file(audio_path)) and input_hash != full_hash:
                raise ValueError(
                    f"Retained master checksum mismatch: {audio_path} is inside track {track_dir} but matches "
                    f"neither its master SHA-256 nor its audio fingerprint; nothing written")
        elif input_hash[:12] != track_dir:
            raise ValueError(f"Unverifiable legacy identity: {audio_path} hashes to {input_hash[:12]}, not its track "
                             f"directory {track_dir}, and no manifest.json proves identity; nothing written")
    content_hash = full_hash[:12]
    print(f"Importing: {audio_path}\n  Hash: {content_hash}")
    song_dir = library_dir / content_hash
    if song_dir.is_symlink():
        raise ValueError("Track directory must not be a symlink")
    existing = song_dir.exists()
    previous_manifest_data = None
    if existing:
        if not song_dir.is_dir():
            raise ValueError("Track path is not a directory")
        # Check the full legacy fingerprint before accepting a shortened-key match.
        manifest_path = song_dir / "manifest.json"
        if manifest_path.exists():
            previous_manifest_data = json.loads(manifest_path.read_text())
            previous_hash = previous_manifest_data["audioFingerprint"]["md5"]
        elif (song_dir / "original.wav").is_file():
            previous_hash = compute_content_hash(song_dir / "original.wav", truncate=32)
        else:
            raise ValueError("Incomplete existing track: retained original is missing")
        if previous_hash != full_hash:
            raise ValueError("Hash collision or unverifiable legacy input; existing track left intact")
        if replace and not force:
            raise ValueError("--replace requires --force")
        if not force:
            for name in ("original.wav", "analysis.pcav", "audio.m4a", "metadata.json", "plan_workzones.json"):
                if not (song_dir / name).is_file():
                    raise ValueError(
                        f"Incomplete existing track: {name} missing; use --force to stage a recovery revision")
            print(f"Track already exists: {song_dir}")
            return content_hash

    if existing and previous_manifest_data and not explicit_source:
        retained_name = previous_manifest_data["source"]["file"]
        if Path(retained_name).name != retained_name:
            raise ValueError("Invalid retained source path")
        source_file = (song_dir / retained_name).resolve(strict=True)
        if sha256_file(source_file) != previous_manifest_data["source"]["sha256"]:
            raise ValueError("Retained source checksum mismatch")
    library_dir.mkdir(parents=True, exist_ok=True)
    # Same filesystem: promotion is one directory rename, never per-file replacement.
    with staging_directory(library_dir, content_hash) as tmp:
        stage = tmp / "track"
        stage.mkdir()
        sha = functools.cache(sha256_file)  # staged files are private to this run: hash each once
        retained_source = stage / ("source" + source_file.suffix.lower())
        shutil.copy2(source_file, retained_source)
        original_wav = stage / "original.wav"
        if audio_path.suffix.lower() == '.wav':
            shutil.copy2(audio_path, original_wav)
            captured_input = original_wav
        else:
            captured_input = retained_source
            if source_file != audio_path:
                captured_input = stage / ("input" + audio_path.suffix.lower())
                shutil.copy2(audio_path, captured_input)
            convert_to_wav(captured_input, original_wav)
        # Detect a moving input before deriving an identity from different bytes.
        if sha256_file(source_file) != sha(retained_source):
            raise ValueError("Source changed during import")
        if (sha(captured_input) != input_sha256 if input_sha256 else compute_content_hash(captured_input,
                                                                                          truncate=32) != input_hash):
            raise ValueError("Input changed during import")
        processing = previous_manifest_data.get(
            "processing") if previous_manifest_data and not explicit_source else None
        if processing_record:
            processing = json.loads(processing_record.read_text())
        if processing is not None:
            if processing.get("inputSha256") != sha(retained_source) or processing.get("outputSha256") != sha(
                    original_wav):
                raise ValueError("Processing provenance does not match source and derived master")
        elif explicit_source and source_file != audio_path:
            raise ValueError("A derived master requires --processing-record provenance")

        # The streaming encode runs on its own core while the detector works; collected before the manifest.
        with background_encode(original_wav, stage / "audio.m4a") as encode:
            if captured_input != original_wav:  # the decoded input was compressed; analysis reads the master
                samples, sample_rate = load_audio(original_wav)
            grid_metrics: dict = {}
            analysis = analyze_audio(original_wav, min_bpm, max_bpm, first_downbeat=first_downbeat,
                                     audio=(samples, sample_rate), grid_out=grid_metrics)
            validate_json(analysis, "analysis")
            info = sf.info(original_wav)
            if analysis["duration"] != info.frames / info.samplerate or analysis["sampleRate"] != sample_rate:
                raise ValueError("Analysis timebase does not match the retained master")
            bar_starts = {b["bar"]: b["t"] for b in analysis["beats"] if b["bar"] > 0 and b["beat"] == 1}
            for zone in analysis["workZones"]:
                start_bar = zone["startBar"]
                next_bar = start_bar + zone["durationBars"]
                if start_bar not in bar_starts or next_bar > max(bar_starts) + 1:
                    raise ValueError("Work zone lies outside the measured bar grid")
                start = bar_starts[start_bar]
                end = bar_starts.get(next_bar, analysis["duration"])
                if abs(zone["startSeconds"] - start) > .006 or abs(zone["durationSeconds"] - (end - start)) > .006:
                    raise ValueError("Work zone times disagree with the measured grid")
            beat_times = analysis["raw"]["beatTimes"]
            first_idx = min(range(len(beat_times)), key=lambda i: abs(beat_times[i] - analysis["firstDownbeat"]))
            with timed_stage("peaks"):
                levels = build_peak_mipmaps(samples, sample_rate)
            pcav_data = PCAVData(
                analysis=AnalysisSection(duration_seconds=info.frames / info.samplerate, bpm=analysis["bpm"],
                                         key=analysis.get("key")),
                beats=BeatsSection(beat_times, first_idx, 4), peaks=levels[0], overview_peaks=levels[1],
            )
            write_pcav_file(stage / "analysis.pcav", pcav_data)
            from pcav import read_pcav_file
            read_pcav_file(stage / "analysis.pcav")
            write_json(stage / "analysis.json", analysis)
            write_json(stage / "plan_workzones.json", {"version": 1, "zones": analysis["workZones"]})
            suno_meta = extract_suno_metadata(retained_source)
            existing_meta = json.loads((song_dir / "metadata.json").read_text()) if existing and (
                    song_dir / "metadata.json").is_file() else {}
            metadata = {
                "hash": content_hash,
                "title": title if title is not None else existing_meta.get("title", source_file.stem),
                "artist": artist if artist is not None else existing_meta.get("artist", "Pain Cave"),
                "source": source,
                "sourceId": source_id or suno_meta.get("id") or existing_meta.get("sourceId", ""),
                "genres": genres if genres is not None else existing_meta.get("genres", []),
                "bpm": analysis["bpm"], "durationSeconds": int(round(info.frames / info.samplerate)),
                "key": analysis.get("key"),
                "createdAt": suno_meta.get("created") or existing_meta.get("createdAt") or get_file_created_time(
                    retained_source),
                "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            validate_json(metadata, "metadata")
            write_json(stage / "metadata.json", metadata)
            with timed_stage("aac+loudness"):  # wait for the encode started before analysis
                loudness = finish_m4a_encode(encode)
            playback = probe_audio(stage / "audio.m4a")
            if abs(float(playback["duration"]) - info.frames / info.samplerate) > 0.05:
                raise ValueError("Playback duration differs from analyzed master by more than 50 ms")
            versions = {}
            for package in ("numpy", "scipy", "soundfile", "beat_this", "torch"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            versions["ffmpeg"] = \
                subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[
                    0]
            with timed_stage("checksums"):
                manifest = {
                    "version": 1, "pcavVersion": 2,
                    "audioFingerprint": {"algorithm": "legacy-mono-float32-md5", "md5": full_hash},
                    "source": {"file": retained_source.name, "name": previous_manifest_data["source"][
                        "name"] if previous_manifest_data and not explicit_source else source_file.name,
                               "sha256": sha(retained_source)},
                    "master": {"file": "original.wav", "sha256": sha(original_wav), "frames": info.frames,
                               "sampleRate": sample_rate, "channels": info.channels, "subtype": info.subtype,
                               "duration": info.frames / info.samplerate},
                    "processing": processing, "playback": playback,
                    "loudness": loudness,
                    "grid": {"firstDownbeatIndex": first_idx, "beatsPerBar": 4, "reviewed": False,
                             "beatCount": len(beat_times), "firstBeat": beat_times[0], "lastBeat": beat_times[-1],
                             "confidenceKind": "grid-heard",
                             "meanConfidence": sum(b["confidence"] for b in analysis["beats"]) / len(beat_times),
                             **grid_metrics},
                    "config": {"minBpm": min_bpm, "maxBpm": max_bpm, "firstDownbeat": first_downbeat,
                               "beatCheckpoint": models.checkpoint_info(),
                               "peakFilterHz": 300, "peakWindows": [256, 8192], "aacBitrate": "256k",
                               "loudnessFilter": "ebur128=peak=true"},
                    "tools": versions,
                    "toolSources": {name: sha256_file(Path(__file__).parent / name) for name in
                                    ("analyze.py", "models.py", "pcav.py", "import_track.py", "content_hash.py")},
                    "artifacts": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in
                                  sorted(stage.iterdir())},
                }
            write_json(stage / "manifest.json", manifest)
            revision = sha256_file(stage / "manifest.json")
        if existing and replace:
            replace_active(song_dir, stage, tmp)
            print(f"Done: {song_dir}")
            print(f"Replaced: {song_dir} (active analysis updated; thumbnails, plans and candidates kept)")
            return content_hash
        if existing:
            revisions = song_dir / "analyses"
            if revisions.is_symlink():
                raise ValueError("Analysis revision directory must not be a symlink")
            revisions.mkdir(exist_ok=True)
            destination = revisions / revision
            if destination.exists():
                raise FileExistsError("Analysis revision already exists")
        else:
            destination = song_dir
        # A concurrently created track is never replaced (a complete track is nonempty).
        stage.rename(destination)
        print(f"Done: {destination}")
        if existing:
            print(f"Revision: {destination} (candidate; active files and authored edits unchanged)")
    print(f"timing: total {time.monotonic() - total_start:.1f}s", file=sys.stderr)
    return content_hash


def main():
    parser = argparse.ArgumentParser(
        description="Import songs into the Pain Cave track library"
    )
    parser.add_argument("audio_file", type=Path, nargs="?", help="Path to audio file")
    parser.add_argument("--library", type=Path, default=DEFAULT_TRACKS_DIR,
                        help="Track library directory (default: $PAINCAVE_TRACKS_DIR)")
    parser.add_argument("-t", "--title", type=str, help="Track title (default: filename)")
    parser.add_argument("-a", "--artist", type=str, help="Artist name (default: Pain Cave)")
    parser.add_argument("--source", type=str, default="SUNO", choices=["SUNO"],
                        help="Audio source platform")
    parser.add_argument("--source-id", type=str, help="ID from source platform")
    parser.add_argument("-g", "--genres", type=str, help="Comma-separated genre slugs")
    parser.add_argument("--min-bpm", type=int, default=100, help="Minimum BPM (default: 100)")
    parser.add_argument("--max-bpm", type=int, default=180, help="Maximum BPM (default: 180)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Stage a candidate analysis revision (write-once by convention); preserve active files")
    parser.add_argument("--replace", action="store_true",
                        help="With --force: make the new analysis the active one (thumbnails, plans and candidates are kept)")

    parser.add_argument("--source-file", type=Path, help="Retain the unprocessed source download")
    parser.add_argument("--processing-record", type=Path, help="Audio preparation provenance JSON")
    parser.add_argument("--first-downbeat", type=float, help="Override bar 1 start (snapped to nearest grid beat)")
    parser.add_argument("--check-model", action="store_true",
                        help="Print the beat detector checkpoint (downloading it if needed) and exit")
    args = parser.parse_args()

    if args.check_model:
        checkpoint = models.ensure_checkpoint()
        print(f"beat_this {importlib.metadata.version('beat_this')}, torch {importlib.metadata.version('torch')}")
        print(f"checkpoint {checkpoint['name']}: {checkpoint['file']}\n  sha256 {checkpoint['sha256']}")
        return

    if args.audio_file is None:
        raise SystemExit("Error: audio_file is required")
    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    genres = [g.strip() for g in args.genres.split(",") if g.strip()] if args.genres else None

    def terminate(signum, _frame):
        # One-shot (a group signal and uv's forwarded copy can both arrive) and no I/O: a failed write would
        # replace SystemExit with an Exception that `except Exception` blocks swallow. Unwinding removes the stage.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)

    try:
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
            replace=args.replace,
            source_file=args.source_file,
            processing_record=args.processing_record,
            first_downbeat=args.first_downbeat,
        )
    except NoBeatGrid as e:
        raise SystemExit(f"Error: refused {args.audio_file.name}: {e}; nothing written")

    print(f"\nHash: {content_hash}")


if __name__ == "__main__":
    main()
