#!/usr/bin/env python3
"""
A track's audio.m4a tags (docs/output-formats.md → audio.m4a tags): title, artist, genres, BPM, artwork and Pain Cave's
own fields, from the folder's metadata.json and thumb.avif, so the file reads right in any player or DJ tool. Only the
tags change: a tagged copy replaces the file only once its audio matches the original packet for packet and in
duration. A file already carrying its tags is left alone.

    tag_track.py <track_dir>... --genres genres.json            # tag what's missing or stale
    tag_track.py <track_dir>... --genres genres.json --check    # list it, write nothing (exit 1 when any)
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm

FREEFORM = "----:com.paincave:"
# The artwork's own fingerprint (thumb.avif's SHA-256): says whether `covr` is current without converting it again, and
# doesn't depend on ffmpeg's JPEG encoder staying byte for byte the same
COVER = f"{FREEFORM}COVER"
# Every tag Pain Cave writes: absent from `wanted_tags` means removed. ©cmt and ©too (Suno's provenance, the encoder)
# are analysis's and never touched.
OWNED = ("\xa9nam", "\xa9ART", "\xa9gen", "tmpo", "\xa9alb", "\xa9day", "covr",
         f"{FREEFORM}HASH", f"{FREEFORM}BPM", f"{FREEFORM}SOURCE", f"{FREEFORM}SOURCE_ID", COVER)


def load_genre_names(path: Path) -> dict[str, str]:
    """Genre slug → name, from paincave's shared/data/genres.json"""
    return {genre["slug"]: genre["name"] for genre in json.loads(path.read_text())}


def cover_jpeg(artwork: Path) -> bytes:
    """The artwork as JPEG (MP4 cover art is JPEG or PNG; the thumbnails are AVIF). The same input gives the same bytes"""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(artwork), "-frames:v", "1", "-q:v", "3", "-f", "mjpeg", "-"],
        capture_output=True, check=True,
    )
    return result.stdout


def wanted_tags(track_dir: Path, genre_names: dict[str, str]) -> dict[str, list]:
    """The tags this track's audio.m4a should carry, but `covr` itself: the artwork's fingerprint stands for it"""
    meta = json.loads((track_dir / "metadata.json").read_text())
    text = lambda value: [MP4FreeForm(str(value).encode())]
    tags: dict[str, list] = {
        "\xa9nam": [meta["title"]],
        "\xa9ART": [meta.get("artist", "Pain Cave")],
        "\xa9day": [meta["createdAt"][:10]],
        f"{FREEFORM}HASH": text(meta["hash"]),
    }
    if meta.get("genres"):
        tags["\xa9gen"] = [genre_names.get(slug, slug) for slug in meta["genres"]]
    if meta.get("bpm"):
        tags["tmpo"] = [round(meta["bpm"])]
        tags[f"{FREEFORM}BPM"] = text(f"{meta['bpm']:.2f}")
    if meta.get("cutFrom"):
        tags["\xa9alb"] = [meta["cutFrom"]]
    if meta.get("source"):
        tags[f"{FREEFORM}SOURCE"] = text(meta["source"])
    if meta.get("sourceId"):
        tags[f"{FREEFORM}SOURCE_ID"] = text(meta["sourceId"])
    artwork = track_dir / "thumb.avif"
    if artwork.exists():
        tags[COVER] = text(hashlib.sha256(artwork.read_bytes()).hexdigest())
    return tags


def _plain(values: list) -> list:
    return [bytes(value) if isinstance(value, bytes) else value for value in values]


def is_tagged(audio: Path, wanted: dict[str, list]) -> bool:
    tags = MP4(audio).tags or {}
    return (all(_plain(tags.get(key, [])) == _plain(wanted.get(key, [])) for key in OWNED if key != "covr")
            and ("covr" in tags) == (COVER in wanted))


def audio_identity(path: Path) -> tuple[str, str]:
    """The audio packets' MD5 and the audio stream's duration: what tagging must not change"""
    packets = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return packets, duration


def track_audio(track_dir: Path) -> Path:
    audio = track_dir / "audio.m4a"
    if not (track_dir / "metadata.json").is_file() or not audio.is_file():
        raise ValueError(f"Not a track folder (no metadata.json and audio.m4a): {track_dir}")
    return audio


def tag_track(track_dir: Path, genre_names: dict[str, str]) -> bool:
    """Tag the track's audio.m4a; False when it already carried its tags (nothing written)"""
    audio = track_audio(track_dir)
    wanted = wanted_tags(track_dir, genre_names)
    if is_tagged(audio, wanted):
        return False
    handle, name = tempfile.mkstemp(prefix=".audio.m4a.tagging-", dir=track_dir)
    os.close(handle)
    temp = Path(name)
    try:
        shutil.copyfile(audio, temp)
        shutil.copymode(audio, temp)
        file = MP4(temp)
        if file.tags is None:
            file.add_tags()
        for key in OWNED:
            file.tags.pop(key, None)
        file.tags.update(wanted)
        if COVER in wanted:
            file.tags["covr"] = [MP4Cover(cover_jpeg(track_dir / "thumb.avif"), imageformat=MP4Cover.FORMAT_JPEG)]
        file.save()
        if audio_identity(temp) != audio_identity(audio):
            raise RuntimeError(f"Tagging changed the audio of {audio}; left as it was")
        with open(temp, "rb") as written:
            os.fsync(written.fileno())
        os.replace(temp, audio)
    finally:
        temp.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a track's audio.m4a tags from its metadata.json and artwork")
    parser.add_argument("track_dirs", nargs="+", type=Path, help="Track folders <library>/<hash>")
    parser.add_argument("--genres", type=Path, required=True, help="paincave's shared/data/genres.json")
    parser.add_argument("--check", action="store_true", help="List folders whose tags are missing or stale; write nothing")
    args = parser.parse_args()

    def terminate(signum, _frame):
        # One-shot and no I/O, as import_track.py's: unwinding removes a half-made tagged copy
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    genre_names = load_genre_names(args.genres)
    failed = False
    for track_dir in args.track_dirs:
        try:
            if args.check:
                if not is_tagged(track_audio(track_dir), wanted_tags(track_dir, genre_names)):
                    print(f"Stale: {track_dir}")
                    failed = True
            else:
                print(f"{'Tagged' if tag_track(track_dir, genre_names) else 'Up to date'}: {track_dir}")
        except Exception as error:
            print(f"Error: {track_dir}: {error}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
