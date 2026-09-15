#!/usr/bin/env python3
"""
Content-based audio hashing for the Pain Cave song library.

Computes MD5 hash of raw audio samples (ignoring file headers/metadata),
truncated to 12 characters for readability and filesystem friendliness.
"""

import hashlib
import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio_samples(audio_path: Path, data: bytes | None = None) -> tuple[np.ndarray, int]:
    """Decode an audio file to mono samples and their sample rate.

    m4a/mp3/aac are decoded by ffmpeg to 44.1 kHz; other formats are read at
    their native sample rate, from `data` (the file's bytes) when given.
    """
    suffix = audio_path.suffix.lower()

    # For m4a/mp3, decode with ffmpeg to get consistent PCM data
    if suffix in ['.m4a', '.mp3', '.aac']:
        with tempfile.TemporaryDirectory(prefix="paincave-hash-") as tmp:
            tmp_path = str(Path(tmp) / "decoded.wav")
            result = subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio_path), '-ar', '44100', '-ac', '1', tmp_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            audio, rate = sf.read(tmp_path)
    else:
        audio, rate = sf.read(io.BytesIO(data) if data is not None else audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio, rate


def fingerprint(audio: np.ndarray, truncate: int = 12) -> str:
    """MD5 of mono samples as float32, the definition behind every track key."""
    return hashlib.md5(audio.astype(np.float32).tobytes()).hexdigest()[:truncate]


def compute_content_hash(audio_path: Path, truncate: int = 12) -> str:
    """Compute a content hash for an audio file.

    The hash is based on the decoded samples, so tags and headers do not
    affect it. Format and sample rate do: the same audio as WAV at different
    rates, or as M4A, hashes differently. It also differs from the paincave
    server's hash.

    Args:
        audio_path: Path to the audio file
        truncate: Number of characters to return (default 12)

    Returns:
        Truncated MD5 hash string (lowercase hex)
    """
    return fingerprint(load_audio_samples(audio_path)[0], truncate)


def main():
    """CLI for testing content hash computation."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute content hash for audio files")
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument("--full", action="store_true", help="Show full MD5 hash")

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}")
        return 1

    truncate = 32 if args.full else 12
    content_hash = compute_content_hash(args.audio_file, truncate=truncate)
    print(content_hash)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
