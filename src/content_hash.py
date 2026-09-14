#!/usr/bin/env python3
"""
Content-based audio hashing for the Pain Cave song library.

Computes MD5 hash of raw audio samples (ignoring file headers/metadata),
truncated to 12 characters for readability and filesystem friendliness.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio_samples(audio_path: Path) -> np.ndarray:
    """Load audio file and return raw samples as a numpy array.

    m4a/mp3/aac are decoded by ffmpeg to 44.1 kHz; other formats are read at
    their native sample rate. Returns mono audio.
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
            audio, _ = sf.read(tmp_path)
    else:
        audio, _ = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio


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
    audio = load_audio_samples(audio_path)

    # Convert to bytes for hashing
    # Use float32 to ensure consistent representation
    audio_bytes = audio.astype(np.float32).tobytes()

    # Compute MD5 hash
    md5 = hashlib.md5(audio_bytes).hexdigest()

    return md5[:truncate]


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
