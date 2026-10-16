#!/usr/bin/env python3
"""
Four-stem split for Songlab reference analysis: Demucs `htdemucs_ft` into drums, bass, other, vocals. The
fine-tuned bag of four models, one per stem: cleaner stems for about four times the time, and the only model
Pain Cave uses (owner 2026-10-16: "default on using demucs_ft ALWAYS ... for everything").

Only `reference.py` imports this, and Demucs itself is imported inside `separate`, so the Pain Cave import
path never loads it. Demucs lives in the optional `reference` dependency group (`uv run --group reference`).

The stems are cached as FLAC under the cache directory, keyed by the input's SHA-256 and the model name,
so a re-run with another tempo bound or bar 1 reuses them instead of separating again.
"""

import hashlib
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "htdemucs_ft"
STEMS = ("drums", "bass", "other", "vocals")
RATE = 44100  # Demucs models are trained at 44.1 kHz stereo
OVERLAP = 0.25

Progress = Callable[[float], None]


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_stereo(audio_path: Path) -> np.ndarray:
    """Decode any format to float32 stereo at 44.1 kHz, shape (2, samples). Mono input is duplicated."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_path), "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32).reshape(-1, 2).T.copy()


def stem_dir(cache_dir: Path, sha256: str) -> Path:
    return cache_dir / f"stems-{sha256[:16]}-{MODEL}"


def cached_stems(cache_dir: Path, sha256: str) -> dict[str, Path] | None:
    """The cached stem files for this input, or None unless all four are there."""
    folder = stem_dir(cache_dir, sha256)
    files = {name: folder / f"{name}.flac" for name in STEMS}
    return files if all(path.is_file() for path in files.values()) else None


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _with_certificates() -> None:
    """The python.org framework build has no CA bundle; torch hub needs one to fetch the checkpoint."""
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


class _ProgressBar:
    """Stands in for `tqdm.tqdm` inside `demucs.apply`, which reports chunk completion only through it."""

    def __init__(self, report: Progress):
        self.report = report

    def __call__(self, futures, **_kwargs):
        futures = list(futures)
        for done, future in enumerate(futures, start=1):
            yield future
            self.report(done / len(futures))


def separate(mix: np.ndarray, report: Progress = lambda _fraction: None) -> dict[str, np.ndarray]:
    """Split a (2, samples) mix at 44.1 kHz into the four stems, each (2, samples) float32."""
    import torch
    from demucs import apply
    from demucs.pretrained import get_model

    _with_certificates()
    model = get_model(MODEL)
    model.eval()
    wav = torch.from_numpy(mix)
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std()
    if std == 0:
        raise ValueError("audio is digital silence")
    original_tqdm = apply.tqdm.tqdm
    apply.tqdm.tqdm = _ProgressBar(report)
    try:
        with torch.no_grad():
            # shifts=0: the random time shift would make two runs differ
            out = apply.apply_model(model, ((wav - mean) / std)[None], device=_device(), shifts=0, split=True,
                                    overlap=OVERLAP, progress=True)[0]
    finally:
        apply.tqdm.tqdm = original_tqdm
    out = out * std + mean
    return {name: out[model.sources.index(name)].numpy().astype(np.float32) for name in STEMS}


VOCALS_CHUNK = 300  # seconds separated at a time: a 45-minute class would need ~4 GB for all four stems at once
VOCALS_CONTEXT = 10  # seconds either side of a chunk, heard but dropped, so its edges sound like its middle


def separate_vocals(mix: np.ndarray, report: Progress = lambda _fraction: None) -> np.ndarray:
    """
    The vocals of a (2, samples) mix at 44.1 kHz, mono float32, for a recording of any length. `htdemucs_ft` weighs
    each stem by one of its four models alone (one-hot weights), so its vocals model run by itself gives the bag's
    vocals for a quarter of the time. Separated in chunks, each with context either side.
    """
    import torch
    from demucs import apply
    from demucs.pretrained import get_model

    _with_certificates()
    bag = get_model(MODEL)
    index = bag.sources.index("vocals")
    weights = np.array(bag.weights)
    if sorted(weights[:, index]) != [0] * (len(bag.models) - 1) + [1]:
        raise RuntimeError(f"{MODEL} doesn't take its vocals from one model alone: {bag.weights}")
    model = bag.models[int(weights[:, index].argmax())]
    model.eval()
    wav = torch.from_numpy(mix)
    reference = wav.mean(0)
    mean, std = reference.mean(), reference.std()
    if std == 0:
        raise ValueError("audio is digital silence")
    del reference
    wav = (wav - mean) / std
    length, chunk, context = wav.shape[1], VOCALS_CHUNK * RATE, VOCALS_CONTEXT * RATE
    starts = range(0, length, chunk)
    vocals = np.zeros(length, dtype=np.float32)
    original_tqdm = apply.tqdm.tqdm
    try:
        for done, start in enumerate(starts):
            apply.tqdm.tqdm = _ProgressBar(lambda fraction: report((done + fraction) / len(starts)))
            heard_from, end = max(0, start - context), min(length, start + chunk)
            with torch.no_grad():
                out = apply.apply_model(model, wav[None, :, heard_from:min(length, end + context)], device=_device(),
                                        shifts=0, split=True, overlap=OVERLAP, progress=True)[0]
            voice = (out[model.sources.index("vocals")] * std + mean).mean(0).numpy()
            vocals[start:end] = voice[start - heard_from:end - heard_from]
    finally:
        apply.tqdm.tqdm = original_tqdm
    return vocals


def write_stems(stems: dict[str, np.ndarray], folder: Path) -> dict[str, Path]:
    """Write each stem as 16-bit FLAC; each file appears whole or not at all."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, samples in stems.items():
        path = folder / f"{name}.flac"
        temporary = folder / f".{name}.tmp.flac"
        try:
            sf.write(temporary, samples.T, RATE, subtype="PCM_16")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        files[name] = path
    return files


def load_stems(files: dict[str, Path]) -> dict[str, np.ndarray]:
    """Read cached stems back as mono float arrays at 44.1 kHz."""
    stems = {}
    for name, path in files.items():
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != RATE:
            raise ValueError(f"cached stem {path} is {rate} Hz, expected {RATE}")
        stems[name] = samples.mean(axis=1)
    return stems


def stems_for(audio_path: Path, mix: np.ndarray, cache_dir: Path, report: Progress) -> dict[str, Path]:
    """Cached stem files for `audio_path`, separating first when the cache is cold."""
    sha256 = file_sha256(audio_path)
    files = cached_stems(cache_dir, sha256)
    if files is not None:
        report(1.0)
        return files
    return write_stems(separate(mix, report), stem_dir(cache_dir, sha256))
