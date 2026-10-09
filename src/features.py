#!/usr/bin/env python3
"""
Per-bar measurements for Songlab reference analysis: loudness of the mix and of each stem, plus the
waveform peaks the Lab draws. Pure numpy on decoded samples; times are seconds, bars come from the grid.
"""

import numpy as np

SILENCE_DB = -120.0
REL_DB = 24.0  # a stem counts as playing in a bar within this many dB of its own loudest bar...
FLOOR_DB = -50.0  # ...and louder than this in absolute terms (dBFS RMS)
PEAKS_PER_SECOND = 50


def rms_db(samples: np.ndarray) -> float:
    """RMS level in dBFS, floored at SILENCE_DB for silence or an empty slice."""
    if not len(samples):
        return SILENCE_DB
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(SILENCE_DB, 20 * np.log10(rms)) if rms > 0 else SILENCE_DB


def bar_slices(bars: list[dict], rate: int, length: int) -> list[slice]:
    return [slice(min(length, int(round(b["start"] * rate))), min(length, int(round(b["end"] * rate)))) for b in bars]


def bar_loudness(samples: np.ndarray, rate: int, bars: list[dict]) -> list[float]:
    """Mix loudness per bar in dBFS RMS, one decimal."""
    return [round(rms_db(samples[s]), 1) for s in bar_slices(bars, rate, len(samples))]


def stem_levels(stems: dict[str, np.ndarray], rate: int, bars: list[dict]) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Each stem's per-bar RMS relative to its own loudest bar (linear, 0-1, three decimals), and that peak in dBFS.

    The absolute level of a bar is `peakDb + 20*log10(level)`.
    """
    levels, peaks = {}, {}
    for name, samples in stems.items():
        db = np.array([rms_db(samples[s]) for s in bar_slices(bars, rate, len(samples))])
        peak = float(db.max()) if len(db) else SILENCE_DB
        peaks[name] = round(peak, 1)
        levels[name] = [round(float(10 ** ((d - peak) / 20)), 3) if peak > SILENCE_DB else 0.0 for d in db]
    return levels, peaks


def is_active(level: float, peak_db: float, rel_db: float = REL_DB, floor_db: float = FLOOR_DB) -> bool:
    """Whether a stem plays in a bar: within `rel_db` of its loudest bar and above `floor_db` absolute."""
    if level <= 0:
        return False
    return 20 * np.log10(level) >= -rel_db and peak_db + 20 * np.log10(level) >= floor_db


def peaks(samples: np.ndarray, rate: int, per_second: int = PEAKS_PER_SECOND) -> list[list[float]]:
    """[min, max] per window of 1/per_second seconds, three decimals."""
    window = max(1, rate // per_second)
    count = int(np.ceil(len(samples) / window))
    padded = np.zeros(count * window, dtype=np.float32)
    padded[:len(samples)] = samples
    frames = padded.reshape(count, window)
    return [[round(float(lo), 3), round(float(hi), 3)] for lo, hi in zip(frames.min(axis=1), frames.max(axis=1))]


def activity_matrix(bars: list[dict], levels: dict[str, list[float]], peaks_db: dict[str, float]) -> str:
    """A text table, one row per bar: which stems play (letter) and the mix loudness, for reading by ear."""
    names = list(levels)
    lines = ["bar   time   " + " ".join(n[0].upper() for n in names) + "   dB"]
    for i, bar in enumerate(bars):
        marks = " ".join(n[0].upper() if is_active(levels[n][i], peaks_db[n]) else "." for n in names)
        minutes, seconds = divmod(bar["start"], 60)
        lines.append(f"{bar['bar']:>3}  {int(minutes)}:{seconds:05.2f}  {marks}  {bar.get('loudness', '')}")
    return "\n".join(lines)
