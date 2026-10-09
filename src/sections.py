#!/usr/bin/env python3
"""
Section boundaries and repetition labels for Songlab reference analysis, from the audio alone.

A dependency-light baseline (numpy and scipy only): bar-synchronous harmony (chroma), timbre (MFCC) and,
when the stems stage ran, which stems play. Each feature gives a bar-by-bar self-similarity matrix; a
Foote checkerboard kernel slid along the diagonal scores how much the song changes at the start of each
bar (novelty). Peaks become boundaries, nudged toward 4-bar phrases; segments that resemble each other
bar for bar share a letter (A, B, C in order of first appearance).

Labels are repetition groups, not names: deciding which group is the chorus is Songlab's job, with the
lyrics' repeats and Mike's notes.
"""

import numpy as np
from scipy import fft, signal
from scipy.cluster import hierarchy

METHOD = "ssm"
VERSION = 1
N_FFT = 4096
HOP = 1024
CHROMA_LOW_HZ, CHROMA_HIGH_HZ = 55.0, 5000.0
MEL_BANDS, MEL_LOW_HZ, MEL_HIGH_HZ = 40, 30.0, 11000.0
MFCC_COUNT = 13
KERNEL_BARS = (4, 8)  # half-widths of the checkerboard kernels: phrase and section scale
MIN_SECTION_BARS = 2
PHRASE_BARS = 4
PHRASE_WEIGHTS = (1.0, 0.55, 0.75, 0.55)  # novelty weight by bar position inside a 4-bar phrase
PEAK_THRESHOLD = 0.3  # of the weighted novelty's range above its median
LABEL_DISTANCE = 0.45  # average-linkage cut: segments closer than this share a letter
QUIET_DB = -40.0  # trailing bars quieter than this are a fade, folded into the last section


def _spectrogram(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT (bins x frames), bin frequencies and frame centre times."""
    freqs, times, z = signal.stft(samples, rate, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=True)
    return np.abs(z), freqs, times


def chroma(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """12 x frames pitch-class energy, log-compressed."""
    keep = (freqs >= CHROMA_LOW_HZ) & (freqs <= CHROMA_HIGH_HZ)
    pitch_class = np.round(69 + 12 * np.log2(freqs[keep] / 440.0)).astype(int) % 12
    out = np.zeros((12, mag.shape[1]))
    np.add.at(out, pitch_class, mag[keep] ** 2)
    return np.log1p(100 * out / (out.max() or 1))


def _mel_filters(freqs: np.ndarray, rate: int) -> np.ndarray:
    def mel(f):
        return 2595 * np.log10(1 + f / 700)

    def hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    edges = hz(np.linspace(mel(MEL_LOW_HZ), mel(min(MEL_HIGH_HZ, rate / 2)), MEL_BANDS + 2))
    filters = np.zeros((MEL_BANDS, len(freqs)))
    for i in range(MEL_BANDS):
        lo, centre, hi = edges[i:i + 3]
        rising = (freqs - lo) / (centre - lo)
        falling = (hi - freqs) / (hi - centre)
        filters[i] = np.clip(np.minimum(rising, falling), 0, None)
    return filters


def mfcc(mag: np.ndarray, freqs: np.ndarray, rate: int) -> np.ndarray:
    """MFCC 1..13 x frames (coefficient 0, overall level, left out: loudness is measured separately)."""
    log_mel = np.log10(_mel_filters(freqs, rate) @ (mag ** 2) + 1e-10)
    return fft.dct(log_mel, type=2, norm="ortho", axis=0)[1:MFCC_COUNT + 1]


def beat_sync(frames: np.ndarray, times: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Mean of the frames between consecutive beat times: features x (beats - 1)."""
    edges = np.searchsorted(times, beat_times)
    out = np.zeros((frames.shape[0], len(beat_times) - 1))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if b > a:
            out[:, i] = frames[:, a:b].mean(axis=1)
        elif a < frames.shape[1]:
            out[:, i] = frames[:, min(a, frames.shape[1] - 1)]
    return out


def bar_stack(per_beat: np.ndarray, first: int, bar_count: int, per_bar: int) -> np.ndarray:
    """Bars x (features * beats per bar): each bar's beats side by side, so a bar's pattern counts, not just its mix."""
    rows = []
    for bar in range(bar_count):
        start = first + bar * per_bar
        rows.append(per_beat[:, start:start + per_bar].T.reshape(-1))
    return np.array(rows)


def self_similarity(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of z-scored rows, mapped to 0-1."""
    z = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-9)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    unit = z / np.where(norms > 0, norms, 1)
    return (unit @ unit.T + 1) / 2


def checkerboard(half: int) -> np.ndarray:
    sign = np.concatenate([-np.ones(half), np.ones(half)])
    taper = signal.windows.gaussian(2 * half, std=half / 2)
    return np.outer(sign * taper, sign * taper)


def novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    """Change score at the start of each bar (index b = boundary between bar b-1 and bar b), 0-1."""
    n = len(ssm)
    padded = np.pad(ssm, half, mode="reflect")
    kernel = checkerboard(half)
    score = np.array([np.sum(padded[b:b + 2 * half, b:b + 2 * half] * kernel) for b in range(n)])
    score = np.clip(score, 0, None)
    score[0] = 0
    return score / (score.max() or 1)


def phrase_offset(score: np.ndarray) -> int:
    """Which bar position (0-3, counting from bar 1) phrases start on: the one carrying the most novelty."""
    return int(np.argmax([score[o::PHRASE_BARS].sum() for o in range(PHRASE_BARS)]))


def pick_boundaries(score: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Bar indices (0-based) where sections start, always including 0, and the phrase-weighted novelty."""
    offset = phrase_offset(score)
    weights = np.array([PHRASE_WEIGHTS[(b - offset) % PHRASE_BARS] for b in range(len(score))])
    weighted = score * weights
    median = float(np.median(weighted))
    height = median + PEAK_THRESHOLD * (float(weighted.max()) - median)
    peaks, _ = signal.find_peaks(np.concatenate([[0], weighted[1:], [0]]), height=height, distance=MIN_SECTION_BARS)
    return [0] + [int(p) for p in peaks if 0 < p < len(score)], weighted


def segment_similarity(ssm: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """How alike two segments are bar for bar (aligned at their starts), discounted for unequal lengths."""
    la, lb = a[1] - a[0], b[1] - b[0]
    m = min(la, lb)
    if m == 0:
        return 0.0
    aligned = float(np.mean([ssm[a[0] + i, b[0] + i] for i in range(m)]))
    return aligned * (0.75 + 0.25 * m / max(la, lb))


def label_segments(ssm: np.ndarray, segments: list[tuple[int, int]]) -> list[str]:
    """Letters by repetition group, A first."""
    if len(segments) == 1:
        return ["A"]
    n = len(segments)
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance[i, j] = distance[j, i] = 1 - segment_similarity(ssm, segments[i], segments[j])
    condensed = distance[np.triu_indices(n, 1)]
    groups = hierarchy.fcluster(hierarchy.linkage(condensed, method="average"), LABEL_DISTANCE, criterion="distance")
    letters: dict[int, str] = {}
    for g in groups:
        letters.setdefault(int(g), chr(ord("A") + len(letters)) if len(letters) < 26 else "Z")
    return [letters[int(g)] for g in groups]


def analyze_sections(
        samples: np.ndarray,
        rate: int,
        beats: list[dict],
        bars: list[dict],
        stem_levels: dict[str, list[float]] | None = None,
        beats_per_bar: int = 4,
) -> tuple[dict, list[float]]:
    """The `sections.methods.ssm` block and the per-bar novelty (one per bar, boundary at the bar's start).

    `beats` and `bars` are the analysis grid; `stem_levels` the per-bar 0-1 stem levels, when measured.
    Sections are contiguous and cover every bar: `endBar` is exclusive.
    """
    n = len(bars)
    first = next(i for i, b in enumerate(beats) if b["bar"] == 1 and b["beat"] == 1)
    beat_times = np.array([b["time"] for b in beats] + [bars[-1]["end"]])
    mag, freqs, times = _spectrogram(samples, rate)
    loudness = np.array([b.get("loudness", 0.0) for b in bars])

    matrices = [
        self_similarity(bar_stack(beat_sync(chroma(mag, freqs), times, beat_times), first, n, beats_per_bar)),
        self_similarity(bar_stack(beat_sync(mfcc(mag, freqs, rate), times, beat_times), first, n, beats_per_bar)),
    ]
    if stem_levels:
        levels_db = np.array([[max(-40.0, 20 * np.log10(v)) if v > 0 else -40.0 for v in stem_levels[name]]
                              for name in sorted(stem_levels)]).T
        matrices.append(self_similarity(levels_db))
    ssm = np.mean(matrices, axis=0)

    score = np.mean([novelty(ssm, h) for h in KERNEL_BARS if 2 * h <= n] or [np.zeros(n)], axis=0)
    starts, weighted = pick_boundaries(score)

    # A quiet tail (the fade) belongs to the section it fades out of.
    tail = n
    while tail > 1 and loudness[tail - 1] < QUIET_DB:
        tail -= 1
    starts = [s for s in starts if s < tail or s == 0]
    segments = list(zip(starts, starts[1:] + [n]))
    labels = label_segments(ssm, segments)
    sections = [{"startBar": a + 1, "endBar": b + 1, "label": label,
                 "confidence": round(float(weighted[a]) if a else 1.0, 2)}
                for (a, b), label in zip(segments, labels)]
    block = {"tool": METHOD, "version": VERSION, "sections": sections,
             "boundariesSeconds": [bars[a]["start"] for a, _ in segments[1:]]}
    return block, [round(float(v), 3) for v in score]
