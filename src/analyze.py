#!/usr/bin/env python3
"""
Audio analysis for Pain Cave spin class workouts.

Extracts:
- Beat grid with downbeats and bar positions
- Work zones (regions with consistent kicks vs breakdowns)

Beats come from the Beat This! detector (CPJKU, MIT); a constant-tempo grid is fitted on top of its
detections, because plans are bar ranges over an unbroken grid and the detector emits nothing in quiet
intros and breakdowns.

Output uses camelCase JSON keys for direct consumption by TypeScript.
"""

import argparse
import functools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

import models
from content_hash import load_audio_samples

MIN_DURATION_SECONDS = 1
MIN_BEATS = 8  # two bars of 4/4: a tempo median, a downbeat and one complete bar
BEATS_PER_BAR = 4
MAX_GRID_RESIDUAL_MS = 60  # above this the detections do not describe a rideable grid
LOCAL_WINDOWS = (32, 64, 128)  # beats each side of a grid beat: 8, 16 and 32 bars
LOCAL_MARGIN = 8  # extra beat slots fitted past each end before the grid is trimmed to the audio


class NoBeatGrid(ValueError):
    """Audio no ride can use: shorter than 1 s, digital silence, or without a measurable beat grid."""


def load_audio(audio_path: Path, data: bytes | None = None) -> tuple[np.ndarray, int]:
    """Load audio as mono, with the same decode as the content hash. `data` is the file's bytes, if already read."""
    print(f"Loading audio: {audio_path}")

    audio, sr = load_audio_samples(audio_path, data)

    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain nonempty finite samples")

    print(f"Duration: {len(audio) / sr:.1f}s ({len(audio) / sr / 60:.1f} minutes)")
    print(f"Sample rate: {sr} Hz")

    return audio, sr


def require_sound(audio: np.ndarray, sr: int) -> None:
    """Refuse audio too short or too quiet to carry a beat grid."""
    duration = len(audio) / sr
    if duration < MIN_DURATION_SECONDS:
        raise NoBeatGrid(f"audio is {duration:.3f} s long; at least {MIN_DURATION_SECONDS} s required")
    if not np.any(audio):
        raise NoBeatGrid("audio is digital silence")


@functools.cache
def load_detector(checkpoint: str = models.CHECKPOINT):
    """Load the Beat This! detector once per process; the checkpoint downloads into the torch hub cache."""
    import torch
    from beat_this.inference import Audio2Beats

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    return Audio2Beats(checkpoint_path=checkpoint, device="cpu", dbn=False)


def extract_beats(audio: np.ndarray, sr: int, min_bpm: int = 100, max_bpm: int = 180) -> tuple[
    np.ndarray, np.ndarray]:
    """Detect beats and downbeats in already-decoded mono samples (Beat This! resamples to 22.05 kHz itself).

    Times are quantized to the detector's 20 ms frames and stop wherever it hears no beat; the tempo bounds
    apply to `fit_beat_grid`, which turns these detections into the continuous grid the app rides on.
    """
    print("Extracting beats with Beat This!...")

    beats, downbeats = load_detector()(audio, sr)
    beats, downbeats = np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)

    print(f"Detected {len(beats)} beats, {len(downbeats)} downbeats")

    return beats, downbeats


@dataclass(frozen=True)
class GridFit:
    """A beat grid fitted to detections, covering the whole track."""

    period: float           # seconds per beat of the global line: the reference tempo
    phase: float            # time of grid index 0 on that line; may lie before the audio
    beat_times: np.ndarray  # every grid beat with 0 <= t < duration
    index: np.ndarray       # grid index of each beat time, counted from `phase`
    heard: np.ndarray       # per grid beat: a detection landed on it
    bar_phase: int          # grid index modulo 4 that carries the downbeats
    residual_ms: float      # RMS distance between heard grid beats and their detections

    @property
    def bpm(self) -> float:
        """Tempo of the median grid interval; local tempo varies with the drift the grid follows."""
        return 60 / float(np.median(np.diff(self.beat_times)))


def _nearest_index(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """For each target, the index of the closest of the (sorted, nonempty) values."""
    after = np.searchsorted(values, targets)
    left = np.clip(after - 1, 0, len(values) - 1)
    right = np.clip(after, 0, len(values) - 1)
    return np.where(np.abs(targets - values[left]) <= np.abs(targets - values[right]), left, right)


def _fit_line(k: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Least-squares t = t0 + k·period, computed in closed form so the result never depends on a BLAS."""
    spread = k - k.mean()
    variance = float(spread @ spread)
    if variance <= 0:
        raise NoBeatGrid("no beat grid: every detection falls on one beat position")
    period = float(spread @ (t - t.mean()) / variance)
    return period, float(t.mean() - period * k.mean())


def _beat_indices(beats: np.ndarray, period: float) -> np.ndarray:
    """Number the detections, counting beats from one detection to the next.

    Stepping locally keeps an imprecise starting period (the median of 20 ms-quantized intervals is a
    multiple of 20 ms) from accumulating into a misnumbered beat halfway through the track. Detections
    closer together than half a period share an index, so a double-tempo reading collapses onto the
    octave-corrected grid instead of stretching it.
    """
    return np.concatenate([[0.0], np.cumsum(np.round(np.diff(beats) / period))])


def _local_line(k: np.ndarray, t: np.ndarray, n: float, fallback: tuple[float, float],
                bounds: tuple[float, float]) -> tuple[float, float]:
    """Least-squares line over the detections neighbouring grid beat `n`, widening until 8 of them fit.

    Past either end of the detections the window stays anchored on the last of them, so the grid leaves
    the music on one straight line instead of on a slope refitted from a handful of stragglers. A window
    whose tempo falls outside the bounds is a section the detector heard in half- or double-time; widen
    past it rather than bend the grid to it.
    """
    center = min(max(n, k[0]), k[-1])
    for half in LOCAL_WINDOWS:
        lo = int(np.searchsorted(k, center - half, "left"))
        hi = int(np.searchsorted(k, center + half, "right"))
        if hi - lo >= MIN_BEATS and k[hi - 1] > k[lo]:
            period, t0 = _fit_line(k[lo:hi], t[lo:hi])
            if period > 0 and bounds[0] <= 60 / period <= bounds[1]:
                return period, t0
    return fallback


def _place(k: np.ndarray, t: np.ndarray, index: np.ndarray, fallback: tuple[float, float],
           bounds: tuple[float, float]) -> np.ndarray:
    """Time of each index, from the line fitted over the detections around it."""

    def at(n: float) -> float:
        local_period, local_t0 = _local_line(k, t, n, fallback, bounds)
        return local_t0 + local_period * n

    return np.fromiter(map(at, index), float, len(index))


def _grid_times(k: np.ndarray, t: np.ndarray, period: float, t0: float, duration: float,
                bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices and times, each beat placed by a line fitted to its own neighbourhood.

    On a track that holds one tempo every local line coincides with the global one; on a track that
    drifts the grid follows it, and over a silent stretch the detections on both sides carry it across.
    """
    low, high = math.floor(-t0 / period) - LOCAL_MARGIN, math.ceil((duration - t0) / period) + LOCAL_MARGIN
    for _ in range(8):
        index = np.arange(low, high + 1)
        times = _place(k, t, index, (period, t0), bounds)
        if times[0] < 0 and times[-1] >= duration:
            break
        low -= LOCAL_MARGIN if times[0] >= 0 else 0
        high += LOCAL_MARGIN if times[-1] < duration else 0
    else:
        raise NoBeatGrid("no beat grid: the fitted grid does not reach both ends of the audio")
    inside = (times >= 0) & (times < duration)
    return index[inside], times[inside]


def _drop_offbeats(beats: np.ndarray, period: float) -> np.ndarray:
    """Keep the detections that land a whole number of beats after the last one kept.

    In some sections the detector hears half-beats and other off-beats. They would otherwise be numbered
    as beats of their own, tilt the fit and pull the local lines with them, so they go before anything is
    fitted. Anything more than two beats out is past a silence the grid has to cross anyway, where the
    starting period is too rough to judge: those re-anchor the walk instead.
    """
    kept = [0]
    for i in range(1, len(beats)):
        steps = (beats[i] - beats[kept[-1]]) / period
        whole = round(steps)
        if whole >= 1 and (whole > 2 or abs(steps - whole) <= 0.25):
            kept.append(i)
    return beats[kept]


def _number_and_fit(beats: np.ndarray, period: float) -> tuple[np.ndarray, float, float]:
    """Number the detections, fit t = t0 + k·period, then renumber with the fitted period and refit.

    The first period comes from a median of 20 ms-quantized intervals and can be a percent out, which is
    enough to lose a beat when the numbering has to step across a silent stretch; the renumber spends
    that better period on the gaps.
    """
    for _ in range(2):
        k = _beat_indices(beats, period)
        period, t0 = _fit_line(k, beats)
    return k, period, t0


def _octave_correct(period: float, min_bpm: float, max_bpm: float) -> float:
    """Halve or double the period until its tempo lies within the bounds."""
    for _ in range(8):
        if 60 / period > max_bpm:
            period *= 2
        elif 60 / period < min_bpm:
            period /= 2
        else:
            return period
    raise NoBeatGrid(f"no beat grid: tempo {60 / period:.1f} BPM outside {min_bpm}-{max_bpm} BPM")


def fit_beat_grid(
        det_beats: np.ndarray,
        det_downbeats: np.ndarray,
        duration: float,
        min_bpm: int = 100,
        max_bpm: int = 180,
) -> GridFit:
    """Fit a beat grid to the detections and extend it across the whole track.

    One global least-squares line octave-corrects the tempo and numbers the detections; each grid beat
    is then placed by a line fitted over the detections within 8 bars of it, so a slow drift (a SUNO
    master can accelerate by more than a BPM over three minutes) is followed instead of averaged away.
    A sudden tempo change is not modelled: the grid bends across it and the residual gate rejects the
    track. Detections are quantized to 20 ms frames, but each local fit over dozens of them recovers a
    period far finer than one frame.
    """
    beats = np.sort(np.asarray(det_beats, dtype=float))
    if len(beats) < MIN_BEATS or not np.isfinite(beats).all():
        raise NoBeatGrid(f"no beat grid: {len(beats)} usable beat detections; at least {MIN_BEATS} required")
    if not math.isfinite(duration) or duration <= 0:
        raise NoBeatGrid("no beat grid: audio has no positive duration")

    intervals = np.diff(beats)
    # Half-tempo and double-tempo spacings stay in: the octave correction below decides between them.
    plausible = intervals[(intervals >= 30 / max_bpm) & (intervals <= 120 / min_bpm)]
    if not len(plausible):
        raise NoBeatGrid(f"no beat grid: beat spacing implies a tempo far outside {min_bpm}-{max_bpm} BPM")
    period = _octave_correct(float(np.median(plausible)), min_bpm, max_bpm)
    # 1% of slack: a grid that follows the drift of a track sitting on a bound may cross it by a hair.
    bounds = (min_bpm * 0.99, max_bpm * 1.01)
    beats = _drop_offbeats(beats, period)
    if len(beats) < MIN_BEATS:
        raise NoBeatGrid(f"no beat grid: {len(beats)} detections left once off-beats were dropped; "
                         f"at least {MIN_BEATS} required")

    inliers = beats
    for _ in range(3):
        k, period, t0 = _number_and_fit(inliers, period)
        # Judged against the grid each detection would get, not against one straight line: a track that
        # drifts leaves a single line by most of a beat at its ends, and those are not stray detections.
        keep = np.abs(inliers - _place(k, inliers, k, (period, t0), bounds)) <= 0.2 * period
        if keep.all():
            break
        inliers, k = inliers[keep], k[keep]
        if len(inliers) < MIN_BEATS:
            raise NoBeatGrid(f"no beat grid: only {len(inliers)} detections fit one tempo curve; "
                             f"at least {MIN_BEATS} required")
    else:  # the last pass dropped outliers, so refit without them
        k, period, t0 = _number_and_fit(inliers, period)
    if not min_bpm <= 60 / period <= max_bpm:
        raise NoBeatGrid(f"no beat grid: fitted tempo {60 / period:.1f} BPM outside {min_bpm}-{max_bpm} BPM")

    index, grid = _grid_times(k, inliers, period, t0, duration, bounds)
    _require_rideable(grid, period, duration, None)  # only the final grid has to hold the tempo bounds

    # Second pass: number every detection by the grid beat it actually landed on, and drop the ones that
    # landed between beats. The detector hears half-beats in some sections, and the first numbering,
    # which only ever steps forward, took those for real beats and dragged the local lines with them.
    nearest = _nearest_index(grid, inliers)
    on_beat = np.abs(inliers - grid[nearest]) <= 0.25 * period
    if on_beat.sum() < MIN_BEATS:
        raise NoBeatGrid(f"no beat grid: only {int(on_beat.sum())} detections sit on the fitted grid; "
                         f"at least {MIN_BEATS} required")
    k, inliers = index[nearest[on_beat]].astype(float), inliers[on_beat]
    index, grid = _grid_times(k, inliers, period, t0, duration, bounds)
    _require_rideable(grid, period, duration, bounds)

    tolerance = max(0.07, 0.2 * period)
    offset = grid - beats[_nearest_index(beats, grid)]
    heard = np.abs(offset) <= tolerance
    if not heard.any():
        raise NoBeatGrid("no beat grid: the fitted grid matches none of the detections")
    residual_ms = float(np.sqrt(np.mean(offset[heard] ** 2)) * 1000)
    if residual_ms > MAX_GRID_RESIDUAL_MS:
        raise NoBeatGrid(f"no beat grid: detections scatter {residual_ms:.0f} ms around the fitted "
                         f"{60 / period:.1f} BPM grid; at most {MAX_GRID_RESIDUAL_MS} ms allowed")

    return GridFit(period, t0, grid, index, heard,
                   _vote_bar_phase(index, grid, det_downbeats, tolerance, heard), residual_ms)


def _require_rideable(grid: np.ndarray, period: float, duration: float, bounds: tuple[float, float] | None) -> None:
    """Refuse a grid that is too short, runs backwards, or (once final) holds an interval out of bounds."""
    if len(grid) < MIN_BEATS:
        raise NoBeatGrid(f"no beat grid: {len(grid)} beats fit in {duration:.1f} s at {60 / period:.1f} BPM")
    spacing = np.diff(grid)
    if spacing.min() <= 0:
        raise NoBeatGrid("no beat grid: the fitted grid does not advance in time")
    fastest, slowest = 60 / spacing.min(), 60 / spacing.max()
    if bounds and not bounds[0] <= slowest <= fastest <= bounds[1]:
        raise NoBeatGrid(f"no beat grid: the grid runs {slowest:.1f}-{fastest:.1f} BPM, "
                         f"outside {bounds[0]:.0f}-{bounds[1]:.0f} BPM")


def _vote_bar_phase(index: np.ndarray, grid: np.ndarray, det_downbeats, tolerance: float,
                    heard: np.ndarray) -> int:
    """The grid index modulo 4 that most detected downbeats land on, each weighted by how close it lands."""
    downbeats = np.asarray(det_downbeats, dtype=float)
    votes = np.zeros(BEATS_PER_BAR)
    if len(downbeats):
        nearest = _nearest_index(grid, downbeats)
        distance = np.abs(downbeats - grid[nearest])
        near = distance <= tolerance
        for phase, weight in zip(index[nearest[near]] % BEATS_PER_BAR, 1 - distance[near] / (2 * tolerance)):
            votes[phase] += weight
    if not votes.any():  # no downbeats survived: bar 1 starts on the first beat actually heard
        return int(index[np.argmax(heard)] % BEATS_PER_BAR)
    return int(np.argmax(votes))


def tempo_range(fit: GridFit, window: float = 30) -> tuple[float, float]:
    """Lowest and highest local tempo of the grid, taken as the median interval of each 30 s window."""
    tempos = []
    for start in np.arange(0, fit.beat_times[-1], window):
        beats = fit.beat_times[(fit.beat_times >= start) & (fit.beat_times < start + window)]
        if len(beats) >= MIN_BEATS:  # a partial window at the end says nothing about tempo
            tempos.append(60 / float(np.median(np.diff(beats))))
    return (round(min(tempos), 1), round(max(tempos), 1)) if tempos else (round(fit.bpm, 1), round(fit.bpm, 1))


def first_bar_time(fit: GridFit) -> float:
    """Bar 1: the first grid beat on the voted downbeat phase.

    The grid covers the whole track, so an intro is plannable bars rather than a run-up to the first
    beat the detector heard; the phase the drops vote for extends back through it. Only the 0-3 grid
    beats before bar 1 are pickups.
    """
    on_phase = np.flatnonzero(fit.index % BEATS_PER_BAR == fit.bar_phase)
    return float(fit.beat_times[on_phase[0] if len(on_phase) else 0])


def band_pass(audio: np.ndarray, sr: int, low: int = 20, high: int = 150) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass; the default 20-150 Hz band holds kick and bass."""
    nyquist = sr / 2
    sos = signal.butter(4, [low / nyquist, min(high / nyquist, 0.99)], btype='band', output='sos')
    return signal.sosfiltfilt(sos, audio, padlen=min(27, len(audio) - 1))


def compute_band_energy(filtered: np.ndarray, sr: int) -> np.ndarray:
    """RMS energy per second of a band-passed signal."""
    print("Computing band energy...")

    filtered = np.nan_to_num(filtered, nan=0.0)

    window_samples = sr
    energy = []

    for i in range(0, len(filtered), window_samples):
        chunk = filtered[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        if np.isnan(rms) or np.isinf(rms):
            rms = 0.0
        energy.append(rms)

    return np.array(energy)


def compute_kick_density(
        bass_energy: np.ndarray,
        beats: list[dict],
        duration: float | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Time-weighted bass energy over complete bar intervals, including beat four."""
    downbeats = [b for b in beats if b["bar"] > 0 and b["beat"] == 1]
    if not downbeats or not len(bass_energy):
        return np.array([], dtype=float), []
    end_time = duration if duration is not None else float(len(bass_energy))
    maximum = float(np.max(bass_energy))
    values, info = [], []
    for i, beat in enumerate(downbeats):
        start = beat["t"]
        end = downbeats[i + 1]["t"] if i + 1 < len(downbeats) else end_time
        if end <= start:
            continue
        weighted = 0.0
        for second in range(max(0, int(start)), min(len(bass_energy), int(np.ceil(end)))):
            overlap = max(0, min(end, second + 1) - max(start, second))
            weighted += float(bass_energy[second]) * overlap
        density = weighted / (end - start) / maximum if maximum > 0 else 0.0
        values.append(density)
        info.append({"bar": beat["bar"], "start_t": start, "end_t": end, "kick_density": round(density, 3)})
    return np.array(values), info


def detect_kick_zones(
        kick_density: np.ndarray,
        bar_info: list[dict],
        threshold: float = 0.45,
        min_zone_bars: int = 4,
        duration: float | None = None
) -> list[dict]:
    """Detect contiguous zones of kick presence vs absence."""
    print("Detecting kick zones...")

    if len(kick_density) == 0:
        return []

    zones = []
    current_zone_start = 0
    current_zone_type = "work" if kick_density[0] >= threshold else "breakdown"

    for i, density in enumerate(kick_density):
        zone_type = "work" if density >= threshold else "breakdown"

        if zone_type != current_zone_type:
            zone_length = i - current_zone_start
            if zone_length >= min_zone_bars:
                start_sec = bar_info[current_zone_start]["start_t"]
                end_sec = bar_info[i]["start_t"]
                zones.append({
                    "type": current_zone_type,
                    "startBar": bar_info[current_zone_start]["bar"],
                    "durationBars": zone_length,
                    "startSeconds": round(start_sec, 2),
                    "durationSeconds": round(end_sec - start_sec, 2),
                })

            current_zone_start = i
            current_zone_type = zone_type

    zone_length = len(kick_density) - current_zone_start
    if zone_length >= min_zone_bars:
        # For final zone, use track duration if available, otherwise estimate next measure boundary
        if duration is not None:
            final_end_t = duration
        else:
            final_end_t = bar_info[-1]["end_t"]
        start_sec = bar_info[current_zone_start]["start_t"]
        zones.append({
            "type": current_zone_type,
            "startBar": bar_info[current_zone_start]["bar"],
            "durationBars": zone_length,
            "startSeconds": round(start_sec, 2),
            "durationSeconds": round(final_end_t - start_sec, 2),
        })

    # Only return work zones (not breakdowns), strip type field
    work_zones = [{k: v for k, v in z.items() if k != "type"} for z in zones if z["type"] == "work"]
    print(f"Detected {len(work_zones)} work zones (filtered from {len(zones)} total)")
    return work_zones


def compute_beat_confidence(beat_times: np.ndarray, expected_bpm: float) -> np.ndarray:
    """Compute confidence score for each beat based on interval consistency."""
    if len(beat_times) < 2:
        return np.ones(len(beat_times))

    expected_interval = 60.0 / expected_bpm
    intervals = np.diff(beat_times)
    tolerance = expected_interval * 0.15

    confidences = []
    for i, t in enumerate(beat_times):
        if i == 0:
            deviation = abs(intervals[0] - expected_interval)
        elif i == len(beat_times) - 1:
            deviation = abs(intervals[-1] - expected_interval)
        else:
            deviation = (abs(intervals[i - 1] - expected_interval) + abs(intervals[i] - expected_interval)) / 2

        confidence = np.exp(-(deviation / tolerance) ** 2)
        confidences.append(float(confidence))

    return np.array(confidences)


def build_beats_array(
        beat_times: np.ndarray,
        first_downbeat_t: float,
        confidences: np.ndarray | None = None
) -> list[dict]:
    """Number the grid: bar 1 starts at the first downbeat, earlier beats are pickups in bar 0."""
    if not len(beat_times):
        return []
    first = int(np.argmin(np.abs(np.asarray(beat_times, dtype=float) - first_downbeat_t)))
    beats = []
    for i, t in enumerate(beat_times):
        offset = i - first
        confidence = float(confidences[i]) if confidences is not None else 1.0
        beats.append({
            "t": round(float(t), 3),
            "bar": max(0, offset // BEATS_PER_BAR + 1),
            "beat": int(offset % BEATS_PER_BAR + 1),
            "confidence": round(confidence, 2),
        })
    return beats


def analyze_audio(
        audio_path: Path,
        min_bpm: int = 100,
        max_bpm: int = 180,
        first_downbeat: float | None = None,
        audio: tuple[np.ndarray, int] | None = None,
        grid_out: dict | None = None,
) -> dict:
    """Main analysis pipeline. `audio` is the file's already decoded (samples, rate), if any.

    `grid_out`, when given, collects grid-fit diagnostics for the manifest; the returned analysis
    itself carries only the fields the shared schema allows.
    """

    if not 0 < min_bpm < max_bpm:
        raise ValueError("BPM bounds must satisfy 0 < min_bpm < max_bpm")
    audio, sr = audio if audio is not None else load_audio(audio_path)
    duration = len(audio) / sr
    require_sound(audio, sr)

    beats_start = time.monotonic()
    det_beats, det_downbeats = extract_beats(audio, sr, min_bpm, max_bpm)
    print(f"timing: beats {time.monotonic() - beats_start:.1f}s", file=sys.stderr)

    if np.any(det_beats < 0) or np.any(det_beats >= duration):
        raise ValueError("Detected beats lie outside audio duration")
    if len(det_beats) < MIN_BEATS:
        raise NoBeatGrid(
            f"no beat grid: {len(det_beats)} beats detected between {min_bpm} and {max_bpm} BPM; at least {MIN_BEATS} required")

    fit = fit_beat_grid(det_beats, det_downbeats, duration, min_bpm, max_bpm)
    inferred = int((~fit.heard).sum())
    print(f"Grid: {fit.bpm:.1f} BPM, {len(fit.beat_times)} beats ({inferred} inferred), "
          f"residual {fit.residual_ms:.1f} ms")

    zones_start = time.monotonic()
    bass = band_pass(audio, sr)
    low_band = compute_band_energy(bass, sr)

    if first_downbeat is not None:
        if not np.isfinite(first_downbeat) or not 0 <= first_downbeat < duration:
            raise ValueError("First downbeat must identify a beat within the audio")
        first_downbeat_t = float(fit.beat_times[np.argmin(np.abs(fit.beat_times - first_downbeat))])
        print(f"Using user-specified first downbeat: {first_downbeat_t:.3f}s")
    else:
        first_downbeat_t = first_bar_time(fit)

    # A grid beat with no detection under it is a guess the app should be able to tell apart.
    confidences = compute_beat_confidence(fit.beat_times, fit.bpm)
    confidences = np.where(fit.heard, confidences, np.minimum(confidences, 0.5))

    beats_data = build_beats_array(fit.beat_times, first_downbeat_t, confidences)

    kick_density, kick_bar_info = compute_kick_density(low_band, beats_data, duration)

    work_zones = detect_kick_zones(kick_density, kick_bar_info, duration=duration)
    print(f"timing: bass+zones {time.monotonic() - zones_start:.1f}s", file=sys.stderr)

    if grid_out is not None:
        grid_out.update(
            detectedBeatCount=len(det_beats), detectedDownbeatCount=len(det_downbeats),
            inferredBeatCount=inferred, residualMs=round(fit.residual_ms, 2), barPhase=fit.bar_phase,
            tempoRangeBpm=list(tempo_range(fit)),
        )

    return {
        "version": "3.0",
        "duration": duration,
        "sampleRate": sr,
        "bpm": round(fit.bpm, 1),
        "key": None,
        "firstDownbeat": round(first_downbeat_t, 3),
        "beats": beats_data,
        "workZones": work_zones,
        "raw": {
            "beatTimes": [b["t"] for b in beats_data],
            "bassEnergy": [round(float(v), 4) for v in low_band],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audio analysis for Pain Cave spin class workouts"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to audio file (m4a, mp3, wav)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to write JSON output (default: stdout)"
    )
    parser.add_argument(
        "--min-bpm",
        type=int,
        default=100,
        help="Minimum BPM for the fitted beat grid (default: 100)"
    )
    parser.add_argument(
        "--max-bpm",
        type=int,
        default=180,
        help="Maximum BPM for the fitted beat grid (default: 180)"
    )
    parser.add_argument(
        "--first-downbeat",
        type=float,
        help="Time in seconds for bar 1, beat 1 (overrides auto-detection)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    analysis = analyze_audio(
        args.input,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        first_downbeat=args.first_downbeat,
    )

    output_json = json.dumps(analysis, indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"\nAnalysis written to: {args.output}")
    else:
        print("\n" + output_json)


if __name__ == "__main__":
    main()
