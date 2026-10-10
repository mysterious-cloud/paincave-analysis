#!/usr/bin/env python3
"""
The `melody` stage of reference analysis: the sung notes in the vocal stem, on the grid's beats.

CREPE (torchcrepe, MIT, `full` model) tracks the vocal stem's pitch every 10 ms, only where the stem is loud enough
to be a voice: Demucs leaves synth bleed in the vocal stem, quietly. Notes come from that pitch track by rule
(`notes`, pure numpy): a note starts where the pitch moves and stays, or the voice re-attacks. Pitch and
periodicity are cached beside the stems, so tuning the rules never runs CREPE again.

Only `reference.py` imports this, and torchcrepe is imported inside `track`; it lives in the `reference`
dependency group with Demucs, so the Pain Cave import path never loads it.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

MODEL = "full"
CREPE_RATE = 16000
HOP_SECONDS = 0.01
FMIN, FMAX = 65.0, 1100.0  # C2 to C#6: every singing voice
WINDOW = 1024  # samples at 44.1 kHz for the level of a frame

GATE_DB = 35.0  # a frame is sung only within this many dB of the stem's loudest frame
PERIODICITY = 0.45  # CREPE's confidence that the frame is pitched
SMOOTH = 5  # frames of median filter on the pitch
SPLIT_SEMITONES = 0.7  # a pitch this far from the note's own...
SPLIT_HOLD = 5  # ...for this many frames starts a new note (shorter is vibrato or a scoop)
REATTACK_DB = 6.0  # a level rise this sharp...
REATTACK_FRAMES = 3  # ...within this many frames starts a new note on the same pitch
MIN_NOTE = 7  # frames: shorter notes are dropped
MERGE_GAP = 5  # frames: the same pitch either side of a gap this short is one note
PAD = 5  # frames tracked either side of a sung stretch
FAINT_DB = -20.0  # notes quieter than this below the stem's loudest frame are likely bleed or backing: readers weigh them less

Progress = Callable[[float], None]


def frame_levels(mono: np.ndarray, rate: int) -> np.ndarray:
    """Each 10 ms frame's RMS level in dBFS, centred on the frame time."""
    hop = round(rate * HOP_SECONDS)
    frames = len(mono) // hop + 1
    padded = np.pad(mono.astype(np.float64), (WINDOW // 2, WINDOW // 2))
    power = np.convolve(padded ** 2, np.ones(WINDOW) / WINDOW, mode="valid")
    return 10 * np.log10(np.maximum(power[::hop][:frames], 1e-12))


def gated(levels: np.ndarray) -> np.ndarray:
    return levels >= levels.max() - GATE_DB


def stretches(mask: np.ndarray, pad: int = 0) -> list[tuple[int, int]]:
    """The [start, end) frame runs where `mask` holds, each widened by `pad` frames and merged where they touch."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    runs: list[tuple[int, int]] = []
    for start, end in zip(edges[::2], edges[1::2]):
        start, end = max(0, start - pad), min(len(mask), end + pad)
        if runs and start <= runs[-1][1]:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((int(start), int(end)))
    return runs


def track(vocals: np.ndarray, rate: int, cache: Path, progress: Progress = lambda f: None) -> tuple[np.ndarray, np.ndarray]:
    """Pitch in Hz and periodicity for every 10 ms frame of the vocal stem (mono) (0 where it isn't sung or tracked),
    from `cache` when it's there."""
    if cache.is_file():
        found = np.load(cache)
        return found["pitch"], found["periodicity"]
    import torch
    import torchcrepe

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    levels = frame_levels(vocals, rate)
    pitch, periodicity = np.zeros(len(levels), np.float32), np.zeros(len(levels), np.float32)
    runs = stretches(gated(levels), PAD)
    total, done = sum(end - start for start, end in runs) or 1, 0
    hop = round(CREPE_RATE * HOP_SECONDS)
    for start, end in runs:
        piece = vocals[round(start * HOP_SECONDS * rate):round(end * HOP_SECONDS * rate)]
        audio = torch.from_numpy(resample_poly(piece, CREPE_RATE, rate).astype(np.float32))[None]
        with torch.no_grad():
            hz, conf = torchcrepe.predict(audio, CREPE_RATE, hop, FMIN, FMAX, MODEL, return_periodicity=True,
                                          batch_size=2048, device=device, decoder=torchcrepe.decode.viterbi)
        n = min(end - start, hz.shape[-1])
        pitch[start:start + n] = hz[0, :n].cpu().numpy()
        periodicity[start:start + n] = conf[0, :n].cpu().numpy()
        done += end - start
        progress(done / total)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.stem}.tmp.npz")
    np.savez_compressed(temporary, pitch=pitch, periodicity=periodicity)
    temporary.replace(cache)
    return pitch, periodicity


Note = tuple[int, int, int, float, float]  # start frame, end frame (exclusive), MIDI note, mean periodicity, level


def notes(pitch: np.ndarray, periodicity: np.ndarray, levels: np.ndarray) -> list[Note]:
    """The sung notes as (start frame, end frame exclusive, MIDI note, mean periodicity, mean level in dB below the
    stem's loudest frame).

    A frame is sung when it's pitched and loud enough; within a sung stretch a new note starts where the
    smoothed pitch sits more than SPLIT_SEMITONES from the note so far for SPLIT_HOLD frames (from where it
    left), or where the level jumps REATTACK_DB within REATTACK_FRAMES (a re-sung note). Short notes are
    dropped; the same pitch either side of a short gap is joined.
    """
    n = min(len(pitch), len(periodicity), len(levels))
    pitch, periodicity, levels = pitch[:n], periodicity[:n], levels[:n]
    sung = (periodicity >= PERIODICITY) & (pitch > 0) & gated(levels)
    midi = np.zeros(n)
    midi[sung] = 69 + 12 * np.log2(pitch[sung] / 440)
    for start, end in stretches(sung):
        midi[start:end] = median_filter(midi[start:end], size=SMOOTH, mode="nearest")

    pieces: list[tuple[int, int]] = []
    for start, end in stretches(sung):
        # `heard` is where the note's own pitch begins: a scoop too short to be a note joins the note it leads into
        note, heard, away = start, start, 0
        for i in range(start + 1, end):
            reference = np.median(midi[heard:i - away]) if i - away > heard else midi[heard]
            away = away + 1 if abs(midi[i] - reference) > SPLIT_SEMITONES else 0
            reattack = i - note >= MIN_NOTE and levels[i] - levels[max(note, i - REATTACK_FRAMES):i].min() > REATTACK_DB
            if away >= SPLIT_HOLD:
                moved = i - away + 1
                if moved - note >= MIN_NOTE:
                    pieces.append((note, moved))
                    note = moved
                heard, away = moved, 0
            elif reattack and away == 0:
                pieces.append((note, i))
                note = heard = i
        pieces.append((note, end))

    peak = levels.max()
    found: list[Note] = []
    for start, end in pieces:
        if end - start < MIN_NOTE:
            continue
        if found and found[-1][2] == round(float(np.median(midi[start:end]))) and 0 < start - found[-1][1] <= MERGE_GAP:
            start = found.pop()[0]
        frames = slice(start, end)
        found.append((start, end, int(round(float(np.median(midi[frames])))), float(periodicity[frames].mean()),
                      float(levels[frames].mean() - peak)))
    return found


def beat_position(seconds: np.ndarray, beats: list[dict], first_downbeat: int) -> np.ndarray:
    """Seconds to beats from bar 1's downbeat (beat 0), between the grid's own beats; before the first beat and
    after the last, at the nearest beat interval."""
    times = np.array([b["time"] for b in beats], dtype=np.float64)
    index = np.arange(len(times), dtype=np.float64) - first_downbeat
    position = np.interp(seconds, times, index)
    before, after = seconds < times[0], seconds > times[-1]
    position[before] = index[0] - (times[0] - seconds[before]) / (times[1] - times[0])
    position[after] = index[-1] + (seconds[after] - times[-1]) / (times[-1] - times[-2])
    return position


def block(found: list[Note], beats: list[dict], first_downbeat: int) -> dict:
    """The analysis' `melody` block: each note's start and length in beats, MIDI pitch, confidence and level."""
    starts = beat_position(np.array([n[0] * HOP_SECONDS for n in found]), beats, first_downbeat)
    ends = beat_position(np.array([n[1] * HOP_SECONDS for n in found]), beats, first_downbeat)
    return {
        "tool": "crepe",
        "version": 1,
        "model": MODEL,
        "gateDb": GATE_DB,
        "faintDb": FAINT_DB,
        "notes": [{"start": round(float(s), 3), "length": round(float(e - s), 3), "pitch": n[2],
                   "confidence": round(n[3], 2), "level": round(n[4], 1)}
                  for n, s, e in zip(found, starts, ends)],
    }


def melody(vocals: np.ndarray, rate: int, beats: list[dict], first_downbeat: int, cache: Path,
           progress: Progress = lambda f: None) -> dict:
    """The `melody` block for the vocal stem (mono, at `rate`)"""
    pitch, periodicity = track(vocals, rate, cache, progress)
    return block(notes(pitch, periodicity, frame_levels(vocals, rate)), beats, first_downbeat)
