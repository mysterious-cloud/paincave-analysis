#!/usr/bin/env python3
"""
Audio analysis for Pain Cave spin class workouts.

Extracts:
- Beat grid with downbeats and bar positions
- Time-varying tempo curve
- Work zones (regions with consistent kicks vs breakdowns)
- Musical key

Output uses camelCase JSON keys for direct consumption by TypeScript.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file and convert to mono. Uses ffmpeg for m4a/mp3."""
    print(f"Loading audio: {audio_path}")

    suffix = audio_path.suffix.lower()

    # For m4a/mp3, use ffmpeg to decode
    if suffix in ['.m4a', '.mp3', '.aac']:
        with tempfile.TemporaryDirectory(prefix="paincave-analysis-") as tmp:
            tmp_path = str(Path(tmp) / "decoded.wav")
            result = subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio_path), '-ar', '44100', '-ac', '1', tmp_path],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            audio, sr = sf.read(tmp_path)
    else:
        audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain nonempty finite samples")

    print(f"Duration: {len(audio) / sr:.1f}s ({len(audio) / sr / 60:.1f} minutes)")
    print(f"Sample rate: {sr} Hz")

    return audio, sr


def extract_beats(audio_path: Path, min_bpm: int = 100, max_bpm: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Extract beats and downbeats using madmom."""
    print("Extracting beats with madmom...")

    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    # RNN processor for beat/downbeat activations
    rnn = RNNDownBeatProcessor()
    activations = rnn(str(audio_path))

    # DBN processor for beat tracking
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[4],  # 4/4 time for EDM
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=100
    )

    # Returns array of [time, beat_position] where beat_position is 1-4
    beat_data = dbn(activations)

    if len(beat_data) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    beat_times = beat_data[:, 0]
    beat_positions = beat_data[:, 1].astype(int)

    print(f"Detected {len(beat_times)} beats")

    return beat_times, beat_positions


def detect_key(audio_path: Path) -> str | None:
    """Detect musical key using madmom CNN."""
    print("Detecting key...")

    try:
        from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label

        proc = CNNKeyRecognitionProcessor()
        probs = proc(str(audio_path))
        key = key_prediction_to_label(probs)

        print(f"Key: {key}")
        return key
    except Exception as e:
        print(f"Warning: Key detection failed: {e}")
        return None


def correct_octave_errors(bpm_curve: np.ndarray, reference_bpm: float, tolerance: float = 0.25) -> np.ndarray:
    """Correct octave errors in BPM detection."""
    corrected = bpm_curve.copy()

    for i, bpm in enumerate(bpm_curve):
        ratio = bpm / reference_bpm

        if 1 - tolerance <= ratio <= 1 + tolerance:
            continue

        if 1.8 <= ratio <= 2.3:
            corrected[i] = bpm / 2
        elif 0.4 <= ratio <= 0.55:
            corrected[i] = bpm * 2
        elif 1.2 <= ratio <= 1.45:
            corrected[i] = bpm * 3 / 4
        elif 1.45 < ratio < 1.8:
            corrected[i] = bpm * 2 / 3

    return corrected


def compute_tempo_curve(beat_times: np.ndarray, window: int = 8, reference_bpm: float | None = None) -> tuple[np.ndarray, list[dict]]:
    """Compute local BPM from beat intervals."""
    print("Computing tempo curve...")

    intervals = np.diff(beat_times)
    if not np.isfinite(beat_times).all() or np.any(intervals <= 0):
        raise ValueError("Beat times must be finite and strictly increasing")
    if len(intervals) == 0:
        return np.array([], dtype=float), []
    local_bpm = 60 / intervals

    smoothed_bpm = median_filter(local_bpm, size=window)

    if reference_bpm:
        initial_global = reference_bpm
        print(f"Using reference BPM: {reference_bpm:.1f}")
    else:
        initial_global = float(np.percentile(smoothed_bpm, 25))

    corrected_bpm = correct_octave_errors(smoothed_bpm, initial_global)
    corrected_bpm = median_filter(corrected_bpm, size=window)

    segments = detect_tempo_segments(corrected_bpm, threshold=3.0)

    global_bpm = float(np.median(corrected_bpm))
    print(f"Global BPM: {global_bpm:.1f}")

    return corrected_bpm, segments


def detect_tempo_segments(bpm_curve: np.ndarray, threshold: float = 3.0) -> list[dict]:
    """Detect segments where tempo is stable."""
    segments = []
    if len(bpm_curve) == 0:
        return segments
    current_start = 0
    current_bpm = bpm_curve[0]

    for i, bpm in enumerate(bpm_curve):
        if abs(bpm - current_bpm) > threshold:
            if i > current_start:
                segments.append({
                    "start_beat": int(current_start),
                    "end_beat": int(i),
                    "bpm": round(float(np.median(bpm_curve[current_start:i])), 1)
                })
            current_start = i
            current_bpm = bpm

    if len(bpm_curve) > current_start:
        segments.append({
            "start_beat": int(current_start),
            "end_beat": int(len(bpm_curve)),
            "bpm": round(float(np.median(bpm_curve[current_start:])), 1)
        })

    return segments


def compute_band_energy(audio: np.ndarray, sr: int, low: int = 20, high: int = 150) -> np.ndarray:
    """Compute energy in a frequency band (e.g., bass 20-150 Hz)."""
    print(f"Computing {low}-{high} Hz band energy...")

    nyquist = sr / 2
    low_norm = low / nyquist
    high_norm = min(high / nyquist, 0.99)

    sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')

    filtered = signal.sosfiltfilt(sos, audio, padlen=min(27, len(audio) - 1))

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


def detect_first_kick(
    audio: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    beat_positions: np.ndarray
) -> float:
    """
    Detect the correct first downbeat by analyzing where kick patterns start.

    In EDM, kick patterns typically start on beat 1 (downbeat). If madmom's
    downbeat detection is off, we correct it by finding where kicks actually
    start in the loudest section and aligning to that.
    """
    print("Detecting first kick for bar alignment...")

    nyquist = sr / 2
    sos = signal.butter(4, [20 / nyquist, 150 / nyquist], btype='band', output='sos')
    bass = signal.sosfiltfilt(sos, audio)

    envelope = np.abs(bass)
    smooth_samples = int(0.01 * sr)
    envelope = np.convolve(envelope, np.ones(smooth_samples) / smooth_samples, mode='same')

    # Find the loudest section (likely the main drop/chorus with kicks)
    duration = len(audio) / sr
    window_sec = 10  # Analyze 10-second windows
    window_samples = window_sec * sr
    max_energy = 0
    loudest_start = 0

    for start in range(0, len(envelope) - window_samples, window_samples // 2):
        energy = np.mean(envelope[start:start + window_samples])
        if energy > max_energy:
            max_energy = energy
            loudest_start = start

    loudest_start_time = loudest_start / sr
    loudest_end_time = min(loudest_start_time + window_sec, duration)
    print(f"  Analyzing loudest section: {loudest_start_time:.1f}s - {loudest_end_time:.1f}s")

    # Find beats in the loudest section and check which position has kick onsets
    kick_threshold = np.max(envelope) * 0.4
    beat_energies = {1: [], 2: [], 3: [], 4: []}

    for t, pos in zip(beat_times, beat_positions):
        if loudest_start_time < t < loudest_end_time:
            sample_idx = int(t * sr)
            # Check for kick onset (sharp rise in bass energy)
            window = int(0.05 * sr)
            if sample_idx + window < len(envelope):
                peak = np.max(envelope[sample_idx:sample_idx + window])
                beat_energies[int(pos)].append(peak)

    # In 4-on-the-floor EDM, all beats should have kicks
    # But the FIRST kick of a phrase should be on beat 1
    # Check where kick patterns START (transitions from quiet to loud)

    # Find kick onset points (first beat where energy goes above threshold)
    onset_positions = []
    prev_loud = False
    for t, pos in zip(beat_times, beat_positions):
        sample_idx = int(t * sr)
        if sample_idx + int(0.05 * sr) < len(envelope):
            peak = np.max(envelope[sample_idx:sample_idx + int(0.05 * sr)])
            is_loud = peak > kick_threshold

            if is_loud and not prev_loud:
                onset_positions.append(int(pos))
            prev_loud = is_loud

    if onset_positions:
        # Count which beat position kick patterns start on most often
        from collections import Counter
        position_counts = Counter(onset_positions)
        most_common_onset = position_counts.most_common(1)[0][0]
        print(f"  Kick patterns start on beat position: {most_common_onset} (count: {position_counts})")

        # If kicks consistently start on beat 3 (should be beat 1), shift by 2
        if most_common_onset == 3:
            # Find first downbeat and shift back by 2 beats
            downbeat_indices = np.where(beat_positions == 1)[0]
            if len(downbeat_indices) > 0 and downbeat_indices[0] >= 2:
                # The actual first downbeat should be 2 beats earlier
                corrected_idx = downbeat_indices[0] - 2
                corrected_time = beat_times[corrected_idx]
                print(f"  Correcting downbeat: madmom says {beat_times[downbeat_indices[0]]:.3f}s, "
                      f"adjusted to {corrected_time:.3f}s")
                return float(corrected_time)

    # Default: use madmom's first downbeat
    downbeat_mask = beat_positions == 1
    if np.any(downbeat_mask):
        first_downbeat = beat_times[downbeat_mask][0]
        print(f"  Using madmom's first downbeat: {first_downbeat:.3f}s")
        return float(first_downbeat)

    return beat_times[0] if len(beat_times) > 0 else 0.0


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
            deviation = (abs(intervals[i-1] - expected_interval) + abs(intervals[i] - expected_interval)) / 2

        confidence = np.exp(-(deviation / tolerance) ** 2)
        confidences.append(float(confidence))

    return np.array(confidences)


def build_beats_array(
    beat_times: np.ndarray,
    beat_positions: np.ndarray,
    first_downbeat_t: float | None = None,
    confidences: np.ndarray | None = None
) -> list[dict]:
    """Build the beats array with bar and beat information."""
    beats = []

    if first_downbeat_t is not None and len(beat_times) > 0:
        distances = np.abs(beat_times - first_downbeat_t)
        first_downbeat_idx = np.argmin(distances)

        bar_num = 1
        beat_in_bar = 1

        for i, time in enumerate(beat_times):
            conf = confidences[i] if confidences is not None else 1.0

            if i < first_downbeat_idx:
                pickup_beat = (i - first_downbeat_idx) % 4 + 1
                beats.append({
                    "t": round(float(time), 3),
                    "bar": 0,
                    "beat": int(pickup_beat),
                    "confidence": round(float(conf), 2)
                })
            else:
                position_from_first = i - first_downbeat_idx
                bar_num = (position_from_first // 4) + 1
                beat_in_bar = (position_from_first % 4) + 1

                beats.append({
                    "t": round(float(time), 3),
                    "bar": int(bar_num),
                    "beat": int(beat_in_bar),
                    "confidence": round(float(conf), 2)
                })
    else:
        bar_num = 1
        for i, (time, pos) in enumerate(zip(beat_times, beat_positions)):
            if pos == 1 and len(beats) > 0:
                bar_num += 1

            conf = confidences[i] if confidences is not None else 1.0
            beats.append({
                "t": round(float(time), 3),
                "bar": bar_num,
                "beat": int(pos),
                "confidence": round(conf, 2)
            })

    return beats


def analyze_audio(
    audio_path: Path,
    min_bpm: int = 100,
    max_bpm: int = 180,
    reference_bpm: float | None = None,
    first_downbeat: float | None = None,
) -> dict:
    """Main analysis pipeline."""

    if not 0 < min_bpm < max_bpm:
        raise ValueError("BPM bounds must satisfy 0 < min_bpm < max_bpm")
    if reference_bpm is not None and (not np.isfinite(reference_bpm) or reference_bpm <= 0):
        raise ValueError("Reference BPM must be finite and positive")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr

    if duration < 1 or not np.any(audio):
        beat_times, beat_positions = np.array([], dtype=float), np.array([], dtype=int)
    else:
        beat_times, beat_positions = extract_beats(audio_path, min_bpm, max_bpm)

    if np.any(beat_times < 0) or np.any(beat_times >= duration):
        raise ValueError("Detected beats lie outside audio duration")

    tempo_curve, tempo_segments = compute_tempo_curve(beat_times, reference_bpm=reference_bpm)
    global_bpm = float(np.median(tempo_curve)) if len(tempo_curve) else 0.0

    low_band = compute_band_energy(audio, sr, low=20, high=150)

    if first_downbeat is not None:
        if not np.isfinite(first_downbeat) or not 0 <= first_downbeat < duration or not len(beat_times):
            raise ValueError("First downbeat must identify a beat within the audio")
        first_downbeat_t = float(beat_times[np.argmin(np.abs(beat_times - first_downbeat))])
        print(f"Using user-specified first downbeat: {first_downbeat_t:.3f}s")
    else:
        first_downbeat_t = detect_first_kick(audio, sr, beat_times, beat_positions) if len(beat_times) else 0.0

    beat_confidences = compute_beat_confidence(beat_times, global_bpm)

    beats_data = build_beats_array(beat_times, beat_positions, first_downbeat_t, beat_confidences)

    kick_density, kick_bar_info = compute_kick_density(low_band, beats_data, duration)

    work_zones = detect_kick_zones(kick_density, kick_bar_info, duration=duration)

    # Detect musical key
    key = detect_key(audio_path)

    return {
        "version": "3.0",
        "duration": duration,
        "sampleRate": sr,
        "bpm": round(global_bpm, 1),
        "key": key,
        "firstDownbeat": round(first_downbeat_t, 3),
        "beats": beats_data,
        "workZones": work_zones,
        "raw": {
            "beatTimes": [round(float(t), 3) for t in beat_times],
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
        help="Minimum BPM for beat tracking (default: 100)"
    )
    parser.add_argument(
        "--max-bpm",
        type=int,
        default=180,
        help="Maximum BPM for beat tracking (default: 180)"
    )
    parser.add_argument(
        "--reference-bpm",
        type=float,
        help="Reference BPM from DAW (e.g., Ableton) for octave error correction"
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
        reference_bpm=args.reference_bpm,
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
