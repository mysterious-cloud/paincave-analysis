#!/usr/bin/env python3
"""
Extract waveform peaks from audio files for visualization.

Generates multiple resolution peak files for rendering waveforms at different
zoom levels in a Canvas-based React application.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from analyze import load_audio


def lowpass_filter(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Apply a low-pass Butterworth filter."""
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(4, normalized_cutoff, btype='low', output='sos')
    return sosfilt(sos, samples)


def compute_peaks(samples: np.ndarray, samples_per_peak: int) -> list[list[float]]:
    """
    Compute min/max peaks for non-overlapping windows.

    Args:
        samples: Audio samples as a 1D numpy array
        samples_per_peak: Number of samples to combine into each peak

    Returns:
        List of [min, max] pairs, each normalized to -1.0 to 1.0
    """
    num_samples = len(samples)
    num_peaks = num_samples // samples_per_peak

    # Truncate to fit evenly into windows
    truncated = samples[:num_peaks * samples_per_peak]

    # Reshape into windows
    windows = truncated.reshape(num_peaks, samples_per_peak)

    # Compute min and max for each window
    mins = np.min(windows, axis=1)
    maxs = np.max(windows, axis=1)

    # Round to 3 decimal places and convert to list
    peaks = [
        [round(float(min_val), 3), round(float(max_val), 3)]
        for min_val, max_val in zip(mins, maxs)
    ]

    return peaks


def generate_peaks_json(
    samples: np.ndarray,
    sample_rate: int,
    samples_per_peak: int,
) -> dict:
    """
    Generate a peaks data structure for a given resolution.

    Args:
        samples: Audio samples as a 1D numpy array
        sample_rate: Sample rate of the audio
        samples_per_peak: Number of samples per peak

    Returns:
        Dictionary with sampleRate, duration, samplesPerPeak, and peaks
    """
    duration = len(samples) / sample_rate
    peaks = compute_peaks(samples, samples_per_peak)

    return {
        "sampleRate": sample_rate,
        "duration": round(duration, 3),
        "samplesPerPeak": samples_per_peak,
        "peaks": peaks,
    }


TINY_PEAKS_BARS = 120


def generate_tiny_peaks(samples: np.ndarray, num_bars: int = TINY_PEAKS_BARS) -> dict:
    """
    Generate coarse waveform for track list thumbnails.

    Produces normalized amplitude values (0.0–1.0) where 1.0 is the
    loudest window. Unlike the other resolutions which store [min, max] pairs,
    tiny peaks store a single normalized amplitude per bar.

    Args:
        samples: Audio samples as a 1D numpy array
        num_bars: Number of output bars (default: TINY_PEAKS_BARS)

    Returns:
        Dictionary with version and bars array
    """
    total = len(samples)
    window = total // num_bars
    if window < 1:
        return {"version": 1, "bars": [0.0] * num_bars}

    bars = []
    for i in range(num_bars):
        start = i * window
        end = start + window
        chunk = samples[start:end]
        bars.append(float(np.max(np.abs(chunk))))

    # Normalize so the loudest bar = 1.0
    peak = max(bars) if bars else 1.0
    if peak > 0:
        bars = [round(b / peak, 3) for b in bars]
    else:
        bars = [0.0] * num_bars

    return {"version": 1, "bars": bars}


def main():
    parser = argparse.ArgumentParser(
        description="Extract waveform peaks from audio files for visualization."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file (m4a, mp3, wav)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for the JSON files",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=None,
        help="Low-pass filter cutoff frequency in Hz (e.g., 500 for bass emphasis)",
    )

    args = parser.parse_args()

    audio_path = args.audio_file.resolve()
    output_dir = args.output.resolve()

    # Validate input file
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define resolutions
    resolutions = [
        ("overview.json", 16384),  # Zoomed out view
        ("medium.json", 4096),     # Medium zoom
        ("detail.json", 1024),     # Zoomed in view
    ]

    samples, sample_rate = load_audio(audio_path)
    duration = len(samples) / sample_rate
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Total samples: {len(samples):,}")

    if args.lowpass:
        print(f"Applying low-pass filter at {args.lowpass} Hz...")
        samples = lowpass_filter(samples, sample_rate, args.lowpass)

    # Generate each resolution
    for filename, samples_per_peak in resolutions:
        output_path = output_dir / filename
        print(f"\nGenerating {filename} ({samples_per_peak} samples/peak)...")

        peaks_data = generate_peaks_json(samples, sample_rate, samples_per_peak)
        num_peaks = len(peaks_data["peaks"])

        with open(output_path, "w") as f:
            json.dump(peaks_data, f)

        file_size = output_path.stat().st_size
        print(f"  Peaks: {num_peaks:,}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    # Generate tiny peaks for track list thumbnails
    tiny_path = output_dir / "tiny.json"
    print(f"\nGenerating tiny.json (~{TINY_PEAKS_BARS} bars)...")
    tiny_data = generate_tiny_peaks(samples)
    with open(tiny_path, "w") as f:
        json.dump(tiny_data, f)
    print(f"  Bars: {len(tiny_data['bars'])}")
    print(f"  File size: {tiny_path.stat().st_size:,} bytes")

    print(f"\nDone! Output files written to: {output_dir}")


if __name__ == "__main__":
    main()
