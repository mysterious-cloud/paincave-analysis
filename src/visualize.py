#!/usr/bin/env python3
"""
Visualize audio analysis data from Pain Cave.

Generates stacked charts for tempo, intensity, and features.
- Intensity chart shows song boundaries as colored regions with events strip
- BPM shown as text labels on top axis
- Each chart has its own time axis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_analysis(path: Path) -> dict:
    """Load analysis JSON file."""
    with open(path) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def get_song_at_time(songs: list[dict], t: float) -> tuple[int, dict | None]:
    """Get song index and data at given time."""
    for i, song in enumerate(songs):
        start = song.get("start", song.get("start_bar", 0))
        end = song.get("end", song.get("end_bar", 0))
        if start <= t < end:
            return i, song
    return -1, None


def detect_flux_peaks(flux_data: list[dict], songs: list[dict], duration: float) -> list[float]:
    """Detect significant spectral flux peaks - only the big spikes for exercise cues."""
    if not flux_data:
        return []

    times = np.array([d["t"] for d in flux_data])
    values = np.array([d["v"] for d in flux_data])

    # Absolute threshold - only the really notable spikes (visible on the graph)
    ABSOLUTE_THRESHOLD = 0.35  # Only peaks above this value
    MIN_PROMINENCE = 0.15      # Must stand out from surrounding values

    from scipy.signal import find_peaks

    # Find peaks with absolute threshold - no per-song normalization
    peaks, _ = find_peaks(values, height=ABSOLUTE_THRESHOLD, prominence=MIN_PROMINENCE, distance=10)

    return times[peaks].tolist()


def plot_intensity_with_songs(analysis: dict, ax: plt.Axes):
    """Plot intensity curve with song-colored regions, events strip, BPM labels, and flux peaks."""
    intensity_data = analysis.get("intensity", {}).get("data", [])
    songs = analysis.get("songs", [])
    events = analysis.get("events", [])
    tempo_curve = analysis.get("tempo", {}).get("curve", [])
    flux_data = analysis.get("features", {}).get("flux", [])
    duration = analysis.get("duration", 1800)

    if not intensity_data:
        return

    times = [d.get("start_t", 0) for d in intensity_data]
    values = [d["v"] for d in intensity_data]

    # Create bar-to-time lookup
    bar_times = {d["bar"]: d.get("start_t", 0) for d in intensity_data}

    # Build tempo lookup for BPM labels
    tempo_times = []
    tempo_bpms = []
    for t in tempo_curve:
        bar = t["bar"]
        if bar in bar_times:
            tempo_times.append(bar_times[bar])
            tempo_bpms.append(t["bpm"])

    # Song colors - use more distinct colors
    song_colors = plt.cm.Set3(np.linspace(0, 1, max(len(songs), 1)))

    # Track last shown BPM to avoid duplicate labels
    last_shown_bpm = None

    # Pre-compute boundary values for smooth transitions between songs
    # Each boundary uses the average of adjacent songs' edge values
    boundary_values = {}
    if songs:
        for i, song in enumerate(songs):
            t_start = song.get("start", song.get("start_bar", 0))
            t_end = song.get("end", song.get("end_bar", duration))

            # Get this song's edge values
            song_vals = [v for t, v in zip(times, values) if t_start <= t < t_end]
            if song_vals:
                # Store this song's contribution to boundaries
                if t_start not in boundary_values:
                    boundary_values[t_start] = []
                boundary_values[t_start].append(song_vals[0])

                if t_end not in boundary_values:
                    boundary_values[t_end] = []
                boundary_values[t_end].append(song_vals[-1])

        # Average boundary values where songs meet
        for t in boundary_values:
            boundary_values[t] = np.mean(boundary_values[t])

    # Plot intensity as colored regions per song
    if songs:
        for i, song in enumerate(songs):
            # Get time range for this song
            if "start" in song:
                t_start, t_end = song["start"], song["end"]
            else:
                t_start = bar_times.get(song.get("start_bar", 1), 0)
                t_end = bar_times.get(song.get("end_bar", len(bar_times)), duration)

            # Get intensity data points within this song
            song_times = []
            song_values = []
            for t, v in zip(times, values):
                if t_start <= t < t_end:
                    song_times.append(t)
                    song_values.append(v)

            if song_times:
                # Use pre-computed boundary values for smooth transitions
                start_val = boundary_values.get(t_start, song_values[0])
                end_val = boundary_values.get(t_end, song_values[-1])

                song_times = [t_start] + song_times + [t_end]
                song_values = [start_val] + song_values + [end_val]

                ax.fill_between(song_times, song_values, alpha=0.7, color=song_colors[i])
                ax.plot(song_times, song_values, color='black', linewidth=0.3, alpha=0.5)

            # Add BPM label(s) at top of each song region
            # Detect tempo changes within song and show multiple labels if needed
            song_tempo_data = [(t, b) for t, b in zip(tempo_times, tempo_bpms) if t_start <= t <= t_end]

            if song_tempo_data:
                song_bpms_only = [b for _, b in song_tempo_data]
                bpm_std = np.std(song_bpms_only)
                bpm_min = min(song_bpms_only)
                bpm_max = max(song_bpms_only)

                if bpm_std > 3.0 and (bpm_max - bpm_min) > 10:
                    # Significant tempo change - detect segments and show multiple labels
                    # Split song into segments where BPM is stable
                    segments = []
                    seg_start = song_tempo_data[0][0]
                    seg_bpms = [song_tempo_data[0][1]]

                    for j in range(1, len(song_tempo_data)):
                        t_prev, bpm_prev = song_tempo_data[j-1]
                        t_curr, bpm_curr = song_tempo_data[j]

                        # If BPM jumps significantly, start new segment
                        if abs(bpm_curr - np.median(seg_bpms)) > 8:
                            if len(seg_bpms) >= 4:  # Only show if segment is substantial
                                seg_end = t_prev
                                segments.append((seg_start, seg_end, int(round(np.median(seg_bpms)))))
                            seg_start = t_curr
                            seg_bpms = [bpm_curr]
                        else:
                            seg_bpms.append(bpm_curr)

                    # Add final segment
                    if len(seg_bpms) >= 4:
                        segments.append((seg_start, t_end, int(round(np.median(seg_bpms)))))

                    # Show label for each segment at the START of that tempo period
                    # Skip if same as last shown BPM
                    for seg_start_t, seg_end, seg_bpm in segments:
                        if seg_bpm != last_shown_bpm:
                            ax.annotate(f"{seg_bpm}", xy=(seg_start_t, 96), fontsize=7, fontweight='bold',
                                       ha='left', va='top', color='#333',
                                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                                edgecolor='gray', alpha=0.85))
                            last_shown_bpm = seg_bpm
                else:
                    # Stable tempo - show single median at start of song
                    median_bpm = int(round(np.median(song_bpms_only)))
                    if median_bpm != last_shown_bpm:
                        ax.annotate(f"{median_bpm}", xy=(t_start, 96), fontsize=8, fontweight='bold',
                                   ha='left', va='top', color='#333',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                            edgecolor='gray', alpha=0.85))
                        last_shown_bpm = median_bpm
            else:
                # Fallback: use global BPM at start of song
                median_bpm = int(round(analysis.get("tempo", {}).get("global_bpm", 120)))
                if median_bpm != last_shown_bpm:
                    ax.annotate(f"{median_bpm}", xy=(t_start, 96), fontsize=8, fontweight='bold',
                               ha='left', va='top', color='#333',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                        edgecolor='gray', alpha=0.85))
                    last_shown_bpm = median_bpm
    else:
        # No songs - just plot as single color
        ax.fill_between(times, values, alpha=0.6, color='steelblue')
        ax.plot(times, values, color='darkblue', linewidth=0.5)

    # Draw song boundary lines (vertical dashed lines at song transitions)
    if songs:
        for i, song in enumerate(songs):
            t_start = song.get("start", song.get("start_bar", 0))
            if t_start > 0:  # Don't draw line at t=0
                ax.axvline(x=t_start, color='black', linestyle='--', linewidth=1, alpha=0.6)

            # Add song title label at bottom
            t_end = song.get("end", song.get("end_bar", duration))
            t_mid = (t_start + t_end) / 2
            title = song.get("title", f"Track {i+1}")
            ax.annotate(title, xy=(t_mid, 3), fontsize=7, ha='center', va='bottom',
                       color='#333', alpha=0.8, rotation=0,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='none', alpha=0.7))

    # Spectral flux peaks as dots at top (potential exercise cue points)
    flux_peaks = detect_flux_peaks(flux_data, songs, duration)
    if flux_peaks:
        ax.scatter(flux_peaks, [98] * len(flux_peaks), marker='v', s=15, c='purple',
                  alpha=0.7, zorder=10, label='Cue points')

    # Events strip at bottom (y=0 to y=2.5) - half height
    event_colors = {"build": "green", "drop": "red", "breakdown": "blue"}
    event_height = 2.5

    for event in events:
        event_type = event.get("type", "")
        color = event_colors.get(event_type, "gray")

        if "bar" in event:
            # Point event (drop)
            t = bar_times.get(event["bar"], 0)
            ax.axvline(x=t, ymin=0, ymax=0.025, color=color, linewidth=3, alpha=0.8)
        elif "start_bar" in event and "end_bar" in event:
            # Range event (build, breakdown)
            t_start = bar_times.get(event["start_bar"], 0)
            t_end = bar_times.get(event["end_bar"], t_start + 10)
            ax.axhspan(0, event_height, xmin=t_start/duration, xmax=t_end/duration,
                      color=color, alpha=0.6)

    ax.set_ylabel("Intensity (0-100)")
    ax.set_title("Workout Intensity (BPM labels, ▼ cue points)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3, axis='y')


def plot_tempo_with_labels(analysis: dict, ax: plt.Axes):
    """Plot tempo with BPM text labels on top axis."""
    intensity_data = analysis.get("intensity", {}).get("data", [])
    tempo_curve = analysis.get("tempo", {}).get("curve", [])
    songs = analysis.get("songs", [])
    duration = analysis.get("duration", 1800)

    if not tempo_curve or not intensity_data:
        return

    bar_times = {d["bar"]: d.get("start_t", 0) for d in intensity_data}

    times = []
    bpms = []
    for t in tempo_curve:
        bar = t["bar"]
        if bar in bar_times:
            times.append(bar_times[bar])
            bpms.append(t["bpm"])

    if not times:
        return

    # Plot tempo curve
    ax.plot(times, bpms, 'steelblue', linewidth=1, alpha=0.8)
    ax.fill_between(times, bpms, alpha=0.3, color='steelblue')

    # Add BPM text labels at top
    # Group by song and compute median BPM per song
    bpm_labels = []

    if songs:
        for song in songs:
            if "start" in song:
                t_start, t_end = song["start"], song["end"]
            else:
                t_start = bar_times.get(song.get("start_bar", 1), 0)
                t_end = bar_times.get(song.get("end_bar", len(bar_times)), duration)

            # Get BPMs in this song range
            song_bpms = [b for t, b in zip(times, bpms) if t_start <= t < t_end]
            if song_bpms:
                median_bpm = int(round(np.median(song_bpms)))
                t_mid = (t_start + t_end) / 2
                bpm_labels.append((t_mid, median_bpm, song.get("title", "")))
    else:
        # Single global BPM
        global_bpm = analysis.get("tempo", {}).get("global_bpm", 0)
        if global_bpm:
            bpm_labels.append((duration / 2, int(round(global_bpm)), ""))

    # Add BPM labels at top of plot
    y_top = max(bpms) + 5 if bpms else 150
    for t_mid, bpm, title in bpm_labels:
        ax.annotate(f"{bpm}", xy=(t_mid, y_top), fontsize=9, fontweight='bold',
                   ha='center', va='bottom', color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Global BPM reference line
    global_bpm = analysis.get("tempo", {}).get("global_bpm", 0)
    if global_bpm:
        ax.axhline(y=global_bpm, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_ylabel("BPM")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Tempo")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Set y limits with padding for labels
    if bpms:
        ax.set_ylim(min(bpms) - 10, max(bpms) + 15)


def plot_single_feature(data: list[dict], ax: plt.Axes, label: str, color: str, duration: float,
                        transform: str | None = None, smooth_window: int = 0):
    """Plot a single feature over time.

    Args:
        transform: Optional transformation - 'lufs' for LUFS-specific scaling
        smooth_window: Rolling average window size (0 = no smoothing)
    """
    if not data:
        return

    times = np.array([d["t"] for d in data])
    values = np.array([d["v"] for d in data])

    # Apply smoothing if requested
    if smooth_window > 1 and len(values) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        values = np.convolve(values, kernel, mode='same')

    # Apply LUFS transformation to expand useful range
    ylim = None
    if transform == 'lufs':
        # Histogram equalization - guarantees uniform distribution across visual range
        # This maximizes visible variation regardless of the original distribution
        from scipy.stats import rankdata
        ranks = rankdata(values, method='average')
        values = (ranks - 1) / (len(ranks) - 1)  # Normalize ranks to 0-1

        ylim = (0, 1.05)

    ax.fill_between(times, values, alpha=0.4, color=color)
    ax.plot(times, values, color=color, linewidth=0.5, alpha=0.8)

    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)
    if ylim:
        ax.set_ylim(ylim)


def create_visualization(analysis: dict, output_path: Path | None = None):
    """Create stacked visualization with time on x-axis."""
    duration = analysis.get("duration", 1800)
    duration_str = format_time(duration)

    features = analysis.get("features", {})

    # Count plots: intensity+songs (with BPM labels) + raw features
    feature_keys = ["lufs", "onset_rate", "low_band", "flux"]
    active_features = [k for k in feature_keys if features.get(k)]

    num_plots = 1 + len(active_features)  # intensity + features (no separate tempo chart)

    # Create figure - double width (32"), intensity taller, features shorter
    # All share same x-axis for clean stacking
    height_ratios = [3] + [0.6] * len(active_features)
    fig, axes = plt.subplots(num_plots, 1, figsize=(32, sum(height_ratios) * 1.5),
                              sharex=True, layout='constrained',
                              gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.05})
    fig.suptitle(f"Audio Analysis - {duration_str} ({duration:.0f}s)", fontsize=16)

    # Handle single plot case (axes won't be array)
    if num_plots == 1:
        axes = [axes]

    idx = 0

    # Intensity with songs, events, and BPM labels
    plot_intensity_with_songs(analysis, axes[idx])
    idx += 1

    # Individual features - (key, label, color, transform, smooth_window)
    feature_config = [
        ("lufs", "Loudness", "steelblue", "lufs", 8),  # Transform + smooth
        ("onset_rate", "Onset Density", "coral", None, 0),
        ("low_band", "Bass Energy (20-150Hz)", "forestgreen", None, 0),
        ("flux", "Spectral Flux", "purple", None, 0),
    ]

    for key, label, color, transform, smooth in feature_config:
        data = features.get(key, [])
        if data:
            plot_single_feature(data, axes[idx], label, color, duration, transform, smooth)
            idx += 1

    # Add time markers to each axis
    for ax in axes:
        ax.xaxis.set_major_locator(plt.MultipleLocator(300))  # Every 5 min
        ax.xaxis.set_minor_locator(plt.MultipleLocator(60))   # Every 1 min

    # Only show x-axis label on bottom chart (shared x-axis)
    axes[-1].set_xlabel("Time (seconds)", fontsize=11)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Pain Cave audio analysis")
    parser.add_argument("input", type=Path, help="Path to analysis.json")
    parser.add_argument("--output", "-o", type=Path, help="Output image path (default: show)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return

    analysis = load_analysis(args.input)
    create_visualization(analysis, args.output)


if __name__ == "__main__":
    main()
