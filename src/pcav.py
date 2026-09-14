#!/usr/bin/env python3
"""
PCAV binary format reader/writer.

Implements the PCAV format specification (docs/pcav-format.md) for storing
audio analysis data: beats, peaks, and metadata in a compact binary format.

Primary usage: standalone .pcav files (via write_pcav_file/read_pcav_file).
Also supports embedding in WAV files (via write_pcav_to_wav/read_pcav_from_wav).
"""

import struct
import math
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy import signal

# Section type constants
SECTION_ANALYSIS = 0x01
SECTION_BEATS = 0x02
SECTION_PEAKS = 0x03           # Editor peaks (window=256, ~187.5 peaks/s at 48kHz)
SECTION_PEAKS_OVERVIEW = 0x08  # Overview peaks (window=8192, ~5.86/s)
SECTION_THUMB_64 = 0x10
SECTION_THUMB_128 = 0x11
SECTION_THUMB_256 = 0x12

# Peak window sizes (samples per peak at 48kHz)
MIPMAP_WINDOWS = [256, 8192]

PCAV_MAGIC = b"PCAV"
PCAV_VERSION = 2
MAX_SECTION_BYTES = 16 * 1024 * 1024
MAX_BEATS = 100_000
MAX_PEAKS = 2_000_000

# Key encoding table (0x00 = unknown, 0x01-0x0C = major, 0x0D-0x18 = minor)
# Includes both sharp and flat enharmonic equivalents
KEY_MAP = {
    "C major": 0x01, "C# major": 0x02, "Db major": 0x02,
    "D major": 0x03, "D# major": 0x04, "Eb major": 0x04,
    "E major": 0x05, "F major": 0x06, "F# major": 0x07, "Gb major": 0x07,
    "G major": 0x08, "G# major": 0x09, "Ab major": 0x09,
    "A major": 0x0A, "A# major": 0x0B, "Bb major": 0x0B, "B major": 0x0C,
    "C minor": 0x0D, "C# minor": 0x0E, "Db minor": 0x0E,
    "D minor": 0x0F, "D# minor": 0x10, "Eb minor": 0x10,
    "E minor": 0x11, "F minor": 0x12, "F# minor": 0x13, "Gb minor": 0x13,
    "G minor": 0x14, "G# minor": 0x15, "Ab minor": 0x15,
    "A minor": 0x16, "A# minor": 0x17, "Bb minor": 0x17, "B minor": 0x18,
}
# Reverse map uses sharps for consistency
KEY_MAP_REVERSE = {
    0x01: "C major", 0x02: "C# major", 0x03: "D major", 0x04: "D# major",
    0x05: "E major", 0x06: "F major", 0x07: "F# major", 0x08: "G major",
    0x09: "G# major", 0x0A: "A major", 0x0B: "A# major", 0x0C: "B major",
    0x0D: "C minor", 0x0E: "C# minor", 0x0F: "D minor", 0x10: "D# minor",
    0x11: "E minor", 0x12: "F minor", 0x13: "F# minor", 0x14: "G minor",
    0x15: "G# minor", 0x16: "A minor", 0x17: "A# minor", 0x18: "B minor",
}


@dataclass
class AnalysisSection:
    """Track-level analysis data."""
    duration_seconds: float  # v2: float64, measured frame count / sample rate
    bpm: float            # stored as uint16 × 10
    key: str | None       # musical key or None


@dataclass
class BeatsSection:
    """Beat timing data."""
    beat_times: list[float]  # beat times in seconds
    first_downbeat_index: int | None = None
    beats_per_bar: int = 4


@dataclass
class PeaksSection:
    """Waveform peaks (default 50 peaks/second)."""
    peaks: list[tuple[int, int]]  # (min, max) as int8 pairs
    sample_rate: int = 48000
    window_samples: int = 256


@dataclass
class ThumbnailSection:
    """Thumbnail image (WebP bytes)."""
    size: int      # 64, 128, or 256
    data: bytes    # raw WebP


@dataclass
class PCAVData:
    """Container for all PCAV section data."""
    analysis: AnalysisSection | None = None
    beats: BeatsSection | None = None
    peaks: PeaksSection | None = None
    overview_peaks: PeaksSection | None = None  # Coarse overview (section 0x08)
    thumb_64: ThumbnailSection | None = None
    thumb_128: ThumbnailSection | None = None
    thumb_256: ThumbnailSection | None = None


# --- Section Encoders ---

def encode_analysis(section: AnalysisSection) -> bytes:
    if not math.isfinite(section.duration_seconds) or section.duration_seconds <= 0:
        raise ValueError("Duration must be finite and positive")
    if not math.isfinite(section.bpm) or not 0 <= section.bpm <= 6553.5:
        raise ValueError("BPM must be finite and in range (0 means unknown)")
    if section.key is not None and section.key not in KEY_MAP:
        raise ValueError("Unknown musical key")
    return struct.pack("<dHBx", section.duration_seconds, round(section.bpm * 10), KEY_MAP.get(section.key, 0))


def decode_analysis(data: bytes, version: int = PCAV_VERSION) -> AnalysisSection:
    fmt = "<HHBx" if version == 1 else "<dHBx"
    if len(data) != struct.calcsize(fmt):
        raise ValueError("Invalid ANALYSIS length")
    duration, bpm, key = struct.unpack(fmt, data)
    if not math.isfinite(duration) or duration <= 0 or key > 24:
        raise ValueError("Invalid ANALYSIS values")
    return AnalysisSection(duration, bpm / 10, KEY_MAP_REVERSE.get(key))


def validate_beats(section: BeatsSection) -> None:
    if len(section.beat_times) > MAX_BEATS:
        raise ValueError("Too many beats")
    previous = -1
    for t in section.beat_times:
        if not math.isfinite(t) or t < 0 or t <= previous:
            raise ValueError("Beat times must be finite, nonnegative and strictly increasing")
        previous = t
    if not 1 <= section.beats_per_bar <= 32:
        raise ValueError("Invalid meter")
    idx = section.first_downbeat_index
    if idx is not None and (not isinstance(idx, int) or not 0 <= idx < len(section.beat_times)):
        raise ValueError("Invalid first downbeat index")


def encode_beats(section: BeatsSection) -> bytes:
    validate_beats(section)
    # Absolute microseconds: quantization error never accumulates between beats.
    times = [round(t * 1_000_000) for t in section.beat_times]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Beats collapse at microsecond resolution")
    if any(t >= 2**53 for t in times):
        raise ValueError("Beat timestamp exceeds safe integer range")
    if len(times) * 8 + 12 > MAX_SECTION_BYTES:
        raise ValueError("Too many beats")
    idx = section.first_downbeat_index
    return struct.pack("<IiB3x", len(times), -1 if idx is None else idx, section.beats_per_bar) + struct.pack(f"<{len(times)}Q", *times)


def decode_beats(data: bytes, version: int = PCAV_VERSION) -> BeatsSection:
    if version == 1:
        if len(data) < 4:
            raise ValueError("Truncated BEATS header")
        first, count = struct.unpack("<HH", data[:4])
        if len(data) != 4 + max(0, count - 1):
            raise ValueError("BEATS count does not match length")
        times = [first / 1000] if count else []
        time_ms = first
        for interval in data[4:]:
            time_ms += interval * 10
            times.append(time_ms / 1000)
        result = BeatsSection(times)
    else:
        if len(data) < 12:
            raise ValueError("Truncated BEATS header")
        count, idx, meter = struct.unpack("<IiB3x", data[:12])
        if count > MAX_BEATS or len(data) != 12 + count * 8 or idx < -1:
            raise ValueError("Invalid BEATS count or phase")
        times_us = struct.unpack(f"<{count}Q", data[12:])
        if any(t >= 2**53 for t in times_us):
            raise ValueError("Beat timestamp exceeds safe integer range")
        result = BeatsSection([t / 1_000_000 for t in times_us], None if idx == -1 else idx, meter)
    validate_beats(result)
    return result


def encode_peaks(section: PeaksSection) -> bytes:
    if not 1 <= section.sample_rate <= 768000 or not 1 <= section.window_samples <= 2**24:
        raise ValueError("Invalid peak timebase")
    if len(section.peaks) > MAX_PEAKS or 12 + 2 * len(section.peaks) > MAX_SECTION_BYTES:
        raise ValueError("Too many peaks")
    data = bytearray(struct.pack("<III", len(section.peaks), section.sample_rate, section.window_samples))
    for lo, hi in section.peaks:
        if not -128 <= lo <= hi <= 127:
            raise ValueError("Invalid peak range")
        data.extend(struct.pack("bb", lo, hi))
    return bytes(data)


def decode_peaks(data: bytes, version: int = PCAV_VERSION, window: int = 256) -> PeaksSection:
    if len(data) < (2 if version == 1 else 12):
        raise ValueError("Truncated PEAKS header")
    rate = 48000
    if version == 1:
        count = struct.unpack("<H", data[:2])[0]
        start = 2
        if len(data) != 2 + count * 2:
            if len(data) < 4:
                raise ValueError("Truncated PEAKS count")
            count = struct.unpack("<I", data[:4])[0]
            start = 4
        elif window == 256:
            # Original v1 editor format was 10 peaks/s, with a uint16 count.
            window = 4800
    else:
        count, rate, window = struct.unpack("<III", data[:12])
        start = 12
        if not 1 <= rate <= 768000 or not 1 <= window <= 2**24:
            raise ValueError("Invalid peak timebase")
    if count > MAX_PEAKS or len(data) != start + count * 2:
        raise ValueError("PEAKS count does not match length")
    peaks = list(struct.iter_unpack("bb", data[start:]))
    if any(lo > hi for lo, hi in peaks):
        raise ValueError("Invalid peak range")
    return PeaksSection(peaks, rate, window)


def encode_thumbnail(section: ThumbnailSection) -> bytes:
    """Encode thumbnail section (raw WebP bytes)."""
    return section.data


def decode_thumbnail(data: bytes, size: int) -> ThumbnailSection:
    """Decode thumbnail section."""
    return ThumbnailSection(size=size, data=data)


# --- PCAV Chunk Assembly ---

def build_pcav_payload(pcav: PCAVData) -> bytes:
    """Build complete PCAV payload from sections."""
    if pcav.analysis and pcav.beats:
        if any(t >= pcav.analysis.duration_seconds for t in pcav.beats.beat_times):
            raise ValueError("Beat lies outside audio duration")
    sections: list[tuple[int, bytes]] = []

    if pcav.analysis:
        sections.append((SECTION_ANALYSIS, encode_analysis(pcav.analysis)))
    if pcav.beats:
        sections.append((SECTION_BEATS, encode_beats(pcav.beats)))
    if pcav.peaks:
        sections.append((SECTION_PEAKS, encode_peaks(pcav.peaks)))
    if pcav.overview_peaks:
        sections.append((SECTION_PEAKS_OVERVIEW, encode_peaks(pcav.overview_peaks)))
    if pcav.thumb_64:
        sections.append((SECTION_THUMB_64, encode_thumbnail(pcav.thumb_64)))
    if pcav.thumb_128:
        sections.append((SECTION_THUMB_128, encode_thumbnail(pcav.thumb_128)))
    if pcav.thumb_256:
        sections.append((SECTION_THUMB_256, encode_thumbnail(pcav.thumb_256)))

    section_count = len(sections)

    # Header: magic (4) + version (1) + section_count (1) = 6 bytes
    # Directory: 9 bytes per section
    header_size = 6 + 9 * section_count

    # Build directory and concatenate section data
    directory = b""
    section_data = b""
    current_offset = header_size

    for section_type, data in sections:
        directory += struct.pack("<BII", section_type, current_offset, len(data))
        section_data += data
        current_offset += len(data)

    # Assemble payload
    payload = PCAV_MAGIC + struct.pack("BB", PCAV_VERSION, section_count)
    payload += directory + section_data

    return payload


def parse_pcav_payload(payload: bytes) -> PCAVData:
    if len(payload) < 6 or payload[:4] != PCAV_MAGIC:
        raise ValueError("Invalid PCAV header")
    version, count = payload[4:6]
    if version not in (1, 2):
        raise ValueError(f"Unsupported PCAV version: {version}")
    header_size = 6 + count * 9
    if len(payload) < header_size:
        raise ValueError("Truncated PCAV directory")
    entries = [struct.unpack_from("<BII", payload, 6 + i * 9) for i in range(count)]
    seen = set()
    end = header_size
    for kind, offset, length in sorted(entries, key=lambda e: e[1]):
        if kind in seen or offset < end or length > MAX_SECTION_BYTES or offset + length > len(payload):
            raise ValueError("Invalid, overlapping or truncated PCAV section")
        seen.add(kind)
        end = offset + length
    pcav = PCAVData()
    for kind, offset, length in entries:
        data = payload[offset:offset + length]
        if kind == SECTION_ANALYSIS:
            pcav.analysis = decode_analysis(data, version)
        elif kind == SECTION_BEATS:
            pcav.beats = decode_beats(data, version)
        elif kind in (SECTION_PEAKS, SECTION_PEAKS_OVERVIEW):
            section = decode_peaks(data, version, 256 if kind == SECTION_PEAKS else 8192)
            if kind == SECTION_PEAKS:
                pcav.peaks = section
            else:
                pcav.overview_peaks = section
        elif kind in (SECTION_THUMB_64, SECTION_THUMB_128, SECTION_THUMB_256):
            size = {SECTION_THUMB_64: 64, SECTION_THUMB_128: 128, SECTION_THUMB_256: 256}[kind]
            setattr(pcav, f"thumb_{size}", decode_thumbnail(data, size))
    if version == 2 and pcav.analysis and pcav.beats:
        if any(t >= pcav.analysis.duration_seconds for t in pcav.beats.beat_times):
            raise ValueError("Beat lies outside audio duration")
    return pcav


# --- WAV File Operations ---

def find_riff_chunks(f: BinaryIO) -> list[tuple[str, int, int]]:
    """Find all chunks in a RIFF file. Returns [(id, offset, size), ...]."""
    f.seek(0)
    riff_header = f.read(12)
    if riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    chunks = []
    offset = 12

    while True:
        f.seek(offset)
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            break

        chunk_id = chunk_header[:4].decode("ascii", errors="replace")
        chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
        chunks.append((chunk_id, offset, chunk_size))

        # Move to next chunk (align to 2-byte boundary)
        offset += 8 + chunk_size
        if chunk_size % 2:
            offset += 1

    return chunks


def read_pcav_from_wav(wav_path: Path) -> PCAVData | None:
    """Read PCAV chunk from WAV file. Returns None if not found."""
    with open(wav_path, "rb") as f:
        chunks = find_riff_chunks(f)

        for chunk_id, offset, size in chunks:
            if chunk_id == "PCAV":
                f.seek(offset + 8)  # Skip chunk header
                payload = f.read(size)
                return parse_pcav_payload(payload)

    return None


def write_pcav_to_wav(wav_path: Path, pcav: PCAVData) -> None:
    """Write or update PCAV chunk in WAV file."""
    payload = build_pcav_payload(pcav)

    # Pad to 2-byte boundary per RIFF spec
    if len(payload) % 2:
        payload += b"\x00"

    with open(wav_path, "r+b") as f:
        chunks = find_riff_chunks(f)

        # Check if PCAV already exists
        pcav_chunk = None
        for chunk_id, offset, size in chunks:
            if chunk_id == "PCAV":
                pcav_chunk = (offset, size)
                break

        if pcav_chunk:
            # Remove existing PCAV chunk and rewrite
            _rewrite_wav_without_chunk(f, chunks, "PCAV")
            chunks = find_riff_chunks(f)

        # Append new PCAV chunk at end
        f.seek(0, 2)  # Seek to end
        file_end = f.tell()

        # Write chunk: ID (4) + size (4) + payload
        chunk_size = len(payload)
        f.write(b"PCAV")
        f.write(struct.pack("<I", chunk_size))
        f.write(payload)

        # Update RIFF size in header
        new_file_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", new_file_size - 8))


def write_pcav_file(pcav_path: Path, pcav: PCAVData) -> None:
    """Write PCAV data to a standalone .pcav file."""
    payload = build_pcav_payload(pcav)
    with open(pcav_path, "wb") as f:
        f.write(payload)


def read_pcav_file(pcav_path: Path) -> PCAVData:
    """Read PCAV data from a standalone .pcav file."""
    with open(pcav_path, "rb") as f:
        payload = f.read()
    return parse_pcav_payload(payload)


def _rewrite_wav_without_chunk(f: BinaryIO, chunks: list, exclude_id: str) -> None:
    """Rewrite WAV file excluding a specific chunk."""
    # Read all data
    f.seek(0)
    original_data = f.read()

    # Rebuild without the excluded chunk
    f.seek(0)
    f.truncate()

    # Write RIFF header (will update size at end)
    f.write(b"RIFF")
    f.write(b"\x00\x00\x00\x00")  # Placeholder size
    f.write(b"WAVE")

    # Write all chunks except excluded
    for chunk_id, offset, size in chunks:
        if chunk_id == exclude_id:
            continue

        # Calculate actual chunk data size including padding
        chunk_data_size = size
        if size % 2:
            chunk_data_size += 1

        # Write chunk header and data
        chunk_start = offset
        chunk_end = offset + 8 + chunk_data_size
        f.write(original_data[chunk_start:chunk_end])

    # Update RIFF size
    file_size = f.tell()
    f.seek(4)
    f.write(struct.pack("<I", file_size - 8))


# --- High-level Helpers ---

def lowpass_filter(samples: np.ndarray, sample_rate: int, cutoff_hz: int = 300) -> np.ndarray:
    """Low-pass filter audio to isolate bass/kick energy for waveform display."""
    nyquist = sample_rate / 2
    normalized_cutoff = min(cutoff_hz / nyquist, 0.99)
    sos = signal.butter(4, normalized_cutoff, btype='low', output='sos')
    return signal.sosfiltfilt(sos, samples, padlen=min(15, len(samples) - 1)).astype(np.float64)


def build_peak_mipmaps(samples, sample_rate: int, filter_hz: int = 300) -> list[PeaksSection]:
    """Generate peak levels from audio samples: editor (window=256) and overview (window=8192).

    When filter_hz is set (default 300Hz), applies a low-pass filter before computing
    peaks so the waveform shape shows kicks/bass energy rather than high-frequency noise.

    Returns [editor_peaks, overview_peaks].
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Peaks require nonempty finite mono samples")
    if not isinstance(sample_rate, int) or not 1 <= sample_rate <= 768000:
        raise ValueError("Invalid sample rate")
    if filter_hz:
        samples = lowpass_filter(samples, sample_rate, filter_hz)
    peak_amp = np.max(np.abs(samples))
    if peak_amp > 0:
        samples = samples / peak_amp
    levels = []
    for window in MIPMAP_WINDOWS:
        starts = np.arange(0, len(samples), window)
        # Include the final partial window, including one-sample inputs.
        mins = np.minimum.reduceat(samples, starts)
        maxs = np.maximum.reduceat(samples, starts)
        peaks = [(int(round(lo * 127)), int(round(hi * 127))) for lo, hi in zip(mins, maxs)]
        levels.append(PeaksSection(peaks, sample_rate, window))
    return levels


def has_stale_peak_sections(pcav_path: Path) -> bool:
    """Check if PCAV file contains intermediate peak sections (0x04-0x07) that should be removed."""
    data = pcav_path.read_bytes()[:200]
    if len(data) < 6 or data[:4] != PCAV_MAGIC:
        return False
    section_count = data[5]
    for i in range(section_count):
        offset = 6 + i * 9
        if offset + 9 > len(data):
            break
        if 0x04 <= data[offset] <= 0x07:
            return True
    return False


def peaks_from_samples(samples, sample_rate: int, peaks_per_second: int = 10) -> PeaksSection:
    """Generate PCAV peaks from audio samples.

    Args:
        samples: Audio samples (numpy array or list)
        sample_rate: Sample rate in Hz
        peaks_per_second: Peaks per second (default: 10)

    Returns:
        PeaksSection with int8 min/max pairs
    """
    samples = np.asarray(samples)
    samples_per_peak = sample_rate // peaks_per_second
    num_peaks = len(samples) // samples_per_peak

    if num_peaks == 0:
        return PeaksSection(peaks=[])

    # Truncate and reshape
    truncated = samples[:num_peaks * samples_per_peak]
    windows = truncated.reshape(num_peaks, samples_per_peak)

    # Compute min/max and convert to int8 (-128 to 127)
    mins = np.min(windows, axis=1)
    maxs = np.max(windows, axis=1)

    peaks = []
    for min_val, max_val in zip(mins, maxs):
        min_int8 = int(round(min_val * 127))
        max_int8 = int(round(max_val * 127))
        peaks.append((min_int8, max_int8))

    return PeaksSection(peaks=peaks)


def beats_from_times(beat_times: list[float]) -> BeatsSection:
    """Create BeatsSection from list of beat times in seconds."""
    return BeatsSection(beat_times=list(beat_times))


def analysis_from_dict(data: dict) -> AnalysisSection:
    """Create AnalysisSection from analysis dict."""
    return AnalysisSection(
        duration_seconds=float(data.get("duration", 0)),
        bpm=float(data.get("bpm", 0)),
        key=data.get("key"),
    )
