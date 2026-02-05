#!/usr/bin/env python3
"""
PCAV binary format reader/writer.

Implements the PCAV format specification (docs/pcav-format.md) for storing
audio analysis data: beats, peaks, and metadata in a compact binary format.

Primary usage: standalone .pcav files (via write_pcav_file/read_pcav_file).
Also supports embedding in WAV files (via write_pcav_to_wav/read_pcav_from_wav).
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

# Section type constants
SECTION_ANALYSIS = 0x01
SECTION_BEATS = 0x02
SECTION_PEAKS = 0x03
SECTION_THUMB_64 = 0x10
SECTION_THUMB_128 = 0x11
SECTION_THUMB_256 = 0x12

PCAV_MAGIC = b"PCAV"
PCAV_VERSION = 1

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
    duration_seconds: int  # uint16
    bpm: float            # stored as uint16 × 10
    key: str | None       # musical key or None


@dataclass
class BeatsSection:
    """Beat timing data."""
    beat_times: list[float]  # beat times in seconds


@dataclass
class PeaksSection:
    """Waveform peaks at 10 peaks/second."""
    peaks: list[tuple[int, int]]  # (min, max) as int8 pairs


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
    thumb_64: ThumbnailSection | None = None
    thumb_128: ThumbnailSection | None = None
    thumb_256: ThumbnailSection | None = None


# --- Section Encoders ---

def encode_analysis(section: AnalysisSection) -> bytes:
    """Encode ANALYSIS section (6 bytes)."""
    duration = min(section.duration_seconds, 65535)
    bpm_encoded = int(round(section.bpm * 10))
    key_byte = KEY_MAP.get(section.key, 0x00) if section.key else 0x00
    return struct.pack("<HHBx", duration, bpm_encoded, key_byte)


def decode_analysis(data: bytes) -> AnalysisSection:
    """Decode ANALYSIS section."""
    duration, bpm_encoded, key_byte = struct.unpack("<HHBx", data[:6])
    key = KEY_MAP_REVERSE.get(key_byte)
    return AnalysisSection(
        duration_seconds=duration,
        bpm=bpm_encoded / 10.0,
        key=key,
    )


def encode_beats(section: BeatsSection) -> bytes:
    """Encode BEATS section as first beat + interval deltas."""
    if not section.beat_times:
        return struct.pack("<HH", 0, 0)

    beats = section.beat_times
    first_beat_ms = int(round(beats[0] * 1000))
    beat_count = len(beats)

    # Compute intervals in centiseconds (10ms units)
    intervals = []
    for i in range(1, len(beats)):
        delta_sec = beats[i] - beats[i - 1]
        delta_cs = int(round(delta_sec * 100))
        intervals.append(min(max(delta_cs, 0), 255))  # clamp to uint8

    return struct.pack("<HH", first_beat_ms, beat_count) + bytes(intervals)


def decode_beats(data: bytes) -> BeatsSection:
    """Decode BEATS section."""
    first_beat_ms, beat_count = struct.unpack("<HH", data[:4])
    if beat_count == 0:
        return BeatsSection(beat_times=[])

    intervals = list(data[4:4 + beat_count - 1])

    # Reconstruct beat times
    beat_times = [first_beat_ms / 1000.0]
    for interval_cs in intervals:
        beat_times.append(beat_times[-1] + interval_cs / 100.0)

    return BeatsSection(beat_times=beat_times)


def encode_peaks(section: PeaksSection) -> bytes:
    """Encode PEAKS section as int8 min/max pairs."""
    peak_count = len(section.peaks)
    data = struct.pack("<H", peak_count)

    for min_val, max_val in section.peaks:
        # Clamp to int8 range
        min_clamped = max(-128, min(127, min_val))
        max_clamped = max(-128, min(127, max_val))
        data += struct.pack("bb", min_clamped, max_clamped)

    return data


def decode_peaks(data: bytes) -> PeaksSection:
    """Decode PEAKS section."""
    peak_count = struct.unpack("<H", data[:2])[0]
    peaks = []

    for i in range(peak_count):
        offset = 2 + i * 2
        min_val, max_val = struct.unpack("bb", data[offset:offset + 2])
        peaks.append((min_val, max_val))

    return PeaksSection(peaks=peaks)


def encode_thumbnail(section: ThumbnailSection) -> bytes:
    """Encode thumbnail section (raw WebP bytes)."""
    return section.data


def decode_thumbnail(data: bytes, size: int) -> ThumbnailSection:
    """Decode thumbnail section."""
    return ThumbnailSection(size=size, data=data)


# --- PCAV Chunk Assembly ---

def build_pcav_payload(pcav: PCAVData) -> bytes:
    """Build complete PCAV payload from sections."""
    sections: list[tuple[int, bytes]] = []

    if pcav.analysis:
        sections.append((SECTION_ANALYSIS, encode_analysis(pcav.analysis)))
    if pcav.beats:
        sections.append((SECTION_BEATS, encode_beats(pcav.beats)))
    if pcav.peaks:
        sections.append((SECTION_PEAKS, encode_peaks(pcav.peaks)))
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
    """Parse PCAV payload into sections."""
    if len(payload) < 6 or payload[:4] != PCAV_MAGIC:
        raise ValueError("Invalid PCAV magic bytes")

    version = payload[4]
    if version != PCAV_VERSION:
        raise ValueError(f"Unsupported PCAV version: {version}")

    section_count = payload[5]
    pcav = PCAVData()

    # Parse directory
    for i in range(section_count):
        dir_offset = 6 + i * 9
        section_type, data_offset, data_length = struct.unpack(
            "<BII", payload[dir_offset:dir_offset + 9]
        )

        section_data = payload[data_offset:data_offset + data_length]

        if section_type == SECTION_ANALYSIS:
            pcav.analysis = decode_analysis(section_data)
        elif section_type == SECTION_BEATS:
            pcav.beats = decode_beats(section_data)
        elif section_type == SECTION_PEAKS:
            pcav.peaks = decode_peaks(section_data)
        elif section_type == SECTION_THUMB_64:
            pcav.thumb_64 = decode_thumbnail(section_data, 64)
        elif section_type == SECTION_THUMB_128:
            pcav.thumb_128 = decode_thumbnail(section_data, 128)
        elif section_type == SECTION_THUMB_256:
            pcav.thumb_256 = decode_thumbnail(section_data, 256)

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

def peaks_from_samples(samples, sample_rate: int, peaks_per_second: int = 10) -> PeaksSection:
    """Generate PCAV peaks from audio samples.

    Args:
        samples: Audio samples (numpy array or list)
        sample_rate: Sample rate in Hz
        peaks_per_second: Peaks per second (default: 10)

    Returns:
        PeaksSection with int8 min/max pairs
    """
    import numpy as np

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
        duration_seconds=int(round(data.get("duration", 0))),
        bpm=float(data.get("bpm", 0)),
        key=data.get("key"),
    )
