# PCAV Format Specification

Version 1.0

## Overview

PCAV (Pain Cave Audio) is a compact binary format for storing audio analysis data: beat timing, waveform peaks, and track metadata.

**File:** `analysis.pcav` in each track directory.

## Structure

```
Offset  Size     Field
------  -------  -----
0       4        Magic "PCAV" (validation)
4       1        Version (currently 1)
5       1        Section count (N)
6       9×N      Section directory
6+9N    ...      Section data (concatenated)
```

### Section Directory Entry (9 bytes each)

```
Offset  Size  Field
------  ----  -----
0       1     Section type
1       4     Data offset (from start of file)
5       4     Data length (bytes)
```

## Section Types

| Type | Name           | Description                              |
|------|----------------|------------------------------------------|
| 0x01 | ANALYSIS       | Track metadata (bpm, key, duration)       |
| 0x02 | BEATS          | Beat timing data                          |
| 0x03 | PEAKS          | Editor peaks (window=256, ~187.5/s at 48kHz) |
| 0x08 | PEAKS_OVERVIEW | Overview peaks (window=8192, ~5.86/s)     |

Types 0x04-0x07 were previously used for intermediate mipmap levels but are now deprecated. Readers should silently ignore unknown section types.

All sections are optional. Readers should check the directory and handle missing sections gracefully.

## Section Formats

### 0x01 ANALYSIS (6 bytes)

```
Offset  Size  Field
------  ----  -----
0       2     Duration in seconds (uint16)
2       2     BPM × 10 (uint16, e.g., 1280 = 128.0 BPM)
4       1     Key (see encoding table)
5       1     Reserved (0x00)
```

**Key Encoding:**

```
0x00 = Unknown
0x01 = C major    0x0D = C minor
0x02 = C# major   0x0E = C# minor
0x03 = D major    0x0F = D minor
0x04 = D# major   0x10 = D# minor
0x05 = E major    0x11 = E minor
0x06 = F major    0x12 = F minor
0x07 = F# major   0x13 = F# minor
0x08 = G major    0x14 = G minor
0x09 = G# major   0x15 = G# minor
0x0A = A major    0x16 = A minor
0x0B = A# major   0x17 = A# minor
0x0C = B major    0x18 = B minor
```

Flat keys map to their sharp equivalents (e.g., Ab = G#).

### 0x02 BEATS

Beat timing stored as first beat position + intervals between consecutive beats.

```
Offset  Size   Field
------  -----  -----
0       2      First beat time in milliseconds (uint16)
2       2      Beat count (uint16)
4       N      Beat intervals in centiseconds (uint8[])
```

**Reconstructing beat times:**
```python
beats = [first_beat_ms / 1000.0]
for interval_cs in intervals:
    beats.append(beats[-1] + interval_cs / 100.0)
```

**Notes:**
- Intervals are centiseconds (10ms units), uint8 range 0-255 = 0-2550ms
- Supports tempos down to ~23 BPM (2550ms between beats)
- For N beats, there are N-1 intervals

### 0x03 PEAKS / 0x08 PEAKS_OVERVIEW

Waveform peak data at two resolutions. Both share the same encoding:

```
Offset  Size   Field
------  -----  -----
0       4      Peak count (uint32)
4       N×2    Peak data: [min: int8, max: int8][]
```

**Notes:**
- Values are signed bytes, -128 to +127, representing -1.0 to +1.0
- To convert: `float_value = int8_value / 127.0`
- Legacy files may use uint16 peak count header (auto-detected by readers)

**Resolutions (at 48kHz sample rate):**

| Section | Window | Peaks/sec | Use case                     |
|---------|--------|-----------|------------------------------|
| 0x03    | 256    | 187.5     | Editor waveform rendering    |
| 0x08    | 8192   | 5.86      | Track list thumbnails        |

The editor uses range aggregation on the high-resolution peaks (0x03), so it renders correctly at all zoom levels from a single peak set. The overview section (0x08) is used by track list thumbnails via HTTP range requests (~2KB fetch instead of the full file).

## Size Estimates

**Total file size (editor + overview peaks):**

| Track Length | Analysis + Beats | Peaks (2 levels) | Total   |
|--------------|------------------|-------------------|---------|
| 3 min        | ~370 B           | ~62 KB            | ~63 KB  |
| 10 min       | ~1.2 KB          | ~206 KB           | ~207 KB |
| 30 min       | ~3.6 KB          | ~618 KB           | ~622 KB |

**Range request sizes:**

| Section  | 3 min track | 30 min track |
|----------|-------------|--------------|
| Editor   | ~60 KB      | ~600 KB      |
| Overview | ~2 KB       | ~19 KB       |

Track list thumbnails only fetch the header (~100 bytes) + overview section (~2KB).

## Reading Example (Python)

```python
from pcav import read_pcav_file
from pathlib import Path

pcav = read_pcav_file(Path("analysis.pcav"))

print(f"BPM: {pcav.analysis.bpm}")
print(f"Beats: {len(pcav.beats.beat_times)}")
print(f"Peaks: {len(pcav.peaks.peaks)}")
```

## Reading Example (TypeScript)

```typescript
// Fetch just the header to find section offsets
const header = await fetch(url, { headers: { Range: 'bytes=0-100' } });
const dir = parsePcavDirectory(header);

// Fetch only the section you need
const peaksRange = `bytes=${dir.peaks.offset}-${dir.peaks.offset + dir.peaks.length}`;
const peaks = await fetch(url, { headers: { Range: peaksRange } });
```

## Version History

The header version byte has only ever been `1` (`PCAV_VERSION` in `src/pcav.py`). The layout change below shipped without a bump; readers handle both via auto-detection.

- **2026-02-05** (`d51fe6e`) — Initial specification: ANALYSIS, BEATS, PEAKS (10 peaks/s, uint16 count).
- **2026-02-11** (`6370f8a`) — PEAKS count widened to uint32; two peak levels, editor (0x03) + overview (0x08); 0x04-0x07 deprecated.
