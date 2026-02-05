# Output Formats

This document describes all output files produced by `bin/import`.

## Output Structure

```
{hash}/
├── original.wav         # Archive audio (pristine)
├── audio.m4a            # Streaming audio (AAC 256kbps)
├── metadata.json        # Track metadata
├── analysis.pcav        # Binary analysis data (beats + peaks)
├── plan_workzones.json  # Auto-detected work zones
├── thumb_64.webp        # Thumbnail 64×64 (added by paincave-thumbnail)
├── thumb_128.webp       # Thumbnail 128×128 (added by paincave-thumbnail)
└── thumb_256.webp       # Thumbnail 256×256 (added by paincave-thumbnail)
```

The `{hash}` directory name is a 12-character content hash derived from the audio samples.

---

## metadata.json

Track metadata in JSON format.

```json
{
  "hash": "98f755fef882",
  "title": "The Strength Within",
  "artist": "Pain Cave",
  "source": "SUNO",
  "sourceId": "23ebb5df-8fe3-4d50-8abb-e4a29ffabb53",
  "genres": ["motivational", "epic"],
  "bpm": 120.0,
  "durationSeconds": 192,
  "key": "Ab major",
  "createdAt": "2026-01-22T20:49:50Z",
  "updatedAt": "2026-02-05T16:10:51Z"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `hash` | string | 12-char content hash (directory name) |
| `title` | string | Track title |
| `artist` | string | Artist name |
| `source` | string | Source platform (e.g., "SUNO") |
| `sourceId` | string | ID from source platform |
| `genres` | string[] | Genre slugs |
| `bpm` | number | Beats per minute |
| `durationSeconds` | integer | Track duration in seconds |
| `key` | string \| null | Musical key (e.g., "Ab major", "E minor") |
| `createdAt` | string | ISO 8601 creation timestamp |
| `updatedAt` | string | ISO 8601 last update timestamp |

### Notes

- `createdAt` uses the source platform's creation time if available, otherwise file modification time
- `key` uses flat notation from analysis but can be either sharps or flats
- Schema: `../paincave/shared/schemas/metadata.schema.json`

---

## analysis.pcav

Binary file containing beats and waveform peaks. See [pcav-format.md](pcav-format.md) for full specification.

### Quick Reference

| Section | Content | Size (3 min track) |
|---------|---------|-------------------|
| ANALYSIS | duration, BPM, key | 6 bytes |
| BEATS | beat times (delta-encoded) | ~360 bytes |
| PEAKS | waveform min/max at 10/sec | ~3.6 KB |

### Reading in Python

```python
from pcav import read_pcav_file

pcav = read_pcav_file("analysis.pcav")

print(f"BPM: {pcav.analysis.bpm}")
print(f"Key: {pcav.analysis.key}")
print(f"Duration: {pcav.analysis.duration_seconds}s")

# Beat times in seconds
for t in pcav.beats.beat_times[:10]:
    print(f"Beat at {t:.2f}s")

# Peaks as (min, max) int8 pairs
for min_val, max_val in pcav.peaks.peaks[:10]:
    print(f"Peak: {min_val} to {max_val}")
```

### Reading in TypeScript

```typescript
// Fetch and parse
const buffer = await fetch("analysis.pcav").then(r => r.arrayBuffer());
const view = new DataView(buffer);

// Verify magic "PCAV"
const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
if (magic !== "PCAV") throw new Error("Invalid PCAV file");

const version = view.getUint8(4);
const sectionCount = view.getUint8(5);

// Parse section directory (9 bytes each, starting at offset 6)
for (let i = 0; i < sectionCount; i++) {
  const dirOffset = 6 + i * 9;
  const type = view.getUint8(dirOffset);
  const dataOffset = view.getUint32(dirOffset + 1, true);
  const dataLength = view.getUint32(dirOffset + 5, true);

  // Read section data from dataOffset
}
```

---

## plan_workzones.json

Auto-detected work zones (high-energy sections) for workout planning.

```json
{
  "version": 1,
  "zones": [
    {
      "startBar": 20,
      "durationBars": 7,
      "startSeconds": 40.01,
      "durationSeconds": 13.99
    },
    {
      "startBar": 28,
      "durationBars": 7,
      "startSeconds": 56.0,
      "durationSeconds": 13.99
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | integer | Format version (currently 1) |
| `zones` | array | List of work zones |

### Zone Fields

| Field | Type | Description |
|-------|------|-------------|
| `startBar` | integer | Starting bar number (1-indexed) |
| `durationBars` | integer | Duration in bars |
| `startSeconds` | number | Start time in seconds |
| `durationSeconds` | number | Duration in seconds |

### Notes

- Work zones are detected from bass energy (20-150 Hz kick density)
- Minimum zone length is 4 bars
- These are suggestions for workout intervals, not definitive boundaries
- The `plan_` prefix indicates this is planning data (future: multiple plans per track)

---

## Audio Files

### original.wav

- Pristine archive copy
- 44.1 kHz sample rate
- 16-bit PCM
- Not modified after import

### audio.m4a

- Streaming format
- AAC codec at 256 kbps
- Suitable for web/mobile playback

---

## Thumbnails (External)

Added by `paincave-thumbnail`, not by this pipeline.

| File | Size | Format |
|------|------|--------|
| `thumb_64.webp` | 64×64 px | WebP |
| `thumb_128.webp` | 128×128 px | WebP |
| `thumb_256.webp` | 256×256 px | WebP |

---

## Content Hash

The directory name is a 12-character hash derived from raw audio samples:

1. Decode audio to mono PCM
2. Convert to float32
3. Compute MD5
4. Truncate to 12 characters

This ensures:
- Same audio content → same hash (regardless of format/metadata)
- Different audio → different hash
- Filesystem-friendly names
