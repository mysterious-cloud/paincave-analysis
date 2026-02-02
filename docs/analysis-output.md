# analysis.json Output Contract

The `analysis.json` file is the primary output of this project and the contract between the Python analysis pipeline and the TypeScript consumer apps (UI, Studio, server).

## Source of Truth

**The canonical schema lives in the main paincave repo:**

```
../paincave/shared/schemas/analysis.schema.json
```

This JSON Schema defines every field, its type, and whether it's required. Both the TypeScript interface (`AnalysisJson` in `shared/repositories/types.ts`) and this pipeline's output must conform to it.

**If you change the output shape of `analyze.py`, the schema must be updated first in the main repo.**

## Current Version: 3.0

All field names are **camelCase** for direct TypeScript consumption.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Schema version (`"3.0"`) |
| `duration` | number | Track duration in seconds |
| `sampleRate` | integer | Audio sample rate in Hz (44100) |
| `bpm` | number | Global BPM (median of tempo curve) |
| `firstDownbeat` | number | Time in seconds of bar 1, beat 1 |
| `beats` | array | Beat grid — see below |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `hash` | string | Content hash of the audio file |
| `key` | string \| null | Musical key (e.g. `"C major"`) |
| `workZones` | array | Bar ranges classified as work/breakdown |
| `raw` | object | Raw signal data for visualization |

### beats[]

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `t` | number | yes | Time in seconds |
| `bar` | integer | yes | Bar number (1-based) |
| `beat` | integer | yes | Beat within bar (1-based) |
| `confidence` | number | no | Detection confidence (0–1) |

### workZones[]

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | no | `"work"` or `"breakdown"` |
| `startBar` | integer | yes | First bar (1-based) |
| `endBar` | integer | yes | Last bar (1-based, exclusive) |
| `startSec` | number | yes | Start time in seconds |
| `endSec` | number | yes | End time in seconds |
| `avgDensity` | number | no | Average kick density (0–1) |
| `hint` | string | no | Human-readable zone description |

### raw

| Field | Type | Description |
|-------|------|-------------|
| `beatTimes` | number[] | All detected beat times in seconds |
| `bassEnergy` | number[] | Bass band energy (20–150 Hz) per second |

## Validation

Use the JSON Schema to validate output before writing:

```python
import json
import jsonschema
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "paincave" / "shared" / "schemas" / "analysis.schema.json"

def validate_analysis(data: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(data, schema)
```

## Peaks Output

Peaks are a separate concern from analysis. Written to `peaks/` subdirectory with four resolutions:

| File | samples/peak | Use |
|------|-------------|-----|
| `overview.json` | 16384 | Zoomed-out waveform |
| `medium.json` | 4096 | Medium zoom |
| `detail.json` | 1024 | Zoomed-in waveform |
| `tiny.json` | ~N bars | Track list thumbnails (normalized 0–1 amplitude) |

Each peaks file (except tiny) has the shape:

```json
{
  "sampleRate": 44100,
  "duration": 1800.5,
  "samplesPerPeak": 4096,
  "peaks": [[min, max], [min, max], ...]
}
```

`tiny.json` has a different shape:

```json
{
  "version": 1,
  "bars": [0.0, 0.45, 0.92, ...]
}
```

## Track Directory Structure

When the pipeline writes to `../paincave-tracks/`, each track gets:

```
{content_hash}/
  original.wav        # Source audio (input file)
  audio.m4a           # Streaming-optimized AAC
  analysis.json       # This contract
  metadata.md         # Track metadata (title, artist, SUNO info)
  peaks/
    overview.json
    medium.json
    detail.json
    tiny.json
```
