# Audio Architecture

Protected audio streaming for Pain Cave workouts.

## Local Development

```
User → Vite dev server → /tracks/{hash}/audio.m4a
                              ↓
                    Served from public/
```

- Audio files stored in `tracks/{hash}/`
- Vite serves static files directly during development
- No auth required locally (add when needed)

## Production

```
User (with JWT) → Cloudflare Worker → R2 Bucket
                        ↓
                  API /api/audio/verify
```

- Audio files stored in Cloudflare R2 (zero egress cost)
- Cloudflare Worker validates access via API endpoint
- `audio.mysterious.cloud` → Worker route

## Folder Structure

```
tracks/
  {content-hash}/           # SHA256 hash of audio
    audio.m4a               # The audio file
    analysis.json           # Beat grid, kick zones, raw data
    peaks/                  # Waveform data for visualization
      data.json             # Peaks metadata
      *.dat                 # Binary peak files
```

- **Content hash** uniquely identifies each track
- **Analysis** contains beat grid, tempo, kick zones
- **Peaks** used by waveform component

## File Formats

### analysis.json

```json
{
  "version": "2.0",
  "duration": 180.5,
  "sample_rate": 44100,
  "bpm": 128,
  "first_downbeat": 0.33,

  "beats": [
    { "t": 0.33, "bar": 1, "beat": 1 },
    { "t": 0.75, "bar": 1, "beat": 2 }
  ],

  "kick_zones": [
    { "type": "kicks", "start_bar": 1, "end_bar": 16, "start_t": 0.33, "end_t": 30.0, "avg_density": 0.85 }
  ],

  "raw": {
    "beat_times": [0.33, 0.75, 1.17, ...],
    "bass_energy": [0.2, 0.8, 0.9, ...]
  }
}
```

### Database Schema (tracks table)

```typescript
{
  hash: string          // Primary key, content hash
  title: string
  artist: string | null
  bpm: number | null
  duration: number
  firstDownbeat: number | null
  kickZones: KickZone[] | null
  styleTags: string[] | null
  plans: Plan[] | null  // Workout variations (JSON)
  // ... SUNO metadata fields
}
```

## Audio Analysis

Advanced analysis for tracks using madmom, essentia, and pyloudnorm. Extracts:
- Beat grid with downbeats and bar positions
- Time-varying tempo curve
- Kick zones (high bass energy sections)
- Raw beat times for UI alignment

### Setup

Requires [uv](https://github.com/astral-sh/uv) and [ffmpeg](https://ffmpeg.org/):

```bash
# Install uv (macOS)
brew install uv

# Install ffmpeg (for m4a/mp3 support)
brew install ffmpeg
```

### Running Analysis

```bash
cd analysis

# Basic analysis
uv run src/analyze.py ../tracks/{hash}/audio.m4a \
  --output ../tracks/{hash}/analysis.json

# With custom BPM range (prevents half-time confusion)
uv run src/analyze.py audio.m4a --min-bpm 120 --max-bpm 150
```

First run creates a virtual environment and installs dependencies (~2-3 min).

### Field Reference

| Field | Description |
|-------|-------------|
| `beats` | Every beat with timestamp, bar number, beat position (1-4) |
| `bpm` | Global BPM of the track |
| `first_downbeat` | Time of first bar 1, beat 1 |
| `kick_zones` | Sections with strong kick presence |
| `raw.beat_times` | Raw beat timestamps for UI snapping |
| `raw.bass_energy` | Per-beat bass energy values |

## Authoring Workflow

1. Generate songs in SUNO
2. Blend in Ableton → export as M4A
3. Create folder: `tracks/{hash}/`
4. Copy audio as `audio.m4a`
5. Run analysis:
   ```bash
   cd analysis
   uv run src/analyze.py ../tracks/{hash}/audio.m4a \
     --output ../tracks/{hash}/analysis.json
   ```
6. Generate peaks for waveform visualization
7. Add track to database via API or migration script
8. Test in Track Editor

## API Endpoints

### GET /api/tracks
List all tracks with metadata.

### GET /api/tracks/:hash
Get single track by content hash.

### PUT /api/tracks/:hash
Update track metadata (title, artist, styleTags).

### PUT /api/tracks/:hash/plans
Update plans for a track (Zod-validated).

## Production Migration

1. Create R2 bucket
2. Upload audio to R2
3. Deploy Cloudflare Worker (validates via `/api/audio/verify`)
4. Point `audio.mysterious.cloud` → Worker
5. Deploy API to fly.io
