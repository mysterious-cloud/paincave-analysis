# Pain Cave Analysis

Audio analysis pipeline for Pain Cave — extracts beat grids, tempo, work zones from tracks for workout playback.

## Philosophy

- **Ship fast, no slop.** Analysis output must be accurate and reliable.
- **Brief code.** Fewer lines without sacrificing clarity.
- **Deterministic.** Same input produces same output.
- **Offline-first.** Everything runs locally, no cloud dependencies.

## Tech Stack

- **Runtime:** Python 3.11+ via uv
- **Beat Detection:** madmom (RNN + Dynamic Bayesian Network)
- **Audio I/O:** SoundFile, ffmpeg
- **Numerics:** NumPy, SciPy
- **Validation:** jsonschema

## Project Structure

```
src/           # Source modules
bin/           # CLI wrapper (bin/import)
docs/          # Specs and standards
```

## Output Structure

```
{hash}/
├── original.wav         # Archive (pristine audio)
├── audio.m4a            # Streaming format
├── metadata.json        # Track metadata
├── analysis.pcav        # Binary: beats + peaks
├── plan_workzones.json  # Auto-detected work zones
├── thumb_64.webp        # Thumbnails (added by paincave-thumbnail)
├── thumb_128.webp
└── thumb_256.webp
```

**Key separation:**
- `analysis.pcav` — objective audio analysis (beats, peaks)
- `plan_*.json` — workout planning data (can have multiple plans per track)
- `thumb_*.webp` — visual assets (added separately)

## How It Works

1. madmom RNN generates beat/downbeat activation functions
2. Dynamic Bayesian Network decodes beat positions (constrained 4/4)
3. Work zones detected from bass energy (20-150 Hz)
4. Musical key detected via madmom CNN
5. Beats + peaks written to `analysis.pcav` (binary)
6. Work zones written to `plan_workzones.json`
7. M4A converted for streaming

## Environment Variables

- `PAINCAVE_TRACKS_DIR` — Default tracks directory. CLI `--library` overrides.

## CLI Reference

### bin/import

Import a song into the track library.

```bash
bin/import track.wav --title "Track Name" --artist "Artist"
bin/import track.m4a --genres "dark-techno,industrial"
bin/import track.mp3 --min-bpm 120 --max-bpm 140 --force
```

| Flag | Default | Description |
|------|---------|-------------|
| `audio_file` | (required) | Path to audio file |
| `--library` | `$PAINCAVE_TRACKS_DIR` | Track library directory |
| `-t, --title` | filename | Track title |
| `-a, --artist` | "Pain Cave" | Artist name |
| `--source` | "SUNO" | Audio source platform |
| `--source-id` | (auto-detect) | ID from source platform |
| `-g, --genres` | (none) | Comma-separated genre slugs |
| `--min-bpm` | 100 | Minimum BPM for beat tracking |
| `--max-bpm` | 180 | Maximum BPM for beat tracking |
| `-f, --force` | false | Overwrite existing track |

## Source Modules

| File | Purpose |
|------|---------|
| `src/import_track.py` | Main entry point — orchestrates pipeline |
| `src/analyze.py` | Beat detection, tempo, work zones, key |
| `src/pcav.py` | PCAV binary format read/write |
| `src/content_hash.py` | Audio content hashing |

## Standards Documents

- [docs/output-formats.md](docs/output-formats.md) — Complete output format reference
- [docs/pcav-format.md](docs/pcav-format.md) — PCAV binary format specification

## Ecosystem

This repo is one step in a multi-project pipeline:
- **paincave-analysis** — This repo. Analyzes audio, outputs .pcav + JSON
- **paincave-thumbnail** — Generates AI thumbnails (thumb_*.webp)
- **paincave/studio** — Extracts assets for deployment, manages track library

Key reference: [`../paincave/shared/schemas/`](../paincave/shared/schemas/) contains JSON schemas.

## AI Assistant Guidelines

### Code Generation Rules
- **No slop.** Every generated line must be intentional.
- **Brief is better.** Fewer lines, same clarity.
- **Verify with `uv run`** before considering anything done.
- Performance matters — analysis takes 2-3 min for a 30-min mix.

### Before Committing Code
1. Zero errors when running
2. Feature manually tested with real audio
3. Docs updated if needed
