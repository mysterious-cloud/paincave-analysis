# Pain Cave Analysis

Audio analysis pipeline for Pain Cave — extracts beat grids, tempo curves, intensity profiles, and structural events from DJ mixes for workout playback.

## Philosophy

- **Ship fast, no slop.** Analysis output must be accurate and reliable.
- **Brief code.** Fewer lines without sacrificing clarity.
- **Deterministic.** Same input produces same output.
- **Offline-first.** Everything runs locally, no cloud dependencies.
- **Zero warnings, zero errors.** Verify before committing.
- **Reuse, don't reinvent.** Check existing modules before building new ones.

## Tech Stack

- **Runtime:** Python 3.11+ via uv
- **Beat Detection:** madmom (RNN + Dynamic Bayesian Network)
- **Audio I/O:** SoundFile
- **Numerics:** NumPy, SciPy
- **Visualization:** Matplotlib (debugging only)

## Project Structure

```
src/           # Source modules (flat layout)
docs/          # Specs and standards
```

## How It Works

1. madmom RNN generates beat/downbeat activation functions
2. Dynamic Bayesian Network decodes beat positions (constrained 4/4)
3. Tempo curve extracted from beat intervals, smoothed per-bar
4. Intensity = 50% low-band energy (20-150 Hz) + 50% LUFS loudness
5. Structural events (build, drop, breakdown) via pattern recognition
6. Song boundaries via multi-indicator clustering
7. Output is a single analysis.json per track

## Environment Variables

- `PAINCAVE_TRACKS_DIR` — Default tracks directory. CLI args override.

## Conventions

- Flat `src/` layout, no packages or `__init__.py`
- `analysis.json` is the sole output contract with the main app
- All CLI commands use argparse
- Visualization is for debugging, not production

## Standards Documents

- [docs/analysis-output.md](docs/analysis-output.md) — analysis.json schema, field reference
- [docs/audio-architecture.md](docs/audio-architecture.md) — Two-track audio system, coaching architecture

## Running

```bash
uv run src/analyze.py input.m4a --output analysis.json
uv run src/analyze.py input.m4a --songs songs.json --output analysis.json
uv run src/peaks.py input.m4a --output peaks.json
uv run src/visualize.py analysis.json
```

## AI Assistant Guidelines

### Code Generation Rules
- **No slop.** Every generated line must be intentional.
- **Brief is better.** Fewer lines, same clarity.
- **Verify with `uv run`** before considering anything done.
- **No unnecessary abstractions** — this is a CLI tool.
- `analysis.json` schema is the contract — changes require coordination with the main app.
- Performance matters — analysis takes 2-3 min for a 30-min mix, don't make it slower.

### Before Committing Code
1. Zero errors when running
2. Code is formatted
3. Feature manually tested with a real audio file
4. Standards docs reviewed — add, update, or remove as needed

### Communication Style
- Be direct. Skip preamble.
- Propose solutions, don't ask permission for obvious fixes.
- When uncertain, ask one clear question.
