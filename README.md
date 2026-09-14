# paincave-analysis

**Status: active.**

The analysis step of the Pain Cave track pipeline. It takes one audio file and writes a track directory: content hash, `original.wav`, beats/BPM/key and waveform peaks (`analysis.pcav`), work zones, `metadata.json`, and a streaming `audio.m4a`.

Pipeline (`../paincave/bin/import.sh`): paincave-audio (effects) → **paincave-analysis** → paincave-thumbnail.

## Setup

Needs Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), and `ffmpeg` + `ffprobe` on `PATH`.

```bash
uv sync
```

Metadata validation reads `../paincave/shared/schemas/metadata.schema.json`, so the `paincave` repo must be checked out as a sibling.

## Usage

```bash
bin/import track.wav --title "Dark Descent" --genres "dark-techno,industrial"
# same as:
uv run python src/import_track.py track.wav --title "Dark Descent" --genres "dark-techno,industrial"
```

| Flag | Default | |
|------|---------|--|
| `audio_file` | required | Input audio |
| `--library` | `$PAINCAVE_TRACKS_DIR`, else `../paincave-tracks` | Track library directory |
| `-t, --title` | filename stem | |
| `-a, --artist` | `Pain Cave` | |
| `--source` | `SUNO` | Only choice: `SUNO` |
| `--source-id` | SUNO id from the file's comment tag | |
| `-g, --genres` | none | Comma-separated genre slugs |
| `--min-bpm` / `--max-bpm` | `100` / `180` | Beat-tracking tempo range |
| `-f, --force` | off | Re-import an existing hash. **Deletes the track dir first**, see below |

No `.env` is loaded. Export `PAINCAVE_TRACKS_DIR` or pass `--library`. The last stdout line is `Hash: <hash>`, which `../paincave/bin/import.sh` parses.

Debug CLIs: `uv run python src/analyze.py <file> [-o out.json]` dumps the full analysis JSON, and `uv run python src/content_hash.py <file> [--full]` prints the hash.

## Docs

- [docs/output-formats.md](docs/output-formats.md): track directory, `metadata.json`, work zones, hash algorithm
- [docs/pcav-format.md](docs/pcav-format.md): `analysis.pcav` binary spec
- Architecture and plans live in the main repo: [../paincave/docs/architecture.md](../paincave/docs/architecture.md), [../paincave/docs/plan.md](../paincave/docs/plan.md)

## Known issues

- **`--force` deletes the whole track directory** (`shutil.rmtree`), including `thumb*.webp` and `plans/` that other tools put there.
- **The hash does not match the server's.** `src/content_hash.py` and `../paincave/server/src/services/audioService.ts` decode differently, so the same file gets different hashes. The Python hash also changes with input format and sample rate. See [output-formats.md](docs/output-formats.md#content-hash).
- A failed import leaves a partial track directory. Re-running without `--force` then skips it and exits 0.
- `mutagen` is declared in `pyproject.toml` but not imported anywhere.
