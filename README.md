# paincave-analysis

**Status: active.**

The analysis step of the Pain Cave track pipeline. It takes one audio file and writes a track directory: content hash, `original.wav`, a beat grid with its BPM and waveform peaks (`analysis.pcav`), work zones, `metadata.json`, and a streaming `audio.m4a`.

Pipeline (`../paincave/bin/import.sh`): paincave-audio (effects) → **paincave-analysis** → paincave-thumbnail.

## Setup

Needs Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), and `ffmpeg` + `ffprobe` on `PATH`.

```bash
uv sync
```

Beats come from [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, MIT), which pulls in CPU `torch` (~1 GB of wheels). Its `small0` checkpoint is fetched by name into the torch hub cache the first time a track is analyzed; `uv run python src/import_track.py --check-model` downloads it up front and prints its path and SHA-256, which every `manifest.json` records.

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
| `-t, --title` | source filename stem | |
| `-a, --artist` | `Pain Cave` | |
| `--source` | `SUNO` | `SUNO`, or `CLASS` for a song paincave Studio cut from a recorded class |
| `--source-id` | SUNO id from the file's comment tag | |
| `-g, --genres` | none | Comma-separated genre slugs |
| `--min-bpm` / `--max-bpm` | `100` / `180` | Tempo range the fitted grid must fall in |
| `-f, --force` | off | Stage a candidate revision under `{hash}/analyses/`; preserve active files |
| `--replace` | off | With `--force`: swap the new analysis in as the active files; thumbnails, plans and candidates are kept |
| `--first-downbeat` | detected | Bar 1 time, snapped to a grid beat |
| `--source-file` / `--processing-record` | input / none | Preserve and verify source lineage for a derived master |
| `--check-model` | off | Print the detector checkpoint (downloading it if needed) and exit |

Audio under 1 s, digital silence, or without a beat grid (too few detections, a tempo outside the bounds, or detections that do not fit one constant tempo) is refused: exit 1, `Error: refused <file>: <reason>; nothing written` on stderr. See [output-formats.md](docs/output-formats.md#refused-input).

The detector reports beats and downbeats; a grid is fitted on top of them so bars run unbroken through quiet intros and breakdowns. The grid is locally linear — each beat sits on a line fitted over the detections within 8 bars of it — so it follows slow tempo drift, and off-beat detections are dropped before fitting. A last pass puts the bars on the kicks: every beat the detector heard slides onto the strongest bass onset within 60 ms of it, provided that onset reaches the track's 90th percentile, and the grid is fitted once more over those times. Bar 1 is the first grid beat on the voted downbeat phase, so an intro is plannable bars and at most three beats are pickups; see [Beat grid](docs/output-formats.md#beat-grid) for the fit and its limits. Musical key is no longer detected: `key` is `null` everywhere, and PCAV stores 0x00 (unknown).

With `--force`, omitted `--title`, `--artist` and `--genres` (and `--source-id` when the source has no SUNO id) keep the active track's `metadata.json` values.

`bin/import` loads no `.env`: export `PAINCAVE_TRACKS_DIR` or pass `--library`. `../paincave/bin/analyze.sh` and `import.sh` first export paincave's `.env`, `.env.local`, `.env.development` and `.env.development.local`; `import.sh` also always passes `--library`. The last stdout line is `Hash: <hash>`, which `../paincave/bin/import.sh` parses along with `Track already exists:` and `Revision: <dir> (candidate; ...)`.

Debug CLIs: `uv run python src/analyze.py <file> [-o out.json]` dumps the full analysis JSON, and `uv run python src/content_hash.py <file> [--full]` prints the hash.

## Reference mode (Songlab)

Songlab's reference-song analysis: one audio file in, one `analysis.json` out, with the song's tempo and bar grid, and on request (`--stages stems,sections,allin1,melody,peaks`) which Demucs stems play in each bar, section boundaries with repetition labels, All-In-One's named sections (on Apple's GPU), the sung notes of the vocal stem (CREPE), and waveform peaks. It has nothing to do with the Pain Cave library: no hash, staging, PCAV, AAC, work zones or `metadata.json`.

```bash
bin/analyze-reference song.wav -o analysis.json
# same as:
uv run python src/reference.py song.wav -o analysis.json
```

| Flag | Default | |
|------|---------|--|
| `audio_file` | required | Input audio (wav, flac, m4a, mp3, ...) |
| `-o, --output` | required | Where to write `analysis.json` |
| `--min-bpm` / `--max-bpm` | `60` / `200` | Tempo range the fitted grid must fall in (the Pain Cave import keeps `100` / `180`) |
| `--first-downbeat` | detected | Bar 1 time, snapped to a grid beat |

It runs the import path's Beat This! detection and `fit_beat_grid`, so the grid has the same strengths and limits (slow drift is followed, a tempo step is not; see [Beat grid](docs/output-formats.md#beat-grid)). The bars are numbered from the first downbeat, with pickups as bar 0, and meter is assumed 4/4. Refused input exits 1 with `Error: refused <file>: <reason>; nothing written` and writes no file. Fields, units and bar numbering: [docs/reference-format.md](docs/reference-format.md).

## Docs

- [docs/output-formats.md](docs/output-formats.md): track directory, `metadata.json`, work zones, hash algorithm
- [docs/pcav-format.md](docs/pcav-format.md): `analysis.pcav` binary spec
- [docs/reference-format.md](docs/reference-format.md): `analysis.json` of reference mode (Songlab)
- Architecture and plans live in the main repo: [../paincave/docs/architecture.md](../paincave/docs/architecture.md), [../paincave/docs/plan.md](../paincave/docs/plan.md)

## Verification

```bash
uv run pytest -q   # or: uv run python -m unittest discover -s tests
uv run python tests/export_pcav_fixtures.py ../paincave/tests/fixtures/pcav
```

PCAV v2 preserves phase, exact duration and waveform timebase. Source/master checksums, processing config, measurements, tool versions and the detector checkpoint live in `manifest.json`. See the output contract before consuming candidate revisions.

## Remaining boundaries

- Reanalysis creates a candidate revision unless `--force --replace` is given, which makes it the active analysis (prints `Replaced: <dir> (...)`; `../paincave/bin/import.sh` must treat that like `Revision:` and skip thumbnails). Publication/review of candidates in the app is not implemented. Existing root files stay active. Candidates are write-once by convention only (writable folders), and each run stores full copies of `source.*`, `original.wav` and `audio.m4a` with no dedupe. SIGTERM removes the run's `.staging-*` folder; a SIGKILLed run's folder stays until the next import into that library sweeps it.
- **The hash does not match the server's.** `src/content_hash.py` and `../paincave/server/src/services/audioService.ts` decode differently, so the same file gets different hashes. The Python hash also changes with input format and sample rate. See [output-formats.md](docs/output-formats.md#content-hash).
- New imports are validated and staged before promotion. Partial directories fail explicitly; `--force` can stage a recovery candidate only while identity still verifies (legacy: `original.wav` present and hashing to the folder name; with `manifest.json`: input byte-identical to the recorded master or matching its fingerprint, and the retained source present with its recorded checksum), and it never repairs the active files.
- The grid follows **slow drift but not a step change in tempo**: a spliced tempo change bends the grid, and only a large step is refused (by the tempo bounds). The residual is measured against the detections the grid kept, so it does not count the off-beats dropped before fitting. The manifest's `grid.residualMs`, `grid.inferredBeatCount` and `grid.tempoRangeBpm` are the evidence, and `--first-downbeat` is the only manual lever. Kick alignment moves only the beats with a kick under them (`grid.kickAlignedCount`, `grid.kickLagMs`, `grid.kickResidualMs`), so through an intro or a breakdown the grid still rides on the detector's timing.
