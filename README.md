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
| `--library` | `$PAINCAVE_TRACKS_DIR`, else `../paincave-tracks/library` | Track library directory |
| `-t, --title` | source filename stem | |
| `-a, --artist` | `Pain Cave` | |
| `--source` | `SUNO` | `SUNO`, or `CLASS` for a song paincave Studio cut from a recorded class |
| `--source-id` | SUNO id from the file's comment tag | |
| `-g, --genres` | none | Comma-separated genre slugs |
| `--min-bpm` / `--max-bpm` | `100` / `180` | Tempo range the fitted grid must fall in |
| `-f, --force` | off | Stage a candidate revision under `{hash}/analyses/`; preserve active files |
| `--replace` | off | With `--force`: swap the new analysis in as the active files; thumbnails, plans and candidates are kept |
| `--first-downbeat` | detected | Bar 1 time, snapped to a grid beat |
| `--bpm` | fitted | With `--first-downbeat` (required): a constant grid at this BPM with bar 1 on that time, instead of the fitted grid; no tempo bounds |
| `--source-file` / `--processing-record` | input / none | Preserve and verify source lineage for a derived master |
| `--check-model` | off | Print the detector checkpoint (downloading it if needed) and exit |

Audio under 1 s, digital silence, or without a beat grid (too few detections, a tempo outside the bounds, or detections that do not fit one constant tempo; a `--bpm` grid is refused for none of these) is refused: exit 1, `Error: refused <file>: <reason>; nothing written` on stderr. See [output-formats.md](docs/output-formats.md#refused-input).

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

## Workout download (research)

Downloads the original audio and the site's captions of a LES MILLS On Demand workout you can play, as research input for Studio's class breakdown. Research content stays on the laptop: it lands in the research workspace's inbox, `../paincave-tracks/research/inbox/` (git-ignored, never exported), where Studio lists it at `paincave-studio.localhost/research`; never in the main library, a commit or an export. Needs Google Chrome and FFmpeg (with ffprobe).

```bash
bin/download-workout 'https://my.lesmillsondemand.com/workout/THE%20TRIP%20%2343%2040%20min'
# same as:
uv run --group workout python src/download_workout.py 'URL'
```

A dedicated Chrome window opens on its own profile (`~/.paincave-analysis/workout-chrome`). Sign in there on the first run; the login is kept for later runs. Click Play if playback doesn't start. The window closes once the HLS master playlist has been captured, and the download continues in the terminal until it prints `Done`. Don't run two copies on the same profile at once.

| Flag | Default | |
|------|---------|--|
| `url` | required | `https://my.lesmillsondemand.com/workout/...`, quoted |
| `-o, --output` | `../paincave-tracks/research/inbox` | Parent of the workout folders; an existing folder is never overwritten |
| `--language` | `en` | Caption language; exits 1 if the workout has none in it (captions are never generated) |
| `--profile` | `~/.paincave-analysis/workout-chrome` | Dedicated Chrome profile; never your everyday one |
| `--timeout` | `300` | Seconds for sign-in and finding the playlist |
| `--har` | | Read the playlist from a Chrome HAR export (DevTools → Network, with response bodies, only this workout) instead of opening Chrome |

Each workout folder (`THE_TRIP_43_40_min/`) holds `<name>.m4a` (the original audio stream, not re-encoded), `<name>.en.vtt` / `.srt` / `.txt` (merged captions, original timestamps), `subtitle-segments/` (source WebVTT) and `metadata.json` (source URL, retrieval time, ffprobe output, caption counts; no playback tokens). The audio must match the playlist duration within 2 s and decode without errors. All of it is written to a temporary folder and moved into place only when complete.

It supports the unprotected HLS (Mux) delivery seen so far: DRM-protected or live streams are refused, and site changes may need updates. Keep the profile and any HAR private; both hold session data.

## Class beats (Studio's cut editor)

Every beat and downbeat of a long recording, for Studio's cut editor (paincave `docs/studio.md`, phase 2b): the Beat This! detector as the import runs it (`small0`), over the whole recording, with no grid fit and no tempo bounds, so a class's songs and their tempo changes all come through. The cut editor fits its own grid per song.

```bash
bin/class-beats THE_TRIP_43.m4a -o class.beats.json   # about 30 s for a 45-minute class on an M1 Max
```

ffmpeg decodes straight to the detector's 22.05 kHz mono, so a long class stays small in memory. Output: `{ "checkpoint": { "name", "sha256" }, "beats": [...], "downbeats": [...] }`, seconds to the millisecond, written whole or not at all; exit 1 with the reason for anything ffmpeg can't decode.

## Class transcript (subtitles for a class without them)

For Studio's *Make subtitles* (paincave `docs/studio.md`, phase 2b): the coach's voice split out with Demucs `htdemucs_ft` (its vocals model alone, which is the bag's vocals: one-hot weights; in 5-minute chunks, so a 45-minute class never holds four stems in memory at once: about 3 GB in all, not 7), transcribed by whisper.cpp `large-v3` with every token timed by DTW, then cut into caption-sized lines (`lines_of`: at sentence ends, pauses over 0.8 s, a comma past 8 words, at most 14 words or 6 s).

```bash
bin/class-transcript THE_TRIP_43.m4a -o THE_TRIP_43.whisper.vtt   # 5 min for THE TRIP 43's 44 on an M1 Max (Demucs 2, whisper 3); 6½ for a 45½-minute class
```

Setup, once: `brew install whisper-cpp` and `ggml-large-v3.bin` in `~/.local/share/whisper/` (or pass `--model`), from Hugging Face `ggerganov/whisper.cpp`. Needs the `reference` group (Demucs).

whisper.cpp's flags were measured on THE TRIP 43 and a 45-minute Apple Fitness class: `-mc 0` (no text carried between 30-second windows: with it, "Attack. The intensity is all." looped over the Apple class's last 30 minutes); no voice activity detection (it leaves token times on its speech-only timeline); `-dtw large.v3` with `-nfa` (whisper's own token times put a segment's first word at the silence before it, "Hey" at 0:00 for 0:16.8; DTW needs flash attention off).

Progress goes to stdout as `PROGRESS {"stage": "voice" | "words", "fraction"}` lines; the VTT starts with a `NOTE` saying how it was made and is written whole or not at all; exit 1 with the reason (no model, not audio, whisper failed). Against THE TRIP 43's real subtitles: 99% of the coach's 591 lines heard, 92% of their words, lines as long as theirs (median 1.7 s, 4 words); 38 of 548 lines aren't in them (lyrics, and calls the subtitles leave out). Singing lands in the vocals stem, so songs with more of it bring more lyrics through.

## Remove voice (a class song without the coach)

For Studio's *Without the coach* (paincave `docs/studio.md`, phase 2b): Demucs `htdemucs_ft`'s stereo vocals subtracted from a song only inside the given spans (where the coach talks), with 100 ms fades; outside them the output is the input sample for sample. The vocals are resampled back to the song's own rate (ffmpeg), and the result keeps its rate and channels, written as 16-bit PCM whole or not at all.

```bash
bin/remove-voice "Class · 02.wav" -o "Class · 02.nocoach.wav" --spans spans.json   # [[start, end], …] in seconds; 13 s for a 3-minute song on an M1 Max
```

Progress as `PROGRESS` lines; exit 1 with the reason (bad spans, not audio). On THE TRIP 43's song 02, whisper.cpp heard 97% of the coach's words before and 5% after. Singing under a span goes too (the vocals stem holds both). Needs the `reference` group (Demucs).

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
- The grid follows **slow drift but not a step change in tempo**: a spliced tempo change bends the grid, and only a large step is refused (by the tempo bounds). The residual is measured against the detections the grid kept, so it does not count the off-beats dropped before fitting. The manifest's `grid.residualMs`, `grid.inferredBeatCount` and `grid.tempoRangeBpm` are the evidence. The manual levers are `--first-downbeat` (bar 1 on the fitted grid) and `--bpm` with it (a constant grid in place of the fit, for a song the fit refuses or misplaces; `grid.source: "manual"`). Kick alignment moves only the beats with a kick under them (`grid.kickAlignedCount`, `grid.kickLagMs`, `grid.kickResidualMs`), so through an intro or a breakdown the grid still rides on the detector's timing.
