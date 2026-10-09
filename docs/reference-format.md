# Reference format

What `src/reference.py` (via `bin/analyze-reference`) writes for [Songlab](../../songlab): one `analysis.json` describing a reference song's bar grid. Verified against the source; if they disagree, the source wins. Unrelated to the Pain Cave track directory in [output-formats.md](output-formats.md), which this mode never touches.

```bash
bin/analyze-reference song.wav -o analysis.json [--min-bpm 60] [--max-bpm 200] [--first-downbeat SECONDS] \
    [--stages grid,stems,sections,peaks] [--cache-dir DIR] [--peaks PATH] [--progress] [--print-activity]
```

Decode, Beat This! detection and the same grid fit as the import path (see [Beat grid](output-formats.md#beat-grid): `analyze.fit_beat_grid`, kick alignment included), with tempo bounds that default to 60-200 BPM instead of 100-180. No library, hash, staging, PCAV, AAC, work zones, `metadata.json` or manifest. Later layers (stems, sections, chords) add fields to this file and hang on `grid.bars`.

## Example

```json
{
  "version": 1,
  "kind": "songlab-reference-analysis",
  "source": { "file": "song.wav", "durationSeconds": 40.0, "sampleRate": 44100 },
  "config": { "minBpm": 60, "maxBpm": 200, "firstDownbeat": null },
  "tempo": { "bpm": 120.0, "rangeBpm": [120.0, 120.0] },
  "meter": [4, 4],
  "grid": {
    "beats": [
      { "time": 0.499, "bar": 0, "beat": 3, "confidence": 1.0 },
      { "time": 0.999, "bar": 0, "beat": 4, "confidence": 1.0 },
      { "time": 1.499, "bar": 1, "beat": 1, "confidence": 1.0 }
    ],
    "firstDownbeatIndex": 2,
    "bars": [
      { "bar": 1, "start": 1.499, "end": 3.499 },
      { "bar": 2, "start": 3.499, "end": 5.499 }
    ],
    "residualMs": 20.82,
    "inferredBeatCount": 1
  }
}
```

The file is pretty-printed (2-space indent) with a trailing newline. Beats and bars are cut to three lines here.

## Fields

All times are seconds from the start of the decoded audio, rounded to milliseconds. Keys are camelCase.

| Field | Type | |
|-------|------|--|
| `version` | integer | Format version, `1`. Bumped on any breaking change; adding fields is not breaking |
| `kind` | string | Always `"songlab-reference-analysis"` |
| `source.file` | string | Input file's basename, no directory |
| `source.durationSeconds` | number | Decoded length |
| `source.sampleRate` | integer | Hz of the decode: native for wav/flac/ogg/aiff, 44100 for m4a/mp3/aac (ffmpeg) |
| `config.minBpm`, `config.maxBpm` | integer | Tempo bounds the grid was fitted within |
| `config.firstDownbeat` | number or null | The `--first-downbeat` given, as typed (not the snapped time); null when bar 1 was detected |
| `tempo.bpm` | number | 60 / the median grid interval, one decimal |
| `tempo.rangeBpm` | [number, number] | Lowest and highest tempo over 30 s windows, one decimal; is `[bpm, bpm]` when no window holds 8 beats |
| `meter` | [4, 4] | **Assumed, not detected.** The grid counts four beats to a bar, so a 3/4 or 6/8 song comes out as 4/4 bars that do not line up with its real ones |
| `grid.beats[]` | array | Every grid beat inside the audio, in time order |
| `grid.beats[].time` | number | Beat position |
| `grid.beats[].bar` | integer | Bar this beat belongs to; `0` for a pickup (see below) |
| `grid.beats[].beat` | integer | 1-4 within its bar |
| `grid.beats[].confidence` | number | 0-1. Where a detection landed on the beat: 1.0 for perfectly even surrounding intervals, falling as they stray from the tempo. Where the grid inferred the beat with no detection under it (quiet intro, breakdown): capped at 0.5, so treat its time as a guess |
| `grid.firstDownbeatIndex` | integer | Index into `grid.beats` of bar 1, beat 1; equals the number of pickup beats |
| `grid.bars[]` | array | Complete bars, numbered from 1 (below) |
| `grid.bars[].bar` | integer | 1-based bar number |
| `grid.bars[].start`, `.end` | number | Bar span in seconds; `start` is the bar's first beat time |
| `grid.residualMs` | number | RMS distance in ms between the grid and the detections it kept; the gate refuses above 60. Lower is a firmer grid |
| `grid.inferredBeatCount` | integer | Grid beats with no detection under them |

## Bar numbering

- **Bar 1 starts at the first downbeat.** By default that is the first grid beat on the voted downbeat phase (each detected downbeat votes for its position modulo 4, weighted by how close it lands). The grid covers the whole song, so an intro is numbered bars, not a run-up; only the 0-3 beats before bar 1 are pickups. With `--first-downbeat`, bar 1 is the grid beat nearest the given time and every beat before it, however many, is a pickup.
- **Pickups are bar 0.** Beats before bar 1 carry `"bar": 0` and their real position in the bar (`beat` 2, 3 or 4 for the one to three that precede the downbeat). Bar 0 is not in `grid.bars`; its beats are `grid.beats[0 .. firstDownbeatIndex - 1]`.
- **`grid.bars` holds complete bars only**, with no gaps and no overlap: bar `n` ends exactly where bar `n+1` starts. Bar `n` consists of `grid.beats[firstDownbeatIndex + 4(n-1) .. firstDownbeatIndex + 4n - 1]`. Beats after the last complete bar (up to three) stay in `grid.beats` with the number of the bar they would have started, but that bar is absent from `grid.bars`.
- **The last bar's `end` is not a downbeat** when no later beat exists inside the audio: it is the bar's fourth beat plus one beat interval, capped at `source.durationSeconds`. The last bar can be shorter than the others when the audio ends inside it.
- Tempo may drift (the grid follows slow drift), so bar lengths differ slightly; use `start` and `end`, not `bar * 4 * 60 / bpm`.

## Layers (`--stages`)

`grid` always runs and is everything above. The other stages are opt-in and only add fields, so a reader of the grid-only file reads these too.

- **`stems`**: Demucs `htdemucs` (MIT) splits the song into drums, bass, other and vocals on the 44.1 kHz stereo decode (MPS when available, else CPU; about 20 s and 90 s for a 4-minute song). The stems are cached as 16-bit FLAC in `--cache-dir` (default `cache/` beside `-o`), in a folder named by the input's SHA-256 and the model, so a re-run with other flags skips the split. Demucs is in the `reference` dependency group; `bin/analyze-reference` runs with it, Pain Cave's import never installs it.
- **`sections`**: boundaries and repetition labels from the audio alone (`src/sections.py`, numpy and scipy only). Bar-synchronous chroma (harmony), MFCC (timbre) and, with the stems stage, stem levels each give a bar-by-bar self-similarity matrix; a checkerboard kernel (4 and 8 bars) along their mean scores the change at each bar start. Peaks become boundaries, weighted toward 4-bar phrases (the phrase phase is whichever bar position carries the most change), at least 2 bars apart; a trailing fade below -40 dBFS joins the last section. Segments that match bar for bar share a letter. Letters are repetition groups, not names.
- **`peaks`**: waveform peaks for drawing, written to `--peaks` (default `peaks.json` beside `-o`) before `analysis.json`, compact JSON: `{ "peaksPerSecond": 50, "mix": [[min, max], ...], "stems": { "drums": [[min, max], ...], ... } }`, three decimals. `stems` is `{}` without the stems stage.

Any stage beyond `grid` also adds `loudness` to every bar.

| Field | Type | |
|-------|------|--|
| `grid.bars[].loudness` | number | Mix level over the bar, dBFS RMS, one decimal; `-120` for silence |
| `grid.bars[].stems` | object | `{drums, bass, other, vocals}`: each stem's RMS over the bar relative to that stem's loudest bar, linear 0-1, three decimals |
| `stems.model` | string | `"htdemucs"` |
| `stems.files` | object | Stem name to its FLAC, relative to the cache directory |
| `stems.peakDb` | object | Stem name to its loudest bar in dBFS RMS. A bar's absolute level is `peakDb + 20·log10(level)` |
| `stems.relDb`, `stems.floorDb` | number | The suggested activity rule: a stem plays in a bar when it is within `relDb` (24) dB of its own loudest bar and above `floorDb` (-50) dBFS. Readers may apply their own |
| `grid.bars[].novelty` | number | With `sections`: how much the song changes at this bar's start, 0-1 (bar 1 is 0) |
| `sections.methods` | object | Method name to its result; `ssm` from this stage, others (SongFormer, All-In-One) may be added by separate tools |
| `sections.methods.<m>.tool`, `.version` | string, integer | Which method and its revision |
| `sections.methods.<m>.sections[]` | array | `{startBar, endBar, label, confidence}`: contiguous, covering bar 1 to the last bar; `endBar` is exclusive (the next section's `startBar`); `label` is a letter A-Z by first appearance; `confidence` is the boundary's phrase-weighted novelty (1 for the first) |
| `sections.methods.<m>.boundariesSeconds` | number[] | The start time of every section after the first |

`--progress` prints `PROGRESS {"stage": "grid", "fraction": 0.0}` lines on stdout, one JSON object per line, for each stage from 0 to 1; everything else on stdout is free text. `--print-activity` prints one row per bar with the playing stems and the loudness, for checking by ear.

The stems are measured, not heard: bleed between stems is common (htdemucs puts pads and guitars in `other`), so `other` reads as playing almost everywhere in a dense mix.

`bin/score-sections analysis.json truth.json` scores every method against confirmed sections (`{"sections": [{startBar, endBar, label}]}`, exported from Songlab's Lab): boundary F exactly and within one bar, and pairwise label F over bars.

## Refusals

Same idea as the import path: the CLI exits 1, prints one line to stderr and writes nothing, not even a partial `-o` file. An `-o` file from an earlier run is left as it was. The file is otherwise written whole (temporary file, then rename), creating `-o`'s parent directory only once the analysis has succeeded.

```text
Error: refused <file>: <reason>; nothing written
```

Refused as `refused`: audio under 1 s; digital silence; fewer than 8 beat detections; a detection outside the audio; a beat spacing, fitted tempo or interval outside `--min-bpm`/`--max-bpm` after octave correction; fewer than 8 detections fitting one constant tempo; detections scattering more than 60 ms RMS around the grid; no complete bar after the first downbeat. The reasons are those of [Refused input](output-formats.md#refused-input). A tempo step or a song in changing meter is likely to land here or in a high `residualMs`.

Other failures are `Error: <file>: <reason>; nothing written`, also exit 1: a file that cannot be decoded, a `--first-downbeat` outside the audio. A missing input is `Error: file not found: <path>`, and an output path equal to the input is refused. Bad flags (`--min-bpm` not below `--max-bpm`) exit 2 with argparse's usage message.

## Determinism

The same input file and flags give identical bytes: no timestamps, no randomness (the detector runs on a fixed thread count, as in the import path; Demucs runs without random shifts). Stem levels come from the cached FLAC, so they repeat exactly once the cache exists; a fresh split on another device (MPS vs CPU) can differ in the last digits. `config` records the flags so a consumer can key a cache on file plus config. The detector checkpoint is pinned in `src/models.py`; changing it changes grids, and will bump `version` or add a field naming it.
