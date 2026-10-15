# Output Formats

What `src/import_track.py` (via `bin/import` or `../paincave/bin/analyze.sh`) writes. Verified against the source; if they disagree, the source wins.

## Track directory and promotion

New imports are built in a unique `.staging-*` directory inside the requested library. Metadata/analysis validation, PCAV round-trip, AAC probing, loudness measurement and checksum collection must all succeed before one same-filesystem directory rename exposes the track. Exceptions clean only that run's staging directory. SIGTERM (e.g. the server's analyze timeout) exits with status 143 after the same cleanup; the handler fires once, and ffmpeg children are killed while unwinding. Signal delivery waits for a running C call (up to ~5.5 s measured on a 29-minute track), so callers should allow a grace period before SIGKILL. A live import holds an `flock` on its staging directory; SIGKILL or power loss leaves the directory behind unlocked, and the next import into that library removes every `.staging-<hash>-<8 chars>` directory in the library root that is unlocked, owned by the user, not a symlink and more than 60 s old. Track folders and candidates are never scanned. A pre-existing partial track is reported as incomplete, not successful.

### Refused input

Input under 1 second or digital silence is refused straight after the input is decoded, before identity checks or staging. Audio the beat grid cannot be built from is refused after detection on the staged master, before any artifact is written; the stage is removed. That covers a detection before 0 or more than one detector frame past the end, fewer than 8 detected beats (a detection landing exactly on the duration is dropped, not refused: a track may end on its last beat), a beat spacing that stays outside `--min-bpm`/`--max-bpm` after octave correction, fewer than 8 detections fitting one constant tempo, and detections scattering more than 60 ms RMS around the fitted grid. The CLI exits 1 with one stderr line, `Error: refused <file>: <reason>; nothing written`; `import_song` raises `analyze.NoBeatGrid`. Unlike madmom, Beat This! reports no beats at all for a steady 440 Hz tone (1) or white noise (0) over 10 s, so non-music is now refused rather than given an invented grid.

```text
{library}/{hash}/
├── source.<extension>    # exact source download bytes
├── original.wav          # analyzed master; may be a processed derivative
├── analysis.pcav         # v2 delivery data
├── analysis.json         # exact duration, numbered beats/confidence, machine suggestions
├── plan_workzones.json   # machine suggestions in legacy import shape
├── metadata.json         # validated catalog import seed
├── audio.m4a             # AAC playback rendition
├── manifest.json         # source/master lineage, config, versions, measurements, checksums
├── analyses/<revision>/  # subsequent candidate analysis runs, same artifact layout
├── thumb*.webp           # managed separately
└── plans/                # legacy authored files; never touched by this tool
```

`--force` creates a complete candidate under `analyses/<manifest-sha256>/`. `--force --replace` instead swaps the finished stage in as the active folder: files analysis does not produce (`thumb*.avif`/`.webp`, `thumbnail.json`, `plans/`, `analyses/`, `.thumbnail.lock`) move into the stage, two renames swap the folders, the old analysis is deleted last, and a failed swap restores the old folder and its kept files. `metadata.json` keeps the existing `createdAt`, and title/artist/genres/sourceId when not given. The CLI prints `Replaced: <dir> (active analysis updated; ...)` instead of `Revision:`. Without `--replace`, `--force` **does not replace active files**, metadata edits, thumbnails, plans, or the original source. Each candidate holds full copies of `source.<extension>`, `original.wav` and `audio.m4a` (no dedupe; `updatedAt` makes every revision id new) and is write-once by convention only. Reviewing/promoting a grid together with affected plan revisions is a separate future authoring operation. Audio anywhere inside a track folder `{library}/{hash}/` (its `original.wav`, a candidate's `analyses/<revision>/original.wav`, a renamed copy, any name case), with or without `--force`, either resolves to that directory's identity or is refused with nothing written. With `manifest.json` (identity holds even for a compressed download), the recorded fingerprint must match the directory and the file must be byte-identical to the recorded master or carry that fingerprint (e.g. the retained `source.<extension>`, only when no processing preset was applied); anything else, such as `audio.m4a`, fails with `Retained master checksum mismatch`. Without one (legacy tracks), the file must hash to the directory name (a legacy import from a compressed download does not: its folder name came from the compressed decode), otherwise the import fails with `Unverifiable legacy identity`. So audio inside a track folder never creates a second active track; a copy moved outside the library is an ordinary input. Library subfolders not named by a 12-hex track key, such as the server's upload staging folder `{library}/import/`, hold ordinary inputs. Without `--force`, a complete matching track prints `Track already exists:`. New full fingerprints are checked on duplicates; unverifiable legacy collisions fail safely.

`{library}` is `--library`, then `$PAINCAVE_TRACKS_DIR`, then the sibling `paincave-tracks`. The main wrappers load their environment files; this Python CLI does not. `Hash: <hash>` remains the final CLI stdout line. Existing tracks and candidate reanalysis skip artwork generation in the full wrapper.

SQL owns authored edits. These files seed a new catalog entry; this tool does not insert/update SQL. Existing active artifacts are not silently upgraded to v2. Legacy phase recovery in the main reader is described in [pcav-format.md](pcav-format.md).

## original.wav

- `.wav` input: byte copy (`shutil.copy2`); sample rate, bit depth and channels are unchanged.
- Other input: ffmpeg decodes the first audio stream to float32 PCM WAV at its native rate/channels. The exact compressed source is retained separately.

Beat, peak and work-zone analysis read the retained `original.wav`. A `.wav` input is decoded once, from a single read: the staged `original.wav` must match that read's SHA-256 (`Input changed during import` otherwise), so those samples serve the identity, the analysis and the peaks. Other inputs are decoded for the identity, re-decoded from the staged copy to prove it, and the master is decoded for analysis. The beat detector reads those samples straight from memory (it resamples to 22.05 kHz itself), so nothing decodes the master a second time; the AAC/loudness encode is started before analysis and collected before the manifest. Each stage prints `timing: <stage> <seconds>s` to stderr. Measured 2026-09-16 (M1 Max, 10 cores): 2:39 track 4.1 s, 3:22 track 4.4 s, 3:44 track 6.7 s wall, including loading the detector; the onset strength kick alignment needs costs about 0.2 s of that. The legacy hash comes from the preparation input; SUNO tags and default title come from the retained unprocessed source. `--source-file` requires a matching `--processing-record` with source/output SHA-256 checksums for derived masters. Reanalysis carries existing source lineage forward.

## Beat grid

Beats come from [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, MIT), checkpoint `small0`, run on CPU without the DBN. It is a detector: times land on its 20 ms frames and it reports nothing through quiet intros and breakdowns. Plans are bar ranges over an unbroken grid, so `analyze.fit_beat_grid` turns the detections into one:

1. The initial period is the median detection interval, keeping only intervals within half to double the `--min-bpm`/`--max-bpm` range, then halved or doubled until its tempo is inside that range.
2. Off-beats go first: walking the detections, one is kept only when it lands within a quarter beat of a whole number of beats after the last one kept. In some sections the detector hears half-beats, and numbering those as beats of their own tilts every fit that follows. A detection more than two beats past the last kept one is across a silence, where the starting period is too rough to judge, and re-anchors the walk.
3. The remaining detections are numbered by stepping from one to the next (`round(interval / period)`, so a gap counts its beats and two detections closer than half a period share a number), fitted with least squares to `t = t0 + k · period`, renumbered with that fitted period and refitted (an initial period a percent out loses a beat when the numbering steps across a long silence), then refitted up to three times more after dropping detections more than 20% of a period off the grid they would get. This global line does two jobs: octave correction, and a beat number for every surviving detection.
4. Each grid beat `n` is placed on its own least-squares line, fitted over the detections numbered within 32 beats (8 bars) of it, widening to 64 and then 128 when fewer than 8 fall in the window or when the window's tempo lands outside the bounds, and falling back to the global line when even that is too thin. Past the ends of the detections the window stays anchored on the last of them. Grid indices run from before the first beat inside the audio to past the last, then are trimmed to `0 <= t < duration`.
5. That grid is then used to renumber: every detection is re-assigned to the grid beat it landed on, those more than a quarter beat away are dropped, and the grid is rebuilt once from the survivors.
6. **The bars are pulled onto the kicks.** The detector times a beat a frame or two from its kick, and by section: on "The Strength Within" its detections ran 25 ms late through 60-90 s and 38 ms early through 150-210 s. `analyze.onset_strength` turns the 20-150 Hz bass the work zones already need into one rising-energy value per millisecond (resampled to 1 kHz, high-passed at 40 Hz so a sub-bass drone cannot blur the attack, then the positive first difference of the analytic envelope). Each grid beat with a detection on it takes the strongest onset within 60 ms; an onset reaching the track's 90th percentile is that beat's kick, and the detection time moves onto it. A beat without one — a breakdown, a beat the bass sits out — keeps the time the detector gave. The grid is then fitted once more over those times, same local lines, same bounds gate.
7. The grid must be strictly increasing with every interval inside the tempo bounds (1% of slack); otherwise the track is refused.
8. **Bar 1 is the first grid beat carrying the voted downbeat phase** (`n mod 4`, each detected downbeat weighted by how close it lands). The grid covers the whole track, so an intro is plannable bars rather than a run-up, and only the 0-3 grid beats before bar 1 are pickups in bar 0. `--first-downbeat` overrides the vote, snapping to the nearest grid beat and numbering from there; every beat before it, however many, is bar 0.

`bpm` is `60 /` the median grid interval, one decimal; `grid.tempoRangeBpm` in the manifest gives the lowest and highest tempo over 30 s windows of at least two bars. A beat's `confidence` is 1.0 where a detection landed within `max(70 ms, 20% of the reference period)` of it and 0.5 where the grid inferred it, so a consumer can tell heard beats from filled-in ones.

`grid.kickAlignedCount` counts the beats moved onto a kick, `grid.kickLagMs` is the median kick-minus-grid lag over them before the pass (how far off the kicks the detector's grid sat: +12 ms on "The Rush", -25 ms on "The Strength Within") and `grid.kickResidualMs` the RMS lag afterwards, which stays in the tens of milliseconds because it counts every accepted onset, including the scattered ones a bass-light section throws up. Measured (2026-09-16) over 30 s windows of the shipped tracks, "The Rush" went from +8..+16 ms off its kicks to -3..+10 and "The Strength Within" from -39..+14 to -12..+20: the sections its detector had drifted in, -25 ms at 60-90 s and -38 ms at 150-210 s, now sit at -3 and -6 ms, while its one bass-light window (90-120 s, a kick under 28 of 60 beats) went the other way, +14 to +20 ms. "Unbreakable" hardly moved at all, -48..+22 to -46..+24: its worst windows are its intro and a breakdown, where almost no beat has a kick under it. That is the limit of the pass — where the beats carry no kick the grid keeps the detector's timing, and a lag measured over a handful of bass stabs says little either way.

**Slow drift is followed; a step change is not.** Because each beat sits on its own local line, a track that speeds up gradually keeps its grid: of the three shipped tracks, "The Rush - Ragga Edit" runs 169.0 to 170.4 BPM and fits with a 12.2 ms residual and 7 of 633 beats inferred. A tempo change that happens at a point (an edit, a spliced section) is not modelled: the grid bends through it, and a large enough step is refused because some grid interval then falls outside the tempo bounds. The residual is measured against the detections the grid kept, so it reports how well the grid explains the beats it claims, not how much the detector heard in total; `inferredBeatCount` is the other half of that picture. It rose by a few milliseconds when kick alignment came in (10.2 to 12.2 ms on "The Rush"): the grid now follows the kicks, and the detections are what it has moved away from.

## audio.m4a

AAC at 256 kbps, first audio stream, `+faststart`. The same ffmpeg process measures loudness (`ebur128=peak=true`) from its decode of the master. Probe must find a positive audio duration within 50 ms of the analyzed master. The source and master are not loudness-normalized. Browser edit-list/alignment checks are recorded in the main architecture document.

## analysis.pcav

PCAV v2: exact duration, BPM/key, absolute beat times with first-downbeat index/meter, editor peaks (0x03) and overview peaks (0x08) with explicit rate/window. Peaks are computed from 300 Hz low-pass filtered audio, normalized to full scale. Full spec: [pcav-format.md](pcav-format.md).

## plan_workzones.json

```json
{
  "version": 1,
  "zones": [
    { "startBar": 21, "durationBars": 23, "startSeconds": 28.73, "durationSeconds": 32.66 }
  ]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `startBar` | integer | 1-indexed bar where the zone starts |
| `durationBars` | integer | Zone length in bars (minimum 4) |
| `startSeconds` | number | Start time, 2 decimals |
| `durationSeconds` | number | Length, 2 decimals; a zone reaching the final bar runs to track end |

Zones are runs of bars whose mean 20–150 Hz energy, normalized to the track maximum, is ≥ 0.45. Only "work" runs are kept; breakdowns are dropped. The `plan_` prefix is historical. The file is not a plan.

## metadata.json

Validated against `../paincave/shared/schemas/metadata.schema.json` (`additionalProperties: false`). Invalid metadata or analysis, non-finite values, missing schemas and invalid RFC 3339 timestamps abort before promotion. Date-time checks remain mandatory without optional jsonschema format extras. The shared analysis JSON Schema matches the actual machine work-zone shape; it is separate from editable UI work zones.

| Field | Type | Value written |
|-------|------|---------------|
| `hash` | string `^[a-f0-9]{12}$` | Content hash (= directory name) |
| `title` | string, min 1 | `--title`, else the active track's title (candidates), else source filename stem |
| `artist` | string | `--artist`, else the active track's artist (candidates), else `"Pain Cave"` |
| `source` | `"SUNO"` \| `"CLASS"` | `--source` (default `SUNO`; `CLASS`: a song paincave Studio cut from a recorded class) |
| `sourceId` | string | `--source-id`, else SUNO `id=` from the source's ffprobe `comment` tag, else the active track's `sourceId` (candidates), else `""` |
| `genres` | string[] | `--genres` split on commas, else the active track's genres (candidates), else `[]` |
| `bpm` | number | `60 /` the median interval of the fitted grid, 1 decimal |
| `durationSeconds` | integer | Rounded track duration |
| `key` | string \| null | Always `null`: key detection was removed with madmom |
| `createdAt` | date-time | SUNO `created=` from the source's comment tag, else source file mtime (UTC) |
| `updatedAt` | date-time | Import time (UTC) |

## manifest.json

Version 1 records source/master SHA-256 and original filename, full legacy MD5 fingerprint, exact sample count/rate/channels/subtype, playback probe properties, source processing record, detector config, peak windows/filter, AAC bitrate, dependency/ffmpeg versions, tool-source digests and per-artifact sizes/checksums. Integrated input LUFS, true peak and loudness range come from ffmpeg's `ebur128` summary (0.1 precision; `config.loudnessFilter` records the filter) and are measured without applying normalization; non-finite measurements are null. `normalizationApplied` is false. Manifests written before 2026-09-15 used `loudnorm`, whose loudness range differs by up to 0.7 LU on the current tracks (integrated and true peak within 0.05).

The grid records beat count/coverage, first-downbeat index, meter, `reviewed: false`, the detected beat/downbeat counts, how many grid beats were inferred, the fitted bar phase, the RMS residual in milliseconds, `tempoRangeBpm` (lowest and highest 30 s window tempo), the kick-alignment counters `kickAlignedCount`/`kickLagMs`/`kickResidualMs` (see [Beat grid](#beat-grid)), and `meanConfidence` under `confidenceKind: "grid-heard"` (the mean of the per-beat confidences, so it runs between 0.5 when every beat was inferred and 1.0 when every one was heard, and is not model or physiological confidence; the plain heard share is `1 - inferredBeatCount / beatCount`; manifests written before 2026-09-15 recorded `meanIntervalConfidence` under `confidenceKind: "interval-regularity"`). `config.beatCheckpoint` records the detector checkpoint name and the SHA-256 of the cached file, `config.beatThreads` the fixed torch thread count the detector ran with (the host's core count would make the reduction order, and so the detections, differ between machines), and `tools` its `beat_this` and `torch` versions. Every imported track has at least 8 beats and a positive BPM (see [Refused input](#refused-input)); folders written by older versions can still hold BPM 0. Work-zone energy covers complete bars and the final partial second. `--first-downbeat` explicitly overrides alignment by snapping to a grid beat.

Versioned candidates and filesystem rename protect ordinary processing failures. They are not a power-loss durability or backup guarantee. No automatic revision publication, source deletion, cloud upload, paid API or SQL sync happens here.

## Content hash

Implemented in `src/content_hash.py`, computed from the **input file** (not `original.wav`):

1. Decode to samples:
   - `.m4a`, `.mp3`, `.aac`: `ffmpeg -ar 44100 -ac 1` to a temp WAV, then read with soundfile.
   - Anything else (e.g. `.wav`): read directly with soundfile at its native sample rate.
2. Downmix to mono with `np.mean` across channels (if still multichannel).
3. Cast to float32 and MD5 the raw bytes.
4. Keep the first 12 hex chars.

**It does not give the same hash across formats.** The same master as a 48 kHz WAV, a 44.1 kHz WAV and an M4A produces three different hashes. So does importing a copy, outside its track folder, of an `original.wav` that was converted from a non-WAV input.

**The paincave server hashes differently.** `../paincave/server/src/services/audioService.ts` (`computeContentHash`) takes MD5 over `ffmpeg -f s16le -ac 1 -ar 22050` output. For the same file it will not match the hash computed here.
