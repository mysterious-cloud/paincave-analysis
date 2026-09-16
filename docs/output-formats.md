# Output Formats

What `src/import_track.py` (via `bin/import` or `../paincave/bin/analyze.sh`) writes. Verified against the source; if they disagree, the source wins.

## Track directory and promotion

New imports are built in a unique `.staging-*` directory inside the requested library. Metadata/analysis validation, PCAV round-trip, AAC probing, loudness measurement and checksum collection must all succeed before one same-filesystem directory rename exposes the track. Exceptions clean only that run's staging directory. SIGTERM (e.g. the server's analyze timeout) exits with status 143 after the same cleanup; the handler fires once, and ffmpeg children are killed while unwinding. Signal delivery waits for a running C call (up to ~5.5 s measured on a 29-minute track), so callers should allow a grace period before SIGKILL. A live import holds an `flock` on its staging directory; SIGKILL or power loss leaves the directory behind unlocked, and the next import into that library removes every `.staging-<hash>-<8 chars>` directory in the library root that is unlocked, owned by the user, not a symlink and more than 60 s old. Track folders and candidates are never scanned. A pre-existing partial track is reported as incomplete, not successful.

### Refused input

Input under 1 second or digital silence is refused straight after the input is decoded, before identity checks or staging. Audio the beat grid cannot be built from is refused after detection on the staged master, before any artifact is written; the stage is removed. That covers fewer than 8 detected beats, a beat spacing that stays outside `--min-bpm`/`--max-bpm` after octave correction, fewer than 8 detections fitting one constant tempo, and detections scattering more than 60 ms RMS around the fitted grid. The CLI exits 1 with one stderr line, `Error: refused <file>: <reason>; nothing written`; `import_song` raises `analyze.NoBeatGrid`. Unlike madmom, Beat This! reports no beats at all for a steady 440 Hz tone (1) or white noise (0) over 10 s, so non-music is now refused rather than given an invented grid.

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

Beat, peak and work-zone analysis read the retained `original.wav`. A `.wav` input is decoded once, from a single read: the staged `original.wav` must match that read's SHA-256 (`Input changed during import` otherwise), so those samples serve the identity, the analysis and the peaks. Other inputs are decoded for the identity, re-decoded from the staged copy to prove it, and the master is decoded for analysis. The beat detector reads those samples straight from memory (it resamples to 22.05 kHz itself), so nothing decodes the master a second time; the AAC/loudness encode is started before analysis and collected before the manifest. Each stage prints `timing: <stage> <seconds>s` to stderr. Measured 2026-09-15 (M1 Max, 10 cores): 2:39 track 4.1 s, 3:22 track 4.5 s, 3:44 track 6.7 s wall, including loading the detector. The legacy hash comes from the preparation input; SUNO tags and default title come from the retained unprocessed source. `--source-file` requires a matching `--processing-record` with source/output SHA-256 checksums for derived masters. Reanalysis carries existing source lineage forward.

## Beat grid

Beats come from [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, MIT), checkpoint `small0`, run on CPU without the DBN. It is a detector: times land on its 20 ms frames and it reports nothing through quiet intros and breakdowns. Plans are bar ranges over an unbroken grid, so `analyze.fit_beat_grid` turns the detections into one:

1. The initial period is the median detection interval, keeping only intervals within half to double the `--min-bpm`/`--max-bpm` range, then halved or doubled until its tempo is inside that range.
2. Detections are numbered by stepping from one to the next (`round(interval / period)`, so a gap counts its beats and two detections closer than half a period share a number), fitted with least squares to `t = t0 + k · period`, and refitted up to three times after dropping detections more than 20% of a period off.
3. The grid is every `t0 + n · period` inside the audio, so it reaches back before the first detection and on past the last one.
4. Bar 1 is the first grid beat carrying the voted downbeat phase (`n mod 4`, each detected downbeat weighted by how close it lands) at or after the first beat a detection was heard on. Earlier grid beats are pickups in bar 0. `--first-downbeat` overrides the vote, snapped to the nearest grid beat.

`bpm` is `60 / period`, one decimal, for the whole track. A beat's `confidence` is 1.0 where a detection landed within `max(70 ms, 20% of a period)` of it and 0.5 where the grid inferred it, so a consumer can tell heard beats from filled-in ones.

**One tempo per track.** The grid is constant by construction. Tracks that drift stay within the residual gate but their grid walks away from the music: of the three shipped tracks, "The Rush - Ragga Edit" drifts from 169.0 to 170.4 BPM and gets a 32.6 ms residual with 245 of 632 beats inferred, while the other two sit at 18–23 ms with 38–46 inferred. Per-section tempo is not implemented.

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
| `source` | `"SUNO"` | `--source` (only allowed value) |
| `sourceId` | string | `--source-id`, else SUNO `id=` from the source's ffprobe `comment` tag, else the active track's `sourceId` (candidates), else `""` |
| `genres` | string[] | `--genres` split on commas, else the active track's genres (candidates), else `[]` |
| `bpm` | number | `60 /` the fitted constant grid period, 1 decimal |
| `durationSeconds` | integer | Rounded track duration |
| `key` | string \| null | Always `null`: key detection was removed with madmom |
| `createdAt` | date-time | SUNO `created=` from the source's comment tag, else source file mtime (UTC) |
| `updatedAt` | date-time | Import time (UTC) |

## manifest.json

Version 1 records source/master SHA-256 and original filename, full legacy MD5 fingerprint, exact sample count/rate/channels/subtype, playback probe properties, source processing record, detector config, peak windows/filter, AAC bitrate, dependency/ffmpeg versions, tool-source digests and per-artifact sizes/checksums. Integrated input LUFS, true peak and loudness range come from ffmpeg's `ebur128` summary (0.1 precision; `config.loudnessFilter` records the filter) and are measured without applying normalization; non-finite measurements are null. `normalizationApplied` is false. Manifests written before 2026-09-15 used `loudnorm`, whose loudness range differs by up to 0.7 LU on the current tracks (integrated and true peak within 0.05).

The grid records beat count/coverage, first-downbeat index, meter, `reviewed: false`, the detected beat/downbeat counts, how many grid beats were inferred, the fitted bar phase, the RMS residual in milliseconds, and `meanConfidence` under `confidenceKind: "grid-heard"` (the share of grid beats a detection was heard on, not model or physiological confidence; manifests written before 2026-09-15 recorded `meanIntervalConfidence` under `confidenceKind: "interval-regularity"`). `config.beatCheckpoint` records the detector checkpoint name and the SHA-256 of the cached file, and `tools` its `beat_this` and `torch` versions. Every imported track has at least 8 beats and a positive BPM (see [Refused input](#refused-input)); folders written by older versions can still hold BPM 0. Work-zone energy covers complete bars and the final partial second. `--first-downbeat` explicitly overrides alignment by snapping to a grid beat.

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
