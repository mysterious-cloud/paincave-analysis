# Output Formats

What `src/import_track.py` (via `bin/import` or `../paincave/bin/analyze.sh`) writes. Verified against the source; if they disagree, the source wins.

> **WARNING — `--force` deletes the whole track directory.**
> If `{library}/{hash}/` already exists, `--force` runs `shutil.rmtree` on it before re-importing. Everything in that directory is lost, including files this repo did not create: `thumb*.webp` (paincave-thumbnail) and `plans/`. Back up or regenerate them afterwards.

## Track directory

```
{library}/{hash}/
├── original.wav          # written here
├── analysis.pcav         # written here
├── plan_workzones.json   # written here
├── metadata.json         # written here
├── audio.m4a             # written here
├── thumb*.webp           # NOT written here — paincave-thumbnail
└── plans/                # NOT written here — Studio-era leftover
```

- `{library}` = `--library`, else `$PAINCAVE_TRACKS_DIR`, else `../paincave-tracks` relative to this repo. `bin/import` and `src/import_track.py` load no `.env`. `../paincave/bin/analyze.sh` and `import.sh` first export paincave's `.env`, `.env.local`, `.env.development` and `.env.development.local`, so a `PAINCAVE_TRACKS_DIR` set there applies; `import.sh` also always passes `--library`.
- `{hash}` = 12-char content hash of the input file (see [Content hash](#content-hash)).
- Files are written in the order listed above. A failure part-way leaves a partial directory, and re-running without `--force` prints "Track already exists" and exits 0.
- The paincave app database is the source of truth for plans and metadata edits. These files are the import output, not a live record.

## original.wav

- `.wav` input: byte copy (`shutil.copy2`); sample rate, bit depth and channels are unchanged.
- Any other input: `ffmpeg -i <input> -ar 44100` (ffmpeg's default WAV encoding, channel count preserved).

Beat, key, peak and work-zone analysis read `original.wav`. The hash and the SUNO comment tags come from the input file.

## audio.m4a

`ffmpeg -i original.wav -c:a aac -b:a 256k`.

## analysis.pcav

Binary: ANALYSIS (duration, BPM, key), BEATS (madmom beat times), editor peaks (0x03) and overview peaks (0x08). Peaks are computed from 300 Hz low-pass filtered audio, normalized to full scale. Full spec: [pcav-format.md](pcav-format.md).

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

Validated against `../paincave/shared/schemas/metadata.schema.json` (`additionalProperties: false`). A validation failure only prints `WARNING: metadata validation failed` and the file is still written. A missing schema file raises and aborts the import after `plan_workzones.json` has been written.

| Field | Type | Value written |
|-------|------|---------------|
| `hash` | string `^[a-f0-9]{12}$` | Content hash (= directory name) |
| `title` | string, min 1 | `--title`, else input filename stem |
| `artist` | string | `--artist`, else `"Pain Cave"` |
| `source` | `"SUNO"` | `--source` (only allowed value) |
| `sourceId` | string | `--source-id`, else SUNO `id=` from the ffprobe `comment` tag, else `""` |
| `genres` | string[] | `--genres` split on commas, else `[]` |
| `bpm` | number | Median of octave-corrected local tempo from beat intervals, 1 decimal |
| `durationSeconds` | integer | Rounded track duration |
| `key` | string \| null | madmom label (e.g. `"D# minor"`, `"Bb major"`), `null` if detection fails |
| `createdAt` | date-time | SUNO `created=` from the comment tag, else input file mtime (UTC) |
| `updatedAt` | date-time | Import time (UTC) |

## Content hash

Implemented in `src/content_hash.py`, computed from the **input file** (not `original.wav`):

1. Decode to samples:
   - `.m4a`, `.mp3`, `.aac`: `ffmpeg -ar 44100 -ac 1` to a temp WAV, then read with soundfile.
   - Anything else (e.g. `.wav`): read directly with soundfile at its native sample rate.
2. Downmix to mono with `np.mean` across channels (if still multichannel).
3. Cast to float32 and MD5 the raw bytes.
4. Keep the first 12 hex chars.

**It does not give the same hash across formats.** The same master as a 48 kHz WAV, a 44.1 kHz WAV and an M4A produces three different hashes. So does re-importing an `original.wav` that was converted from a non-WAV input.

**The paincave server hashes differently.** `../paincave/server/src/services/audioService.ts` (`computeContentHash`) takes MD5 over `ffmpeg -f s16le -ac 1 -ar 22050` output. For the same file it will not match the hash computed here.
