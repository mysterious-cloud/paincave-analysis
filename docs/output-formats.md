# Output Formats

What `src/import_track.py` (via `bin/import` or `../paincave/bin/analyze.sh`) writes. Verified against the source; if they disagree, the source wins.

## Track directory and promotion

New imports are built in a unique `.staging-*` directory inside the requested library. Metadata/analysis validation, PCAV round-trip, AAC probing, loudness measurement and checksum collection must all succeed before one same-filesystem directory rename exposes the track. Exceptions clean only that run's staging directory; a killed process (SIGKILL, power loss) leaves it behind and nothing removes it. A pre-existing partial track is reported as incomplete, not successful.

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

`--force` now creates a complete candidate under `analyses/<manifest-sha256>/`. It **does not replace active files**, metadata edits, thumbnails, plans, or the original source. Each candidate holds full copies of `source.<extension>`, `original.wav` and `audio.m4a` (no dedupe; `updatedAt` makes every revision id new) and is write-once by convention only. Reviewing/promoting a grid together with affected plan revisions is a separate future authoring operation. Audio anywhere inside a track folder `{library}/{hash}/` (its `original.wav`, a candidate's `analyses/<revision>/original.wav`, a renamed copy, any name case), with or without `--force`, either resolves to that directory's identity or is refused with nothing written. With `manifest.json` (identity holds even for a compressed download), the recorded fingerprint must match the directory and the file must be byte-identical to the recorded master or carry that fingerprint (e.g. the retained `source.<extension>`, only when no processing preset was applied); anything else, such as `audio.m4a`, fails with `Retained master checksum mismatch`. Without one (legacy tracks), the file must hash to the directory name (a legacy import from a compressed download does not: its folder name came from the compressed decode), otherwise the import fails with `Unverifiable legacy identity`. So audio inside a track folder never creates a second active track; a copy moved outside the library is an ordinary input. Library subfolders not named by a 12-hex track key, such as the server's upload staging folder `{library}/import/`, hold ordinary inputs. Without `--force`, a complete matching track prints `Track already exists:`. New full fingerprints are checked on duplicates; unverifiable legacy collisions fail safely.

`{library}` is `--library`, then `$PAINCAVE_TRACKS_DIR`, then the sibling `paincave-tracks`. The main wrappers load their environment files; this Python CLI does not. `Hash: <hash>` remains the final CLI stdout line. Existing tracks and candidate reanalysis skip artwork generation in the full wrapper.

SQL owns authored edits. These files seed a new catalog entry; this tool does not insert/update SQL. Existing active artifacts are not silently upgraded to v2. Legacy phase recovery in the main reader is described in [pcav-format.md](pcav-format.md).

## original.wav

- `.wav` input: byte copy (`shutil.copy2`); sample rate, bit depth and channels are unchanged.
- Other input: ffmpeg decodes the first audio stream to float32 PCM WAV at its native rate/channels. The exact compressed source is retained separately.

Beat, key, peak and work-zone analysis read the retained `original.wav`. The legacy hash comes from the preparation input; SUNO tags and default title come from the retained unprocessed source. `--source-file` requires a matching `--processing-record` with source/output SHA-256 checksums for derived masters. Reanalysis carries existing source lineage forward.

## audio.m4a

AAC at 256 kbps, first audio stream, `+faststart`. Probe must find a positive audio duration within 50 ms of the analyzed master. The source and master are not loudness-normalized. Browser edit-list/alignment checks are recorded in the main architecture document.

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
| `bpm` | number | Median of octave-corrected local tempo from beat intervals, 1 decimal |
| `durationSeconds` | integer | Rounded track duration |
| `key` | string \| null | madmom label (e.g. `"D# minor"`, `"Bb major"`), `null` if detection fails |
| `createdAt` | date-time | SUNO `created=` from the source's comment tag, else source file mtime (UTC) |
| `updatedAt` | date-time | Import time (UTC) |

## manifest.json

Version 1 records source/master SHA-256 and original filename, full legacy MD5 fingerprint, exact sample count/rate/channels/subtype, playback probe properties, source processing record, detector config, peak windows/filter, AAC bitrate, dependency/ffmpeg versions, tool-source digests and per-artifact sizes/checksums. Integrated input LUFS, true peak and loudness range are measured without applying normalization; non-finite measurements (silence) are null. `normalizationApplied` is false.

The grid records beat count/coverage, first-downbeat index, meter, `reviewed: false`, and mean **interval regularity**, which is not model/physiological confidence. Silence and inputs shorter than one second have BPM 0 and no grid; the main repo's `db:sync-tracks` rejects BPM 0 or `durationSeconds` 0 and stops. Work-zone energy covers complete bars and the final partial second. `--first-downbeat` explicitly overrides alignment by snapping to a detected beat.

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
