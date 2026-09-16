# PCAV Format Specification

**Writer version: 2 (2026-09-14). Readers accept versions 1 and 2.** All integers and floats are little-endian. New writes use v2; existing assets are never silently rewritten.

## Container

| Offset | Bytes | Field |
|---|---|---|
| 0 | 4 | ASCII `PCAV` |
| 4 | 1 | Version: 1 or 2 |
| 5 | 1 | Number of directory entries |
| 6 | 9 × count | Directory |

Each directory entry is `type:uint8, offset:uint32, length:uint32`. Offsets are relative to the file start. Sections must lie after the complete directory, fit within the file, not overlap, and have unique types. Unknown section types are skipped after these checks. At most 255 sections; each section is limited to 16 MiB. Counts must match section lengths exactly before allocation: at most 100,000 beats and 2,000,000 peaks per level.

All sections remain optional. v2 timestamps must be strictly increasing, finite, nonnegative and earlier than exact duration when ANALYSIS is present. Invalid/truncated input raises an error, never a partial grid.

## Version 2 sections

### 0x01 ANALYSIS — 12 bytes

`durationSeconds:float64, bpmTimes10:uint16, key:uint8, reserved:uint8`

Duration is measured frames / sample rate, without whole-second rounding. BPM 0 means unknown (for example silence or insufficient beats). Key is always 0x00 (unknown) for tracks analyzed after 2026-09-15: key detection went away with madmom. Reserved byte is zero. Duration must be positive and finite.

**Key Encoding:**

```
0x00 = Unknown
0x01 = C major    0x0D = C minor
0x02 = C# major   0x0E = C# minor
0x03 = D major    0x0F = D minor
0x04 = D# major   0x10 = D# minor
0x05 = E major    0x11 = E minor
0x06 = F major    0x12 = F minor
0x07 = F# major   0x13 = F# minor
0x08 = G major    0x14 = G minor
0x09 = G# major   0x15 = G# minor
0x0A = A major    0x16 = A minor
0x0B = A# major   0x17 = A# minor
0x0C = B major    0x18 = B minor
```

Flat keys map to their sharp equivalents (e.g., Ab = G#).

### 0x02 BEATS — 12 + 8 × count bytes

`count:uint32, firstDownbeatIndex:int32, beatsPerBar:uint8, reserved:3 bytes, absoluteMicroseconds:uint64[count]`

The downbeat index identifies **bar 1, beat 1**; -1 means unknown. Meter is 1–32 beats/bar; the current analyzer emits 4. Values must be below 2^53 microseconds for exact JavaScript integer representation. Quantize each absolute timestamp independently; maximum error is 0.5 microseconds, with no accumulated delta drift. Do not clamp long intro/gap values.

Before the selected first downbeat, beats use bar 0 and beat `((index - firstDownbeatIndex) mod meter) + 1`, with nonnegative modulo. From the downbeat onward, bars start at 1. Multiple pickup bars still use bar 0; their beat labels remain within 1–meter.

### 0x03 PEAKS / 0x08 PEAKS_OVERVIEW — 12 + 2 × count bytes

`count:uint32, sampleRate:uint32, windowSamples:uint32, [min:int8, max:int8][count]`

Sample rate is 1–768,000 Hz; window size is 1–2^24 samples. Rate is **sampleRate / windowSamples**, not a fixed 48 kHz assumption. Current windows: 256 (editor), 8192 (overview). Peaks include the final partial window. Last-window display coverage can exceed exact media duration by less than one window; playback uses exact duration.

The writer uses mono samples, a 300 Hz low-pass display filter and amplitude normalization for visualization only. Audio is not normalized. Values must satisfy -128 ≤ min ≤ max ≤ 127; readers divide by 127 and clamp -128 to -1.

## Legacy version 1

- ANALYSIS: `<HHBx>`: whole-second uint16 duration, BPM × 10, key, reserved.
- BEATS: `<HH>` first beat milliseconds and count, then exactly max(0, count-1) uint8 centisecond intervals. It contains **no downbeat phase**. Readers accumulate integer milliseconds to avoid additional floating-point summation error; information already lost by old delta quantization cannot be recovered.
- PEAKS: uint32 count plus int8 pairs, with implicit 48 kHz / 256 or 8192 samples. The older uint16-count editor section was 10 peaks/s and is recognized only when its exact length matches. Non-48 kHz legacy files cannot recover their real timebase from PCAV alone; reanalysis is required.
- The main client may recover phase from `plan_workzones.json` only when every machine zone start matches a unique raw beat within 6 ms and implies the same downbeat index. Otherwise phase is unknown (`grid.phaseSource: 'unknown'`) and no numbered beats are returned. A non-OK, non-JSON `Content-Type` or unparseable sidecar counts as no anchors, so a server must send it as JSON. The main app's editor and ride do not surface unknown phase yet. This recovers the shipped Unbreakable grid without editing catalog assets.

## Partial reads

Fetch bytes 0–5, then exactly bytes 0 through `6 + 9 × sectionCount - 1`. Fetch the chosen peak section using its offset and length; pass the container version to the section decoder. Verify 206 Content-Range and exact body length. A 200 response is a whole-file fallback. Reading a fixed 200-byte header can overrun a valid short file.

## Verification

`uv run pytest -q` (equivalently `uv run python -m unittest discover -s tests`) exercises Python validation and deterministic import failures. `uv run python tests/export_pcav_fixtures.py ../paincave/tests/fixtures/pcav` generates shared fixtures; `bun test tests/pcav.test.ts` in paincave reads them. Keep the stripped shipped Unbreakable fixture alongside generated fixtures; it is historical regression evidence, not generated synthetic data.

## History

- 2026-02-05: v1, uint16-count 10/s peaks.
- 2026-02-11: v1 widened peak count and changed resolution without a header version bump.
- 2026-09-14: v2 explicit phase/meter, exact duration, absolute microseconds, per-level timebase; strict bounds and legacy compatibility.
