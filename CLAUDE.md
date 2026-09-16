@README.md

# Pain Cave Analysis — AI rules

Ecosystem rules and philosophy: [../paincave/CLAUDE.md](../paincave/CLAUDE.md). Output contract: [docs/output-formats.md](docs/output-formats.md). Binary spec: [docs/pcav-format.md](docs/pcav-format.md).

## Rules

- **Deterministic output.** The same input file and flags must produce identical `analysis.pcav`, `plan_workzones.json` and hash. No randomness, no wall-clock values outside `metadata.json` `updatedAt`.
- **Schema-validated metadata.** `metadata.json` must validate against `../paincave/shared/schemas/metadata.schema.json`. Change the schema in the paincave repo first, then the writer here. Never add fields the schema lacks (`additionalProperties: false`).
- **Don't change the hash casually.** The hash is the track directory name and DB key. Any change to `src/content_hash.py` re-keys every track. Get the user's sign-off first.
- **PCAV changes update the spec.** Any change to `src/pcav.py` layout goes into `docs/pcav-format.md` in the same commit. Readers must stay backward compatible.
- **Keep the stdout markers.** `../paincave/bin/import.sh` parses the last `Hash: <hash>` line, `Track already exists:`, and `Revision: <dir> (candidate; ...)` (the bare `Revision:` prefix skips paid thumbnail generation; the path ends before ` (candidate`); `../paincave/server/src/routes/tracks.ts` takes the first `Hash: <hash>`. `test_cli_stdout_markers_consumers_parse` locks them. Refusals exit 1 with the stderr line `Error: refused <file>: <reason>; nothing written`, locked by `test_refuses_short_silent_and_gridless_audio`.
- **No redundant passes.** One decode, one band-pass and one SHA-256 per staged file; the beat detector reads the samples already in memory, so nothing decodes the master twice; AAC and loudness share one ffmpeg run, started before analysis so it overlaps detection. `test_each_input_decoded_filtered_and_checksummed_once` locks the counts. Measured 2026-09-15 on a 10-core M1 Max: 2:39 track 4.1 s, 3:44 track 6.7 s wall. Every stage prints `timing: <stage> <s>s` on stderr; don't add a full pass over the audio without measuring it.

- **Detection and grid are separate.** Beat This! (`small0`, `dbn=False`) says where it hears beats; `analyze.fit_beat_grid` turns that into the locally linear grid plans are written against (one line per beat over its 8-bar neighbourhood, so slow drift is followed). Keep that split: detector quirks get handled in the fit, not by patching detections. Changing the checkpoint changes every grid — it is pinned in `src/models.py` and its SHA-256 goes into each `manifest.json`.
- **Interrupts clean up.** SIGTERM unwinds through the staging `with` block (handler: one-shot, no I/O). Anything that holds a staging directory must keep its `flock`.
- **Docs follow code.** When behavior in `src/import_track.py` changes, update `docs/output-formats.md` and README in the same commit.

## Verify

- `uv run pytest -q` passes.
- `uv run python src/import_track.py --help` runs clean.
- For analysis/PCAV/hash changes, import a real track into a scratch `--library` (never the real library with `--force`) and inspect the output.
