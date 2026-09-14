@README.md

# Pain Cave Analysis — AI rules

Ecosystem rules and philosophy: [../paincave/CLAUDE.md](../paincave/CLAUDE.md). Output contract: [docs/output-formats.md](docs/output-formats.md). Binary spec: [docs/pcav-format.md](docs/pcav-format.md).

## Rules

- **Deterministic output.** The same input file and flags must produce identical `analysis.pcav`, `plan_workzones.json` and hash. No randomness, no wall-clock values outside `metadata.json` `updatedAt`.
- **Schema-validated metadata.** `metadata.json` must validate against `../paincave/shared/schemas/metadata.schema.json`. Change the schema in the paincave repo first, then the writer here. Never add fields the schema lacks (`additionalProperties: false`).
- **Don't change the hash casually.** The hash is the track directory name and DB key. Any change to `src/content_hash.py` re-keys every track. Get the user's sign-off first.
- **PCAV changes update the spec.** Any change to `src/pcav.py` layout goes into `docs/pcav-format.md` in the same commit. Readers must stay backward compatible.
- **Keep the `Hash: <hash>` stdout line.** `../paincave/bin/import.sh` parses it.
- **Docs follow code.** When behavior in `src/import_track.py` changes, update `docs/output-formats.md` and README in the same commit.

## Verify

- `uv run python src/import_track.py --help` runs clean.
- For analysis/PCAV/hash changes, import a real track into a scratch `--library` (never the real library with `--force`) and inspect the output.
