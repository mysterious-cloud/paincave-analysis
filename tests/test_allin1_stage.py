"""The allin1 stage: All-In-One's segments snapped to the grid (the model itself is stubbed; see tools/allin1)."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import allin1_stage
import analyze
import reference
import stems

from test_reference import RATE, click_track, detected
from test_reference_layers import fake_separate

BARS = [{"bar": i + 1, "start": 1.25 + 2 * i, "end": 3.25 + 2 * i} for i in range(9)]  # the stub grid: 9 bars of 2 s
SEGMENTS = [{"start": 0.0, "end": 1.2, "label": "start"}, {"start": 1.2, "end": 5.3, "label": "intro"},
            {"start": 5.3, "end": 13.1, "label": "verse"}, {"start": 13.1, "end": 13.4, "label": "chorus"},
            {"start": 13.4, "end": 19.0, "label": "chorus"}, {"start": 19.0, "end": 20.0, "label": "end"}]


class SnapTests(unittest.TestCase):
    def test_segments_snap_to_bars_and_edges_fold_in(self):
        found = allin1_stage.to_sections(SEGMENTS, BARS)
        # intro from bar 1 (the silent start joins it); verse at 5.3 s -> bar 3; the 0.3 s chorus snaps onto the
        # next one's bar and is dropped; the silent end joins the last
        self.assertEqual([(s["startBar"], s["endBar"], s["label"]) for s in found],
                         [(1, 3, "intro"), (3, 7, "verse"), (7, 10, "chorus")])

    def test_block_keeps_the_raw_seconds(self):
        b = allin1_stage.block({"model": "harmonix-all", "bpm": 120, "segments": SEGMENTS}, BARS)
        self.assertEqual((b["tool"], b["bpm"], len(b["segmentsSeconds"])), ("allin1", 120, 6))
        self.assertEqual(b["boundariesSeconds"], [5.25, 13.25])


class StageTests(unittest.TestCase):
    def test_the_stage_runs_on_the_cached_stems_and_adds_a_method(self):
        calls = []

        def fake_run(audio, stems_dir, cache_dir):
            calls.append(stems_dir)
            self.assertEqual(sorted(p.name for p in stems_dir.iterdir()), ["bass.flac", "drums.flac", "other.flac", "vocals.flac"])
            return {"model": "harmonix-all", "bpm": 120, "segments": SEGMENTS}

        with tempfile.TemporaryDirectory() as tmp, patch.object(analyze, "extract_beats", side_effect=detected), \
                patch.object(stems, "separate", side_effect=fake_separate), patch.object(allin1_stage, "run", side_effect=fake_run):
            source, out = Path(tmp) / "a.wav", Path(tmp) / "analysis.json"
            sf.write(source, click_track(detected()[0]), RATE, subtype="PCM_16")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                reference.main([str(source), "-o", str(out), "--stages", "sections,allin1"])  # stems comes with allin1
            doc = json.loads(out.read_text())
        self.assertEqual(len(calls), 1)
        self.assertIn("stems", doc)
        self.assertEqual(sorted(doc["sections"]["methods"]), ["allin1", "ssm"])
        self.assertEqual([s["label"] for s in doc["sections"]["methods"]["allin1"]["sections"]], ["intro", "verse", "chorus"])


if __name__ == "__main__":
    unittest.main()
