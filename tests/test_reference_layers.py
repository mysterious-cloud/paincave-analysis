"""Reference-analysis layers (stems, peaks, progress) with a stubbed beat detector and a stubbed Demucs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze
import features
import reference
import stems

from test_reference import RATE, click_track, detected


def fake_separate(mix, report=lambda _f: None):
    """Drums everywhere; bass only in bars 3-4 (5.25 s to 9.25 s on the stub grid); other and vocals silent."""
    length = mix.shape[1]
    t = np.arange(length) / stems.RATE
    drums = np.tile(mix.mean(axis=0), (2, 1)).astype(np.float32)
    bass = np.zeros((2, length), dtype=np.float32)
    span = (t >= 5.25) & (t < 9.25)
    bass[:, span] = 0.3 * np.sin(2 * np.pi * 55 * t[span])
    report(0.5)
    report(1.0)
    silent = np.zeros((2, length), dtype=np.float32)
    return {"drums": drums, "bass": bass, "other": silent, "vocals": silent.copy()}


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "Some song.wav"
        sf.write(self.source, click_track(detected()[0]), RATE, subtype="PCM_16")
        self.out = self.root / "run" / "analysis.json"
        for patcher in (patch.object(analyze, "extract_beats", side_effect=detected),
                        patch.object(stems, "separate", side_effect=fake_separate)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cli(self, *args):
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            reference.main([str(self.source), "-o", str(self.out), "--cache-dir", str(self.root / "cache"), *args])
        return stdout.getvalue()

    def test_stems_measure_each_bar_and_cache_the_split(self):
        self.run_cli("--stages", "stems")
        doc = json.loads(self.out.read_text())
        self.assertEqual(list(doc)[-1], "stems")
        block = doc["stems"]
        self.assertEqual(block["model"], "htdemucs")
        self.assertEqual(sorted(block["files"]), ["bass", "drums", "other", "vocals"])
        self.assertTrue(all(not Path(f).is_absolute() for f in block["files"].values()))
        self.assertEqual((block["relDb"], block["floorDb"]), (features.REL_DB, features.FLOOR_DB))
        bars = doc["grid"]["bars"]
        for bar in bars:
            self.assertEqual(list(bar), ["bar", "start", "end", "loudness", "stems"])
            self.assertTrue(all(0 <= v <= 1 for v in bar["stems"].values()))
        bass_active = [features.is_active(b["stems"]["bass"], block["peakDb"]["bass"]) for b in bars]
        self.assertEqual([b["bar"] for b, on in zip(bars, bass_active) if on], [3, 4])
        self.assertEqual(block["peakDb"]["vocals"], features.SILENCE_DB)
        self.assertFalse(any(features.is_active(b["stems"]["vocals"], block["peakDb"]["vocals"]) for b in bars))
        self.assertFalse((self.out.parent / "peaks.json").exists())

        self.run_cli("--stages", "stems")  # a second run reuses the cached FLAC stems
        self.assertEqual(stems.separate.call_count, 1)
        self.assertEqual(json.loads(self.out.read_text())["stems"], block)

    def test_peaks_cover_the_mix_and_each_stem(self):
        self.run_cli("--stages", "stems,peaks")
        peaks = json.loads((self.out.parent / "peaks.json").read_text())
        self.assertEqual(list(peaks), ["peaksPerSecond", "mix", "stems"])
        self.assertEqual(peaks["peaksPerSecond"], 50)
        self.assertEqual(len(peaks["mix"]), 20 * 50)
        self.assertEqual(sorted(peaks["stems"]), ["bass", "drums", "other", "vocals"])
        self.assertTrue(all(lo <= hi for lo, hi in peaks["mix"]))
        self.assertEqual(max(hi for _, hi in peaks["stems"]["vocals"]), 0)

    def test_peaks_without_stems_draw_the_mix_only(self):
        self.run_cli("--stages", "peaks")
        peaks = json.loads((self.out.parent / "peaks.json").read_text())
        self.assertEqual(peaks["stems"], {})
        self.assertNotIn("stems", json.loads(self.out.read_text()))
        stems.separate.assert_not_called()

    def test_progress_lines_are_json_per_stage(self):
        lines = [l for l in self.run_cli("--stages", "stems,peaks", "--progress").splitlines()
                 if l.startswith("PROGRESS ")]
        events = [json.loads(l.removeprefix("PROGRESS ")) for l in lines]
        self.assertTrue(all(list(e) == ["stage", "fraction"] for e in events))
        self.assertEqual([e["stage"] for e in events if e["fraction"] in (0, 1)],
                         ["grid", "grid", "stems", "stems", "peaks", "peaks"])
        self.assertIn({"stage": "stems", "fraction": 0.5}, events)
        self.assertEqual(self.run_cli().count("PROGRESS"), 0)

    def test_unknown_stage_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as usage:
            self.run_cli("--stages", "stems,chords")
        self.assertEqual(usage.exception.code, 2)

    def test_activity_matrix_marks_playing_stems(self):
        text = self.run_cli("--stages", "stems", "--print-activity")
        rows = [l for l in text.splitlines() if l[:3].strip().isdigit()]
        self.assertEqual(len(rows), len(json.loads(self.out.read_text())["grid"]["bars"]))
        self.assertIn("D B . .", rows[2])
        self.assertIn("D . . .", rows[0])


class FeatureTests(unittest.TestCase):
    def test_rms_db_and_levels(self):
        self.assertEqual(features.rms_db(np.zeros(10)), features.SILENCE_DB)
        self.assertAlmostEqual(features.rms_db(np.ones(10)), 0.0)
        bars = [{"start": 0, "end": 1}, {"start": 1, "end": 2}]
        signal = np.concatenate([np.ones(100), 0.1 * np.ones(100)])
        levels, peak = features.stem_levels({"x": signal}, 100, bars)
        self.assertEqual(levels["x"], [1.0, 0.1])
        self.assertEqual(peak["x"], 0.0)
        self.assertTrue(features.is_active(0.1, 0.0))
        self.assertFalse(features.is_active(0.01, 0.0))  # 40 dB below its loudest bar
        self.assertFalse(features.is_active(1.0, -60.0))  # its loudest bar is still below the floor

    def test_peaks_window_min_max(self):
        self.assertEqual(features.peaks(np.array([0, 1, -1, 0.5], dtype=np.float32), 4, 2), [[0.0, 1.0], [-1.0, 0.5]])


if __name__ == "__main__":
    unittest.main()
