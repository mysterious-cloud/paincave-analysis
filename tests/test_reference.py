"""Reference-song analysis: synthetic audio, stubbed beat detector (plus one real-detector smoke test)."""
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import analyze
import import_track
import reference

RATE = 44100
SECONDS = 20


def click_track(times, seconds=SECONDS, rate=RATE):
    """A decaying 80 Hz thump on every time: a kick the onset pass can find."""
    audio = np.zeros(int(seconds * rate))
    thump = np.sin(np.arange(int(0.2 * rate)) * 2 * np.pi * 80 / rate) * np.exp(-np.arange(int(0.2 * rate)) / 2000)
    for t in times:
        i = int(t * rate)
        audio[i:i + len(thump)] += thump[:len(audio) - i]
    return audio * 0.8


def detected(*args, **kwargs):
    """Detections on a 120 BPM grid from 0.25 s; the detector hears downbeats on the third beat of each bar."""
    beats = 0.25 + 0.5 * np.arange(39)
    return beats, beats[2::4]


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "Some song.wav"
        sf.write(self.source, click_track(detected()[0]), RATE, subtype="PCM_16")
        self.out = self.root / "out" / "analysis.json"
        patcher = patch.object(analyze, "extract_beats", side_effect=detected)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *args, source=None, out=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            reference.main([str(source or self.source), "-o", str(out or self.out), *args])

    def analysis(self, *args):
        self.run_cli(*args)
        return json.loads(self.out.read_text())

    def test_json_shape(self):
        doc = self.analysis()
        self.assertEqual(list(doc), ["version", "kind", "source", "config", "tempo", "meter", "grid"])
        self.assertEqual((doc["version"], doc["kind"]), (1, "songlab-reference-analysis"))
        self.assertEqual(doc["source"], {"file": "Some song.wav", "durationSeconds": 20.0, "sampleRate": RATE})
        self.assertEqual(doc["config"], {"minBpm": 60, "maxBpm": 200, "firstDownbeat": None})
        self.assertEqual(doc["tempo"], {"bpm": 120.0, "rangeBpm": [120.0, 120.0]})
        self.assertEqual(doc["meter"], [4, 4])
        grid = doc["grid"]
        self.assertEqual(list(grid), ["beats", "firstDownbeatIndex", "bars", "residualMs", "inferredBeatCount"])
        self.assertEqual(len(grid["beats"]), 40)
        for beat in grid["beats"]:
            self.assertEqual(list(beat), ["time", "bar", "beat", "confidence"])
            self.assertTrue(0 <= beat["time"] < 20 and 0 <= beat["confidence"] <= 1 and 1 <= beat["beat"] <= 4)
        self.assertEqual([list(bar) for bar in grid["bars"]], [["bar", "start", "end"]] * len(grid["bars"]))
        self.assertIsInstance(grid["residualMs"], float)
        self.assertIsInstance(grid["inferredBeatCount"], int)

    def test_bars_are_numbered_from_the_first_downbeat_and_pickups_are_bar_zero(self):
        grid = self.analysis()["grid"]
        beats, bars = grid["beats"], grid["bars"]
        # The detector's downbeats fall on grid index 2 (mod 4): two pickup beats, then bar 1 at 1.25 s.
        self.assertEqual(grid["firstDownbeatIndex"], 2)
        self.assertEqual([(b["bar"], b["beat"]) for b in beats[:6]], [(0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)])
        self.assertAlmostEqual(beats[2]["time"], 1.25, delta=0.01)
        self.assertEqual([b["bar"] for b in bars], list(range(1, len(bars) + 1)))
        self.assertEqual(bars[0]["start"], beats[grid["firstDownbeatIndex"]]["time"])
        for bar, following in zip(bars, bars[1:]):
            self.assertEqual(bar["end"], following["start"])
            self.assertAlmostEqual(bar["end"] - bar["start"], 2.0, delta=0.02)
        # 40 beats, 2 pickups: nine complete bars, then two beats that make no bar. The last bar ends at
        # its fourth beat plus one interval, inside the audio.
        self.assertEqual(len(bars), 9)
        self.assertEqual(beats[-1]["bar"], 10)
        self.assertLessEqual(bars[-1]["end"], 20.0)
        self.assertAlmostEqual(bars[-1]["end"], bars[-1]["start"] + 2.0, delta=0.02)

    def test_first_downbeat_override_snaps_to_a_grid_beat_and_numbers_from_it(self):
        doc = self.analysis("--first-downbeat", "5.3")
        grid = doc["grid"]
        self.assertEqual(doc["config"]["firstDownbeat"], 5.3)
        self.assertAlmostEqual(grid["beats"][grid["firstDownbeatIndex"]]["time"], 5.25, delta=0.01)
        self.assertTrue(all(b["bar"] == 0 for b in grid["beats"][:grid["firstDownbeatIndex"]]))
        self.assertEqual(grid["bars"][0]["start"], grid["beats"][grid["firstDownbeatIndex"]]["time"])
        with self.assertRaisesRegex(SystemExit, "First downbeat must identify a beat within the audio"):
            self.run_cli("--first-downbeat", "99")
        with self.assertRaisesRegex(SystemExit, "no complete bar follows the first downbeat"):
            self.run_cli("--first-downbeat", "19.9")

    def test_same_input_twice_gives_identical_bytes(self):
        self.run_cli()
        first = self.out.read_bytes()
        self.out.unlink()
        other = self.root / "elsewhere" / "again.json"
        self.run_cli(out=other)
        self.assertEqual(other.read_bytes(), first)
        self.assertNotRegex(first.decode(), r"\d{4}-\d{2}-\d{2}")  # no timestamps

    def test_same_grid_as_the_import_path(self):
        grid = self.analysis()["grid"]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            imported = analyze.analyze_audio(self.source)
        self.assertEqual([(b["time"], b["bar"], b["beat"], b["confidence"]) for b in grid["beats"]],
                         [(b["t"], b["bar"], b["beat"], b["confidence"]) for b in imported["beats"]])
        self.assertEqual(self.analysis()["tempo"]["bpm"], imported["bpm"])

    def test_refused_input_exits_with_one_line_and_writes_nothing(self):
        self.out.parent.mkdir()
        self.out.write_text("earlier result")
        silent, short, noisy = (self.root / name for name in ("silent.wav", "short.wav", "garbage.wav"))
        sf.write(silent, np.zeros(RATE * 5), RATE)
        sf.write(short, np.sin(np.arange(12000) * .01), 48000)
        noisy.write_text("not audio")
        cases = ((silent, "Error: refused silent.wav: audio is digital silence; nothing written"),
                 (short, "Error: refused short.wav: audio is 0.250 s long; at least 1 s required; nothing written"))
        for source, message in cases:
            with self.subTest(source=source.name), self.assertRaises(SystemExit) as refused:
                self.run_cli(source=source)
            self.assertEqual(refused.exception.code, message)
        with patch.object(analyze, "extract_beats", return_value=(np.arange(7) * .5, np.array([]))), \
                self.assertRaises(SystemExit) as refused:
            self.run_cli()
        self.assertRegex(refused.exception.code, r"^Error: refused Some song\.wav: no beat grid: 7 beats detected")
        self.assertTrue(refused.exception.code.endswith("; nothing written"))
        with self.assertRaisesRegex(SystemExit, r"^Error: garbage\.wav: .*; nothing written$") as unreadable:
            self.run_cli(source=noisy)
        self.assertNotIn("\n", unreadable.exception.code)
        with self.assertRaisesRegex(SystemExit, "^Error: file not found"):
            self.run_cli(source=self.root / "missing.wav")
        # The earlier result is untouched and no temporary file or other output appeared.
        self.assertEqual(self.out.read_text(), "earlier result")
        self.assertEqual([p.name for p in self.out.parent.iterdir()], ["analysis.json"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["Some song.wav", "garbage.wav", "out", "short.wav", "silent.wav"])

    def test_tempo_bounds_default_wide_and_flags_apply(self):
        self.assertEqual(inspect.signature(reference.analyze_reference).parameters["min_bpm"].default, 60)
        self.assertEqual(inspect.signature(reference.analyze_reference).parameters["max_bpm"].default, 200)
        with self.assertRaisesRegex(SystemExit, "no beat grid"):  # 120 BPM is outside 130-200
            self.run_cli("--min-bpm", "130")
        with self.assertRaises(SystemExit) as usage:
            self.run_cli("--min-bpm", "200", "--max-bpm", "100")
        self.assertEqual(usage.exception.code, 2)

    def test_runs_none_of_the_pain_cave_machinery(self):
        probe = ("import sys; sys.path.insert(0, sys.argv[1]); import reference; "
                 "print(sorted({'import_track', 'pcav', 'content_hash'} & set(sys.modules)))")
        shown = subprocess.run([sys.executable, "-c", probe, str(SRC)], capture_output=True, text=True, check=True)
        self.assertEqual(shown.stdout.strip(), "['content_hash']")  # only the decoder `analyze` shares with import
        self.run_cli()
        self.assertEqual([p.name for p in self.out.parent.iterdir()], ["analysis.json"])


class ImportPathUntouchedTests(unittest.TestCase):
    def test_pain_cave_defaults_stay_100_to_180(self):
        for function in (analyze.analyze_audio, import_track.import_song):
            parameters = inspect.signature(function).parameters
            self.assertEqual((parameters["min_bpm"].default, parameters["max_bpm"].default), (100, 180))
            self.assertIsNone(parameters["bpm"].default)  # the grid is fitted unless a manual BPM is given
        with patch.object(sys, "argv", ["import_track.py", "--help"]), redirect_stdout(io.StringIO()) as help_text, \
                self.assertRaises(SystemExit):
            import_track.main()
        self.assertIn("(default: 100)", help_text.getvalue())
        self.assertIn("(default: 180)", help_text.getvalue())

    def test_import_analysis_keeps_its_shape(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(analyze, "extract_beats", side_effect=detected):
            source = Path(tmp) / "a.wav"
            sf.write(source, click_track(detected()[0]), RATE, subtype="PCM_16")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = analyze.analyze_audio(source)
        self.assertEqual(list(result), ["version", "duration", "sampleRate", "bpm", "key", "firstDownbeat",
                                        "beats", "workZones", "raw"])
        self.assertEqual(result["version"], "3.0")
        self.assertEqual(list(result["beats"][0]), ["t", "bar", "beat", "confidence"])


@unittest.skipUnless(analyze.models.checkpoint_path().is_file(), "checkpoint not cached")
class DetectorSmokeTests(unittest.TestCase):
    def test_click_track_at_known_tempo(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "click.wav", Path(tmp) / "analysis.json"
            sf.write(source, click_track(np.arange(0.2, 29.9, 60 / 100), seconds=30), RATE, subtype="PCM_16")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                reference.main([str(source), "-o", str(out)])
            doc = json.loads(out.read_text())
        self.assertAlmostEqual(doc["tempo"]["bpm"], 100, delta=1)
        self.assertGreaterEqual(len(doc["grid"]["bars"]), 10)
        self.assertEqual(doc["grid"]["bars"][0]["bar"], 1)


if __name__ == "__main__":
    unittest.main()
