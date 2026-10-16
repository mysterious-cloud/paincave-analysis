"""Class beats (src/class_beats.py): real decoding, a stubbed detector (accuracy is measured on real classes, where
the imported songs' kick-aligned grids are the reference: paincave docs/studio.md, phase 2b)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import class_beats


class ClassBeatsTest(unittest.TestCase):
    def test_decodes_to_the_detectors_rate_and_writes_beats_and_downbeats(self):
        heard = []

        def detector(audio, rate):
            heard.append((len(audio), rate, audio.dtype))
            # 120 BPM, then 150 BPM: no grid, no bound, every beat as heard
            beats = [1 + 0.5 * i for i in range(8)] + [5 + 0.4 * i for i in range(10)]
            return np.array(beats), np.array(beats[::4])

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "class.wav"
            sf.write(wav, 0.1 * np.sin(np.linspace(0, 2000, 3 * 44100)), 44100)
            out = Path(tmp) / "class.beats.json"
            with patch.object(class_beats.analyze, "load_detector", return_value=detector), \
                    patch.object(sys, "argv", ["class_beats.py", str(wav), "-o", str(out)]):
                class_beats.main()
            self.assertEqual(heard[0][1:], (class_beats.RATE, np.float32))
            self.assertAlmostEqual(heard[0][0] / class_beats.RATE, 3, places=2)
            found = json.loads(out.read_text())
            self.assertEqual(sorted(found), ["beats", "checkpoint", "downbeats"])
            self.assertEqual(found["checkpoint"]["name"], "small0")
            self.assertEqual(found["beats"][7:10], [4.5, 5.0, 5.4])
            self.assertEqual(found["downbeats"], [1.0, 3.0, 5.0, 6.6, 8.2])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["class.beats.json", "class.wav"])

    def test_not_audio_is_refused_and_nothing_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "notes.txt"
            bad.write_text("not audio")
            result = subprocess.run([sys.executable, str(Path(class_beats.__file__)), str(bad), "-o", str(Path(tmp) / "o.json")],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("could not decode", result.stderr)
            self.assertFalse((Path(tmp) / "o.json").exists())


if __name__ == "__main__":
    unittest.main()
