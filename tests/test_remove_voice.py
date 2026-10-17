"""Remove voice (src/remove_voice.py): real decoding, resampling and writing; Demucs stubbed with the exact "voice"
(how well the real model hears a coach is measured on THE TRIP 43: paincave docs/studio.md, class breakdown)."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import remove_voice
import stems

RATE = 48000
SECONDS = 6
VOICE_HZ, MUSIC_HZ = 440.0, 110.0


def tone(hz: float, rate: int, level: float) -> np.ndarray:
    return (level * np.sin(2 * np.pi * hz * np.arange(SECONDS * rate) / rate)).astype(np.float32)


class RemoveVoiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        # The song at 48 kHz: music everywhere, a "voice" (a 440 Hz tone) everywhere too
        self.song = np.stack([tone(MUSIC_HZ, RATE, 0.3) + tone(VOICE_HZ, RATE, 0.3)] * 2)
        self.wav = self.tmp / "song.wav"
        sf.write(self.wav, self.song.T, RATE, subtype="FLOAT")

    def run_main(self, spans) -> np.ndarray:
        spans_file = self.tmp / "spans.json"
        spans_file.write_text(json.dumps(spans))
        out = self.tmp / "out.wav"

        def separate(mix, report, stereo=False):
            # Demucs hears the voice exactly: the 440 Hz tone at its own 44.1 kHz
            self.assertTrue(stereo)
            report(1.0)
            voice = tone(VOICE_HZ, stems.RATE, 0.3)[:mix.shape[1]]
            return np.stack([voice, voice])

        argv = ["remove_voice.py", str(self.wav), "-o", str(out), "--spans", str(spans_file)]
        with patch.object(remove_voice.stems, "separate_vocals", separate), patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            remove_voice.main()
        written, rate = sf.read(out, dtype="float32", always_2d=True)
        self.assertEqual((rate, written.shape[1]), (RATE, 2))
        return written.T

    def test_the_voice_goes_inside_the_spans_and_the_song_is_untouched_outside(self):
        out = self.run_main([[2.0, 4.0]])
        music = np.stack([tone(MUSIC_HZ, RATE, 0.3)] * 2)
        # Inside (clear of the fades): the music alone, give or take resampling and 16 bits
        inside = slice(int(2.2 * RATE), int(3.8 * RATE))
        self.assertLess(np.abs(out[:, inside] - music[:, inside]).max(), 0.01)
        # Outside the fades: the input, to 16 bits
        for outside in (slice(0, int(1.8 * RATE)), slice(int(4.2 * RATE), SECONDS * RATE)):
            self.assertLess(np.abs(out[:, outside] - self.song[:, outside]).max(), 1 / 32768 + 1e-6)
        # The fade in: the voice leaves gradually, no step
        edge = out[0, int(1.85 * RATE):int(2.05 * RATE)] - music[0, int(1.85 * RATE):int(2.05 * RATE)]
        self.assertLess(np.abs(np.diff(edge)).max(), 0.02)

    def test_no_spans_leave_the_song_as_it_is(self):
        out = self.run_main([])
        self.assertLess(np.abs(out - self.song).max(), 1 / 32768 + 1e-6)

    def test_a_bad_span_or_not_audio_is_refused_and_nothing_written(self):
        for spans, audio in (([[3, 2]], self.wav), ([[1, 2]], self.tmp / "notes.txt")):
            (self.tmp / "notes.txt").write_text("not audio")
            spans_file = self.tmp / "spans.json"
            spans_file.write_text(json.dumps(spans))
            argv = ["remove_voice.py", str(audio), "-o", str(self.tmp / "bad.wav"), "--spans", str(spans_file)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit) as exit_:
                remove_voice.main()
            self.assertEqual(exit_.exception.code, 1)
            self.assertFalse((self.tmp / "bad.wav").exists())


if __name__ == "__main__":
    unittest.main()
