"""Separate coach (src/separate_coach.py): real decoding, resampling and writing; Demucs stubbed with the exact "voice"
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
import separate_coach
import stems

RATE = 48000
SECONDS = 6
VOICE_HZ, MUSIC_HZ = 440.0, 110.0
STEP = 1 / 32768 + 1e-6  # one 16-bit step


def tone(hz: float, rate: int, level: float) -> np.ndarray:
    return (level * np.sin(2 * np.pi * hz * np.arange(SECONDS * rate) / rate)).astype(np.float32)


class SeparateCoachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        # The class at 48 kHz: music everywhere, a "voice" (a 440 Hz tone) everywhere too
        self.music = np.stack([tone(MUSIC_HZ, RATE, 0.3)] * 2)
        self.recording = self.music + np.stack([tone(VOICE_HZ, RATE, 0.3)] * 2)
        self.wav = self.tmp / "class.wav"
        sf.write(self.wav, self.recording.T, RATE, subtype="FLOAT")

    def run_main(self, spans, audio: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
        spans_file = self.tmp / "spans.json"
        spans_file.write_text(json.dumps(spans))
        music, coach = self.tmp / "music.flac", self.tmp / "coach.flac"

        def separate(mix, report, stereo=False):
            # Demucs hears the voice exactly: the 440 Hz tone at its own 44.1 kHz
            self.assertTrue(stereo)
            report(1.0)
            voice = tone(VOICE_HZ, stems.RATE, 0.3)[:mix.shape[1]]
            return np.stack([voice, voice])

        argv = ["separate_coach.py", str(audio or self.wav), "--spans", str(spans_file), "-o", str(music), "-c", str(coach)]
        with patch.object(separate_coach.stems, "separate_vocals", separate), patch.object(sys, "argv", argv), \
                redirect_stdout(io.StringIO()) as stdout:
            separate_coach.main()
        self.assertEqual(json.loads(stdout.getvalue().splitlines()[-1].removeprefix("PROGRESS ")), {"stage": "voice", "fraction": 1})
        written = []
        for path in (music, coach):
            self.assertEqual(sf.info(path).subtype, "PCM_16")
            audio_, rate = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(rate, sf.info(audio or self.wav).samplerate)
            written.append(audio_.T)
        self.assertEqual(written[0].shape, written[1].shape)
        return written[0], written[1]

    def test_the_coach_holds_the_voice_inside_the_spans_and_nothing_outside(self):
        music, coach = self.run_main([[2.0, 4.0]])
        self.assertEqual(music.shape, self.recording.shape)
        voice = self.recording - self.music
        # Inside (clear of the fades): the music alone and the voice in the coach, give or take resampling and 16 bits
        inside = slice(int(2.2 * RATE), int(3.8 * RATE))
        self.assertLess(np.abs(music[:, inside] - self.music[:, inside]).max(), 0.01)
        self.assertLess(np.abs(coach[:, inside] - voice[:, inside]).max(), 0.01)
        # Outside the fades: the coach is silent and the music is the recording, to 16 bits
        for outside in (slice(0, int(1.9 * RATE)), slice(int(4.1 * RATE), SECONDS * RATE)):
            self.assertEqual(np.abs(coach[:, outside]).max(), 0)
            self.assertLess(np.abs(music[:, outside] - self.recording[:, outside]).max(), STEP)

    def test_music_and_coach_add_back_to_the_recording_everywhere(self):
        music, coach = self.run_main([[0.5, 1.5], [2.0, 4.0], [5.5, 7.0]])
        self.assertLess(np.abs(music + coach - self.recording).max(), STEP)

    def test_the_coach_fades_in_and_out_without_a_step(self):
        music, coach = self.run_main([[2.0, 4.0]])
        for edge in (slice(int(1.85 * RATE), int(2.05 * RATE)), slice(int(3.95 * RATE), int(4.15 * RATE))):
            self.assertLess(np.abs(np.diff(coach[0, edge])).max(), 0.02)
            self.assertLess(np.abs(np.diff(music[0, edge] - self.music[0, edge])).max(), 0.02)
        # The fade is under way at the span's edge, not a switch: half the voice 50 ms before it
        self.assertAlmostEqual(float(separate_coach.gain(int(1.95 * RATE), 1, RATE, [[2.0, 4.0]])[0]), 0.5, places=2)

    def test_a_mono_recording_stays_mono(self):
        mono = self.tmp / "mono.wav"
        sf.write(mono, self.recording[0], RATE, subtype="PCM_16")
        music, coach = self.run_main([[2.0, 4.0]], mono)
        self.assertEqual(music.shape, (1, SECONDS * RATE))
        recording, _ = sf.read(mono, dtype="float32")
        # A 16-bit recording comes back sample for sample
        self.assertEqual(np.abs(music[0] + coach[0] - recording).max(), 0)
        inside = slice(int(2.2 * RATE), int(3.8 * RATE))
        self.assertLess(np.abs(music[0, inside] - self.music[0, inside]).max(), 0.01)

    def test_no_spans_leave_the_coach_silent_and_the_music_as_recorded(self):
        music, coach = self.run_main([])
        self.assertEqual(np.abs(coach).max(), 0)
        self.assertLess(np.abs(music - self.recording).max(), STEP)

    def test_bad_spans_not_audio_or_an_output_over_the_input_are_refused_and_nothing_written(self):
        (self.tmp / "notes.txt").write_text("not audio")
        music, coach = self.tmp / "music.flac", self.tmp / "coach.flac"
        for spans, audio, outputs in (([[3, 2]], self.wav, (music, coach)), ([[1, "2"]], self.wav, (music, coach)),
                                      ({"start": 1}, self.wav, (music, coach)), ([[1, 2]], self.tmp / "notes.txt", (music, coach)),
                                      ([[1, 2]], self.wav, (self.wav, coach)), ([[1, 2]], self.wav, (music, self.wav)),
                                      ([[1, 2]], self.wav, (music, music))):
            spans_file = self.tmp / "spans.json"
            spans_file.write_text(json.dumps(spans))
            argv = ["separate_coach.py", str(audio), "--spans", str(spans_file), "-o", str(outputs[0]), "-c", str(outputs[1])]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit) as exit_:
                separate_coach.main()
            self.assertEqual(exit_.exception.code, 1)
            self.assertRegex(stderr.getvalue(), r"^Error: [^\n]+\n$")
            self.assertFalse(music.exists() or coach.exists())
            self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["class.wav", "notes.txt", "spans.json"])


if __name__ == "__main__":
    unittest.main()
