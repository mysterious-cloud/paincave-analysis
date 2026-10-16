"""Class transcript (src/class_transcript.py): real decoding and resampling; Demucs stubbed and whisper.cpp faked (accuracy
is measured on THE TRIP 43 against its real subtitles: paincave docs/studio.md, phase 2b)."""
import io
import json
import shutil
import subprocess
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
import class_transcript

# Stands in for whisper-cli: checks what it was given, reports progress on stderr as whisper.cpp does, writes -of.json
# (full JSON, tokens timed by DTW in centiseconds)
FAKE_WHISPER = """#!/usr/bin/env python3
import json, sys, soundfile
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
info = soundfile.info(value("-f"))
assert (info.samplerate, info.channels) == (16000, 1), info
assert {"-ojf", "-sns", "-nfa", "-pp"} <= set(args) and "--vad" not in args, args
assert (value("-l"), value("-mc"), value("-dtw")) == ("en", "0", "large.v3"), args
for p in (40, 100):
    print(f"whisper_print_progress_callback: progress = {p:3d}%", file=sys.stderr)
token = lambda text, t: {"text": text, "offsets": {"from": 0, "to": 3000}, "t_dtw": t}
json.dump({"transcription": [{"tokens": [token("[_BEG_]", -1), token(" Hey", 100), token(" there", 130), token(".", 160),
                                         token(" Let's", 400), token(" ride", 430)]}]}, open(value("-of") + ".json", "w"))
"""


class ClassTranscriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.wav = self.tmp / "class.wav"
        sf.write(self.wav, 0.1 * np.sin(np.linspace(0, 2000, 3 * 44100)), 44100)
        self.model = self.tmp / "ggml-large-v3.bin"
        self.model.write_bytes(b"model")
        self.whisper = self.tmp / "whisper-cli"
        self.whisper.write_text(FAKE_WHISPER.replace("python3", sys.executable, 1))
        self.whisper.chmod(0o755)

    def run_main(self, *extra: str) -> list[dict]:
        out, separated = io.StringIO(), []

        def separate(mix, report):
            separated.append(mix.shape)
            report(0.5)
            return mix.mean(axis=0)

        argv = ["class_transcript.py", str(self.wav), "-o", str(self.tmp / "class.whisper.vtt"),
                "--model", str(self.model), *extra]
        with patch.object(class_transcript.stems, "separate_vocals", separate), \
                patch.object(class_transcript, "WHISPER", str(self.whisper)), \
                patch.object(sys, "argv", argv), redirect_stdout(out):
            class_transcript.main()
        self.assertEqual(separated, [(2, 3 * 44100)])
        return [json.loads(line.removeprefix("PROGRESS ")) for line in out.getvalue().splitlines()]

    def test_writes_the_vtt_with_a_note_and_reports_both_stages(self):
        steps = self.run_main()
        self.assertEqual(steps, [{"stage": "voice", "fraction": 0}, {"stage": "voice", "fraction": 0.5},
                                 {"stage": "words", "fraction": 0}, {"stage": "words", "fraction": 0.4},
                                 {"stage": "words", "fraction": 1.0}])
        self.assertEqual((self.tmp / "class.whisper.vtt").read_text(),
                         "WEBVTT\n\nNOTE Made by paincave-analysis: Demucs htdemucs_ft vocals, whisper.cpp ggml-large-v3\n\n"
                         "00:00:01.000 --> 00:00:01.900\nHey there.\n\n00:00:04.000 --> 00:00:04.600\nLet's ride\n")
        self.assertFalse(list(self.tmp.glob(".*.tmp")))

    def test_a_missing_model_is_refused_and_nothing_written(self):
        self.model.unlink()
        with self.assertRaises(SystemExit) as exit_, patch("sys.stderr", io.StringIO()) as err:
            self.run_main()
        self.assertEqual(exit_.exception.code, 1)
        self.assertIn("no whisper model at", err.getvalue())
        self.assertFalse((self.tmp / "class.whisper.vtt").exists())

    def test_not_audio_is_refused(self):
        bad = self.tmp / "notes.txt"
        bad.write_text("not audio")
        result = subprocess.run([sys.executable, str(SRC / "class_transcript.py"), str(bad), "-o", str(self.tmp / "o.vtt"),
                                 "--model", str(self.model)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("could not decode", result.stderr)
        self.assertFalse((self.tmp / "o.vtt").exists())


def segment(*tokens: tuple[str, float, float]) -> dict:
    """A whisper.cpp full-JSON segment from (text, from, to) tokens, seconds"""
    return {"tokens": [{"text": text, "offsets": {"from": round(start * 1000), "to": round(end * 1000)}}
                       for text, start, end in tokens]}


class LinesTest(unittest.TestCase):
    def test_words_from_tokens(self):
        words = class_transcript.words_of({"transcription": [segment(
            ("[_BEG_]", 0, 0), (" Handle", 1, 1.2), ("bar", 1.2, 1.4), (" up", 1.5, 1.7), (".", 1.7, 1.7), ("[_TT_90]", 1.8, 1.8))]})
        self.assertEqual(words, [(1, 1.4, "Handlebar"), (1.5, 1.7, "up.")])

    def test_dtw_times_a_word_from_its_moment_to_the_next_word_and_a_sentence_runs_across_segments(self):
        def timed(text: str, dtw: float) -> dict:
            # Whisper's own offsets stretch the first word over the silence before it
            return {"text": text, "offsets": {"from": 0, "to": 5290}, "t_dtw": round(dtw * 100)}

        words = class_transcript.words_of({"transcription": [
            {"tokens": [timed(" should", 25.0), timed(" have", 25.2), timed(" a", 25.4)]},
            {"tokens": [timed("slight", 25.6), timed(" bend", 25.9), timed(".", 26.1)]}]})
        self.assertEqual(words, [(25.0, 25.2, "should"), (25.2, 25.4, "have"), (25.4, 25.6, "a"),
                                 (25.6, 25.9, "slight"), (25.9, 26.4, "bend.")])
        self.assertEqual(class_transcript.lines_of(words), [(25.0, 26.4, "should have a slight bend.")])

    def test_lines_end_at_sentences_pauses_long_commas_and_limits(self):
        def spoken(text: str, start: float, step=0.3):
            return [(f" {word}", start + i * step, start + i * step + 0.25) for i, word in enumerate(text.split())]

        lines = class_transcript.lines_of(class_transcript.words_of({"transcription": [
            # A segment of calls across a song: each pause ends a line, as does each sentence
            segment(*spoken("Control", 0), *spoken("thirty seconds left", 10), *spoken("Hover. Nice", 20)),
            segment(*spoken("Short, then one more comma, and on we go, yes and on we go", 30)),
            segment(*spoken(" ".join(["go"] * 20), 40))
        ]}))
        self.assertEqual([text for _, _, text in lines], [
            "Control", "thirty seconds left", "Hover.", "Nice",
            "Short, then one more comma, and on we go,", "yes and on we go",
            " ".join(["go"] * 14), " ".join(["go"] * 6)])
        self.assertEqual(lines[1][:2], (10, 10.85))

    def test_a_line_shows_at_least_half_a_second_unless_the_next_starts(self):
        text = class_transcript.vtt([(1, 1, "Go."), (1.2, 1.4, "Now."), (3, 3.1, "Up.")], "test")
        self.assertEqual(text, "WEBVTT\n\nNOTE test\n\n00:00:01.000 --> 00:00:01.200\nGo.\n\n"
                               "00:00:01.200 --> 00:00:01.700\nNow.\n\n00:00:03.000 --> 00:00:03.500\nUp.\n")


if __name__ == "__main__":
    unittest.main()
