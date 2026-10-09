"""Section boundaries and labels on synthetic audio with a known form."""
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
import reference
import score_sections
import sections
import stems

from test_reference import RATE, click_track, detected
from test_reference_layers import fake_separate

SR = 22050
BAR = 2.0  # seconds: 120 BPM in 4/4
CHORDS = {"A": (261.63, 329.63, 392.00), "B": (369.99, 466.16, 554.37)}  # C major, F# major


def form_audio(form: str, bars_each: int = 8) -> np.ndarray:
    """Sections of sustained chords with a kick on every beat, one letter per section."""
    t = np.arange(int(BAR * bars_each * SR)) / SR
    kick = np.zeros_like(t)
    for beat in np.arange(0, BAR * bars_each, BAR / 4):
        i = int(beat * SR)
        n = min(len(t) - i, int(0.1 * SR))
        kick[i:i + n] += np.sin(2 * np.pi * 60 * t[:n]) * np.exp(-t[:n] * 40)
    parts = [sum(np.sin(2 * np.pi * f * t) for f in CHORDS[letter]) / 6 + 0.5 * kick for letter in form]
    return np.concatenate(parts).astype(np.float32)


def grid(bar_count: int) -> tuple[list[dict], list[dict]]:
    beats = [{"time": round(i * BAR / 4, 3), "bar": i // 4 + 1, "beat": i % 4 + 1, "confidence": 1.0}
             for i in range(bar_count * 4)]
    bars = [{"bar": b + 1, "start": b * BAR, "end": (b + 1) * BAR, "loudness": -12.0} for b in range(bar_count)]
    return beats, bars


class SectionTests(unittest.TestCase):
    def test_abab_gives_boundaries_on_the_section_starts_and_repeating_letters(self):
        beats, bars = grid(32)
        block, novelty = sections.analyze_sections(form_audio("ABAB"), SR, beats, bars)
        found = [(s["startBar"], s["endBar"], s["label"]) for s in block["sections"]]
        self.assertEqual(found, [(1, 9, "A"), (9, 17, "B"), (17, 25, "A"), (25, 33, "B")])
        self.assertEqual(block["boundariesSeconds"], [16.0, 32.0, 48.0])
        self.assertEqual((block["tool"], block["version"]), ("ssm", 1))
        self.assertEqual(len(novelty), 32)
        self.assertEqual(novelty[0], 0)
        self.assertTrue(all(0 <= v <= 1 for v in novelty))

    def test_three_letters_and_a_returning_first(self):
        beats, bars = grid(32)
        block, _ = sections.analyze_sections(form_audio("ABBA"), SR, beats, bars)
        self.assertEqual([s["label"] for s in block["sections"]][0], "A")
        self.assertEqual(block["sections"][-1]["label"], "A")

    def test_sections_cover_every_bar_contiguously(self):
        beats, bars = grid(24)
        block, _ = sections.analyze_sections(form_audio("ABA"), SR, beats, bars)
        spans = [(s["startBar"], s["endBar"]) for s in block["sections"]]
        self.assertEqual(spans[0][0], 1)
        self.assertEqual(spans[-1][1], 25)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))

    def test_a_quiet_tail_joins_the_last_section(self):
        beats, bars = grid(24)
        for bar in bars[-3:]:
            bar["loudness"] = -60.0
        block, _ = sections.analyze_sections(form_audio("ABA"), SR, beats, bars)
        self.assertEqual(block["sections"][-1]["endBar"], 25)
        self.assertTrue(all(s["startBar"] <= 22 for s in block["sections"]))

    def test_phrase_offset_follows_the_novelty(self):
        score = np.zeros(16)
        score[[2, 6, 10]] = 1
        self.assertEqual(sections.phrase_offset(score), 2)


class ScoreTests(unittest.TestCase):
    def test_scores_boundaries_and_labels(self):
        truth = {"sections": [{"startBar": 1, "endBar": 9, "label": "A"}, {"startBar": 9, "endBar": 17, "label": "B"},
                              {"startBar": 17, "endBar": 25, "label": "A"}]}
        early = {"sections": {"methods": {"x": {"sections": [
            {"startBar": 1, "endBar": 8, "label": "A"}, {"startBar": 8, "endBar": 17, "label": "B"},
            {"startBar": 17, "endBar": 25, "label": "C"}]}}}}
        result = score_sections.score(early, truth)["x"]
        self.assertEqual((result["exact"], result["withinOneBar"], result["sections"]), (0.5, 1.0, 3))
        self.assertLess(result["labelPairwiseF"], 1.0)
        self.assertEqual(score_sections.score({"sections": {"methods": {"t": truth}}}, truth)["t"]["labelPairwiseF"], 1.0)


class SectionStageTests(unittest.TestCase):
    def test_cli_stage_adds_novelty_and_the_ssm_method(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(analyze, "extract_beats", side_effect=detected), \
                patch.object(stems, "separate", side_effect=fake_separate):
            source, out = Path(tmp) / "a.wav", Path(tmp) / "analysis.json"
            sf.write(source, click_track(detected()[0]), RATE, subtype="PCM_16")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                reference.main([str(source), "-o", str(out), "--stages", "stems,sections"])
            doc = json.loads(out.read_text())
        self.assertEqual(list(doc["sections"]["methods"]), ["ssm"])
        found = doc["sections"]["methods"]["ssm"]["sections"]
        self.assertEqual(found[0]["startBar"], 1)
        self.assertEqual(found[-1]["endBar"], len(doc["grid"]["bars"]) + 1)
        self.assertTrue(all("novelty" in bar for bar in doc["grid"]["bars"]))


if __name__ == "__main__":
    unittest.main()
