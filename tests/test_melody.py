"""The melody stage: notes from a pitch track by rule, and onto the grid's beats (CREPE itself isn't run here)."""
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import melody
import reference


def hz(midi: float) -> float:
    return 440 * 2 ** ((midi - 69) / 12)


def track(*segments: tuple[int, float | None, float]):
    """Pitch, periodicity and level arrays from (frames, MIDI note or None for silence, level dB) segments."""
    pitch, periodicity, levels = [], [], []
    for frames, note, level in segments:
        pitch += [hz(note) if note is not None else 0.0] * frames
        periodicity += [0.9 if note is not None else 0.1] * frames
        levels += [level] * frames
    return np.array(pitch), np.array(periodicity), np.array(levels, dtype=float)


def summary(found):
    return [(start, end, note) for start, end, note, _, _ in found]


class NoteTests(unittest.TestCase):
    def test_held_notes_split_where_the_pitch_moves_and_stays(self):
        found = melody.notes(*track((30, 64, -6), (40, 67, -6), (20, None, -60)))
        self.assertEqual(summary(found), [(0, 30, 64), (30, 70, 67)])

    def test_vibrato_and_a_short_scoop_stay_one_note(self):
        wobble = 69 + 0.5 * np.sin(np.arange(60) / 3)
        pitch, periodicity, levels = track((3, 66, -6), (60, 69, -6))
        pitch[3:] = hz(wobble)
        self.assertEqual(summary(melody.notes(pitch, periodicity, levels)), [(0, 63, 69)])

    def test_a_resung_note_on_the_same_pitch_is_two_notes(self):
        found = melody.notes(*track((20, 62, -6), (10, 62, -16), (20, 62, -4)))
        self.assertEqual(summary(found), [(0, 30, 62), (30, 50, 62)])

    def test_the_same_pitch_across_a_breath_is_one_note(self):
        found = melody.notes(*track((20, 60, -6), (3, None, -40), (20, 60, -6)))
        self.assertEqual(summary(found), [(0, 43, 60)])

    def test_bleed_under_the_gate_and_blips_are_dropped(self):
        found = melody.notes(*track((30, 64, 0), (40, 71, -50), (4, 72, -6), (30, None, -60)))
        self.assertEqual(summary(found), [(0, 30, 64)])

    def test_level_is_below_the_loudest_frame(self):
        found = melody.notes(*track((20, 60, 0), (20, None, -60), (20, 64, -12)))
        self.assertEqual([round(n[4], 1) for n in found], [0.0, -12.0])


class BeatTests(unittest.TestCase):
    BEATS = [{"time": 0.5 + 0.5 * i} for i in range(12)]  # 120 BPM; bar 1's downbeat is the 3rd beat (1.5 s)

    def test_seconds_to_beats_from_bar_one(self):
        position = melody.beat_position(np.array([1.5, 1.75, 3.5, 0.0, 7.0]), self.BEATS, 2)
        self.assertEqual(position.round(3).tolist(), [0.0, 0.5, 4.0, -3.0, 11.0])

    def test_block_in_beats(self):
        found = [(150, 200, 64, 0.876, -7.04)]  # 1.5 s to 2.0 s
        b = melody.block(found, self.BEATS, 2)
        self.assertEqual((b["tool"], b["model"]), ("crepe", "full"))
        self.assertEqual(b["notes"], [{"start": 0.0, "length": 1.0, "pitch": 64, "confidence": 0.88, "level": -7.0}])


class StageTests(unittest.TestCase):
    def test_melody_brings_the_stems(self):
        self.assertEqual(reference.parse_stages("melody"), {"grid", "stems", "melody"})

    def test_a_cached_track_never_runs_crepe(self):
        pitch, periodicity, _ = track((50, 64, 0), (50, 67, 0))
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "melody.npz"
            np.savez(cache, pitch=pitch, periodicity=periodicity)
            vocals = np.full(441 * 100, 0.3, dtype=np.float32)  # 1 s of steady level at 44.1 kHz
            with unittest.mock.patch.dict(sys.modules, {"torchcrepe": None}):
                b = melody.melody(vocals, 44100, [{"time": 0.5 * i} for i in range(4)], 0, cache)
        self.assertEqual([(n["start"], n["pitch"]) for n in b["notes"]], [(0.0, 64), (1.0, 67)])


if __name__ == "__main__":
    unittest.main()
