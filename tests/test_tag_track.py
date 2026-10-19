"""audio.m4a tags: written from metadata.json and the artwork, the audio untouched, nothing written when up to date."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mutagen.mp4 import MP4

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import tag_track

GENRES = {"house": "House", "tech-house": "Tech House"}
META = {
    "hash": "0123456789ab", "title": "The Rush", "artist": "Pain Cave", "source": "SUNO", "sourceId": "suno-id",
    "genres": ["house", "tech-house"], "bpm": 127.98, "durationSeconds": 2, "key": None,
    "createdAt": "2026-01-22T20:26:55Z", "updatedAt": "2026-01-22T20:26:55Z",
}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


class TagTrackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name) / "0123456789ab"
        self.dir.mkdir()
        self.audio = self.dir / "audio.m4a"
        ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-b:a", "128k",
               "-metadata", "comment=made with suno", "-movflags", "+faststart", str(self.audio))
        self.write_meta(META)

    def tearDown(self):
        self.temp.cleanup()

    def write_meta(self, meta: dict) -> None:
        (self.dir / "metadata.json").write_text(json.dumps(meta))

    def tags(self):
        return MP4(self.audio).tags

    def test_writes_the_tags_and_keeps_the_audio(self):
        before = tag_track.audio_identity(self.audio)
        self.assertTrue(tag_track.tag_track(self.dir, GENRES))
        tags = self.tags()
        self.assertEqual(tags["\xa9nam"], ["The Rush"])
        self.assertEqual(tags["\xa9ART"], ["Pain Cave"])
        self.assertEqual(tags["\xa9gen"], ["House", "Tech House"])
        self.assertEqual(tags["tmpo"], [128])
        self.assertEqual(tags["\xa9day"], ["2026-01-22"])
        self.assertEqual(bytes(tags["----:com.paincave:BPM"][0]), b"127.98")
        self.assertEqual(bytes(tags["----:com.paincave:HASH"][0]), b"0123456789ab")
        self.assertEqual(bytes(tags["----:com.paincave:SOURCE"][0]), b"SUNO")
        self.assertEqual(bytes(tags["----:com.paincave:SOURCE_ID"][0]), b"suno-id")
        self.assertEqual(tags["\xa9cmt"], ["made with suno"])
        self.assertNotIn("covr", tags)
        self.assertNotIn("\xa9alb", tags)
        self.assertEqual(tag_track.audio_identity(self.audio), before)
        self.assertEqual(list(self.dir.glob(".audio.m4a.tagging-*")), [])

    def test_artwork_and_class_album(self):
        ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(self.dir / "thumb.avif"))
        self.write_meta({**META, "source": "CLASS", "cutFrom": "THE TRIP 43"})
        tag_track.tag_track(self.dir, GENRES)
        tags = self.tags()
        self.assertEqual(tags["\xa9alb"], ["THE TRIP 43"])
        self.assertEqual(tags["covr"][0].imageformat, tags["covr"][0].FORMAT_JPEG)
        self.assertTrue(bytes(tags["covr"][0]).startswith(b"\xff\xd8"))
        red = bytes(tags["covr"][0])
        # New artwork is told by its own fingerprint, not by converting it again
        ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", str(self.dir / "thumb.avif"))
        self.assertFalse(tag_track.is_tagged(self.audio, tag_track.wanted_tags(self.dir, GENRES)))
        self.assertTrue(tag_track.tag_track(self.dir, GENRES))
        self.assertNotEqual(bytes(self.tags()["covr"][0]), red)
        (self.dir / "thumb.avif").unlink()
        self.assertTrue(tag_track.tag_track(self.dir, GENRES))
        self.assertNotIn("covr", self.tags())
        self.assertNotIn("----:com.paincave:COVER", self.tags())

    def test_sigterm_leaves_no_copy(self):
        stalled = (
            "import sys, time; src, marker, track, genres = sys.argv[1:]; sys.path.insert(0, src); import tag_track; "
            "tag_track.audio_identity = lambda path: (open(marker, 'w').close(), time.sleep(30)); "
            "sys.argv = ['tag_track.py', track, '--genres', genres]; sys.exit(tag_track.main())"
        )
        genres = Path(self.temp.name) / "genres.json"
        genres.write_text(json.dumps([{"slug": "house", "name": "House"}]))
        marker = Path(self.temp.name) / "verifying"
        original = self.audio.read_bytes()
        child = subprocess.Popen([sys.executable, "-c", stalled, str(SRC), str(marker), str(self.dir), str(genres)])
        for _ in range(200):
            if marker.exists():
                break
            time.sleep(0.05)
        self.assertTrue(marker.exists())
        child.terminate()
        self.assertEqual(child.wait(10), 128 + 15)
        self.assertEqual(list(self.dir.glob(".audio.m4a.tagging-*")), [])
        self.assertEqual(self.audio.read_bytes(), original)

    def test_up_to_date_writes_nothing(self):
        ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(self.dir / "thumb.avif"))
        self.assertTrue(tag_track.tag_track(self.dir, GENRES))
        written = self.audio.stat().st_mtime_ns
        self.assertFalse(tag_track.tag_track(self.dir, GENRES))
        self.assertEqual(self.audio.stat().st_mtime_ns, written)

    def test_a_cleared_field_removes_its_tag(self):
        tag_track.tag_track(self.dir, GENRES)
        self.write_meta({**META, "genres": [], "title": "Renamed"})
        self.assertTrue(tag_track.tag_track(self.dir, GENRES))
        self.assertNotIn("\xa9gen", self.tags())
        self.assertEqual(self.tags()["\xa9nam"], ["Renamed"])

    def test_check_lists_missing_and_stale_tags(self):
        wanted = lambda: tag_track.wanted_tags(self.dir, GENRES)
        self.assertFalse(tag_track.is_tagged(self.audio, wanted()))
        tag_track.tag_track(self.dir, GENRES)
        self.assertTrue(tag_track.is_tagged(self.audio, wanted()))
        self.write_meta({**META, "bpm": 130.0})
        self.assertFalse(tag_track.is_tagged(self.audio, wanted()))

    def test_refuses_a_folder_that_isnt_a_track(self):
        (self.dir / "metadata.json").unlink()
        with self.assertRaises(ValueError):
            tag_track.tag_track(self.dir, GENRES)

    def test_a_changed_audio_leaves_the_original(self):
        original = self.audio.read_bytes()
        identities = iter([("a", "2.0"), ("b", "2.0")])
        with patch.object(tag_track, "audio_identity", side_effect=lambda _: next(identities)):
            with self.assertRaises(RuntimeError):
                tag_track.tag_track(self.dir, GENRES)
        self.assertEqual(self.audio.read_bytes(), original)
        self.assertEqual(list(self.dir.glob(".audio.m4a.tagging-*")), [])


if __name__ == "__main__":
    unittest.main()
