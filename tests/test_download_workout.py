"""Workout downloader: playlist parsing and caption merging, no network."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import download_workout as d


class PlaylistTests(unittest.TestCase):
    def test_attributes_with_commas(self):
        self.assertEqual(d.attributes('TYPE=SUBTITLES,NAME="English, CC",LANGUAGE="en"'),
                         {'TYPE': 'SUBTITLES', 'NAME': 'English, CC', 'LANGUAGE': 'en'})

    def test_select_smallest_muxed_variant_and_resolve_urls(self):
        text = ('#EXTM3U\n#EXT-X-MEDIA:TYPE=SUBTITLES,NAME="English",LANGUAGE="en",URI="subs.m3u8"\n'
                '#EXT-X-STREAM-INF:BANDWIDTH=2000\nlarge.m3u8\n'
                '#EXT-X-STREAM-INF:BANDWIDTH=1000\nsmall.m3u8\n')
        audio, subs = d.parse_master(text, 'https://example.test/video/master.m3u8')
        self.assertEqual(audio, 'https://example.test/video/small.m3u8')
        self.assertEqual(subs[0]['url'], 'https://example.test/video/subs.m3u8')

    def test_prefer_separate_audio_group(self):
        text = ('#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",URI="audio.m3u8"\n'
                '#EXT-X-STREAM-INF:BANDWIDTH=1000,AUDIO="a"\nvideo.m3u8\n')
        audio, _ = d.parse_master(text, 'https://example.test/master.m3u8')
        self.assertEqual(audio, 'https://example.test/audio.m3u8')


class CaptionTests(unittest.TestCase):
    def test_deduplicate_overlap_without_losing_repeated_speech(self):
        first = ('WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000\n\n'
                 '1\n00:00:28.000 --> 00:00:32.000\nGo!\n')
        second = (first + '\n2\n00:00:35.000 --> 00:00:36.000\nGo!\n')
        cues = d.merge_captions([first, second])
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0][:2], (28000, 32000))

    def test_changing_timestamp_map_fails_clearly(self):
        cue = '\n\n00:00:00.000 --> 00:00:01.000\nHello\n'
        with self.assertRaises(d.DownloadError):
            d.merge_captions(['WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:0' + cue,
                              'WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:90000' + cue])

    def test_time_roundtrip_including_minutes_format(self):
        self.assertEqual(d.timestamp_ms('43:05.466'), 2585466)
        self.assertEqual(d.format_time(2585466, ','), '00:43:05,466')


if __name__ == '__main__':
    unittest.main()
