"""Deterministic pipeline tests: real decoding/encoding, stubbed expensive inference."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyze
import content_hash
import import_track
import pcav

# Supports a staged checkout while the schemas stay in the main workspace.
import_track.SCHEMA_DIR = Path(
    os.environ.get("PAINCAVE_SCHEMA_DIR", Path(__file__).resolve().parents[2] / "paincave/shared/schemas"))

# The CLI in a child process, stalled inside analysis (stage populated) in a grandchild that records its pid.
STALLED_IMPORT = """
import subprocess, sys
from pathlib import Path
from unittest.mock import patch
src, schemas, marker, *argv = sys.argv[1:]
sys.path.insert(0, src)
import analyze, import_track
import_track.SCHEMA_DIR = Path(schemas)
stall = lambda *a, **k: subprocess.run(['/bin/sh', '-c', 'echo $$ > "$0.tmp" && mv "$0.tmp" "$0" && exec sleep 60', marker], stderr=subprocess.DEVNULL)
with patch.object(analyze, 'extract_beats', side_effect=stall), patch.object(sys, 'argv', ['import_track.py', *argv]):
    import_track.main()
"""


def detected(*args, **kwargs):
    """Detections on a 120 BPM half-note grid with downbeats on 1.75 s; the fitted grid reaches 4.75 s."""
    return np.arange(0.25, 4.75, 0.5), np.array([1.75, 3.75])


# A different, denser grid: 180 BPM from 0 s, five bars over the fixture, all of them work.
ZONED_ZONES = [{'startBar': 1, 'durationBars': 5, 'startSeconds': 0.0, 'durationSeconds': 6.0}]


def zoned(*args, **kwargs):
    times = np.arange(0, 6, 1 / 3)
    return times, times[::4]


def altered(target, change):
    """Wrap a real function and mutate its result, to inject one inconsistent value."""

    def run(*args, **kwargs):
        result = target(*args, **kwargs)
        change(result)
        return result

    return run


class PcavTests(unittest.TestCase):
    def fixture(self, rate=44100):
        samples = np.sin(np.arange(rate * 10 + 1) * 0.03)
        peaks = pcav.build_peak_mipmaps(samples, rate)
        return pcav.PCAVData(pcav.AnalysisSection(len(samples) / rate, 123.4, None),
                             pcav.BeatsSection([0.123456 + i * 60 / 123.4 for i in range(20)], 3), peaks[0], peaks[1])

    def test_phase_duration_timebase_and_no_drift(self):
        for rate in (44100, 48000):
            data = self.fixture(rate)
            parsed = pcav.parse_pcav_payload(pcav.build_pcav_payload(data))
            self.assertEqual(parsed.beats.first_downbeat_index, 3)
            self.assertEqual(parsed.analysis.duration_seconds, data.analysis.duration_seconds)
            self.assertEqual(parsed.peaks.sample_rate, rate)
            self.assertEqual(len(parsed.peaks.peaks), (rate * 10 + 256) // 256)
            for before, after in zip(data.beats.beat_times, parsed.beats.beat_times):
                self.assertLessEqual(abs(before - after), 0.0000005)

    def test_long_intro_large_gaps(self):
        section = pcav.BeatsSection([70.0, 80.5, 100.123456], 1, 3)
        decoded = pcav.decode_beats(pcav.encode_beats(section))
        self.assertEqual(decoded, section)

    def test_every_truncation_rejected(self):
        data = pcav.build_pcav_payload(self.fixture())
        for end in range(len(data)):
            with self.assertRaises(ValueError, msg=f"truncation {end}"):
                pcav.parse_pcav_payload(data[:end])

    def test_bad_counts_overlap_duplicate_and_events(self):
        base = bytearray(pcav.build_pcav_payload(self.fixture()))
        for offset, value in ((7, 0), (16, 6), (11, 2 ** 32 - 1)):
            data = base.copy()
            struct.pack_into('<I', data, offset, value)
            with self.assertRaises(ValueError): pcav.parse_pcav_payload(data)
        duplicate = base.copy();
        duplicate[15] = duplicate[6]
        with self.assertRaises(ValueError):
            pcav.parse_pcav_payload(duplicate)
        for times in ([1, 1], [2, 1], [-1], [float('nan')], [float('inf')], [0, 0.0000001]):
            with self.assertRaises(ValueError): pcav.encode_beats(pcav.BeatsSection(times))
        with self.assertRaises(ValueError):
            pcav.decode_peaks(struct.pack('<III', 2 ** 32 - 1, 44100, 256))

    def test_short_and_silent_peaks_include_tail(self):
        for size in (1, 2, 15, 255, 256, 257):
            for samples in (np.zeros(size), np.ones(size)):
                result = pcav.build_peak_mipmaps(samples, 48000)
                self.assertEqual(len(result[0].peaks), (size + 255) // 256)
        for samples in ([], [float('nan')]):
            with self.assertRaises(ValueError): pcav.build_peak_mipmaps(samples, 48000)

    def test_v1_both_peak_headers(self):
        for data, expected in ((struct.pack('<Hbb', 1, -128, 127), 4800), (struct.pack('<Ibb', 1, -127, 127), 256)):
            section = pcav.decode_peaks(data, 1)
            self.assertEqual(section.window_samples, expected)
        self.assertEqual(pcav.decode_beats(struct.pack('<HHBB', 200, 3, 50, 51), 1).beat_times, [0.2, 0.7, 1.21])
        with self.assertRaises(ValueError): pcav.decode_beats(struct.pack('<HHB', 200, 3, 50), 1)


class AnalyzerTests(unittest.TestCase):
    def test_pickup_numbers_match_python_and_ts_contract(self):
        beats = analyze.build_beats_array(np.arange(10, dtype=float), 7)
        self.assertEqual([b['beat'] for b in beats[:7]], [2, 3, 4, 1, 2, 3, 4])
        self.assertEqual(beats[7]['bar'], 1)
        self.assertEqual(beats[7]['beat'], 1)


class GridFitTests(unittest.TestCase):
    """The detector is silent in quiet passages and quantizes to 20 ms; plans need an unbroken grid."""

    PERIOD = 60 / 137.3

    def detections(self, count=200, start=1.0, period=None, quantize=0.02):
        times = start + np.arange(count) * (period or self.PERIOD)
        return np.round(times / quantize) * quantize if quantize else times

    def test_gap_is_filled_with_inferred_beats(self):
        beats = np.concatenate([self.detections(24), self.detections(40, start=1.0 + 44 * self.PERIOD)])
        fit = analyze.fit_beat_grid(beats, beats[::4], 60.0)
        spacing = np.diff(fit.beat_times)
        self.assertLess(spacing.max() - spacing.min(), 1e-9)
        self.assertLess(fit.beat_times[0], fit.period)          # the grid reaches back before the first detection
        self.assertGreater(fit.beat_times[-1], 60 - fit.period)  # and on past the last one
        silent = (fit.beat_times > beats[23] + 0.1) & (fit.beat_times < beats[24] - 0.1)
        self.assertEqual(silent.sum(), 20)  # the 20 beat slots the detector heard nothing on
        self.assertFalse(fit.heard[silent].any())
        played = (fit.beat_times >= beats[0] - 0.001) & (fit.beat_times <= beats[23] + 0.001)
        self.assertTrue(fit.heard[played].all())

    def test_quantized_detections_recover_the_true_period(self):
        fit = analyze.fit_beat_grid(self.detections(), self.detections()[::4], 120.0)
        self.assertLess(abs(fit.period - self.PERIOD) * 1000, 0.05)
        self.assertAlmostEqual(fit.bpm, 137.3, places=2)
        self.assertLess(fit.residual_ms, 12)
        self.assertGreater(fit.beat_times[0], 0)
        self.assertLess(fit.beat_times[0], self.PERIOD)

    def test_half_tempo_detections_are_octave_corrected(self):
        beats = self.detections(120, period=2 * self.PERIOD)
        fit = analyze.fit_beat_grid(beats, beats[::2], 120.0, min_bpm=100, max_bpm=180)
        self.assertAlmostEqual(fit.bpm, 137.3, places=1)
        self.assertEqual(fit.heard.sum(), len(beats))
        self.assertEqual((~fit.heard).sum(), len(fit.beat_times) - len(beats))

    def test_downbeat_vote_sets_bar_phase_and_first_bar(self):
        beats = self.detections()
        for offset in range(4):
            fit = analyze.fit_beat_grid(beats, beats[offset::4], 120.0)
            first = analyze.first_bar_time(fit)
            self.assertEqual(fit.bar_phase, int(fit.index[np.argmin(np.abs(fit.beat_times - beats[offset]))] % 4))
            self.assertAlmostEqual(first, beats[offset], delta=0.021)
            numbered = analyze.build_beats_array(fit.beat_times, first)
            self.assertEqual([b['bar'] for b in numbered if b['t'] == round(first, 3)], [1])
        # Without downbeats, bar 1 falls on the first beat the detector actually heard.
        self.assertAlmostEqual(analyze.first_bar_time(analyze.fit_beat_grid(beats, [], 120.0)), beats[0], delta=0.021)

    def test_too_few_or_scattered_detections_have_no_grid(self):
        for beats, message in (
                (self.detections(5), 'usable beat detections'),
                (np.arange(20) * 0.05, 'far outside 100-180 BPM'),
                (np.cumsum(np.concatenate([[1.0], np.tile([0.30, 0.62], 30)])), 'scatter'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(analyze.NoBeatGrid, message):
                analyze.fit_beat_grid(beats, beats[::4], 120.0)

    def test_same_detections_give_the_same_grid(self):
        beats, downbeats = self.detections(), self.detections()[::4]
        first, second = (analyze.fit_beat_grid(beats, downbeats, 120.0) for _ in range(2))
        self.assertEqual((first.period, first.phase, first.bar_phase, first.residual_ms),
                         (second.period, second.phase, second.bar_phase, second.residual_ms))
        self.assertTrue(np.array_equal(first.beat_times, second.beat_times))
        self.assertTrue(np.array_equal(first.heard, second.heard))

    @unittest.skipUnless(analyze.models.checkpoint_path().is_file(), "checkpoint not cached")
    def test_detector_smoke(self):
        sr = 22050
        click = np.zeros(sr * 12)
        for i in range(0, len(click), int(sr * 0.5)):
            click[i:i + 200] = np.sin(np.arange(200) * 0.5) * 0.9
        beats, downbeats = analyze.extract_beats(click, sr)
        self.assertGreater(len(beats), 8)
        self.assertAlmostEqual(analyze.fit_beat_grid(beats, downbeats, 12.0, 100, 180).bpm, 120, delta=1)

    def test_energy_covers_final_second_and_complete_bar(self):
        self.assertEqual(len(analyze.compute_band_energy(analyze.band_pass(np.ones(48001), 48000), 48000)), 2)
        beats = [{'t': 0, 'bar': 1, 'beat': 1}, {'t': 2, 'bar': 2, 'beat': 1}]
        density, info = analyze.compute_kick_density(np.array([0., 1., 0.]), beats, 3)
        self.assertEqual(list(density), [0.5, 0])
        self.assertEqual(info[0]['end_t'], 2)
        self.assertEqual(len(analyze.compute_kick_density(np.array([]), beats)[0]), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "Original song.wav"
        sf.write(self.source, np.sin(np.arange(48000 * 6) * 0.01) * 0.3, 48000, subtype='PCM_24')
        self.library = self.root / "library"
        self.beats_patch = patch.object(analyze, 'extract_beats', side_effect=detected)
        self.beats_patch.start()
        self.addCleanup(self.beats_patch.stop)

    def run_import(self, **kwargs):
        return import_track.import_song(self.source, self.library, **kwargs)

    def snapshot(self, path):
        return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}

    def active(self, folder):
        """Checksum, inode and mtime of every active file; candidate revisions excluded."""
        return {str(p.relative_to(folder)): (import_track.sha256_file(p), p.stat().st_ino, p.stat().st_mtime_ns)
                for p in folder.rglob('*') if p.is_file() and p.relative_to(folder).parts[0] != 'analyses'}

    def assertNothingPublished(self, *tracks):
        self.assertEqual(sorted(p.name for p in self.library.iterdir()), sorted(tracks))
        for h in tracks: self.assertFalse((self.library / h / 'analyses').exists())

    def test_real_encode_import_duplicate_and_safe_reanalysis(self):
        original = self.source.read_bytes()
        h = self.run_import()
        folder = self.library / h
        (folder / 'plans').mkdir()
        (folder / 'plans/authored.json').write_text('authored')
        (folder / 'thumb.webp').write_bytes(b'keep')
        old = self.active(folder)
        self.assertEqual(self.run_import(), h)
        self.assertEqual(self.active(folder), old)
        # The candidate gets a different grid and zones, so an overwrite cannot pass as preservation.
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            self.assertEqual(import_track.import_song(folder / 'original.wav', self.library, force=True), h)
        self.assertEqual(self.active(folder), old)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual([p.name for p in self.library.iterdir()], [h])
        revisions = list((folder / 'analyses').iterdir())
        self.assertEqual(len(revisions), 1)
        revision = revisions[0]
        for name in ('analysis.pcav', 'analysis.json', 'plan_workzones.json'):
            self.assertNotEqual(import_track.sha256_file(folder / name), import_track.sha256_file(revision / name),
                                name)
        for directory, downbeat_index, zones in ((folder, 3, []), (revision, 0, ZONED_ZONES)):
            parsed = pcav.read_pcav_file(directory / 'analysis.pcav')
            self.assertEqual(parsed.beats.first_downbeat_index, downbeat_index)
            self.assertEqual(json.loads((directory / 'plan_workzones.json').read_text())['zones'], zones)
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(manifest['master']['frames'], 288000)
            self.assertFalse(manifest['loudness']['normalizationApplied'])
            self.assertEqual(json.loads((directory / 'metadata.json').read_text())['title'], 'Original song')
            for name, record in manifest['artifacts'].items():
                self.assertEqual(import_track.sha256_file(directory / name), record['sha256'])

    def test_replace_swaps_active_analysis_and_keeps_artwork_plans_and_candidates(self):
        h = self.run_import()
        folder = self.library / h
        (folder / 'plans').mkdir()
        (folder / 'plans/authored.json').write_text('authored')
        (folder / 'thumb.avif').write_bytes(b'keep')
        (folder / 'thumbnail.json').write_text('{}')
        created = json.loads((folder / 'metadata.json').read_text())['createdAt']
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            self.cli(folder / 'original.wav', '--force')
        with self.assertRaisesRegex(ValueError, '--replace requires --force'):
            import_track.import_song(folder / 'original.wav', self.library, replace=True)
        old = self.active(folder)
        # A failed swap puts the old folder and its kept files back; nothing is left in the library root.
        real_rename = Path.rename

        def failing(self_, target):
            if self_.name == 'track': raise OSError('swap failed')
            return real_rename(self_, target)

        with patch.object(analyze, 'extract_beats', side_effect=zoned), patch.object(Path, 'rename', failing), \
                self.assertRaisesRegex(OSError, 'swap failed'):
            self.run_import(force=True, replace=True)
        self.assertEqual(self.active(folder), old)
        self.assertEqual(sorted(p.name for p in self.library.iterdir()), [h])
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            out = self.cli(folder / 'original.wav', '--force', '--replace', '--title', 'Renamed')
        self.assertRegex(out, f'(?m)^Replaced: .*{h} \\(active analysis updated')
        self.assertNotRegex(out, '(?m)^Revision:')
        self.assertEqual(out.splitlines()[-1], f'Hash: {h}')
        self.assertEqual(pcav.read_pcav_file(folder / 'analysis.pcav').beats.first_downbeat_index, 0)
        self.assertEqual(json.loads((folder / 'plan_workzones.json').read_text())['zones'], ZONED_ZONES)
        meta = json.loads((folder / 'metadata.json').read_text())
        self.assertEqual((meta['title'], meta['createdAt']), ('Renamed', created))
        self.assertEqual((folder / 'thumb.avif').read_bytes(), b'keep')
        self.assertEqual((folder / 'thumbnail.json').read_text(), '{}')
        self.assertEqual((folder / 'plans/authored.json').read_text(), 'authored')
        self.assertEqual(len(list((folder / 'analyses').iterdir())), 1)
        self.assertEqual(sorted(p.name for p in self.library.iterdir()), [h])
        self.assertTrue((folder / 'manifest.json').is_file())
        self.assertNotIn('retired', [p.name for p in folder.iterdir()])

    def cli(self, *argv):
        out = io.StringIO()
        with patch.object(sys, 'argv',
                          ['import_track.py', *map(str, argv), '--library', str(self.library)]), redirect_stdout(out):
            import_track.main()
        return out.getvalue()

    def test_cli_stdout_markers_consumers_parse(self):
        # bin/import.sh reads the last "^Hash: " and "^Track already exists:"; the server route reads the first "Hash: ".
        first = self.cli(self.source)
        h = re.search(r'Hash: ([a-f0-9]+)', first).group(1)
        repeat, forced = self.cli(self.source), self.cli(self.library / h / 'original.wav', '--force')
        for output in (first, repeat, forced):
            self.assertEqual(output.splitlines()[-1], f'Hash: {h}')
            self.assertEqual(re.search(r'Hash: ([a-f0-9]+)', output).group(1), h)
        self.assertNotRegex(first, '(?m)^(Track already exists|Revision):')
        self.assertRegex(repeat, f'(?m)^Track already exists: .*{h}$')
        # import.sh skips the paid thumbnail step on "^Revision:" and reports the path before " (candidate".
        revision = re.search(r'(?m)^Revision: (.+/analyses/[0-9a-f]{64}) \(candidate; ', forced)
        self.assertIsNotNone(revision, forced)
        self.assertEqual(Path(revision.group(1)), next((self.library / h / 'analyses').iterdir()).resolve())

    def test_legacy_master_without_manifest_needs_matching_identity(self):
        encoded = self.root / 'download.m4a'
        import_track.convert_to_m4a(self.source, encoded)
        h = import_track.import_song(encoded, self.library)
        folder = self.library / h
        # A pre-manifest track converted from a compressed download: its master hashes to another key.
        (folder / 'manifest.json').unlink()
        master = folder / 'original.wav'
        self.assertNotEqual(import_track.compute_content_hash(master), h)
        (folder / 'nested').mkdir()
        shutil.copy2(master, folder / 'nested/Original copy.WAV')
        old = self.snapshot(self.library)
        for audio in (master, folder / 'nested/Original copy.WAV', folder / 'audio.m4a'):
            for force in (True, False):
                with self.subTest(audio=audio.name, force=force), \
                        self.assertRaisesRegex(ValueError,
                                               f'Unverifiable legacy identity: .* not its track directory {h}'):
                    import_track.import_song(audio, self.library, force=force)
                self.assertEqual(self.snapshot(self.library), old)
                self.assertNothingPublished(h)

    def test_audio_inside_track_folder_keeps_identity_or_is_refused(self):
        encoded = self.root / 'download.m4a'
        import_track.convert_to_m4a(self.source, encoded)
        h = import_track.import_song(encoded, self.library)
        folder = self.library / h
        (folder / 'nested').mkdir()
        copy = folder / 'nested/Original copy.WAV'
        shutil.copy2(folder / 'original.wav', copy)
        self.assertNotEqual(import_track.compute_content_hash(copy), h)
        old = self.active(folder)
        # A renamed byte copy of the master, the retained download, and a case-variant path resolve to the track.
        case_variant = self.library / h.upper() / 'ORIGINAL.WAV'
        for audio in (copy, folder / 'source.m4a', *([case_variant] if case_variant.exists() else [])):
            self.assertEqual(import_track.import_song(audio, self.library), h)
        self.assertNothingPublished(h)
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            self.assertEqual(import_track.import_song(copy, self.library, force=True), h)
        revision = next((folder / 'analyses').iterdir())
        self.assertEqual(import_track.import_song(revision / 'original.wav', self.library, force=True), h)
        self.assertEqual(len(list((folder / 'analyses').iterdir())), 2)
        self.assertEqual([p.name for p in self.library.iterdir()], [h])
        self.assertEqual(self.active(folder), old)
        before = self.snapshot(self.library)
        # A rendition is neither the master nor the fingerprinted audio: refused, not minted as a new track.
        for audio in (folder / 'audio.m4a', revision / 'audio.m4a'):
            for force in (True, False):
                with self.subTest(audio=str(audio.relative_to(folder)), force=force), \
                        self.assertRaisesRegex(ValueError, f'Retained master checksum mismatch: .* inside track {h} '):
                    import_track.import_song(audio, self.library, force=force)
                self.assertEqual(self.snapshot(self.library), before)

    def test_library_subfolder_that_is_not_a_track_holds_plain_inputs(self):
        # The server stages uploads in {library}/import/; a file there named original.wav is an ordinary import.
        staged = self.library / 'import/original.wav'
        staged.parent.mkdir(parents=True)
        shutil.copy2(self.source, staged)
        h = import_track.import_song(staged, self.library)
        self.assertEqual(h, import_track.compute_content_hash(self.source))
        self.assertEqual(sorted(p.name for p in self.library.iterdir()), sorted([h, 'import']))
        self.assertEqual((self.library / h / 'original.wav').read_bytes(), self.source.read_bytes())

    def test_legacy_master_without_manifest_matching_identity_stages_candidate(self):
        h = self.run_import()
        folder = self.library / h
        (folder / 'manifest.json').unlink()
        old = self.active(folder)
        self.assertEqual(import_track.import_song(folder / 'original.wav', self.library), h)
        self.assertNothingPublished(h)
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            self.assertEqual(import_track.import_song(folder / 'original.wav', self.library, force=True), h)
        self.assertEqual(self.active(folder), old)
        self.assertEqual([p.name for p in self.library.iterdir()], [h])
        self.assertEqual(len(list((folder / 'analyses').iterdir())), 1)

    def test_retained_file_and_manifest_tampering_rejected(self):
        h = self.run_import()
        folder = self.library / h
        old = self.snapshot(folder)
        manifest = json.loads((folder / 'manifest.json').read_text())
        wrong_identity = {**manifest, 'audioFingerprint': {**manifest['audioFingerprint'], 'md5': 'f' * 32}}
        samples, rate = sf.read(folder / 'original.wav')
        tampered_master = io.BytesIO();
        sf.write(tampered_master, samples * 0.5, rate, format='WAV', subtype='PCM_24')
        cases = (
            ('manifest.json', json.dumps(wrong_identity).encode(), folder / 'original.wav',
             'Manifest identity does not match track directory'),
            ('original.wav', tampered_master.getvalue(), folder / 'original.wav', 'Retained master checksum mismatch'),
            ('source.wav', tampered_master.getvalue(), self.source, 'Retained source checksum mismatch'),
            ('source.wav', None, self.source, 'source.wav'),
        )
        for name, data, target, message in cases:
            with self.subTest(name=name, message=message):
                (folder / name).unlink()
                if data is not None: (folder / name).write_bytes(data)
                with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                    import_track.import_song(target, self.library, force=True)
                self.assertNothingPublished(h)
                (folder / name).write_bytes(old[name])
                self.assertEqual(self.snapshot(folder), old)

    def test_incomplete_existing_track_fails_explicitly(self):
        h = self.run_import()
        folder = self.library / h
        (folder / 'audio.m4a').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete existing track: audio.m4a missing'):
            self.run_import()
        for name in ('manifest.json', 'original.wav'): (folder / name).unlink()
        old = self.snapshot(folder)
        for force in (False, True):
            with self.assertRaisesRegex(ValueError, 'Incomplete existing track: retained original is missing'):
                self.run_import(force=force)
            self.assertEqual(self.snapshot(folder), old)
            self.assertNothingPublished(h)

    def test_derived_master_requires_processing_record(self):
        derived = self.root / 'processed.wav'
        samples, rate = sf.read(self.source)
        sf.write(derived, samples * 0.5, rate, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'A derived master requires --processing-record provenance'):
            import_track.import_song(derived, self.library, source_file=self.source)
        self.assertNothingPublished()

    def test_inconsistent_timebase_zones_and_playback_never_published(self):
        def shift(field, amount):
            return lambda a: a['workZones'][0].update({field: a['workZones'][0][field] + amount})

        cases = (
            ('analyze_audio', lambda a: a.update(duration=a['duration'] + 0.001),
             'Analysis timebase does not match the retained master'),
            ('analyze_audio', lambda a: a.update(sampleRate=44100),
             'Analysis timebase does not match the retained master'),
            ('analyze_audio', shift('startSeconds', 0.01), 'Work zone times disagree with the measured grid'),
            ('analyze_audio', shift('durationSeconds', -0.01), 'Work zone times disagree with the measured grid'),
            ('analyze_audio', shift('startBar', 1), 'Work zone lies outside the measured bar grid'),
            ('analyze_audio', shift('durationBars', 1), 'Work zone lies outside the measured bar grid'),
            ('probe_audio', lambda p: p.update(duration=str(float(p['duration']) + 0.06)),
             'Playback duration differs from analyzed master'),
        )
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            for force in (False, True):
                old = self.snapshot(self.library)
                for function, change, message in cases:
                    with self.subTest(force=force, message=message), \
                            patch.object(import_track, function,
                                         side_effect=altered(getattr(import_track, function), change)), \
                            self.assertRaisesRegex(ValueError, message):
                        self.run_import(force=force)
                    self.assertEqual(self.snapshot(self.library), old)
                    self.assertNothingPublished(*[p.name for p in self.library.iterdir()])
                h = self.run_import()
                self.assertEqual(json.loads((self.library / h / 'plan_workzones.json').read_text())['zones'],
                                 ZONED_ZONES)

    def test_beats_outside_duration_and_first_downbeat_snapping(self):
        for beats in ([0.5, 7.0], [-0.01, 0.5]):
            with patch.object(analyze, 'extract_beats', return_value=(np.array(beats), np.array([1, 2]))):
                with self.assertRaisesRegex(ValueError, 'Detected beats lie outside audio duration'):
                    self.run_import()
        for value in (6.0, -0.1, float('nan')):
            with self.assertRaisesRegex(ValueError, 'First downbeat must identify a beat within the audio'):
                self.run_import(first_downbeat=value)
        self.assertNothingPublished()
        folder = self.library / self.run_import(first_downbeat=1.3)
        self.assertEqual(json.loads((folder / 'analysis.json').read_text())['firstDownbeat'], 1.25)
        self.assertEqual(pcav.read_pcav_file(folder / 'analysis.pcav').beats.first_downbeat_index, 2)
        self.assertEqual(json.loads((folder / 'manifest.json').read_text())['config']['firstDownbeat'], 1.3)

    def test_failures_preserve_previous_and_never_publish_new(self):
        with self.assertRaises(Exception): self.run_import(title='')
        self.assertEqual(list(self.library.iterdir()), [])
        h = self.run_import()
        old = self.snapshot(self.library / h)
        for function in ('analyze_audio', 'write_pcav_file', 'finish_m4a_encode', 'probe_audio'):
            with patch.object(import_track, function, side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError): self.run_import(force=True)
            self.assertEqual(self.snapshot(self.library / h), old)
        with patch.object(Path, 'rename', side_effect=OSError('promotion failed')):
            with self.assertRaises(OSError): self.run_import(force=True)
        self.assertEqual(self.snapshot(self.library / h), old)
        self.assertFalse(list(self.library.glob('.staging-*')))

    def test_metadata_dates_and_nonfinite_rejected(self):
        with patch.object(import_track, 'extract_suno_metadata', return_value={'created': 'not-a-date'}):
            with self.assertRaisesRegex(ValueError, 'Invalid createdAt: expected RFC 3339 date-time'): self.run_import()
        for field in ('bpm', 'duration'):
            nonfinite = altered(import_track.analyze_audio, lambda a: a.update({field: float('nan')}))
            with patch.object(import_track, 'analyze_audio', side_effect=nonfinite):
                with self.assertRaisesRegex(ValueError, 'not JSON compliant'): self.run_import()
        self.assertNothingPublished()
        with self.assertRaisesRegex(ValueError, 'BPM bounds'): self.run_import(min_bpm=180, max_bpm=100)

    def test_refuses_short_silent_and_gridless_audio(self):
        with patch.object(analyze, 'extract_beats', return_value=(np.arange(7) * .5, np.arange(0, 7, 4) * .5)), \
                self.assertRaisesRegex(analyze.NoBeatGrid, '^no beat grid: 7 beats detected'):
            self.run_import()
        self.assertEqual(list(self.library.iterdir()), [])
        # Too short or silent: refused from the input decode, before anything is staged.
        for samples, rate, message in ((np.zeros(441000), 44100, 'audio is digital silence'),
                                       (np.sin(np.arange(12000) * .01), 48000, 'audio is 0.250 s long')):
            sf.write(self.source, samples, rate)
            with self.subTest(message=message), patch.object(import_track, 'staging_directory',
                                                             side_effect=AssertionError('staged')), \
                    self.assertRaisesRegex(analyze.NoBeatGrid, message):
                self.run_import()
        with self.assertRaises(SystemExit) as refused: self.cli(self.source)
        self.assertEqual(refused.exception.code,
                         'Error: refused Original song.wav: audio is 0.250 s long; at least 1 s required; nothing written')
        self.assertEqual(list(self.library.iterdir()), [])
        sf.write(self.source, np.sin(np.arange(441000) * .01) * .3, 44100)
        self.assertEqual(pcav.read_pcav_file(self.library / self.run_import() / 'analysis.pcav').peaks.sample_rate,
                         44100)

    def test_each_input_decoded_filtered_and_checksummed_once(self):
        # Repeated decodes, filters, checksums and a separate loudness pass made imports ~20% slower.
        encoded = self.root / 'download.m4a'
        import_track.convert_to_m4a(self.source, encoded)
        decode = content_hash.load_audio_samples
        # A WAV's one decode serves identity and analysis; a compressed input is also re-decoded when staged, then its master read.
        for audio, decodes in ((self.source, 1), (encoded, 3)):
            with self.subTest(audio=audio.name), \
                    patch.object(content_hash, 'load_audio_samples', wraps=decode) as hash_decodes, \
                    patch.object(analyze, 'load_audio_samples', wraps=decode) as analysis_decodes, \
                    patch.object(analyze, 'band_pass', wraps=analyze.band_pass) as band_pass, \
                    patch.object(import_track, 'sha256_file', wraps=import_track.sha256_file) as sha256, \
                    patch.object(subprocess, 'run', wraps=subprocess.run) as run, \
                    patch.object(subprocess, 'Popen', wraps=subprocess.Popen) as popen:
                import_track.import_song(audio, self.root / audio.suffix)
                self.assertEqual(hash_decodes.call_count + analysis_decodes.call_count, decodes)
                self.assertEqual(band_pass.call_count, 1)
                self.assertEqual(max(Counter(c.args[0] for c in sha256.call_args_list).values()), 1)
                # The detector reads the samples already decoded for identity, so the only ffmpeg pass over the
                # master encodes AAC and measures loudness (started with Popen so it overlaps analysis).
                samples, rate = analyze.extract_beats.call_args.args[:2]
                self.assertIsInstance(samples, np.ndarray)
                self.assertAlmostEqual(len(samples) / rate, 6, delta=0.1)
                spawned = [c.args[0] for c in run.call_args_list + popen.call_args_list]
                masters = [argv for argv in spawned if
                           '-i' in argv and Path(argv[argv.index('-i') + 1]).name == 'original.wav']
                encodes = [argv for argv in masters if 'aac' in argv]
                self.assertEqual(len(masters), 1, masters)
                self.assertEqual(len(encodes), 1)
                self.assertIn('ebur128=peak=true:framelog=quiet', encodes[0])

    def test_input_changed_after_decode_is_refused(self):
        encoded, altered_wav, altered_m4a = self.root / 'download.m4a', self.root / 'altered.wav', self.root / 'altered.m4a'
        import_track.convert_to_m4a(self.source, encoded)
        samples, rate = sf.read(self.source)
        sf.write(altered_wav, samples * .5, rate, subtype='PCM_24')
        import_track.convert_to_m4a(altered_wav, altered_m4a)
        for audio, replacement in ((self.source, altered_wav), (encoded, altered_m4a)):
            # The file changes after its identity was decoded and before it is staged.
            with self.subTest(audio=audio.name), \
                    patch.object(import_track, 'require_sound',
                                 side_effect=lambda *_, src=replacement, dst=audio: shutil.copyfile(src, dst)), \
                    self.assertRaisesRegex(ValueError, 'Input changed during import'):
                import_track.import_song(audio, self.library)
            self.assertEqual(list(self.library.iterdir()), [])

    def stalled_import(self):
        """Start the CLI in its own session; return it and its grandchild's pid once analysis has stalled in the stage."""
        marker = self.root / f'stalled-{time.monotonic_ns()}.pid'
        proc = subprocess.Popen([sys.executable, '-c', STALLED_IMPORT, str(Path(import_track.__file__).parent),
                                 str(import_track.SCHEMA_DIR),
                                 str(marker), str(self.source), '--library', str(self.library)],
                                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: proc.poll() is None and os.killpg(proc.pid, signal.SIGKILL))
        deadline = time.monotonic() + 60
        while not marker.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                self.fail(proc.communicate()[1].decode())
            time.sleep(.05)
        return proc, int(marker.read_text())

    def test_sigterm_removes_stage_and_children(self):
        # The server's timeout signals the process group and uv forwards more; repeats must not interrupt cleanup.
        for group in (False, True):
            proc, child = self.stalled_import()
            self.assertEqual(len(list(self.library.glob('.staging-*'))), 1)
            if group: os.killpg(proc.pid, signal.SIGTERM)
            os.kill(proc.pid, signal.SIGTERM)
            while group and proc.poll() is None:
                os.kill(proc.pid, signal.SIGTERM)
                time.sleep(.0005)
            self.assertEqual(proc.wait(30), 128 + signal.SIGTERM, proc.stderr.read().decode())
            self.assertEqual(list(self.library.iterdir()), [])
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)

    def test_stale_stages_from_killed_imports_swept_live_ones_kept(self):
        aged = lambda path: os.utime(path, (time.time() - 3600,) * 2, follow_symlinks=False) or path
        dead, _ = self.stalled_import()
        os.killpg(dead.pid, signal.SIGKILL)
        dead.wait(30)
        [leftover] = self.library.glob('.staging-*')
        aged(leftover)
        live, _ = self.stalled_import()  # its sweep removes the leftover
        [stage] = self.library.glob('.staging-*')
        self.assertNotEqual(stage, leftover)
        # Only unlocked, aged, real staging directories go: never a live stage, a fresh one, a symlink, a track or a candidate.
        h = self.run_import()
        decoy = self.root / 'decoy';
        decoy.mkdir();
        (decoy / 'keep').write_text('keep')
        (self.library / '.staging-0123456789ab-symlink1').symlink_to(decoy, target_is_directory=True)
        (self.library / '.staging-0123456789ab-fresh123').mkdir()
        (self.library / '.staging-other').mkdir()
        (self.library / h / 'analyses/.staging-0123456789ab-candidat').mkdir(parents=True)
        for path in (stage, self.library / '.staging-0123456789ab-symlink1', self.library / '.staging-other',
                     self.library / h / 'analyses/.staging-0123456789ab-candidat'):
            aged(path)
        kept = sorted(self.library.rglob('*'))
        with patch.object(analyze, 'extract_beats', side_effect=zoned):
            import_track.import_song(self.library / h / 'original.wav', self.library, force=True)
        self.assertEqual([p for p in kept if not p.exists()], [])
        self.assertEqual((decoy / 'keep').read_text(), 'keep')
        os.kill(live.pid, signal.SIGTERM)
        self.assertEqual(live.wait(30), 128 + signal.SIGTERM)
        self.assertFalse(stage.exists())

    def test_derived_source_provenance_survives_reanalysis(self):
        derived = self.root / 'processed.wav'
        samples, rate = sf.read(self.source)
        sf.write(derived, np.concatenate([np.zeros(rate), samples]), rate, subtype='FLOAT')
        record = self.root / 'processed.wav.json'
        record.write_text(json.dumps({'version': 1, 'inputSha256': import_track.sha256_file(self.source),
                                      'outputSha256': import_track.sha256_file(derived),
                                      'presets': [{'name': 'test_swell'}]}))
        h = import_track.import_song(derived, self.library, source_file=self.source, processing_record=record)
        folder = self.library / h
        self.assertEqual((folder / 'source.wav').read_bytes(), self.source.read_bytes())
        self.assertEqual(json.loads((folder / 'metadata.json').read_text())['title'], 'Original song')
        import_track.import_song(folder / 'original.wav', self.library, force=True)
        revision = next((folder / 'analyses').iterdir())
        manifest = json.loads((revision / 'manifest.json').read_text())
        self.assertEqual(manifest['source']['name'], self.source.name)
        self.assertEqual(manifest['processing'], json.loads(record.read_text()))
        self.assertEqual((revision / 'source.wav').read_bytes(), self.source.read_bytes())
        record.write_text('{}')
        with self.assertRaises(ValueError):
            import_track.import_song(derived, self.library, source_file=self.source, processing_record=record,
                                     force=True)

    def test_converted_master_reanalysis_retains_identity(self):
        encoded = self.root / 'download.m4a'
        import_track.convert_to_m4a(self.source, encoded)
        h = import_track.import_song(encoded, self.library)
        folder = self.library / h
        self.assertEqual((folder / 'source.m4a').read_bytes(), encoded.read_bytes())
        # Without --force the retained master maps to its track, not a duplicate titled "original".
        self.assertEqual(import_track.import_song(folder / 'original.wav', self.library), h)
        self.assertNothingPublished(h)
        self.assertEqual(import_track.import_song(folder / 'original.wav', self.library, force=True), h)
        self.assertEqual(len(list(self.library.iterdir())), 1)
        self.assertEqual(len(list((folder / 'analyses').iterdir())), 1)

    def test_missing_source_collision_and_symlink_rejected(self):
        h = self.run_import()
        folder = self.library / h
        old = self.snapshot(folder)
        with patch.object(import_track, 'fingerprint', return_value=h + 'f' * 20):
            with self.assertRaisesRegex(ValueError, 'Hash collision'): self.run_import(force=True)
        self.assertEqual(self.snapshot(folder), old)
        alternate = self.root / 'alternate';
        alternate.mkdir()
        (alternate / h).symlink_to(folder, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'must not be a symlink'): import_track.import_song(self.source,
                                                                                                   alternate,
                                                                                                   force=True)


if __name__ == '__main__':
    unittest.main()
