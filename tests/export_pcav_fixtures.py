"""Regenerate shared Python/TypeScript binary contract fixtures."""
from pathlib import Path
import json
import struct
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcav import AnalysisSection, BeatsSection, PCAVData, build_peak_mipmaps, write_pcav_file

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
beats = [0.123456 + i * 60 / 123.4 for i in range(20)]
for rate in (44100, 48000):
    samples = np.zeros(rate * 10 + 1)
    samples[::max(1, rate // 2)] = .8
    levels = build_peak_mipmaps(samples, rate)
    data = PCAVData(AnalysisSection(len(samples) / rate, 123.4, 'A minor'), BeatsSection(beats, 3), levels[0], levels[1])
    write_pcav_file(destination / f'v2-{rate}.pcav', data)
    (destination / f'v2-{rate}.json').write_text(json.dumps({
        'duration': len(samples) / rate, 'beatTimes': beats, 'firstDownbeatIndex': 3,
        'firstDownbeat': round(beats[3] * 1_000_000) / 1_000_000,
        'sampleRate': rate, 'peakCount': len(levels[0].peaks), 'overviewCount': len(levels[1].peaks),
    }, indent=2) + '\n')
# Legacy variants exercise uint16/uint32 headers with the original 10/s and 48kHz contracts.
for width in (2, 4):
    sections = [(1, struct.pack('<HHBx', 2, 1200, 0)), (2, struct.pack('<HHBB', 200, 3, 50, 51)),
                (3, struct.pack('<H' if width == 2 else '<I', 2) + bytes([129,127,0,0]))]
    at = 6 + 9 * len(sections)
    directory = bytearray(); payload = bytearray()
    for kind, data in sections:
        directory.extend(struct.pack('<BII', kind, at, len(data))); payload.extend(data); at += len(data)
    (destination / f'v1-count{width * 8}.pcav').write_bytes(b'PCAV' + bytes([1, len(sections)]) + directory + payload)
