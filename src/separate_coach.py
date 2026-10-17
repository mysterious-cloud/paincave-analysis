#!/usr/bin/env python3
"""
A class recording split in two, the music and the coach, for Studio (paincave docs/studio.md, class breakdown).

The vocals are split out with Demucs `htdemucs_ft` (its vocals model, stereo: `stems.separate_vocals`, in 5-minute
chunks) and count as the coach only inside the spans given (the coach's subtitle lines, padded and joined by Studio),
fading in and out over FADE seconds at each edge. The coach file is those vocals and silence everywhere else; the
music file is the recording minus the coach, so outside the spans it is the recording sample for sample. Singing inside
a span goes with the coach: Demucs can't tell a singer from the coach.

    separate_coach.py IN --spans SPANS.json -o MUSIC.flac -c COACH.flac     # SPANS.json: [[start, end], ...] in seconds

Progress goes to stdout as `PROGRESS {"stage": "voice", "fraction"}` lines. Both files are 16-bit FLAC at the
recording's own rate and channels, quantized so that music + coach is the recording to 16 bits; both are written
before either replaces anything, so a failure leaves neither half-done. The recording is decoded by ffmpeg (a class is
often .m4a) and streamed BLOCK frames at a time, so only the vocals are ever held whole.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

import stems

FADE = 0.1  # seconds over which the coach fades in and out at each span's edges
BLOCK = 1 << 20  # frames of the recording read, split and written at a time (about 22 s at 48 kHz)


def progress(fraction: float) -> None:
    print("PROGRESS " + json.dumps({"stage": "voice", "fraction": round(fraction, 3)}), flush=True)


def check_spans(spans) -> None:
    if not isinstance(spans, list):
        raise ValueError("the spans file holds a list of [start, end]")
    for span in spans:
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in span)
                and 0 <= span[0] < span[1]):
            raise ValueError(f"a span is [start, end] in seconds, start before end: {span}")


def probe(path: Path) -> tuple[int, int]:
    """The recording's (rate, channels), by ffprobe; anything without an audio stream is refused."""
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
                             "-of", "json", str(path)], capture_output=True, text=True)
    found = json.loads(result.stdout or "{}").get("streams") if result.returncode == 0 else None
    if not found:
        raise ValueError(f"not audio: {path}: {result.stderr.strip() or 'no audio stream'}")
    rate, channels = int(found[0]["sample_rate"]), int(found[0]["channels"])
    if channels not in (1, 2):
        raise ValueError(f"{path} has {channels} channels; mono or stereo only")
    return rate, channels


def decoded(path: Path, channels: int) -> Iterator[np.ndarray]:
    """The recording at its own rate and channels, float32 (frames, channels), BLOCK frames at a time."""
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-"],
            stdout=subprocess.PIPE, stderr=errors) as proc:
        assert proc.stdout is not None  # piped above
        while chunk := proc.stdout.read(BLOCK * channels * 4):
            yield np.frombuffer(chunk, dtype=np.float32).reshape(-1, channels)
        if proc.wait() != 0:
            errors.seek(0)
            raise RuntimeError(f"ffmpeg failed: {errors.read().decode(errors='replace').strip()}")


def resample(audio: np.ndarray, rate: int, to: int) -> np.ndarray:
    """(channels, n) float32 from `rate` to (n, channels) frames at `to`, by ffmpeg (Demucs works at 44.1 kHz). Fed a
    block at a time and read straight into the result, so nothing but the result is allocated whole."""
    if rate == to:
        return audio.T
    channels, length = audio.shape
    out = np.empty((length * to // rate + to, channels), dtype=np.float32)  # a second's room for the resampler's rounding
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-",
             "-ar", str(to), "-f", "f32le", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors) as proc:
        assert proc.stdin is not None and proc.stdout is not None  # piped above

        def feed():
            try:
                for start in range(0, length, BLOCK):
                    proc.stdin.write(memoryview(np.ascontiguousarray(audio[:, start:start + BLOCK].T)).cast("B"))
            except BrokenPipeError:
                pass  # ffmpeg stopped; its own error says why
            finally:
                proc.stdin.close()

        feeder = threading.Thread(target=feed)
        feeder.start()
        view, filled = memoryview(out).cast("B"), 0
        while filled < len(view) and (read := proc.stdout.readinto(view[filled:])):
            filled += read
        if overflow := proc.stdout.read(1):
            proc.kill()
        feeder.join()
        if proc.wait() != 0 and not overflow:
            errors.seek(0)
            raise RuntimeError(f"ffmpeg failed: {errors.read().decode(errors='replace').strip()}")
    if overflow:
        raise RuntimeError(f"ffmpeg resampled {length} samples at {rate} Hz to more than {len(out)} at {to} Hz")
    return out[:filled // (4 * channels)]


def gain(start: int, length: int, rate: int, spans: list[list[float]]) -> np.ndarray:
    """How much of the vocals is the coach at samples start..start+length: 1 inside a span, 0 outside, ramps of FADE
    seconds at its edges."""
    g = np.zeros(length, dtype=np.float32)
    ramp = max(1, round(FADE * rate))
    for span_start, span_end in spans:
        a, b = round(span_start * rate), round(span_end * rate)
        lo, hi = max(start, a - ramp), min(start + length, b + ramp)
        if b <= a or hi <= lo:
            continue
        t = np.arange(lo, hi)
        edges = np.clip(np.minimum(t - a + ramp, b + ramp - 1 - t) / ramp, 0, 1).astype(np.float32)
        np.maximum(g[lo - start:hi - start], edges, out=g[lo - start:hi - start])
    return g


def pcm16(audio: np.ndarray) -> np.ndarray:
    """Float samples as 16-bit steps, rounded and clipped to ±1 (still float32, exact)."""
    return np.clip(np.rint(audio * 32768), -32768, 32767)


def separate_coach(path: Path, spans: list[list[float]], music_path: Path, coach_path: Path) -> None:
    """Write the recording at `path` as music (`music_path`) and coach (`coach_path`), the coach being its vocals inside
    `spans`."""
    check_spans(spans)
    rate, channels = probe(path)
    progress(0)
    vocals = None
    if spans:
        # Back to the recording's own rate, frames first, the only thing held whole (a mono recording: their mean)
        vocals = resample(stems.separate_vocals(stems.load_stereo(path), progress, stereo=True), stems.RATE, rate)
    temporaries = [p.with_name(f".{p.name}.tmp") for p in (music_path, coach_path)]
    try:
        with sf.SoundFile(temporaries[0], "w", rate, channels, "PCM_16", format="FLAC") as music, \
                sf.SoundFile(temporaries[1], "w", rate, channels, "PCM_16", format="FLAC") as coach:
            start = 0
            for block in decoded(path, channels):
                voice = np.zeros_like(block)
                g = gain(start, len(block), rate, spans)
                if vocals is not None and g.any():
                    heard = vocals[start:start + len(block)]
                    if channels == 1:
                        heard = heard.mean(1, keepdims=True)
                    voice[:len(heard)] = g[:len(heard), None] * heard
                # Quantized apart, so music + coach is the recording to 16 bits (sample for sample if it was 16-bit)
                voice = pcm16(voice)
                rest = pcm16(block)
                rest -= voice
                np.clip(rest, -32768, 32767, out=rest)
                music.write(rest.astype(np.int16))
                coach.write(voice.astype(np.int16))
                start += len(block)
        for temporary, final in zip(temporaries, (music_path, coach_path)):
            os.replace(temporary, final)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    progress(1)


def main():
    parser = argparse.ArgumentParser(description="A class recording split in two, the music and the coach")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--spans", type=Path, required=True, help="JSON: [[start, end], ...] in seconds, where the coach talks")
    parser.add_argument("-o", "--music", type=Path, required=True, help="Where the music FLAC goes (written whole or not at all)")
    parser.add_argument("-c", "--coach", type=Path, required=True, help="Where the coach FLAC goes (written whole or not at all)")
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.audio, args.music, args.coach)]
        if len(set(paths)) < 3:
            raise ValueError("the music, the coach and the recording must be three different files")
        spans = json.loads(args.spans.read_text())
        separate_coach(args.audio, spans, args.music, args.coach)
    except (ValueError, RuntimeError, OSError) as error:  # JSONDecodeError is a ValueError; soundfile's errors RuntimeErrors
        print("Error: " + " ".join(str(error).split()), file=sys.stderr)
        sys.exit(1)
    print(f"coach separated in {len(spans)} spans", file=sys.stderr)


if __name__ == "__main__":
    main()
