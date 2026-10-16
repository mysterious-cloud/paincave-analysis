#!/usr/bin/env python3
"""
Subtitles for a long class that has none, for Studio's cut editor (paincave docs/studio.md, phase 2b).

The coach's voice is split out with Demucs `htdemucs_ft` (its vocals model: `stems.separate_vocals`), then
transcribed by whisper.cpp `large-v3` with every word timed (DTW), and the words are cut into caption-sized lines
(`lines_of`). Singing lands in the vocals too, so lyrics come through as lines. On THE TRIP 43, against its real
subtitles: 99% of the coach's lines heard, 92% of their words; 38 of 548 lines heard aren't in them (lyrics, and
calls the subtitles leave out).

    class_transcript.py AUDIO -o OUT.vtt [--model GGML]

Progress goes to stdout as `PROGRESS {"stage": "voice" | "words", "fraction"}` lines, as `reference.py` writes them.
OUT.vtt is written whole or not at all.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import stems

WHISPER = "whisper-cli"
MODELS = Path.home() / ".local" / "share" / "whisper"
DEFAULT_MODEL = MODELS / "ggml-large-v3.bin"
WHISPER_RATE = 16000  # what whisper listens at
PROGRESS = re.compile(r"progress\s*=\s*(\d+)%")

GAP = 0.8  # seconds of quiet between two words that end a line
SOFT_WORDS = 8  # past this many words a comma ends a line too
MAX_WORDS, MAX_SECONDS = 14, 6.0  # a line never runs longer: a caption is read at a glance
MIN_SECONDS = 0.5  # a line shows at least this long (while the next hasn't started)
TOKEN_SECONDS = 0.3  # how long a DTW-timed token lasts, at most: DTW gives each token a moment, not a span

Word = tuple[float, float, str]


def progress(stage: str, fraction: float) -> None:
    print("PROGRESS " + json.dumps({"stage": stage, "fraction": round(fraction, 3)}), flush=True)


def words_of(document: dict) -> list[Word]:
    """
    Every word whisper heard as (start, end, text), seconds, in time order, from its segments' tokens (whisper.cpp's
    full JSON): a token starting with a space starts a word, any other joins the one before; special tokens
    (`[_BEG_]`, `[_TT_150]`) drop. A token's time is its DTW alignment (`t_dtw`, centiseconds: within 0.1 s of the real
    subtitles on THE TRIP 43) and lasts TOKEN_SECONDS, or up to the next word; whisper's own token offsets stretch a
    word over the silence before it ("Hey" at 0–5.3 s, said at 16.8 s), and are used only without DTW.
    """
    words: list[list] = []
    for segment in document["transcription"]:
        starts_word = True  # a segment's first token starts a word, space or not
        for token in segment["tokens"]:
            text = token["text"]
            if text.startswith("[_") or not text.strip():
                continue
            if token.get("t_dtw", -1) >= 0:
                start = token["t_dtw"] / 100
                end = start + TOKEN_SECONDS
            else:
                start, end = token["offsets"]["from"] / 1000, token["offsets"]["to"] / 1000
            if starts_word or text.startswith(" "):
                words.append([start, end, text.strip()])
            else:
                words[-1][1], words[-1][2] = max(words[-1][1], end), words[-1][2] + text
            starts_word = False
    words.sort(key=lambda word: word[0])
    for word, after in zip(words, words[1:]):
        word[1] = max(word[0], min(word[1], after[0]))
    return [(round(start, 3), round(end, 3), text) for start, end, text in words]


def lines_of(words: list[Word]) -> list[Word]:
    """
    Caption-sized lines from the words: a line ends at a sentence's end, at a pause, at a comma once it's long, and
    always by MAX_WORDS or MAX_SECONDS. Whisper's own segments end at its 30-second windows, mid-sentence, and run to 50
    words over 40 s of calls shouted across a song.
    """
    lines: list[Word] = []
    line: list[Word] = []
    for word in words:
        if line and (word[0] - line[-1][1] > GAP or len(line) >= MAX_WORDS or word[1] - line[0][0] > MAX_SECONDS):
            lines.append((line[0][0], line[-1][1], " ".join(w[2] for w in line)))
            line = []
        line.append(word)
        if word[2][-1] in ".?!" or (len(line) >= SOFT_WORDS and word[2][-1] in ",;:"):
            lines.append((line[0][0], line[-1][1], " ".join(w[2] for w in line)))
            line = []
    if line:
        lines.append((line[0][0], line[-1][1], " ".join(w[2] for w in line)))
    return lines


def stamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def vtt(lines: list[Word], note: str) -> str:
    """WebVTT for the lines, each shown at least MIN_SECONDS unless the next starts sooner."""
    cues = []
    for i, (start, end, text) in enumerate(lines):
        until = lines[i + 1][0] if i + 1 < len(lines) else end + MIN_SECONDS
        cues.append(f"{stamp(start)} --> {stamp(max(end, min(start + MIN_SECONDS, until)))}\n{text}")
    return "WEBVTT\n\nNOTE " + note + "\n\n" + "\n\n".join(cues) + "\n"


def whisper(vocals, model: Path, folder: Path) -> dict:
    """
    The vocals (mono float32 at 44.1 kHz) as whisper.cpp hears them: its full JSON, each token timed by DTW. Measured
    on THE TRIP 43 and the Apple Fitness class (paincave docs/studio.md, phase 2b): carrying text from one 30-second
    window to the next (`-mc 0` stops it) looped "Attack. The intensity is all." over the last 30 minutes of a class;
    voice activity detection keeps token times on its speech-only timeline, so it's off; DTW needs flash attention off.
    """
    wav = folder / "vocals.wav"
    result = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(stems.RATE), "-ac", "1", "-i", "-",
                             "-ar", str(WHISPER_RATE), str(wav)], input=vocals.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
    # The DTW preset is named after the model, quantization aside: ggml-large-v3 → large.v3, ggml-large-v3-turbo-q5_0
    # → large.v3.turbo
    preset = re.sub(r"-q\d.*$", "", model.stem.removeprefix("ggml-")).replace("-", ".")
    proc = subprocess.Popen([WHISPER, "-m", str(model), "-l", "en", "-sns", "-mc", "0", "-nfa", "-dtw", preset, "-pp",
                             "-ojf", "-of", str(folder / "words"), "-f", str(wav)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    assert proc.stderr is not None  # piped above
    last = ""
    for line in proc.stderr:
        if found := PROGRESS.search(line):
            progress("words", int(found.group(1)) / 100)
        elif line.strip():
            last = line.strip()
    if proc.wait() != 0:
        raise RuntimeError(f"{WHISPER} failed: {last}")
    return json.loads((folder / "words.json").read_text())


def transcribe(audio: Path, model: Path) -> str:
    """The class's subtitles as WebVTT, with a note saying how they were made."""
    if not model.is_file():
        raise ValueError(f"no whisper model at {model} (setup: paincave-analysis README, Class transcript)")
    progress("voice", 0)
    try:
        mix = stems.load_stereo(audio)
    except RuntimeError as error:
        raise ValueError(f"could not decode {audio}: {error}") from error
    if not mix.shape[1]:
        raise ValueError(f"no audio in {audio}")
    vocals = stems.separate_vocals(mix, lambda fraction: progress("voice", fraction))
    del mix  # a 45-minute class is about 1 GB decoded: whisper and its model need the room
    progress("words", 0)
    with tempfile.TemporaryDirectory() as folder:
        heard = whisper(vocals, model, Path(folder))
    return vtt(lines_of(words_of(heard)), f"Made by paincave-analysis: Demucs {stems.MODEL} vocals, whisper.cpp {model.stem}")


def main():
    parser = argparse.ArgumentParser(description="Subtitles for a class without them (Demucs vocals + whisper.cpp)")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help="Where the WebVTT goes (written whole or not at all)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"whisper.cpp model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    try:
        subtitles = transcribe(args.audio, args.model)
    except (ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as error:
        print(f"{error.filename} not found (setup: paincave-analysis README, Class transcript)", file=sys.stderr)
        sys.exit(1)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(subtitles)
    os.replace(temporary, args.output)
    print(f"{subtitles.count(' --> ')} lines", file=sys.stderr)


if __name__ == "__main__":
    main()
