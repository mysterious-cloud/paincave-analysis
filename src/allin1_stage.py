#!/usr/bin/env python3
"""
The `allin1` stage of reference analysis: All-In-One's sections, named (intro, verse, chorus...), snapped to the
grid's bars. allin1 runs as a subprocess in its own environment (`bin/allin1`, tools/allin1) on the stems the
stems stage cached, so its pins never touch Pain Cave's and Demucs runs once. Its result is cached beside them.
"""

import json
import subprocess
from pathlib import Path

ALLIN1 = Path(__file__).resolve().parents[1] / "bin" / "allin1"
METHOD = "allin1"
MODEL = "harmonix-all"
EDGE_LABELS = ("start", "end")  # allin1's silence before the music and after it


def run(audio: Path, stems_dir: Path, cache_dir: Path) -> dict:
    """allin1's output for this audio, from the cache when the same stems were analysed before."""
    cached = cache_dir / f"allin1-{stems_dir.name}-{MODEL}.json"
    if not cached.is_file():
        result = subprocess.run([str(ALLIN1), str(audio), "--stems", str(stems_dir), "-o", str(cached), "--model", MODEL],
                                capture_output=True, text=True)
        if result.returncode != 0:
            last = (result.stderr.strip().splitlines() or ["no output"])[-1]
            raise RuntimeError(f"allin1 failed: {last}")
    return json.loads(cached.read_text())


def nearest_bar(bars: list[dict], seconds: float) -> int:
    """The 1-based bar whose start is nearest `seconds`; len(bars) + 1 past the end."""
    starts = [b["start"] for b in bars] + [bars[-1]["end"]]
    return min(range(len(starts)), key=lambda i: abs(starts[i] - seconds)) + 1


def to_sections(segments: list[dict], bars: list[dict]) -> list[dict]:
    """Contiguous sections on bars, `endBar` exclusive. The silent start joins the first section, the silent end the
    last; a section that snaps to no bars is dropped."""
    end = len(bars) + 1
    named = [s for s in segments if s["label"] not in EDGE_LABELS] or segments
    out: list[dict] = []
    for i, s in enumerate(named):
        start = 1 if i == 0 else nearest_bar(bars, s["start"])
        if out and start <= out[-1]["startBar"]:
            continue
        if out:
            out[-1]["endBar"] = start
        out.append({"startBar": start, "endBar": end, "label": s["label"]})
    return out


def block(document: dict, bars: list[dict]) -> dict:
    sections = to_sections(document["segments"], bars)
    return {
        "tool": METHOD,
        "version": 1,
        "model": document.get("model", MODEL),
        "bpm": document.get("bpm"),
        "sections": sections,
        "boundariesSeconds": [bars[s["startBar"] - 1]["start"] for s in sections[1:]],
        "segmentsSeconds": [{"start": round(s["start"], 3), "end": round(s["end"], 3), "label": s["label"]}
                            for s in document["segments"]],
    }
