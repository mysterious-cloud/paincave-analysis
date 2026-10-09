#!/usr/bin/env python3
"""
Score each section method in an `analysis.json` against confirmed sections (Songlab's Lab truth export).

Boundaries: precision, recall and F at exact bars and within one bar. Labels: pairwise F over bars (two bars
share a label in the truth and in the method). Usage: bin/score-sections analysis.json truth.json
"""

import argparse
import json
from itertools import combinations
from pathlib import Path


def starts(sections: list[dict]) -> list[int]:
    return [s["startBar"] for s in sections[1:]]


def boundary_f(found: list[int], truth: list[int], tolerance: int) -> tuple[float, float, float]:
    """Each truth boundary matches at most one found boundary within `tolerance` bars."""
    unmatched = list(truth)
    hits = 0
    for b in found:
        match = next((t for t in unmatched if abs(t - b) <= tolerance), None)
        if match is not None:
            unmatched.remove(match)
            hits += 1
    precision = hits / len(found) if found else 1.0
    recall = hits / len(truth) if truth else 1.0
    return precision, recall, (2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def bar_labels(sections: list[dict]) -> list[str]:
    return [s["label"] for s in sections for _ in range(s["endBar"] - s["startBar"])]


def pairwise_f(found: list[str], truth: list[str]) -> float:
    n = min(len(found), len(truth))
    same_found = {(i, j) for i, j in combinations(range(n), 2) if found[i] == found[j]}
    same_truth = {(i, j) for i, j in combinations(range(n), 2) if truth[i] == truth[j]}
    both = len(same_found & same_truth)
    precision = both / len(same_found) if same_found else 1.0
    recall = both / len(same_truth) if same_truth else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(analysis: dict, truth: dict) -> dict[str, dict]:
    expected = truth["sections"]
    out = {}
    for name, method in analysis.get("sections", {}).get("methods", {}).items():
        found = method["sections"]
        out[name] = {
            "exact": round(boundary_f(starts(found), starts(expected), 0)[2], 3),
            "withinOneBar": round(boundary_f(starts(found), starts(expected), 1)[2], 3),
            "labelPairwiseF": round(pairwise_f(bar_labels(found), bar_labels(expected)), 3),
            "sections": len(found),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score section methods against confirmed sections")
    parser.add_argument("analysis", type=Path)
    parser.add_argument("truth", type=Path, help='JSON with "sections": [{startBar, endBar, label}]')
    args = parser.parse_args(argv)
    truth = json.loads(args.truth.read_text())
    print(json.dumps({"truthSections": len(truth["sections"]),
                      "methods": score(json.loads(args.analysis.read_text()), truth)}, indent=2))


if __name__ == "__main__":
    main()
