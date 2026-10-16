#!/usr/bin/env python3
"""
Sections, labels, beats and tempo from All-In-One (allin1 1.1.0, MIT; Kim & Nam, WASPAA 2023) for Songlab.

Runs in its own environment (this folder's pyproject). NATTEN, which allin1's attention layers call, is swapped for
natten_torch.py (same results to float precision, verified against NATTEN 0.15.1 on CPU), so the model runs on
Apple's GPU: the 8-fold ensemble takes about 10 s per song on an M1 Max instead of about 5 minutes on CPU.

The stems come from reference.py's Demucs cache (htdemucs_ft, FLAC): allin1's own split would run Demucs again,
with plain htdemucs and a random time shift. Usage:
    run.py AUDIO --stems DIR -o OUT.json [--device auto|mps|cpu] [--model harmonix-all]
"""

import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEMS = ("bass", "drums", "other", "vocals")


def use_torch_natten() -> None:
    """Make `from natten.functional import ...` (allin1/models/dinat.py) find the plain-PyTorch version."""
    sys.path.insert(0, str(HERE))
    import natten_torch

    package = types.ModuleType("natten")
    package.functional = natten_torch
    sys.modules["natten"] = package
    sys.modules["natten.functional"] = natten_torch


def device_for(choice: str) -> str:
    import torch

    if choice != "auto":
        return choice
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="All-In-One sections for Songlab")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--stems", type=Path, required=True, help="Folder with drums/bass/other/vocals .flac")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--model", default="harmonix-all")
    args = parser.parse_args(argv)

    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # the checkpoint download on first use
    except ImportError:
        pass
    use_torch_natten()
    import soundfile as sf
    import torch
    import allin1

    device = device_for(args.device)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="allin1-") as tmp:
        work = Path(tmp)
        # allin1 skips its own split when demix/htdemucs/<audio stem>/<name>.wav exist: the folder name is where it
        # looks, the stems in it are ours (htdemucs_ft)
        stem_dir = work / "demix" / "htdemucs" / "audio"
        stem_dir.mkdir(parents=True)
        for name in STEMS:
            samples, rate = sf.read(args.stems / f"{name}.flac", dtype="float32", always_2d=True)
            sf.write(stem_dir / f"{name}.wav", samples, rate, subtype="FLOAT")
        audio = work / f"audio{args.audio.suffix}"
        audio.symlink_to(args.audio.resolve())
        with torch.no_grad():
            result = allin1.analyze(audio, model=args.model, device=device, demix_dir=work / "demix",
                                    spec_dir=work / "spec", keep_byproducts=False, multiprocess=False)
    document = {
        "tool": "allin1",
        "version": allin1.__version__ if hasattr(allin1, "__version__") else "1.1.0",
        "model": args.model,
        "device": device,
        "seconds": round(time.time() - started, 1),
        "bpm": result.bpm,
        "segments": [{"start": s.start, "end": s.end, "label": s.label} for s in result.segments],
        "beats": [round(b, 3) for b in result.beats],
        "downbeats": [round(b, 3) for b in result.downbeats],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"allin1 on {device}: {len(document['segments'])} segments, {result.bpm} BPM, {document['seconds']} s")


if __name__ == "__main__":
    main()
