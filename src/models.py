#!/usr/bin/env python3
"""Identity of the Beat This! checkpoint behind every analysis.

The checkpoint is loaded by name and cached by torch hub, so the manifest records what actually ran:
its name and the SHA-256 of the cached file.
"""

import hashlib
from pathlib import Path

# small0 runs a 3-minute track in ~1.5 s on CPU and agreed with the previous grids on ~95% of downbeats;
# the larger final0 put one section of The Strength Within a beat early.
CHECKPOINT = "small0"


def checkpoint_path(name: str = CHECKPOINT) -> Path:
    """Where torch hub caches the named checkpoint; it may not exist until the detector has run once."""
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / f"beat_this-{name}.ckpt"


def checkpoint_info(name: str = CHECKPOINT) -> dict:
    """Checkpoint name and the SHA-256 of its cached file, or None while it is not cached."""
    path = checkpoint_path(name)
    if not path.is_file():
        return {"name": name, "sha256": None}
    with path.open("rb") as f:
        return {"name": name, "sha256": hashlib.file_digest(f, "sha256").hexdigest()}


def ensure_checkpoint(name: str = CHECKPOINT) -> dict:
    """Download the checkpoint if the cache is cold, then describe it."""
    if not checkpoint_path(name).is_file():
        from beat_this.inference import load_checkpoint

        load_checkpoint(name)
    return {**checkpoint_info(name), "file": str(checkpoint_path(name))}
