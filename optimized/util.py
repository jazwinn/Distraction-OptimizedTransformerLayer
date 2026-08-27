"""Small helpers shared across the package."""

from __future__ import annotations

from typing import Optional

import torch


def _version_or_none(t: torch.Tensor) -> Optional[int]:
    """t._version, or None when the tensor does not track one.

    Inference tensors raise instead of reporting a version, and both cache keys
    below can be handed one -- the accuracy path builds its mask inside
    inference_mode(). None makes the version half of a key best-effort; the
    identity half still carries correctness.
    """
    try:
        return t._version
    except RuntimeError:
        return None
