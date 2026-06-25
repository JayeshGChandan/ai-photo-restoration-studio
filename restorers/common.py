"""
Shared helpers for the restoration backends.
"""

import os
from typing import Optional

import torch


def resolve_device(device: Optional[str] = None, env_var: str = "RESTORE_DEVICE") -> torch.device:
    """Pick a compute device for the restoration models.

    Priority: explicit ``device`` arg > ``env_var`` env override > CUDA (if
    present) > CPU. We default to CPU on Apple Silicon because several of the
    model ops (FFTs in LaMa, some convs in the GAN upsamplers) are either
    unsupported or numerically unreliable on the MPS backend. MPS can still be
    opted into with ``RESTORE_DEVICE=mps`` (each backend falls back to CPU if
    the graph fails to run).
    """
    if device:
        return torch.device(device)
    env = os.environ.get(env_var)
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Absolute path to the bundled local weights directory.
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
