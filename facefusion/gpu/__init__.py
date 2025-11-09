"""CUDA-accelerated helpers for FaceFusion."""
from __future__ import annotations

from typing import Optional

import torch

_try_cuda_graphs = False


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def use_cuda_graphs() -> bool:
    global _try_cuda_graphs
    return _try_cuda_graphs


def enable_cuda_graphs(enable: bool) -> None:
    global _try_cuda_graphs
    _try_cuda_graphs = bool(enable)
