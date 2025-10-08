"""GPU-native preprocessing helpers using CV-CUDA when available."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import cvcuda  # type: ignore
    _HAS_CVCUDA = True
except Exception:
    cvcuda = None  # type: ignore
    _HAS_CVCUDA = False


def to_nchw_float(frame_bgr: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if frame_bgr.dtype != torch.float32:
        frame_bgr = frame_bgr.to(torch.float32)
    frame = frame_bgr / 255.0
    frame = (frame - mean) / std
    return frame.permute(0, 3, 1, 2).contiguous()


def from_nchw_float(frame: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    frame = frame.permute(0, 2, 3, 1)
    frame = frame * std + mean
    frame = frame.clamp(0.0, 1.0) * 255.0
    return frame.to(torch.uint8)


def resize_affine(frame: torch.Tensor, matrices: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    batch = frame.shape[0]
    if _HAS_CVCUDA:
        tensor = cvcuda.Tensor(frame)
        mats = cvcuda.Tensor(matrices)
        warped = cvcuda.warpaffine(tensor, mats, out_size, interpolation=cvcuda.Interp.LINEAR, border_mode=cvcuda.Border.CLAMP)  # type: ignore[attr-defined]
        return torch.as_tensor(warped.cuda(), device=frame.device)
    grid = torch.nn.functional.affine_grid(matrices, torch.Size((batch, frame.shape[3], *out_size)), align_corners=False)
    frame_t = frame.permute(0, 3, 1, 2)
    warped = torch.nn.functional.grid_sample(frame_t, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return warped.permute(0, 2, 3, 1)
