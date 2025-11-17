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
    """Convert BGR uint8 tensor (B,H,W,3) to normalized NCHW float32 on GPU."""
    if frame_bgr.dtype != torch.float32:
        frame_bgr = frame_bgr.to(torch.float32)
    frame = frame_bgr / 255.0
    frame = (frame - mean) / std
    return frame.permute(0, 3, 1, 2).contiguous()


def from_nchw_float(frame: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse of to_nchw_float, returning uint8 (B,H,W,3)."""
    frame = frame.permute(0, 2, 3, 1)
    frame = frame * std + mean
    frame = frame.clamp(0.0, 1.0) * 255.0
    return frame.to(torch.uint8)


def resize_affine(frame: torch.Tensor, matrices: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    """Affine warp batch on GPU, using CV-CUDA when present."""
    batch = frame.shape[0]
    if _HAS_CVCUDA:
        tensor = cvcuda.Tensor(frame)
        mats = cvcuda.Tensor(matrices)
        warped = cvcuda.warpaffine(tensor, mats, out_size, interpolation=cvcuda.Interp.LINEAR, border_mode=cvcuda.Border.REPLICATE)  # type: ignore[attr-defined]
        return torch.as_tensor(warped.cuda(), device=frame.device)
    grid = torch.nn.functional.affine_grid(matrices, torch.Size((batch, frame.shape[3], *out_size)), align_corners=False)
    frame_t = frame.permute(0, 3, 1, 2)
    warped = torch.nn.functional.grid_sample(frame_t, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return warped.permute(0, 2, 3, 1)
