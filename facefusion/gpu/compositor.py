"""CUDA compositor that mirrors CPU paste_back behaviour."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy

try:
	import torch
	_HAS_TORCH = True
except Exception:
	_HAS_TORCH = False

from facefusion.gpu.kernels import warp_face
from facefusion.face_helper import calculate_paste_area


def paste_back_cuda(temp_vision_frame: numpy.ndarray, crop_vision_frame: numpy.ndarray, crop_vision_mask: numpy.ndarray, affine_matrix: numpy.ndarray) -> Optional[numpy.ndarray]:
	"""
	GPU equivalent of face_helper.paste_back. Returns None on any failure so
	the caller can fall back to CPU. Math matches the CPU path:
	- border replicate
	- bilinear warp
	- mask clamp and linear blend
	"""
	if not _HAS_TORCH:
		return None
	if not torch.cuda.is_available():
		return None
	try:
		paste_bounding_box, paste_matrix = calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix)
		x1, y1, x2, y2 = paste_bounding_box
		paste_width = int(x2 - x1)
		paste_height = int(y2 - y1)
		if paste_width <= 0 or paste_height <= 0:
			return None

		device = torch.device("cuda")
		# Prepare tensors
		src = torch.from_numpy(crop_vision_frame.astype(numpy.float32)).to(device)
		mask = torch.from_numpy(crop_vision_mask.astype(numpy.float32)).to(device)
		affine = torch.from_numpy(paste_matrix.astype(numpy.float32)).to(device)

		# Warp crop and mask to paste area
		warped = warp_face(src, affine, paste_width, paste_height)  # float32
		# Mask is single channel; expand for broadcast
		mask_warped = warp_face(mask.unsqueeze(-1), affine, paste_width, paste_height).squeeze(-1)
		mask_warped = mask_warped.clamp(0.0, 1.0)

		# Blend
		paste_region = torch.from_numpy(temp_vision_frame[y1:y2, x1:x2].astype(numpy.float32)).to(device)
		out = paste_region * (1.0 - mask_warped.unsqueeze(-1)) + warped * mask_warped.unsqueeze(-1)
		out = out.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

		# Replace region
		temp_copy = temp_vision_frame.copy()
		temp_copy[y1:y2, x1:x2] = out
		return temp_copy
	except Exception:
		return None
