"""
GPU helpers for optional CUDA-accelerated preprocessing and warping.

These are kept isolated so default behaviour (CPU OpenCV/NumPy paths)
remains byte-for-byte identical to baseline FaceFusion 3.5.0. Enable
them explicitly via the guard functions in this package.
"""

from __future__ import annotations

import os
from typing import Tuple


def gpu_accel_enabled() -> bool:
	"""
	Return True when CUDA acceleration is explicitly requested.

	We keep this opt-in to avoid any unintended numerical drift versus
	the default CPU code paths. Set environment variable
	FACEFUSION_USE_CUDA_ACCEL=1 to enable.
	"""
	disabled = os.environ.get("FACEFUSION_DISABLE_CUDA_ACCEL", "0") == "1"
	if disabled:
		return False
	# Default to enabled when CUDA is present; can be force-enabled on CPU for future extensibility.
	override = os.environ.get("FACEFUSION_USE_CUDA_ACCEL")
	if override is not None:
		return override == "1"
	try:
		import torch
		return torch.cuda.is_available()
	except Exception:
		return False


def current_cuda_device() -> Tuple[str, int]:
	"""Return (provider, device_id) for ONNX Runtime bindings."""
	try:
		import torch
		device_id = torch.cuda.current_device()
		return ("cuda", int(device_id))
	except Exception:
		return ("cuda", 0)


def gpu_compositor_enabled() -> bool:
	"""
	Guard for the optional GPU compositor. Default is disabled to ensure
	bitwise parity with the CPU pasteback; enable with
	FACEFUSION_USE_GPU_COMPOSITOR=1 when you want the CUDA path.
	"""
	return os.environ.get("FACEFUSION_USE_GPU_COMPOSITOR", "0") == "1"
