"""
GPU-accelerated color space conversion utilities using PyTorch.

This module provides CUDA implementations of color space conversions
commonly used in FaceFusion for GPU-resident processing.
"""

import torch

from facefusion.gpu_types import CUDAFrame


def bgr_to_rgb_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert BGR to RGB on GPU by flipping the channel dimension.

	Args:
		image: BGR image (H, W, C) with C=3

	Returns:
		RGB image (H, W, C)
	"""
	return image.flip(-1)


def rgb_to_bgr_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert RGB to BGR on GPU by flipping the channel dimension.

	Args:
		image: RGB image (H, W, C) with C=3

	Returns:
		BGR image (H, W, C)
	"""
	return image.flip(-1)


def bgr_to_grayscale_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert BGR to grayscale on GPU using weighted average.

	Args:
		image: BGR image (H, W, 3) in float32 [0, 255]

	Returns:
		Grayscale image (H, W) in float32 [0, 255]
	"""
	# OpenCV weights for BGR: B=0.114, G=0.587, R=0.299
	weights = torch.tensor([0.114, 0.587, 0.299], dtype=torch.float32, device=image.device)
	weights = weights.view(1, 1, 3)

	# Weighted sum
	grayscale = (image * weights).sum(dim=-1)

	return grayscale


def bgr_to_hsv_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert BGR to HSV color space on GPU.

	Args:
		image: BGR image (H, W, 3) in float32 [0, 255]

	Returns:
		HSV image (H, W, 3) with H in [0, 360], S in [0, 1], V in [0, 255]
	"""
	# Normalize to [0, 1]
	bgr_normalized = image / 255.0

	# Extract BGR channels
	b = bgr_normalized[:, :, 0]
	g = bgr_normalized[:, :, 1]
	r = bgr_normalized[:, :, 2]

	# Calculate V (max of RGB)
	v = torch.max(torch.stack([r, g, b], dim=-1), dim=-1)[0]

	# Calculate S
	min_rgb = torch.min(torch.stack([r, g, b], dim=-1), dim=-1)[0]
	delta = v - min_rgb
	s = torch.where(v == 0, torch.zeros_like(v), delta / v)

	# Calculate H
	h = torch.zeros_like(v)

	# Red is max
	mask_r = (v == r) & (delta != 0)
	h = torch.where(mask_r, 60.0 * (((g - b) / delta) % 6), h)

	# Green is max
	mask_g = (v == g) & (delta != 0)
	h = torch.where(mask_g, 60.0 * (((b - r) / delta) + 2), h)

	# Blue is max
	mask_b = (v == b) & (delta != 0)
	h = torch.where(mask_b, 60.0 * (((r - g) / delta) + 4), h)

	# Ensure H is in [0, 360]
	h = torch.where(h < 0, h + 360, h)

	# Stack to form HSV image (denormalize V back to [0, 255])
	hsv = torch.stack([h, s, v * 255.0], dim=-1)

	return hsv


def hsv_to_bgr_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert HSV to BGR color space on GPU.

	Args:
		image: HSV image (H, W, 3) with H in [0, 360], S in [0, 1], V in [0, 255]

	Returns:
		BGR image (H, W, 3) in float32 [0, 255]
	"""
	# Extract HSV channels
	h = image[:, :, 0]
	s = image[:, :, 1]
	v = image[:, :, 2] / 255.0  # Normalize V to [0, 1]

	# Calculate intermediate values
	c = v * s
	x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
	m = v - c

	# Initialize RGB
	r = torch.zeros_like(h)
	g = torch.zeros_like(h)
	b = torch.zeros_like(h)

	# Assign RGB based on hue sector
	mask_0 = (h >= 0) & (h < 60)
	mask_1 = (h >= 60) & (h < 120)
	mask_2 = (h >= 120) & (h < 180)
	mask_3 = (h >= 180) & (h < 240)
	mask_4 = (h >= 240) & (h < 300)
	mask_5 = (h >= 300) & (h < 360)

	# Sector 0: [0, 60)
	r = torch.where(mask_0, c, r)
	g = torch.where(mask_0, x, g)
	b = torch.where(mask_0, torch.zeros_like(b), b)

	# Sector 1: [60, 120)
	r = torch.where(mask_1, x, r)
	g = torch.where(mask_1, c, g)
	b = torch.where(mask_1, torch.zeros_like(b), b)

	# Sector 2: [120, 180)
	r = torch.where(mask_2, torch.zeros_like(r), r)
	g = torch.where(mask_2, c, g)
	b = torch.where(mask_2, x, b)

	# Sector 3: [180, 240)
	r = torch.where(mask_3, torch.zeros_like(r), r)
	g = torch.where(mask_3, x, g)
	b = torch.where(mask_3, c, b)

	# Sector 4: [240, 300)
	r = torch.where(mask_4, x, r)
	g = torch.where(mask_4, torch.zeros_like(g), g)
	b = torch.where(mask_4, c, b)

	# Sector 5: [300, 360)
	r = torch.where(mask_5, c, r)
	g = torch.where(mask_5, torch.zeros_like(g), g)
	b = torch.where(mask_5, x, b)

	# Add m and denormalize to [0, 255]
	r = (r + m) * 255.0
	g = (g + m) * 255.0
	b = (b + m) * 255.0

	# Stack to form BGR image
	bgr = torch.stack([b, g, r], dim=-1)

	return bgr


def normalize_frame_cuda(
	frame: CUDAFrame,
	mean: list,
	std: list
) -> CUDAFrame:
	"""
	Normalize a frame with mean and std on GPU.

	Args:
		frame: Input frame (H, W, C) with values typically in [0, 1] or [0, 255]
		mean: Mean values for each channel
		std: Standard deviation for each channel

	Returns:
		Normalized frame
	"""
	device = frame.device
	mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
	std_tensor = torch.tensor(std, dtype=torch.float32, device=device)

	# Reshape for broadcasting: (1, 1, C)
	mean_tensor = mean_tensor.view(1, 1, -1)
	std_tensor = std_tensor.view(1, 1, -1)

	# Normalize
	normalized = (frame - mean_tensor) / std_tensor

	return normalized


def denormalize_frame_cuda(
	frame: CUDAFrame,
	mean: list,
	std: list
) -> CUDAFrame:
	"""
	Denormalize a frame with mean and std on GPU.

	Args:
		frame: Normalized frame (H, W, C)
		mean: Mean values used for normalization
		std: Standard deviation values used for normalization

	Returns:
		Denormalized frame
	"""
	device = frame.device
	mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
	std_tensor = torch.tensor(std, dtype=torch.float32, device=device)

	# Reshape for broadcasting: (1, 1, C)
	mean_tensor = mean_tensor.view(1, 1, -1)
	std_tensor = std_tensor.view(1, 1, -1)

	# Denormalize
	denormalized = frame * std_tensor + mean_tensor

	return denormalized


def uint8_to_float32_cuda(frame: CUDAFrame) -> CUDAFrame:
	"""
	Convert uint8 image to float32 [0, 1] on GPU.

	Args:
		frame: uint8 image (H, W, C)

	Returns:
		float32 image (H, W, C) in [0, 1]
	"""
	if frame.dtype == torch.uint8:
		return frame.float() / 255.0
	return frame


def float32_to_uint8_cuda(frame: CUDAFrame) -> CUDAFrame:
	"""
	Convert float32 image to uint8 on GPU.

	Args:
		frame: float32 image (H, W, C) in [0, 1] or [0, 255]

	Returns:
		uint8 image (H, W, C)
	"""
	if frame.dtype == torch.float32:
		# Auto-detect range
		if frame.max() <= 1.0:
			# [0, 1] range
			return (frame * 255.0).clamp(0, 255).byte()
		else:
			# [0, 255] range
			return frame.clamp(0, 255).byte()
	return frame


def adjust_brightness_cuda(image: CUDAFrame, factor: float) -> CUDAFrame:
	"""
	Adjust image brightness on GPU.

	Args:
		image: Input image (H, W, C) in float32 [0, 255]
		factor: Brightness factor (1.0 = no change, >1 = brighter, <1 = darker)

	Returns:
		Brightness-adjusted image
	"""
	adjusted = image * factor
	return torch.clamp(adjusted, 0, 255)


def adjust_contrast_cuda(image: CUDAFrame, factor: float) -> CUDAFrame:
	"""
	Adjust image contrast on GPU.

	Args:
		image: Input image (H, W, C) in float32 [0, 255]
		factor: Contrast factor (1.0 = no change, >1 = more contrast, <1 = less)

	Returns:
		Contrast-adjusted image
	"""
	# Calculate mean intensity
	mean = image.mean()

	# Adjust contrast around mean
	adjusted = mean + factor * (image - mean)

	return torch.clamp(adjusted, 0, 255)


def adjust_saturation_cuda(image: CUDAFrame, factor: float) -> CUDAFrame:
	"""
	Adjust image saturation on GPU.

	Args:
		image: BGR image (H, W, 3) in float32 [0, 255]
		factor: Saturation factor (1.0 = no change, >1 = more saturated, 0 = grayscale)

	Returns:
		Saturation-adjusted image
	"""
	# Convert to grayscale
	grayscale = bgr_to_grayscale_cuda(image)

	# Expand grayscale to 3 channels
	grayscale_3ch = grayscale.unsqueeze(-1).expand_as(image)

	# Blend between grayscale and original based on factor
	adjusted = grayscale_3ch + factor * (image - grayscale_3ch)

	return torch.clamp(adjusted, 0, 255)
