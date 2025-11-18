"""
GPU-accelerated mask operations using custom CUDA kernels and PyTorch.

This module provides CUDA implementations of mask creation and manipulation
operations for GPU-resident processing pipelines.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from facefusion.gpu_types import CUDAFrame, CUDAMask
from facefusion.types import FaceLandmark68, FaceMaskArea, FaceMaskRegion, Padding


def create_box_mask_cuda(
	crop_vision_frame: CUDAFrame,
	face_mask_blur: float,
	face_mask_padding: Padding,
	device: str = 'cuda:0'
) -> CUDAMask:
	"""
	Create a box mask with padding and blur on GPU.

	Args:
		crop_vision_frame: Input frame as CUDA tensor (H, W, C)
		face_mask_blur: Blur amount (0.0-1.0)
		face_mask_padding: Padding (top, right, bottom, left) as percentages
		device: CUDA device string

	Returns:
		Box mask as CUDA tensor (H, W) in float32 [0, 1]
	"""
	# Ensure device is correct
	if isinstance(crop_vision_frame, torch.Tensor):
		device = str(crop_vision_frame.device)

	# Get crop size (W, H)
	h, w = crop_vision_frame.shape[:2]

	# Calculate blur parameters
	blur_amount = int(w * 0.5 * face_mask_blur)
	blur_area = max(blur_amount // 2, 1)

	# Create mask filled with ones (always float32 for masks)
	box_mask = torch.ones((h, w), dtype=torch.float32, device=device)

	# Calculate padding in pixels
	top_pad = max(blur_area, int(h * face_mask_padding[0] / 100))
	right_pad = max(blur_area, int(w * face_mask_padding[1] / 100))
	bottom_pad = max(blur_area, int(h * face_mask_padding[2] / 100))
	left_pad = max(blur_area, int(w * face_mask_padding[3] / 100))

	# Zero out padded regions
	box_mask[:top_pad, :] = 0
	box_mask[-bottom_pad:, :] = 0
	box_mask[:, :left_pad] = 0
	box_mask[:, -right_pad:] = 0

	# Apply Gaussian blur if needed
	if blur_amount > 0:
		box_mask = gaussian_blur_cuda(box_mask, sigma=blur_amount * 0.25)

	return box_mask


def gaussian_blur_cuda(
	mask: CUDAMask,
	sigma: float,
	kernel_size: int = 0
) -> CUDAMask:
	"""
	Apply Gaussian blur to a mask on GPU using separable convolution.

	Args:
		mask: Input mask (H, W)
		sigma: Gaussian kernel standard deviation
		kernel_size: Kernel size (0 for auto-calculation)

	Returns:
		Blurred mask (H, W)
	"""
	if kernel_size == 0:
		# Auto-calculate kernel size (OpenCV compatible)
		kernel_size = int(2 * (4 * sigma + 0.5) + 1)
		kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

	# Ensure odd kernel size
	if kernel_size % 2 == 0:
		kernel_size += 1

	# Create Gaussian kernel
	kernel_1d = _create_gaussian_kernel_1d(kernel_size, sigma, device=mask.device)

	# Reshape mask for conv2d: (1, 1, H, W)
	mask_4d = mask.unsqueeze(0).unsqueeze(0)

	# Separable convolution: horizontal then vertical
	# Horizontal pass
	kernel_h = kernel_1d.view(1, 1, 1, -1)
	padding = kernel_size // 2
	mask_4d = F.conv2d(mask_4d, kernel_h, padding=(0, padding))

	# Vertical pass
	kernel_v = kernel_1d.view(1, 1, -1, 1)
	mask_4d = F.conv2d(mask_4d, kernel_v, padding=(padding, 0))

	# Remove batch and channel dimensions
	return mask_4d.squeeze(0).squeeze(0)


def _create_gaussian_kernel_1d(kernel_size: int, sigma: float, device: str = 'cuda:0') -> Tensor:
	"""
	Create 1D Gaussian kernel.

	Args:
		kernel_size: Size of the kernel (must be odd)
		sigma: Standard deviation
		device: CUDA device

	Returns:
		1D Gaussian kernel tensor
	"""
	# Create coordinate grid
	coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
	coords -= kernel_size // 2

	# Compute Gaussian
	kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
	kernel /= kernel.sum()

	return kernel


def resize_cuda(
	frame: CUDAFrame,
	target_size: Tuple[int, int],
	mode: str = 'bilinear'
) -> CUDAFrame:
	"""
	Resize a frame on GPU using interpolation.

	Args:
		frame: Input frame (H, W, C) or (H, W)
		target_size: Target size as (width, height)
		mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')

	Returns:
		Resized frame
	"""
	# Check if grayscale or color
	is_grayscale = frame.ndim == 2

	if is_grayscale:
		# Add batch and channel dimensions: (1, 1, H, W)
		frame_4d = frame.unsqueeze(0).unsqueeze(0)
	else:
		# Permute to (C, H, W) then add batch: (1, C, H, W)
		frame_4d = frame.permute(2, 0, 1).unsqueeze(0)

	# Resize using PyTorch interpolation
	# target_size is (W, H) but size parameter expects (H, W)
	resized = F.interpolate(
		frame_4d,
		size=(target_size[1], target_size[0]),
		mode=mode,
		align_corners=False if mode != 'nearest' else None
	)

	if is_grayscale:
		# Remove batch and channel dimensions
		return resized.squeeze(0).squeeze(0)
	else:
		# Remove batch and permute back to (H, W, C)
		return resized.squeeze(0).permute(1, 2, 0)


def normalize_frame_cuda(
	frame: CUDAFrame,
	mean: List[float],
	std: List[float]
) -> CUDAFrame:
	"""
	Normalize a frame with mean and std on GPU.

	Args:
		frame: Input frame (H, W, C) with values in [0, 1]
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


def fillConvexPoly_cuda(
	mask_size: Tuple[int, int],
	points: Tensor,
	device: str = 'cuda:0'
) -> CUDAMask:
	"""
	Fill a convex polygon on GPU using scanline algorithm.

	Args:
		mask_size: Mask size as (width, height)
		points: Convex hull points (N, 2) as x,y coordinates
		device: CUDA device

	Returns:
		Binary mask with filled polygon
	"""
	w, h = mask_size

	# Create empty mask
	mask = torch.zeros((h, w), dtype=torch.float32, device=device)

	# Ensure points are on the correct device
	if not points.is_cuda:
		points = points.to(device)

	# Convert to float for sub-pixel accuracy
	points = points.float()

	# Scanline fill using rasterization
	# For each row, find intersections and fill between them
	for y in range(h):
		y_line = float(y)
		intersections = []

		# Find intersections with each edge
		n = points.shape[0]
		for i in range(n):
			p1 = points[i]
			p2 = points[(i + 1) % n]

			x1, y1 = p1[0].item(), p1[1].item()
			x2, y2 = p2[0].item(), p2[1].item()

			# Check if edge crosses this scanline
			if (y1 <= y_line < y2) or (y2 <= y_line < y1):
				# Compute x intersection
				if y2 != y1:
					x_intersect = x1 + (y_line - y1) * (x2 - x1) / (y2 - y1)
					intersections.append(x_intersect)

		# Sort intersections
		intersections.sort()

		# Fill between pairs of intersections
		for i in range(0, len(intersections), 2):
			if i + 1 < len(intersections):
				x_start = int(max(0, intersections[i]))
				x_end = int(min(w - 1, intersections[i + 1]))
				if x_start <= x_end:
					mask[y, x_start:x_end + 1] = 1.0

	return mask


def create_area_mask_cuda(
	crop_vision_frame: CUDAFrame,
	face_landmark_68: FaceLandmark68,
	face_mask_areas: List[FaceMaskArea],
	device: str = 'cuda:0'
) -> CUDAMask:
	"""
	Create an area mask from facial landmarks on GPU.

	Args:
		crop_vision_frame: Input frame (H, W, C)
		face_landmark_68: 68-point facial landmarks (68, 2)
		face_mask_areas: List of face mask areas to include
		device: CUDA device

	Returns:
		Area mask (H, W)
	"""
	import facefusion.choices

	h, w = crop_vision_frame.shape[:2]
	landmark_points = []

	# Collect landmark indices for requested areas
	for face_mask_area in face_mask_areas:
		if face_mask_area in facefusion.choices.face_mask_area_set:
			landmark_points.extend(facefusion.choices.face_mask_area_set.get(face_mask_area))

	# Extract selected landmarks
	selected_landmarks = face_landmark_68[landmark_points]

	# Compute convex hull (CPU operation, then move to GPU)
	import cv2
	import numpy
	convex_hull = cv2.convexHull(selected_landmarks.astype(numpy.int32))
	convex_hull_tensor = torch.from_numpy(convex_hull.squeeze()).to(device)

	# Fill convex polygon on GPU
	area_mask = fillConvexPoly_cuda((w, h), convex_hull_tensor, device=device)

	# Apply Gaussian blur and thresholding
	area_mask = gaussian_blur_cuda(area_mask, sigma=5.0)
	area_mask = torch.clamp(area_mask, 0.0, 1.0)
	area_mask = torch.clamp(area_mask, 0.5, 1.0)
	area_mask = (area_mask - 0.5) * 2.0

	return area_mask


def minimum_reduce_cuda(masks: List[CUDAMask]) -> CUDAMask:
	"""
	Compute element-wise minimum across multiple masks (intersection).

	Args:
		masks: List of masks to intersect

	Returns:
		Minimum mask (intersection of all masks)
	"""
	if len(masks) == 0:
		raise ValueError("Cannot reduce empty mask list")

	if len(masks) == 1:
		return masks[0]

	# Stack masks and compute minimum along the first dimension
	stacked = torch.stack(masks, dim=0)
	return torch.min(stacked, dim=0)[0]


def apply_mask_threshold_cuda(mask: CUDAMask, clip_min: float = 0.0, clip_max: float = 1.0) -> CUDAMask:
	"""
	Apply Gaussian blur and thresholding to a mask (standard FaceFusion pattern).

	Args:
		mask: Input mask (H, W)
		clip_min: Minimum clip value (default: 0.5)
		clip_max: Maximum clip value (default: 1.0)

	Returns:
		Processed mask
	"""
	# Gaussian blur
	blurred = gaussian_blur_cuda(mask, sigma=5.0)

	# Clip to [clip_min, clip_max]
	clipped = torch.clamp(blurred, clip_min, clip_max)

	# Rescale: (clipped - clip_min) * (1 / (clip_max - clip_min)) * 2
	# For clip_min=0.5, clip_max=1.0: (x - 0.5) * 2
	if clip_min == 0.5 and clip_max == 1.0:
		return (clipped - 0.5) * 2.0
	else:
		# General formula
		range_scale = 2.0 / (clip_max - clip_min)
		return (clipped - clip_min) * range_scale


def bgr_to_rgb_cuda(frame: CUDAFrame) -> CUDAFrame:
	"""
	Convert BGR to RGB on GPU by flipping the channel dimension.

	Args:
		frame: BGR frame (H, W, C) with C=3

	Returns:
		RGB frame (H, W, C)
	"""
	# Flip the last dimension (channels)
	return frame.flip(-1)


def rgb_to_bgr_cuda(frame: CUDAFrame) -> CUDAFrame:
	"""
	Convert RGB to BGR on GPU by flipping the channel dimension.

	Args:
		frame: RGB frame (H, W, C) with C=3

	Returns:
		BGR frame (H, W, C)
	"""
	# Same operation as BGR to RGB (symmetric)
	return frame.flip(-1)


def prepare_occlusion_input_cuda(
	crop_vision_frame: CUDAFrame,
	model_size: Tuple[int, int]
) -> CUDAFrame:
	"""
	Prepare input for occlusion model inference on GPU.

	Args:
		crop_vision_frame: Input frame (H, W, C) in uint8 [0, 255]
		model_size: Model input size (W, H)

	Returns:
		Prepared frame (1, H, W, C) in float32 [0, 1]
	"""
	# Resize to model size
	resized = resize_cuda(crop_vision_frame, model_size, mode='bilinear')

	# Convert to float [0, 1]
	if resized.dtype == torch.uint8:
		resized = resized.float() / 255.0

	# Add batch dimension: (H, W, C) -> (1, H, W, C)
	return resized.unsqueeze(0)


def prepare_region_input_cuda(
	crop_vision_frame: CUDAFrame,
	model_size: Tuple[int, int]
) -> CUDAFrame:
	"""
	Prepare input for region parser model inference on GPU.

	Args:
		crop_vision_frame: Input frame (H, W, C) in BGR format
		model_size: Model input size (W, H)

	Returns:
		Prepared frame (1, C, H, W) normalized for BiSeNet
	"""
	# Resize to model size
	resized = resize_cuda(crop_vision_frame, model_size, mode='bilinear')

	# Convert BGR to RGB
	rgb_frame = bgr_to_rgb_cuda(resized)

	# Convert to float [0, 1]
	if rgb_frame.dtype == torch.uint8:
		rgb_frame = rgb_frame.float() / 255.0

	# Normalize with ImageNet mean and std
	normalized = normalize_frame_cuda(
		rgb_frame,
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)

	# Permute to (C, H, W) and add batch dimension: (1, C, H, W)
	return normalized.permute(2, 0, 1).unsqueeze(0)


def process_region_output_cuda(
	region_mask: CUDAMask,
	face_mask_regions: List[FaceMaskRegion],
	target_size: Tuple[int, int]
) -> CUDAMask:
	"""
	Process region parser output to create final mask on GPU.

	Args:
		region_mask: Raw model output (C, H, W) with class logits
		face_mask_regions: List of regions to include in mask
		target_size: Target mask size (W, H)

	Returns:
		Processed region mask (H, W)
	"""
	import facefusion.choices

	# Get argmax to find predicted class for each pixel: (H, W)
	class_map = torch.argmax(region_mask, dim=0)

	# Create binary mask for selected regions
	region_indices = [
		facefusion.choices.face_mask_region_set.get(face_mask_region)
		for face_mask_region in face_mask_regions
	]

	# Create mask where pixels belong to selected regions
	binary_mask = torch.zeros_like(class_map, dtype=torch.float32)
	for region_idx in region_indices:
		if region_idx is not None:
			binary_mask = torch.logical_or(binary_mask, class_map == region_idx)

	binary_mask = binary_mask.float()

	# Resize to target size
	resized_mask = resize_cuda(binary_mask, target_size, mode='bilinear')

	# Apply standard thresholding
	final_mask = apply_mask_threshold_cuda(resized_mask, clip_min=0.5, clip_max=1.0)

	return final_mask
