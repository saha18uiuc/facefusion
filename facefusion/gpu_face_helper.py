"""
GPU-accelerated face helper operations using PyTorch and custom CUDA kernels.

This module provides CUDA implementations of affine transforms, face warping,
and face pasting operations for GPU-resident processing.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
import numpy

from facefusion.gpu_types import CUDAFrame, CUDAMask
from facefusion.types import BoundingBox, FaceLandmark5, Matrix


def warp_affine_cuda(
	frame: CUDAFrame,
	affine_matrix: Matrix,
	output_size: Tuple[int, int],
	mode: str = 'bilinear',
	padding_mode: str = 'zeros'
) -> CUDAFrame:
	"""
	Apply affine transformation to a frame on GPU.

	Args:
		frame: Input frame (H, W, C) or (H, W)
		affine_matrix: 2x3 affine transformation matrix (NumPy array)
		output_size: Output size as (width, height)
		mode: Interpolation mode ('bilinear', 'nearest')
		padding_mode: Padding mode ('zeros', 'border', 'reflection')

	Returns:
		Transformed frame with output_size
	"""
	device = frame.device
	is_grayscale = frame.ndim == 2
	original_dtype = frame.dtype

	# Convert to float32 for interpolation
	if frame.dtype == torch.uint8:
		frame = frame.float()

	# Convert affine matrix to PyTorch tensor on GPU
	if isinstance(affine_matrix, numpy.ndarray):
		affine_matrix = torch.from_numpy(affine_matrix).float().to(device)

	# Add batch dimension and permute to (N, C, H, W)
	if is_grayscale:
		frame_nchw = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
	else:
		frame_nchw = frame.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

	# Create affine grid
	# PyTorch affine_grid expects 2x3 matrix in normalized coordinates
	# Convert from pixel coordinates to normalized [-1, 1]
	theta = _convert_affine_matrix_to_theta(
		affine_matrix,
		input_size=(frame.shape[1], frame.shape[0]),  # (W, H)
		output_size=output_size
	)
	theta = theta.unsqueeze(0)  # Add batch dimension

	# Create sampling grid
	grid = F.affine_grid(
		theta,
		size=(1, frame_nchw.shape[1], output_size[1], output_size[0]),
		align_corners=False
	)

	# Sample using grid_sample
	output = F.grid_sample(
		frame_nchw,
		grid,
		mode=mode,
		padding_mode=padding_mode if padding_mode != 'border' else 'border',
		align_corners=False
	)

	# Remove batch dimension and permute back
	if is_grayscale:
		result = output.squeeze(0).squeeze(0)
	else:
		result = output.squeeze(0).permute(1, 2, 0)

	# Convert back to original dtype
	if original_dtype == torch.uint8:
		result = torch.clamp(result, 0, 255).byte()

	return result


def _convert_affine_matrix_to_theta(
	affine_matrix: torch.Tensor,
	input_size: Tuple[int, int],
	output_size: Tuple[int, int]
) -> torch.Tensor:
	"""
	Convert OpenCV-style affine matrix (pixel coords) to PyTorch theta (normalized).

	Args:
		affine_matrix: 2x3 affine matrix in pixel coordinates
		input_size: Input image size (W, H)
		output_size: Output image size (W, H)

	Returns:
		2x3 theta matrix for affine_grid
	"""
	# Create normalization matrices
	# From pixel to normalized [-1, 1]
	in_w, in_h = input_size
	out_w, out_h = output_size
	device = affine_matrix.device

	# Normalization matrix for output coordinates
	# Maps [0, out_w] -> [-1, 1] and [0, out_h] -> [-1, 1]
	T_out_inv = torch.tensor([
		[2.0 / out_w, 0, -1],
		[0, 2.0 / out_h, -1],
		[0, 0, 1]
	], dtype=torch.float32, device=device)

	# Denormalization matrix for input coordinates
	# Maps [-1, 1] -> [0, in_w] and [-1, 1] -> [0, in_h]
	T_in = torch.tensor([
		[in_w / 2.0, 0, in_w / 2.0],
		[0, in_h / 2.0, in_h / 2.0],
		[0, 0, 1]
	], dtype=torch.float32, device=device)

	# Extend affine matrix to 3x3
	affine_3x3 = torch.cat([
		affine_matrix,
		torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
	], dim=0)

	# Compute combined transformation: T_out_inv @ affine_3x3 @ T_in
	theta_3x3 = T_out_inv @ affine_3x3 @ T_in

	# Return 2x3 matrix
	return theta_3x3[:2, :]


def invert_affine_matrix_cuda(affine_matrix: Matrix, device: str = 'cuda:0') -> torch.Tensor:
	"""
	Invert a 2x3 affine transformation matrix on GPU.

	Args:
		affine_matrix: 2x3 affine matrix (NumPy array or torch.Tensor)
		device: CUDA device

	Returns:
		Inverted 2x3 affine matrix as torch.Tensor
	"""
	if isinstance(affine_matrix, numpy.ndarray):
		affine_matrix = torch.from_numpy(affine_matrix).float().to(device)

	# Extend to 3x3
	affine_3x3 = torch.cat([
		affine_matrix,
		torch.tensor([[0, 0, 1]], dtype=torch.float32, device=device)
	], dim=0)

	# Invert
	inverse_3x3 = torch.inverse(affine_3x3)

	# Return 2x3
	return inverse_3x3[:2, :]


def paste_back_cuda(
	temp_vision_frame: CUDAFrame,
	crop_vision_frame: CUDAFrame,
	crop_vision_mask: CUDAMask,
	affine_matrix: Matrix,
	device: str = 'cuda:0'
) -> CUDAFrame:
	"""
	Paste a cropped face back into the original frame using affine transform on GPU.

	Args:
		temp_vision_frame: Target frame to paste into (H, W, C)
		crop_vision_frame: Cropped face to paste (crop_H, crop_W, C)
		crop_vision_mask: Mask for the cropped face (crop_H, crop_W)
		affine_matrix: Original affine matrix used for cropping
		device: CUDA device

	Returns:
		Frame with pasted face
	"""
	# Calculate paste area
	paste_bounding_box, paste_matrix = calculate_paste_area_cuda(
		temp_vision_frame,
		crop_vision_frame,
		affine_matrix,
		device=device
	)

	x1, y1, x2, y2 = paste_bounding_box
	paste_width = x2 - x1
	paste_height = y2 - y1

	# Inverse warp the mask
	inverse_vision_mask = warp_affine_cuda(
		crop_vision_mask,
		paste_matrix.cpu().numpy() if isinstance(paste_matrix, torch.Tensor) else paste_matrix,
		(paste_width, paste_height),
		mode='bilinear',
		padding_mode='zeros'
	)
	inverse_vision_mask = torch.clamp(inverse_vision_mask, 0.0, 1.0)
	inverse_vision_mask = inverse_vision_mask.unsqueeze(-1)  # Add channel dimension

	# Inverse warp the face
	inverse_vision_frame = warp_affine_cuda(
		crop_vision_frame,
		paste_matrix.cpu().numpy() if isinstance(paste_matrix, torch.Tensor) else paste_matrix,
		(paste_width, paste_height),
		mode='bilinear',
		padding_mode='border'
	)

	# Clone temp frame for in-place modification
	temp_vision_frame = temp_vision_frame.clone()

	# Convert to float32 for blending if needed
	original_dtype = temp_vision_frame.dtype
	if temp_vision_frame.dtype == torch.uint8:
		temp_vision_frame = temp_vision_frame.float()
		inverse_vision_frame = inverse_vision_frame.float()

	# Extract paste region
	paste_vision_frame = temp_vision_frame[y1:y2, x1:x2, :]

	# Alpha blend (float32 math)
	paste_vision_frame = paste_vision_frame * (1 - inverse_vision_mask) + inverse_vision_frame * inverse_vision_mask

	# Paste back
	temp_vision_frame[y1:y2, x1:x2, :] = paste_vision_frame

	# Convert back to original dtype
	if original_dtype == torch.uint8:
		temp_vision_frame = torch.clamp(temp_vision_frame, 0, 255).byte()

	return temp_vision_frame


def calculate_paste_area_cuda(
	temp_vision_frame: CUDAFrame,
	crop_vision_frame: CUDAFrame,
	affine_matrix: Matrix,
	device: str = 'cuda:0'
) -> Tuple[BoundingBox, torch.Tensor]:
	"""
	Calculate the paste area and inverse affine matrix on GPU.

	Args:
		temp_vision_frame: Target frame (H, W, C)
		crop_vision_frame: Cropped frame (crop_H, crop_W, C)
		affine_matrix: Forward affine matrix (2x3)
		device: CUDA device

	Returns:
		Tuple of (paste_bounding_box, paste_matrix)
		- paste_bounding_box: (x1, y1, x2, y2) as NumPy array
		- paste_matrix: 2x3 inverse affine matrix as torch.Tensor
	"""
	temp_height, temp_width = temp_vision_frame.shape[:2]
	crop_height, crop_width = crop_vision_frame.shape[:2]

	# Invert affine matrix
	if isinstance(affine_matrix, numpy.ndarray):
		affine_matrix = torch.from_numpy(affine_matrix).float().to(device)

	inverse_matrix = invert_affine_matrix_cuda(affine_matrix, device=device)

	# Define crop corner points
	crop_points = torch.tensor([
		[0, 0],
		[crop_width, 0],
		[crop_width, crop_height],
		[0, crop_height]
	], dtype=torch.float32, device=device)

	# Transform points to find paste region
	paste_region_points = transform_points_cuda(crop_points, inverse_matrix)

	# Find bounding box
	paste_region_point_min = torch.floor(paste_region_points.min(dim=0)[0])
	paste_region_point_max = torch.ceil(paste_region_points.max(dim=0)[0])

	# Clip to frame boundaries
	x1 = int(torch.clamp(paste_region_point_min[0], 0, temp_width).item())
	y1 = int(torch.clamp(paste_region_point_min[1], 0, temp_height).item())
	x2 = int(torch.clamp(paste_region_point_max[0], 0, temp_width).item())
	y2 = int(torch.clamp(paste_region_point_max[1], 0, temp_height).item())

	paste_bounding_box = numpy.array([x1, y1, x2, y2])

	# Adjust paste matrix
	paste_matrix = inverse_matrix.clone()
	paste_matrix[0, 2] -= x1
	paste_matrix[1, 2] -= y1

	return paste_bounding_box, paste_matrix


def transform_points_cuda(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
	"""
	Transform points using affine matrix on GPU.

	Args:
		points: Points tensor (N, 2) in (x, y) format
		matrix: 2x3 affine transformation matrix

	Returns:
		Transformed points (N, 2)
	"""
	# Add homogeneous coordinate: (N, 2) -> (N, 3)
	ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
	points_homo = torch.cat([points, ones], dim=1)  # (N, 3)

	# Apply transformation: (2, 3) @ (N, 3).T = (2, N)
	transformed = matrix @ points_homo.T  # (2, N)

	# Transpose back to (N, 2)
	return transformed.T


def warp_face_by_face_landmark_5_cuda(
	temp_vision_frame: CUDAFrame,
	face_landmark_5: FaceLandmark5,
	warp_template: torch.Tensor,
	crop_size: Tuple[int, int],
	device: str = 'cuda:0'
) -> Tuple[CUDAFrame, Matrix]:
	"""
	Warp face by 5-point landmarks using similarity transform on GPU.

	Args:
		temp_vision_frame: Input frame (H, W, C)
		face_landmark_5: 5-point facial landmarks (5, 2) as NumPy array
		warp_template: Template landmarks (5, 2) as torch.Tensor
		crop_size: Output crop size (W, H)
		device: CUDA device

	Returns:
		Tuple of (warped_frame, affine_matrix)
	"""
	# Convert landmarks to torch tensor
	if isinstance(face_landmark_5, numpy.ndarray):
		face_landmark_5 = torch.from_numpy(face_landmark_5).float().to(device)

	if not warp_template.is_cuda:
		warp_template = warp_template.to(device)

	# Estimate similarity transform
	affine_matrix = estimate_similarity_transform_cuda(face_landmark_5, warp_template, device=device)

	# Convert to NumPy for warp_affine_cuda (which converts back internally)
	affine_matrix_np = affine_matrix.cpu().numpy()

	# Warp frame
	crop_vision_frame = warp_affine_cuda(
		temp_vision_frame,
		affine_matrix_np,
		crop_size,
		mode='bilinear',
		padding_mode='zeros'
	)

	return crop_vision_frame, affine_matrix_np


def estimate_similarity_transform_cuda(
	src_points: torch.Tensor,
	dst_points: torch.Tensor,
	device: str = 'cuda:0'
) -> torch.Tensor:
	"""
	Estimate similarity transformation (rotation, scale, translation) on GPU.

	Uses least-squares estimation to find the best affine transform that maps
	src_points to dst_points.

	Args:
		src_points: Source points (N, 2)
		dst_points: Destination points (N, 2)
		device: CUDA device

	Returns:
		2x3 affine transformation matrix
	"""
	# Ensure tensors are on the correct device
	if not src_points.is_cuda:
		src_points = src_points.to(device)
	if not dst_points.is_cuda:
		dst_points = dst_points.to(device)

	# Center the points
	src_mean = src_points.mean(dim=0, keepdim=True)
	dst_mean = dst_points.mean(dim=0, keepdim=True)

	src_centered = src_points - src_mean
	dst_centered = dst_points - dst_mean

	# Compute scale
	src_norm = torch.norm(src_centered, dim=1).mean()
	dst_norm = torch.norm(dst_centered, dim=1).mean()
	scale = dst_norm / (src_norm + 1e-8)

	# Normalize
	src_normalized = src_centered / (src_norm + 1e-8)
	dst_normalized = dst_centered / (dst_norm + 1e-8)

	# Compute rotation using SVD
	# H = src.T @ dst
	H = src_normalized.T @ dst_normalized

	U, _, Vt = torch.linalg.svd(H)
	R = Vt.T @ U.T

	# Handle reflection case
	if torch.det(R) < 0:
		Vt[-1, :] *= -1
		R = Vt.T @ U.T

	# Build affine matrix
	# [scale * R | t]
	# where t = dst_mean - scale * R @ src_mean
	scaled_R = scale * R
	t = dst_mean.T - scaled_R @ src_mean.T

	affine_matrix = torch.cat([scaled_R, t], dim=1)  # (2, 3)

	return affine_matrix


def blend_frames_cuda(
	frame1: CUDAFrame,
	frame2: CUDAFrame,
	mask: CUDAMask,
	alpha: float = 1.0
) -> CUDAFrame:
	"""
	Blend two frames using a mask on GPU.

	Args:
		frame1: First frame (H, W, C)
		frame2: Second frame (H, W, C)
		mask: Blending mask (H, W) with values in [0, 1]
		alpha: Global alpha for frame2 (default: 1.0)

	Returns:
		Blended frame
	"""
	# Ensure mask has channel dimension
	if mask.ndim == 2:
		mask = mask.unsqueeze(-1)

	# Blend: frame1 * (1 - mask) + frame2 * mask * alpha
	blended = frame1 * (1 - mask) + frame2 * mask * alpha

	return blended
