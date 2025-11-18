"""
GPU-accelerated blending operations using PyTorch and custom CUDA kernels.

This module provides CUDA implementations of advanced blending techniques including
multiband (Laplacian pyramid) blending, color correction, and Poisson blending.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from facefusion.gpu_types import CUDAFrame, CUDAMask
from facefusion.gpu_mask_ops import gaussian_blur_cuda, resize_cuda


def create_gaussian_pyramid_cuda(
	image: CUDAFrame,
	levels: int
) -> List[CUDAFrame]:
	"""
	Create a Gaussian pyramid on GPU.

	Args:
		image: Input image (H, W, C) or (H, W)
		levels: Number of pyramid levels

	Returns:
		List of downsampled images, from original size to smallest
	"""
	pyramid = [image]

	for level in range(levels - 1):
		# Blur current level
		blurred = gaussian_blur_cuda_separable_3ch(pyramid[-1], sigma=1.0)

		# Downsample by 2x
		downsampled = resize_cuda(
			blurred,
			(pyramid[-1].shape[1] // 2, pyramid[-1].shape[0] // 2),
			mode='bilinear'
		)

		pyramid.append(downsampled)

	return pyramid


def create_laplacian_pyramid_cuda(
	image: CUDAFrame,
	levels: int
) -> List[CUDAFrame]:
	"""
	Create a Laplacian pyramid on GPU.

	Args:
		image: Input image (H, W, C) in float32
		levels: Number of pyramid levels

	Returns:
		List of Laplacian images (high-frequency detail at each level)
	"""
	# Create Gaussian pyramid
	gaussian_pyramid = create_gaussian_pyramid_cuda(image, levels + 1)

	laplacian_pyramid = []

	for i in range(levels):
		# Get current and next (smaller) Gaussian level
		current = gaussian_pyramid[i]
		next_level = gaussian_pyramid[i + 1]

		# Upsample next level to match current size
		upsampled = resize_cuda(
			next_level,
			(current.shape[1], current.shape[0]),
			mode='bilinear'
		)

		# Laplacian = current - upsampled_next
		laplacian = current - upsampled
		laplacian_pyramid.append(laplacian)

	# Add the smallest Gaussian level as the residual
	laplacian_pyramid.append(gaussian_pyramid[-1])

	return laplacian_pyramid


def reconstruct_from_laplacian_cuda(laplacian_pyramid: List[CUDAFrame]) -> CUDAFrame:
	"""
	Reconstruct image from Laplacian pyramid on GPU.

	Args:
		laplacian_pyramid: List of Laplacian levels plus final residual

	Returns:
		Reconstructed image
	"""
	# Start with smallest level (residual)
	reconstructed = laplacian_pyramid[-1]

	# Iterate backward through pyramid levels
	for i in range(len(laplacian_pyramid) - 2, -1, -1):
		# Upsample reconstructed to match current level
		target_h, target_w = laplacian_pyramid[i].shape[:2]
		upsampled = resize_cuda(
			reconstructed,
			(target_w, target_h),
			mode='bilinear'
		)

		# Add current Laplacian level
		reconstructed = upsampled + laplacian_pyramid[i]

	return reconstructed


def gaussian_blur_cuda_separable_3ch(
	image: CUDAFrame,
	sigma: float,
	kernel_size: int = 0
) -> CUDAFrame:
	"""
	Apply Gaussian blur to 3-channel image on GPU.

	Args:
		image: Input image (H, W, C) with C=3
		sigma: Gaussian kernel standard deviation
		kernel_size: Kernel size (0 for auto-calculation)

	Returns:
		Blurred image
	"""
	# Handle both grayscale and color images
	if image.ndim == 2:
		return gaussian_blur_cuda(image, sigma, kernel_size)

	# Blur each channel separately
	h, w, c = image.shape
	result = torch.zeros_like(image)

	for ch in range(c):
		result[:, :, ch] = gaussian_blur_cuda(image[:, :, ch], sigma, kernel_size)

	return result


def multiband_blend_cuda(
	source: CUDAFrame,
	target: CUDAFrame,
	mask: CUDAMask,
	levels: int = 5
) -> CUDAFrame:
	"""
	Multi-band (Laplacian pyramid) blending for seamless integration on GPU.

	Args:
		source: Source image (H, W, C) in float32 [0, 255]
		target: Target image (H, W, C) in float32 [0, 255]
		mask: Blending mask (H, W) or (H, W, 3) in float32 [0, 1]
		levels: Number of pyramid levels

	Returns:
		Blended image in float32 [0, 255]
	"""
	# Ensure mask is single channel
	if mask.ndim == 3:
		mask_single = mask[:, :, 0]
	else:
		mask_single = mask

	# Create Laplacian pyramids for source and target
	source_pyramid = create_laplacian_pyramid_cuda(source, levels)
	target_pyramid = create_laplacian_pyramid_cuda(target, levels)

	# Create Gaussian pyramid for mask
	mask_pyramid = create_gaussian_pyramid_cuda(mask_single, levels)

	# Blend at each pyramid level
	blended_pyramid = []
	for level in range(levels):
		# Expand mask to 3 channels for this level
		level_mask = mask_pyramid[level].unsqueeze(-1).expand_as(source_pyramid[level])

		# Blend at this level
		blended_level = source_pyramid[level] * level_mask + target_pyramid[level] * (1 - level_mask)
		blended_pyramid.append(blended_level)

	# Reconstruct image from blended pyramid
	result = reconstruct_from_laplacian_cuda(blended_pyramid)

	# Clip to valid range
	result = torch.clamp(result, 0, 255)

	return result


def basic_blend_cuda(
	source: CUDAFrame,
	target: CUDAFrame,
	mask: CUDAMask
) -> CUDAFrame:
	"""
	Basic alpha blending on GPU.

	Args:
		source: Source image (H, W, C)
		target: Target image (H, W, C)
		mask: Blending mask (H, W) or (H, W, 3) in float32 [0, 1]

	Returns:
		Blended image
	"""
	# Ensure mask has channel dimension
	if mask.ndim == 2:
		mask = mask.unsqueeze(-1)

	# Blend: source * mask + target * (1 - mask)
	blended = source * mask + target * (1 - mask)

	return blended


def color_correction_cuda(
	source: CUDAFrame,
	target: CUDAFrame,
	mask: CUDAMask
) -> CUDAFrame:
	"""
	Transfer color statistics from target to source region on GPU.

	Args:
		source: Source image (H, W, C) in BGR format, float32 [0, 255]
		target: Target image (H, W, C) in BGR format, float32 [0, 255]
		mask: Binary mask (H, W) or (H, W, 3) in float32 [0, 1]

	Returns:
		Color-corrected source image
	"""
	# Convert BGR to LAB
	source_lab = bgr_to_lab_cuda(source)
	target_lab = bgr_to_lab_cuda(target)

	# Create binary mask for statistics calculation
	if mask.ndim == 3:
		mask_binary = (mask[:, :, 0] > 0.5).float()
	else:
		mask_binary = (mask > 0.5).float()

	# Expand to 3 channels
	mask_3ch = mask_binary.unsqueeze(-1).expand_as(source_lab)

	# Process each LAB channel
	result = source_lab.clone()

	for channel in range(3):
		source_channel = source_lab[:, :, channel]
		target_channel = target_lab[:, :, channel]
		channel_mask = mask_binary

		# Calculate masked statistics
		source_masked = source_channel * channel_mask
		target_masked = target_channel * channel_mask

		mask_sum = channel_mask.sum()

		if mask_sum > 0:
			# Source statistics
			source_mean = source_masked.sum() / mask_sum
			source_variance = ((source_channel - source_mean) ** 2 * channel_mask).sum() / mask_sum
			source_std = torch.sqrt(source_variance + 1e-6)

			# Target statistics
			target_mean = target_masked.sum() / mask_sum
			target_variance = ((target_channel - target_mean) ** 2 * channel_mask).sum() / mask_sum
			target_std = torch.sqrt(target_variance + 1e-6)

			# Apply color transfer: scale and shift
			result[:, :, channel] = (source_channel - source_mean) * (target_std / source_std) + target_mean

	# Clip values to valid LAB range
	result[:, :, 0] = torch.clamp(result[:, :, 0], 0, 100)     # L channel
	result[:, :, 1] = torch.clamp(result[:, :, 1], -127, 127)  # A channel
	result[:, :, 2] = torch.clamp(result[:, :, 2], -127, 127)  # B channel

	# Convert back to BGR
	corrected = lab_to_bgr_cuda(result)

	return corrected


def bgr_to_lab_cuda(image: CUDAFrame) -> CUDAFrame:
	"""
	Convert BGR image to LAB color space on GPU.

	Args:
		image: BGR image (H, W, 3) in float32 [0, 255]

	Returns:
		LAB image (H, W, 3) in float32
	"""
	# Normalize to [0, 1]
	bgr_normalized = image / 255.0

	# BGR to RGB
	rgb = bgr_normalized.flip(-1)

	# RGB to XYZ
	# Apply gamma correction
	rgb_linear = torch.where(
		rgb > 0.04045,
		((rgb + 0.055) / 1.055) ** 2.4,
		rgb / 12.92
	)

	# RGB to XYZ transformation matrix (D65 illuminant)
	transform_matrix = torch.tensor([
		[0.4124564, 0.3575761, 0.1804375],
		[0.2126729, 0.7151522, 0.0721750],
		[0.0193339, 0.1191920, 0.9503041]
	], dtype=torch.float32, device=image.device)

	# Reshape for matrix multiplication: (H*W, 3)
	h, w, c = rgb_linear.shape
	rgb_flat = rgb_linear.view(-1, 3)

	# XYZ = RGB @ transform_matrix.T
	xyz_flat = rgb_flat @ transform_matrix.T

	# Reshape back: (H, W, 3)
	xyz = xyz_flat.view(h, w, 3)

	# Normalize by D65 white point
	white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32, device=image.device)
	xyz_normalized = xyz / white_point.view(1, 1, 3)

	# XYZ to LAB
	# Apply f function
	epsilon = 0.008856
	kappa = 903.3

	xyz_f = torch.where(
		xyz_normalized > epsilon,
		xyz_normalized ** (1.0 / 3.0),
		(kappa * xyz_normalized + 16.0) / 116.0
	)

	# Calculate L, A, B
	L = 116.0 * xyz_f[:, :, 1] - 16.0
	A = 500.0 * (xyz_f[:, :, 0] - xyz_f[:, :, 1])
	B = 200.0 * (xyz_f[:, :, 1] - xyz_f[:, :, 2])

	# Stack to form LAB image
	lab = torch.stack([L, A, B], dim=-1)

	return lab


def lab_to_bgr_cuda(lab: CUDAFrame) -> CUDAFrame:
	"""
	Convert LAB image to BGR color space on GPU.

	Args:
		lab: LAB image (H, W, 3) in float32

	Returns:
		BGR image (H, W, 3) in float32 [0, 255]
	"""
	# Extract L, A, B channels
	L = lab[:, :, 0]
	A = lab[:, :, 1]
	B = lab[:, :, 2]

	# LAB to XYZ
	# Calculate f_y, f_x, f_z
	f_y = (L + 16.0) / 116.0
	f_x = A / 500.0 + f_y
	f_z = f_y - B / 200.0

	# Inverse f function
	epsilon = 0.008856
	kappa = 903.3

	xyz_normalized = torch.stack([
		torch.where(f_x ** 3 > epsilon, f_x ** 3, (116.0 * f_x - 16.0) / kappa),
		torch.where(f_y ** 3 > epsilon, f_y ** 3, (116.0 * f_y - 16.0) / kappa),
		torch.where(f_z ** 3 > epsilon, f_z ** 3, (116.0 * f_z - 16.0) / kappa)
	], dim=-1)

	# Denormalize by D65 white point
	white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32, device=lab.device)
	xyz = xyz_normalized * white_point.view(1, 1, 3)

	# XYZ to RGB transformation matrix (inverse of RGB to XYZ)
	inverse_transform = torch.tensor([
		[ 3.2404542, -1.5371385, -0.4985314],
		[-0.9692660,  1.8760108,  0.0415560],
		[ 0.0556434, -0.2040259,  1.0572252]
	], dtype=torch.float32, device=lab.device)

	# Reshape for matrix multiplication: (H*W, 3)
	h, w, c = xyz.shape
	xyz_flat = xyz.view(-1, 3)

	# RGB = XYZ @ inverse_transform.T
	rgb_linear_flat = xyz_flat @ inverse_transform.T

	# Reshape back: (H, W, 3)
	rgb_linear = rgb_linear_flat.view(h, w, 3)

	# Apply inverse gamma correction
	rgb = torch.where(
		rgb_linear > 0.0031308,
		1.055 * (rgb_linear ** (1.0 / 2.4)) - 0.055,
		12.92 * rgb_linear
	)

	# Clip to valid range
	rgb = torch.clamp(rgb, 0, 1)

	# RGB to BGR
	bgr = rgb.flip(-1)

	# Denormalize to [0, 255]
	bgr_denormalized = bgr * 255.0

	return bgr_denormalized


def enhanced_mask_cuda(
	base_mask: CUDAMask,
	frame_shape: Tuple[int, int],
	feather_amount: int = 15,
	num_blur_passes: int = 3
) -> CUDAMask:
	"""
	Create enhanced mask with smooth feathering on GPU.

	Args:
		base_mask: Base mask (H, W) in float32 [0, 1]
		frame_shape: Frame shape (height, width)
		feather_amount: Pixel amount for feathering
		num_blur_passes: Number of blur passes for smoothing

	Returns:
		Enhanced mask (H, W, 3) in float32 [0, 1]
	"""
	mask = base_mask.clone()

	# Ensure proper range [0, 1]
	if mask.max() > 1.0:
		mask = mask / 255.0

	# Apply multiple blur passes for smooth feathering
	for _ in range(num_blur_passes):
		mask = gaussian_blur_cuda(mask, sigma=float(feather_amount))

	# Erode slightly using max pooling (approximate erosion)
	kernel_size = feather_amount
	if kernel_size % 2 == 0:
		kernel_size += 1

	# Negative max pooling = erosion (invert, max pool, invert)
	mask_4d = (1 - mask).unsqueeze(0).unsqueeze(0)
	eroded_4d = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
	mask = 1 - eroded_4d.squeeze(0).squeeze(0)

	# Final blur for ultra-soft edge
	mask = gaussian_blur_cuda(mask, sigma=float(feather_amount * 2))

	# Normalize to [0, 1]
	mask = torch.clamp(mask, 0, 1)

	# Expand to 3 channels
	mask_3ch = torch.stack([mask] * 3, dim=-1)

	return mask_3ch
