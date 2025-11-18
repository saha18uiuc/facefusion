"""
GPU-aware face masker module that dispatches to CPU or GPU implementations.

This module provides a unified interface for mask creation that automatically
uses GPU operations when GPU mode is enabled.
"""

from typing import List

from facefusion import state_manager
from facefusion.face_masker import (
	create_box_mask as create_box_mask_cpu,
	create_occlusion_mask as create_occlusion_mask_cpu,
	create_area_mask as create_area_mask_cpu,
	create_region_mask as create_region_mask_cpu
)
from facefusion.gpu_types import (
	get_processing_mode,
	is_cuda_tensor,
	ensure_cuda_tensor,
	ensure_numpy_array
)
from facefusion.types import FaceLandmark68, FaceMaskArea, FaceMaskRegion, Mask, Padding, VisionFrame


def create_box_mask(
	crop_vision_frame: VisionFrame,
	face_mask_blur: float,
	face_mask_padding: Padding
) -> Mask:
	"""
	Create box mask using GPU or CPU based on processing mode.

	Args:
		crop_vision_frame: Input frame
		face_mask_blur: Blur amount
		face_mask_padding: Padding (top, right, bottom, left)

	Returns:
		Box mask
	"""
	processing_mode = get_processing_mode()

	if processing_mode == 'gpu' and is_cuda_tensor(crop_vision_frame):
		from facefusion.gpu_mask_ops import create_box_mask_cuda

		device = str(crop_vision_frame.device)
		mask_cuda = create_box_mask_cuda(
			crop_vision_frame,
			face_mask_blur,
			face_mask_padding,
			device=device
		)

		# Return as NumPy if needed for compatibility
		return ensure_numpy_array(mask_cuda)
	else:
		# CPU path
		frame_np = ensure_numpy_array(crop_vision_frame)
		return create_box_mask_cpu(frame_np, face_mask_blur, face_mask_padding)


def create_occlusion_mask(crop_vision_frame: VisionFrame) -> Mask:
	"""
	Create occlusion mask using GPU or CPU based on processing mode.

	Args:
		crop_vision_frame: Input frame

	Returns:
		Occlusion mask
	"""
	# Occlusion mask requires ONNX inference which handles GPU internally
	# For now, use CPU implementation with NumPy arrays
	frame_np = ensure_numpy_array(crop_vision_frame)
	return create_occlusion_mask_cpu(frame_np)


def create_area_mask(
	crop_vision_frame: VisionFrame,
	face_landmark_68: FaceLandmark68,
	face_mask_areas: List[FaceMaskArea]
) -> Mask:
	"""
	Create area mask using GPU or CPU based on processing mode.

	Args:
		crop_vision_frame: Input frame
		face_landmark_68: 68-point facial landmarks
		face_mask_areas: List of areas to include

	Returns:
		Area mask
	"""
	processing_mode = get_processing_mode()

	if processing_mode == 'gpu' and is_cuda_tensor(crop_vision_frame):
		from facefusion.gpu_mask_ops import create_area_mask_cuda

		device = str(crop_vision_frame.device)
		mask_cuda = create_area_mask_cuda(
			crop_vision_frame,
			face_landmark_68,
			face_mask_areas,
			device=device
		)

		# Return as NumPy if needed for compatibility
		return ensure_numpy_array(mask_cuda)
	else:
		# CPU path
		frame_np = ensure_numpy_array(crop_vision_frame)
		return create_area_mask_cpu(frame_np, face_landmark_68, face_mask_areas)


def create_region_mask(
	crop_vision_frame: VisionFrame,
	face_mask_regions: List[FaceMaskRegion]
) -> Mask:
	"""
	Create region mask using GPU or CPU based on processing mode.

	Args:
		crop_vision_frame: Input frame
		face_mask_regions: List of regions to include

	Returns:
		Region mask
	"""
	# Region mask requires ONNX inference which handles GPU internally
	# For now, use CPU implementation with NumPy arrays
	frame_np = ensure_numpy_array(crop_vision_frame)
	return create_region_mask_cpu(frame_np, face_mask_regions)
