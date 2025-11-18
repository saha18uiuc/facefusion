"""
GPU-aware face helper module that dispatches to CPU or GPU implementations.

This module provides a unified interface for face operations that automatically
uses GPU operations when GPU mode is enabled.
"""

from typing import Tuple

from facefusion import state_manager
from facefusion.face_helper import (
	warp_face_by_face_landmark_5 as warp_face_by_face_landmark_5_cpu,
	paste_back as paste_back_cpu
)
from facefusion.gpu_types import (
	get_processing_mode,
	is_cuda_tensor,
	ensure_cuda_tensor,
	ensure_numpy_array,
	get_cuda_device
)
from facefusion.types import FaceLandmark5, Mask, Matrix, VisionFrame, WarpTemplate


def warp_face_by_face_landmark_5(
	temp_vision_frame: VisionFrame,
	face_landmark_5: FaceLandmark5,
	warp_template: WarpTemplate,
	crop_size: Tuple[int, int]
) -> Tuple[VisionFrame, Matrix]:
	"""
	Warp face by 5-point landmarks using GPU or CPU based on processing mode.

	Args:
		temp_vision_frame: Input frame
		face_landmark_5: 5-point facial landmarks
		warp_template: Template name
		crop_size: Output crop size (width, height)

	Returns:
		Tuple of (warped_frame, affine_matrix)
	"""
	processing_mode = get_processing_mode()

	if processing_mode == 'gpu' and is_cuda_tensor(temp_vision_frame):
		from facefusion.gpu_face_helper import warp_face_by_face_landmark_5_cuda
		from facefusion.face_helper import get_warp_template
		import torch

		device = str(temp_vision_frame.device)

		# Get warp template as torch tensor
		warp_template_np = get_warp_template(warp_template)
		warp_template_tensor = torch.from_numpy(warp_template_np).float().to(device)

		crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5_cuda(
			temp_vision_frame,
			face_landmark_5,
			warp_template_tensor,
			crop_size,
			device=device
		)

		# Return frame in same format as input
		return crop_vision_frame, affine_matrix
	else:
		# CPU path
		from facefusion.face_helper import get_warp_template

		frame_np = ensure_numpy_array(temp_vision_frame)
		warp_template_np = get_warp_template(warp_template)

		crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5_cpu(
			frame_np,
			face_landmark_5,
			warp_template,
			crop_size
		)

		# If input was CUDA, convert output back to CUDA
		if is_cuda_tensor(temp_vision_frame):
			device = str(temp_vision_frame.device)
			crop_vision_frame = ensure_cuda_tensor(crop_vision_frame, device=device)

		return crop_vision_frame, affine_matrix


def paste_back(
	temp_vision_frame: VisionFrame,
	crop_vision_frame: VisionFrame,
	crop_vision_mask: Mask,
	affine_matrix: Matrix
) -> VisionFrame:
	"""
	Paste cropped face back into original frame using GPU or CPU.

	Args:
		temp_vision_frame: Target frame to paste into
		crop_vision_frame: Cropped face to paste
		crop_vision_mask: Mask for the cropped face
		affine_matrix: Affine transformation matrix

	Returns:
		Frame with pasted face
	"""
	processing_mode = get_processing_mode()

	if processing_mode == 'gpu' and is_cuda_tensor(temp_vision_frame):
		from facefusion.gpu_face_helper import paste_back_cuda

		device = str(temp_vision_frame.device)

		# Ensure all inputs are CUDA tensors on the same device
		crop_vision_frame_cuda = ensure_cuda_tensor(crop_vision_frame, device=device)
		crop_vision_mask_cuda = ensure_cuda_tensor(crop_vision_mask, device=device)

		result = paste_back_cuda(
			temp_vision_frame,
			crop_vision_frame_cuda,
			crop_vision_mask_cuda,
			affine_matrix,
			device=device
		)

		return result
	else:
		# CPU path
		temp_frame_np = ensure_numpy_array(temp_vision_frame)
		crop_frame_np = ensure_numpy_array(crop_vision_frame)
		crop_mask_np = ensure_numpy_array(crop_vision_mask)

		result = paste_back_cpu(
			temp_frame_np,
			crop_frame_np,
			crop_mask_np,
			affine_matrix
		)

		# If input was CUDA, convert output back to CUDA
		if is_cuda_tensor(temp_vision_frame):
			device = str(temp_vision_frame.device)
			result = ensure_cuda_tensor(result, device=device)

		return result
