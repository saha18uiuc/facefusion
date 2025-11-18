"""
GPU-accelerated face swapper implementation.

This module provides GPU-resident versions of face swapping operations,
keeping frames on GPU throughout the entire pipeline.
"""

from typing import List, Tuple

import cv2
import numpy
import torch

import facefusion.choices
from facefusion import face_masker, state_manager
from facefusion.face_helper import get_warp_template
from facefusion.gpu_types import (
	CUDAFrame,
	CUDAMask,
	ensure_cuda_tensor,
	ensure_numpy_array,
	is_cuda_tensor,
	get_cuda_device
)
from facefusion.gpu_mask_ops import (
	create_box_mask_cuda,
	create_area_mask_cuda,
	minimum_reduce_cuda,
	resize_cuda
)
from facefusion.gpu_face_helper import (
	warp_affine_cuda,
	paste_back_cuda,
	estimate_similarity_transform_cuda
)
from facefusion.gpu_blending import color_correction_cuda
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.types import Face, VisionFrame
from facefusion.vision import unpack_resolution


def swap_face_gpu(
	source_face: Face,
	target_face: Face,
	temp_vision_frame: CUDAFrame,
	model_template: str,
	model_size: Tuple[int, int],
	pixel_boost_size: Tuple[int, int],
	device: str = 'cuda:0'
) -> CUDAFrame:
	"""
	Swap face on GPU, keeping all tensors on GPU throughout.

	Args:
		source_face: Source face to swap from
		target_face: Target face to swap to
		temp_vision_frame: Current frame as CUDA tensor (H, W, C)
		model_template: Warp template name
		model_size: Model input size (W, H)
		pixel_boost_size: Pixel boost size (W, H)
		device: CUDA device

	Returns:
		Frame with swapped face as CUDA tensor
	"""
	pixel_boost_total = pixel_boost_size[0] // model_size[0]

	# Get smoothed landmarks (temporal smoother is CPU-based for now)
	from facefusion.processors.modules.face_swapper.core import get_temporal_smoother

	temporal_smoother = get_temporal_smoother()
	target_landmarks_5 = target_face.landmark_set.get('5/68')

	# Check for scene cuts
	if temporal_smoother.should_reset(target_landmarks_5):
		temporal_smoother.reset()

	# Smooth landmarks
	smoothed_landmarks_5 = temporal_smoother.smooth_landmarks(target_landmarks_5)

	# Warp face on GPU
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5_gpu(
		temp_vision_frame,
		smoothed_landmarks_5,
		model_template,
		pixel_boost_size,
		device
	)

	# Create masks on GPU
	crop_masks = []
	face_mask_types = state_manager.get_item('face_mask_types')

	if 'box' in face_mask_types:
		box_mask = create_box_mask_cuda(
			crop_vision_frame,
			state_manager.get_item('face_mask_blur'),
			state_manager.get_item('face_mask_padding'),
			device=device
		)
		crop_masks.append(box_mask)

	if 'occlusion' in face_mask_types:
		# Occlusion requires ONNX inference - use CPU version
		crop_np = ensure_numpy_array(crop_vision_frame)
		occlusion_mask = face_masker.create_occlusion_mask(crop_np)
		occlusion_mask_cuda = ensure_cuda_tensor(occlusion_mask, device=device)
		crop_masks.append(occlusion_mask_cuda)

	# Pixel boost and forward pass (CPU for now - ONNX inference)
	crop_vision_frame_np = ensure_numpy_array(crop_vision_frame)
	pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame_np, pixel_boost_total, model_size)

	temp_vision_frames = []
	for pixel_boost_vision_frame in pixel_boost_vision_frames:
		# ONNX inference happens here (CPU)
		from facefusion.processors.modules.face_swapper.core import (
			prepare_crop_frame,
			forward_swap_face,
			normalize_crop_frame
		)

		pixel_boost_vision_frame = prepare_crop_frame(pixel_boost_vision_frame)
		pixel_boost_vision_frame = forward_swap_face(source_face, target_face, pixel_boost_vision_frame)
		pixel_boost_vision_frame = normalize_crop_frame(pixel_boost_vision_frame)
		temp_vision_frames.append(pixel_boost_vision_frame)

	crop_vision_frame_np = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

	# Convert back to GPU
	crop_vision_frame = ensure_cuda_tensor(crop_vision_frame_np, device=device)

	if 'area' in face_mask_types:
		# Transform landmarks using affine matrix
		face_landmark_68 = cv2.transform(
			target_face.landmark_set.get('68').reshape(1, -1, 2),
			affine_matrix
		).reshape(-1, 2)

		area_mask = create_area_mask_cuda(
			crop_vision_frame,
			face_landmark_68,
			state_manager.get_item('face_mask_areas'),
			device=device
		)
		crop_masks.append(area_mask)

	if 'region' in face_mask_types:
		# Region mask requires ONNX inference - use CPU version
		crop_np = ensure_numpy_array(crop_vision_frame)
		region_mask = face_masker.create_region_mask(crop_np, state_manager.get_item('face_mask_regions'))
		region_mask_cuda = ensure_cuda_tensor(region_mask, device=device)
		crop_masks.append(region_mask_cuda)

	# Merge masks on GPU
	crop_mask = minimum_reduce_cuda(crop_masks)
	crop_mask = torch.clamp(crop_mask, 0, 1)

	# Paste back on GPU
	paste_vision_frame = paste_back_cuda(
		temp_vision_frame,
		crop_vision_frame,
		crop_mask,
		affine_matrix,
		device=device
	)

	# Enhanced blending on GPU
	if target_face.bounding_box is not None:
		try:
			paste_vision_frame = apply_enhanced_blending_gpu(
				paste_vision_frame,
				temp_vision_frame,
				target_face.bounding_box,
				device=device
			)
		except Exception as e:
			# Fallback to standard result
			import traceback
			print(f"Enhanced blending failed: {e}")
			traceback.print_exc()
			pass

	# Ensure output is uint8 in range [0, 255]
	if paste_vision_frame.dtype == torch.float32:
		paste_vision_frame = torch.clamp(paste_vision_frame, 0, 255).byte()
	elif paste_vision_frame.dtype != torch.uint8:
		paste_vision_frame = paste_vision_frame.byte()

	return paste_vision_frame


def warp_face_by_face_landmark_5_gpu(
	temp_vision_frame: CUDAFrame,
	face_landmark_5: numpy.ndarray,
	warp_template: str,
	crop_size: Tuple[int, int],
	device: str = 'cuda:0'
) -> Tuple[CUDAFrame, numpy.ndarray]:
	"""
	Warp face by 5-point landmarks on GPU.

	Args:
		temp_vision_frame: Input frame as CUDA tensor (H, W, C)
		face_landmark_5: 5-point landmarks (5, 2) as NumPy array
		warp_template: Template name
		crop_size: Output crop size (W, H)
		device: CUDA device

	Returns:
		Tuple of (warped_frame, affine_matrix)
	"""
	# Get warp template
	warp_template_np = get_warp_template(warp_template)
	warp_template_tensor = torch.from_numpy(warp_template_np).float().to(device)
	face_landmark_5_tensor = torch.from_numpy(face_landmark_5).float().to(device)

	# Estimate similarity transform on GPU
	affine_matrix = estimate_similarity_transform_cuda(
		face_landmark_5_tensor,
		warp_template_tensor,
		device=device
	)

	# Warp frame on GPU
	crop_vision_frame = warp_affine_cuda(
		temp_vision_frame,
		affine_matrix.cpu().numpy(),
		crop_size,
		mode='bilinear',
		padding_mode='zeros'
	)

	return crop_vision_frame, affine_matrix.cpu().numpy()


def apply_enhanced_blending_gpu(
	paste_vision_frame: CUDAFrame,
	temp_vision_frame: CUDAFrame,
	bounding_box: numpy.ndarray,
	device: str = 'cuda:0'
) -> CUDAFrame:
	"""
	Apply enhanced blending with color correction on GPU.

	Args:
		paste_vision_frame: Frame with pasted face (H, W, C)
		temp_vision_frame: Original frame (H, W, C)
		bounding_box: Face bounding box (x1, y1, x2, y2)
		device: CUDA device

	Returns:
		Blended frame
	"""
	# Get face center
	bbox = bounding_box
	center_x = int((bbox[0] + bbox[2]) / 2)
	center_y = int((bbox[1] + bbox[3]) / 2)

	# Extract face region
	x1, y1, x2, y2 = map(int, bbox)
	margin = 20
	x1 = max(0, x1 - margin)
	y1 = max(0, y1 - margin)
	x2 = min(temp_vision_frame.shape[1], x2 + margin)
	y2 = min(temp_vision_frame.shape[0], y2 + margin)

	# Extract regions on GPU
	swapped_region = paste_vision_frame[y1:y2, x1:x2]
	original_region = temp_vision_frame[y1:y2, x1:x2]

	# Convert to float32 for blending
	original_dtype = paste_vision_frame.dtype
	if swapped_region.dtype == torch.uint8:
		swapped_region = swapped_region.float()
	if original_region.dtype == torch.uint8:
		original_region = original_region.float()

	# Create mask for this region
	region_h = y2 - y1
	region_w = x2 - x1
	region_mask = torch.ones((region_h, region_w), dtype=torch.float32, device=device) * 0.8

	# Apply color correction on GPU
	blended_region = color_correction_cuda(
		swapped_region,
		original_region,
		region_mask
	)

	# Blend the corrected region back
	alpha = 0.7
	paste_vision_frame = paste_vision_frame.clone()

	# Convert paste_vision_frame to float32 if needed
	if paste_vision_frame.dtype == torch.uint8:
		paste_vision_frame = paste_vision_frame.float()

	paste_vision_frame[y1:y2, x1:x2] = (
		blended_region * alpha + paste_vision_frame[y1:y2, x1:x2] * (1 - alpha)
	)

	# Convert back to original dtype
	if original_dtype == torch.uint8:
		paste_vision_frame = torch.clamp(paste_vision_frame, 0, 255).byte()

	return paste_vision_frame


def process_frame_gpu(
	inputs: dict,
	device: str = 'cuda:0'
) -> Tuple[CUDAFrame, CUDAMask]:
	"""
	Process a single frame with GPU-resident tensors.

	Args:
		inputs: Input dictionary with CUDA tensors or NumPy arrays
		device: CUDA device

	Returns:
		Tuple of (processed_frame, mask) as CUDA tensors
	"""
	from facefusion.face_analyser import get_average_face, scale_face
	from facefusion.face_selector import select_faces
	from facefusion.processors.modules.face_swapper.core import extract_source_face

	# Extract inputs
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	# Ensure inputs are CUDA tensors
	temp_vision_frame_cuda = ensure_cuda_tensor(temp_vision_frame, device=device)
	temp_vision_mask_cuda = ensure_cuda_tensor(temp_vision_mask, device=device)

	# Face extraction and selection (CPU-based for now)
	source_face = extract_source_face(source_vision_frames)
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if source_face and target_faces:
		# Get model options
		from facefusion.processors.modules.face_swapper.core import get_model_options

		model_template = get_model_options().get('template')
		model_size = get_model_options().get('size')
		pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))

		for target_face in target_faces:
			# Scale face (CPU-based)
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)

			# Swap face on GPU
			temp_vision_frame_cuda = swap_face_gpu(
				source_face,
				target_face,
				temp_vision_frame_cuda,
				model_template,
				model_size,
				pixel_boost_size,
				device=device
			)

	return temp_vision_frame_cuda, temp_vision_mask_cuda
