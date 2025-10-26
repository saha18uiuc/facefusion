from functools import lru_cache
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy
from cv2.typing import Size

from facefusion.types import Anchors, Angle, BoundingBox, Distance, FaceDetectorModel, FaceLandmark5, FaceLandmark68, Mask, Matrix, Points, Scale, Score, Translation, VisionFrame, WarpTemplate, WarpTemplateSet
from facefusion.temporal_filters import reset_all_filters as reset_alignment_filters, affine_to_se2, resolve_filter_key
from facefusion.thread_helper import thread_lock

# Detect CUDA support in OpenCV if available
def _has_cv2_cuda() -> bool:
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_POSE_ANGLE_THRESHOLD = math.radians(12.0)
_SCALE_THRESHOLD = 0.12
_FREEZE_FRAMES = 2
_POISSON_ITERATIONS = 6
_POISSON_LAMBDA = 3.0

_SEAM_STATE_LOCK = thread_lock()
_SEAM_STATES: Dict[str, '_SeamTemporalState'] = {}


class _SeamTemporalState:
	def __init__(self, height: int, width: int) -> None:
		self.height = int(height)
		self.width = int(width)
		self.prev_gray: Optional[numpy.ndarray] = None
		self.prev_mask: Optional[numpy.ndarray] = None
		self.freeze_counter: int = 0
		self.last_angle: Optional[float] = None
		self.last_scale: Optional[float] = None
		self.curr_gray: Optional[numpy.ndarray] = None
		self.curr_mask: Optional[numpy.ndarray] = None
		self.pending_angle: Optional[float] = None
		self.pending_scale: Optional[float] = None
		self.base_x, self.base_y = _make_base_grid(self.width, self.height)
		self.ofa = _create_ofa(self.width, self.height)

	def resize(self, height: int, width: int) -> None:
		height = int(height)
		width = int(width)
		if height != self.height or width != self.width:
			self.height = height
			self.width = width
			self.base_x, self.base_y = _make_base_grid(self.width, self.height)
			self.ofa = _create_ofa(self.width, self.height)
			self.prev_gray = None
			self.prev_mask = None
			self.freeze_counter = 0
			self.last_angle = None
			self.last_scale = None

	def prepare(self, raw_mask: numpy.ndarray, bg_roi: numpy.ndarray, theta: float, scale: float) -> Tuple[numpy.ndarray, numpy.ndarray]:
		curr_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
		mask = numpy.clip(raw_mask.astype(numpy.float32, copy = False), 0.0, 1.0)
		if self.prev_gray is not None and self.prev_mask is not None and self.prev_gray.shape == curr_gray.shape:
			flow = self._calc_flow(self.prev_gray, curr_gray)
			if flow is not None and flow.shape[:2] == mask.shape:
				propagated = _warp_with_flow(self, self.prev_mask.astype(numpy.float32, copy = False), flow)
				weight = _blend_weight_from_iou(propagated, mask)
				mask = numpy.clip(weight * propagated + (1.0 - weight) * mask, 0.0, 1.0)
		delta_angle = _angular_difference(theta, self.last_angle)
		delta_scale = abs(scale - self.last_scale) / max(abs(self.last_scale) if self.last_scale else 1.0, 1e-3) if self.last_scale is not None else 0.0
		if self.freeze_counter > 0 and self.prev_mask is not None:
			mask = numpy.clip(0.8 * self.prev_mask + 0.2 * mask, 0.0, 1.0)
			self.freeze_counter -= 1
		elif abs(delta_angle) > _POSE_ANGLE_THRESHOLD or delta_scale > _SCALE_THRESHOLD:
			if self.prev_mask is not None:
				mask = numpy.clip(0.85 * self.prev_mask + 0.15 * mask, 0.0, 1.0)
			self.freeze_counter = _FREEZE_FRAMES
		self.curr_gray = curr_gray
		self.curr_mask = mask
		self.pending_angle = theta
		self.pending_scale = scale
		return mask, curr_gray

	def finalize(self) -> None:
		if self.curr_gray is not None:
			self.prev_gray = self.curr_gray
		if self.curr_mask is not None:
			self.prev_mask = self.curr_mask.copy()
		else:
			self.prev_mask = None
		self.last_angle = self.pending_angle
		self.last_scale = self.pending_scale
		self.curr_gray = None
		self.curr_mask = None
		self.pending_angle = None
		self.pending_scale = None

	def _calc_flow(self, prev_gray: numpy.ndarray, curr_gray: numpy.ndarray) -> Optional[numpy.ndarray]:
		if prev_gray.shape != curr_gray.shape:
			return None
		if self.ofa is not None:
			try:
				prev_gpu = cv2.cuda_GpuMat()
				curr_gpu = cv2.cuda_GpuMat()
				prev_gpu.upload(prev_gray)
				curr_gpu.upload(curr_gray)
				flow_gpu = self.ofa.calc(prev_gpu, curr_gpu, None)
				dense_gpu = self.ofa.convertToDense(flow_gpu)
				flow = dense_gpu.download()
				return flow
			except Exception:
				self.ofa = None
		try:
			flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 2, 21, 3, 5, 1.2, 0)
			return flow
		except Exception:
			return None


def _make_base_grid(width: int, height: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	grid_x, grid_y = numpy.meshgrid(numpy.arange(width, dtype = numpy.float32), numpy.arange(height, dtype = numpy.float32))
	return grid_x, grid_y


def _create_ofa(width: int, height: int):
	if not _has_cv2_cuda():
		return None
	if not hasattr(cv2, 'cuda') or not hasattr(cv2.cuda, 'NvidiaOpticalFlow_2_0'):
		return None
	try:
		return cv2.cuda_NvidiaOpticalFlow_2_0.create(
			(width, height),
			perfPreset = cv2.cuda.NVIDIA_OF_PERF_LEVEL_FAST,
			outputGridSize = cv2.cuda.NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE_4,
			enableTemporalHints = True,
			enableExternalHints = False,
			enableCostBuffer = False
		)
	except Exception:
		return None


def _warp_with_flow(state: _SeamTemporalState, image: numpy.ndarray, flow: numpy.ndarray) -> numpy.ndarray:
	if image.shape[:2] != (state.height, state.width):
		image = cv2.resize(image, (state.width, state.height), interpolation = cv2.INTER_LINEAR)
	map_x = state.base_x + flow[..., 0].astype(numpy.float32)
	map_y = state.base_y + flow[..., 1].astype(numpy.float32)
	return cv2.remap(image, map_x, map_y, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT101, borderValue = 0)


def _mask_iou(mask_a: numpy.ndarray, mask_b: numpy.ndarray) -> float:
	a_bin = mask_a > 0.5
	b_bin = mask_b > 0.5
	intersection = numpy.logical_and(a_bin, b_bin).sum()
	union = numpy.logical_or(a_bin, b_bin).sum()
	if union == 0:
		return 1.0
	return float(intersection) / float(union)


def _blend_weight_from_iou(propagated: numpy.ndarray, current: numpy.ndarray) -> float:
	iou = _mask_iou(propagated, current)
	if iou >= 0.85:
		return 0.2
	if iou >= 0.70:
		return 0.35
	if iou >= 0.50:
		return 0.5
	return 0.7


def _angular_difference(current: float, previous: Optional[float]) -> float:
	if previous is None:
		return 0.0
	delta = current - previous
	while delta <= -math.pi:
		delta += 2.0 * math.pi
	while delta > math.pi:
		delta -= 2.0 * math.pi
	return delta


def _build_ring_mask(mask: numpy.ndarray) -> Optional[numpy.ndarray]:
	if mask.size == 0:
		return None
	mask_uint8 = numpy.clip(mask * 255.0, 0.0, 255.0).astype(numpy.uint8)
	height, width = mask_uint8.shape
	max_dim = max(height, width)
	outer = max(9, (max_dim // 6) | 1)
	inner = max(5, ((outer // 2) | 1))
	outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer, outer))
	inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner, inner))
	dilated = cv2.dilate(mask_uint8, outer_kernel, iterations = 1)
	eroded = cv2.erode(mask_uint8, inner_kernel, iterations = 1)
	ring = cv2.subtract(dilated, eroded)
	if not numpy.any(ring):
		return None
	return ring.astype(bool)


def _apply_screened_poisson_ring(composite: numpy.ndarray, background: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
	ring = _build_ring_mask(mask)
	if ring is None:
		return composite
	src = composite.astype(numpy.float32, copy = False)
	tgt = background.astype(numpy.float32, copy = False)
	kernel = numpy.array([
		[0.0, 1.0, 0.0],
		[1.0, 0.0, 1.0],
		[0.0, 1.0, 0.0]
	], dtype = numpy.float32) * 0.25
	ring_idx = ring.astype(bool)
	ring_idx3 = ring_idx[..., None]
	for _ in range(_POISSON_ITERATIONS):
		lap = cv2.filter2D(src, -1, kernel, borderType = cv2.BORDER_REFLECT101)
		updated = (lap + _POISSON_LAMBDA * tgt) / (1.0 + _POISSON_LAMBDA)
		src[ring_idx3] = updated[ring_idx3]
	return numpy.clip(src, 0.0, 255.0).astype(composite.dtype)


def _get_seam_state(track_token: Optional[str], shape: Tuple[int, int]) -> Optional[_SeamTemporalState]:
	if not track_token:
		return None
	key = resolve_filter_key(track_token)
	height, width = shape
	with _SEAM_STATE_LOCK:
		state = _SEAM_STATES.get(key)
		if state is None:
			state = _SeamTemporalState(height, width)
			_SEAM_STATES[key] = state
		else:
			state.resize(height, width)
	return state


def _reset_seam_states() -> None:
	with _SEAM_STATE_LOCK:
		_SEAM_STATES.clear()

WARP_TEMPLATE_SET : WarpTemplateSet =\
{
	'arcface_112_v1': numpy.array(
	[
		[ 0.35473214, 0.45658929 ],
		[ 0.64526786, 0.45658929 ],
		[ 0.50000000, 0.61154464 ],
		[ 0.37913393, 0.77687500 ],
		[ 0.62086607, 0.77687500 ]
	]),
	'arcface_112_v2': numpy.array(
	[
		[ 0.34191607, 0.46157411 ],
		[ 0.65653393, 0.45983393 ],
		[ 0.50022500, 0.64050536 ],
		[ 0.37097589, 0.82469196 ],
		[ 0.63151696, 0.82325089 ]
	]),
	'arcface_128': numpy.array(
	[
		[ 0.36167656, 0.40387734 ],
		[ 0.63696719, 0.40235469 ],
		[ 0.50019687, 0.56044219 ],
		[ 0.38710391, 0.72160547 ],
		[ 0.61507734, 0.72034453 ]
	]),
	'dfl_whole_face': numpy.array(
	[
		[ 0.35342266, 0.39285716 ],
		[ 0.62797622, 0.39285716 ],
		[ 0.48660713, 0.54017860 ],
		[ 0.38839287, 0.68750011 ],
		[ 0.59821427, 0.68750011 ]
	]),
	'ffhq_512': numpy.array(
	[
		[ 0.37691676, 0.46864664 ],
		[ 0.62285697, 0.46912813 ],
		[ 0.50123859, 0.61331904 ],
		[ 0.39308822, 0.72541100 ],
		[ 0.61150205, 0.72490465 ]
	]),
	'mtcnn_512': numpy.array(
	[
		[ 0.36562865, 0.46733799 ],
		[ 0.63305391, 0.46585885 ],
		[ 0.50019127, 0.61942959 ],
		[ 0.39032951, 0.77598822 ],
		[ 0.61178945, 0.77476328 ]
	]),
	'styleganex_384': numpy.array(
	[
		[ 0.42353745, 0.52289879 ],
		[ 0.57725008, 0.52319972 ],
		[ 0.50123859, 0.61331904 ],
		[ 0.43364461, 0.68337652 ],
		[ 0.57015325, 0.68306005 ]
	])
}


def _normalize_mask(mask : Mask) -> numpy.ndarray:
	if mask.size == 0:
		return numpy.zeros_like(mask, dtype = numpy.float32)
	return numpy.clip(mask.astype(numpy.float32), 0.0, 1.0)


def _feather_mask(mask : Mask, feather_strength : float = 1.6) -> numpy.ndarray:
	mask_f32 = _normalize_mask(mask)
	if mask_f32.size == 0:
		return mask_f32
	feather = max(0.6, float(feather_strength))
	return cv2.GaussianBlur(mask_f32, ksize = (0, 0), sigmaX = feather, sigmaY = feather)


def _color_correct_face(face_roi : VisionFrame, background_roi : VisionFrame, mask : Mask) -> VisionFrame:
	mask_f32 = _normalize_mask(mask)
	if mask_f32.size == 0:
		return face_roi
	mask_binary = mask_f32 > 0.05
	if not mask_binary.any():
		return face_roi
	if face_roi.dtype != numpy.uint8:
		face_roi = numpy.clip(face_roi, 0, 255).astype(numpy.uint8, copy=False)
	if background_roi.dtype != numpy.uint8:
		background_roi = numpy.clip(background_roi, 0, 255).astype(numpy.uint8, copy=False)
	face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB).astype(numpy.float32)
	background_lab = cv2.cvtColor(background_roi, cv2.COLOR_BGR2LAB).astype(numpy.float32)
	flat_mask = mask_binary.reshape(-1)
	face_pixels = face_lab.reshape(-1, 3)[flat_mask]
	bg_pixels = background_lab.reshape(-1, 3)[flat_mask]
	if face_pixels.shape[0] < 32 or bg_pixels.shape[0] < 32:
		return face_roi
	face_mean = face_pixels.mean(axis = 0)
	face_std = face_pixels.std(axis = 0)
	bg_mean = bg_pixels.mean(axis = 0)
	bg_std = bg_pixels.std(axis = 0)
	face_std = numpy.where(face_std < 1e-3, 1.0, face_std)
	scale = bg_std / face_std
	corrected_lab = (face_lab - face_mean) * scale + bg_mean
	corrected_lab = numpy.clip(corrected_lab, 0.0, 255.0).astype(numpy.uint8)
	corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
	result = face_roi.copy()
	result[mask_binary] = corrected_bgr[mask_binary]
	return result


def _linear_light_blend(background_roi : VisionFrame, face_roi : VisionFrame, mask_3c : numpy.ndarray) -> VisionFrame:
	if mask_3c.size == 0:
		return face_roi
	bg_f = numpy.clip(background_roi.astype(numpy.float32) / 255.0, 0.0, 1.0)
	face_f = numpy.clip(face_roi.astype(numpy.float32) / 255.0, 0.0, 1.0)
	mask_f = numpy.clip(mask_3c.astype(numpy.float32), 0.0, 1.0)
	bg_linear = numpy.power(bg_f + 1e-6, 2.2)
	face_linear = numpy.power(face_f + 1e-6, 2.2)
	blended_linear = bg_linear * (1.0 - mask_f) + face_linear * mask_f
	blended = numpy.power(numpy.clip(blended_linear, 0.0, 1.0), 1.0 / 2.2)
	return numpy.clip(blended * 255.0, 0.0, 255.0).astype(background_roi.dtype)


def reset_affine_smoothers() -> None:
	reset_alignment_filters()


def reset_cpu_temporal_states() -> None:
	_reset_seam_states()


def estimate_matrix_by_face_landmark_5(face_landmark_5 : FaceLandmark5, warp_template : WarpTemplate, crop_size : Size) -> Matrix:
	warp_template_norm = WARP_TEMPLATE_SET.get(warp_template) * crop_size
	affine_matrix = cv2.estimateAffinePartial2D(face_landmark_5, warp_template_norm, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
	return affine_matrix


def warp_face_with_matrix(temp_vision_frame: VisionFrame, affine_matrix: Matrix, crop_size: Size) -> Tuple[VisionFrame, Matrix]:
	"""Warp ``temp_vision_frame`` using ``affine_matrix`` and return the crop and matrix."""
	affine_matrix = numpy.asarray(affine_matrix, dtype=numpy.float32)
	if _has_cv2_cuda():
		try:
			gpu_img = cv2.cuda_GpuMat()
			gpu_img.upload(temp_vision_frame)
			crop_gpu = cv2.cuda.warpAffine(gpu_img, affine_matrix, crop_size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
			crop_vision_frame = crop_gpu.download()
			return crop_vision_frame, affine_matrix
		except Exception:
			pass
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
	return crop_vision_frame, affine_matrix


def warp_face_by_face_landmark_5(temp_vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5, warp_template : WarpTemplate, crop_size : Size) -> Tuple[VisionFrame, Matrix]:
	affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)
	return warp_face_with_matrix(temp_vision_frame, affine_matrix, crop_size)


def warp_face_by_bounding_box(temp_vision_frame : VisionFrame, bounding_box : BoundingBox, crop_size : Size) -> Tuple[VisionFrame, Matrix]:
	source_points = numpy.array([ [ bounding_box[0], bounding_box[1] ], [bounding_box[2], bounding_box[1] ], [ bounding_box[0], bounding_box[3] ] ]).astype(numpy.float32)
	target_points = numpy.array([ [ 0, 0 ], [ crop_size[0], 0 ], [ 0, crop_size[1] ] ]).astype(numpy.float32)
	affine_matrix = cv2.getAffineTransform(source_points, target_points)
	if bounding_box[2] - bounding_box[0] > crop_size[0] or bounding_box[3] - bounding_box[1] > crop_size[1]:
		interpolation_method = cv2.INTER_AREA
	else:
		interpolation_method = cv2.INTER_LINEAR
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, flags = interpolation_method)
	return crop_vision_frame, affine_matrix


def warp_face_by_translation(temp_vision_frame : VisionFrame, translation : Translation, scale : float, crop_size : Size) -> Tuple[VisionFrame, Matrix]:
	affine_matrix = numpy.array([ [ scale, 0, translation[0] ], [ 0, scale, translation[1] ] ])
	crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
	return crop_vision_frame, affine_matrix


def paste_back(temp_vision_frame : VisionFrame,
			crop_vision_frame : VisionFrame,
			crop_mask : Mask,
			affine_matrix : Matrix,
			track_token : Optional[str] = None,
			**_unused : object) -> VisionFrame:
	paste_bounding_box, paste_matrix = calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix)
	x1, y1, x2, y2 = paste_bounding_box
	paste_width = x2 - x1
	paste_height = y2 - y1
	if paste_width <= 0 or paste_height <= 0:
		return temp_vision_frame

	inverse_mask : Optional[numpy.ndarray] = None
	inverse_vision_frame : Optional[numpy.ndarray] = None

	if _has_cv2_cuda():
		try:
			gpu_mask = cv2.cuda_GpuMat()
			gpu_mask.upload(crop_mask.astype(numpy.float32))
			inv_mask_gpu = cv2.cuda.warpAffine(gpu_mask, paste_matrix, (paste_width, paste_height))
			inverse_mask = inv_mask_gpu.download()

			gpu_crop = cv2.cuda_GpuMat()
			gpu_crop.upload(crop_vision_frame)
			inv_frame_gpu = cv2.cuda.warpAffine(gpu_crop, paste_matrix, (paste_width, paste_height), borderMode = cv2.BORDER_REPLICATE)
			inverse_vision_frame = inv_frame_gpu.download()
		except Exception:
			inverse_mask = None
			inverse_vision_frame = None

	if inverse_mask is None or inverse_vision_frame is None:
		inverse_mask = cv2.warpAffine(crop_mask, paste_matrix, (paste_width, paste_height))
		inverse_vision_frame = cv2.warpAffine(crop_vision_frame, paste_matrix, (paste_width, paste_height), borderMode = cv2.BORDER_REPLICATE)

	inverse_mask = numpy.clip(inverse_mask.astype(numpy.float32, copy = False), 0.0, 1.0)
	if inverse_mask.ndim == 3:
		inverse_mask = inverse_mask[..., 0]

	output_frame = temp_vision_frame.copy()
	background_roi = output_frame[y1:y2, x1:x2]
	background_copy = background_roi.copy()
	components = affine_to_se2(affine_matrix)
	state = _get_seam_state(track_token, (paste_height, paste_width))
	if state is not None:
		try:
			stabilized_mask, _ = state.prepare(inverse_mask, background_copy, components.theta, components.scale)
		except Exception:
			state = None
			stabilized_mask = inverse_mask
	else:
		stabilized_mask = inverse_mask

	feathered_mask = _feather_mask(stabilized_mask)
	corrected_face = _color_correct_face(inverse_vision_frame, background_copy, feathered_mask)
	mask3 = cv2.merge([ feathered_mask, feathered_mask, feathered_mask ])
	blend = _linear_light_blend(background_copy, corrected_face, mask3)
	if state is not None:
		try:
			blend = _apply_screened_poisson_ring(blend, background_copy, stabilized_mask)
			state.finalize()
		except Exception:
			state.finalize()

	output_frame[y1:y2, x1:x2] = blend.astype(output_frame.dtype)
	return output_frame


def calculate_paste_area(temp_vision_frame : VisionFrame, crop_vision_frame : VisionFrame, affine_matrix : Matrix) -> Tuple[BoundingBox, Matrix]:
	temp_height, temp_width = temp_vision_frame.shape[:2]
	crop_height, crop_width = crop_vision_frame.shape[:2]
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	crop_points = numpy.array([ [ 0, 0 ], [ crop_width, 0 ], [ crop_width, crop_height ], [ 0, crop_height ] ])
	paste_region_points = transform_points(crop_points, inverse_matrix)
	paste_region_point_min = numpy.floor(paste_region_points.min(axis = 0)).astype(int)
	paste_region_point_max = numpy.ceil(paste_region_points.max(axis = 0)).astype(int)
	x1, y1 = numpy.clip(paste_region_point_min, 0, [ temp_width, temp_height ])
	x2, y2 = numpy.clip(paste_region_point_max, 0, [ temp_width, temp_height ])
	paste_bounding_box = numpy.array([ x1, y1, x2, y2 ])
	paste_matrix = inverse_matrix.copy()
	paste_matrix[0, 2] -= x1
	paste_matrix[1, 2] -= y1
	return paste_bounding_box, paste_matrix


@lru_cache()
def create_static_anchors(feature_stride : int, anchor_total : int, stride_height : int, stride_width : int) -> Anchors:
	x, y = numpy.mgrid[:stride_width, :stride_height]
	anchors = numpy.stack((y, x), axis = -1)
	anchors = (anchors * feature_stride).reshape((-1, 2))
	anchors = numpy.stack([ anchors ] * anchor_total, axis = 1).reshape((-1, 2))
	return anchors


def create_rotation_matrix_and_size(angle : Angle, size : Size) -> Tuple[Matrix, Size]:
	rotation_matrix = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1)
	rotation_size = numpy.dot(numpy.abs(rotation_matrix[:, :2]), size)
	rotation_matrix[:, -1] += (rotation_size - size) * 0.5 #type:ignore[misc]
	rotation_size = int(rotation_size[0]), int(rotation_size[1])
	return rotation_matrix, rotation_size


def create_bounding_box(face_landmark_68 : FaceLandmark68) -> BoundingBox:
	x1, y1 = numpy.min(face_landmark_68, axis = 0)
	x2, y2 = numpy.max(face_landmark_68, axis = 0)
	bounding_box = normalize_bounding_box(numpy.array([ x1, y1, x2, y2 ]))
	return bounding_box


def normalize_bounding_box(bounding_box : BoundingBox) -> BoundingBox:
	x1, y1, x2, y2 = bounding_box
	x1, x2 = sorted([ x1, x2 ])
	y1, y2 = sorted([ y1, y2 ])
	return numpy.array([ x1, y1, x2, y2 ])


def transform_points(points : Points, matrix : Matrix) -> Points:
	points = points.reshape(-1, 1, 2)
	points = cv2.transform(points, matrix) #type:ignore[assignment]
	points = points.reshape(-1, 2)
	return points


def transform_bounding_box(bounding_box : BoundingBox, matrix : Matrix) -> BoundingBox:
	points = numpy.array(
	[
		[ bounding_box[0], bounding_box[1] ],
		[ bounding_box[2], bounding_box[1] ],
		[ bounding_box[2], bounding_box[3] ],
		[ bounding_box[0], bounding_box[3] ]
	])
	points = transform_points(points, matrix)
	x1, y1 = numpy.min(points, axis = 0)
	x2, y2 = numpy.max(points, axis = 0)
	return normalize_bounding_box(numpy.array([ x1, y1, x2, y2 ]))


def distance_to_bounding_box(points : Points, distance : Distance) -> BoundingBox:
	x1 = points[:, 0] - distance[:, 0]
	y1 = points[:, 1] - distance[:, 1]
	x2 = points[:, 0] + distance[:, 2]
	y2 = points[:, 1] + distance[:, 3]
	bounding_box = numpy.column_stack([ x1, y1, x2, y2 ])
	return bounding_box


def distance_to_face_landmark_5(points : Points, distance : Distance) -> FaceLandmark5:
	x = points[:, 0::2] + distance[:, 0::2]
	y = points[:, 1::2] + distance[:, 1::2]
	face_landmark_5 = numpy.stack((x, y), axis = -1)
	return face_landmark_5


def scale_face_landmark_5(face_landmark_5 : FaceLandmark5, scale : Scale) -> FaceLandmark5:
	face_landmark_5_scale = face_landmark_5 - face_landmark_5[2]
	face_landmark_5_scale *= scale
	face_landmark_5_scale += face_landmark_5[2]
	return face_landmark_5_scale


def convert_to_face_landmark_5(face_landmark_68 : FaceLandmark68) -> FaceLandmark5:
	face_landmark_5 = numpy.array(
	[
		numpy.mean(face_landmark_68[36:42], axis = 0),
		numpy.mean(face_landmark_68[42:48], axis = 0),
		face_landmark_68[30],
		face_landmark_68[48],
		face_landmark_68[54]
	])
	return face_landmark_5


def estimate_face_angle(face_landmark_68 : FaceLandmark68) -> Angle:
	x1, y1 = face_landmark_68[0]
	x2, y2 = face_landmark_68[16]
	theta = numpy.arctan2(y2 - y1, x2 - x1)
	theta = numpy.degrees(theta) % 360
	angles = numpy.linspace(0, 360, 5)
	index = numpy.argmin(numpy.abs(angles - theta))
	face_angle = int(angles[index] % 360)
	return face_angle


def apply_nms(bounding_boxes : List[BoundingBox], scores : List[Score], score_threshold : float, nms_threshold : float) -> Sequence[int]:
	bounding_boxes_norm = [ (x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bounding_boxes ]
	keep_indices = cv2.dnn.NMSBoxes(bounding_boxes_norm, scores, score_threshold = score_threshold, nms_threshold = nms_threshold)
	return keep_indices


def get_nms_threshold(face_detector_model : FaceDetectorModel, face_detector_angles : List[Angle]) -> float:
	if face_detector_model == 'many':
		return 0.1
	if len(face_detector_angles) == 2:
		return 0.3
	if len(face_detector_angles) == 3:
		return 0.2
	if len(face_detector_angles) == 4:
		return 0.1
	return 0.4


def merge_matrix(temp_matrices : List[Matrix]) -> Matrix:
	matrix = numpy.vstack([temp_matrices[0], [0, 0, 1]])

	for temp_matrix in temp_matrices[1:]:
		temp_matrix = numpy.vstack([ temp_matrix, [ 0, 0, 1 ] ])
		matrix = numpy.dot(temp_matrix, matrix)

	return matrix[:2, :]
