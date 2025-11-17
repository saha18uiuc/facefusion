"""
Lightweight landmark tracker to reduce expensive per-frame face detection.
Pulls from revert_to_8392 branch (fast path) without altering output quality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, List, Optional

import cv2
import numpy

from facefusion import state_manager
from facefusion.face_helper import convert_to_face_landmark_5, estimate_face_angle, reset_affine_smoothers
from facefusion.types import Face, VisionFrame

try:
	from facefusion.gpu.compositor import reset_all_temporal_states  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	reset_all_temporal_states = None  # type: ignore


@dataclass
class _TrackedFace:
	track_id: int
	face: Face
	points_68: numpy.ndarray
	misses: int = 0
	last_detection_frame: int = 0
	confidence_history: List[float] = None

	def __post_init__(self) -> None:
		if self.confidence_history is None:
			self.confidence_history = []


class _OneEuroFilter:
	def __init__(self, fps: float = 30.0, min_cutoff: float = 1.2, beta: float = 0.05, dcut: float = 1.0) -> None:
		self._te = 1.0 / max(1e-3, fps)
		self._min_cutoff = float(min_cutoff)
		self._beta = float(beta)
		self._dcut = float(dcut)
		self._x_prev: Optional[float] = None
		self._dx_prev: float = 0.0

	def _alpha(self, cutoff: float) -> float:
		tau = 1.0 / (2.0 * math.pi * cutoff)
		return 1.0 / (1.0 + tau / self._te)

	def reset(self, value: float) -> None:
		self._x_prev = value
		self._dx_prev = 0.0

	def filter(self, value: float) -> float:
		if self._x_prev is None:
			self.reset(value)
			return value
		dx = (value - self._x_prev) / self._te
		alpha_d = self._alpha(self._dcut)
		self._dx_prev = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
		cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
		alpha = self._alpha(cutoff)
		filtered = alpha * value + (1.0 - alpha) * self._x_prev
		self._x_prev = filtered
		return filtered


class FaceTracker:
	def __init__(self) -> None:
		self._tracks: Dict[int, _TrackedFace] = {}
		self._next_track_id = 0
		self._prev_gray: Optional[numpy.ndarray] = None
		self._frame_index: int = 0
		self._lock = Lock()
		self._detection_interval = 6
		self._max_missed = 2
		self._min_points = 10
		self._match_iou = 0.3
		self._config_signature: Optional[tuple[int, int, int, float]] = None
		self._filters: Dict[int, List[_OneEuroFilter]] = {}
		self._face_to_track: Dict[int, int] = {}
		self._track_to_face_id: Dict[int, int] = {}
		fps = state_manager.get_item('output_video_fps')
		try:
			self._fps = float(fps) if fps else 30.0
		except Exception:
			self._fps = 30.0
		self._confidence_window_size = 10
		self._confidence_drop_threshold = 0.15

	def reset(self) -> None:
		with self._lock:
			self._tracks.clear()
			self._next_track_id = 0
			self._prev_gray = None
			self._frame_index = 0
			self._filters.clear()
			self._face_to_track.clear()
			self._track_to_face_id.clear()
			reset_affine_smoothers()
			if reset_all_temporal_states:
				reset_all_temporal_states()

	def process_frame(self, vision_frame: VisionFrame, detect_fn: Callable[[VisionFrame], List[Face]]) -> List[Face]:
		with self._lock:
			return self._process_frame_locked(vision_frame, detect_fn)

	# Internal helpers -----------------------------------------------------
	def _process_frame_locked(self, vision_frame: VisionFrame, detect_fn: Callable[[VisionFrame], List[Face]]) -> List[Face]:
		self._refresh_config()
		gray = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY)
		self._frame_index += 1

		need_detection = (self._prev_gray is None or not self._tracks or
		                  (self._detection_interval > 0 and (self._frame_index % self._detection_interval == 0)))

		if need_detection:
			faces = detect_fn(vision_frame)
			self._prev_gray = gray
			return self._update_tracks_from_detection(faces, gray)

		if self._prev_gray is None:
			self._prev_gray = gray
			return detect_fn(vision_frame)

		flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		self._prev_gray = gray
		faces = self._propagate_tracks(flow)
		if not faces:
			faces = detect_fn(vision_frame)
			return self._update_tracks_from_detection(faces, gray)
		return faces

	def _refresh_config(self) -> None:
		cfg = (
			int(state_manager.get_item('face_tracker_interval') or 6),
			int(state_manager.get_item('face_tracker_max_missed') or 2),
			int(state_manager.get_item('face_tracker_min_points') or 10),
			float(state_manager.get_item('face_tracker_match_iou') or 0.3)
		)
		if cfg != self._config_signature:
			self._config_signature = cfg
			self._detection_interval, self._max_missed, self._min_points, self._match_iou = cfg
			self._filters.clear()
			self._face_to_track.clear()
			self._track_to_face_id.clear()

	def _update_tracks_from_detection(self, faces: List[Face], gray: numpy.ndarray) -> List[Face]:
		self._frame_index = self._frame_index or 1
		self._prev_gray = gray
		updated_faces: List[Face] = []
		for idx, face in enumerate(faces):
			points_68 = face.landmark_set.get('68/5')
			if points_68 is None or len(points_68) < self._min_points:
				continue
			track_id = self._face_to_track.get(idx)
			if track_id is None:
				track_id = self._next_track_id
				self._next_track_id += 1
			self._tracks[track_id] = _TrackedFace(track_id, face, points_68, last_detection_frame=self._frame_index)
			self._track_to_face_id[track_id] = idx
			self._face_to_track[idx] = track_id
			self._init_filters_for_track(track_id, points_68)
			updated_faces.append(face)
		return updated_faces

	def _init_filters_for_track(self, track_id: int, points_68: numpy.ndarray) -> None:
		filters = [_OneEuroFilter(self._fps) for _ in range(points_68.shape[0] * 2)]
		for i, coord in enumerate(points_68.flatten()):
			filters[i].reset(float(coord))
		self._filters[track_id] = filters

	def _propagate_tracks(self, flow: numpy.ndarray) -> List[Face]:
		if not self._tracks:
			return []
		updated: List[Face] = []
		face_to_track_new: Dict[int, int] = {}
		for track_id, tracked in list(self._tracks.items()):
			points = tracked.points_68.astype(numpy.float32)
			offsets = cv2.remap(flow[..., 0], points[:, 0], points[:, 1], cv2.INTER_LINEAR), \
			          cv2.remap(flow[..., 1], points[:, 0], points[:, 1], cv2.INTER_LINEAR)
			dx = offsets[0].reshape(-1)
			dy = offsets[1].reshape(-1)
			new_points = points.copy()
			new_points[:, 0] += dx
			new_points[:, 1] += dy
			new_points = self._smooth_points(track_id, new_points)
			tracked.points_68 = new_points

			bbox = tracked.face.bounding_box.copy()
			bbox[0::2] += numpy.mean(dx)
			bbox[1::2] += numpy.mean(dy)
			landmarks_5 = convert_to_face_landmark_5(new_points)
			angle = estimate_face_angle(new_points)
			face = tracked.face._replace(
				bounding_box=bbox,
				landmark_set={**tracked.face.landmark_set, '68/5': new_points, '5/68': landmarks_5},
				angle=angle
			)
			tracked.face = face
			tracked.misses = 0
			updated.append(face)
			face_to_track_new[len(updated) - 1] = track_id

		for track_id in list(self._tracks.keys()):
			if track_id not in face_to_track_new.values():
				self._tracks[track_id].misses += 1
				if self._tracks[track_id].misses > self._max_missed:
					del self._tracks[track_id]

		self._face_to_track = face_to_track_new
		return updated

	def _smooth_points(self, track_id: int, points: numpy.ndarray) -> numpy.ndarray:
		filters = self._filters.get(track_id)
		if not filters:
			return points
		flat = points.flatten()
		smoothed = []
		for idx, value in enumerate(flat):
			if idx >= len(filters):
				smoothed.append(value)
				continue
			smoothed.append(filters[idx].filter(float(value)))
		return numpy.array(smoothed, dtype=numpy.float32).reshape(points.shape)


_GLOBAL_TRACKER: Optional[FaceTracker] = None


def get_tracker() -> FaceTracker:
	global _GLOBAL_TRACKER
	if _GLOBAL_TRACKER is None:
		_GLOBAL_TRACKER = FaceTracker()
	return _GLOBAL_TRACKER

