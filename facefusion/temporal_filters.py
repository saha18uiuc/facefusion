"""Similarity-transform smoothing with bounded derivatives."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy

WINDOW = 9
ORDER = 2
_MAX_ROT_STEP = math.radians(3.0)
_MAX_LOGS_STEP = 0.02
_MAX_TRANS_STEP = 6.0


@dataclass
class Se2Components:
    scale: float
    theta: float
    tx: float
    ty: float


class _SimilarityHistory:
    """Maintain a causal history of similarity parameters and smooth them."""

    def __init__(self) -> None:
        self._scale: Deque[float] = deque(maxlen=WINDOW)
        self._theta: Deque[float] = deque(maxlen=WINDOW)
        self._tx: Deque[float] = deque(maxlen=WINDOW)
        self._ty: Deque[float] = deque(maxlen=WINDOW)
        self._theta_ref: Optional[float] = None
        self._last: Optional[Se2Components] = None

    def push(self, comp: Se2Components) -> Se2Components:
        """Insert a new observation and return the smoothed components."""
        theta = float(comp.theta)
        if self._theta_ref is None:
            self._theta_ref = theta
        else:
            theta = _unwrap(theta, self._theta_ref)
            self._theta_ref = theta

        self._scale.append(max(comp.scale, 1e-6))
        self._theta.append(theta)
        self._tx.append(comp.tx)
        self._ty.append(comp.ty)

        smoothed = Se2Components(
            scale=self._smooth_channel(self._scale, _MAX_LOGS_STEP, is_log=True),
            theta=self._smooth_channel(self._theta, _MAX_ROT_STEP, is_angle=True),
            tx=self._smooth_channel(self._tx, _MAX_TRANS_STEP),
            ty=self._smooth_channel(self._ty, _MAX_TRANS_STEP),
        )
        self._last = smoothed
        return smoothed

    def _smooth_channel(self, channel: Deque[float], max_step: float, is_log: bool = False, is_angle: bool = False) -> float:
        arr = numpy.array(channel, dtype=numpy.float64)
        if is_log:
            arr = numpy.log(numpy.clip(arr, 1e-6, None))
        if arr.size == 0:
            return 0.0
        target = arr[-1]
        if arr.size < 3:
            return _bounded_step(self._last_value(channel, is_log, is_angle), target, max_step, is_log)

        x = numpy.arange(arr.size, dtype=numpy.float64)
        try:
            coeffs = numpy.polyfit(x, arr, ORDER)
            smoothed = numpy.polyval(coeffs, x[-1])
        except numpy.linalg.LinAlgError:
            smoothed = float(target)

        return _bounded_step(self._last_value(channel, is_log, is_angle), float(smoothed), max_step, is_log)

    def _last_value(self, channel: Deque[float], is_log: bool, is_angle: bool) -> float:
        if self._last is None:
            if is_log:
                return math.log(max(channel[-1], 1e-6))
            return channel[-1]
        if is_log:
            return math.log(max(self._last.scale, 1e-6))
        if is_angle:
            return self._last.theta
        if channel is self._tx:
            return self._last.tx
        if channel is self._ty:
            return self._last.ty
        return self._last.scale


def _bounded_step(previous: float, current: float, max_step: float, is_log: bool) -> float:
    delta = current - previous
    if delta > max_step:
        current = previous + max_step
    elif delta < -max_step:
        current = previous - max_step
    if is_log:
        return float(math.exp(current))
    return float(current)


def _unwrap(theta: float, reference: float) -> float:
    delta = theta - reference
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return reference + delta


def affine_to_similarity(matrix: numpy.ndarray) -> Se2Components:
    if matrix.shape != (2, 3):
        raise ValueError('Expected 2x3 affine matrix.')
    a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    d, e, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])

    sx = math.sqrt(max(a * a + d * d, 1e-12))
    sy = math.sqrt(max(b * b + e * e, 1e-12))
    scale = 0.5 * (sx + sy)
    if (a * e - b * d) < 0.0:
        scale = -scale
    theta = math.atan2(d, a)
    return Se2Components(scale=scale, theta=theta, tx=tx, ty=ty)


def similarity_to_affine(components: Se2Components) -> numpy.ndarray:
    s = float(components.scale)
    theta = float(components.theta)
    tx = float(components.tx)
    ty = float(components.ty)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return numpy.array(
        [
            [s * cos_t, -s * sin_t, tx],
            [s * sin_t,  s * cos_t, ty],
        ],
        dtype=numpy.float32,
    )


class _FilterStore:
    def __init__(self) -> None:
        self._filters: Dict[str, _SimilarityHistory] = {}

    def get(self, key: str) -> _SimilarityHistory:
        filt = self._filters.get(key)
        if filt is None:
            filt = _SimilarityHistory()
            self._filters[key] = filt
        return filt

    def reset(self) -> None:
        self._filters.clear()


_STORE = _FilterStore()


def resolve_filter_key(track_token: Optional[str]) -> str:
    return track_token if track_token else '__default__'


def reset_all_filters() -> None:
    _STORE.reset()


def filter_affine(matrix: numpy.ndarray, key: str) -> numpy.ndarray:
    components = affine_to_similarity(matrix)
    smooth = _STORE.get(key).push(components)
    return similarity_to_affine(smooth)
