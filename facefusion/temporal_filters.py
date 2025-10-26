"""Low-latency temporal filters for face alignment smoothing."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from time import time
from typing import Optional, Tuple

import numpy


class OneEuroFilter:
    """One Euro filter implementation for real-time smoothing.

    The filter adapts its cutoff frequency based on the magnitude of the
    derivative so it can remain responsive to rapid motion while suppressing
    jitter during slow motion. See: Casiez et al., "The 1€ Filter", UIST 2012.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.03, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._prev_t: Optional[float] = None
        self._prev_x: Optional[float] = None
        self._prev_dx: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        now = time() if t is None else float(t)
        if self._prev_t is None:
            self._prev_t = now
            self._prev_x = float(x)
            self._prev_dx = 0.0
            return float(x)

        dt = max(1e-3, now - self._prev_t)
        dx = (float(x) - float(self._prev_x)) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._prev_dx

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * float(self._prev_x)

        self._prev_t = now
        self._prev_x = x_hat
        self._prev_dx = dx_hat
        return x_hat


class Kalman1D:
    """Minimal 1D constant-velocity Kalman filter."""

    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 5e-3) -> None:
        self.q = float(process_variance)
        self.r = float(measurement_variance)
        self.x: Optional[float] = None
        self.p: float = 1.0

    def reset(self) -> None:
        self.x = None
        self.p = 1.0

    def __call__(self, measurement: float) -> float:
        if self.x is None:
            self.x = float(measurement)
            self.p = 1.0
            return self.x
        # Predict
        self.p += self.q
        # Update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (float(measurement) - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


@dataclass
class Se2Components:
    scale: float
    theta: float
    tx: float
    ty: float


class Se2Filter:
    """Composite filter for similarity transform (scale, rotation, translation)."""

    def __init__(self) -> None:
        self._scale_filter = OneEuroFilter(min_cutoff=1.2, beta=0.04)
        self._theta_filter = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        self._tx_filter = OneEuroFilter(min_cutoff=1.0, beta=0.06)
        self._ty_filter = OneEuroFilter(min_cutoff=1.0, beta=0.06)
        self._scale_kalman = Kalman1D(process_variance=2e-4, measurement_variance=2e-3)
        self._theta_kalman = Kalman1D(process_variance=5e-4, measurement_variance=4e-3)
        self._tx_kalman = Kalman1D(process_variance=8e-4, measurement_variance=6e-3)
        self._ty_kalman = Kalman1D(process_variance=8e-4, measurement_variance=6e-3)
        self._theta_ref: Optional[float] = None
        self._lock = threading.Lock()

    @staticmethod
    def _unwrap(theta: float, reference: float) -> float:
        """Unwrap angles to avoid jumps around +/-pi."""
        delta = theta - reference
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        while delta > math.pi:
            delta -= 2.0 * math.pi
        return reference + delta

    def reset(self) -> None:
        with self._lock:
            self._scale_filter = OneEuroFilter(min_cutoff=1.2, beta=0.04)
            self._theta_filter = OneEuroFilter(min_cutoff=1.0, beta=0.05)
            self._tx_filter = OneEuroFilter(min_cutoff=1.0, beta=0.06)
            self._ty_filter = OneEuroFilter(min_cutoff=1.0, beta=0.06)
            self._scale_kalman.reset()
            self._theta_kalman.reset()
            self._tx_kalman.reset()
            self._ty_kalman.reset()
            self._theta_ref = None

    def __call__(self, components: Se2Components, timestamp: Optional[float] = None) -> Se2Components:
        with self._lock:
            theta = float(components.theta)
            if self._theta_ref is None:
                self._theta_ref = theta
            else:
                theta = self._unwrap(theta, self._theta_ref)

            s_hat = self._scale_filter(components.scale, timestamp)
            th_hat = self._theta_filter(theta, timestamp)
            tx_hat = self._tx_filter(components.tx, timestamp)
            ty_hat = self._ty_filter(components.ty, timestamp)

            s_hat = self._scale_kalman(s_hat)
            th_hat = self._theta_kalman(th_hat)
            tx_hat = self._tx_kalman(tx_hat)
            ty_hat = self._ty_kalman(ty_hat)

            self._theta_ref = th_hat

            return Se2Components(s_hat, th_hat, tx_hat, ty_hat)


def affine_to_se2(matrix: numpy.ndarray) -> Se2Components:
    """Convert a 2x3 affine similarity matrix into scale/rotation/translation."""
    if matrix.shape != (2, 3):
        raise ValueError('Expected a 2x3 affine matrix.')
    a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    d, e, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])

    scale_x = math.sqrt(max(a * a + d * d, 1e-12))
    scale_y = math.sqrt(max(b * b + e * e, 1e-12))
    scale = 0.5 * (scale_x + scale_y)
    det = a * e - b * d
    if det < 0.0:
        scale = -scale

    theta = math.atan2(d, a)
    return Se2Components(scale, theta, tx, ty)


def se2_to_affine(components: Se2Components) -> numpy.ndarray:
    """Convert scale/rotation/translation back into a 2x3 affine matrix."""
    s = float(components.scale)
    theta = float(components.theta)
    tx = float(components.tx)
    ty = float(components.ty)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    matrix = numpy.array(
        [
            [s * cos_t, -s * sin_t, tx],
            [s * sin_t,  s * cos_t, ty],
        ],
        dtype=numpy.float32,
    )
    return matrix


class TransformFilterStore:
    """Thread-safe registry of SE(2) filters keyed by track tokens."""

    def __init__(self) -> None:
        self._filters: dict[str, Se2Filter] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Se2Filter:
        with self._lock:
            filt = self._filters.get(key)
            if filt is None:
                filt = Se2Filter()
                self._filters[key] = filt
            return filt

    def reset(self) -> None:
        with self._lock:
            for filt in self._filters.values():
                filt.reset()
            self._filters.clear()


_TRANSFORM_STORE = TransformFilterStore()


def reset_all_filters() -> None:
    _TRANSFORM_STORE.reset()


def filter_affine(matrix: numpy.ndarray, key: str, timestamp: Optional[float] = None) -> numpy.ndarray:
    """Return a smoothed affine matrix for ``key``."""
    components = affine_to_se2(matrix)
    filt = _TRANSFORM_STORE.get(key)
    filtered = filt(components, timestamp)
    return se2_to_affine(filtered)


def resolve_filter_key(track_token: Optional[str]) -> str:
    return track_token if track_token is not None else '__default__'
