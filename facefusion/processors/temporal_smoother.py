"""
Temporal Smoother Module for FaceFusion 3.5.0
Eliminates jitter by smoothing face transformations across frames.

Usage:
    from facefusion.processors.temporal_smoother import TemporalSmoother

    smoother = TemporalSmoother(window_size=5, alpha=0.25)
    smoothed_landmarks = smoother.smooth_landmarks(current_landmarks)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import cv2


class TemporalSmoother:
    """
    Smooth face transformations across frames to reduce jitter.

    This addresses the negative Δ tPSNR spikes (down to -2.31 dB) in your current
    implementation by maintaining temporal consistency in landmark positions and
    transformation matrices.
    """

    def __init__(
        self,
        window_size: int = 5,
        alpha: float = 0.25,
        enable_adaptive: bool = True
    ):
        """
        Initialize temporal smoother.

        Args:
            window_size: Number of frames to consider for smoothing history
                        5 frames ≈ 0.2s at 25fps - good balance
            alpha: Smoothing factor for exponential moving average
                   0.25 = 75% history, 25% current frame
                   Lower = more smoothing, higher = more responsive
            enable_adaptive: Adapt smoothing based on motion magnitude
        """
        self.window_size = window_size
        self.alpha = alpha
        self.enable_adaptive = enable_adaptive

        # History buffers
        self.landmark_history = deque(maxlen=window_size)
        self.transform_history = deque(maxlen=window_size)
        self.bbox_history = deque(maxlen=window_size)

        # Motion tracking
        self.motion_magnitude_history = deque(maxlen=window_size)

    def smooth_landmarks(self, current_landmarks: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to face landmarks.

        Args:
            current_landmarks: (N, 2) array of facial landmark coordinates

        Returns:
            Smoothed landmark coordinates
        """
        self.landmark_history.append(current_landmarks.copy())

        # First frame - no smoothing possible
        if len(self.landmark_history) == 1:
            return current_landmarks

        # Calculate motion magnitude for adaptive smoothing
        if self.enable_adaptive and len(self.landmark_history) >= 2:
            motion = np.mean(np.abs(
                current_landmarks - self.landmark_history[-2]
            ))
            self.motion_magnitude_history.append(motion)

            # Adapt alpha based on motion
            # High motion = higher alpha (less smoothing to stay responsive)
            # Low motion = lower alpha (more smoothing to reduce jitter)
            avg_motion = np.mean(self.motion_magnitude_history)
            adaptive_alpha = np.clip(
                self.alpha * (1 + motion / (avg_motion + 1e-6)),
                0.1,
                0.5
            )
        else:
            adaptive_alpha = self.alpha

        # Exponential moving average with adaptive alpha
        smoothed = current_landmarks.copy()

        # Weight more recent frames higher
        history = list(self.landmark_history)[:-1]  # Exclude current
        for i, prev_landmarks in enumerate(reversed(history)):
            weight = (1 - adaptive_alpha) * (0.8 ** i)  # Decay for older frames
            smoothed = adaptive_alpha * smoothed + weight * prev_landmarks

        # Normalize weights
        total_weight = adaptive_alpha + sum(
            (1 - adaptive_alpha) * (0.8 ** i)
            for i in range(len(history))
        )
        smoothed = smoothed / total_weight

        return smoothed

    def smooth_landmark_individual(
        self,
        current_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Apply per-landmark smoothing for maximum jitter reduction.

        More aggressive than smooth_landmarks() - use when jitter is severe.
        """
        self.landmark_history.append(current_landmarks.copy())

        if len(self.landmark_history) == 1:
            return current_landmarks

        smoothed = np.zeros_like(current_landmarks)

        # Smooth each landmark independently
        for landmark_idx in range(current_landmarks.shape[0]):
            # Get history for this specific landmark
            landmark_sequence = [
                lm[landmark_idx] for lm in self.landmark_history
            ]

            # Use median filter for outlier rejection + mean for smoothing
            if len(landmark_sequence) >= 3:
                # Remove outliers using median
                landmark_array = np.array(landmark_sequence)
                median = np.median(landmark_array, axis=0)
                distances = np.linalg.norm(landmark_array - median, axis=1)

                # Keep landmarks within 1.5 * median distance
                threshold = 1.5 * np.median(distances)
                valid_landmarks = landmark_array[distances <= threshold]

                # Average valid landmarks
                if len(valid_landmarks) > 0:
                    smoothed[landmark_idx] = np.mean(valid_landmarks, axis=0)
                else:
                    smoothed[landmark_idx] = current_landmarks[landmark_idx]
            else:
                smoothed[landmark_idx] = np.mean(landmark_sequence, axis=0)

        return smoothed

    def smooth_transform_matrix(
        self,
        current_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply temporal smoothing to affine transformation matrices.

        Args:
            current_matrix: (2, 3) or (3, 3) affine transformation matrix

        Returns:
            Smoothed transformation matrix
        """
        self.transform_history.append(current_matrix.copy())

        if len(self.transform_history) == 1:
            return current_matrix

        # Average transformation matrices with exponential decay
        smoothed = np.zeros_like(current_matrix)
        total_weight = 0.0

        for i, prev_matrix in enumerate(reversed(list(self.transform_history))):
            weight = self.alpha * (0.7 ** i)
            smoothed += weight * prev_matrix
            total_weight += weight

        smoothed /= total_weight

        return smoothed

    def smooth_bbox(
        self,
        current_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Smooth bounding box coordinates to reduce jitter.

        Args:
            current_bbox: (x1, y1, x2, y2) bounding box

        Returns:
            Smoothed bounding box
        """
        bbox_array = np.array(current_bbox)
        self.bbox_history.append(bbox_array)

        if len(self.bbox_history) == 1:
            return current_bbox

        # Use median filter for bbox (more robust to outliers)
        smoothed_bbox = np.median(list(self.bbox_history), axis=0)

        # Ensure integer coordinates
        smoothed_bbox = np.round(smoothed_bbox).astype(int)

        return tuple(smoothed_bbox)

    def reset(self):
        """Reset all history buffers (call at scene cuts)."""
        self.landmark_history.clear()
        self.transform_history.clear()
        self.bbox_history.clear()
        self.motion_magnitude_history.clear()

    def should_reset(self, current_landmarks: np.ndarray) -> bool:
        """
        Detect if there's a scene cut and reset should be called.

        Args:
            current_landmarks: Current frame landmarks

        Returns:
            True if reset is recommended
        """
        if len(self.landmark_history) < 2:
            return False

        # Calculate large motion that might indicate scene cut
        prev_landmarks = self.landmark_history[-1]
        motion = np.mean(np.abs(current_landmarks - prev_landmarks))

        # If motion is > 3x average motion, likely a scene cut
        if len(self.motion_magnitude_history) >= 3:
            avg_motion = np.mean(self.motion_magnitude_history)
            if motion > 3.0 * avg_motion:
                return True

        return False
