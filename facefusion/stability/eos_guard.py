"""
End-of-Sequence (EOS) Stability Guard.
Prevents jitter spikes in the last frames of video processing by freezing
slow-varying parameters and applying quantization with hysteresis to affine matrices.

This addresses the common issue where tracker confidence drops near the end of clips,
causing noticeable quality degradation in tail frames.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any


class EOSStabilityGuard:
    """
    Stability guard for end-of-sequence frames.

    Features:
    1. Parameter freezing: Lock slow-varying parameters (color gains, gamma, etc.)
    2. Affine quantization: Round affine matrix components to reduce micro-jitter
    3. Hysteresis: Only accept changes that cross quantum boundaries

    These techniques are particularly effective for the last N frames where
    tracker reliability decreases.
    """

    def __init__(
        self,
        tail_window_frames: int = 12,
        quant_rotation: float = 1/2048,
        quant_translation: float = 1/16
    ):
        """
        Initialize EOS stability guard.

        Args:
            tail_window_frames: Number of final frames to protect
            quant_rotation: Quantization step for rotation/scale components (radians)
            quant_translation: Quantization step for translation (pixels)
        """
        self.tail_window_frames = tail_window_frames
        self.quant_rotation = quant_rotation
        self.quant_translation = quant_translation

        # State tracking
        self.frozen_params: Optional[Dict[str, Any]] = None
        self.prev_affine_quantized: Optional[np.ndarray] = None
        self.is_frozen = False

    def in_tail_window(self, frame_idx: int, total_frames: int) -> bool:
        """
        Check if current frame is in the tail window.

        Args:
            frame_idx: Current frame index (0-based)
            total_frames: Total number of frames

        Returns:
            True if in tail window
        """
        return frame_idx >= (total_frames - self.tail_window_frames)

    def freeze_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Freeze slow-varying parameters at their current values.

        Args:
            params: Dictionary of parameters to potentially freeze
                   (e.g., {'color_gains': ..., 'gamma': ..., 'mask_feather': ...})

        Returns:
            Frozen parameters (first call freezes, subsequent calls return frozen values)
        """
        if not self.is_frozen:
            # First time in tail window - freeze current values
            self.frozen_params = params.copy()
            self.is_frozen = True

        return self.frozen_params.copy()

    def quantize_affine(self, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Quantize affine matrix components to reduce micro-jitter.

        The affine matrix has the form:
        [[a, b, tx],
         [c, d, ty]]

        Where (a,b,c,d) encode rotation/scale and (tx,ty) encode translation.

        Args:
            affine_matrix: 2x3 affine transformation matrix

        Returns:
            Quantized affine matrix
        """
        affine_q = affine_matrix.copy()

        # Quantize rotation/scale components (a, b, c, d)
        affine_q[0, 0] = np.round(affine_q[0, 0] / self.quant_rotation) * self.quant_rotation
        affine_q[0, 1] = np.round(affine_q[0, 1] / self.quant_rotation) * self.quant_rotation
        affine_q[1, 0] = np.round(affine_q[1, 0] / self.quant_rotation) * self.quant_rotation
        affine_q[1, 1] = np.round(affine_q[1, 1] / self.quant_rotation) * self.quant_rotation

        # Quantize translation components (tx, ty)
        affine_q[0, 2] = np.round(affine_q[0, 2] / self.quant_translation) * self.quant_translation
        affine_q[1, 2] = np.round(affine_q[1, 2] / self.quant_translation) * self.quant_translation

        return affine_q.astype(np.float32)

    def stabilize_affine(self, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Apply quantization and hysteresis to affine matrix.

        Hysteresis prevents rapid oscillation by only accepting changes that
        move to a different quantum cell.

        Args:
            affine_matrix: 2x3 affine transformation matrix

        Returns:
            Stabilized affine matrix
        """
        # Quantize the new matrix
        affine_quantized = self.quantize_affine(affine_matrix)

        # Initialize on first call
        if self.prev_affine_quantized is None:
            self.prev_affine_quantized = affine_quantized
            return affine_quantized

        # Hysteresis: only update if quantized values changed
        if np.allclose(affine_quantized, self.prev_affine_quantized, rtol=0, atol=1e-8):
            # No quantum boundary crossed - keep previous
            return self.prev_affine_quantized

        # Quantum boundary crossed - accept new value
        self.prev_affine_quantized = affine_quantized
        return affine_quantized

    def reset(self):
        """Reset guard state (call at start of new video/sequence)."""
        self.frozen_params = None
        self.prev_affine_quantized = None
        self.is_frozen = False


class MultiTrackEOSGuard:
    """
    Manages EOS guards for multiple face tracks.
    Each track gets its own independent guard instance.
    """

    def __init__(
        self,
        tail_window_frames: int = 12,
        quant_rotation: float = 1/2048,
        quant_translation: float = 1/16
    ):
        """
        Initialize multi-track EOS guard manager.

        Args:
            tail_window_frames: Number of final frames to protect
            quant_rotation: Quantization step for rotation/scale
            quant_translation: Quantization step for translation
        """
        self.tail_window_frames = tail_window_frames
        self.quant_rotation = quant_rotation
        self.quant_translation = quant_translation
        self.guards: Dict[int, EOSStabilityGuard] = {}

    def get_guard(self, track_id: int) -> EOSStabilityGuard:
        """
        Get or create guard for a specific track.

        Args:
            track_id: Unique track identifier

        Returns:
            EOS guard instance for this track
        """
        if track_id not in self.guards:
            self.guards[track_id] = EOSStabilityGuard(
                self.tail_window_frames,
                self.quant_rotation,
                self.quant_translation
            )
        return self.guards[track_id]

    def reset(self, track_id: Optional[int] = None):
        """
        Reset guard state for a track or all tracks.

        Args:
            track_id: Track to reset, or None to reset all
        """
        if track_id is None:
            self.guards.clear()
        elif track_id in self.guards:
            self.guards[track_id].reset()


def apply_eos_protection(
    frame_idx: int,
    total_frames: int,
    affine_matrix: np.ndarray,
    params: Dict[str, Any],
    guard: EOSStabilityGuard
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply EOS protection to current frame if in tail window.

    Args:
        frame_idx: Current frame index
        total_frames: Total frames in video
        affine_matrix: Current affine transformation matrix
        params: Current processing parameters
        guard: EOS guard instance

    Returns:
        Tuple of (protected_affine_matrix, protected_params)
    """
    if guard.in_tail_window(frame_idx, total_frames):
        # In tail window - apply protection
        protected_affine = guard.stabilize_affine(affine_matrix)
        protected_params = guard.freeze_parameters(params)
        return protected_affine, protected_params
    else:
        # Not in tail window - pass through
        return affine_matrix, params


def lock_affine_scale_and_rotation(
    affine_matrix: np.ndarray,
    locked_affine: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Lock scale and rotation components of affine matrix, allowing only translation.

    This is an additional protection for extreme tail frames where even
    small geometric changes cause artifacts.

    Args:
        affine_matrix: Current affine matrix (2x3)
        locked_affine: Reference affine to lock to (if None, uses identity scale/rotation)

    Returns:
        Affine matrix with locked scale/rotation
    """
    result = affine_matrix.copy()

    if locked_affine is not None:
        # Lock to reference affine's scale/rotation
        result[0, 0] = locked_affine[0, 0]  # a
        result[0, 1] = locked_affine[0, 1]  # b
        result[1, 0] = locked_affine[1, 0]  # c
        result[1, 1] = locked_affine[1, 1]  # d
    else:
        # Lock to identity (scale=1, rotation=0)
        result[0, 0] = 1.0
        result[0, 1] = 0.0
        result[1, 0] = 0.0
        result[1, 1] = 1.0

    # Keep translation (tx, ty) from original
    # result[0, 2] and result[1, 2] unchanged

    return result
