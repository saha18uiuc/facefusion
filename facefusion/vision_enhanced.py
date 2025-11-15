"""
Enhanced vision operations for FaceFusion 3.5.0 with CUDA optimizations.
Provides drop-in replacements for critical operations with improved performance and stability.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from facefusion.types import VisionFrame, Mask, Matrix

# Try to import CUDA operations
CUDA_AVAILABLE = False
try:
    from facefusion.cuda import ff_cuda_ops
    CUDA_AVAILABLE = True
except ImportError:
    pass

# Try to import stability modules
STABILITY_AVAILABLE = False
try:
    from facefusion.stability.fhr_refine import TinyFHR, refine_landmarks_to_subpixel, TORCH_AVAILABLE as FHR_AVAILABLE
    from facefusion.stability.pose_filter import PoseFilter, extract_pose_from_landmarks
    from facefusion.stability.eos_guard import EOSStabilityGuard
    STABILITY_AVAILABLE = True
except ImportError:
    FHR_AVAILABLE = False


def warp_face_enhanced(
    vision_frame: VisionFrame,
    affine_matrix: Matrix,
    output_size: Tuple[int, int],
    use_cuda: bool = True
) -> VisionFrame:
    """
    Enhanced face warping with optional CUDA acceleration and deterministic rounding.

    Args:
        vision_frame: Input frame
        affine_matrix: 2x3 affine transformation matrix
        output_size: Output size (width, height)
        use_cuda: Whether to use CUDA kernels (if available)

    Returns:
        Warped vision frame
    """
    if use_cuda and CUDA_AVAILABLE and len(vision_frame.shape) == 3 and vision_frame.shape[2] == 3:
        try:
            # Prepare output
            output = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

            # Flatten affine matrix to [a, b, tx, c, d, ty]
            affine_flat = np.array([
                affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[0, 2],
                affine_matrix[1, 0], affine_matrix[1, 1], affine_matrix[1, 2]
            ], dtype=np.float32)

            # Call CUDA kernel
            ff_cuda_ops.warp_bilinear_rgb(vision_frame, output, affine_flat, 0, 0)

            return output
        except Exception as e:
            # Fallback to OpenCV if CUDA fails
            print(f"CUDA warp failed, falling back to OpenCV: {e}")

    # Fallback: OpenCV with deterministic interpolation
    return cv2.warpAffine(
        vision_frame,
        affine_matrix,
        output_size,
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_LINEAR
    )


def paste_back_enhanced(
    temp_vision_frame: VisionFrame,
    crop_vision_frame: VisionFrame,
    crop_mask: Mask,
    affine_matrix: Matrix,
    use_cuda: bool = True,
    seam_band: Tuple[int, int] = (30, 225)
) -> VisionFrame:
    """
    Enhanced paste-back with CUDA-accelerated compositing and seam-band feathering.

    Args:
        temp_vision_frame: Base frame
        crop_vision_frame: Crop to paste
        crop_mask: Crop mask (0-1 float or 0-255 uint8)
        affine_matrix: Affine transformation matrix
        use_cuda: Whether to use CUDA kernels
        seam_band: Alpha value range for seam-band feathering (lo, hi)

    Returns:
        Composited vision frame
    """
    # Calculate paste area (same as original)
    from facefusion.face_helper import calculate_paste_area

    paste_bounding_box, paste_matrix = calculate_paste_area(
        temp_vision_frame, crop_vision_frame, affine_matrix
    )
    x1, y1, x2, y2 = paste_bounding_box
    paste_width = x2 - x1
    paste_height = y2 - y1

    temp_vision_frame = temp_vision_frame.copy()
    paste_region = temp_vision_frame[y1:y2, x1:x2]

    if use_cuda and CUDA_AVAILABLE and paste_width > 0 and paste_height > 0 and paste_region.shape[-1] == 3:
        try:
            mask_2d, mask_copied = _ensure_contiguous(
                crop_mask if crop_mask.ndim == 2 else crop_mask[..., 0],
                'mask'
            )
            if mask_2d.dtype != np.float32:
                mask_buf = _get_cached_buffer(mask_2d.shape, np.float32, 'mask-f32')
                mask_buf[...] = mask_2d.astype(np.float32)
                mask_2d = mask_buf
            crop_frame, crop_copied = _ensure_contiguous(crop_vision_frame, 'crop')
            if crop_frame.dtype != np.uint8:
                crop_buf = _get_cached_buffer(crop_frame.shape, np.uint8, 'crop-u8')
                crop_buf[...] = crop_frame.astype(np.uint8)
                crop_frame = crop_buf
            bg_roi, bg_copied = _ensure_contiguous(paste_region, 'bg')
            out_roi = bg_roi if not bg_copied else _get_cached_buffer(bg_roi.shape, bg_roi.dtype, 'bg-out')
            paste_flat = np.array([
                paste_matrix[0, 0], paste_matrix[0, 1], paste_matrix[0, 2],
                paste_matrix[1, 0], paste_matrix[1, 1], paste_matrix[1, 2]
            ], dtype=np.float32)

            ff_cuda_ops.warp_composite_rgb(
                crop_frame, mask_2d, bg_roi, out_roi, paste_flat,
                seam_band[0], seam_band[1]
            )

            if bg_copied:
                np.copyto(paste_region, out_roi)
            return temp_vision_frame
        except Exception as e:
            print(f"CUDA fused paste failed, falling back to OpenCV: {e}")

    # Warp mask and frame on CPU fallback
    inverse_mask = cv2.warpAffine(crop_mask, paste_matrix, (paste_width, paste_height))
    inverse_mask = inverse_mask.clip(0, 1)

    inverse_frame = cv2.warpAffine(
        crop_vision_frame, paste_matrix, (paste_width, paste_height),
        borderMode=cv2.BORDER_REPLICATE
    )

    # Fallback: OpenCV blending
    inverse_mask_3d = np.expand_dims(inverse_mask, axis=-1)
    paste_result = paste_region * (1 - inverse_mask_3d) + inverse_frame * inverse_mask_3d
    temp_vision_frame[y1:y2, x1:x2] = paste_result.astype(temp_vision_frame.dtype)

    return temp_vision_frame


def apply_color_correction_enhanced(
    vision_frame: VisionFrame,
    gain_r: float,
    gain_g: float,
    gain_b: float,
    use_cuda: bool = True,
    linear_space: bool = True
) -> VisionFrame:
    """
    Apply color gain with optional linear-light processing and CUDA acceleration.

    Args:
        vision_frame: Input frame
        gain_r: Red channel gain
        gain_g: Green channel gain
        gain_b: Blue channel gain
        use_cuda: Whether to use CUDA
        linear_space: Whether to work in linear light (recommended)

    Returns:
        Color-corrected frame
    """
    if use_cuda and CUDA_AVAILABLE and linear_space and len(vision_frame.shape) == 3 and vision_frame.shape[2] == 3:
        try:
            output = np.zeros_like(vision_frame)
            ff_cuda_ops.apply_color_gain_linear(
                vision_frame, output, gain_r, gain_g, gain_b
            )
            return output
        except Exception as e:
            print(f"CUDA color gain failed, falling back to NumPy: {e}")

    # Fallback: NumPy implementation
    if linear_space:
        # Convert to float and linear
        frame_float = vision_frame.astype(np.float32) / 255.0

        # sRGB to linear
        mask_low = frame_float <= 0.04045
        frame_linear = np.where(
            mask_low,
            frame_float / 12.92,
            np.power((frame_float + 0.055) / 1.055, 2.4)
        )

        # Apply gains
        gains = np.array([gain_b, gain_g, gain_r], dtype=np.float32)  # BGR order
        frame_linear = frame_linear * gains

        # Linear to sRGB
        mask_low_out = frame_linear <= 0.0031308
        frame_srgb = np.where(
            mask_low_out,
            frame_linear * 12.92,
            1.055 * np.power(frame_linear, 1.0 / 2.4) - 0.055
        )

        return (frame_srgb * 255).clip(0, 255).astype(np.uint8)
    else:
        # Simple gain in sRGB space
        gains = np.array([gain_b, gain_g, gain_r], dtype=np.float32)
        result = vision_frame.astype(np.float32) * gains
        return result.clip(0, 255).astype(np.uint8)


def apply_mask_hysteresis(
    alpha_current: Mask,
    alpha_previous: Mask,
    threshold_in: int = 4,
    threshold_out: int = 6,
    use_cuda: bool = True
) -> Mask:
    """
    Apply hysteresis to mask to prevent single-frame flicker.

    Args:
        alpha_current: Current alpha mask
        alpha_previous: Previous alpha mask
        threshold_in: Threshold for inward movement
        threshold_out: Threshold for outward movement
        use_cuda: Whether to use CUDA

    Returns:
        Stabilized alpha mask
    """
    # Ensure uint8
    if alpha_current.dtype != np.uint8:
        alpha_current = (alpha_current * 255).astype(np.uint8)
    if alpha_previous.dtype != np.uint8:
        alpha_previous = (alpha_previous * 255).astype(np.uint8)

    if use_cuda and CUDA_AVAILABLE:
        try:
            output = np.zeros_like(alpha_current)
            ff_cuda_ops.mask_edge_hysteresis(
                alpha_current,
                alpha_previous,
                output,
                threshold_in,
                threshold_out
            )
            return output
        except Exception as e:
            print(f"CUDA mask hysteresis failed, falling back to NumPy: {e}")

    # Fallback: NumPy implementation
    delta = alpha_current.astype(np.int16) - alpha_previous.astype(np.int16)
    moved_in = delta < -threshold_in
    moved_out = delta > threshold_out
    stable = ~(moved_in | moved_out)

    output = np.where(stable, alpha_previous, alpha_current)
    return output.astype(np.uint8)


# Global state for stability features
_fhr_model = None
_pose_filters = {}
_eos_guards = {}
_previous_masks = {}
_paste_buffer_cache: Dict[Tuple[Tuple[int, ...], str, str], np.ndarray] = {}


def _get_cached_buffer(shape: Tuple[int, ...], dtype: np.dtype, tag: str) -> np.ndarray:
    """Reuse scratch buffers to avoid reallocating per frame."""
    key = (tuple(shape), np.dtype(dtype).str, tag)
    buf = _paste_buffer_cache.get(key)
    if buf is None or buf.shape != tuple(shape) or buf.dtype != np.dtype(dtype):
        buf = np.empty(shape, dtype=dtype)
        _paste_buffer_cache[key] = buf
    return buf


def _ensure_contiguous(array: np.ndarray, tag: str) -> Tuple[np.ndarray, bool]:
    """Return a C-contiguous view, copying into cached buffer only if needed."""
    if array.flags.c_contiguous:
        return array, False
    buf = _get_cached_buffer(array.shape, array.dtype, tag)
    np.copyto(buf, array)
    return buf, True


def get_fhr_model() -> Optional['TinyFHR']:
    """Get or create FHR model for landmark refinement."""
    global _fhr_model
    if _fhr_model is None and STABILITY_AVAILABLE and FHR_AVAILABLE:
        try:
            import torch
            _fhr_model = TinyFHR(num_landmarks=68).eval()
            if torch.cuda.is_available():
                _fhr_model = _fhr_model.cuda()
        except Exception as e:
            print(f"Failed to create FHR model: {e}")
    return _fhr_model


def get_pose_filter(face_id: int) -> Optional[PoseFilter]:
    """Get or create pose filter for a face."""
    if not STABILITY_AVAILABLE:
        return None
    if face_id not in _pose_filters:
        _pose_filters[face_id] = PoseFilter(beta=0.85, deadband=(0.5, 0.5, 0.5, 0.002))
    return _pose_filters[face_id]


def get_eos_guard(track_id: int) -> Optional[EOSStabilityGuard]:
    """Get or create EOS guard for a track."""
    if not STABILITY_AVAILABLE:
        return None
    if track_id not in _eos_guards:
        _eos_guards[track_id] = EOSStabilityGuard(tail_window_frames=12)
    return _eos_guards[track_id]


def reset_stability_state():
    """Reset all stability state (call at start of new video)."""
    global _fhr_model, _pose_filters, _eos_guards, _previous_masks, _paste_buffer_cache
    _fhr_model = None
    _pose_filters.clear()
    _eos_guards.clear()
    _previous_masks.clear()
    _paste_buffer_cache.clear()


# Configuration
def is_cuda_available() -> bool:
    """Check if CUDA operations are available."""
    return CUDA_AVAILABLE


def is_stability_available() -> bool:
    """Check if stability features are available."""
    return STABILITY_AVAILABLE


def get_optimization_status() -> dict:
    """Get status of all optimizations."""
    return {
        'cuda_available': CUDA_AVAILABLE,
        'stability_available': STABILITY_AVAILABLE,
        'fhr_available': FHR_AVAILABLE if STABILITY_AVAILABLE else False,
        'pose_filter_active': len(_pose_filters) > 0,
        'eos_guards_active': len(_eos_guards) > 0
    }
