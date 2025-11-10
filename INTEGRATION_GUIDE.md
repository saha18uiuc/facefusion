# FaceFusion 3.5.0 Optimization Integration Guide

## Quick Start: Drop-in Replacement

The easiest way to use these optimizations is through the enhanced vision module.

### Option 1: Modify face_helper.py (Minimal Changes)

Add at the top of `facefusion/face_helper.py`:

```python
# Add these imports at the top
try:
    from facefusion.vision_enhanced import (
        warp_face_enhanced,
        paste_back_enhanced,
        is_cuda_available
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
```

Replace the warp function:

```python
def warp_face_by_face_landmark_5(temp_vision_frame: VisionFrame, face_landmark_5: FaceLandmark5, warp_template: WarpTemplate, crop_size: Size) -> Tuple[VisionFrame, Matrix]:
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, warp_template, crop_size)

    # Use enhanced warp if available
    if ENHANCED_AVAILABLE and is_cuda_available():
        crop_vision_frame = warp_face_enhanced(
            temp_vision_frame, affine_matrix, crop_size, use_cuda=True
        )
    else:
        crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)

    return crop_vision_frame, affine_matrix
```

Replace the paste function:

```python
def paste_back(temp_vision_frame: VisionFrame, crop_vision_frame: VisionFrame, crop_vision_mask: Mask, affine_matrix: Matrix) -> VisionFrame:
    # Use enhanced paste if available
    if ENHANCED_AVAILABLE and is_cuda_available():
        return paste_back_enhanced(
            temp_vision_frame, crop_vision_frame, crop_vision_mask, affine_matrix,
            use_cuda=True, seam_band=(30, 225)
        )

    # Original implementation (fallback)
    paste_bounding_box, paste_matrix = calculate_paste_area(temp_vision_frame, crop_vision_frame, affine_matrix)
    x1, y1, x2, y2 = paste_bounding_box
    paste_width = x2 - x1
    paste_height = y2 - y1
    inverse_vision_mask = cv2.warpAffine(crop_vision_mask, paste_matrix, (paste_width, paste_height)).clip(0, 1)
    inverse_vision_mask = numpy.expand_dims(inverse_vision_mask, axis=-1)
    inverse_vision_frame = cv2.warpAffine(crop_vision_frame, paste_matrix, (paste_width, paste_height), borderMode=cv2.BORDER_REPLICATE)
    temp_vision_frame = temp_vision_frame.copy()
    paste_vision_frame = temp_vision_frame[y1:y2, x1:x2]
    paste_vision_frame = paste_vision_frame * (1 - inverse_vision_mask) + inverse_vision_frame * inverse_vision_mask
    temp_vision_frame[y1:y2, x1:x2] = paste_vision_frame.astype(temp_vision_frame.dtype)
    return temp_vision_frame
```

### Option 2: Wrapper in face_swapper.py (More Control)

Create a wrapper in `facefusion/processors/modules/face_swapper/core.py`:

```python
# Add at module level
try:
    from facefusion.vision_enhanced import (
        paste_back_enhanced,
        get_eos_guard,
        get_pose_filter,
        reset_stability_state,
        is_cuda_available
    )
    from facefusion.stability.pose_filter import extract_pose_from_landmarks
    STABILITY_ENABLED = True
except ImportError:
    STABILITY_ENABLED = False

# Global state
_current_video_frame_count = 0
_current_face_track_id = 0


def swap_face_with_stability(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame, frame_idx: int, total_frames: int) -> VisionFrame:
    """Enhanced swap_face with stability features."""

    # Original warp and processing
    model_template = get_model_options().get('template')
    model_size = get_model_options().get('size')
    pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))
    pixel_boost_total = pixel_boost_size[0] // model_size[0]

    # Warp face (enhanced if available)
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, pixel_boost_size
    )

    # Apply stability enhancements to landmarks and pose if available
    if STABILITY_ENABLED:
        # Extract 68 landmarks for pose filtering
        landmarks_68 = target_face.landmark_set.get('68')
        if landmarks_68 is not None:
            # Extract and filter pose
            pose = extract_pose_from_landmarks(landmarks_68)
            pose_filter = get_pose_filter(_current_face_track_id)
            if pose_filter:
                filtered_pose = pose_filter.update(pose)

        # Apply EOS guard if in tail window
        eos_guard = get_eos_guard(_current_face_track_id)
        if eos_guard and eos_guard.in_tail_window(frame_idx, total_frames):
            # Stabilize affine matrix
            affine_matrix = eos_guard.stabilize_affine(affine_matrix)

            # Freeze slow parameters
            params = {
                'face_mask_blur': state_manager.get_item('face_mask_blur'),
                'face_mask_padding': state_manager.get_item('face_mask_padding')
            }
            frozen_params = eos_guard.freeze_parameters(params)

    # Create masks
    temp_vision_frames = []
    crop_masks = []

    if 'box' in state_manager.get_item('face_mask_types'):
        box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
        crop_masks.append(box_mask)

    if 'occlusion' in state_manager.get_item('face_mask_types'):
        occlusion_mask = create_occlusion_mask(crop_vision_frame)
        crop_masks.append(occlusion_mask)

    # Process with model
    pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
    for pixel_boost_vision_frame in pixel_boost_vision_frames:
        pixel_boost_vision_frame = prepare_crop_frame(pixel_boost_vision_frame)
        pixel_boost_vision_frame = forward_swap_face(source_face, target_face, pixel_boost_vision_frame)
        pixel_boost_vision_frame = normalize_crop_frame(pixel_boost_vision_frame)
        temp_vision_frames.append(pixel_boost_vision_frame)

    crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

    # Additional masks
    if 'area' in state_manager.get_item('face_mask_types'):
        face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
        area_mask = create_area_mask(crop_vision_frame, face_landmark_68, state_manager.get_item('face_mask_areas'))
        crop_masks.append(area_mask)

    if 'region' in state_manager.get_item('face_mask_types'):
        region_mask = create_region_mask(crop_vision_frame, state_manager.get_item('face_mask_regions'))
        crop_masks.append(region_mask)

    # Combine masks
    crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)

    # Paste back with enhancements
    if STABILITY_ENABLED and is_cuda_available():
        paste_vision_frame = paste_back_enhanced(
            temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix,
            use_cuda=True, seam_band=(30, 225)
        )
    else:
        paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

    return paste_vision_frame


# Update the main swap_face to use wrapper
def swap_face(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    """Main swap_face function (updated to use stability features)."""

    # Try to get frame index and total frames from context if available
    frame_idx = getattr(temp_vision_frame, '_frame_idx', 0)
    total_frames = getattr(temp_vision_frame, '_total_frames', 1000)

    if STABILITY_ENABLED:
        return swap_face_with_stability(source_face, target_face, temp_vision_frame, frame_idx, total_frames)
    else:
        # Fallback to original implementation
        # ... (original code)
        pass
```

### Option 3: Workflow Integration (Frame Loop)

For best results, integrate at the workflow level in `facefusion/workflows/`:

```python
# In your workflow's frame processing loop

from facefusion.vision_enhanced import (
    reset_stability_state,
    get_optimization_status,
    get_eos_guard
)

def process_video(video_path, output_path):
    # Reset stability state at start of video
    reset_stability_state()

    # Get frame count
    total_frames = count_video_frame_total(video_path)

    print("Optimization status:", get_optimization_status())

    for frame_idx in range(total_frames):
        frame = read_video_frame(video_path, frame_idx)

        # Attach metadata for stability features
        frame._frame_idx = frame_idx
        frame._total_frames = total_frames

        # Process frame
        result_frame = process_frame_with_swapping(frame)

        # Write output
        write_frame(output_path, result_frame)
```

## Advanced: Custom Stability Parameters

### Tune Pose Filter

```python
from facefusion.stability.pose_filter import PoseFilter

# More aggressive smoothing (higher beta)
pose_filter = PoseFilter(
    beta=0.90,  # Default: 0.85
    deadband=(0.3, 0.3, 0.3, 0.001)  # Tighter deadband
)
```

### Tune EOS Guard

```python
from facefusion.stability.eos_guard import EOSStabilityGuard

# Protect last 24 frames instead of 12
eos_guard = EOSStabilityGuard(
    tail_window_frames=24,
    quant_rotation=1/4096,  # Finer quantization
    quant_translation=1/32
)
```

### Enable FHR Refinement

```python
from facefusion.stability.fhr_refine import refine_landmarks_to_subpixel, TinyFHR

# Create FHR model
fhr_model = TinyFHR(num_landmarks=68).eval().cuda()

# In landmark detection
landmarks_int = detect_face_landmark(frame, bbox, angle)
landmarks_refined = refine_landmarks_to_subpixel(
    face_crop, landmarks_int, fhr_model, device='cuda'
)
```

## Performance Monitoring

Add benchmarking to your workflow:

```python
import time
import numpy as np

# Timing
start = time.time()
result = swap_face(source, target, frame)
elapsed = time.time() - start

print(f"Frame {i}: {elapsed*1000:.1f} ms")

# tPSNR tracking
if i > 0:
    mse = np.mean((result.astype(float) - prev_result.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    print(f"  tPSNR: {psnr:.2f} dB")

prev_result = result
```

## Debugging

### Check CUDA Usage

```python
from facefusion.vision_enhanced import get_optimization_status

status = get_optimization_status()
print(f"CUDA available: {status['cuda_available']}")
print(f"Stability features: {status['stability_available']}")
print(f"FHR available: {status['fhr_available']}")
```

### Profile Kernels

```bash
# Use NVIDIA Nsight Systems
nsys profile --trace=cuda,nvtx python run.py ...

# Or use PyTorch profiler
python -m torch.utils.bottleneck run.py ...
```

### Verify Determinism

```python
# Run twice with same input
result1 = warp_face_enhanced(frame, matrix, size, use_cuda=True)
result2 = warp_face_enhanced(frame, matrix, size, use_cuda=True)

# Should be bit-exact
assert np.array_equal(result1, result2), "Not deterministic!"
```

## Migration Checklist

- [ ] Build CUDA kernels successfully
- [ ] Verify `import facefusion.cuda.ff_cuda_ops` works
- [ ] Test on single frame (compare to original)
- [ ] Test on short video (5 seconds)
- [ ] Measure tPSNR and compare to targets
- [ ] Test tail stability (last 12 frames)
- [ ] Full video test
- [ ] Benchmark performance vs base 3.5.0

## Rollback Plan

If issues arise, the system automatically falls back:

1. **CUDA fails** → Falls back to OpenCV (with deterministic flags where possible)
2. **PyTorch unavailable** → FHR disabled, integer landmarks used
3. **Build fails** → Original FaceFusion 3.5.0 behavior (no code changes needed)

To completely disable:

```python
# In vision_enhanced.py, set:
FORCE_DISABLE_CUDA = True
FORCE_DISABLE_STABILITY = True
```

---

**Next Steps**:
1. Build kernels: `cd facefusion/cuda && mkdir build && cmake ..`
2. Test import: `python -c "from facefusion.cuda import ff_cuda_ops"`
3. Run benchmark: `python benchmark_optimizations.py`
4. Integrate into your workflow using Option 1 or 2 above
