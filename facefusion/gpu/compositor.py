"""High-quality GPU ROI compositor with multi-band blending."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy
import torch
import torch.nn.functional as F

from facefusion.gpu.kernels import warp_face


_HAS_OFA = bool(hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'NvidiaOpticalFlow_2_0'))


@dataclass
class CompositeResult:
    """Container for ROI composition outputs."""
    frame_bgr: torch.Tensor
    warped_srgb: torch.Tensor
    mask: torch.Tensor
    blended_linear: torch.Tensor
    source_linear: torch.Tensor


@dataclass
class _ColorState:
    src_mean: numpy.ndarray
    src_std: numpy.ndarray
    tgt_mean: numpy.ndarray
    tgt_std: numpy.ndarray
    # Linear-space gain/bias for stable relighting
    gain_rgb: Optional[torch.Tensor] = None  # Per-channel multiplicative gain
    bias_rgb: Optional[torch.Tensor] = None  # Per-channel additive bias


@dataclass
class _SeamBandState:
    """Cache for temporal seam band reuse to eliminate edge flicker."""
    prev_seam_bgr: Optional[torch.Tensor] = None
    prev_seam_mask: Optional[torch.Tensor] = None
    prev_center_x: float = 0.0
    prev_center_y: float = 0.0
    seam_width: int = 12  # Width of seam band in pixels


class _TemporalState:
    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.prev_bg_gray: Optional[numpy.ndarray] = None
        self.prev_comp_bgr: Optional[numpy.ndarray] = None
        self.prev_mask: Optional[numpy.ndarray] = None
        self._pending_gray: Optional[numpy.ndarray] = None
        # Optical flow-guided fusion state
        self.prev_face_patch: Optional[torch.Tensor] = None  # Previous swapped face in linear space
        if not _HAS_OFA:
            raise RuntimeError('OFA unavailable')
        self._flow = cv2.cuda_NvidiaOpticalFlow_2_0.create(
            (self.width, self.height),
            perfPreset=cv2.cuda.NVIDIA_OF_PERF_LEVEL_FAST,
            outputGridSize=cv2.cuda.NVIDIA_OF_OUTPUT_VECTOR_GRID_SIZE_4,
            hintGridSize=cv2.cuda.NVIDIA_OF_HINT_VECTOR_GRID_SIZE_4,
            enableTemporalHints=True
        )
        x_coords = numpy.tile(numpy.arange(self.width, dtype=numpy.float32), (self.height, 1))
        y_coords = numpy.repeat(numpy.arange(self.height, dtype=numpy.float32)[:, None], self.width, axis=1)
        self._grid_x = cv2.cuda_GpuMat()
        self._grid_y = cv2.cuda_GpuMat()
        self._grid_x.upload(x_coords)
        self._grid_y.upload(y_coords)

    def calc_flow(self, prev_gray: numpy.ndarray, curr_gray: numpy.ndarray) -> Tuple['cv2.cuda.GpuMat', 'cv2.cuda.GpuMat']:
        prev_gpu = cv2.cuda_GpuMat()
        curr_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(prev_gray)
        curr_gpu.upload(curr_gray)
        flow_gpu = self._flow.calc(prev_gpu, curr_gpu, None)
        cost_gpu = self._flow.getFlowConfidence()
        return flow_gpu, cost_gpu

    def prepare_flow(self, curr_gray: numpy.ndarray) -> Optional[Tuple['cv2.cuda.GpuMat', 'cv2.cuda.GpuMat']]:
        self._pending_gray = curr_gray
        if self.prev_bg_gray is None:
            return None
        return self.calc_flow(self.prev_bg_gray, curr_gray)

    def warp_previous(
        self,
        prev_comp_bgr: numpy.ndarray,
        flow_gpu: 'cv2.cuda.GpuMat',
        cost_gpu: 'cv2.cuda.GpuMat',
        threshold: float
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        planes = cv2.cuda.split(flow_gpu)
        map_x = cv2.cuda.add(self._grid_x, planes[0])
        map_y = cv2.cuda.add(self._grid_y, planes[1])
        src_gpu = cv2.cuda_GpuMat()
        src_gpu.upload(prev_comp_bgr)
        warped_gpu = cv2.cuda.remap(src_gpu, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        warped = warped_gpu.download()
        mask_gpu = cv2.cuda.compare(cost_gpu, threshold, cv2.CMP_LT)
        mask_gpu = mask_gpu.convertTo(cv2.CV_32F, 1.0 / 255.0)
        mask_gpu = cv2.cuda.blur(mask_gpu, (5, 5))
        reliable = mask_gpu.download()
        return warped, reliable

    def warp_mask(
        self,
        prev_mask: numpy.ndarray,
        flow_gpu: 'cv2.cuda.GpuMat',
        cost_gpu: 'cv2.cuda.GpuMat',
        threshold: float
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        planes = cv2.cuda.split(flow_gpu)
        map_x = cv2.cuda.add(self._grid_x, planes[0])
        map_y = cv2.cuda.add(self._grid_y, planes[1])
        src_gpu = cv2.cuda_GpuMat()
        src_gpu.upload(prev_mask)
        warped_gpu = cv2.cuda.remap(src_gpu, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        warped = warped_gpu.download()
        mask_gpu = cv2.cuda.compare(cost_gpu, threshold, cv2.CMP_LT)
        mask_gpu = mask_gpu.convertTo(cv2.CV_32F, 1.0 / 255.0)
        mask_gpu = cv2.cuda.blur(mask_gpu, (5, 5))
        reliable = mask_gpu.download()
        return warped, reliable

    def reset(self) -> None:
        self.prev_bg_gray = None
        self.prev_comp_bgr = None
        self.prev_mask = None
        self._pending_gray = None
        self.prev_face_patch = None

    def warp_face_patch(
        self,
        prev_patch_linear: torch.Tensor,
        flow_gpu: 'cv2.cuda.GpuMat',
        cost_gpu: 'cv2.cuda.GpuMat',
        threshold: float
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Warp previous face patch forward using optical flow. Returns (warped_patch, reliability_mask)."""
        try:
            # Convert to numpy for cv2 warping
            prev_np = (prev_patch_linear.detach().cpu().numpy() * 255.0).astype(numpy.uint8)

            planes = cv2.cuda.split(flow_gpu)
            map_x = cv2.cuda.add(self._grid_x, planes[0])
            map_y = cv2.cuda.add(self._grid_y, planes[1])

            src_gpu = cv2.cuda_GpuMat()
            src_gpu.upload(prev_np)
            warped_gpu = cv2.cuda.remap(src_gpu, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            warped = warped_gpu.download()

            # Reliability mask
            mask_gpu = cv2.cuda.compare(cost_gpu, threshold, cv2.CMP_LT)
            mask_gpu = mask_gpu.convertTo(cv2.CV_32F, 1.0 / 255.0)
            mask_gpu = cv2.cuda.blur(mask_gpu, (5, 5))
            reliable = mask_gpu.download()

            # Convert back to tensor
            warped_tensor = torch.from_numpy(warped.astype(numpy.float32) / 255.0).to(prev_patch_linear.device)
            reliable_tensor = torch.from_numpy(reliable).to(prev_patch_linear.device)

            return warped_tensor, reliable_tensor
        except Exception:
            return None

class GpuRoiComposer:
    """Compose warped face crops back into a frame ROI on the GPU."""

    def __init__(self, device: torch.device, roi_size: Tuple[int, int], levels: int = 3) -> None:
        self.device = torch.device(device)
        self.roi_height, self.roi_width = roi_size
        self.levels = max(1, levels)
        self._gaussian_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._temporal_states: Dict[str, _TemporalState] = {}
        self._color_states: Dict[str, _ColorState] = {}
        self._seam_states: Dict[str, _SeamBandState] = {}
        self.temporal_alpha = 0.2
        self.temporal_tau = 0.35
        self.color_momentum = 0.1
        self._sdf_half_width = 1.5
        # Temporal snap: if face patch is similar to previous frame, reduce movement
        self._prev_warped: Dict[str, torch.Tensor] = {}
        self.snap_threshold = 0.95  # SSIM threshold for temporal snapping (higher = more aggressive)
        # Seam band temporal reuse parameters
        self.seam_motion_threshold = 3.0  # pixels - reuse seam if movement < this
        # Anomaly detection parameters
        self._prev_quality: Dict[str, float] = {}  # Track quality per face
        self.anomaly_threshold = 0.70  # If quality drops below this, trigger aggressive smoothing

    # ------------------------------------------------------------------
    def compose(
        self,
        src_bgr: torch.Tensor,
        mask: torch.Tensor,
        bg_bgr: torch.Tensor,
        affine: torch.Tensor,
        track_token: Optional[str] = None,
        extras: Optional[dict] = None,
    ) -> CompositeResult:
        if extras is None:
            extras = {}
        track_key = str(track_token) if track_token is not None else None
        bg_reference = extras.get('bg_reference') if extras else None

        src_bgr = self._prepare_color_tensor(src_bgr)
        bg_bgr = self._prepare_color_tensor(bg_bgr, roi=True)
        mask = self._prepare_mask_tensor(mask)
        affine = affine.to(self.device, dtype=torch.float32).contiguous().view(-1)

        src_h, src_w, _ = src_bgr.shape
        roi_h, roi_w = bg_bgr.shape
        src_rgba = torch.ones((src_h, src_w, 4), device=self.device, dtype=torch.float32)
        src_rgba[..., :3] = src_bgr
        if 'sdf' in extras:
            sdf = extras['sdf'].to(self.device, dtype=torch.float32)
            if sdf.shape != src_bgr[..., 0].shape:
                raise ValueError('SDF tensor must match source crop spatial size.')
            src_rgba[..., 3] = sdf

        warped = warp_face(src_rgba, affine, roi_w, roi_h)
        warped_rgb = torch.clamp(warped[..., :3], 0.0, 1.0)
        warped_sdf = warped[..., 3] if warped.size(-1) > 3 else None

        if warped_sdf is not None:
            sdf_alpha = self._compute_sdf_alpha(warped_sdf)
            mask = torch.clamp(mask * sdf_alpha, 0.0, 1.0)

        # Apply temporal snap BEFORE color transfer to reduce warping micro-jitter
        if track_key is not None:
            warped_rgb = self._apply_temporal_snap(track_key, warped_rgb)

        state: Optional[_TemporalState] = None
        curr_gray: Optional[numpy.ndarray] = None
        flow_bundle: Optional[Tuple['cv2.cuda.GpuMat', 'cv2.cuda.GpuMat']] = None
        mask_np = mask.detach().cpu().numpy().astype(numpy.float32)

        if _HAS_OFA and track_key is not None and isinstance(bg_reference, numpy.ndarray):
            state = self._ensure_temporal_state(track_key)
            if state is not None:
                try:
                    curr_gray = cv2.cvtColor(bg_reference, cv2.COLOR_BGR2GRAY)
                    flow_bundle = state.prepare_flow(curr_gray)
                    mask, mask_np = self._stabilize_mask(state, mask, mask_np, flow_bundle)
                except Exception:
                    state.reset()
                    curr_gray = None
                    flow_bundle = None
                    mask_np = mask.detach().cpu().numpy().astype(numpy.float32)

        if track_key is not None and isinstance(bg_reference, numpy.ndarray):
            try:
                warped_rgb = self._apply_color_transfer(track_key, warped_rgb, mask, bg_reference)
            except Exception:
                pass

        warped_rgb_t = warped_rgb.permute(2, 0, 1).unsqueeze(0)
        bg_rgb_t = bg_bgr.permute(2, 0, 1).unsqueeze(0)
        mask_t = mask.unsqueeze(0).unsqueeze(0)

        src_lin = self._srgb_to_linear(warped_rgb_t)
        bg_lin = self._srgb_to_linear(bg_rgb_t)

        # Optical flow-guided temporal fusion: conditionally blend with warped-previous
        # AND anomaly detection: if quality drops significantly, use more aggressive smoothing
        is_anomaly = False
        if state is not None and flow_bundle is not None and state.prev_face_patch is not None:
            try:
                flow_gpu, cost_gpu = flow_bundle
                warp_result = state.warp_face_patch(state.prev_face_patch, flow_gpu, cost_gpu, float(self.temporal_tau))
                if warp_result is not None:
                    warped_prev_linear, flow_reliable = warp_result
                    # Compute similarity between current and warped-previous in linear space
                    # Use just the face region (where mask > 0.1)
                    similarity = self._compute_similarity(src_lin.squeeze(0).permute(1, 2, 0), warped_prev_linear)

                    # Anomaly detection: track quality per frame
                    prev_quality = self._prev_quality.get(track_key, 1.0) if track_key else 1.0
                    current_quality = similarity

                    # If quality dropped significantly from previous frame, mark as anomaly
                    if current_quality < self.anomaly_threshold and (prev_quality - current_quality) > 0.15:
                        is_anomaly = True
                        # VERY aggressive smoothing for anomaly frames to prevent catastrophic dips
                        src_lin_permuted = src_lin.squeeze(0).permute(1, 2, 0)
                        fused = 0.1 * src_lin_permuted + 0.9 * warped_prev_linear  # 90% previous!
                        src_lin = fused.unsqueeze(0).permute(0, 3, 1, 2)
                    elif similarity > 0.85:
                        # Normal high similarity: blend 70% toward stabilized warped-previous
                        src_lin_permuted = src_lin.squeeze(0).permute(1, 2, 0)
                        fused = 0.3 * src_lin_permuted + 0.7 * warped_prev_linear
                        src_lin = fused.unsqueeze(0).permute(0, 3, 1, 2)

                    # Update quality history
                    if track_key:
                        self._prev_quality[track_key] = current_quality
            except Exception:
                pass  # Fall back to using current frame

        blended_lin = self._multi_band_blend(src_lin, bg_lin, mask_t)

        # Cache current face patch in linear space for next frame's optical flow fusion
        if state is not None:
            state.prev_face_patch = src_lin.squeeze(0).permute(1, 2, 0).detach().clone()

        out_srgb = self._linear_to_srgb(blended_lin)
        out_srgb = self._apply_dither(out_srgb)
        out_srgb = torch.clamp(out_srgb, 0.0, 1.0)

        out_uint8 = torch.clamp(out_srgb * 255.0 + 0.5, 0.0, 255.0).to(torch.uint8)
        frame_bgr = out_uint8.squeeze(0).permute(1, 2, 0).contiguous()

        if state is not None and curr_gray is not None:
            try:
                frame_bgr, blended_lin = self._apply_temporal(state, curr_gray, flow_bundle, frame_bgr, blended_lin, mask_np)
            except Exception:
                state.reset()

        # Apply seam band blending with temporal reuse (after temporal smoothing) to eliminate edge flicker
        if track_key is not None:
            frame_bgr_float = frame_bgr.float() / 255.0
            frame_bgr_float = self._apply_seam_band_blending(track_key, frame_bgr_float, mask)
            frame_bgr = torch.clamp(frame_bgr_float * 255.0 + 0.5, 0.0, 255.0).to(torch.uint8)

        return CompositeResult(
            frame_bgr=frame_bgr,
            warped_srgb=warped_rgb_t.squeeze(0),
            mask=mask_t.squeeze(0),
            blended_linear=blended_lin.squeeze(0),
            source_linear=src_lin.squeeze(0)
        )

    # ------------------------------------------------------------------
    def _prepare_color_tensor(self, tensor: torch.Tensor, roi: bool = False) -> torch.Tensor:
        tensor = tensor.to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        if tensor.max() > 1.5:
            tensor = tensor * (1.0 / 255.0)
        if tensor.dim() == 3:
            h, w, c = tensor.shape
            if c != 3:
                raise ValueError('Color tensor must have 3 channels.')
        elif tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        else:
            raise ValueError('Unexpected color tensor shape.')
        if roi:
            if tensor.shape[0] != self.roi_height or tensor.shape[1] != self.roi_width:
                raise ValueError('Background ROI shape mismatch.')
        return tensor.contiguous()

    def _prepare_mask_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError('Mask tensor must be 2D.')
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape != (self.roi_height, self.roi_width):
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=(self.roi_height, self.roi_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        return tensor.contiguous()

    def _get_gaussian_kernels(self, channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._gaussian_cache.get(channels)
        if cached is not None:
            return cached
        base = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=self.device, dtype=torch.float32)
        base = (base / base.sum()).view(1, 1, 1, 5)
        kernel_h = base.repeat(channels, 1, 1, 1)
        kernel_v = base.view(1, 1, 5, 1).repeat(channels, 1, 1, 1)
        self._gaussian_cache[channels] = (kernel_v, kernel_h)
        return kernel_v, kernel_h

    def _gaussian_blur(self, tensor: torch.Tensor) -> torch.Tensor:
        channels = tensor.shape[1]
        kernel_v, kernel_h = self._get_gaussian_kernels(channels)
        padding = (2, 2)
        tmp = F.conv2d(tensor, kernel_h, padding=(0, padding[1]), groups=channels)
        blurred = F.conv2d(tmp, kernel_v, padding=(padding[0], 0), groups=channels)
        return blurred

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        new_h = max(1, h // 2)
        new_w = max(1, w // 2)
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    def _upsample(self, tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)

    def _multi_band_blend(self, src_lin: torch.Tensor, dst_lin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.levels == 1:
            return src_lin * mask + dst_lin * (1.0 - mask)

        lap_src = []
        lap_dst = []
        mask_levels = []

        curr_src = src_lin
        curr_dst = dst_lin
        curr_mask = mask
        gauss_src = [curr_src]
        gauss_dst = [curr_dst]
        gauss_mask = [curr_mask]

        for _ in range(self.levels - 1):
            blurred_src = self._gaussian_blur(curr_src)
            blurred_dst = self._gaussian_blur(curr_dst)
            blurred_mask = self._gaussian_blur(curr_mask)

            lap_src.append(curr_src - blurred_src)
            lap_dst.append(curr_dst - blurred_dst)
            mask_levels.append(torch.clamp(blurred_mask, 0.0, 1.0))

            curr_src = self._downsample(blurred_src)
            curr_dst = self._downsample(blurred_dst)
            curr_mask = self._downsample(blurred_mask)

            gauss_src.append(curr_src)
            gauss_dst.append(curr_dst)
            gauss_mask.append(curr_mask)

        base_mask = torch.clamp(gauss_mask[-1], 0.0, 1.0)
        blended = gauss_src[-1] * base_mask + gauss_dst[-1] * (1.0 - base_mask)

        for level in reversed(range(self.levels - 1)):
            blended = self._upsample(blended, gauss_src[level].shape[-2:])
            lap_blend = lap_src[level] * mask_levels[level] + lap_dst[level] * (1.0 - mask_levels[level])
            blended = blended + lap_blend

        return blended

    def _ensure_temporal_state(self, track_token: str) -> Optional[_TemporalState]:
        state = self._temporal_states.get(track_token)
        if state is None:
            try:
                state = _TemporalState(self.roi_width, self.roi_height)
            except Exception:
                return None
            self._temporal_states[track_token] = state
        return state

    def _stabilize_mask(
        self,
        state: _TemporalState,
        mask_tensor: torch.Tensor,
        mask_numpy: numpy.ndarray,
        flow_bundle: Optional[Tuple['cv2.cuda.GpuMat', 'cv2.cuda.GpuMat']]
    ) -> Tuple[torch.Tensor, numpy.ndarray]:
        if state.prev_mask is None or flow_bundle is None:
            state.prev_mask = mask_numpy.copy()
            return mask_tensor, mask_numpy

        flow_gpu, cost_gpu = flow_bundle
        warped_mask, reliable = state.warp_mask(state.prev_mask, flow_gpu, cost_gpu, float(self.temporal_tau))
        reliable_t = torch.from_numpy(reliable).to(self.device, dtype=torch.float32)
        warped_t = torch.from_numpy(warped_mask).to(self.device, dtype=torch.float32)
        mask_tensor = torch.clamp(mask_tensor * (1.0 - reliable_t) + warped_t * reliable_t, 0.0, 1.0)
        mask_numpy = mask_tensor.detach().cpu().numpy().astype(numpy.float32)
        return mask_tensor, mask_numpy

    def _apply_temporal(
        self,
        state: _TemporalState,
        curr_gray: numpy.ndarray,
        flow_bundle: Optional[Tuple['cv2.cuda.GpuMat', 'cv2.cuda.GpuMat']],
        frame_bgr: torch.Tensor,
        blended_lin: torch.Tensor,
        mask_numpy: numpy.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.prev_bg_gray is None or state.prev_comp_bgr is None:
            state.prev_bg_gray = curr_gray
            state.prev_comp_bgr = frame_bgr.cpu().numpy()
            state.prev_mask = mask_numpy.copy()
            return frame_bgr, blended_lin

        try:
            if flow_bundle is None:
                flow_bundle = state.calc_flow(state.prev_bg_gray, curr_gray)
            if flow_bundle is None:
                state.prev_bg_gray = curr_gray
                state.prev_comp_bgr = frame_bgr.cpu().numpy()
                state.prev_mask = mask_numpy.copy()
                return frame_bgr, blended_lin
            flow_gpu, cost_gpu = flow_bundle
            warped_prev, reliable = state.warp_previous(state.prev_comp_bgr, flow_gpu, cost_gpu, float(self.temporal_tau))
        except Exception:
            state.reset()
            state.prev_bg_gray = curr_gray
            state.prev_comp_bgr = frame_bgr.cpu().numpy()
            state.prev_mask = mask_numpy.copy()
            return frame_bgr, blended_lin

        prev_tensor = torch.from_numpy(warped_prev).to(self.device, dtype=torch.float32) * (1.0 / 255.0)
        curr_tensor = frame_bgr.to(torch.float32) * (1.0 / 255.0)

        prev_lin = self._srgb_to_linear(prev_tensor.permute(2, 0, 1).unsqueeze(0))
        curr_lin = self._srgb_to_linear(curr_tensor.permute(2, 0, 1).unsqueeze(0))

        reliable_tensor = torch.from_numpy(reliable).to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        reliable_tensor = torch.clamp(reliable_tensor, 0.0, 1.0)

        temp_lin = prev_lin * self.temporal_alpha + curr_lin * (1.0 - self.temporal_alpha)
        blended_lin = curr_lin * (1.0 - reliable_tensor) + temp_lin * reliable_tensor

        out_srgb = self._linear_to_srgb(blended_lin)
        out_uint8 = torch.clamp(out_srgb * 255.0 + 0.5, 0.0, 255.0).to(torch.uint8)
        frame_bgr = out_uint8.squeeze(0).permute(1, 2, 0).contiguous()

        state.prev_bg_gray = curr_gray
        state.prev_comp_bgr = frame_bgr.cpu().numpy()
        state.prev_mask = mask_numpy.copy()

        return frame_bgr, blended_lin

    def _apply_color_transfer(
        self,
        track_token: str,
        warped_rgb: torch.Tensor,
        mask: torch.Tensor,
        bg_reference: numpy.ndarray
    ) -> torch.Tensor:
        """Apply color transfer in LINEAR light space with quantized, lowpass gain/bias."""
        mask_cpu = mask.detach().cpu().numpy()
        region = mask_cpu > 0.1
        if region.sum() < 16:
            return warped_rgb

        # Convert to linear space for perceptually uniform color matching
        warped_linear = self._srgb_to_linear(warped_rgb.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0)
        bg_float = torch.from_numpy(bg_reference.astype(numpy.float32) / 255.0).to(self.device)
        bg_linear = self._srgb_to_linear(bg_float.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0)

        # Get masked regions
        mask_3d = mask.unsqueeze(-1).expand_as(warped_linear)
        region_tensor = mask > 0.1

        warped_masked = warped_linear[region_tensor.unsqueeze(-1).expand_as(warped_linear)].view(-1, 3)
        bg_masked = bg_linear[region_tensor.unsqueeze(-1).expand_as(bg_linear)].view(-1, 3)

        if warped_masked.shape[0] < 16:
            return warped_rgb

        # Compute per-channel statistics in linear space
        src_mean_now = warped_masked.mean(dim=0)
        src_std_now = warped_masked.std(dim=0) + 1e-6
        tgt_mean_now = bg_masked.mean(dim=0)
        tgt_std_now = bg_masked.std(dim=0) + 1e-6

        # Compute raw gain and bias
        raw_gain = tgt_std_now / src_std_now
        raw_bias = tgt_mean_now - raw_gain * src_mean_now

        # Quantize to prevent micro-pumping (round to nearest 1/256)
        quantize_steps = 256.0
        raw_gain = torch.round(raw_gain * quantize_steps) / quantize_steps
        raw_bias = torch.round(raw_bias * quantize_steps) / quantize_steps

        # Get or create state
        state = self._color_states.get(track_token)
        if state is None:
            # First frame - initialize with quantized values
            state = _ColorState(
                src_mean=src_mean_now.detach().cpu().numpy(),
                src_std=src_std_now.detach().cpu().numpy(),
                tgt_mean=tgt_mean_now.detach().cpu().numpy(),
                tgt_std=tgt_std_now.detach().cpu().numpy(),
                gain_rgb=raw_gain.detach().clone(),
                bias_rgb=raw_bias.detach().clone()
            )
            self._color_states[track_token] = state
        else:
            # Lowpass filter gain/bias to prevent frame-to-frame pumping
            momentum = self.color_momentum
            state.gain_rgb = (1.0 - momentum) * state.gain_rgb + momentum * raw_gain
            state.bias_rgb = (1.0 - momentum) * state.bias_rgb + momentum * raw_bias
            # Quantize filtered values too
            state.gain_rgb = torch.round(state.gain_rgb * quantize_steps) / quantize_steps
            state.bias_rgb = torch.round(state.bias_rgb * quantize_steps) / quantize_steps

        # Apply color correction in linear space
        adjusted_linear = warped_linear * state.gain_rgb + state.bias_rgb
        adjusted_linear = torch.clamp(adjusted_linear, 0.0, 1.0)

        # Convert back to sRGB
        adjusted_srgb = self._linear_to_srgb(adjusted_linear.unsqueeze(0).permute(0, 3, 1, 2)).squeeze(0).permute(1, 2, 0)
        return torch.clamp(adjusted_srgb, 0.0, 1.0)

    @staticmethod
    def _reinhard_transfer(
        lab_image: numpy.ndarray,
        src_mean: numpy.ndarray,
        src_std: numpy.ndarray,
        tgt_mean: numpy.ndarray,
        tgt_std: numpy.ndarray
    ) -> numpy.ndarray:
        adjusted = lab_image.copy()
        ratios = tgt_std / src_std
        for idx in range(3):
            adjusted[..., idx] = (adjusted[..., idx] - src_mean[idx]) * ratios[idx] + tgt_mean[idx]
        adjusted[..., 0] = numpy.clip(adjusted[..., 0], 0.0, 100.0)
        adjusted[..., 1] = numpy.clip(adjusted[..., 1], -128.0, 127.0)
        adjusted[..., 2] = numpy.clip(adjusted[..., 2], -128.0, 127.0)
        return adjusted

    @staticmethod
    def _srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return torch.where(
            tensor <= 0.04045,
            tensor / 12.92,
            torch.pow(torch.clamp((tensor + 0.055) / 1.055, min=0.0), 2.4)
        )

    @staticmethod
    def _linear_to_srgb(tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return torch.where(
            tensor <= 0.0031308,
            tensor * 12.92,
            1.055 * torch.pow(torch.clamp(tensor, min=0.0), 1.0 / 2.4) - 0.055
        )

    def _apply_dither(self, tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[-2:]
        y = torch.arange(h, device=self.device, dtype=torch.float32).view(1, 1, h, 1)
        x = torch.arange(w, device=self.device, dtype=torch.float32).view(1, 1, 1, w)
        noise = torch.sin((x * 12.9898 + y * 78.233) * 43758.5453)
        noise = torch.frac(noise) - 0.5
        noise = noise.repeat(1, tensor.shape[1], 1, 1)
        return tensor + noise * (1.0 / 512.0)

    def _compute_sdf_alpha(self, sdf: torch.Tensor) -> torch.Tensor:
        width = float(self._sdf_half_width)
        t = torch.clamp((sdf + width) / (2.0 * width), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _compute_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Fast approximate SSIM using mean squared difference."""
        if img1.shape != img2.shape:
            return 0.0
        # Simple MSE-based similarity (faster than full SSIM)
        mse = torch.mean((img1 - img2) ** 2).item()
        # Convert MSE to similarity score (0-1 range, 1 = identical)
        # Assuming images are in [0, 1] range, MSE of 0.01 is ~90% similar
        similarity = max(0.0, 1.0 - (mse * 50.0))  # Scale MSE to similarity
        return similarity

    def _apply_temporal_snap(self, track_token: Optional[str], warped_rgb: torch.Tensor) -> torch.Tensor:
        """Apply temporal snapping: if current face is very similar to previous, blend toward previous."""
        if track_token is None:
            return warped_rgb

        prev_warped = self._prev_warped.get(track_token)
        if prev_warped is None:
            # First frame for this track, just cache and return
            self._prev_warped[track_token] = warped_rgb.detach().clone()
            return warped_rgb

        # Check similarity
        similarity = self._compute_similarity(warped_rgb, prev_warped)

        if similarity >= self.snap_threshold:
            # Very similar to previous frame - blend heavily toward previous to reduce jitter
            # Use 0.3 weight for current, 0.7 for previous (aggressive smoothing)
            snapped = 0.3 * warped_rgb + 0.7 * prev_warped
            self._prev_warped[track_token] = snapped.detach().clone()
            return snapped
        else:
            # Different enough - use current frame but cache it
            self._prev_warped[track_token] = warped_rgb.detach().clone()
            return warped_rgb

    def _apply_seam_band_blending(
        self,
        track_token: Optional[str],
        blended_bgr: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply multiband blending on seam edge with temporal reuse to eliminate edge flicker."""
        if track_token is None:
            return blended_bgr

        # Get or create seam state
        seam_state = self._seam_states.get(track_token)
        if seam_state is None:
            seam_state = _SeamBandState()
            self._seam_states[track_token] = seam_state

        # Compute mask center for motion tracking
        mask_binary = (mask > 0.5).float()
        if mask_binary.sum() < 10:
            return blended_bgr  # No valid mask

        ys, xs = torch.where(mask_binary > 0)
        center_x = xs.float().mean().item()
        center_y = ys.float().mean().item()

        # Check if we should reuse previous seam (low motion)
        if seam_state.prev_seam_bgr is not None:
            motion = ((center_x - seam_state.prev_center_x) ** 2 +
                     (center_y - seam_state.prev_center_y) ** 2) ** 0.5

            if motion < self.seam_motion_threshold:
                # Low motion - reuse previous seam band
                # Extract seam band region (dilated - original)
                kernel_size = seam_state.seam_width
                # Use max pooling as dilation
                mask_4d = mask.unsqueeze(0).unsqueeze(0)
                dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size*2+1, stride=1, padding=kernel_size)
                dilated = dilated.squeeze(0).squeeze(0)

                # Seam band = dilated - original
                seam_band = torch.clamp(dilated - mask, 0.0, 1.0)

                # Blend previous seam into current frame at seam band
                seam_band_3d = seam_band.unsqueeze(-1)
                blended_bgr = blended_bgr * (1.0 - seam_band_3d) + seam_state.prev_seam_bgr * seam_band_3d

        # Update state with current frame
        seam_state.prev_seam_bgr = blended_bgr.detach().clone()
        seam_state.prev_center_x = center_x
        seam_state.prev_center_y = center_y

        return blended_bgr

    def reset_temporal_state(self) -> None:
        for state in self._temporal_states.values():
            state.reset()
        self._temporal_states.clear()
        self._color_states.clear()
        self._prev_warped.clear()
        self._seam_states.clear()
        self._prev_quality.clear()


_COMPOSER_CACHE: Dict[Tuple[int, int, int], GpuRoiComposer] = {}


def get_composer(device: torch.device, height: int, width: int) -> GpuRoiComposer:
    """Return a cached composer for ``device`` and ROI size."""
    device = torch.device(device)
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (index, height, width)
    composer = _COMPOSER_CACHE.get(key)
    if composer is None:
        composer = GpuRoiComposer(device, (height, width))
        _COMPOSER_CACHE[key] = composer
    return composer


def reset_all_temporal_states() -> None:
	for composer in _COMPOSER_CACHE.values():
		composer.reset_temporal_state()
