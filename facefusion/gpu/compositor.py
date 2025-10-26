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


class _TemporalState:
    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.prev_bg_gray: Optional[numpy.ndarray] = None
        self.prev_comp_bgr: Optional[numpy.ndarray] = None
        self.prev_mask: Optional[numpy.ndarray] = None
        self._pending_gray: Optional[numpy.ndarray] = None
        self.motion_level: float = 0.0
        self.occlusion_ratio: float = 0.0
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
        self.occlusion_ratio = 1.0 - float(cv2.cuda.mean(mask_gpu)[0])
        self.motion_level = self._measure_motion(flow_gpu)
        reliable = mask_gpu.download()
        return warped, reliable

    def reset(self) -> None:
        self.prev_bg_gray = None
        self.prev_comp_bgr = None
        self.prev_mask = None
        self._pending_gray = None
        self.motion_level = 0.0
        self.occlusion_ratio = 0.0

    def _measure_motion(self, flow_gpu: 'cv2.cuda.GpuMat') -> float:
        try:
            planes = cv2.cuda.split(flow_gpu)
            mag = cv2.cuda.magnitude(planes[0], planes[1])
            return float(cv2.cuda.mean(mag)[0])
        except Exception:
            flow = flow_gpu.download()
            return float(numpy.mean(numpy.linalg.norm(flow, axis=2)))

class GpuRoiComposer:
    """Compose warped face crops back into a frame ROI on the GPU."""

    def __init__(self, device: torch.device, roi_size: Tuple[int, int], levels: int = 3) -> None:
        self.device = torch.device(device)
        self.roi_height, self.roi_width = roi_size
        self.levels = max(1, levels)
        self._gaussian_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._temporal_states: Dict[str, _TemporalState] = {}
        self._color_states: Dict[str, _ColorState] = {}
        self.temporal_alpha = 0.2
        self.temporal_tau = 0.35
        self.color_momentum = 0.1
        self._sdf_half_width = 1.5

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

        current_stream = torch.cuda.current_stream(self.device) if torch.cuda.is_available() and self.device.type == 'cuda' else None
        if current_stream is not None:
            if src_bgr.is_cuda:
                src_bgr.record_stream(current_stream)
            if bg_bgr.is_cuda:
                bg_bgr.record_stream(current_stream)
            if mask.is_cuda:
                mask.record_stream(current_stream)

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
        if current_stream is not None:
            warp_fence = torch.cuda.Event(enable_timing=False, blocking=False)
            warp_fence.record(current_stream)
            current_stream.wait_event(warp_fence)
        warped_rgb = torch.clamp(warped[..., :3], 0.0, 1.0)
        warped_sdf = warped[..., 3] if warped.size(-1) > 3 else None

        if warped_sdf is not None:
            sdf_alpha = self._compute_sdf_alpha(warped_sdf)
            mask = torch.clamp(mask * sdf_alpha, 0.0, 1.0)

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
        blended_lin = self._multi_band_blend(src_lin, bg_lin, mask_t)

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
        temporal_state = self._temporal_states.get(track_token)
        motion_level = 0.0
        if temporal_state is not None:
            motion_level = getattr(temporal_state, 'motion_level', 0.0)
        momentum = self.color_momentum
        if motion_level > 1.2:
            momentum = 0.02
        elif motion_level > 0.6:
            momentum = 0.05
        else:
            momentum = 0.12
        mask_cpu = mask.detach().cpu().numpy()
        region = mask_cpu > 0.1
        if region.sum() < 16:
            return warped_rgb

        warped_cpu = numpy.clip(warped_rgb.detach().cpu().numpy().astype(numpy.float32), 0.0, 1.0)
        bg_float = numpy.clip(bg_reference.astype(numpy.float32) / 255.0, 0.0, 1.0)

        warped_lab = cv2.cvtColor(warped_cpu, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_float, cv2.COLOR_BGR2LAB)

        src_mean = warped_lab[region].mean(axis=0)
        src_std = warped_lab[region].std(axis=0) + 1e-6
        tgt_mean_now = bg_lab[region].mean(axis=0)
        tgt_std_now = bg_lab[region].std(axis=0) + 1e-6

        state = self._color_states.get(track_token)
        if state is None:
            state = _ColorState(src_mean=src_mean, src_std=src_std, tgt_mean=tgt_mean_now, tgt_std=tgt_std_now)
            self._color_states[track_token] = state
        else:
            state.tgt_mean = (1.0 - momentum) * state.tgt_mean + momentum * tgt_mean_now
            state.tgt_std = (1.0 - momentum) * state.tgt_std + momentum * tgt_std_now
            # Keep source statistics stable once initialised

        adjusted_lab = self._reinhard_transfer(warped_lab, state.src_mean, state.src_std, state.tgt_mean, state.tgt_std)
        adjusted_bgr = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
        adjusted_bgr = numpy.clip(adjusted_bgr, 0.0, 1.0).astype(numpy.float32)
        return torch.from_numpy(adjusted_bgr).to(self.device)

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

    def reset_temporal_state(self) -> None:
        for state in self._temporal_states.values():
            state.reset()
        self._temporal_states.clear()
        self._color_states.clear()


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
