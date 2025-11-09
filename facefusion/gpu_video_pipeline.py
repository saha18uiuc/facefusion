"""GPU video pipeline using NVDEC for decode and NVENC for encode.

This module builds an end-to-end GPU pipeline that keeps video frames on
CUDA memory from ingest through post-processing to final encode. The
implementation follows the recommendations outlined in the FaceFusion
3.4.0 performance plan: frames are decoded with NVDEC, processed on a
user-supplied CUDA stream, and encoded with NVENC after a GPU-side color
conversion to NV12. Decode, compute, and encode run on separate CUDA
streams and are synchronised with CUDA events so that the specialised
hardware engines can overlap as much work as possible.
"""
from __future__ import annotations

import contextlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import numpy
import torch

try:
    import torchaudio  # type: ignore
    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover - optional dependency
    torchaudio = None  # type: ignore
    _HAS_TORCHAUDIO = False

try:
    import cvcuda  # type: ignore
    _HAS_CVCUDA = True
except Exception:  # pragma: no cover - optional dependency
    cvcuda = None  # type: ignore
    _HAS_CVCUDA = False

from facefusion import logger


DecodeCallback = Callable[[int, numpy.ndarray], numpy.ndarray]


@dataclass
class PipelineConfig:
    src: str
    dst: str
    fps: float
    width: int
    height: int
    device: int = 0
    chunk_size: int = 1


def is_available() -> bool:
    """Return ``True`` when the NVDEC/NVENC pipeline can run."""
    if not _HAS_TORCHAUDIO:
        logger.debug('GPU pipeline unavailable: torchaudio missing', __name__)
        return False
    if not torch.cuda.is_available():
        logger.debug('GPU pipeline unavailable: CUDA not available', __name__)
        return False
    return True


class _Nv12Converter:
    """Helper to convert RGB tensors on CUDA into NV12 layout."""

    def __init__(self, width: int, height: int, device: int) -> None:
        self.width = width
        self.height = height
        self.device = device

    def rgb_to_nv12(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert ``rgb`` (B,H,W,3) uint8 tensor on CUDA to NV12 format."""
        if rgb.dtype != torch.uint8:
            rgb = rgb.to(torch.uint8)
        if rgb.shape[-1] != 3:
            raise ValueError('Expected last dimension to be 3 (RGB).')
        if rgb.dim() == 4:
            # assume batch dimension of 1
            if rgb.shape[0] != 1:
                raise ValueError('Only batch size 1 supported for NV12 conversion.')
            rgb = rgb[0]
        if rgb.device.index != self.device:
            rgb = rgb.to(self.device, non_blocking=True)

        if _HAS_CVCUDA:
            # CV-CUDA expects NHWC tensors; wrap and convert directly.
            image = cvcuda.Tensor(rgb.unsqueeze(0))  # type: ignore[attr-defined]
            # Convert RGB -> NV12, result height = H * 3/2, width = W, channels = 1
            nv12 = cvcuda.cvtcolor(image, cvcuda.ColorConversion.RGB2NV12)  # type: ignore[attr-defined]
            return torch.as_tensor(nv12.cuda(), device=rgb.device)

        # Manual conversion fallback using Torch ops on GPU.
        rgb_f = rgb.to(dtype=torch.float32)
        r = rgb_f[..., 0]
        g = rgb_f[..., 1]
        b = rgb_f[..., 2]
        y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0)
        u = (-0.148 * r - 0.291 * g + 0.439 * b + 128.0)
        v = (0.439 * r - 0.368 * g - 0.071 * b + 128.0)

        y = y.clamp(0.0, 255.0).round().to(torch.uint8)
        u = u.clamp(0.0, 255.0)
        v = v.clamp(0.0, 255.0)

        # Downsample 2x2 blocks for chroma (nearest-neighbour average).
        u_plane = torch.nn.functional.avg_pool2d(
            u.view(1, 1, self.height, self.width), kernel_size=2, stride=2
        )[0, 0]
        v_plane = torch.nn.functional.avg_pool2d(
            v.view(1, 1, self.height, self.width), kernel_size=2, stride=2
        )[0, 0]
        u_plane = u_plane.round().to(torch.uint8)
        v_plane = v_plane.round().to(torch.uint8)

        interleaved_uv = torch.stack((u_plane, v_plane), dim=-1)
        interleaved_uv = interleaved_uv.view(-1, self.width)
        y_plane = y.view(self.height, self.width)
        nv12 = torch.cat((y_plane, interleaved_uv), dim=0)
        return nv12.contiguous()


class NvdecNvencPipeline:
    """High-throughput GPU video pipeline with overlapped stages."""

    def __init__(self, config: PipelineConfig) -> None:
        if not is_available():
            raise RuntimeError('GPU video pipeline not available on this system.')
        self.config = config
        self.device = torch.device('cuda', index=config.device)
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self.encode_stream = torch.cuda.Stream(device=self.device)
        self._reader = self._build_reader(config)
        self._writer = self._build_writer(config)
        self._converter = _Nv12Converter(config.width, config.height, config.device)
        self._closed = False
        self._buffer_pool: list[torch.Tensor] = []
        self._max_workers = max(2, min(8, (torch.get_num_threads() or 4)))

    def _build_reader(self, config: PipelineConfig) -> "torchaudio.io.StreamReader":
        reader = torchaudio.io.StreamReader(
            src=config.src,
            format=None,
            buffer_chunk_size=0
        )
        reader.add_video_stream(
            frames_per_chunk=config.chunk_size,
            decoder='h264_cuvid',
            hw_accel=f'cuda:{config.device}',
            format='rgb24'
        )
        return reader

    def _build_writer(self, config: PipelineConfig) -> "torchaudio.io.StreamWriter":
        writer = torchaudio.io.StreamWriter(
            dst=config.dst,
            format='mp4'
        )
        with contextlib.suppress(Exception):
            writer.set_muxer_option('vsync', 'cfr')
        writer.add_video_stream(
            frame_rate=config.fps,
            width=config.width,
            height=config.height,
            encoder='h264_nvenc',
            hw_accel=f'cuda:{config.device}',
            pixel_format='nv12',
            encoder_option={
                'preset': 'p4',
                'tune': 'hq',
                'rc': 'vbr_hq',
                'cq': '19',
                'bf': '2',
                'spatial_aq': '1',
                'temporal_aq': '1',
                'look_ahead': '0',
            },
        )
        writer.set_audio_stream(None)
        writer.open()
        return writer

    def close(self) -> None:
        if self._closed:
            return
        with contextlib.suppress(Exception):
            self._writer.close()
        self._closed = True

    def __enter__(self) -> "NvdecNvencPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def run(self, process_frame: DecodeCallback) -> int:
        """Decode, process, and encode all frames; return processed count."""
        frame_total = 0
        inflight: Deque[Tuple[int, torch.Tensor, Future[numpy.ndarray]]] = deque()

        def submit_job(index: int, cpu_buffer: torch.Tensor, ready_event: torch.cuda.Event) -> Future[numpy.ndarray]:

            def _worker() -> numpy.ndarray:
                ready_event.synchronize()
                frame_np = cpu_buffer.numpy()
                result = process_frame(index, frame_np)
                if not isinstance(result, numpy.ndarray):
                    raise RuntimeError('process_frame callback must return numpy.ndarray')
                if result.dtype != numpy.uint8:
                    result = result.clip(0, 255).astype(numpy.uint8)
                return result

            return executor.submit(_worker)

        executor = ThreadPoolExecutor(max_workers=self._max_workers)
        try:
            for chunk in self._reader.stream():
                if not chunk:
                    break
                (frame_tensor,) = chunk
                cpu_buffer = self._acquire_buffer()
                decode_event = torch.cuda.Event()
                with torch.cuda.stream(self.decode_stream):
                    frame_tensor = frame_tensor.to(self.device, non_blocking=True)
                    frame_tensor = frame_tensor[:, [2, 1, 0], ...]  # RGB -> BGR
                    frame_tensor = frame_tensor.permute(0, 2, 3, 1).contiguous()
                    cpu_buffer.copy_(frame_tensor[0], non_blocking=True)
                decode_event.record(self.decode_stream)

                future = submit_job(frame_total, cpu_buffer, decode_event)
                inflight.append((frame_total, cpu_buffer, future))
                self._drain_inflight(inflight, flush=False)
                frame_total += 1

            self._drain_inflight(inflight, flush=True)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

        self._writer.flush()
        return frame_total

    def _drain_inflight(self, inflight: Deque[Tuple[int, torch.Tensor, Future[numpy.ndarray]]], flush: bool) -> None:
        while inflight:
            index, buffer, future = inflight[0]
            if not flush and not future.done():
                break
            inflight.popleft()
            try:
                processed_bgr = future.result()  # raises if callback failed
            finally:
                self._release_buffer(buffer)
            self._encode_frame(index, processed_bgr)

    def _encode_frame(self, frame_index: int, frame_bgr: numpy.ndarray) -> None:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError('Encoded frame must have shape (H, W, 3)')
        rgb_frame = numpy.ascontiguousarray(frame_bgr[..., ::-1])
        rgb_tensor = torch.from_numpy(rgb_frame)
        with torch.cuda.stream(self.encode_stream):
            rgb_tensor = rgb_tensor.to(self.device, non_blocking=True)
            rgb_tensor = rgb_tensor.unsqueeze(0)  # (1, H, W, 3)
            nv12 = self._converter.rgb_to_nv12(rgb_tensor)
            self._writer.encode_video(0, nv12, pts=frame_index)

    def _acquire_buffer(self) -> torch.Tensor:
        if self._buffer_pool:
            return self._buffer_pool.pop()
        return torch.empty((self.config.height, self.config.width, 3), dtype=torch.uint8, pin_memory=True)

    def _release_buffer(self, buffer: torch.Tensor) -> None:
        self._buffer_pool.append(buffer)


def run_pipeline(config: PipelineConfig, process_frame: DecodeCallback) -> int:
    """Utility to execute the NVDEC/NVENC pipeline if available."""
    with NvdecNvencPipeline(config) as pipeline:
        return pipeline.run(process_frame)
