import subprocess
from typing import List, Optional, Tuple, Union

import numpy

from facefusion import logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame, get_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.filesystem import filter_audio_paths
from facefusion.processors.core import get_processors_modules
from facefusion import ffmpeg_builder
from facefusion.time_helper import calculate_end_time
from facefusion.types import ErrorCode
from facefusion.vision import conditional_merge_vision_mask, detect_video_resolution, extract_vision_mask, pack_resolution, read_static_image, read_static_images, read_static_video_frame, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, scale_resolution, write_image

try:
	import torch
	from facefusion.gpu_types import (
		get_processing_mode,
		numpy_to_cuda,
		cuda_to_numpy,
		get_cuda_device
	)
	GPU_AVAILABLE = True
except ImportError:
	GPU_AVAILABLE = False


def _get_streaming_video_props() -> Tuple[int, int, float, int, int]:
	target_path = state_manager.get_item('target_path')
	trim_frame_start, trim_frame_end = restrict_trim_frame(target_path, state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	base_res = detect_video_resolution(target_path)
	output_video_scale = state_manager.get_item('output_video_scale')
	output_res = scale_resolution(base_res, output_video_scale)
	temp_res = restrict_video_resolution(target_path, output_res)
	temp_fps = restrict_video_fps(target_path, state_manager.get_item('output_video_fps'))
	logger.info(translator.get('processing').format(resolution = pack_resolution(temp_res), fps = temp_fps), __name__)
	return temp_res[0], temp_res[1], temp_fps, trim_frame_start, trim_frame_end


def _frame_to_seconds(frame_index: int, fps: float) -> Optional[float]:
	if frame_index and fps:
		return frame_index / fps
	return None


def _append_trim_args(cmd: List[str], start_sec: Optional[float], end_sec: Optional[float]) -> None:
	if start_sec is not None:
		cmd.extend([ '-ss', f'{start_sec:.3f}' ])
	if end_sec is not None and end_sec > 0:
		cmd.extend([ '-to', f'{end_sec:.3f}' ])


def _launch_decoder_ffmpeg(input_path: str, width: int, height: int, start_sec: Optional[float], end_sec: Optional[float]) -> subprocess.Popen:
	cmd = [
		'ffmpeg',
		'-hide_banner',
		'-loglevel', 'error',
		'-hwaccel', 'cuda',
		'-hwaccel_output_format', 'cuda',
	]
	_append_trim_args(cmd, start_sec, end_sec)
	cmd += [
		'-i', input_path,
		'-vf', f'scale_npp={width}:{height}:interp_algo=lanczos',
		'-pix_fmt', 'bgr24',
		'-f', 'rawvideo',
		'pipe:1'
	]
	return subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, bufsize = 10 ** 8)


def _launch_encoder_ffmpeg(output_path: str, input_path: str, width: int, height: int, fps: float, start_sec: Optional[float], end_sec: Optional[float]) -> subprocess.Popen:
	video_encoder = state_manager.get_item('output_video_encoder') or 'libx264'
	video_quality = state_manager.get_item('output_video_quality') or 75
	video_preset = state_manager.get_item('output_video_preset') or 'medium'

	# Map CPU encoder choices to NVENC equivalents for streaming while preserving quality intent.
	if video_encoder in [ 'libx264', 'libx264rgb' ]:
		video_encoder = 'h264_nvenc'
	if video_encoder == 'libx265':
		video_encoder = 'hevc_nvenc'

	video_quality_args = ffmpeg_builder.set_video_quality(video_encoder, video_quality)
	video_preset_args = ffmpeg_builder.set_video_preset(video_encoder, video_preset)

	audio_encoder = state_manager.get_item('output_audio_encoder') or 'aac'
	audio_quality = state_manager.get_item('output_audio_quality') or 75
	audio_args = ffmpeg_builder.set_audio_quality(audio_encoder, audio_quality)

	cmd = [
		'ffmpeg',
		'-hide_banner',
		'-loglevel', 'error',
		'-f', 'rawvideo',
		'-pix_fmt', 'bgr24',
		'-s', f'{width}x{height}',
		'-r', str(fps),
		'-i', 'pipe:0',
	]
	_append_trim_args(cmd, start_sec, end_sec)
	cmd += [
		'-i', input_path,
	] + ffmpeg_builder.set_video_encoder(video_encoder) + video_quality_args + video_preset_args + [
		'-map', '0:v:0',
		'-map', '1:a?',
	] + ffmpeg_builder.set_audio_encoder(audio_encoder) + audio_args + [
		'-pix_fmt', 'yuv420p',
		'-y',
		output_path
	]
	return subprocess.Popen(cmd, stdin = subprocess.PIPE, stderr = subprocess.PIPE, bufsize = 10 ** 8)


def _resolve_audio_frames(source_audio_path : Optional[str], temp_video_fps : float, frame_number : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	source_audio_frame = get_audio_frame(source_audio_path, temp_video_fps, frame_number) if source_audio_path else create_empty_audio_frame()
	source_voice_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number) if source_audio_path else create_empty_audio_frame()

	if not numpy.any(source_audio_frame):
		source_audio_frame = create_empty_audio_frame()
	if not numpy.any(source_voice_frame):
		source_voice_frame = create_empty_audio_frame()

	return source_audio_frame, source_voice_frame


def _bgr_to_rgba(frame: numpy.ndarray) -> numpy.ndarray:
	alpha = numpy.full(frame.shape[:2], 255, dtype = frame.dtype)
	return numpy.dstack((frame, alpha))


def _rgba_to_bgr(frame: numpy.ndarray) -> numpy.ndarray:
	return frame[:, :, :3]


def _bgr_to_rgba_cuda(frame) -> 'torch.Tensor':
	"""Convert BGR CUDA tensor to RGBA by adding alpha channel."""
	if not GPU_AVAILABLE:
		raise RuntimeError("GPU mode not available")

	alpha = torch.full((frame.shape[0], frame.shape[1], 1), 255, dtype=frame.dtype, device=frame.device)
	return torch.cat([frame, alpha], dim=-1)


def _rgba_to_bgr_cuda(frame) -> 'torch.Tensor':
	"""Convert RGBA CUDA tensor to BGR by removing alpha channel."""
	return frame[:, :, :3]


def extract_vision_mask_cuda(frame, device: str):
	"""Extract vision mask from RGBA frame on GPU."""
	if not GPU_AVAILABLE:
		raise RuntimeError("GPU mode not available")

	# Check if frame has alpha channel
	if frame.shape[-1] == 4:
		alpha_mask = frame[:, :, 3].float() / 255.0
		return alpha_mask
	else:
		# No alpha, return full mask
		return torch.ones((frame.shape[0], frame.shape[1]), dtype=torch.float32, device=device)


def conditional_merge_vision_mask_cuda(frame, mask):
	"""Merge vision frame with mask on GPU."""
	if not GPU_AVAILABLE:
		raise RuntimeError("GPU mode not available")

	# If frame has alpha channel, update it with mask
	if frame.shape[-1] == 4:
		frame = frame.clone()
		frame[:, :, 3] = (mask * 255.0).byte()
	return frame


def _process_streaming_frame(bgr_frame: numpy.ndarray,
                             frame_number: int,
                             reference_vision_frame: numpy.ndarray,
                             source_vision_frames: List[numpy.ndarray],
                             processor_modules: List,
                             source_audio_path: Optional[str],
                             temp_video_fps: float,
                             device: Optional[str] = None) -> numpy.ndarray:
	"""
	Process a single frame through the processor chain.

	Args:
		bgr_frame: Input BGR frame as NumPy array
		frame_number: Current frame number
		reference_vision_frame: Reference frame
		source_vision_frames: Source faces
		processor_modules: List of processor modules
		source_audio_path: Path to source audio
		temp_video_fps: Video FPS
		device: CUDA device (if GPU mode)

	Returns:
		Processed BGR frame as NumPy array
	"""
	processing_mode = get_processing_mode() if GPU_AVAILABLE else 'cpu'

	if processing_mode == 'gpu' and device is not None:
		# GPU path: convert to CUDA and keep on GPU throughout
		bgr_frame_cuda = numpy_to_cuda(bgr_frame, device=device)
		target_vision_frame = _bgr_to_rgba_cuda(bgr_frame_cuda)
		temp_vision_frame = target_vision_frame.clone()
		temp_vision_mask = extract_vision_mask_cuda(temp_vision_frame, device=device)

		# Convert reference and source frames to CUDA once
		reference_vision_frame_cuda = numpy_to_cuda(reference_vision_frame, device=device)
		source_vision_frames_cuda = [numpy_to_cuda(frame, device=device) for frame in source_vision_frames]

		source_audio_frame, source_voice_frame = _resolve_audio_frames(source_audio_path, temp_video_fps, frame_number)

		# Process with CUDA tensors directly (processors handle CPU conversion internally if needed)
		for processor_module in processor_modules:
			# Pass CUDA tensors directly - processor will detect and use GPU path
			temp_vision_frame, temp_vision_mask = processor_module.process_frame(
			{
				'reference_vision_frame': reference_vision_frame,  # NumPy for face detection
				'source_vision_frames': source_vision_frames,  # NumPy for face detection
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frame': cuda_to_numpy(target_vision_frame[:, :, :3]),  # NumPy for face detection
				'temp_vision_frame': temp_vision_frame[:, :, :3],  # CUDA tensor for processing
				'temp_vision_mask': temp_vision_mask  # CUDA tensor for processing
			})

			# Outputs should already be CUDA tensors from GPU-aware processor
			# No need to convert unless processor falls back to CPU

		temp_vision_frame = conditional_merge_vision_mask_cuda(temp_vision_frame, temp_vision_mask)
		result_bgr = _rgba_to_bgr_cuda(temp_vision_frame)

		# Convert back to NumPy only at the end for encoder
		return cuda_to_numpy(result_bgr)
	else:
		# CPU path (original implementation)
		target_vision_frame = _bgr_to_rgba(bgr_frame)
		temp_vision_frame = target_vision_frame.copy()
		temp_vision_mask = extract_vision_mask(temp_vision_frame)
		source_audio_frame, source_voice_frame = _resolve_audio_frames(source_audio_path, temp_video_fps, frame_number)

		for processor_module in processor_modules:
			temp_vision_frame, temp_vision_mask = processor_module.process_frame(
			{
				'reference_vision_frame': reference_vision_frame,
				'source_vision_frames': source_vision_frames,
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frame': target_vision_frame[:, :, :3],
				'temp_vision_frame': temp_vision_frame[:, :, :3],
				'temp_vision_mask': temp_vision_mask
			})

		temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)
		return _rgba_to_bgr(temp_vision_frame)


def run_streaming_video_job(start_time: float) -> ErrorCode:
	target_path = state_manager.get_item('target_path')
	output_path = state_manager.get_item('output_path')

	width, height, temp_video_fps, trim_frame_start, trim_frame_end = _get_streaming_video_props()
	frame_size = width * height * 3
	start_sec = _frame_to_seconds(trim_frame_start, temp_video_fps)
	end_sec = _frame_to_seconds(trim_frame_end, temp_video_fps) if trim_frame_end else None

	reference_vision_frame = read_static_video_frame(target_path, state_manager.get_item('reference_frame_number'))
	source_paths = state_manager.get_item('source_paths') or []
	source_vision_frames = read_static_images(source_paths)
	source_audio_path = get_first(filter_audio_paths(source_paths))
	processor_modules = list(get_processors_modules(state_manager.get_item('processors')))

	# Determine GPU device if GPU mode is enabled
	processing_mode = get_processing_mode() if GPU_AVAILABLE else 'cpu'
	device = None

	# Debug logging
	gpu_mode_flag = state_manager.get_item('gpu_mode')
	logger.info(f"GPU mode flag from state: {gpu_mode_flag}", __name__)
	logger.info(f"GPU_AVAILABLE: {GPU_AVAILABLE}", __name__)
	logger.info(f"Processing mode: {processing_mode}", __name__)

	if processing_mode == 'gpu':
		device_ids = state_manager.get_item('execution_device_ids') or ['0']
		device = get_cuda_device(str(device_ids[0]))
		logger.info(f"GPU mode enabled, using device: {device}", __name__)

	decoder = _launch_decoder_ffmpeg(target_path, width, height, start_sec, end_sec)
	encoder = _launch_encoder_ffmpeg(output_path, target_path, width, height, temp_video_fps, start_sec, end_sec)

	if decoder.stdout is None or encoder.stdin is None:
		logger.error("Failed to start streaming encoder/decoder.", __name__)
		return 1

	frame_number = 0

	try:
		while True:
			raw = decoder.stdout.read(frame_size)
			if not raw or len(raw) < frame_size:
				break

			bgr_frame = numpy.frombuffer(raw, dtype = numpy.uint8).reshape((height, width, 3))
			processed_bgr = _process_streaming_frame(
				bgr_frame,
				frame_number,
				reference_vision_frame,
				source_vision_frames,
				processor_modules,
				source_audio_path,
				temp_video_fps,
				device=device  # Pass GPU device if GPU mode
			)
			if processed_bgr is None:
				return 1
			encoder.stdin.write(processed_bgr.tobytes())
			frame_number += 1

	except Exception as error:
		logger.error(f"Streaming pipeline failed: {error}", __name__)
		return 1
	finally:
		if decoder.stdout:
			decoder.stdout.close()
		if encoder.stdin:
			encoder.stdin.close()
		decoder.wait()
		encoder.wait()

	logger.info(translator.get('processing_video_succeeded').format(seconds = calculate_end_time(start_time)), __name__)
	return 0
