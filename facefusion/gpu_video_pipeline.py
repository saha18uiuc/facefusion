import subprocess
from functools import lru_cache
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


@lru_cache(maxsize=8)
def _ffmpeg_supports_filter(filter_name: str) -> bool:
	try:
		result = subprocess.run(
			['ffmpeg', '-hide_banner', '-v', 'quiet', '-filters'],
			check=True,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True
		)
		return filter_name in result.stdout
	except Exception:
		return False


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
	# Try CUDA hardware decode first, but don't require hwaccel_output_format
	# This allows fallback to software decode if CUDA isn't available
	use_hwscale = state_manager.get_item('prefer_hwscale')
	use_hwscale = True if use_hwscale is None else bool(use_hwscale)

	# Pick best available scale filter: prefer scale_npp > scale_cuda > scale
	if use_hwscale:
		if _ffmpeg_supports_filter('scale_npp'):
			scale_filter = 'scale_npp'
		elif _ffmpeg_supports_filter('scale_cuda'):
			# scale_cuda can be finicky with formats; add explicit format bridge when we use it
			scale_filter = 'scale_cuda'
		else:
			use_hwscale = False
			scale_filter = 'scale'
	else:
		scale_filter = 'scale'

	cmd = [
		'ffmpeg',
		'-hide_banner',
		'-loglevel', 'warning',  # Changed from 'error' to see warnings
		'-hwaccel', 'cuda',
		'-hwaccel_device', '0',
	]

	# Prefer hardware scaling to stay on GPU; fall back to CPU scaling when disabled
	if use_hwscale:
		cmd += [ '-hwaccel_output_format', 'cuda' ]

	_append_trim_args(cmd, start_sec, end_sec)

	# Build filter chain; handle scale_cuda/scale_npp format expectations
	filter_chain: List[str] = []
	if scale_filter in ('scale_cuda', 'scale_npp'):
		filter_chain.append(f"{scale_filter}={width}:{height}")
		# Ensure output is in a CUDA-friendly pixel format; nv12 is broadly supported
		filter_chain.append('format=nv12')
	else:
		filter_chain.append(f"{scale_filter}={width}:{height}:flags=lanczos")
	filter_chain.append('format=bgr24')

	cmd += [
		'-i', input_path,
		'-vf', ','.join(filter_chain),
		'-pix_fmt', 'bgr24',
		'-f', 'rawvideo',
		'pipe:1'
	]

	logger.info(f"Decoder command: {' '.join(cmd)}", __name__)
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
		try:
			if frame_number == 0:
				print(f"[DEBUG] GPU path activated for frame {frame_number}")

			# GPU path: convert to CUDA and keep on GPU throughout
			if not bgr_frame.flags.writeable:
				bgr_frame = bgr_frame.copy()
			bgr_frame_cuda = numpy_to_cuda(bgr_frame, device=device)

			# Ensure uint8 dtype
			if bgr_frame_cuda.dtype != torch.uint8:
				bgr_frame_cuda = bgr_frame_cuda.byte()

			target_vision_frame = _bgr_to_rgba_cuda(bgr_frame_cuda)
			temp_vision_frame = target_vision_frame[:, :, :3].clone()  # BGR only for processing
			temp_vision_mask = extract_vision_mask_cuda(target_vision_frame, device=device)

			source_audio_frame, source_voice_frame = _resolve_audio_frames(source_audio_path, temp_video_fps, frame_number)

			# Process with CUDA tensors directly
			if frame_number == 0:
				print(f"[DEBUG] Processing through {len(processor_modules)} processor modules")

			for idx, processor_module in enumerate(processor_modules):
				if frame_number == 0:
					print(f"[DEBUG] Calling processor {idx}: {type(processor_module).__name__}")

				temp_vision_frame, temp_vision_mask = processor_module.process_frame(
				{
					'reference_vision_frame': reference_vision_frame,  # NumPy for face detection
					'source_vision_frames': source_vision_frames,  # NumPy for face detection
					'source_audio_frame': source_audio_frame,
					'source_voice_frame': source_voice_frame,
					'target_vision_frame': cuda_to_numpy(target_vision_frame[:, :, :3]),  # NumPy for face detection
					'temp_vision_frame': temp_vision_frame,  # CUDA tensor BGR (3 channels)
					'temp_vision_mask': temp_vision_mask  # CUDA tensor mask
				})

				if frame_number == 0:
					print(f"[DEBUG] Processor {idx} returned frame shape: {temp_vision_frame.shape}, mask shape: {temp_vision_mask.shape}")

				# Normalize outputs to CUDA tensors if a processor returned NumPy (happens when GPU path falls back internally)
				if isinstance(temp_vision_frame, numpy.ndarray):
					temp_vision_frame = numpy_to_cuda(temp_vision_frame, device=device)
				if isinstance(temp_vision_mask, numpy.ndarray):
					temp_vision_mask = numpy_to_cuda(temp_vision_mask, device=device).float() / 255.0

				# Processor returns BGR (3 channels), need to add alpha back for merging
				if temp_vision_frame.shape[-1] == 3:
					# Convert BGR back to RGBA by adding mask as alpha channel
					if isinstance(temp_vision_mask, torch.Tensor):
						alpha_channel = (temp_vision_mask * 255.0).byte().unsqueeze(-1) if temp_vision_mask.ndim == 2 else (temp_vision_mask * 255.0).byte()
					else:
						alpha_np = (temp_vision_mask * 255.0).astype(numpy.uint8)
						alpha_channel = numpy_to_cuda(alpha_np, device=device).unsqueeze(-1)
					temp_vision_frame = torch.cat([temp_vision_frame, alpha_channel], dim=-1)

				# Now merge mask (updates alpha channel)
				temp_vision_frame = conditional_merge_vision_mask_cuda(temp_vision_frame, temp_vision_mask if isinstance(temp_vision_mask, torch.Tensor) else numpy_to_cuda(temp_vision_mask, device=device))

				# Extract BGR for encoder
				result_bgr = temp_vision_frame[:, :, :3]

				# Ensure uint8 dtype
				if result_bgr.dtype != torch.uint8:
					result_bgr = torch.clamp(result_bgr, 0, 255).byte()

				# Validate dimensions
				if result_bgr.ndim != 3 or result_bgr.shape[-1] != 3:
					raise ValueError(f"Invalid result dimensions: {result_bgr.shape}, expected (H, W, 3)")

				# Convert back to NumPy for encoder
				result_np = cuda_to_numpy(result_bgr)

				# Final validation
				if result_np.dtype != numpy.uint8:
					result_np = numpy.clip(result_np, 0, 255).astype(numpy.uint8)

				if result_np.shape != (bgr_frame.shape[0], bgr_frame.shape[1], 3):
					raise ValueError(f"Result shape {result_np.shape} doesn't match input shape {bgr_frame.shape}")

				return result_np

		except Exception as e:
			logger.error(f"GPU frame processing failed: {e}, falling back to CPU for this frame", __name__)
			import traceback
			traceback.print_exc()
			# Fall through to CPU path
	# CPU path (original implementation or fallback from GPU error)
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
	result_bgr = _rgba_to_bgr(temp_vision_frame)

	# Ensure correct dtype
	if result_bgr.dtype != numpy.uint8:
		result_bgr = numpy.clip(result_bgr, 0, 255).astype(numpy.uint8)

	return result_bgr


def run_streaming_video_job(start_time: float) -> ErrorCode:
	target_path = state_manager.get_item('target_path')
	output_path = state_manager.get_item('output_path')

	width, height, temp_video_fps, trim_frame_start, trim_frame_end = _get_streaming_video_props()
	frame_size = width * height * 3
	start_sec = _frame_to_seconds(trim_frame_start, temp_video_fps)
	end_sec = _frame_to_seconds(trim_frame_end, temp_video_fps) if trim_frame_end else None

	processor_names = state_manager.get_item('processors') or []
	reference_vision_frame = read_static_video_frame(target_path, state_manager.get_item('reference_frame_number'))
	source_paths = state_manager.get_item('source_paths') or []
	source_vision_frames = read_static_images(source_paths)
	source_audio_path = get_first(filter_audio_paths(source_paths))
	processor_modules = list(get_processors_modules(processor_names))

	if not processor_modules:
		logger.error("No processor modules configured; cannot process frames. Set --processors (e.g. face_swapper).", __name__)
		return 1

	if not source_vision_frames and ('face_swapper' in processor_names or 'deep_swapper' in processor_names):
		logger.error("No source images loaded. Check --source-paths for your face swap inputs.", __name__)
		return 1

	if reference_vision_frame is None:
		logger.error("Failed to load reference frame from target video; cannot proceed.", __name__)
		return 1

	# Determine GPU device if GPU mode is enabled
	processing_mode = get_processing_mode() if GPU_AVAILABLE else 'cpu'
	device = None

	# Debug logging
	gpu_mode_flag = state_manager.get_item('gpu_mode')
	logger.info(f"GPU mode flag from state: {gpu_mode_flag}", __name__)
	logger.info(f"GPU_AVAILABLE: {GPU_AVAILABLE}", __name__)
	logger.info(f"Processing mode: {processing_mode}", __name__)
	logger.info(f"Number of processor modules loaded: {len(processor_modules)}", __name__)
	logger.info(f"Processor modules: {[p.__name__ for p in processor_modules]}", __name__)
	logger.info(f"Source paths: {source_paths}", __name__)
	logger.info(f"Source vision frames count: {len(source_vision_frames)}", __name__)

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
				# Check if decoder had errors
				if decoder.stderr:
					stderr_output = decoder.stderr.read().decode('utf-8', errors='ignore')
					if stderr_output:
						logger.error(f"Decoder stderr: {stderr_output}", __name__)
				logger.info(f"Decoder finished after {frame_number} frames", __name__)
				break

			bgr_frame = numpy.frombuffer(raw, dtype = numpy.uint8).reshape((height, width, 3))

			if frame_number == 0:
				logger.info(f"Processing first frame - shape: {bgr_frame.shape}, dtype: {bgr_frame.dtype}", __name__)

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

			if frame_number == 0:
				logger.info(f"Processed first frame - shape: {processed_bgr.shape if processed_bgr is not None else 'None'}", __name__)

			if processed_bgr is None:
				logger.error(f"Frame {frame_number} processing returned None!", __name__)
				return 1
			encoder.stdin.write(processed_bgr.tobytes())
			frame_number += 1

			if frame_number % 30 == 0:
				logger.info(f"Processed {frame_number} frames...", __name__)

	except Exception as error:
		logger.error(f"Streaming pipeline failed: {error}", __name__)
		import traceback
		traceback.print_exc()
		return 1
	finally:
		if decoder.stdout:
			decoder.stdout.close()
		if decoder.stderr:
			stderr_output = decoder.stderr.read().decode('utf-8', errors='ignore')
			if stderr_output:
				logger.warning(f"Decoder final stderr: {stderr_output}", __name__)
			decoder.stderr.close()
		if encoder.stdin:
			encoder.stdin.close()
		if encoder.stderr:
			stderr_output = encoder.stderr.read().decode('utf-8', errors='ignore')
			if stderr_output:
				logger.warning(f"Encoder stderr: {stderr_output}", __name__)
			encoder.stderr.close()
		decoder.wait()
		encoder.wait()

	if frame_number == 0:
		logger.error("Streaming pipeline processed 0 frames; failing to avoid empty output.", __name__)
		return 1

	logger.info(f"Total frames processed: {frame_number}", __name__)
	logger.info(translator.get('processing_video_succeeded').format(seconds = calculate_end_time(start_time)), __name__)
	return 0
