from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Optional, Tuple

import numpy
from tqdm import tqdm

from facefusion import ffmpeg
from facefusion import logger, process_manager, state_manager, translator, video_manager
from facefusion.audio import create_empty_audio_frame, get_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_video
from facefusion.filesystem import filter_audio_paths, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, move_temp_file, resolve_temp_frame_paths
from facefusion.time_helper import calculate_end_time
from facefusion.types import ErrorCode
from facefusion.vision import conditional_merge_vision_mask, detect_video_resolution, extract_vision_mask, pack_resolution, read_static_image, read_static_images, read_static_video_frame, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, scale_resolution, write_image
from facefusion.workflows.core import is_process_stopping

# Simple per-job cache for decoded reference and source frames (enabled only for targeted models).
_FRAME_CACHE_TARGET_PATH: Optional[str] = None
_FRAME_CACHE_REFERENCE_NUMBER: Optional[int] = None
_FRAME_CACHE_REFERENCE_FRAME: Optional[numpy.ndarray] = None
_FRAME_CACHE_SOURCE_PATHS: Tuple[str, ...] = ()
_FRAME_CACHE_SOURCE_FRAMES: List[numpy.ndarray] = []
_CACHE_ENABLED: bool = False
_RUNTIME_SOURCE_AUDIO_PATH: Optional[str] = None
_RUNTIME_TEMP_VIDEO_FPS: Optional[float] = None


def process(start_time : float) -> ErrorCode:
	tasks =\
	[
		setup,
		extract_frames,
		process_video,
		merge_frames,
		restore_audio,
		partial(finalize_video, start_time)
	]
	process_manager.start()

	for task in tasks:
		error_code = task() # type:ignore[operator]

		if error_code > 0:
			process_manager.end()
			return error_code

	process_manager.end()
	return 0


def setup() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if analyse_video(state_manager.get_item('target_path'), trim_frame_start, trim_frame_end):
		return 3

	logger.debug(translator.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))
	logger.debug(translator.get('creating_temp'), __name__)
	create_temp_directory(state_manager.get_item('target_path'))
	return 0


def extract_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_resolution = restrict_video_resolution(state_manager.get_item('target_path'), output_video_resolution)
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	logger.info(translator.get('extracting_frames').format(resolution=pack_resolution(temp_video_resolution), fps=temp_video_fps), __name__)

	if ffmpeg.extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps, trim_frame_start, trim_frame_end):
		logger.debug(translator.get('extracting_frames_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('extracting_frames_failed'), __name__)
		return 1
	return 0


def process_video() -> ErrorCode:
	_prepare_runtime_cache()
	temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))

	if temp_frame_paths:
		with tqdm(total = len(temp_frame_paths), desc = translator.get('processing'), unit = 'frame', ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
			progress.set_postfix(execution_providers = state_manager.get_item('execution_providers'))

			with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
				futures = []

				for frame_number, temp_frame_path in enumerate(temp_frame_paths):
					future = executor.submit(process_temp_frame, temp_frame_path, frame_number)
					futures.append(future)

				for future in as_completed(futures):
					if is_process_stopping():
						for __future__ in futures:
							__future__.cancel()

					if not future.cancelled():
						future.result()
						progress.update()

		for processor_module in get_processors_modules(state_manager.get_item('processors')):
			processor_module.post_process()
		if is_process_stopping():
			return 4
	else:
		logger.error(translator.get('temp_frames_not_found'), __name__)
		return 1
	return 0


def merge_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))

	logger.info(translator.get('merging_video').format(resolution = pack_resolution(output_video_resolution), fps = state_manager.get_item('output_video_fps')), __name__)
	if ffmpeg.merge_video(state_manager.get_item('target_path'), temp_video_fps, output_video_resolution, state_manager.get_item('output_video_fps'), trim_frame_start, trim_frame_end):
		logger.debug(translator.get('merging_video_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('merging_video_failed'), __name__)
		return 1
	return 0


def restore_audio() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if state_manager.get_item('output_audio_volume') == 0:
		logger.info(translator.get('skipping_audio'), __name__)
		move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	else:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		if source_audio_path:
			if ffmpeg.replace_audio(state_manager.get_item('target_path'), source_audio_path, state_manager.get_item('output_path')):
				video_manager.clear_video_pool()
				logger.debug(translator.get('replacing_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('replacing_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
		else:
			if ffmpeg.restore_audio(state_manager.get_item('target_path'), state_manager.get_item('output_path'), trim_frame_start, trim_frame_end):
				video_manager.clear_video_pool()
				logger.debug(translator.get('restoring_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('restoring_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	return 0


def finalize_video(start_time : float) -> ErrorCode:
	logger.debug(translator.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))

	if is_video(state_manager.get_item('output_path')):
		logger.info(translator.get('processing_video_succeeded').format(seconds = calculate_end_time(start_time)), __name__)
	else:
		logger.error(translator.get('processing_video_failed'), __name__)
		return 1
	return 0


def _prepare_frame_cache() -> None:
	"""
	Decode reference/source frames once per job and reuse across threads.
	"""
	global _FRAME_CACHE_TARGET_PATH, _FRAME_CACHE_REFERENCE_NUMBER, _FRAME_CACHE_REFERENCE_FRAME
	global _FRAME_CACHE_SOURCE_PATHS, _FRAME_CACHE_SOURCE_FRAMES

	target_path = state_manager.get_item('target_path')
	reference_number = state_manager.get_item('reference_frame_number')
	source_paths: Tuple[str, ...] = tuple(state_manager.get_item('source_paths') or [])

	if target_path != _FRAME_CACHE_TARGET_PATH or reference_number != _FRAME_CACHE_REFERENCE_NUMBER:
		_FRAME_CACHE_REFERENCE_FRAME = read_static_video_frame(target_path, reference_number)
		_FRAME_CACHE_TARGET_PATH = target_path
		_FRAME_CACHE_REFERENCE_NUMBER = reference_number

	if source_paths != _FRAME_CACHE_SOURCE_PATHS:
		_FRAME_CACHE_SOURCE_FRAMES = read_static_images(list(source_paths))
		_FRAME_CACHE_SOURCE_PATHS = source_paths


def _prepare_runtime_cache() -> None:
	"""
	Prepare all per-job reusable state to minimize per-frame overhead.
	"""
	global _RUNTIME_SOURCE_AUDIO_PATH, _RUNTIME_TEMP_VIDEO_FPS
	global _FRAME_CACHE_TARGET_PATH, _FRAME_CACHE_REFERENCE_NUMBER, _CACHE_ENABLED

	_FRAME_CACHE_TARGET_PATH = state_manager.get_item('target_path')
	_FRAME_CACHE_REFERENCE_NUMBER = state_manager.get_item('reference_frame_number')

	_CACHE_ENABLED = state_manager.get_item('face_swapper_model') == 'hyperswap_1c_256'

	if _CACHE_ENABLED:
		_prepare_frame_cache()

	source_paths = state_manager.get_item('source_paths')
	_RUNTIME_SOURCE_AUDIO_PATH = get_first(filter_audio_paths(source_paths))
	_RUNTIME_TEMP_VIDEO_FPS = restrict_video_fps(_FRAME_CACHE_TARGET_PATH, state_manager.get_item('output_video_fps'))


def _resolve_audio_frames(source_audio_path : Optional[str], temp_video_fps : float, frame_number : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	"""
	Load (or create empty) audio and voice frames for the current frame index.
	"""
	source_audio_frame = get_audio_frame(source_audio_path, temp_video_fps, frame_number) if source_audio_path else create_empty_audio_frame()
	source_voice_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number) if source_audio_path else create_empty_audio_frame()

	if not numpy.any(source_audio_frame):
		source_audio_frame = create_empty_audio_frame()
	if not numpy.any(source_voice_frame):
		source_voice_frame = create_empty_audio_frame()

	return source_audio_frame, source_voice_frame


def process_frame_runtime(target_vision_frame : numpy.ndarray,
                          frame_number : int,
                          reference_vision_frame : numpy.ndarray,
                          source_vision_frames : List[numpy.ndarray],
                          source_audio_path : Optional[str],
                          temp_video_fps : float) -> numpy.ndarray:
	"""
	Per-frame logic that works on already-decoded inputs; returns processed frame.
	"""
	temp_vision_frame = target_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)
	source_audio_frame, source_voice_frame = _resolve_audio_frames(source_audio_path, temp_video_fps, frame_number)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
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

	return conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)


def process_temp_frame(temp_frame_path : str, frame_number : int) -> bool:
	"""
	Per-frame worker:
	- Read target RGBA frame.
	- Use cached reference/source frames if enabled.
	- Run processors.
	- Write back result.
	"""
	if _CACHE_ENABLED:
		reference_vision_frame = _FRAME_CACHE_REFERENCE_FRAME
		source_vision_frames = _FRAME_CACHE_SOURCE_FRAMES
		target_vision_frame = read_static_image(temp_frame_path, 'rgba')

		if target_vision_frame is None or reference_vision_frame is None:
			return False

		processed_frame = process_frame_runtime(
			target_vision_frame,
			frame_number,
			reference_vision_frame,
			source_vision_frames,
			_RUNTIME_SOURCE_AUDIO_PATH,
			_RUNTIME_TEMP_VIDEO_FPS
		)
		return write_image(temp_frame_path, processed_frame)

	# Fallback: baseline per-frame loading for non-cached models.
	target_path = state_manager.get_item('target_path')
	reference_frame_number = state_manager.get_item('reference_frame_number')
	reference_vision_frame = read_static_video_frame(target_path, reference_frame_number)
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	temp_video_fps = restrict_video_fps(target_path, state_manager.get_item('output_video_fps'))
	target_vision_frame = read_static_image(temp_frame_path, 'rgba')

	if target_vision_frame is None or reference_vision_frame is None:
		return False

	processed_frame = process_frame_runtime(
		target_vision_frame,
		frame_number,
		reference_vision_frame,
		source_vision_frames,
		source_audio_path,
		temp_video_fps
	)
	return write_image(temp_frame_path, processed_frame)
