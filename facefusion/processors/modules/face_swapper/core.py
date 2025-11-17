from argparse import ArgumentParser
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy

import facefusion.choices
import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, face_tracker, inference_manager, logger, state_manager, translator, video_manager
from facefusion.common_helper import get_first, is_macos
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_average_face, get_many_faces, get_one_face, scale_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_area_mask, create_box_mask, create_occlusion_mask, create_region_mask
from facefusion.face_selector import select_faces, sort_faces_by_order
from facefusion.filesystem import filter_image_paths, has_image, in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.model_helper import get_static_model_initializer
from facefusion.processors.modules.face_swapper import choices as face_swapper_choices
from facefusion.processors.modules.face_swapper.types import FaceSwapperInputs
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadScope, Embedding, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_images, read_static_video_frame, unpack_resolution
from facefusion.gpu import gpu_accel_enabled, gpu_compositor_enabled, current_cuda_device

# Performance optimization modules
from facefusion.processors.temporal_smoother import TemporalSmoother
from facefusion.processors.enhanced_blending import EnhancedBlender

# Global instances for temporal smoothing and enhanced blending
_temporal_smoother = None
_enhanced_blender = None

# CUDA memory management
try:
	import torch
	import gc
	_torch_available = True
except ImportError:
	_torch_available = False
try:
	import onnxruntime as ort  # type: ignore
	_ort_available = True
except Exception:
	_ort_available = False

# Processing statistics
_frame_count = 0
_batch_cleanup_interval = 10
_cuda_mean_std_cache = {}
_iobinding_cache = {}

_swap_faulted = False


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'blendswap_256':
		{
			'__metadata__':
			{
				'vendor': 'mapooon',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
				}
			},
			'type': 'blendswap',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'ghost_1_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_2_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_3_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hififace_unofficial_256':
		{
			'__metadata__':
			{
				'vendor': 'GuijiAI',
				'license': 'Unknown',
				'year': 2021
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.hash'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.onnx')
				}
			},
			'type': 'hififace',
			'template': 'mtcnn_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1a_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1b_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.hash')
				}
			},
			'sources':
				{
					'face_swapper':
					{
						'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.onnx'),
						'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.onnx')
					}
				},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1c_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'inswapper_128':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'inswapper_128_fp16':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'simswap_256':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.485, 0.456, 0.406 ],
			'standard_deviation': [ 0.229, 0.224, 0.225 ]
		},
		'simswap_unofficial_512':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (512, 512),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'uniface_256':
		{
			'__metadata__':
			{
				'vendor': 'xc-csc101',
				'license': 'Unknown',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.hash'),
					'path': resolve_relative_path('../.assets/models/uniface_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.onnx'),
					'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
				}
			},
			'type': 'uniface',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ get_model_name() ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ get_model_name() ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_temporal_smoother() -> TemporalSmoother:
	"""Get or create temporal smoother instance."""
	global _temporal_smoother
	if _temporal_smoother is None:
		_temporal_smoother = TemporalSmoother(
			window_size=5,
			alpha=0.25,
			enable_adaptive=True
		)
	return _temporal_smoother


def get_enhanced_blender() -> EnhancedBlender:
	"""Get or create enhanced blender instance."""
	global _enhanced_blender
	if _enhanced_blender is None:
		_enhanced_blender = EnhancedBlender(
			method='multiband',
			enable_color_correction=True,
			pyramid_levels=5,
			feather_amount=15
		)
	return _enhanced_blender


def reset_temporal_smoother() -> None:
	"""Reset temporal smoother (call at scene cuts)."""
	global _temporal_smoother, _frame_count
	if _temporal_smoother is not None:
		_temporal_smoother.reset()
	_frame_count = 0


def cleanup_cuda_memory(aggressive: bool = False) -> None:
	"""
	Clean up CUDA memory to prevent OOM.

	Args:
		aggressive: If True, performs more thorough cleanup
		"""
	if not _torch_available or state_manager.get_item('abort_cuda'):
		return

	if torch.cuda.is_available():
		# Guard CUDA maintenance; device may already be faulted after an illegal access
		try:
			if aggressive:
				try:
					torch.cuda.synchronize()
				except Exception:
					# Ignore sync failures when context is bad
					pass
			# Empty CUDA cache; ignore if context is invalid
			torch.cuda.empty_cache()
			if aggressive:
				gc.collect()
				try:
					torch.cuda.empty_cache()
				except Exception:
					pass
		except Exception as exc:
			logger.warning(f"CUDA cleanup skipped due to error: {exc}", __name__)


def get_optimal_batch_size() -> int:
	"""
	Determine optimal batch size based on available VRAM.

	Returns:
		Recommended batch size for face swapping
	"""
	if not _torch_available or not torch.cuda.is_available():
		return 1  # No batching for CPU

	try:
		device = torch.cuda.current_device()
		props = torch.cuda.get_device_properties(device)
		vram_gb = props.total_memory / (1024 ** 3)
		compute_capability = props.major + (props.minor / 10)

		# Base batch size on VRAM (conservative for face swapping)
		if vram_gb >= 24:
			base_batch = 64
		elif vram_gb >= 16:
			base_batch = 48
		elif vram_gb >= 12:
			base_batch = 32
		elif vram_gb >= 8:
			base_batch = 24
		else:
			base_batch = 16

		# Adjust for compute capability
		if compute_capability >= 8.0:  # Ampere or newer
			batch_multiplier = 1.2
		elif compute_capability >= 7.5:  # Turing
			batch_multiplier = 1.1
		else:
			batch_multiplier = 1.0

		optimal_batch = int(base_batch * batch_multiplier)
		return max(1, optimal_batch)

	except Exception:
		return 1  # Safe fallback


def get_model_options() -> ModelOptions:
	model_name = get_model_name()
	return create_static_model_set('full').get(model_name)


def get_model_name() -> str:
	model_name = state_manager.get_item('face_swapper_model')

	if is_macos() and has_execution_provider('coreml') and model_name == 'inswapper_128_fp16':
		return 'inswapper_128'
	return model_name


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--face-swapper-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'face_swapper_model', 'hyperswap_1a_256'), choices = face_swapper_choices.face_swapper_models)
		known_args, _ = program.parse_known_args()
		face_swapper_pixel_boost_choices = face_swapper_choices.face_swapper_set.get(known_args.face_swapper_model)
		group_processors.add_argument('--face-swapper-pixel-boost', help = translator.get('help.pixel_boost', __package__), default = config.get_str_value('processors', 'face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices)), choices = face_swapper_pixel_boost_choices)
		group_processors.add_argument('--face-swapper-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'face_swapper_weight', '0.5'), choices = face_swapper_choices.face_swapper_weight_range)
		facefusion.jobs.job_store.register_step_keys([ 'face_swapper_model', 'face_swapper_pixel_boost', 'face_swapper_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('face_swapper_model', args.get('face_swapper_model'))
	apply_state_item('face_swapper_pixel_boost', args.get('face_swapper_pixel_boost'))
	apply_state_item('face_swapper_weight', args.get('face_swapper_weight'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if not has_image(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_image_source') + translator.get('exclamation_mark'), __name__)
		return False

	source_image_paths = filter_image_paths(state_manager.get_item('source_paths'))
	source_frames = read_static_images(source_image_paths)
	source_faces = get_many_faces(source_frames)

	if not get_one_face(source_faces):
		logger.error(translator.get('no_source_face_detected') + translator.get('exclamation_mark'), __name__)
		return False

	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False

	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False

	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False

	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	reset_temporal_smoother()  # Reset temporal smoother after processing
	# Reset tracker state to avoid cross-job bleed
	try:
		face_tracker.get_tracker().reset()
	except Exception:
		pass

	# Aggressive CUDA memory cleanup after processing
	cleanup_cuda_memory(aggressive=True)

	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		get_static_model_initializer.cache_clear()
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()

	# Final CUDA cleanup
	cleanup_cuda_memory(aggressive=True)


def swap_face(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))
	pixel_boost_total = pixel_boost_size[0] // model_size[0]
	use_gpu_comp = gpu_compositor_enabled()

	# Apply temporal smoothing to target face landmarks to reduce jitter
	temporal_smoother = get_temporal_smoother()
	target_landmarks_5 = target_face.landmark_set.get('5/68')

	# Check for scene cuts and reset if needed
	if temporal_smoother.should_reset(target_landmarks_5):
		temporal_smoother.reset()

	# Smooth the landmarks
	smoothed_landmarks_5 = temporal_smoother.smooth_landmarks(target_landmarks_5)

	# Use smoothed landmarks for warping
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, smoothed_landmarks_5, model_template, pixel_boost_size)
	temp_vision_frames = []
	crop_masks = []

	if 'box' in state_manager.get_item('face_mask_types'):
		box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
		crop_masks.append(box_mask)

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
	for pixel_boost_vision_frame in pixel_boost_vision_frames:
		# Prefer CUDA preprocessing when available and safe (no pixel boost tiling).
		prepared_gpu = None
		if pixel_boost_total == 1:
			prepared_gpu = prepare_crop_frame_cuda(pixel_boost_vision_frame)
		if prepared_gpu is not None:
			pixel_boost_vision_frame = prepared_gpu
		else:
			pixel_boost_vision_frame = prepare_crop_frame(pixel_boost_vision_frame)
		pixel_boost_vision_frame = forward_swap_face(source_face, target_face, pixel_boost_vision_frame)
		pixel_boost_vision_frame = normalize_crop_frame(pixel_boost_vision_frame)
		temp_vision_frames.append(pixel_boost_vision_frame)
	crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

	if 'area' in state_manager.get_item('face_mask_types'):
		# Use original target landmarks for area mask (not smoothed)
		face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		area_mask = create_area_mask(crop_vision_frame, face_landmark_68, state_manager.get_item('face_mask_areas'))
		crop_masks.append(area_mask)

	if 'region' in state_manager.get_item('face_mask_types'):
		region_mask = create_region_mask(crop_vision_frame, state_manager.get_item('face_mask_regions'))
		crop_masks.append(region_mask)

	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)

	# Standard paste_back for base result, with optional GPU compositor.
	paste_vision_frame = None
	if use_gpu_comp:
		try:
			from facefusion.gpu.compositor import paste_back_cuda
			paste_vision_frame = paste_back_cuda(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
		except Exception:
			paste_vision_frame = None
	if paste_vision_frame is None:
		paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

	# Apply enhanced blending for improved quality and color consistency
	# This addresses color warping and boundary artifacts
	enhanced_blender = get_enhanced_blender()

	# Create a blended mask from crop_mask for enhanced blending
	# We need to warp the crop_mask back to original frame coordinates
	mask_h, mask_w = crop_mask.shape[:2]
	if len(crop_mask.shape) == 2:
		mask_3ch = numpy.stack([crop_mask] * 3, axis=-1)
	else:
		mask_3ch = crop_mask

	# For enhanced blending, we'll apply color correction to the entire pasted result
	# This helps match colors between the swapped face and surrounding areas
	try:
		# Get face center for blending
		if target_face.bounding_box is not None:
			bbox = target_face.bounding_box
			center_x = int((bbox[0] + bbox[2]) / 2)
			center_y = int((bbox[1] + bbox[3]) / 2)
			center = (center_x, center_y)

			# Extract face region for enhanced blending
			x1, y1, x2, y2 = map(int, bbox)
			# Expand bbox slightly for better blending
			margin = 20
			x1 = max(0, x1 - margin)
			y1 = max(0, y1 - margin)
			x2 = min(temp_vision_frame.shape[1], x2 + margin)
			y2 = min(temp_vision_frame.shape[0], y2 + margin)

			# Extract regions
			swapped_region = paste_vision_frame[y1:y2, x1:x2]
			original_region = temp_vision_frame[y1:y2, x1:x2]

			# Create mask for this region
			region_h = y2 - y1
			region_w = x2 - x1
			region_mask = numpy.ones((region_h, region_w, 3), dtype=numpy.float32) * 0.8

			# Apply enhanced blending to the region
			blended_region = enhanced_blender._color_correction(
				swapped_region,
				original_region,
				region_mask
			)

			# Blend the corrected region back
			alpha = 0.7  # Blend factor
			paste_vision_frame[y1:y2, x1:x2] = (
				blended_region * alpha + paste_vision_frame[y1:y2, x1:x2] * (1 - alpha)
			).astype(numpy.uint8)

	except Exception:
		# If enhanced blending fails, use standard result
		pass

	return paste_vision_frame


def forward_swap_face(source_face : Face, target_face : Face, crop_vision_frame : VisionFrame) -> VisionFrame:
	face_swapper = get_inference_pool().get('face_swapper')
	model_type = get_model_options().get('type')
	face_swapper_inputs = {}

	if is_macos() and has_execution_provider('coreml') and model_type in [ 'ghost', 'uniface' ]:
		face_swapper.set_providers([ facefusion.choices.execution_provider_set.get('cpu') ])

	for face_swapper_input in face_swapper.get_inputs():
		if face_swapper_input.name == 'source':
			if model_type in [ 'blendswap', 'uniface' ]:
				face_swapper_inputs[face_swapper_input.name] = prepare_source_frame(source_face)
			else:
				source_embedding = prepare_source_embedding(source_face)
				source_embedding = balance_source_embedding(source_embedding, target_face.embedding)
				face_swapper_inputs[face_swapper_input.name] = source_embedding
		if face_swapper_input.name == 'target':
			face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

	with conditional_thread_semaphore():
		# Prefer CUDA IOBinding when available to avoid host<->device churn.
		if _use_iobinding():
			result = _run_faceswapper_iobinding(face_swapper, face_swapper_inputs)
			if result is not None:
				return result
		# Fallback uses numpy; convert torch inputs if needed.
		run_inputs = {}
		for k, v in face_swapper_inputs.items():
			if isinstance(v, torch.Tensor):
				run_inputs[k] = v.detach().cpu().numpy()
			else:
				run_inputs[k] = v
		crop_vision_frame = face_swapper.run(None, run_inputs)[0][0]

	return crop_vision_frame


def forward_convert_embedding(face_embedding : Embedding) -> Embedding:
	embedding_converter = get_inference_pool().get('embedding_converter')

	with conditional_thread_semaphore():
		face_embedding = embedding_converter.run(None,
		{
			'input': face_embedding
		})[0]

	return face_embedding


def prepare_source_frame(source_face : Face) -> VisionFrame:
	model_type = get_model_options().get('type')
	source_vision_frame = read_static_image(get_first(state_manager.get_item('source_paths')))

	if model_type == 'blendswap':
		source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmark_set.get('5/68'), 'arcface_112_v2', (112, 112))

	if model_type == 'uniface':
		source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmark_set.get('5/68'), 'ffhq_512', (256, 256))

	source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
	source_vision_frame = source_vision_frame.transpose(2, 0, 1)
	source_vision_frame = numpy.expand_dims(source_vision_frame, axis = 0).astype(numpy.float32)
	return source_vision_frame


def prepare_source_embedding(source_face : Face) -> Embedding:
	model_type = get_model_options().get('type')

	if model_type == 'ghost':
		source_embedding = source_face.embedding.reshape(-1, 512)
		source_embedding, _ = convert_source_embedding(source_embedding)
		source_embedding = source_embedding.reshape(1, -1)
		return source_embedding

	if model_type == 'hyperswap':
		source_embedding = source_face.embedding_norm.reshape((1, -1))
		return source_embedding

	if model_type == 'inswapper':
		model_path = get_model_options().get('sources').get('face_swapper').get('path')
		model_initializer = get_static_model_initializer(model_path)
		source_embedding = source_face.embedding.reshape((1, -1))
		source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
		return source_embedding

	source_embedding = source_face.embedding.reshape(-1, 512)
	_, source_embedding_norm = convert_source_embedding(source_embedding)
	source_embedding = source_embedding_norm.reshape(1, -1)
	return source_embedding


def balance_source_embedding(source_embedding : Embedding, target_embedding : Embedding) -> Embedding:
	model_type = get_model_options().get('type')
	face_swapper_weight = state_manager.get_item('face_swapper_weight')
	face_swapper_weight = numpy.interp(face_swapper_weight, [ 0, 1 ], [ 0.35, -0.35 ]).astype(numpy.float32)

	if model_type in [ 'hififace', 'hyperswap', 'inswapper', 'simswap' ]:
		target_embedding = target_embedding / numpy.linalg.norm(target_embedding)

	source_embedding = source_embedding.reshape(1, -1)
	target_embedding = target_embedding.reshape(1, -1)
	source_embedding = source_embedding * (1 - face_swapper_weight) + target_embedding * face_swapper_weight
	return source_embedding


# ---------------------------------------------------------------------------
# CUDA helpers (IOBinding + preprocessing on GPU) guarded to preserve output quality.
# These keep math identical but move tensors to CUDA when available.
# ---------------------------------------------------------------------------


def _use_iobinding() -> bool:
	if not (_ort_available and _torch_available):
		return False
	if not gpu_accel_enabled():
		return False
	try:
		return torch.cuda.is_available()
	except Exception:
		return False


def _run_faceswapper_iobinding(face_swapper, face_swapper_inputs: dict) -> Optional[VisionFrame]:
	"""
	Run face_swapper with ONNX Runtime IOBinding on CUDA to avoid extra copies.
	Falls back to None on any failure so caller can use the standard path.
	"""
	try:
		if 'CUDAExecutionProvider' not in [p[0] if isinstance(p, (list, tuple)) else p for p in face_swapper.get_providers()]:
			return None
		device_type, device_id = current_cuda_device()
		cache_key = id(face_swapper)
		cache_entry = _iobinding_cache.get(cache_key)
		if cache_entry:
			binding = cache_entry['binding']
			bound_inputs = cache_entry['inputs']
			bound_outputs = cache_entry['outputs']
		else:
			binding = face_swapper.io_binding()
			bound_inputs = {}
			bound_outputs = {}

		# Prepare and bind inputs (reuse buffers when shape matches)
		for name, arr in face_swapper_inputs.items():
			if isinstance(arr, torch.Tensor):
				tensor = arr
				if tensor.device.type != 'cuda':
					tensor = tensor.to(torch.device(f"{device_type}:{device_id}"))
			elif isinstance(arr, numpy.ndarray):
				tensor = torch.from_numpy(arr).contiguous().to(torch.device(f"{device_type}:{device_id}"))
			else:
				return None
			prev = bound_inputs.get(name)
			if prev is None or tuple(prev.shape) != tuple(tensor.shape):
				bound_inputs[name] = tensor
				binding.bind_input(
					name=name,
					device_type=device_type,
					device_id=device_id,
					element_type=numpy.float32,
					shape=tuple(tensor.shape),
					buffer_ptr=tensor.data_ptr()
				)
			else:
				# Update data pointer if tensor changed
				binding.bind_input(
					name=name,
					device_type=device_type,
					device_id=device_id,
					element_type=numpy.float32,
					shape=tuple(tensor.shape),
					buffer_ptr=tensor.data_ptr()
				)

		# Bind outputs to CUDA; reuse buffer if shape known
		outputs_cfg = face_swapper.get_outputs()
		for out in outputs_cfg:
			prev_out = bound_outputs.get(out.name)
			if prev_out is None:
				binding.bind_output(out.name, device_type, device_id)
			else:
				binding.bind_output(
					name=out.name,
					device_type=device_type,
					device_id=device_id,
					buffer_ptr=prev_out.data_ptr(),
					shape=tuple(prev_out.shape),
					element_type=numpy.float32
				)

		face_swapper.run_with_iobinding(binding)
		outputs = binding.copy_outputs_to_cpu()
		if not outputs:
			return None
		# Cache outputs buffers for reuse if shapes fixed
		try:
			for ort_out, out_cfg in zip(outputs, outputs_cfg):
				tensor_out = torch.from_numpy(ort_out).to(torch.float32)  # CPU tensor
				bound_outputs[out_cfg.name] = tensor_out
		except Exception:
			pass

		_iobinding_cache[cache_key] = {
			'binding': binding,
			'inputs': bound_inputs,
			'outputs': bound_outputs,
		}

		return outputs[0][0]
	except Exception:
		return None


def convert_source_embedding(source_embedding : Embedding) -> Tuple[Embedding, Embedding]:
	source_embedding = forward_convert_embedding(source_embedding)
	source_embedding = source_embedding.ravel()
	source_embedding_norm = source_embedding / numpy.linalg.norm(source_embedding)
	return source_embedding, source_embedding_norm


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0).astype(numpy.float32)
	return crop_vision_frame


def prepare_crop_frame_cuda(crop_vision_frame : VisionFrame) -> VisionFrame | None:
	if not gpu_accel_enabled() or not _torch_available:
		return None
	if not torch.cuda.is_available():
		return None
	try:
		opts = get_model_options()
		mean_key = tuple(opts.get('mean'))
		std_key = tuple(opts.get('standard_deviation'))
		cache_key = (mean_key, std_key)
		mean_std = _cuda_mean_std_cache.get(cache_key)
		if mean_std is None:
			model_mean = torch.tensor(opts.get('mean'), device="cuda", dtype=torch.float32).view(1, 1, 1, 3)
			model_standard_deviation = torch.tensor(opts.get('standard_deviation'), device="cuda", dtype=torch.float32).view(1, 1, 1, 3)
			mean_std = (model_mean, model_standard_deviation)
			_cuda_mean_std_cache[cache_key] = mean_std
		else:
			model_mean, model_standard_deviation = mean_std

		tensor = torch.from_numpy(crop_vision_frame).to(torch.float32).to("cuda", non_blocking=True)
		tensor = tensor.unsqueeze(0)  # 1,H,W,3
		tensor = tensor[:, :, :, [2, 1, 0]]  # BGR->RGB
		tensor = tensor * (1.0 / 255.0)
		tensor = (tensor - model_mean) / model_standard_deviation
		tensor = tensor.permute(0, 3, 1, 2).contiguous()
		return tensor
	except Exception:
		return None


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	if _torch_available and isinstance(crop_vision_frame, torch.Tensor):
		tensor = crop_vision_frame
		if tensor.device.type == 'cuda':
			tensor = tensor.detach().cpu()
		tensor = tensor.permute(1, 2, 0)
		crop_vision_frame_np = tensor.numpy()
	else:
		crop_vision_frame_np = crop_vision_frame.transpose(1, 2, 0)

	if model_type in [ 'ghost', 'hififace', 'hyperswap', 'uniface' ]:
		crop_vision_frame_np = crop_vision_frame_np * model_standard_deviation + model_mean

	crop_vision_frame_np = crop_vision_frame_np.clip(0, 1)
	crop_vision_frame_np = crop_vision_frame_np[:, :, ::-1] * 255
	return crop_vision_frame_np


def extract_source_face(source_vision_frames : List[VisionFrame]) -> Optional[Face]:
	source_faces = []

	if source_vision_frames:
		for source_vision_frame in source_vision_frames:
			temp_faces = get_many_faces([source_vision_frame])
			temp_faces = sort_faces_by_order(temp_faces, 'large-small')

			if temp_faces:
				source_faces.append(get_first(temp_faces))

	return get_average_face(source_faces)


def process_frame(inputs : FaceSwapperInputs) -> ProcessorOutputs:
	global _frame_count

	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	source_face = extract_source_face(source_vision_frames)
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	# If a prior frame faulted the CUDA context, skip heavy work to avoid crashes
	global _swap_faulted
	if _swap_faulted:
		return temp_vision_frame, temp_vision_mask

	if source_face and target_faces:
		for target_face in target_faces:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			try:
				temp_vision_frame = swap_face(source_face, target_face, temp_vision_frame)
			except Exception as exc:
				logger.error('Recovered from CUDA/ORT swap failure: ' + str(exc), __name__)
				_swap_faulted = True
				try:
					state_manager.set_item('abort_cuda', True)
				except Exception:
					pass
				try:
					cleanup_cuda_memory(aggressive=True)
				except Exception:
					pass
				# Bail out after a failed swap to avoid cascading GPU faults
				break

	# Increment frame counter
	_frame_count += 1

	# Periodic CUDA memory cleanup to prevent OOM
	# Clean up every N frames to maintain stable memory usage
	if _frame_count % _batch_cleanup_interval == 0:
		cleanup_cuda_memory(aggressive=False)

	# Aggressive cleanup every 50 frames
	if _frame_count % 50 == 0:
		cleanup_cuda_memory(aggressive=True)

	return temp_vision_frame, temp_vision_mask
