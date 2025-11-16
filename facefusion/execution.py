import os
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from functools import lru_cache
from typing import List, Optional

from onnxruntime import get_available_providers, set_default_logger_severity

import facefusion.choices
from pathlib import Path

from facefusion.types import ExecutionDevice, ExecutionProvider, InferenceSessionProvider, ValueAndUnit

set_default_logger_severity(3)


def has_execution_provider(execution_provider : ExecutionProvider) -> bool:
	return execution_provider in get_available_execution_providers()


def get_available_execution_providers() -> List[ExecutionProvider]:
	inference_session_providers = get_available_providers()
	available_execution_providers : List[ExecutionProvider] = []

	for execution_provider, execution_provider_value in facefusion.choices.execution_provider_set.items():
		if execution_provider_value in inference_session_providers:
			index = facefusion.choices.execution_providers.index(execution_provider)
			available_execution_providers.insert(index, execution_provider)

	return available_execution_providers


def _ensure_cache_dir(*segments: str) -> str:
	path_segments = [ str(segment) for segment in segments ]
	# If an absolute path is provided as the first segment, honor it directly.
	base = Path(path_segments[0])
	if base.is_absolute():
		cache_path = base.joinpath(*path_segments[1:])
	else:
		cache_path = Path('.caches').joinpath(*path_segments)
	cache_path.mkdir(parents = True, exist_ok = True)
	return str(cache_path)


def _stage_trt_cache(cache_root: str) -> str:
	"""If cache is on slower storage (e.g., Drive), stage to tmpfs for faster reads."""
	try:
		cache_path = Path(cache_root)
		if '/content/drive' in str(cache_path) and cache_path.exists():
			fast_path = Path('/dev/shm/trt_cache_stage')
			if not fast_path.exists():
				shutil.copytree(cache_path, fast_path, dirs_exist_ok=True)
			return str(fast_path)
	except Exception:
		pass
	return cache_root


def create_inference_session_providers(execution_device_id : str, execution_providers : List[ExecutionProvider], model_identifier : Optional[str] = None, allow_tensorrt : bool = False) -> List[InferenceSessionProvider]:
	# Honor the requested order. Only wire TensorRT options when explicitly allowed.
	requested_execution_providers = list(execution_providers)
	inference_session_providers : List[InferenceSessionProvider] = []

	# Defensive: avoid empty provider lists
	if not requested_execution_providers:
		raise RuntimeError("No execution providers requested; cannot create inference session providers")

	for execution_provider in requested_execution_providers:
		if execution_provider == 'cuda':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id,
				'cudnn_conv_algo_search': resolve_cudnn_conv_algo_search()
			}))
		if execution_provider == 'tensorrt' and allow_tensorrt:
			env_root = os.environ.get('TENSORRT_CACHE_DIR')
			repo_root_cache = 'trt_cache' if Path('trt_cache').exists() else None
			default_root = '.caches/tensorrt'
			trt_cache_root = env_root or repo_root_cache or default_root
			trt_cache_root = _stage_trt_cache(trt_cache_root)
			cache_dir = _ensure_cache_dir(trt_cache_root, model_identifier or 'shared', execution_device_id)
			provider_options = {
				'device_id': execution_device_id,
				'trt_engine_cache_enable': True,
				'trt_engine_cache_path': cache_dir,
				'trt_timing_cache_enable': True,
				'trt_timing_cache_path': cache_dir,
				'trt_fp16_enable': False,
				# Opt level 1 minimizes build time; cached engine still runs at full speed
				'trt_builder_optimization_level': 1
			}
			# Optional static-shape hint: when set, ask TRT to build sequentially (fewer profiles, smaller engine)
			if os.environ.get('TRT_STATIC_SHAPES') == '1':
				provider_options['trt_force_sequential_engine_build'] = True
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider), provider_options))
		if execution_provider in [ 'directml', 'rocm' ]:
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id
			}))
		if execution_provider == 'migraphx':
			cache_dir = _ensure_cache_dir('migraphx', model_identifier or 'shared', execution_device_id)
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_id': execution_device_id,
				'migraphx_model_cache_dir': cache_dir
			}))
		if execution_provider == 'openvino':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'device_type': resolve_openvino_device_type(execution_device_id),
				'precision': 'FP32'
			}))
		if execution_provider == 'coreml':
			inference_session_providers.append((facefusion.choices.execution_provider_set.get(execution_provider),
			{
				'SpecializationStrategy': 'FastPrediction',
				'ModelCacheDirectory': '.caches'
			}))

	if 'cpu' in execution_providers:
		inference_session_providers.append(facefusion.choices.execution_provider_set.get('cpu'))

	return inference_session_providers


def resolve_cudnn_conv_algo_search() -> str:
	execution_devices = detect_static_execution_devices()
	product_names = ('GeForce GTX 1630', 'GeForce GTX 1650', 'GeForce GTX 1660')

	for execution_device in execution_devices:
		if execution_device.get('product').get('name').startswith(product_names):
			return 'DEFAULT'

	return 'EXHAUSTIVE'


def resolve_openvino_device_type(execution_device_id : str) -> str:
	if execution_device_id == '0':
		return 'GPU'
	return 'GPU.' + execution_device_id


def run_nvidia_smi() -> subprocess.Popen[bytes]:
	commands = [ shutil.which('nvidia-smi'), '--query', '--xml-format' ]
	return subprocess.Popen(commands, stdout = subprocess.PIPE)


@lru_cache()
def detect_static_execution_devices() -> List[ExecutionDevice]:
	return detect_execution_devices()


def detect_execution_devices() -> List[ExecutionDevice]:
	execution_devices : List[ExecutionDevice] = []

	try:
		output, _ = run_nvidia_smi().communicate()
		root_element = ElementTree.fromstring(output)
	except Exception:
		root_element = ElementTree.Element('xml')

	for gpu_element in root_element.findall('gpu'):
		execution_devices.append(
		{
			'driver_version': root_element.findtext('driver_version'),
			'framework':
			{
				'name': 'CUDA',
				'version': root_element.findtext('cuda_version')
			},
			'product':
			{
				'vendor': 'NVIDIA',
				'name': gpu_element.findtext('product_name').replace('NVIDIA', '').strip()
			},
			'video_memory':
			{
				'total': create_value_and_unit(gpu_element.findtext('fb_memory_usage/total')),
				'free': create_value_and_unit(gpu_element.findtext('fb_memory_usage/free'))
			},
			'temperature':
			{
				'gpu': create_value_and_unit(gpu_element.findtext('temperature/gpu_temp')),
				'memory': create_value_and_unit(gpu_element.findtext('temperature/memory_temp'))
			},
			'utilization':
			{
				'gpu': create_value_and_unit(gpu_element.findtext('utilization/gpu_util')),
				'memory': create_value_and_unit(gpu_element.findtext('utilization/memory_util'))
			}
		})

	return execution_devices


def create_value_and_unit(text : str) -> Optional[ValueAndUnit]:
	if ' ' in text:
		value, unit = text.split()

		return\
		{
			'value': int(value),
			'unit': str(unit)
		}
	return None
