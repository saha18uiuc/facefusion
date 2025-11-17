import importlib
import random
from time import sleep, time
from typing import List

from onnxruntime import ExecutionMode, GraphOptimizationLevel, InferenceSession, SessionOptions

from facefusion import logger, process_manager, state_manager, translator
from facefusion.app_context import detect_app_context
from facefusion.common_helper import is_windows
from facefusion.execution import create_inference_session_providers, has_execution_provider
from facefusion.exit_helper import fatal_exit
from facefusion.filesystem import get_file_name, is_file
from facefusion.time_helper import calculate_end_time
from facefusion.types import DownloadSet, ExecutionProvider, InferencePool, InferencePoolSet

# Import CUDA graph manager and IO Binding
try:
	from facefusion import cuda_graph_manager
except:
	cuda_graph_manager = None

try:
	from facefusion.cuda_io_binding import wrap_session_with_io_binding
except:
	wrap_session_with_io_binding = None

INFERENCE_POOL_SET : InferencePoolSet =\
{
	'cli': {},
	'ui': {}
}


def get_inference_pool(module_name : str, model_names : List[str], model_source_set : DownloadSet) -> InferencePool:
	while process_manager.is_checking():
		sleep(0.5)
	execution_device_ids = state_manager.get_item('execution_device_ids')
	execution_providers = resolve_execution_providers(module_name)
	app_context = detect_app_context()

	for execution_device_id in execution_device_ids:
		inference_context = get_inference_context(module_name, model_names, execution_device_id, execution_providers)

		if app_context == 'cli' and INFERENCE_POOL_SET.get('ui').get(inference_context):
			INFERENCE_POOL_SET['cli'][inference_context] = INFERENCE_POOL_SET.get('ui').get(inference_context)
		if app_context == 'ui' and INFERENCE_POOL_SET.get('cli').get(inference_context):
			INFERENCE_POOL_SET['ui'][inference_context] = INFERENCE_POOL_SET.get('cli').get(inference_context)
		if not INFERENCE_POOL_SET.get(app_context).get(inference_context):
			INFERENCE_POOL_SET[app_context][inference_context] = create_inference_pool(model_source_set, execution_device_id, execution_providers)

	current_inference_context = get_inference_context(module_name, model_names, random.choice(execution_device_ids), execution_providers)
	return INFERENCE_POOL_SET.get(app_context).get(current_inference_context)


def create_inference_pool(model_source_set : DownloadSet, execution_device_id : str, execution_providers : List[ExecutionProvider]) -> InferencePool:
	inference_pool : InferencePool = {}

	for model_name in model_source_set.keys():
		model_path = model_source_set.get(model_name).get('path')
		if is_file(model_path):
			inference_pool[model_name] = create_inference_session(model_path, execution_device_id, execution_providers)

	return inference_pool


def clear_inference_pool(module_name : str, model_names : List[str]) -> None:
	execution_device_ids = state_manager.get_item('execution_device_ids')
	execution_providers = resolve_execution_providers(module_name)
	app_context = detect_app_context()

	if is_windows() and has_execution_provider('directml'):
		INFERENCE_POOL_SET[app_context].clear()

	for execution_device_id in execution_device_ids:
		inference_context = get_inference_context(module_name, model_names, execution_device_id, execution_providers)
		if INFERENCE_POOL_SET.get(app_context).get(inference_context):
			del INFERENCE_POOL_SET[app_context][inference_context]


def create_inference_session(model_path : str, execution_device_id : str, execution_providers : List[ExecutionProvider]) -> InferenceSession:
	model_file_name = get_file_name(model_path)
	start_time = time()

	try:
		# Create optimized session options only for compatible models
		session_options = None
		is_hyperswap = 'hyperswap_1c_256' in model_path.lower()

		if is_hyperswap:
			logger.info(f'Detected hyperswap_1c_256, applying advanced optimizations...', __name__)
			try:
				session_options = SessionOptions()

				# Enable all graph optimizations for maximum performance
				# This includes constant folding, layer fusion, and layout optimizations
				session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

				# Enable parallel execution for independent operators
				parallel_mode = getattr(ExecutionMode, 'ORT_PARALLEL', None)
				if parallel_mode is not None:
					session_options.execution_mode = parallel_mode
				else:
					logger.warn('ExecutionMode.ORT_PARALLEL not available in this onnxruntime build, keeping default execution mode', __name__)

				# Set intra-op and inter-op thread counts for optimal performance
				execution_thread_count = state_manager.get_item('execution_thread_count')
				if execution_thread_count and execution_thread_count > 0:
					session_options.intra_op_num_threads = execution_thread_count
					session_options.inter_op_num_threads = 1  # Most models benefit from single inter-op thread
					logger.info(f'SessionOptions configured: threads={execution_thread_count}, graph_opt=ALL', __name__)
			except Exception as e:
				logger.warn(f'Failed to create SessionOptions: {str(e)}, using defaults', __name__)
				session_options = None

		# Create inference session with selective CUDA graph support
		inference_session_providers = create_inference_session_providers(execution_device_id, execution_providers, model_path)

		try:
			if session_options:
				inference_session = InferenceSession(model_path, sess_options = session_options, providers = inference_session_providers)
				logger.info(f'✓ ONNX graph optimizations enabled for {model_file_name}', __name__)
			else:
				inference_session = InferenceSession(model_path, providers = inference_session_providers)
		except Exception as session_error:
			# If SessionOptions fails, try without it
			logger.warn(f'Session creation with options failed: {str(session_error)}, retrying without options', __name__)
			inference_session = InferenceSession(model_path, providers = inference_session_providers)

		logger.debug(translator.get('loading_model_succeeded').format(model_name = model_file_name, seconds = calculate_end_time(start_time)), __name__)

		# CUDA graphs enabled via execution.py when compatible
		# ONNX Runtime automatically captures graphs during first few inferences
		# No explicit warmup needed - graphs capture naturally with real data
		if has_execution_provider('cuda') and 'cuda' in execution_providers and cuda_graph_manager:
			if cuda_graph_manager.should_enable_cuda_graphs(model_path):
				logger.info(f'CUDA graphs enabled for {model_file_name} - will capture automatically during inference', __name__)

		# Wrap with IO Binding for optimized GPU memory management
		if has_execution_provider('cuda') and 'cuda' in execution_providers and wrap_session_with_io_binding:
			if 'hyperswap_1c_256' in model_path.lower():
				logger.info(f'Attempting to enable IO Binding for {model_file_name}...', __name__)
				try:
					wrapped_session = wrap_session_with_io_binding(inference_session, int(execution_device_id), model_path)
					if wrapped_session != inference_session:
						inference_session = wrapped_session
						logger.info(f'✓ IO Binding enabled for {model_file_name}', __name__)
					else:
						logger.warn(f'IO Binding returned same session - not applied', __name__)
				except Exception as io_error:
					logger.warn(f'IO Binding failed: {str(io_error)}, continuing without it', __name__)

		# Warm up once to trigger graph capture / kernel selection and reduce first-frame latency
		if is_hyperswap:
			try:
				_warmup_inference_session(inference_session)
				logger.debug(f'Warmup inference complete for {model_file_name}', __name__)
			except Exception as warmup_error:
				logger.debug(f'Warmup skipped due to error: {warmup_error}', __name__)

		return inference_session

	except Exception as e:
		logger.error(translator.get('loading_model_failed').format(model_name = model_file_name), __name__)
		logger.error(f'Error details: {str(e)}', __name__)
		fatal_exit(1)


def get_inference_context(module_name : str, model_names : List[str], execution_device_id : str, execution_providers : List[ExecutionProvider]) -> str:
	inference_context = '.'.join([ module_name ] + model_names + [ str(execution_device_id) ] + list(execution_providers))
	return inference_context


def resolve_execution_providers(module_name : str) -> List[ExecutionProvider]:
	module = importlib.import_module(module_name)

	if hasattr(module, 'resolve_execution_providers'):
		return getattr(module, 'resolve_execution_providers')()
	return state_manager.get_item('execution_providers')


def _warmup_inference_session(inference_session : InferenceSession) -> None:
	"""
	Run a single dummy inference to trigger CUDA graph capture / kernel selection.
	Uses minimal static shapes (1 for dynamic dims) so warmup is fast.
	"""
	try:
		inputs = inference_session.get_inputs()
	except Exception:
		return

	warmup_feed = {}
	for meta in inputs:
		# Build a static shape with 1s for unknown dims
		shape = []
		for dim in getattr(meta, 'shape', []) or []:
			if isinstance(dim, int) and dim > 0:
				shape.append(dim)
			else:
				shape.append(1)

		if not shape:
			continue

		# Default to float32 warmup tensor to avoid type mismatches
		import numpy
		warmup_feed[meta.name] = numpy.zeros(shape, dtype = numpy.float32)

	if warmup_feed:
		inference_session.run(None, warmup_feed)
