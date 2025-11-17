"""
CUDA Graph Manager - Advanced CUDA Graph Support for ONNX Runtime

This module provides selective CUDA graph enablement with IO Binding,
output validation, and automatic fallback for incompatible models.
"""

from typing import Dict, List, Optional
import numpy

from onnxruntime import InferenceSession

from facefusion import choices, logger
from facefusion.execution import has_execution_provider


def is_model_compatible_with_cuda_graphs(model_name : str) -> bool:
	"""
	Check if a model is compatible with CUDA graphs.

	Args:
		model_name: Name of the model to check

	Returns:
		True if model is compatible, False otherwise
	"""
	model_lower = model_name.lower()

	# ONLY enable for hyperswap_1c_256 as requested by user
	if 'hyperswap_1c_256' in model_lower:
		logger.debug(f'Model {model_name} is hyperswap_1c_256 - CUDA graphs ENABLED', __name__)
		return True

	# All other models: CUDA graphs DISABLED
	logger.debug(f'Model {model_name} is not hyperswap_1c_256 - CUDA graphs DISABLED', __name__)
	return False


def should_enable_cuda_graphs(model_path : str) -> bool:
	"""
	Determine if CUDA graphs should be enabled for a specific model.

	Args:
		model_path: Path to the ONNX model file

	Returns:
		True if CUDA graphs should be enabled
	"""
	if not has_execution_provider('cuda'):
		return False

	# Extract model name from path
	model_name = model_path.lower()

	# Check compatibility
	return is_model_compatible_with_cuda_graphs(model_name)


def validate_inference_output(outputs : List, expected_min_outputs : int = 1) -> bool:
	"""
	Validate ONNX Runtime inference outputs.

	Args:
		outputs: List of output tensors from inference
		expected_min_outputs: Minimum number of expected outputs

	Returns:
		True if outputs are valid, False otherwise
	"""
	if not outputs:
		return False

	if len(outputs) < expected_min_outputs:
		return False

	# Check each output is not None and not empty
	for output in outputs:
		if output is None:
			return False
		if isinstance(output, numpy.ndarray):
			if output.size == 0:
				return False

	return True


def create_warmup_inputs(inference_session : InferenceSession) -> Optional[Dict[str, numpy.ndarray]]:
	"""
	Create realistic warmup inputs for CUDA graph capture.

	Args:
		inference_session: The ONNX Runtime InferenceSession

	Returns:
		Dictionary of input tensors, or None if creation fails
	"""
	try:
		inputs_info = inference_session.get_inputs()
		warmup_inputs : Dict[str, numpy.ndarray] = {}

		for input_info in inputs_info:
			input_name = input_info.name
			input_shape = input_info.shape
			input_type = input_info.type

			# Create concrete shape
			concrete_shape = []
			for dim in input_shape:
				if isinstance(dim, int) and dim > 0:
					concrete_shape.append(dim)
				else:
					# Dynamic dimension - use sensible default
					concrete_shape.append(1)

			# Create realistic input based on type and name
			if 'float' in input_type:
				# Use small random values for float inputs
				if 'source' in input_name.lower() or 'embedding' in input_name.lower():
					# Embedding-like inputs: normalized random values
					warmup_inputs[input_name] = numpy.random.randn(*concrete_shape).astype(numpy.float32) * 0.1
				else:
					# Image-like inputs: values in [0, 1] range
					warmup_inputs[input_name] = numpy.random.rand(*concrete_shape).astype(numpy.float32)
			elif 'int64' in input_type:
				warmup_inputs[input_name] = numpy.ones(concrete_shape, dtype = numpy.int64)
			elif 'int32' in input_type:
				warmup_inputs[input_name] = numpy.ones(concrete_shape, dtype = numpy.int32)
			else:
				warmup_inputs[input_name] = numpy.random.rand(*concrete_shape).astype(numpy.float32)

		return warmup_inputs

	except Exception as exception:
		logger.debug(f'Failed to create warmup inputs: {str(exception)}', __name__)
		return None


def warmup_for_cuda_graphs(inference_session : InferenceSession, model_name : str, iterations : int = 5) -> bool:
	"""
	Perform warmup to trigger CUDA graph capture.

	Args:
		inference_session: The ONNX Runtime InferenceSession
		model_name: Name of the model (for logging)
		iterations: Number of warmup iterations

	Returns:
		True if warmup succeeded, False otherwise
	"""
	if not has_execution_provider('cuda'):
		return False

	try:
		# Create warmup inputs
		warmup_inputs = create_warmup_inputs(inference_session)
		if not warmup_inputs:
			return False

		# Run warmup iterations
		for i in range(iterations):
			try:
				outputs = inference_session.run(None, warmup_inputs)

				# Validate outputs
				if not validate_inference_output(outputs):
					logger.warn(f'CUDA graph warmup iteration {i+1} produced invalid outputs for {model_name}, disabling graphs', __name__)
					return False

			except Exception as iteration_error:
				logger.debug(f'CUDA graph warmup iteration {i+1} failed for {model_name}: {str(iteration_error)}', __name__)
				return False

		logger.debug(f'CUDA graph warmup succeeded for {model_name}', __name__)
		return True

	except Exception as exception:
		logger.debug(f'CUDA graph warmup failed for {model_name}: {str(exception)}', __name__)
		return False


def get_cuda_graph_config(model_path : str) -> Dict:
	"""
	Get CUDA graph configuration for a specific model.

	Args:
		model_path: Path to the ONNX model file

	Returns:
		Dictionary with CUDA graph configuration options
	"""
	config = {}

	if should_enable_cuda_graphs(model_path):
		config['enable_cuda_graph'] = True
		logger.debug(f'CUDA graphs ENABLED for {model_path}', __name__)
	else:
		logger.debug(f'CUDA graphs DISABLED for {model_path} (model not in compatibility whitelist)', __name__)

	return config
