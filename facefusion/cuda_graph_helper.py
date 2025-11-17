"""
CUDA Graph Helper Module

This module provides utilities for warming up ONNX Runtime inference sessions
to enable CUDA graph capture and replay, significantly improving inference performance.
"""

import numpy
from typing import Dict, List

from onnxruntime import InferenceSession

from facefusion import logger
from facefusion.execution import has_execution_provider


def warmup_inference_session(inference_session : InferenceSession, warmup_iterations : int = 3) -> None:
	"""
	Warm up an ONNX Runtime inference session to trigger CUDA graph capture.

	Args:
		inference_session: The ONNX Runtime InferenceSession to warm up
		warmup_iterations: Number of warmup iterations (default: 3)

	Note:
		This function only performs warmup when CUDA execution provider is available.
		The warmup runs dummy inputs through the model to trigger CUDA graph capture.
	"""
	if not has_execution_provider('cuda'):
		return

	try:
		# Get input information from the model
		inputs_info = inference_session.get_inputs()

		# Skip warmup for models with complex/dynamic shapes that might cause issues
		# CUDA graphs will still be captured on first real inference
		has_dynamic_shapes = False
		for input_info in inputs_info:
			for dim in input_info.shape:
				if not isinstance(dim, int) or dim <= 0:
					has_dynamic_shapes = True
					break
			if has_dynamic_shapes:
				break

		if has_dynamic_shapes:
			logger.debug('Skipping warmup for model with dynamic shapes - CUDA graphs will be captured on first inference', __name__)
			return

		# Create dummy inputs with the correct shapes and types
		dummy_inputs : Dict[str, numpy.ndarray] = {}

		for input_info in inputs_info:
			input_name = input_info.name
			input_shape = input_info.shape
			input_type = input_info.type

			# Create concrete shape (all dims should be positive integers at this point)
			concrete_shape = list(input_shape)

			# Create dummy input based on type with small random values
			if 'float' in input_type:
				dummy_inputs[input_name] = numpy.random.randn(*concrete_shape).astype(numpy.float32) * 0.01
			elif 'int64' in input_type:
				dummy_inputs[input_name] = numpy.ones(concrete_shape, dtype = numpy.int64)
			elif 'int32' in input_type:
				dummy_inputs[input_name] = numpy.ones(concrete_shape, dtype = numpy.int32)
			else:
				# Default to float32 for unknown types
				dummy_inputs[input_name] = numpy.random.randn(*concrete_shape).astype(numpy.float32) * 0.01

		# Run warmup iterations to trigger CUDA graph capture
		# Use try-except for each iteration in case warmup fails
		for i in range(warmup_iterations):
			try:
				result = inference_session.run(None, dummy_inputs)
				# Verify result is not empty
				if not result or len(result) == 0:
					logger.debug(f'Warmup iteration {i+1} returned empty result, skipping remaining warmup', __name__)
					return
			except Exception as iteration_exception:
				logger.debug(f'Warmup iteration {i+1} failed: {str(iteration_exception)}, skipping remaining warmup', __name__)
				return

	except Exception as exception:
		# Log warmup failure but don't crash - the model will still work without CUDA graphs
		logger.debug(f'CUDA graph warmup failed: {str(exception)}', __name__)


def get_warmup_iterations() -> int:
	"""
	Get the number of warmup iterations for CUDA graph capture.

	Returns:
		Number of warmup iterations (default: 3)

	Note:
		3 iterations is generally sufficient for ONNX Runtime to capture CUDA graphs.
		The first iteration may involve JIT compilation, the second and third trigger capture.
	"""
	return 3
