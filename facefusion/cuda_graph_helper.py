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

		# Create dummy inputs with the correct shapes and types
		dummy_inputs : Dict[str, numpy.ndarray] = {}

		for input_info in inputs_info:
			input_name = input_info.name
			input_shape = input_info.shape
			input_type = input_info.type

			# Handle dynamic dimensions (replace None or symbolic names with concrete values)
			concrete_shape = []
			for dim in input_shape:
				if isinstance(dim, int) and dim > 0:
					concrete_shape.append(dim)
				else:
					# Use default batch size of 1 for dynamic dimensions
					concrete_shape.append(1)

			# Create dummy input based on type
			if 'float' in input_type:
				dummy_inputs[input_name] = numpy.zeros(concrete_shape, dtype = numpy.float32)
			elif 'int64' in input_type:
				dummy_inputs[input_name] = numpy.zeros(concrete_shape, dtype = numpy.int64)
			elif 'int32' in input_type:
				dummy_inputs[input_name] = numpy.zeros(concrete_shape, dtype = numpy.int32)
			else:
				# Default to float32 for unknown types
				dummy_inputs[input_name] = numpy.zeros(concrete_shape, dtype = numpy.float32)

		# Run warmup iterations to trigger CUDA graph capture
		for _ in range(warmup_iterations):
			inference_session.run(None, dummy_inputs)

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
