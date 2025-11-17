"""
CUDA IO Binding - Optimized GPU memory management for ONNX Runtime

This module provides IO Binding support to eliminate CPU-GPU data transfers
and keep tensors resident on GPU throughout inference pipeline.
"""

from typing import Dict, List, Optional
import numpy
from onnxruntime import InferenceSession
try:
	from onnxruntime import OrtValue
except Exception:
	OrtValue = None

from facefusion import logger
from facefusion.execution import has_execution_provider


class CUDAIOBindingSession:
	"""
	Wrapper for InferenceSession that uses IO Binding for CUDA.

	Benefits:
	- Eliminates CPU→GPU and GPU→CPU data transfers
	- Keeps tensors on GPU between inferences
	- Works seamlessly with CUDA graphs
	- Falls back to standard .run() if CUDA unavailable
	"""

	def __init__(self, inference_session: InferenceSession, device_id: int = 0):
		"""
		Initialize IO Binding session.

		Args:
			inference_session: The ONNX Runtime InferenceSession
			device_id: CUDA device ID
		"""
		self.session = inference_session
		self.device_id = device_id
		self.use_io_binding = has_execution_provider('cuda')

		if self.use_io_binding:
			self.device_type = 'cuda'
			self.input_names = [inp.name for inp in self.session.get_inputs()]
			self.output_names = [out.name for out in self.session.get_outputs()]
			self.io_binding = self.session.io_binding()
			self._bind_outputs_once()
			logger.debug(f'IO Binding enabled on CUDA device {device_id}', __name__)
		else:
			logger.debug('IO Binding disabled - CUDA not available', __name__)

	def run(self, output_names: Optional[List[str]], input_feed: Dict[str, numpy.ndarray]) -> List[numpy.ndarray]:
		"""
		Run inference with IO Binding if available.

		Args:
			output_names: List of output names (ignored for IO Binding)
			input_feed: Dictionary mapping input names to numpy arrays

		Returns:
			List of output numpy arrays
		"""
		if not self.use_io_binding:
			# Fallback to standard run
			return self.session.run(output_names, input_feed)

		try:
			# Clear and re-bind inputs for this batch/frame
			self.io_binding.clear_binding_inputs()

			# Bind inputs
			for input_name in self.input_names:
				if input_name in input_feed:
					input_tensor = input_feed[input_name]

					# Handle pre-GPU data if caller passes an OrtValue on CUDA
					if OrtValue and isinstance(input_tensor, OrtValue) and input_tensor.device_name().lower() == 'cuda':
						self.io_binding.bind_ortvalue_input(input_name, input_tensor)
						continue

					# Ensure contiguous layout and preferred dtype to reduce copies
					if not input_tensor.flags['C_CONTIGUOUS']:
						input_tensor = numpy.ascontiguousarray(input_tensor)
					if input_tensor.dtype not in (numpy.float16, numpy.float32):
						input_tensor = input_tensor.astype(numpy.float32, copy = False)

					self.io_binding.bind_cpu_input(input_name, input_tensor)

			# Ensure outputs remain bound to GPU (only binds once unless session recreated)
			if not getattr(self, 'outputs_bound', False):
				self._bind_outputs_once()

			# Run inference with IO binding
			self.session.run_with_iobinding(self.io_binding)

			# Get outputs and copy back to CPU
			outputs = self.io_binding.copy_outputs_to_cpu()

			return outputs

		except Exception as e:
			# If IO binding fails, fall back to standard run
			logger.debug(f'IO Binding failed, falling back to standard run: {str(e)}', __name__)
			return self.session.run(output_names, input_feed)

	def get_inputs(self):
		"""Get input metadata."""
		return self.session.get_inputs()

	def get_outputs(self):
		"""Get output metadata."""
		return self.session.get_outputs()

	def set_providers(self, providers):
		"""Set execution providers."""
		return self.session.set_providers(providers)

	def _bind_outputs_once(self) -> None:
		"""Bind outputs to the GPU device once and reuse."""
		for output_name in self.output_names:
			self.io_binding.bind_output(output_name, device_type = self.device_type, device_id = self.device_id)
		self.outputs_bound = True


def wrap_session_with_io_binding(inference_session: InferenceSession, device_id: int = 0, model_path: str = None) -> InferenceSession:
	"""
	Wrap an InferenceSession with IO Binding support for CUDA.

	Args:
		inference_session: The ONNX Runtime InferenceSession to wrap
		device_id: CUDA device ID
		model_path: Path to model file (for selective enablement)

	Returns:
		Wrapped session with IO Binding support
	"""
	# Only enable IO Binding for compatible models
	if not has_execution_provider('cuda'):
		return inference_session

	# Check if model is compatible (same logic as CUDA graphs)
	if model_path:
		model_name = model_path.lower()
		# ONLY enable for hyperswap_1c_256 as requested
		if 'hyperswap_1c_256' not in model_name:
			logger.debug(f'IO Binding disabled for {model_path} - not hyperswap_1c_256', __name__)
			return inference_session

	logger.debug(f'Wrapping session with IO Binding for device {device_id}', __name__)
	return CUDAIOBindingSession(inference_session, device_id)
