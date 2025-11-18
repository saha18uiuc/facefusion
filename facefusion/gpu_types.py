"""
GPU-resident tensor types and utilities for CUDA-accelerated processing.

This module extends the base types to support GPU-resident processing where frames
stay on CUDA throughout the entire pipeline, eliminating CPU↔GPU transfers.
"""

from typing import Any, Literal, Tuple, TypeAlias, Union

import torch
from numpy.typing import NDArray

# GPU-resident frame types
CUDAFrame: TypeAlias = torch.Tensor  # GPU-resident vision frame (H, W, C)
CUDAMask: TypeAlias = torch.Tensor  # GPU-resident mask (H, W) or (H, W, 1)

# Hybrid types that support both CPU and GPU
VisionFrameAny: TypeAlias = Union[NDArray[Any], CUDAFrame]
MaskAny: TypeAlias = Union[NDArray[Any], CUDAMask]

# Processing mode
ProcessingMode = Literal['cpu', 'gpu']

# Processor outputs for GPU mode
CUDAProcessorOutputs: TypeAlias = Tuple[CUDAFrame, CUDAMask]


def is_gpu_mode_available() -> bool:
	"""
	Check if GPU mode is available on this system.

	Returns:
		True if CUDA is available and PyTorch is properly configured
	"""
	return torch.cuda.is_available()


def get_processing_mode() -> ProcessingMode:
	"""
	Get the current processing mode from state manager.

	Returns:
		'gpu' if GPU mode is enabled and available, 'cpu' otherwise
	"""
	# Import here to avoid circular dependency
	from facefusion import state_manager

	gpu_mode_enabled = state_manager.get_item('gpu_mode')
	if gpu_mode_enabled and is_gpu_mode_available():
		return 'gpu'
	return 'cpu'


def is_cuda_tensor(frame: Any) -> bool:
	"""
	Check if a frame is a CUDA tensor.

	Args:
		frame: Frame to check (can be NumPy array or torch.Tensor)

	Returns:
		True if frame is a CUDA tensor
	"""
	return isinstance(frame, torch.Tensor) and frame.is_cuda


def is_numpy_array(frame: Any) -> bool:
	"""
	Check if a frame is a NumPy array.

	Args:
		frame: Frame to check

	Returns:
		True if frame is a NumPy array
	"""
	import numpy
	return isinstance(frame, numpy.ndarray)


def numpy_to_cuda(frame: NDArray[Any], device: str = 'cuda:0') -> CUDAFrame:
	"""
	Convert NumPy array to CUDA tensor with zero-copy when possible.

	Args:
		frame: NumPy array (H, W, C) in uint8 or float32
		device: CUDA device string (default: 'cuda:0')

	Returns:
		CUDA tensor with same dtype and shape
	"""
	# Use from_numpy for zero-copy when possible (only works for CPU tensors)
	# Then move to GPU
	tensor = torch.from_numpy(frame)
	return tensor.to(device=device, non_blocking=True)


def cuda_to_numpy(frame: CUDAFrame) -> NDArray[Any]:
	"""
	Convert CUDA tensor to NumPy array.

	Args:
		frame: CUDA tensor (H, W, C)

	Returns:
		NumPy array with same dtype and shape
	"""
	# Move to CPU first, then convert to numpy
	# contiguous() ensures memory is contiguous for numpy
	return frame.detach().cpu().contiguous().numpy()


def ensure_cuda_tensor(frame: VisionFrameAny, device: str = 'cuda:0') -> CUDAFrame:
	"""
	Ensure frame is a CUDA tensor, converting if necessary.

	Args:
		frame: Frame (can be NumPy array or CUDA tensor)
		device: Target CUDA device (default: 'cuda:0')

	Returns:
		CUDA tensor
	"""
	if is_cuda_tensor(frame):
		# Already CUDA, ensure it's on the right device
		if frame.device.type == 'cuda' and str(frame.device) == device:
			return frame
		return frame.to(device=device, non_blocking=True)
	else:
		# NumPy array, convert to CUDA
		return numpy_to_cuda(frame, device=device)


def ensure_numpy_array(frame: VisionFrameAny) -> NDArray[Any]:
	"""
	Ensure frame is a NumPy array, converting if necessary.

	Args:
		frame: Frame (can be NumPy array or CUDA tensor)

	Returns:
		NumPy array
	"""
	if is_cuda_tensor(frame):
		return cuda_to_numpy(frame)
	else:
		return frame


def get_cuda_device(device_id: str = '0') -> str:
	"""
	Get CUDA device string from device ID.

	Args:
		device_id: Device ID (e.g., '0', '1')

	Returns:
		CUDA device string (e.g., 'cuda:0')
	"""
	return f'cuda:{device_id}'


def synchronize_cuda(device: str = 'cuda:0') -> None:
	"""
	Synchronize CUDA device to ensure all operations are complete.

	Args:
		device: CUDA device string (default: 'cuda:0')
	"""
	torch.cuda.synchronize(device=device)


def get_cuda_memory_info(device: str = 'cuda:0') -> dict:
	"""
	Get CUDA memory information for a device.

	Args:
		device: CUDA device string (default: 'cuda:0')

	Returns:
		Dictionary with 'allocated', 'reserved', 'free' in bytes
	"""
	device_obj = torch.device(device)
	allocated = torch.cuda.memory_allocated(device_obj)
	reserved = torch.cuda.memory_reserved(device_obj)

	# Get total memory
	props = torch.cuda.get_device_properties(device_obj)
	total = props.total_memory
	free = total - allocated

	return {
		'allocated': allocated,
		'reserved': reserved,
		'free': free,
		'total': total
	}
