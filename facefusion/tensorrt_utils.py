from pathlib import Path
from typing import Iterable, Sequence

import onnx
from onnxsim import simplify

from facefusion import logger


def ensure_static_hyperswap(src : str,
	dst : str,
	input_name : str = 'input',
	shape : Sequence[int] = (1, 3, 256, 256)) -> str:
	"""Create a static-shape hyperswap ONNX once and reuse it.

	If the destination file already exists, it is returned untouched.
	If conversion fails for any reason, the original src path is returned so the
	caller can fall back to dynamic shapes.
	"""
	src_path = Path(src)
	dst_path = Path(dst)
	if dst_path.exists():
		return str(dst_path)
	try:
		logger.info(f"[TRT] building static hyperswap ONNX at {dst_path}", __name__)
		model = onnx.load(str(src_path))
		# If the provided input_name is not present, fall back to the first input
		inputs = list(model.graph.input)
		if not inputs:
			raise RuntimeError('model has no inputs')
		name = input_name if any(i.name == input_name for i in inputs) else inputs[0].name
		# Derive shape from model if available; otherwise use user-provided shape.
		dims = []
		for d in next(i.type.tensor_type.shape.dim for i in inputs if i.name == name):
			dims.append(d.dim_value if d.dim_value > 0 else None)
		# If any dim is unknown, don't force static—return dynamic model.
		if any(v is None for v in dims):
			logger.info('[TRT] hyperswap input has dynamic dims; keeping dynamic ONNX', __name__)
			return str(src_path)
		model_opt, check = simplify(
			model,
			overwrite_input_shapes = { name: dims or list(shape) },
			dynamic_input_shape = False
		)
		if not check:
			raise RuntimeError('onnx-simplify validation failed')
		dst_path.parent.mkdir(parents = True, exist_ok = True)
		onnx.save(model_opt, str(dst_path))
		logger.info(f"[TRT] saved static hyperswap ONNX to {dst_path}", __name__)
		return str(dst_path)
	except Exception as exception:
		logger.info(f"[TRT] static hyperswap generation failed: {exception}; using dynamic model", __name__)
		return str(src_path)
