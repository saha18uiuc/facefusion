"""
CUDA operations for FaceFusion 3.5.0
High-performance kernels for face swapping with deterministic operations.
"""

# Import will be attempted when module is loaded
# If build fails, this will raise ImportError which is handled by vision_enhanced.py
try:
    from . import ff_cuda_ops
    __all__ = ['ff_cuda_ops']
except ImportError:
    # Build not completed or CUDA not available
    pass
