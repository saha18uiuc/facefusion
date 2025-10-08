"""FaceFusion package initialisation."""
from __future__ import annotations

import os

# Configure PyTorch CUDA allocator to reduce fragmentation stalls when many
# small kernels are launched (recommended in performance tuning guidance).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')

# Expose version metadata if needed by clients.
__all__ = ['__version__']
__version__ = '3.4.0-enhanced'
