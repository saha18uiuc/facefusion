import os

# Configure CUDA allocator defaults unless explicitly overridden by the user.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,expandable_segments:True')
