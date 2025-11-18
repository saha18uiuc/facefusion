#!/usr/bin/env python3
"""Quick test to verify GPU mode setup."""

import sys

print("Testing GPU mode setup...")
print()

# Test 1: Check PyTorch
print("1. Checking PyTorch installation...")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ GPU device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"   ✗ PyTorch not found: {e}")
    sys.exit(1)

# Test 2: Check GPU types module
print()
print("2. Checking GPU types module...")
try:
    from facefusion.gpu_types import (
        get_processing_mode,
        is_gpu_mode_available,
        numpy_to_cuda,
        cuda_to_numpy
    )
    print(f"   ✓ GPU types module loaded")
    print(f"   ✓ GPU mode available: {is_gpu_mode_available()}")
except ImportError as e:
    print(f"   ✗ GPU types import failed: {e}")
    sys.exit(1)

# Test 3: Check state manager
print()
print("3. Checking state manager...")
try:
    from facefusion import state_manager
    state_manager.init_item('gpu_mode', True)
    mode = get_processing_mode()
    print(f"   ✓ Processing mode with gpu_mode=True: {mode}")

    state_manager.init_item('gpu_mode', False)
    mode = get_processing_mode()
    print(f"   ✓ Processing mode with gpu_mode=False: {mode}")
except Exception as e:
    print(f"   ✗ State manager test failed: {e}")
    sys.exit(1)

# Test 4: Quick tensor conversion
print()
print("4. Testing tensor conversions...")
try:
    import numpy as np
    test_array = np.random.rand(100, 100, 3).astype(np.uint8)
    print(f"   ✓ Created test array: {test_array.shape}, {test_array.dtype}")

    cuda_tensor = numpy_to_cuda(test_array)
    print(f"   ✓ Converted to CUDA: {cuda_tensor.shape}, {cuda_tensor.device}")

    back_to_numpy = cuda_to_numpy(cuda_tensor)
    print(f"   ✓ Converted back to NumPy: {back_to_numpy.shape}, {back_to_numpy.dtype}")

    if np.array_equal(test_array, back_to_numpy):
        print(f"   ✓ Round-trip conversion successful!")
    else:
        print(f"   ✗ Arrays don't match after round-trip")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Tensor conversion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("All tests passed! GPU mode is ready.")
print("=" * 50)
print()
print("Run your command with --gpu-mode to enable GPU-resident processing:")
print("  python facefusion.py headless-run --gpu-mode ...")
