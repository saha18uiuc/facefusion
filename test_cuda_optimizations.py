#!/usr/bin/env python3
"""
Quick validation script for CUDA optimizations integration.
Tests that all modules are properly loaded and accessible.
"""

import sys
import os

def test_module_imports():
    """Test that all optimization modules can be imported."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Test 1: Temporal Smoother
    tests_total += 1
    try:
        from facefusion.processors.modules.temporal_smoother import TemporalSmoother
        smoother = TemporalSmoother()
        print("✅ Temporal Smoother: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Temporal Smoother: FAILED - {e}")

    # Test 2: Enhanced Blender
    tests_total += 1
    try:
        from facefusion.processors.modules.enhanced_blending import EnhancedBlender
        blender = EnhancedBlender()
        print("✅ Enhanced Blender: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Enhanced Blender: FAILED - {e}")

    # Test 3: Batch Processor
    tests_total += 1
    try:
        from facefusion.processors.batch_processor import OptimizedBatchProcessor
        processor = OptimizedBatchProcessor()
        print("✅ Batch Processor: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Batch Processor: FAILED - {e}")

    # Test 4: Quality Monitor
    tests_total += 1
    try:
        from facefusion.quality.quality_monitor import QualityMonitor
        monitor = QualityMonitor(enable_monitoring=False)
        print("✅ Quality Monitor: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Quality Monitor: FAILED - {e}")

    # Test 5: Face Swapper Integration
    tests_total += 1
    try:
        from facefusion.processors.modules import face_swapper
        # Check if our functions exist
        assert hasattr(face_swapper, 'get_temporal_smoother'), "Missing get_temporal_smoother"
        assert hasattr(face_swapper, 'get_enhanced_blender'), "Missing get_enhanced_blender"
        assert hasattr(face_swapper, 'paste_back_enhanced'), "Missing paste_back_enhanced"
        print("✅ Face Swapper Integration: OK")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Face Swapper Integration: FAILED - {e}")

    print("\n" + "=" * 60)
    print(f"Import Tests: {tests_passed}/{tests_total} passed")
    print("=" * 60)

    return tests_passed == tests_total


def test_temporal_smoother_functionality():
    """Test basic temporal smoother functionality."""
    print("\n" + "=" * 60)
    print("Testing Temporal Smoother Functionality")
    print("=" * 60)

    try:
        import numpy as np
        from facefusion.processors.modules.temporal_smoother import TemporalSmoother

        # Create smoother
        smoother = TemporalSmoother(window_size=5, alpha=0.25)

        # Test with dummy landmarks (5 points, 2D)
        landmarks1 = np.array([[100, 100], [200, 100], [150, 150], [100, 200], [200, 200]], dtype=np.float32)
        landmarks2 = np.array([[101, 101], [201, 101], [151, 151], [101, 201], [201, 201]], dtype=np.float32)

        # Smooth landmarks
        smoothed1 = smoother.smooth_landmarks(landmarks1)
        smoothed2 = smoother.smooth_landmarks(landmarks2)

        # Check scene cut detection
        should_reset = smoother.should_reset(landmarks1)

        print(f"✅ Smoothing works: Input shape {landmarks1.shape} -> Output shape {smoothed1.shape}")
        print(f"✅ Scene cut detection works: {should_reset}")
        print(f"✅ Temporal Smoother: FUNCTIONAL")

        return True

    except Exception as e:
        print(f"❌ Temporal Smoother Functionality: FAILED - {e}")
        return False


def test_enhanced_blender_functionality():
    """Test basic enhanced blender functionality."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Blender Functionality")
    print("=" * 60)

    try:
        import numpy as np
        import cv2
        from facefusion.processors.modules.enhanced_blending import EnhancedBlender

        # Create blender
        blender = EnhancedBlender(method='multiband')

        # Create dummy images and mask
        source = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.ones((256, 256), dtype=np.uint8) * 255
        center = (128, 128)

        # Test blending
        result = blender.blend_face(source, target, mask, center)

        print(f"✅ Blending works: Input shape {source.shape} -> Output shape {result.shape}")
        print(f"✅ Enhanced Blender: FUNCTIONAL")

        return True

    except Exception as e:
        print(f"❌ Enhanced Blender Functionality: FAILED - {e}")
        return False


def test_gpu_availability():
    """Test if CUDA/GPU is available."""
    print("\n" + "=" * 60)
    print("Testing GPU Availability")
    print("=" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"✅ CUDA Available: YES")
            print(f"   - Devices: {device_count}")
            print(f"   - Device 0: {device_name}")
            print(f"   - VRAM: {vram:.2f} GB")
        else:
            print(f"⚠️  CUDA Available: NO (will use CPU)")

        return True

    except Exception as e:
        print(f"❌ GPU Check: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FaceFusion CUDA Optimizations - Validation Script")
    print("=" * 60)
    print()

    all_tests = []

    # Run tests
    all_tests.append(("Imports", test_module_imports()))
    all_tests.append(("Temporal Smoother", test_temporal_smoother_functionality()))
    all_tests.append(("Enhanced Blender", test_enhanced_blender_functionality()))
    all_tests.append(("GPU", test_gpu_availability()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in all_tests if result)
    total = len(all_tests)

    for test_name, result in all_tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {passed}/{total} test suites passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed! CUDA optimizations are properly integrated.")
        print("\nNext steps:")
        print("1. Test on a real video:")
        print("   python facefusion.py run --source face.jpg --target video.mp4 --output out.mp4")
        print("2. Monitor for:")
        print("   - No jitter in output")
        print("   - No color warping")
        print("   - Faster processing time")
        print("   - No OOM errors")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
