#!/usr/bin/env python3
"""
Test script for FaceFusion 3.5.0 optimizations.

Usage:
    python test_optimizations.py --source face.jpg --target video.mp4 --output result.mp4
"""

import sys
import time
import argparse
from pathlib import Path

def test_imports():
    """Test that all optimization modules can be imported."""
    print("\n" + "="*60)
    print("Testing Module Imports")
    print("="*60 + "\n")

    try:
        from facefusion.processors.temporal_smoother import TemporalSmoother
        print("✅ temporal_smoother imported successfully")
    except Exception as e:
        print(f"❌ Failed to import temporal_smoother: {e}")
        return False

    try:
        from facefusion.processors.enhanced_blending import EnhancedBlender
        print("✅ enhanced_blending imported successfully")
    except Exception as e:
        print(f"❌ Failed to import enhanced_blending: {e}")
        return False

    try:
        from facefusion.quality.quality_monitor import QualityMonitor
        print("✅ quality_monitor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import quality_monitor: {e}")
        return False

    print("\n✅ All optimization modules imported successfully!\n")
    return True


def test_temporal_smoother():
    """Test temporal smoother functionality."""
    print("="*60)
    print("Testing Temporal Smoother")
    print("="*60 + "\n")

    try:
        import numpy as np
        from facefusion.processors.temporal_smoother import TemporalSmoother

        # Create smoother
        smoother = TemporalSmoother(window_size=5, alpha=0.25)
        print("✅ TemporalSmoother created")

        # Test with dummy landmarks
        landmarks = np.random.rand(5, 2) * 100  # 5 landmarks, x,y coordinates

        # Smooth multiple frames
        for i in range(10):
            smoothed = smoother.smooth_landmarks(landmarks)
            landmarks = landmarks + np.random.randn(5, 2) * 2  # Add small variation

        print(f"✅ Smoothing test passed (processed 10 frames)")

        # Test reset
        smoother.reset()
        print("✅ Reset test passed")

        return True

    except Exception as e:
        print(f"❌ Temporal smoother test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_blender():
    """Test enhanced blender functionality."""
    print("\n" + "="*60)
    print("Testing Enhanced Blender")
    print("="*60 + "\n")

    try:
        import numpy as np
        import cv2
        from facefusion.processors.enhanced_blending import EnhancedBlender

        # Create blender
        blender = EnhancedBlender(method='basic')  # Use basic for quick test
        print("✅ EnhancedBlender created")

        # Create dummy images
        source = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Test blending
        result = blender.blend_face(source, target, mask, center=(50, 50))
        print(f"✅ Blending test passed (result shape: {result.shape})")

        return True

    except Exception as e:
        print(f"❌ Enhanced blender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_monitor():
    """Test quality monitor functionality."""
    print("\n" + "="*60)
    print("Testing Quality Monitor")
    print("="*60 + "\n")

    try:
        import numpy as np
        from facefusion.quality.quality_monitor import QualityMonitor

        # Create monitor
        monitor = QualityMonitor(enable_monitoring=True, stride=1)
        print("✅ QualityMonitor created")

        # Test with dummy frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            monitor.monitor_frame(frame)

        print(f"✅ Monitored 10 frames")

        # Get summary
        summary = monitor.get_summary()
        print(f"✅ Summary generated (monitored {summary['frames_monitored']} frames)")

        # Print summary
        monitor.print_summary()

        return True

    except Exception as e:
        print(f"❌ Quality monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_face_swapper_integration():
    """Test that face_swapper can import the optimization modules."""
    print("\n" + "="*60)
    print("Testing Face Swapper Integration")
    print("="*60 + "\n")

    try:
        from facefusion.processors.modules.face_swapper import core as face_swapper
        print("✅ face_swapper.core imported successfully")

        # Check if our helper functions exist
        assert hasattr(face_swapper, 'get_temporal_smoother'), "get_temporal_smoother not found"
        print("✅ get_temporal_smoother() found")

        assert hasattr(face_swapper, 'get_enhanced_blender'), "get_enhanced_blender not found"
        print("✅ get_enhanced_blender() found")

        assert hasattr(face_swapper, 'reset_temporal_smoother'), "reset_temporal_smoother not found"
        print("✅ reset_temporal_smoother() found")

        print("\n✅ Face swapper integration verified!")
        return True

    except Exception as e:
        print(f"❌ Face swapper integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FaceFusion 3.5.0 Optimization Test Suite")
    print("="*70)

    results = []

    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Temporal Smoother", test_temporal_smoother()))
    results.append(("Enhanced Blender", test_enhanced_blender()))
    results.append(("Quality Monitor", test_quality_monitor()))
    results.append(("Face Swapper Integration", test_face_swapper_integration()))

    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*70}\n")

    if passed == total:
        print("🎉 All tests passed! Optimizations are ready to use.")
        print("\nNext steps:")
        print("1. Run a test face swap to verify functionality")
        print("2. Measure performance with 'time python facefusion.py run ...'")
        print("3. Compare quality metrics with baseline")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
