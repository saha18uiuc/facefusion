"""
Quality Monitoring Module for FaceFusion 3.5.0
Real-time tracking of temporal PSNR, jitter detection, and quality metrics.

Usage:
    from facefusion.quality.quality_monitor import QualityMonitor

    monitor = QualityMonitor()
    for frame in processed_frames:
        monitor.monitor_frame(frame)

    stats = monitor.get_summary()
"""

import cv2
import numpy
from typing import List, Dict, Optional, Tuple
from collections import deque
import json
from pathlib import Path


class QualityMonitor:
    """
    Real-time quality monitoring during video processing.

    Tracks:
    - Temporal PSNR (frame-to-frame stability)
    - Jitter detection
    - Color consistency
    - Boundary artifacts
    """

    def __init__(
        self,
        enable_monitoring: bool = True,
        crop_size: Tuple[int, int] = (480, 480),
        stride: int = 3,
        jitter_threshold: float = 3.0
    ):
        """
        Initialize quality monitor.

        Args:
            enable_monitoring: Enable/disable monitoring (for production vs dev)
            crop_size: Size of center crop for tPSNR calculation
            stride: Process every Nth frame for faster monitoring
            jitter_threshold: dB drop considered as jitter
        """
        self.enable = enable_monitoring
        self.crop_size = crop_size
        self.stride = stride
        self.jitter_threshold = jitter_threshold

        # Frame history
        self.frame_history = deque(maxlen=10)

        # Metrics history
        self.tpsnr_history = []
        self.color_variance_history = []
        self.edge_sharpness_history = []

        # Jitter tracking
        self.jitter_frames = []
        self.jitter_magnitudes = []

        # Frame counter
        self.frame_count = 0

    def monitor_frame(
        self,
        processed_frame: numpy.ndarray,
        original_frame: Optional[numpy.ndarray] = None
    ) -> Dict[str, float]:
        """
        Monitor single frame quality.

        Args:
            processed_frame: Processed output frame
            original_frame: Optional original frame for comparison

        Returns:
            Dictionary of quality metrics for this frame
        """
        if not self.enable:
            return {}

        metrics = {}

        # Only process every Nth frame (stride)
        if self.frame_count % self.stride != 0:
            self.frame_count += 1
            return metrics

        # Temporal PSNR (frame-to-frame)
        if len(self.frame_history) > 0:
            tpsnr = self._compute_temporal_psnr(
                self.frame_history[-1],
                processed_frame
            )
            self.tpsnr_history.append(tpsnr)
            metrics['tpsnr'] = tpsnr

            # Detect jitter
            if self._is_jitter(tpsnr):
                self.jitter_frames.append(self.frame_count)
                jitter_magnitude = self._compute_jitter_magnitude(tpsnr)
                self.jitter_magnitudes.append(jitter_magnitude)
                metrics['jitter_detected'] = True
                metrics['jitter_magnitude'] = jitter_magnitude

                print(f"⚠️  Jitter detected at frame {self.frame_count}: "
                      f"tPSNR={tpsnr:.2f} dB (drop: {jitter_magnitude:.2f} dB)")

        # Add to frame history
        self.frame_history.append(processed_frame.copy())

        # Increment counter
        self.frame_count += 1

        return metrics

    def _compute_temporal_psnr(
        self,
        frame1: numpy.ndarray,
        frame2: numpy.ndarray
    ) -> float:
        """
        Compute temporal PSNR between consecutive frames.

        Higher tPSNR = more stable (less change between frames).
        """
        # Center crop for face region focus
        h, w = frame1.shape[:2]
        crop_h, crop_w = self.crop_size

        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2

        # Ensure crop is within bounds
        y2 = min(y1 + crop_h, h)
        x2 = min(x1 + crop_w, w)

        frame1_crop = frame1[y1:y2, x1:x2]
        frame2_crop = frame2[y1:y2, x1:x2]

        return self._compute_psnr(frame1_crop, frame2_crop)

    def _compute_psnr(
        self,
        frame1: numpy.ndarray,
        frame2: numpy.ndarray
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two frames.
        """
        # Compute MSE
        mse = numpy.mean((frame1.astype(float) - frame2.astype(float)) ** 2)

        if mse == 0:
            return float('inf')

        # Compute PSNR
        max_pixel = 255.0
        psnr = 20 * numpy.log10(max_pixel / numpy.sqrt(mse))

        return psnr

    def _is_jitter(self, tpsnr: float) -> bool:
        """
        Detect if current tPSNR indicates jitter.

        Jitter = sudden drop in tPSNR compared to recent average.
        """
        if len(self.tpsnr_history) < 5:
            return False

        # Recent average (last 5 frames)
        recent_mean = numpy.mean(self.tpsnr_history[-5:])

        # Check for significant drop
        drop = recent_mean - tpsnr

        return drop > self.jitter_threshold

    def _compute_jitter_magnitude(self, tpsnr: float) -> float:
        """
        Compute magnitude of jitter event.
        """
        recent_mean = numpy.mean(self.tpsnr_history[-5:])
        return recent_mean - tpsnr

    def get_summary(self) -> Dict[str, any]:
        """
        Get summary statistics for all processed frames.

        Returns:
            Dictionary of summary statistics
        """
        if not self.tpsnr_history:
            return {
                'frames_monitored': 0,
                'monitoring_enabled': self.enable
            }

        tpsnr_array = numpy.array(self.tpsnr_history)

        summary = {
            # Basic info
            'frames_monitored': len(self.tpsnr_history),
            'total_frames': self.frame_count,
            'monitoring_enabled': self.enable,

            # Temporal PSNR statistics
            'tpsnr': {
                'mean': float(numpy.mean(tpsnr_array)),
                'std': float(numpy.std(tpsnr_array)),
                'min': float(numpy.min(tpsnr_array)),
                'max': float(numpy.max(tpsnr_array)),
                'p05': float(numpy.percentile(tpsnr_array, 5)),
                'p50': float(numpy.percentile(tpsnr_array, 50)),
                'p95': float(numpy.percentile(tpsnr_array, 95))
            },

            # Jitter statistics
            'jitter': {
                'count': len(self.jitter_frames),
                'percentage': (len(self.jitter_frames) / len(self.tpsnr_history)) * 100 if self.tpsnr_history else 0,
                'frame_numbers': self.jitter_frames,
                'avg_magnitude': float(numpy.mean(self.jitter_magnitudes)) if self.jitter_magnitudes else 0,
                'max_magnitude': float(numpy.max(self.jitter_magnitudes)) if self.jitter_magnitudes else 0
            }
        }

        return summary

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("Quality Monitoring Summary")
        print("="*60)

        print(f"\n📊 Frames Monitored: {summary['frames_monitored']} / {summary['total_frames']}")

        if 'tpsnr' in summary:
            print(f"\n🎬 Temporal PSNR (Frame-to-Frame Stability):")
            print(f"   Mean:   {summary['tpsnr']['mean']:.2f} dB")
            print(f"   Std:    {summary['tpsnr']['std']:.2f} dB")
            print(f"   Min:    {summary['tpsnr']['min']:.2f} dB")
            print(f"   P05:    {summary['tpsnr']['p05']:.2f} dB")
            print(f"   Median: {summary['tpsnr']['p50']:.2f} dB")
            print(f"   P95:    {summary['tpsnr']['p95']:.2f} dB")

        if 'jitter' in summary:
            jitter_pct = summary['jitter']['percentage']
            if jitter_pct > 0:
                print(f"\n⚠️  Jitter Events:")
                print(f"   Count:      {summary['jitter']['count']} frames ({jitter_pct:.1f}%)")
                print(f"   Avg Drop:   {summary['jitter']['avg_magnitude']:.2f} dB")
                print(f"   Max Drop:   {summary['jitter']['max_magnitude']:.2f} dB")

                if summary['jitter']['count'] <= 10:
                    print(f"   Frames:     {summary['jitter']['frame_numbers']}")
            else:
                print(f"\n✅ No Jitter Detected!")

        print("\n" + "="*60 + "\n")

    def save_report(self, output_path: str):
        """
        Save detailed report to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        summary = self.get_summary()

        # Add full time series data
        summary['time_series'] = {
            'tpsnr': [float(x) for x in self.tpsnr_history]
        }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Quality report saved to: {output_path}")

    def reset(self):
        """Reset all monitoring data."""
        self.frame_history.clear()
        self.tpsnr_history.clear()
        self.color_variance_history.clear()
        self.edge_sharpness_history.clear()
        self.jitter_frames.clear()
        self.jitter_magnitudes.clear()
        self.frame_count = 0
