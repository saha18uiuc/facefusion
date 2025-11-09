"""
Quality Monitoring Module for FaceFusion 3.5.0
Real-time tracking of temporal PSNR, jitter detection, and quality metrics.

Usage:
    from quality_monitor import QualityMonitor
    
    monitor = QualityMonitor()
    for frame in processed_frames:
        monitor.monitor_frame(frame)
    
    stats = monitor.get_summary()
"""

import cv2
import numpy as np
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
        processed_frame: np.ndarray,
        original_frame: Optional[np.ndarray] = None
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
        
        # Color consistency
        color_variance = self._compute_color_variance(processed_frame)
        self.color_variance_history.append(color_variance)
        metrics['color_variance'] = color_variance
        
        # Edge sharpness (indicator of quality)
        edge_sharpness = self._compute_edge_sharpness(processed_frame)
        self.edge_sharpness_history.append(edge_sharpness)
        metrics['edge_sharpness'] = edge_sharpness
        
        # If original provided, compute PSNR vs original
        if original_frame is not None:
            original_psnr = self._compute_psnr(original_frame, processed_frame)
            metrics['psnr_vs_original'] = original_psnr
        
        # Add to frame history
        self.frame_history.append(processed_frame.copy())
        
        # Increment counter
        self.frame_count += 1
        
        return metrics
    
    def _compute_temporal_psnr(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
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
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two frames.
        """
        # Compute MSE
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Compute PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def _is_jitter(self, tpsnr: float) -> bool:
        """
        Detect if current tPSNR indicates jitter.
        
        Jitter = sudden drop in tPSNR compared to recent average.
        """
        if len(self.tpsnr_history) < 5:
            return False
        
        # Recent average (last 5 frames)
        recent_mean = np.mean(self.tpsnr_history[-5:])
        
        # Check for significant drop
        drop = recent_mean - tpsnr
        
        return drop > self.jitter_threshold
    
    def _compute_jitter_magnitude(self, tpsnr: float) -> float:
        """
        Compute magnitude of jitter event.
        """
        recent_mean = np.mean(self.tpsnr_history[-5:])
        return recent_mean - tpsnr
    
    def _compute_color_variance(self, frame: np.ndarray) -> float:
        """
        Compute color variance in frame (indicates color consistency).
        
        Lower variance = more consistent colors.
        """
        # Convert to LAB for perceptual color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Compute variance in each channel
        l_var = np.var(lab[:, :, 0])
        a_var = np.var(lab[:, :, 1])
        b_var = np.var(lab[:, :, 2])
        
        # Average variance across channels
        avg_variance = (l_var + a_var + b_var) / 3
        
        return float(avg_variance)
    
    def _compute_edge_sharpness(self, frame: np.ndarray) -> float:
        """
        Compute edge sharpness (indicates image quality).
        
        Higher values = sharper edges.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian (edge detector)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Variance of Laplacian indicates sharpness
        sharpness = np.var(laplacian)
        
        return float(sharpness)
    
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
        
        tpsnr_array = np.array(self.tpsnr_history)
        
        summary = {
            # Basic info
            'frames_monitored': len(self.tpsnr_history),
            'total_frames': self.frame_count,
            'monitoring_enabled': self.enable,
            
            # Temporal PSNR statistics
            'tpsnr': {
                'mean': float(np.mean(tpsnr_array)),
                'std': float(np.std(tpsnr_array)),
                'min': float(np.min(tpsnr_array)),
                'max': float(np.max(tpsnr_array)),
                'p05': float(np.percentile(tpsnr_array, 5)),
                'p50': float(np.percentile(tpsnr_array, 50)),
                'p95': float(np.percentile(tpsnr_array, 95))
            },
            
            # Jitter statistics
            'jitter': {
                'count': len(self.jitter_frames),
                'percentage': (len(self.jitter_frames) / len(self.tpsnr_history)) * 100,
                'frame_numbers': self.jitter_frames,
                'avg_magnitude': float(np.mean(self.jitter_magnitudes)) if self.jitter_magnitudes else 0,
                'max_magnitude': float(np.max(self.jitter_magnitudes)) if self.jitter_magnitudes else 0
            },
            
            # Color consistency
            'color_variance': {
                'mean': float(np.mean(self.color_variance_history)),
                'std': float(np.std(self.color_variance_history))
            } if self.color_variance_history else {},
            
            # Edge sharpness
            'edge_sharpness': {
                'mean': float(np.mean(self.edge_sharpness_history)),
                'std': float(np.std(self.edge_sharpness_history))
            } if self.edge_sharpness_history else {}
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
        
        if 'color_variance' in summary and summary['color_variance']:
            print(f"\n🎨 Color Consistency:")
            print(f"   Variance: {summary['color_variance']['mean']:.2f} "
                  f"(±{summary['color_variance']['std']:.2f})")
        
        if 'edge_sharpness' in summary and summary['edge_sharpness']:
            print(f"\n🔍 Edge Sharpness:")
            print(f"   Mean: {summary['edge_sharpness']['mean']:.2f}")
        
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
            'tpsnr': [float(x) for x in self.tpsnr_history],
            'color_variance': [float(x) for x in self.color_variance_history],
            'edge_sharpness': [float(x) for x in self.edge_sharpness_history]
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Quality report saved to: {output_path}")
    
    def compare_with_baseline(
        self,
        baseline_stats: Dict[str, any]
    ) -> Dict[str, float]:
        """
        Compare current stats with baseline (e.g., base 3.5.0).
        
        Args:
            baseline_stats: Summary statistics from baseline run
            
        Returns:
            Dictionary of deltas (current - baseline)
        """
        current_stats = self.get_summary()
        
        deltas = {
            'tpsnr_mean_delta': (
                current_stats['tpsnr']['mean'] -
                baseline_stats['tpsnr']['mean']
            ),
            'tpsnr_std_delta': (
                current_stats['tpsnr']['std'] -
                baseline_stats['tpsnr']['std']
            ),
            'tpsnr_min_delta': (
                current_stats['tpsnr']['min'] -
                baseline_stats['tpsnr']['min']
            ),
            'jitter_count_delta': (
                current_stats['jitter']['count'] -
                baseline_stats['jitter']['count']
            ),
            'jitter_percentage_delta': (
                current_stats['jitter']['percentage'] -
                baseline_stats['jitter']['percentage']
            )
        }
        
        return deltas
    
    def reset(self):
        """Reset all monitoring data."""
        self.frame_history.clear()
        self.tpsnr_history.clear()
        self.color_variance_history.clear()
        self.edge_sharpness_history.clear()
        self.jitter_frames.clear()
        self.jitter_magnitudes.clear()
        self.frame_count = 0


class ComparisonMonitor:
    """
    Monitor for comparing two versions side-by-side.
    
    Useful for comparing your optimized version vs base 3.5.0.
    """
    
    def __init__(self, crop_size: Tuple[int, int] = (480, 480)):
        """
        Initialize comparison monitor.
        
        Args:
            crop_size: Size of center crop for comparisons
        """
        self.crop_size = crop_size
        self.tpsnr_deltas = []
        self.psnr_deltas = []
        self.frame_count = 0
        
    def compare_frames(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        prev_frame_a: Optional[np.ndarray] = None,
        prev_frame_b: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compare two frames (e.g., optimized vs baseline).
        
        Args:
            frame_a: First frame (e.g., optimized version)
            frame_b: Second frame (e.g., baseline)
            prev_frame_a: Previous frame from version A (for tPSNR)
            prev_frame_b: Previous frame from version B (for tPSNR)
            
        Returns:
            Comparison metrics
        """
        metrics = {}
        
        # Direct PSNR between A and B
        psnr_ab = self._compute_psnr(frame_a, frame_b)
        metrics['psnr_between'] = psnr_ab
        
        # Temporal PSNR for each version
        if prev_frame_a is not None:
            tpsnr_a = self._compute_temporal_psnr(prev_frame_a, frame_a)
            metrics['tpsnr_a'] = tpsnr_a
            
        if prev_frame_b is not None:
            tpsnr_b = self._compute_temporal_psnr(prev_frame_b, frame_b)
            metrics['tpsnr_b'] = tpsnr_b
            
        # Compute deltas
        if 'tpsnr_a' in metrics and 'tpsnr_b' in metrics:
            delta_tpsnr = metrics['tpsnr_a'] - metrics['tpsnr_b']
            metrics['delta_tpsnr'] = delta_tpsnr
            self.tpsnr_deltas.append(delta_tpsnr)
            
            # Flag if A is worse than B
            if delta_tpsnr < -2.0:  # More than 2dB worse
                metrics['a_worse_than_b'] = True
                print(f"⚠️  Frame {self.frame_count}: Version A worse by {abs(delta_tpsnr):.2f} dB")
        
        self.frame_count += 1
        
        return metrics
    
    def _compute_psnr(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute PSNR between two frames."""
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def _compute_temporal_psnr(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """Compute temporal PSNR with center crop."""
        h, w = frame1.shape[:2]
        crop_h, crop_w = self.crop_size
        
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        y2 = min(y1 + crop_h, h)
        x2 = min(x1 + crop_w, w)
        
        crop1 = frame1[y1:y2, x1:x2]
        crop2 = frame2[y1:y2, x1:x2]
        
        return self._compute_psnr(crop1, crop2)
    
    def get_summary(self) -> Dict[str, any]:
        """Get comparison summary."""
        if not self.tpsnr_deltas:
            return {}
        
        deltas = np.array(self.tpsnr_deltas)
        
        return {
            'frames_compared': len(self.tpsnr_deltas),
            'delta_tpsnr': {
                'mean': float(np.mean(deltas)),
                'std': float(np.std(deltas)),
                'min': float(np.min(deltas)),
                'max': float(np.max(deltas)),
                'p05': float(np.percentile(deltas, 5)),
                'p50': float(np.percentile(deltas, 50)),
                'p95': float(np.percentile(deltas, 95))
            },
            'frames_a_worse': int(np.sum(deltas < -2.0)),
            'frames_a_better': int(np.sum(deltas > 2.0))
        }


# Example usage
if __name__ == "__main__":
    # Example: Integrate into FaceFusion processing loop
    
    # In facefusion/core.py or main processing loop:
    #
    # from quality_monitor import QualityMonitor
    #
    # # Initialize monitor
    # monitor = QualityMonitor(
    #     enable_monitoring=True,
    #     crop_size=(480, 480),
    #     stride=3,  # Check every 3rd frame for efficiency
    #     jitter_threshold=3.0
    # )
    #
    # # Process frames
    # for frame in video_frames:
    #     processed_frame = process_frame(frame)
    #     monitor.monitor_frame(processed_frame)
    #
    # # Print summary
    # monitor.print_summary()
    #
    # # Save detailed report
    # monitor.save_report('quality_report.json')
    
    print("Quality Monitoring module ready for integration")
    print("Track tPSNR, detect jitter, validate improvements in real-time")
