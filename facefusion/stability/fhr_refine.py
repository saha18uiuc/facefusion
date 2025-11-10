"""
Fractional Heatmap Regression (FHR) for sub-pixel landmark refinement.
Reduces quantization jitter in high-resolution video face alignment.

Based on research: "Towards Highly Accurate and Stable Face Alignment for High-Resolution Videos"
https://arxiv.org/abs/1811.00342
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TinyFHR(nn.Module if TORCH_AVAILABLE else object):
    """
    Ultra-small head that predicts sub-pixel offsets (dx,dy) for each landmark.

    This lightweight model refines integer landmark positions to sub-pixel precision,
    reducing frame-to-frame jitter caused by quantization.

    Input: cropped face tensor (B,C,H,W) in range [0,1]
    Output: offsets (B, L, 2) in pixels (typically [-0.5,+0.5])
    """

    def __init__(self, num_landmarks: int = 68, in_channels: int = 3):
        """
        Initialize FHR refinement head.

        Args:
            num_landmarks: Number of facial landmarks (default: 68)
            in_channels: Number of input channels (default: 3 for RGB)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FHR refinement")

        super().__init__()
        self.num_landmarks = num_landmarks

        # Lightweight feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Offset predictor
        self.fc = nn.Linear(32, num_landmarks * 2)

        # Initialize with small weights for stability
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict sub-pixel offsets for landmarks.

        Args:
            x: Input face crop tensor (B, C, H, W)

        Returns:
            Offset tensor (B, num_landmarks, 2)
        """
        # Extract features
        features = self.conv(x).flatten(1)

        # Predict offsets
        offsets = self.fc(features).view(-1, self.num_landmarks, 2)

        # Clamp to reasonable range [-1, 1] pixels
        offsets = torch.clamp(offsets, -1.0, 1.0)

        return offsets


def refine_landmarks_to_subpixel(
    face_crop: np.ndarray,
    landmarks_int: np.ndarray,
    model: Optional[TinyFHR] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Refine integer landmark coordinates to sub-pixel precision.

    Args:
        face_crop: Face crop image (H, W, 3) in range [0, 255], uint8
        landmarks_int: Integer landmark coordinates (L, 2)
        model: FHR model instance (will create if None)
        device: Device for inference ('cuda' or 'cpu')

    Returns:
        Refined landmarks with sub-pixel precision (L, 2)
    """
    if not TORCH_AVAILABLE:
        # Fallback: return integer landmarks without refinement
        return landmarks_int.astype(np.float32)

    # Create model if not provided
    if model is None:
        model = TinyFHR(num_landmarks=landmarks_int.shape[0])
        model = model.eval().to(device)

    # Prepare input tensor
    face_tensor = torch.from_numpy(face_crop).float() / 255.0
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    face_tensor = face_tensor.to(device)

    # Get offsets
    with torch.inference_mode():
        offsets = model(face_tensor)  # (1, L, 2)

    # Convert to numpy and apply offsets
    offsets_np = offsets.cpu().numpy()[0]  # (L, 2)
    refined_landmarks = landmarks_int.astype(np.float32) + offsets_np

    return refined_landmarks


def refine_landmarks_batch(
    face_crops: np.ndarray,
    landmarks_batch: np.ndarray,
    model: Optional[TinyFHR] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Batch refinement of landmarks for multiple faces.

    Args:
        face_crops: Face crop images (B, H, W, 3) in range [0, 255], uint8
        landmarks_batch: Integer landmark coordinates (B, L, 2)
        model: FHR model instance (will create if None)
        device: Device for inference ('cuda' or 'cpu')

    Returns:
        Refined landmarks with sub-pixel precision (B, L, 2)
    """
    if not TORCH_AVAILABLE:
        # Fallback: return integer landmarks without refinement
        return landmarks_batch.astype(np.float32)

    batch_size = face_crops.shape[0]
    num_landmarks = landmarks_batch.shape[1]

    # Create model if not provided
    if model is None:
        model = TinyFHR(num_landmarks=num_landmarks)
        model = model.eval().to(device)

    # Prepare input tensor
    face_tensor = torch.from_numpy(face_crops).float() / 255.0
    face_tensor = face_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)
    face_tensor = face_tensor.to(device)

    # Get offsets
    with torch.inference_mode():
        offsets = model(face_tensor)  # (B, L, 2)

    # Convert to numpy and apply offsets
    offsets_np = offsets.cpu().numpy()  # (B, L, 2)
    refined_landmarks = landmarks_batch.astype(np.float32) + offsets_np

    return refined_landmarks


def enable_fhr_for_landmarks(enable: bool = True) -> bool:
    """
    Check if FHR can be enabled based on PyTorch availability.

    Args:
        enable: Whether to enable FHR

    Returns:
        True if FHR can be enabled, False otherwise
    """
    return enable and TORCH_AVAILABLE
