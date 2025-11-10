"""
3D Pose Filter with Exponential Moving Average (EMA) and Deadband.
Provides temporal smoothing for face pose to reduce jitter in video processing.

Based on research: "Stabilized Temporal 3D Face Alignment Using Landmark Displacement"
https://www.mdpi.com/2079-9292/12/17/3735
"""

import numpy as np
from typing import Tuple, Optional


class PoseFilter:
    """
    Temporal filter for 3D face pose (yaw, pitch, roll, scale) using EMA with deadband.

    The filter uses:
    1. Exponential Moving Average (EMA) for smooth temporal tracking
    2. Bias correction for accurate warmup
    3. Deadband to prevent micro-jitter when pose is stable

    This approach is particularly effective for late-video stability where
    motion slows or stops.
    """

    def __init__(
        self,
        beta: float = 0.85,
        deadband: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.002)
    ):
        """
        Initialize pose filter.

        Args:
            beta: EMA smoothing factor (0-1). Higher = more smoothing
                  0.85 provides good balance between responsiveness and stability
            deadband: Threshold tuple (yaw, pitch, roll, scale) in degrees/units.
                     Changes below threshold are suppressed.
        """
        self.beta = beta
        self.deadband = np.array(deadband, dtype=np.float32)
        self.m = None  # EMA state
        self.t = 0  # Time step
        self.last_output = None  # Last output for deadband

    def update(self, pose: np.ndarray) -> np.ndarray:
        """
        Update filter with new pose measurement.

        Args:
            pose: Current pose [yaw, pitch, roll, scale] as numpy array

        Returns:
            Filtered pose [yaw, pitch, roll, scale]
        """
        pose = np.asarray(pose, dtype=np.float32)
        self.t += 1

        # Initialize on first update
        if self.m is None:
            self.m = pose.copy()
            self.last_output = pose.copy()
            return pose.copy()

        # EMA update
        self.m = self.beta * self.m + (1.0 - self.beta) * pose

        # Bias correction (important during warmup)
        bias_corrected = self.m / (1.0 - (self.beta ** self.t))

        # Deadband: suppress small changes to prevent micro-jitter
        delta = np.abs(pose - bias_corrected)
        if np.all(delta < self.deadband):
            # Change is within deadband - keep last output
            return self.last_output.copy()

        # Significant change detected - update output
        self.last_output = bias_corrected.copy()
        return bias_corrected

    def reset(self):
        """Reset filter state."""
        self.m = None
        self.t = 0
        self.last_output = None


class Multi3DPoseFilter:
    """
    Manages multiple pose filters for tracking multiple faces.
    Each face gets its own independent filter instance.
    """

    def __init__(
        self,
        beta: float = 0.85,
        deadband: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.002)
    ):
        """
        Initialize multi-face pose filter manager.

        Args:
            beta: EMA smoothing factor
            deadband: Threshold tuple for pose changes
        """
        self.beta = beta
        self.deadband = deadband
        self.filters = {}  # face_id -> PoseFilter

    def update(self, face_id: int, pose: np.ndarray) -> np.ndarray:
        """
        Update pose for a specific face.

        Args:
            face_id: Unique identifier for the face
            pose: Current pose [yaw, pitch, roll, scale]

        Returns:
            Filtered pose
        """
        if face_id not in self.filters:
            self.filters[face_id] = PoseFilter(self.beta, self.deadband)

        return self.filters[face_id].update(pose)

    def reset(self, face_id: Optional[int] = None):
        """
        Reset filter state for a face or all faces.

        Args:
            face_id: Face to reset, or None to reset all
        """
        if face_id is None:
            self.filters.clear()
        elif face_id in self.filters:
            del self.filters[face_id]


def extract_pose_from_landmarks(landmarks_68: np.ndarray) -> np.ndarray:
    """
    Extract simplified 3D pose (yaw, pitch, roll, scale) from 68 landmarks.

    This is a lightweight approximation suitable for filtering.

    Args:
        landmarks_68: Facial landmarks (68, 2)

    Returns:
        Pose array [yaw, pitch, roll, scale]
    """
    # Calculate face orientation from outer landmarks
    left_point = landmarks_68[0]  # Left face edge
    right_point = landmarks_68[16]  # Right face edge
    nose_tip = landmarks_68[30]  # Nose tip
    chin = landmarks_68[8]  # Chin

    # Calculate yaw (horizontal rotation) from face width asymmetry
    nose_x = nose_tip[0]
    face_center_x = (left_point[0] + right_point[0]) / 2
    face_width = right_point[0] - left_point[0]
    yaw = (nose_x - face_center_x) / (face_width / 2) * 30.0  # Scale to ~degrees

    # Calculate pitch (vertical tilt) from nose-chin distance
    nose_y = nose_tip[1]
    chin_y = chin[1]
    face_height = np.linalg.norm(landmarks_68[8] - landmarks_68[27])
    pitch = (chin_y - nose_y) / face_height * 20.0  # Scale to ~degrees

    # Calculate roll (head tilt) from eye line angle
    left_eye = np.mean(landmarks_68[36:42], axis=0)
    right_eye = np.mean(landmarks_68[42:48], axis=0)
    eye_vec = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

    # Calculate scale from face size
    scale = np.linalg.norm(right_point - left_point) / 100.0  # Normalize by typical width

    return np.array([yaw, pitch, roll, scale], dtype=np.float32)


def apply_pose_to_affine_matrix(
    base_affine: np.ndarray,
    filtered_pose: np.ndarray,
    original_pose: np.ndarray,
    image_center: Tuple[float, float]
) -> np.ndarray:
    """
    Apply filtered pose to affine transformation matrix.

    This modulates the affine matrix based on the pose filtering,
    allowing smooth geometric transformations.

    Args:
        base_affine: Base 2x3 affine matrix
        filtered_pose: Filtered pose [yaw, pitch, roll, scale]
        original_pose: Original unfiltered pose
        image_center: Center point for rotation (cx, cy)

    Returns:
        Modified 2x3 affine matrix
    """
    # Calculate pose delta
    pose_delta = filtered_pose - original_pose

    # Extract components
    roll_delta = pose_delta[2]  # Only apply roll to 2D affine
    scale_delta = pose_delta[3]

    # Create rotation matrix for roll delta
    roll_rad = np.radians(roll_delta)
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)

    cx, cy = image_center

    # Build correction matrix
    # Translation to origin
    T1 = np.array([[1, 0, -cx], [0, 1, -cy]], dtype=np.float32)

    # Rotation and scale
    R = np.array([
        [cos_r * (1 + scale_delta * 0.1), -sin_r, 0],
        [sin_r, cos_r * (1 + scale_delta * 0.1), 0]
    ], dtype=np.float32)

    # Translation back
    T2 = np.array([[1, 0, cx], [0, 1, cy]], dtype=np.float32)

    # Combine transformations
    # Convert 2x3 matrices to 3x3 for multiplication
    T1_3x3 = np.vstack([T1, [0, 0, 1]])
    R_3x3 = np.vstack([R, [0, 0, 1]])
    T2_3x3 = np.vstack([T2, [0, 0, 1]])
    base_3x3 = np.vstack([base_affine, [0, 0, 1]])

    # Apply: base_affine * T2 * R * T1
    result_3x3 = base_3x3 @ T2_3x3 @ R_3x3 @ T1_3x3

    # Convert back to 2x3
    return result_3x3[:2, :]
