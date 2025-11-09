"""
Enhanced Face Blending Module for FaceFusion 3.5.0
Reduces color warping and boundary artifacts through advanced blending techniques.

Usage:
    from facefusion.processors.enhanced_blending import EnhancedBlender

    blender = EnhancedBlender(method='multiband')
    result = blender.blend_face(swapped_face, target_frame, mask, center)
"""

import cv2
import numpy
from typing import Tuple, Optional, Literal
from enum import Enum


class BlendingMethod(Enum):
    """Available blending methods with quality/performance trade-offs."""
    BASIC = "basic"           # Fast, lower quality
    SEAMLESS = "seamless"     # Good balance
    MULTIBAND = "multiband"   # Best quality, ~5-7% slower
    POISSON = "poisson"       # Alternative to seamless


class EnhancedBlender:
    """
    Advanced face blending with color correction and multi-scale techniques.

    Addresses color warping issues present in performance-optimized versions
    by matching color statistics and using multi-band blending.
    """

    def __init__(
        self,
        method: str = "multiband",
        enable_color_correction: bool = True,
        pyramid_levels: int = 5,
        feather_amount: int = 15
    ):
        """
        Initialize enhanced blender.

        Args:
            method: Blending method ('basic', 'seamless', 'multiband', 'poisson')
            enable_color_correction: Apply color transfer before blending
            pyramid_levels: Number of pyramid levels for multiband blending
            feather_amount: Pixel amount for mask feathering
        """
        self.method = BlendingMethod(method)
        self.enable_color_correction = enable_color_correction
        self.pyramid_levels = pyramid_levels
        self.feather_amount = feather_amount

    def blend_face(
        self,
        swapped_face: numpy.ndarray,
        target_frame: numpy.ndarray,
        mask: numpy.ndarray,
        center: Optional[Tuple[int, int]] = None,
        landmarks: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Blend swapped face into target frame with advanced techniques.

        Args:
            swapped_face: Face-swapped region
            target_frame: Original target frame
            mask: Binary mask of face region
            center: Center point of face (x, y)
            landmarks: Optional face landmarks for refined masking

        Returns:
            Blended result frame
        """
        # Create enhanced mask with proper feathering
        enhanced_mask = self._create_enhanced_mask(
            mask,
            landmarks,
            target_frame.shape[:2]
        )

        # Apply color correction if enabled
        if self.enable_color_correction:
            swapped_corrected = self._color_correction(
                swapped_face,
                target_frame,
                enhanced_mask
            )
        else:
            swapped_corrected = swapped_face

        # Apply selected blending method
        if self.method == BlendingMethod.MULTIBAND:
            result = self._multiband_blend(
                swapped_corrected,
                target_frame,
                enhanced_mask
            )
        elif self.method == BlendingMethod.SEAMLESS and center is not None:
            result = self._seamless_clone(
                swapped_corrected,
                target_frame,
                enhanced_mask,
                center
            )
        elif self.method == BlendingMethod.POISSON and center is not None:
            result = self._poisson_blend(
                swapped_corrected,
                target_frame,
                enhanced_mask,
                center
            )
        else:  # BASIC or fallback
            result = self._basic_blend(
                swapped_corrected,
                target_frame,
                enhanced_mask
            )

        return result

    def _create_enhanced_mask(
        self,
        base_mask: numpy.ndarray,
        landmarks: Optional[numpy.ndarray],
        frame_shape: Tuple[int, int]
    ) -> numpy.ndarray:
        """
        Create high-quality face mask with smooth feathering.

        Critical for preventing visible seams and jitter at boundaries.
        """
        # Ensure mask is proper format
        if len(base_mask.shape) == 3:
            mask = cv2.cvtColor(base_mask, cv2.COLOR_BGR2GRAY)
        else:
            mask = base_mask.copy()

        # Ensure uint8 format
        if mask.dtype != numpy.uint8:
            mask = (mask * 255).astype(numpy.uint8)

        # If landmarks provided, refine mask to face boundary
        if landmarks is not None:
            # Create convex hull from landmarks
            hull = cv2.convexHull(landmarks.astype(numpy.int32))
            refined_mask = numpy.zeros(frame_shape, dtype=numpy.uint8)
            cv2.fillConvexPoly(refined_mask, hull, 255)

            # Combine with base mask
            mask = cv2.bitwise_and(mask, refined_mask)

        # Apply multiple blur passes for smooth feathering
        # This is crucial for temporal stability
        for blur_iteration in range(3):
            mask = cv2.GaussianBlur(
                mask,
                (self.feather_amount, self.feather_amount),
                0
            )

        # Erode slightly and blur again for ultra-soft edge
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.feather_amount, self.feather_amount)
        )
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(
            mask,
            (self.feather_amount * 2 + 1, self.feather_amount * 2 + 1),
            0
        )

        # Normalize to 0-1 range for blending
        mask_normalized = mask.astype(numpy.float32) / 255.0

        # Expand to 3 channels for color blending
        mask_normalized = numpy.stack([mask_normalized] * 3, axis=-1)

        return mask_normalized

    def _color_correction(
        self,
        source: numpy.ndarray,
        target: numpy.ndarray,
        mask: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Transfer color statistics from target to source region.

        Reduces color warping by matching source face colors to target frame.
        """
        # Convert to LAB color space (perceptually uniform)
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(numpy.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(numpy.float32)

        # Create binary mask for statistics calculation
        mask_binary = (mask > 0.5).astype(numpy.uint8)
        if len(mask_binary.shape) == 3:
            mask_binary = mask_binary[:, :, 0]

        # Calculate mean and std for each LAB channel in masked region
        result = source_lab.copy()

        for channel in range(3):
            # Source statistics
            source_channel = source_lab[:, :, channel]
            source_masked = source_channel[mask_binary > 0]

            if len(source_masked) == 0:
                continue

            source_mean = numpy.mean(source_masked)
            source_std = numpy.std(source_masked) + 1e-6  # Avoid division by zero

            # Target statistics
            target_channel = target_lab[:, :, channel]
            target_masked = target_channel[mask_binary > 0]

            if len(target_masked) == 0:
                continue

            target_mean = numpy.mean(target_masked)
            target_std = numpy.std(target_masked)

            # Apply color transfer: scale and shift
            result[:, :, channel] = (
                (source_channel - source_mean) * (target_std / source_std) +
                target_mean
            )

        # Clip values to valid LAB range
        result[:, :, 0] = numpy.clip(result[:, :, 0], 0, 100)   # L channel
        result[:, :, 1] = numpy.clip(result[:, :, 1], -127, 127)  # A channel
        result[:, :, 2] = numpy.clip(result[:, :, 2], -127, 127)  # B channel

        # Convert back to BGR
        result = result.astype(numpy.uint8)
        corrected = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return corrected

    def _multiband_blend(
        self,
        source: numpy.ndarray,
        target: numpy.ndarray,
        mask: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Multi-band (Laplacian pyramid) blending for seamless integration.

        Best quality method - eliminates visible seams and color discontinuities.
        """
        # Ensure mask is single channel for pyramid
        if len(mask.shape) == 3:
            mask_single = mask[:, :, 0]
        else:
            mask_single = mask

        # Create Laplacian pyramids for source and target
        source_pyramid = self._create_laplacian_pyramid(
            source,
            self.pyramid_levels
        )
        target_pyramid = self._create_laplacian_pyramid(
            target,
            self.pyramid_levels
        )

        # Create Gaussian pyramid for mask
        mask_pyramid = self._create_gaussian_pyramid(
            mask_single,
            self.pyramid_levels
        )

        # Blend at each pyramid level
        blended_pyramid = []
        for level in range(self.pyramid_levels):
            # Expand mask to 3 channels for this level
            level_mask = numpy.stack([mask_pyramid[level]] * 3, axis=-1)

            # Blend at this level
            blended_level = (
                source_pyramid[level] * level_mask +
                target_pyramid[level] * (1 - level_mask)
            )
            blended_pyramid.append(blended_level)

        # Reconstruct image from blended pyramid
        result = self._reconstruct_from_laplacian(blended_pyramid)

        # Clip to valid range
        result = numpy.clip(result, 0, 255).astype(numpy.uint8)

        return result

    def _create_laplacian_pyramid(
        self,
        image: numpy.ndarray,
        levels: int
    ) -> list:
        """Create Laplacian pyramid for multi-scale blending."""
        pyramid = []
        current = image.astype(numpy.float32)

        for level in range(levels - 1):
            # Downsample
            down = cv2.pyrDown(current)

            # Upsample back to current size
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))

            # Laplacian = current - upsampled
            laplacian = cv2.subtract(current, up)
            pyramid.append(laplacian)

            current = down

        # Add the final (coarsest) level
        pyramid.append(current)

        return pyramid

    def _create_gaussian_pyramid(
        self,
        image: numpy.ndarray,
        levels: int
    ) -> list:
        """Create Gaussian pyramid for mask."""
        pyramid = [image.copy().astype(numpy.float32)]
        current = image

        for level in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current.astype(numpy.float32))

        return pyramid

    def _reconstruct_from_laplacian(self, pyramid: list) -> numpy.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        current = pyramid[-1]

        for level in range(len(pyramid) - 2, -1, -1):
            # Upsample current level
            current = cv2.pyrUp(
                current,
                dstsize=(pyramid[level].shape[1], pyramid[level].shape[0])
            )

            # Add Laplacian at this level
            current = cv2.add(current, pyramid[level])

        return current

    def _seamless_clone(
        self,
        source: numpy.ndarray,
        target: numpy.ndarray,
        mask: numpy.ndarray,
        center: Tuple[int, int]
    ) -> numpy.ndarray:
        """
        OpenCV seamless cloning (Poisson blending).

        Good balance between quality and performance.
        """
        # Convert mask to 8-bit single channel
        if len(mask.shape) == 3:
            mask_8u = (mask[:, :, 0] * 255).astype(numpy.uint8)
        else:
            mask_8u = (mask * 255).astype(numpy.uint8)

        try:
            result = cv2.seamlessClone(
                source,
                target,
                mask_8u,
                center,
                cv2.NORMAL_CLONE
            )
        except cv2.error:
            # Fallback to basic blend if seamless clone fails
            result = self._basic_blend(source, target, mask)

        return result

    def _poisson_blend(
        self,
        source: numpy.ndarray,
        target: numpy.ndarray,
        mask: numpy.ndarray,
        center: Tuple[int, int]
    ) -> numpy.ndarray:
        """
        Poisson blending with mixed gradients.

        Alternative to seamless clone, sometimes better for texture preservation.
        """
        # Convert mask
        if len(mask.shape) == 3:
            mask_8u = (mask[:, :, 0] * 255).astype(numpy.uint8)
        else:
            mask_8u = (mask * 255).astype(numpy.uint8)

        try:
            result = cv2.seamlessClone(
                source,
                target,
                mask_8u,
                center,
                cv2.MIXED_CLONE  # Mixed gradients preserve more texture
            )
        except cv2.error:
            result = self._basic_blend(source, target, mask)

        return result

    def _basic_blend(
        self,
        source: numpy.ndarray,
        target: numpy.ndarray,
        mask: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Basic alpha blending.

        Fastest method, use only if performance is critical.
        """
        # Ensure mask has 3 channels
        if len(mask.shape) == 2:
            mask = numpy.stack([mask] * 3, axis=-1)

        # Alpha blend
        result = (
            source.astype(numpy.float32) * mask +
            target.astype(numpy.float32) * (1 - mask)
        )

        result = numpy.clip(result, 0, 255).astype(numpy.uint8)

        return result
