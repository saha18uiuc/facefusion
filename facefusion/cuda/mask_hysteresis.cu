/**
 * Mask Edge Hysteresis CUDA Kernel
 *
 * Prevents single-frame mask edge flicker by applying in/out thresholds.
 * Only pixels that move beyond the threshold band are updated.
 *
 * This eliminates 1-2 pixel seam jumps that cause visible tPSNR drops.
 */

#include <cuda_runtime.h>

/**
 * Mask edge hysteresis kernel (2D grid).
 *
 * @param alpha Current alpha mask (unsigned char, 0-255)
 * @param alphaPrev Previous alpha mask
 * @param alphaOut Output alpha mask
 * @param W Mask width
 * @param H Mask height
 * @param stridePx Stride in pixels
 * @param Tin Threshold for inward movement (positive)
 * @param Tout Threshold for outward movement (positive)
 *
 * Movement is considered significant only if:
 * - alpha - alphaPrev < -Tin (moved in)
 * - alpha - alphaPrev > +Tout (moved out)
 * Otherwise, alphaPrev is kept (hysteresis).
 */
template<int BLOCK_X, int BLOCK_Y>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y)
void mask_edge_hysteresis_2d(
    const unsigned char* __restrict__ alpha,
    const unsigned char* __restrict__ alphaPrev,
    unsigned char* __restrict__ alphaOut,
    int W, int H, int stridePx,
    int Tin, int Tout
)
{
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (x >= W || y >= H) return;

    const int idx = y * stridePx + x;

    const int a = (int)alpha[idx];
    const int ap = (int)alphaPrev[idx];
    const int d = a - ap;

    // Determine if movement crosses threshold
    const bool moved_in = (d < -Tin);
    const bool moved_out = (d > Tout);
    const bool stable = !(moved_in || moved_out);

    // Keep previous value if stable (hysteresis), otherwise accept new value
    alphaOut[idx] = stable ? (unsigned char)ap : (unsigned char)a;
}

/**
 * Mask edge hysteresis kernel (1D grid, for full-image processing).
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void mask_edge_hysteresis_1d(
    const unsigned char* __restrict__ alpha,
    const unsigned char* __restrict__ alphaPrev,
    unsigned char* __restrict__ alphaOut,
    int W, int H, int stridePx,
    int Tin, int Tout
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    const int a = (int)alpha[idx];
    const int ap = (int)alphaPrev[idx];
    const int d = a - ap;

    const bool moved_in = (d < -Tin);
    const bool moved_out = (d > Tout);
    const bool stable = !(moved_in || moved_out);

    alphaOut[idx] = stable ? (unsigned char)ap : (unsigned char)a;
}

/**
 * Mask edge hysteresis with region-of-interest (only process seam ring).
 *
 * This variant processes only pixels within a specified ring around the mask edge,
 * which is much faster when the seam band is narrow.
 *
 * @param indices Flat indices of pixels in the seam ring
 * @param numIndices Number of indices
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void mask_edge_hysteresis_roi(
    const unsigned char* __restrict__ alpha,
    const unsigned char* __restrict__ alphaPrev,
    unsigned char* __restrict__ alphaOut,
    const int* __restrict__ indices,
    int numIndices,
    int Tin, int Tout
)
{
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i >= numIndices) return;

    const int idx = indices[i];

    const int a = (int)alpha[idx];
    const int ap = (int)alphaPrev[idx];
    const int d = a - ap;

    const bool moved_in = (d < -Tin);
    const bool moved_out = (d > Tout);
    const bool stable = !(moved_in || moved_out);

    alphaOut[idx] = stable ? (unsigned char)ap : (unsigned char)a;
}

/**
 * Helper: identify seam ring indices (pixels where alpha is in [bandLo, bandHi]).
 *
 * This can be called once per face to build the index list, then reused for
 * hysteresis on each frame.
 *
 * @param alpha Input alpha mask
 * @param indices Output index buffer (preallocated, size W*H)
 * @param count Output counter (atomically incremented)
 * @param W Mask width
 * @param H Mask height
 * @param stridePx Stride
 * @param bandLo Lower band threshold
 * @param bandHi Upper band threshold
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void identify_seam_ring(
    const unsigned char* __restrict__ alpha,
    int* __restrict__ indices,
    int* __restrict__ count,
    int W, int H, int stridePx,
    int bandLo, int bandHi
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    const int a = (int)alpha[idx];

    if (a >= bandLo && a <= bandHi) {
        // This pixel is in the seam band - add to list
        int pos = atomicAdd(count, 1);
        indices[pos] = idx;
    }
}

// Explicit template instantiations
template __global__ void mask_edge_hysteresis_2d<16, 16>(
    const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int);
template __global__ void mask_edge_hysteresis_2d<32, 8>(
    const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int);

template __global__ void mask_edge_hysteresis_1d<256>(
    const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int);
template __global__ void mask_edge_hysteresis_1d<512>(
    const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int);

template __global__ void mask_edge_hysteresis_roi<256>(
    const unsigned char*, const unsigned char*, unsigned char*, const int*, int, int, int);
template __global__ void mask_edge_hysteresis_roi<512>(
    const unsigned char*, const unsigned char*, unsigned char*, const int*, int, int, int);

template __global__ void identify_seam_ring<256>(
    const unsigned char*, int*, int*, int, int, int, int, int);
template __global__ void identify_seam_ring<512>(
    const unsigned char*, int*, int*, int, int, int, int, int);
