/**
 * Linear-Light Color Gain CUDA Kernel
 *
 * Applies color normalization in linear light space (not sRGB) for steadier
 * perceived tone. Prevents gamma pumping artifacts.
 *
 * Gains are computed and filtered on CPU (low-pass + quantized) to avoid
 * micro-pumping; this kernel simply applies them deterministically.
 */

#include <cuda_runtime.h>

/**
 * sRGB to linear light conversion.
 * Standard sRGB EOTF (Electro-Optical Transfer Function).
 */
__device__ inline float srgb_to_linear(float u)
{
    u = fminf(fmaxf(u / 255.0f, 0.0f), 1.0f);
    if (u <= 0.04045f) {
        return u / 12.92f;
    } else {
        return powf((u + 0.055f) / 1.055f, 2.4f);
    }
}

/**
 * Linear light to sRGB conversion.
 * Standard sRGB inverse EOTF.
 */
__device__ inline float linear_to_srgb(float L)
{
    L = fminf(fmaxf(L, 0.0f), 1.0f);
    float u;
    if (L <= 0.0031308f) {
        u = 12.92f * L;
    } else {
        u = 1.055f * powf(L, 1.0f / 2.4f) - 0.055f;
    }
    return 255.0f * u;
}

/**
 * Apply color gain in linear light space for RGBA images.
 *
 * @param in Input image (uchar4 per pixel, RGBA)
 * @param out Output image (uchar4 per pixel, RGBA)
 * @param W Image width
 * @param H Image height
 * @param stridePx Stride in pixels
 * @param gR Red channel gain
 * @param gG Green channel gain
 * @param gB Blue channel gain
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void apply_color_gain_linear_rgba(
    const uchar4* __restrict__ in,
    uchar4* __restrict__ out,
    int W, int H, int stridePx,
    float gR, float gG, float gB
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    uchar4 p = in[idx];

    // Convert to linear space
    float R = srgb_to_linear((float)p.x);
    float G = srgb_to_linear((float)p.y);
    float B = srgb_to_linear((float)p.z);

    // Apply gains with deterministic rounding
    float R2 = __fmul_rn(R, gR);
    float G2 = __fmul_rn(G, gG);
    float B2 = __fmul_rn(B, gB);

    // Convert back to sRGB
    uchar4 q;
    q.x = (unsigned char)__float2int_rn(linear_to_srgb(R2));
    q.y = (unsigned char)__float2int_rn(linear_to_srgb(G2));
    q.z = (unsigned char)__float2int_rn(linear_to_srgb(B2));
    q.w = p.w;  // Preserve alpha

    out[idx] = q;
}

/**
 * Apply color gain in linear light space for RGB images.
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void apply_color_gain_linear_rgb(
    const uchar3* __restrict__ in,
    uchar3* __restrict__ out,
    int W, int H, int stridePx,
    float gR, float gG, float gB
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    uchar3 p = in[idx];

    float R = srgb_to_linear((float)p.x);
    float G = srgb_to_linear((float)p.y);
    float B = srgb_to_linear((float)p.z);

    float R2 = __fmul_rn(R, gR);
    float G2 = __fmul_rn(G, gG);
    float B2 = __fmul_rn(B, gB);

    uchar3 q;
    q.x = (unsigned char)__float2int_rn(linear_to_srgb(R2));
    q.y = (unsigned char)__float2int_rn(linear_to_srgb(G2));
    q.z = (unsigned char)__float2int_rn(linear_to_srgb(B2));

    out[idx] = q;
}

/**
 * Fast approximate sRGB<->linear conversion (lookup table or polynomial approximation).
 * This variant trades some accuracy for speed when exact color science isn't critical.
 */
__device__ inline float srgb_to_linear_fast(float u)
{
    u = fminf(fmaxf(u / 255.0f, 0.0f), 1.0f);
    // Approximate gamma 2.2 instead of exact sRGB curve
    return powf(u, 2.2f);
}

__device__ inline float linear_to_srgb_fast(float L)
{
    L = fminf(fmaxf(L, 0.0f), 1.0f);
    // Approximate gamma 1/2.2
    return 255.0f * powf(L, 1.0f / 2.2f);
}

/**
 * Fast color gain variant (approximate gamma, faster than exact sRGB).
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void apply_color_gain_fast_rgb(
    const uchar3* __restrict__ in,
    uchar3* __restrict__ out,
    int W, int H, int stridePx,
    float gR, float gG, float gB
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    uchar3 p = in[idx];

    float R = srgb_to_linear_fast((float)p.x);
    float G = srgb_to_linear_fast((float)p.y);
    float B = srgb_to_linear_fast((float)p.z);

    float R2 = __fmul_rn(R, gR);
    float G2 = __fmul_rn(G, gG);
    float B2 = __fmul_rn(B, gB);

    uchar3 q;
    q.x = (unsigned char)__float2int_rn(linear_to_srgb_fast(R2));
    q.y = (unsigned char)__float2int_rn(linear_to_srgb_fast(G2));
    q.z = (unsigned char)__float2int_rn(linear_to_srgb_fast(B2));

    out[idx] = q;
}

// Explicit template instantiations
template __global__ void apply_color_gain_linear_rgba<256>(
    const uchar4*, uchar4*, int, int, int, float, float, float);
template __global__ void apply_color_gain_linear_rgba<512>(
    const uchar4*, uchar4*, int, int, int, float, float, float);

template __global__ void apply_color_gain_linear_rgb<256>(
    const uchar3*, uchar3*, int, int, int, float, float, float);
template __global__ void apply_color_gain_linear_rgb<512>(
    const uchar3*, uchar3*, int, int, int, float, float, float);

template __global__ void apply_color_gain_fast_rgb<256>(
    const uchar3*, uchar3*, int, int, int, float, float, float);
template __global__ void apply_color_gain_fast_rgb<512>(
    const uchar3*, uchar3*, int, int, int, float, float, float);
