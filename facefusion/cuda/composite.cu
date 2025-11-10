/**
 * Alpha Composite with Seam-Band Feathering CUDA Kernel
 *
 * Fused composite that handles:
 * 1. Regular alpha blending
 * 2. Seam-band feathering to suppress one-frame halos
 *
 * Uses deterministic rounding and vectorized operations for stability and performance.
 */

#include <cuda_runtime.h>

/**
 * Alpha composite kernel with optional seam-band feathering for RGBA images.
 *
 * @param fg Foreground image (uchar4 per pixel, RGBA)
 * @param bg Background image (uchar4 per pixel, RGBA)
 * @param alpha Alpha mask (unsigned char, 0-255)
 * @param out Output image (uchar4 per pixel, RGBA)
 * @param W Image width
 * @param H Image height
 * @param stridePx Stride in pixels
 * @param bandLo Lower threshold for seam band (alpha value 0-255)
 * @param bandHi Upper threshold for seam band (alpha value 0-255)
 *
 * Template parameters:
 * @tparam TILE_W Tile width
 * @tparam TILE_H Tile height
 */
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void composite_rgba_u8(
    const uchar4* __restrict__ fg,
    const uchar4* __restrict__ bg,
    const unsigned char* __restrict__ alpha,
    uchar4* __restrict__ out,
    int W, int H, int stridePx,
    int bandLo, int bandHi
)
{
    const int x = blockIdx.x * TILE_W + threadIdx.x % TILE_W;
    const int y = blockIdx.y * TILE_H + threadIdx.x / TILE_W;

    if (x >= W || y >= H) return;

    const int idx = y * stridePx + x;
    const unsigned char alpha_val = alpha[idx];
    const float a = (float)alpha_val / 255.0f;

    // Load pixels
    const uchar4 F = fg[idx];
    const uchar4 B = bg[idx];

    // Cast to float
    float Fr = (float)F.x, Fg = (float)F.y, Fb = (float)F.z, Fa = (float)F.w;
    float Br = (float)B.x, Bg = (float)B.y, Bb = (float)B.z, Ba = (float)B.w;

    // Apply seam-band feathering if in band range
    // This nudges alpha toward stability in the seam ring
    float a_eff = a;
    if (alpha_val >= bandLo && alpha_val <= bandHi) {
        // Light feathering: blend toward 0.15 to reduce single-frame flicker
        a_eff = __fadd_rn(__fmul_rn(a, 0.85f), 0.15f);
    }

    // Alpha blending with deterministic rounding
    float r = __fadd_rn(__fmul_rn(Fr, a_eff), __fmul_rn(Br, __fadd_rn(1.0f, -a_eff)));
    float g = __fadd_rn(__fmul_rn(Fg, a_eff), __fmul_rn(Bg, __fadd_rn(1.0f, -a_eff)));
    float b = __fadd_rn(__fmul_rn(Fb, a_eff), __fmul_rn(Bb, __fadd_rn(1.0f, -a_eff)));
    float A = __fadd_rn(__fmul_rn(Fa, a_eff), __fmul_rn(Ba, __fadd_rn(1.0f, -a_eff)));

    // Pack to uint8 with deterministic rounding
    uchar4 O;
    O.x = (unsigned char)__float2int_rn(fminf(fmaxf(r, 0.0f), 255.0f));
    O.y = (unsigned char)__float2int_rn(fminf(fmaxf(g, 0.0f), 255.0f));
    O.z = (unsigned char)__float2int_rn(fminf(fmaxf(b, 0.0f), 255.0f));
    O.w = (unsigned char)__float2int_rn(fminf(fmaxf(A, 0.0f), 255.0f));

    out[idx] = O;
}

/**
 * Alpha composite kernel for RGB images (no alpha channel in output).
 */
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void composite_rgb_u8(
    const uchar3* __restrict__ fg,
    const uchar3* __restrict__ bg,
    const unsigned char* __restrict__ alpha,
    uchar3* __restrict__ out,
    int W, int H, int stridePx,
    int bandLo, int bandHi
)
{
    const int x = blockIdx.x * TILE_W + threadIdx.x % TILE_W;
    const int y = blockIdx.y * TILE_H + threadIdx.x / TILE_W;

    if (x >= W || y >= H) return;

    const int idx = y * stridePx + x;
    const unsigned char alpha_val = alpha[idx];
    const float a = (float)alpha_val / 255.0f;

    const uchar3 F = fg[idx];
    const uchar3 B = bg[idx];

    float Fr = (float)F.x, Fg = (float)F.y, Fb = (float)F.z;
    float Br = (float)B.x, Bg = (float)B.y, Bb = (float)B.z;

    float a_eff = a;
    if (alpha_val >= bandLo && alpha_val <= bandHi) {
        a_eff = __fadd_rn(__fmul_rn(a, 0.85f), 0.15f);
    }

    float r = __fadd_rn(__fmul_rn(Fr, a_eff), __fmul_rn(Br, __fadd_rn(1.0f, -a_eff)));
    float g = __fadd_rn(__fmul_rn(Fg, a_eff), __fmul_rn(Bg, __fadd_rn(1.0f, -a_eff)));
    float b = __fadd_rn(__fmul_rn(Fb, a_eff), __fmul_rn(Bb, __fadd_rn(1.0f, -a_eff)));

    uchar3 O;
    O.x = (unsigned char)__float2int_rn(fminf(fmaxf(r, 0.0f), 255.0f));
    O.y = (unsigned char)__float2int_rn(fminf(fmaxf(g, 0.0f), 255.0f));
    O.z = (unsigned char)__float2int_rn(fminf(fmaxf(b, 0.0f), 255.0f));

    out[idx] = O;
}

/**
 * Simple alpha blend without seam-band feathering (faster, for non-critical regions).
 */
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void alpha_blend_simple(
    const uchar3* __restrict__ fg,
    const uchar3* __restrict__ bg,
    const unsigned char* __restrict__ alpha,
    uchar3* __restrict__ out,
    int W, int H, int stridePx
)
{
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total = H * stridePx;

    if (idx >= total) return;

    const float a = (float)alpha[idx] / 255.0f;
    const uchar3 F = fg[idx];
    const uchar3 B = bg[idx];

    const float r = __fadd_rn(__fmul_rn((float)F.x, a),
                              __fmul_rn((float)B.x, __fadd_rn(1.0f, -a)));
    const float g = __fadd_rn(__fmul_rn((float)F.y, a),
                              __fmul_rn((float)B.y, __fadd_rn(1.0f, -a)));
    const float b = __fadd_rn(__fmul_rn((float)F.z, a),
                              __fmul_rn((float)B.z, __fadd_rn(1.0f, -a)));

    uchar3 O;
    O.x = (unsigned char)__float2int_rn(fminf(fmaxf(r, 0.0f), 255.0f));
    O.y = (unsigned char)__float2int_rn(fminf(fmaxf(g, 0.0f), 255.0f));
    O.z = (unsigned char)__float2int_rn(fminf(fmaxf(b, 0.0f), 255.0f));

    out[idx] = O;
}

// Explicit template instantiations
template __global__ void composite_rgba_u8<16, 16>(
    const uchar4*, const uchar4*, const unsigned char*, uchar4*, int, int, int, int, int);
template __global__ void composite_rgba_u8<32, 8>(
    const uchar4*, const uchar4*, const unsigned char*, uchar4*, int, int, int, int, int);

template __global__ void composite_rgb_u8<16, 16>(
    const uchar3*, const uchar3*, const unsigned char*, uchar3*, int, int, int, int, int);
template __global__ void composite_rgb_u8<32, 8>(
    const uchar3*, const uchar3*, const unsigned char*, uchar3*, int, int, int, int, int);

template __global__ void alpha_blend_simple<256>(
    const uchar3*, const uchar3*, const unsigned char*, uchar3*, int, int, int);
template __global__ void alpha_blend_simple<512>(
    const uchar3*, const uchar3*, const unsigned char*, uchar3*, int, int, int);
