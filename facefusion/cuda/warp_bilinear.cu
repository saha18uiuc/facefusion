/**
 * Deterministic Bilinear Warp CUDA Kernel
 *
 * Manual bilinear interpolation with explicit round-to-nearest-even intrinsics
 * to ensure bit-stable output and eliminate sub-pixel jitter.
 *
 * Uses vectorized uchar4 loads/stores for bandwidth efficiency.
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

/**
 * Deterministic bilinear warp kernel for RGBA images (uint8).
 *
 * @param src Source image buffer (uchar4 per pixel)
 * @param srcW Source width in pixels
 * @param srcH Source height in pixels
 * @param srcStridePx Source stride in pixels (may differ from width for padding)
 * @param dst Destination image buffer (uchar4 per pixel)
 * @param dstW Destination width in pixels
 * @param dstH Destination height in pixels
 * @param dstStridePx Destination stride in pixels
 * @param M Affine transformation matrix (2x3, row-major: [a,b,tx,c,d,ty])
 * @param roiX ROI offset X in destination
 * @param roiY ROI offset Y in destination
 *
 * Template parameters:
 * @tparam TILE_W Tile width (threads per block in X direction)
 * @tparam TILE_H Tile height (threads per block in Y direction)
 */
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void warp_bilinear_rgba_u8(
    const uchar4* __restrict__ src,
    int srcW, int srcH, int srcStridePx,
    uchar4* __restrict__ dst,
    int dstW, int dstH, int dstStridePx,
    const float* __restrict__ M,
    int roiX, int roiY
)
{
    // Calculate destination pixel coordinates
    const int x = blockIdx.x * TILE_W + threadIdx.x % TILE_W;
    const int y = blockIdx.y * TILE_H + threadIdx.x / TILE_W;

    if (x >= dstW || y >= dstH) return;

    // Load affine matrix components
    // M is stored as [a, b, tx, c, d, ty]
    const float a = M[0], b = M[1], tx = M[2];
    const float c = M[3], d = M[4], ty = M[5];

    // Apply affine transformation with deterministic rounding
    // Map dst(x+roiX, y+roiY) -> src(xf, yf)
    const float xf = __fadd_rn(__fadd_rn(__fmul_rn(a, (float)(x + roiX)),
                                         __fmul_rn(b, (float)(y + roiY))), tx);
    const float yf = __fadd_rn(__fadd_rn(__fmul_rn(c, (float)(x + roiX)),
                                         __fmul_rn(d, (float)(y + roiY))), ty);

    // Calculate bilinear interpolation coordinates
    const int x0 = max(0, min(srcW - 1, (int)floorf(xf)));
    const int y0 = max(0, min(srcH - 1, (int)floorf(yf)));
    const int x1 = min(srcW - 1, x0 + 1);
    const int y1 = min(srcH - 1, y0 + 1);

    const float fx = xf - (float)x0;
    const float fy = yf - (float)y0;

    // Vectorized loads (4 bytes per pixel = uchar4)
    const uchar4 p00 = src[y0 * srcStridePx + x0];
    const uchar4 p10 = src[y0 * srcStridePx + x1];
    const uchar4 p01 = src[y1 * srcStridePx + x0];
    const uchar4 p11 = src[y1 * srcStridePx + x1];

    // Cast to float for interpolation
    float4 c00 = make_float4(p00.x, p00.y, p00.z, p00.w);
    float4 c10 = make_float4(p10.x, p10.y, p10.z, p10.w);
    float4 c01 = make_float4(p01.x, p01.y, p01.z, p01.w);
    float4 c11 = make_float4(p11.x, p11.y, p11.z, p11.w);

    // Bilinear interpolation with round-to-nearest intrinsics
    // LERP(a, b, t) = a + (b - a) * t
    // Using __fadd_rn and __fmul_rn to avoid FMA contraction
    #define LERP(a, b, t) (__fadd_rn((a), __fmul_rn(__fadd_rn((b), -(a)), (t))))

    // Interpolate in X direction
    float4 cx0 = make_float4(
        LERP(c00.x, c10.x, fx),
        LERP(c00.y, c10.y, fx),
        LERP(c00.z, c10.z, fx),
        LERP(c00.w, c10.w, fx)
    );
    float4 cx1 = make_float4(
        LERP(c01.x, c11.x, fx),
        LERP(c01.y, c11.y, fx),
        LERP(c01.z, c11.z, fx),
        LERP(c01.w, c11.w, fx)
    );

    // Interpolate in Y direction
    float4 cxy = make_float4(
        LERP(cx0.x, cx1.x, fy),
        LERP(cx0.y, cx1.y, fy),
        LERP(cx0.z, cx1.z, fy),
        LERP(cx0.w, cx1.w, fy)
    );

    #undef LERP

    // Pack back to uint8 with deterministic rounding
    uchar4 out;
    out.x = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.x, 0.0f), 255.0f));
    out.y = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.y, 0.0f), 255.0f));
    out.z = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.z, 0.0f), 255.0f));
    out.w = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.w, 0.0f), 255.0f));

    // Vectorized store
    dst[y * dstStridePx + x] = out;
}

/**
 * RGB variant (3-channel)
 */
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void warp_bilinear_rgb_u8(
    const uchar3* __restrict__ src,
    int srcW, int srcH, int srcStridePx,
    uchar3* __restrict__ dst,
    int dstW, int dstH, int dstStridePx,
    const float* __restrict__ M,
    int roiX, int roiY
)
{
    const int x = blockIdx.x * TILE_W + threadIdx.x % TILE_W;
    const int y = blockIdx.y * TILE_H + threadIdx.x / TILE_W;

    if (x >= dstW || y >= dstH) return;

    const float a = M[0], b = M[1], tx = M[2];
    const float c = M[3], d = M[4], ty = M[5];

    const float xf = __fadd_rn(__fadd_rn(__fmul_rn(a, (float)(x + roiX)),
                                         __fmul_rn(b, (float)(y + roiY))), tx);
    const float yf = __fadd_rn(__fadd_rn(__fmul_rn(c, (float)(x + roiX)),
                                         __fmul_rn(d, (float)(y + roiY))), ty);

    const int x0 = max(0, min(srcW - 1, (int)floorf(xf)));
    const int y0 = max(0, min(srcH - 1, (int)floorf(yf)));
    const int x1 = min(srcW - 1, x0 + 1);
    const int y1 = min(srcH - 1, y0 + 1);

    const float fx = xf - (float)x0;
    const float fy = yf - (float)y0;

    const uchar3 p00 = src[y0 * srcStridePx + x0];
    const uchar3 p10 = src[y0 * srcStridePx + x1];
    const uchar3 p01 = src[y1 * srcStridePx + x0];
    const uchar3 p11 = src[y1 * srcStridePx + x1];

    float3 c00 = make_float3(p00.x, p00.y, p00.z);
    float3 c10 = make_float3(p10.x, p10.y, p10.z);
    float3 c01 = make_float3(p01.x, p01.y, p01.z);
    float3 c11 = make_float3(p11.x, p11.y, p11.z);

    #define LERP(a, b, t) (__fadd_rn((a), __fmul_rn(__fadd_rn((b), -(a)), (t))))

    float3 cx0 = make_float3(
        LERP(c00.x, c10.x, fx),
        LERP(c00.y, c10.y, fx),
        LERP(c00.z, c10.z, fx)
    );
    float3 cx1 = make_float3(
        LERP(c01.x, c11.x, fx),
        LERP(c01.y, c11.y, fx),
        LERP(c01.z, c11.z, fx)
    );

    float3 cxy = make_float3(
        LERP(cx0.x, cx1.x, fy),
        LERP(cx0.y, cx1.y, fy),
        LERP(cx0.z, cx1.z, fy)
    );

    #undef LERP

    uchar3 out;
    out.x = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.x, 0.0f), 255.0f));
    out.y = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.y, 0.0f), 255.0f));
    out.z = (unsigned char)__float2int_rn(fminf(fmaxf(cxy.z, 0.0f), 255.0f));

    dst[y * dstStridePx + x] = out;
}

// Explicit template instantiations for common tile sizes
template __global__ void warp_bilinear_rgba_u8<16, 16>(
    const uchar4*, int, int, int, uchar4*, int, int, int, const float*, int, int);
template __global__ void warp_bilinear_rgba_u8<32, 8>(
    const uchar4*, int, int, int, uchar4*, int, int, int, const float*, int, int);

template __global__ void warp_bilinear_rgb_u8<16, 16>(
    const uchar3*, int, int, int, uchar3*, int, int, int, const float*, int, int);
template __global__ void warp_bilinear_rgb_u8<32, 8>(
    const uchar3*, int, int, int, uchar3*, int, int, int, const float*, int, int);
