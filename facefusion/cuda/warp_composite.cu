/**
 * Fused warp + composite CUDA kernel.
 *
 * Warps the generated face patch and mask directly into the destination ROI
 * and performs alpha blending with seam-band stabilization in a single pass.
 *
 * This eliminates two cv2.warpAffine calls and a separate composite kernel
 * launch per face per frame while preserving deterministic math.
 */

#include <cuda_runtime.h>

/**
 * Fused warp + composite for RGB uint8 images with float32 mask input.
 *
 * @tparam TILE_W Tile width processed per block
 * @tparam TILE_H Tile height processed per block
 */
template<int TILE_W, int TILE_H>
__global__ __launch_bounds__(TILE_W * TILE_H, 2)
void warp_composite_rgb_mask_u8(
    const uchar3* __restrict__ src,
    int srcW, int srcH, int srcStridePx,
    const float* __restrict__ mask,
    int maskStridePx,
    const uchar3* __restrict__ bg,
    uchar3* __restrict__ out,
    int dstW, int dstH, int dstStridePx,
    const float* __restrict__ M,
    int bandLo, int bandHi
)
{
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tile_threads = TILE_W * TILE_H;
    if (linear_tid >= tile_threads) {
        return;
    }

    const int x = blockIdx.x * TILE_W + (linear_tid % TILE_W);
    const int y = blockIdx.y * TILE_H + (linear_tid / TILE_W);

    if (x >= dstW || y >= dstH) {
        return;
    }

    const float a = M[0];
    const float b = M[1];
    const float tx = M[2];
    const float c = M[3];
    const float d = M[4];
    const float ty = M[5];

    const float xf = __fadd_rn(__fadd_rn(__fmul_rn(a, (float)x), __fmul_rn(b, (float)y)), tx);
    const float yf = __fadd_rn(__fadd_rn(__fmul_rn(c, (float)x), __fmul_rn(d, (float)y)), ty);

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

    #define LERP(a, b, t) (__fadd_rn((a), __fmul_rn(__fadd_rn((b), -(a)), (t))))

    const float3 cx0 = make_float3(
        LERP((float)p00.x, (float)p10.x, fx),
        LERP((float)p00.y, (float)p10.y, fx),
        LERP((float)p00.z, (float)p10.z, fx)
    );
    const float3 cx1 = make_float3(
        LERP((float)p01.x, (float)p11.x, fx),
        LERP((float)p01.y, (float)p11.y, fx),
        LERP((float)p01.z, (float)p11.z, fx)
    );

    const float3 warped_rgb = make_float3(
        LERP(cx0.x, cx1.x, fy),
        LERP(cx0.y, cx1.y, fy),
        LERP(cx0.z, cx1.z, fy)
    );

    const float m00 = mask[y0 * maskStridePx + x0];
    const float m10 = mask[y0 * maskStridePx + x1];
    const float m01 = mask[y1 * maskStridePx + x0];
    const float m11 = mask[y1 * maskStridePx + x1];

    const float mx0 = LERP(m00, m10, fx);
    const float mx1 = LERP(m01, m11, fx);
    float alpha = LERP(mx0, mx1, fy);

    #undef LERP

    alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    const int alpha_u8 = (int)__float2int_rn(alpha * 255.0f);
    float alpha_norm = (float)alpha_u8 / 255.0f;

    if (alpha_u8 >= bandLo && alpha_u8 <= bandHi) {
        alpha_norm = __fadd_rn(__fmul_rn(alpha_norm, 0.85f), 0.15f);
    }

    const uchar3 bg_pixel = bg[y * dstStridePx + x];

    const float inv_alpha = __fadd_rn(1.0f, -alpha_norm);
    const float r = __fadd_rn(__fmul_rn(warped_rgb.x, alpha_norm), __fmul_rn((float)bg_pixel.x, inv_alpha));
    const float g = __fadd_rn(__fmul_rn(warped_rgb.y, alpha_norm), __fmul_rn((float)bg_pixel.y, inv_alpha));
    const float b = __fadd_rn(__fmul_rn(warped_rgb.z, alpha_norm), __fmul_rn((float)bg_pixel.z, inv_alpha));

    uchar3 out_pixel;
    out_pixel.x = (unsigned char)__float2int_rn(fminf(fmaxf(r, 0.0f), 255.0f));
    out_pixel.y = (unsigned char)__float2int_rn(fminf(fmaxf(g, 0.0f), 255.0f));
    out_pixel.z = (unsigned char)__float2int_rn(fminf(fmaxf(b, 0.0f), 255.0f));

    out[y * dstStridePx + x] = out_pixel;
}

// Explicit instantiations

template __global__ void warp_composite_rgb_mask_u8<16, 16>(
    const uchar3*, int, int, int,
    const float*, int,
    const uchar3*, uchar3*,
    int, int, int,
    const float*, int, int);

template __global__ void warp_composite_rgb_mask_u8<32, 8>(
    const uchar3*, int, int, int,
    const float*, int,
    const uchar3*, uchar3*,
    int, int, int,
    const float*, int, int);
