/**
 * Kernel launcher wrappers for Python bindings.
 * These provide C-linkage functions that can be called from bindings.cpp
 */

#include <cuda_runtime.h>

// Include kernel declarations
template<int TILE_W, int TILE_H>
__global__ void warp_bilinear_rgb_u8(
    const uchar3*, int, int, int, uchar3*, int, int, int, const float*, int, int);

template<int TILE_W, int TILE_H>
__global__ void composite_rgb_u8(
    const uchar3*, const uchar3*, const unsigned char*, uchar3*, int, int, int, int, int);

template<int BLOCK_SIZE>
__global__ void apply_color_gain_linear_rgb(
    const uchar3*, uchar3*, int, int, int, float, float, float);

template<int BLOCK_SIZE>
__global__ void mask_edge_hysteresis_1d(
    const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int);

// Launchers
extern "C" {

void launch_warp_bilinear_rgb_u8(
    const uchar3* src, int srcW, int srcH, int srcStride,
    uchar3* dst, int dstW, int dstH, int dstStride,
    const float* M, int roiX, int roiY,
    dim3 grid, dim3 block
) {
    warp_bilinear_rgb_u8<16, 16><<<grid, block>>>(
        src, srcW, srcH, srcStride,
        dst, dstW, dstH, dstStride,
        M, roiX, roiY
    );
}

void launch_composite_rgb_u8(
    const uchar3* fg, const uchar3* bg, const unsigned char* alpha,
    uchar3* out, int W, int H, int stride, int bandLo, int bandHi,
    dim3 grid, dim3 block
) {
    composite_rgb_u8<16, 16><<<grid, block>>>(
        fg, bg, alpha, out, W, H, stride, bandLo, bandHi
    );
}

void launch_apply_color_gain_linear_rgb(
    const uchar3* in, uchar3* out, int W, int H, int stride,
    float gR, float gG, float gB, int grid_size, int block_size
) {
    apply_color_gain_linear_rgb<256><<<grid_size, block_size>>>(
        in, out, W, H, stride, gR, gG, gB
    );
}

void launch_mask_edge_hysteresis_1d(
    const unsigned char* alpha, const unsigned char* prev, unsigned char* out,
    int W, int H, int stride, int Tin, int Tout, int grid_size, int block_size
) {
    mask_edge_hysteresis_1d<256><<<grid_size, block_size>>>(
        alpha, prev, out, W, H, stride, Tin, Tout
    );
}

} // extern "C"
