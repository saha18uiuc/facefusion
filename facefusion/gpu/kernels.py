"""CUDA kernels compiled at runtime for GPU-accelerated ROI warping."""
from __future__ import annotations

import functools

import torch
from torch.utils.cpp_extension import load_inline


@functools.lru_cache(maxsize=1)
def _load_module() -> object:
    # Bilinear sample kernel that matches OpenCV's INTER_LINEAR with border
    # replicate semantics. We avoid Catmull-Rom here to keep the output
    # aligned with baseline quality.
    cuda_src = r"""
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <ATen/cuda/CUDAUtils.h>

    __device__ __forceinline__ float clamp_coord(float v, float limit) {
        if (v < 0.0f) return 0.0f;
        if (v > limit) return limit;
        return v;
    }

    __global__ void warp_bilinear_kernel(
        const float* __restrict__ src,
        float* __restrict__ out,
        const float* __restrict__ affine,
        int src_w,
        int src_h,
        int channels,
        int out_w,
        int out_h,
        size_t src_stride_bytes,
        size_t out_stride_bytes
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= out_w || y >= out_h) return;

        float src_x = affine[0] * static_cast<float>(x) + affine[1] * static_cast<float>(y) + affine[2];
        float src_y = affine[3] * static_cast<float>(x) + affine[4] * static_cast<float>(y) + affine[5];

        // Border replicate
        src_x = clamp_coord(src_x, static_cast<float>(src_w - 1));
        src_y = clamp_coord(src_y, static_cast<float>(src_h - 1));

        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);
        float dx = src_x - static_cast<float>(x0);
        float dy = src_y - static_cast<float>(y0);

        const char* base = reinterpret_cast<const char*>(src);
        const char* row0 = base + static_cast<size_t>(y0) * src_stride_bytes;
        const char* row1 = base + static_cast<size_t>(y1) * src_stride_bytes;

        char* out_base = reinterpret_cast<char*>(out);
        char* out_row = out_base + static_cast<size_t>(y) * out_stride_bytes;
        float* out_pixel = reinterpret_cast<float*>(out_row) + static_cast<size_t>(x) * channels;

        for (int c = 0; c < channels; ++c) {
            const float* p00 = reinterpret_cast<const float*>(row0) + static_cast<size_t>(x0) * channels + c;
            const float* p01 = reinterpret_cast<const float*>(row0) + static_cast<size_t>(x1) * channels + c;
            const float* p10 = reinterpret_cast<const float*>(row1) + static_cast<size_t>(x0) * channels + c;
            const float* p11 = reinterpret_cast<const float*>(row1) + static_cast<size_t>(x1) * channels + c;

            float top = *p00 + dx * (*p01 - *p00);
            float bot = *p10 + dx * (*p11 - *p10);
            out_pixel[c] = top + dy * (bot - top);
        }
    }

    torch::Tensor warp_face(
        torch::Tensor src,
        torch::Tensor affine,
        int64_t out_w,
        int64_t out_h
    ) {
        TORCH_CHECK(src.is_cuda(), "src must be CUDA");
        TORCH_CHECK(affine.is_cuda(), "affine must be CUDA");
        TORCH_CHECK(src.dtype() == torch::kFloat32, "src tensor expects float32");
        TORCH_CHECK(affine.dtype() == torch::kFloat32, "affine tensor expects float32");
        TORCH_CHECK(src.dim() == 3, "src tensor must be HxWxC");
        TORCH_CHECK(affine.numel() == 6, "affine must have 6 elements");
        TORCH_CHECK(out_w > 0 && out_h > 0, "output width/height must be positive");

        auto sizes = src.sizes();
        int src_h = static_cast<int>(sizes[0]);
        int src_w = static_cast<int>(sizes[1]);
        int channels = static_cast<int>(sizes[2]);

        auto options = src.options();
        auto out = torch::empty({out_h, out_w, channels}, options);

        const dim3 block(16, 16);
        const dim3 grid(
            static_cast<unsigned int>((out_w + block.x - 1) / block.x),
            static_cast<unsigned int>((out_h + block.y - 1) / block.y)
        );

        auto stream = at::cuda::getCurrentCUDAStream();
        size_t src_stride_bytes = static_cast<size_t>(src.stride(0)) * src.element_size();
        size_t out_stride_bytes = static_cast<size_t>(out.stride(0)) * out.element_size();

        warp_bilinear_kernel<<<grid, block, 0, stream.stream()>>>(
            src.data_ptr<float>(),
            out.data_ptr<float>(),
            affine.data_ptr<float>(),
            src_w,
            src_h,
            channels,
            static_cast<int>(out_w),
            static_cast<int>(out_h),
            src_stride_bytes,
            out_stride_bytes
        );
        AT_CUDA_CHECK(cudaGetLastError());
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("warp_face", &warp_face, "Bilinear warp of source ROI into destination frame");
    }
    """
    return load_inline(
        name="facefusion_cuda_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["warp_face"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def warp_face(src_rgb: torch.Tensor, affine: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Warp ``src_rgb`` (H, W, C) into an ROI with bilinear sampling."""
    module = _load_module()
    return module.warp_face(src_rgb, affine, int(width), int(height))
