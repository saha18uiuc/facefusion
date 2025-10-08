"""CUDA kernels compiled at runtime for GPU-accelerated post-processing."""
from __future__ import annotations

import functools
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline


@functools.lru_cache(maxsize=1)
def _load_module() -> object:
    cuda_src = r"""
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <ATen/cuda/CUDAUtils.h>

    __global__ void warp_blend_kernel(
        cudaTextureObject_t src_tex,
        const uchar4* __restrict__ bg,
        const float* __restrict__ alpha,
        uchar4* __restrict__ out,
        const float* __restrict__ affine,
        int width,
        int height,
        size_t bg_stride,
        size_t out_stride
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }

        float sx = affine[0] * x + affine[1] * y + affine[2];
        float sy = affine[3] * x + affine[4] * y + affine[5];
        float4 sample = tex2D<float4>(src_tex, sx + 0.5f, sy + 0.5f);

        const uchar4* bg_row = reinterpret_cast<const uchar4*>(reinterpret_cast<const char*>(bg) + y * bg_stride);
        uchar4 bg_px = bg_row[x];

        float a = alpha[y * width + x];
        float inv_a = 1.0f - a;

        float3 bgf = make_float3(bg_px.x * (1.0f/255.0f), bg_px.y * (1.0f/255.0f), bg_px.z * (1.0f/255.0f));
        float3 sf = make_float3(sample.x, sample.y, sample.z);
        float3 outc;
        outc.x = sf.x * a + bgf.x * inv_a;
        outc.y = sf.y * a + bgf.y * inv_a;
        outc.z = sf.z * a + bgf.z * inv_a;

        uchar4* out_row = reinterpret_cast<uchar4*>(reinterpret_cast<char*>(out) + y * out_stride);
        out_row[x] = make_uchar4(
            static_cast<unsigned char>(fminf(fmaxf(outc.x * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf(outc.y * 255.0f, 0.0f), 255.0f)),
            static_cast<unsigned char>(fminf(fmaxf(outc.z * 255.0f, 0.0f), 255.0f)),
            255
        );
    }

    static cudaTextureObject_t create_texture(const torch::Tensor& src) {
        cudaTextureObject_t tex = 0;
        cudaResourceDesc res_desc{};
        cudaTextureDesc tex_desc{};

        auto sizes = src.sizes();
        int height = static_cast<int>(sizes[0]);
        int width = static_cast<int>(sizes[1]);

        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = const_cast<void*>(src.data_ptr());
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;
        res_desc.res.pitch2D.pitchInBytes = width * sizeof(float4);

        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        TORCH_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr) == cudaSuccess, "Failed to create texture");
        return tex;
    }

    static void destroy_texture(cudaTextureObject_t tex) {
        if (tex != 0) {
            cudaDestroyTextureObject(tex);
        }
    }

    torch::Tensor warp_blend(
        torch::Tensor src,
        torch::Tensor alpha,
        torch::Tensor bg,
        torch::Tensor affine
    ) {
        TORCH_CHECK(src.is_cuda(), "src must be CUDA");
        TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
        TORCH_CHECK(bg.is_cuda(), "bg must be CUDA");
        TORCH_CHECK(affine.is_cuda(), "affine must be CUDA");
        TORCH_CHECK(src.dtype() == torch::kFloat32, "src tensor expects float32");
        TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha tensor expects float32");
        TORCH_CHECK(bg.dtype() == torch::kUInt8, "bg tensor expects uint8");
        TORCH_CHECK(affine.numel() == 6, "affine must have 6 elements");

        TORCH_CHECK(alpha.dim() == 2, "alpha tensor must be 2D (H, W)");
        auto height = alpha.size(0);
        auto width = alpha.size(1);

        TORCH_CHECK(bg.size(0) == height && bg.size(1) == width,
                    "background ROI dimensions must match alpha dimensions");

        auto options = bg.options();
        auto out = torch::empty_like(bg);

        cudaTextureObject_t tex = create_texture(src);

        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        auto stream = at::cuda::getCurrentCUDAStream();
        size_t bg_stride_bytes = bg.stride(0) * bg.element_size();
        size_t out_stride_bytes = out.stride(0) * out.element_size();

        warp_blend_kernel<<<grid, block, 0, stream.stream()>>>(
            tex,
            reinterpret_cast<const uchar4*>(bg.data_ptr()),
            alpha.data_ptr<float>(),
            reinterpret_cast<uchar4*>(out.data_ptr()),
            affine.data_ptr<float>(),
            width,
            height,
            bg_stride_bytes,
            out_stride_bytes
        );
        destroy_texture(tex);
        AT_CUDA_CHECK(cudaGetLastError());
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("warp_blend", &warp_blend, "Fused affine warp, bilinear sample, and alpha blend");
    }
    """
    return load_inline(
        name="facefusion_cuda_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["warp_blend"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def warp_blend(src_rgba: torch.Tensor, alpha: torch.Tensor, background_bgr: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    module = _load_module()
    return module.warp_blend(src_rgba, alpha, background_bgr, affine)
