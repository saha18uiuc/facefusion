"""CUDA kernels compiled at runtime for GPU-accelerated ROI warping."""
from __future__ import annotations

import functools

import torch
from torch.utils.cpp_extension import load_inline


@functools.lru_cache(maxsize=1)
def _load_module() -> object:
    cuda_src = r"""
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <ATen/cuda/CUDAUtils.h>
    #include <math.h>

    __device__ __forceinline__ float catmull_rom(float p0, float p1, float p2, float p3, float t) {
        float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
        float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
        float a2 = -0.5f * p0 + 0.5f * p2;
        return ((a0 * t + a1) * t + a2) * t + p1;
    }

    __device__ __forceinline__ float4 sample_catmull_rom(cudaTextureObject_t tex, float sx, float sy) {
        float fx = floorf(sx);
        float fy = floorf(sy);
        float tx = sx - fx;
        float ty = sy - fy;

        int ix = static_cast<int>(fx);
        int iy = static_cast<int>(fy);

        float4 rows[4];
        for (int j = -1; j <= 2; ++j) {
            float4 p0 = tex2D<float4>(tex, static_cast<float>(ix - 1) + 0.5f, static_cast<float>(iy + j) + 0.5f);
            float4 p1 = tex2D<float4>(tex, static_cast<float>(ix) + 0.5f, static_cast<float>(iy + j) + 0.5f);
            float4 p2 = tex2D<float4>(tex, static_cast<float>(ix + 1) + 0.5f, static_cast<float>(iy + j) + 0.5f);
            float4 p3 = tex2D<float4>(tex, static_cast<float>(ix + 2) + 0.5f, static_cast<float>(iy + j) + 0.5f);

            rows[j + 1].x = catmull_rom(p0.x, p1.x, p2.x, p3.x, tx);
            rows[j + 1].y = catmull_rom(p0.y, p1.y, p2.y, p3.y, tx);
            rows[j + 1].z = catmull_rom(p0.z, p1.z, p2.z, p3.z, tx);
            rows[j + 1].w = catmull_rom(p0.w, p1.w, p2.w, p3.w, tx);
        }

        float4 result;
        result.x = catmull_rom(rows[0].x, rows[1].x, rows[2].x, rows[3].x, ty);
        result.y = catmull_rom(rows[0].y, rows[1].y, rows[2].y, rows[3].y, ty);
        result.z = catmull_rom(rows[0].z, rows[1].z, rows[2].z, rows[3].z, ty);
        result.w = catmull_rom(rows[0].w, rows[1].w, rows[2].w, rows[3].w, ty);
        return result;
    }

    __global__ void warp_catmull_kernel(
        cudaTextureObject_t src_tex,
        float* __restrict__ out,
        const float* __restrict__ affine,
        int width,
        int height,
        int channels,
        size_t out_stride_bytes
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) {
            return;
        }

        float sx = affine[0] * static_cast<float>(x) + affine[1] * static_cast<float>(y) + affine[2];
        float sy = affine[3] * static_cast<float>(x) + affine[4] * static_cast<float>(y) + affine[5];

        float4 sample = sample_catmull_rom(src_tex, sx, sy);

        char* base = reinterpret_cast<char*>(out);
        float* row_ptr = reinterpret_cast<float*>(base + static_cast<size_t>(y) * out_stride_bytes);
        float* pixel_ptr = row_ptr + static_cast<size_t>(x) * channels;

        if (channels >= 1) {
            pixel_ptr[0] = sample.x;
        }
        if (channels >= 2) {
            pixel_ptr[1] = sample.y;
        }
        if (channels >= 3) {
            pixel_ptr[2] = sample.z;
        }
        if (channels >= 4) {
            pixel_ptr[3] = sample.w;
        }
    }

    static cudaTextureObject_t create_texture(const torch::Tensor& src) {
        cudaTextureObject_t tex = 0;
        cudaResourceDesc res_desc{};
        cudaTextureDesc tex_desc{};

        TORCH_CHECK(src.dim() == 3, "src tensor must be HxWxC");
        auto sizes = src.sizes();
        int height = static_cast<int>(sizes[0]);
        int width = static_cast<int>(sizes[1]);
        int channels = static_cast<int>(sizes[2]);
        TORCH_CHECK(channels == 4, "src tensor expects 4 channels (B, G, R, extra)");

        res_desc.resType = cudaResourceTypePitch2D;
        res_desc.res.pitch2D.devPtr = const_cast<void*>(src.data_ptr());
        res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();
        res_desc.res.pitch2D.width = width;
        res_desc.res.pitch2D.height = height;
        res_desc.res.pitch2D.pitchInBytes = width * sizeof(float4);

        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModePoint;
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

    torch::Tensor warp_face(
        torch::Tensor src,
        torch::Tensor affine,
        int64_t width,
        int64_t height
    ) {
        TORCH_CHECK(src.is_cuda(), "src must be CUDA");
        TORCH_CHECK(affine.is_cuda(), "affine must be CUDA");
        TORCH_CHECK(src.dtype() == torch::kFloat32, "src tensor expects float32");
        TORCH_CHECK(affine.dtype() == torch::kFloat32, "affine tensor expects float32");
        TORCH_CHECK(affine.numel() == 6, "affine must have 6 elements");
        TORCH_CHECK(width > 0 && height > 0, "output width/height must be positive");

        int channels = static_cast<int>(src.size(2));
        auto options = src.options();
        auto out = torch::empty({height, width, channels}, options);

        cudaTextureObject_t tex = create_texture(src);

        const dim3 block(16, 16);
        const dim3 grid(
            static_cast<unsigned int>((width + block.x - 1) / block.x),
            static_cast<unsigned int>((height + block.y - 1) / block.y)
        );

        auto stream = at::cuda::getCurrentCUDAStream();
        size_t out_stride_bytes = static_cast<size_t>(out.stride(0)) * out.element_size();

        warp_catmull_kernel<<<grid, block, 0, stream.stream()>>>(
            tex,
            out.data_ptr<float>(),
            affine.data_ptr<float>(),
            static_cast<int>(width),
            static_cast<int>(height),
            channels,
            out_stride_bytes
        );
        destroy_texture(tex);
        AT_CUDA_CHECK(cudaGetLastError());
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("warp_face", &warp_face, "Catmull-Rom warp of source ROI into destination frame");
    }
    """
    return load_inline(
        name="facefusion_cuda_kernels",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["warp_face"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def warp_face(src_rgba: torch.Tensor, affine: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Warp ``src_rgba`` (BGR + extra channel) into an ROI with Catmull-Rom sampling."""
    module = _load_module()
    return module.warp_face(src_rgba, affine, int(width), int(height))
