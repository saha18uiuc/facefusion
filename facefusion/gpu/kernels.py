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

    __device__ __forceinline__ float mitchell1d(float x) {
        const float B = 1.0f / 3.0f;
        const float C = 1.0f / 3.0f;
        x = fabsf(x);
        if (x < 1.0f) {
            return ((12.0f - 9.0f * B - 6.0f * C) * x * x * x
                    + (-18.0f + 12.0f * B + 6.0f * C) * x * x
                    + (6.0f - 2.0f * B)) / 6.0f;
        } else if (x < 2.0f) {
            return ((-B - 6.0f * C) * x * x * x
                    + (6.0f * B + 30.0f * C) * x * x
                    + (-12.0f * B - 48.0f * C) * x
                    + (8.0f * B + 24.0f * C)) / 6.0f;
        }
        return 0.0f;
    }

    __device__ __forceinline__ float4 sample_mitchell(cudaTextureObject_t tex, float sx, float sy) {
        float fx = floorf(sx);
        float fy = floorf(sy);
        float tx = sx - fx;
        float ty = sy - fy;

        float wx[4];
        float wy[4];
        for (int i = 0; i < 4; ++i) {
            wx[i] = mitchell1d(tx - (static_cast<float>(i) - 1.0f));
            wy[i] = mitchell1d(ty - (static_cast<float>(i) - 1.0f));
        }

        float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float weight_sum = 0.0f;

        for (int j = 0; j < 4; ++j) {
            float py = static_cast<float>(fy + j - 1) + 0.5f;
            for (int i = 0; i < 4; ++i) {
                float px = static_cast<float>(fx + i - 1) + 0.5f;
                float4 sample = tex2D<float4>(tex, px, py);
                float w = wx[i] * wy[j];
                accum.x += sample.x * w;
                accum.y += sample.y * w;
                accum.z += sample.z * w;
                accum.w += sample.w * w;
                weight_sum += w;
            }
        }

        if (weight_sum > 0.0f) {
            float inv = 1.0f / weight_sum;
            accum.x *= inv;
            accum.y *= inv;
            accum.z *= inv;
            accum.w *= inv;
        }

        return accum;
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

        float4 sample = sample_mitchell(src_tex, sx, sy);

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
        m.def("warp_face", &warp_face, "Mitchell-Netravali warp of source ROI into destination frame");
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
