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

    __device__ __forceinline__ void catmull_rom_weights(float t, float4& w) {
        float t2 = t * t;
        float t3 = t2 * t;
        w.x = -0.5f * t3 + t2 - 0.5f * t;
        w.y =  1.5f * t3 - 2.5f * t2 + 1.0f;
        w.z = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
        w.w =  0.5f * t3 - 0.5f * t2;
    }

    __device__ __forceinline__ float4 sample_bicubic_fast(cudaTextureObject_t tex, float sx, float sy) {
        float px = floorf(sx);
        float py = floorf(sy);
        float fx = sx - px;
        float fy = sy - py;

        float4 wx;
        float4 wy;
        catmull_rom_weights(fx, wx);
        catmull_rom_weights(fy, wy);

        float2 gx = make_float2(wx.x + wx.y, wx.z + wx.w);
        float2 gy = make_float2(wy.x + wy.y, wy.z + wy.w);

        float2 ox = make_float2(
            gx.x > 1e-3f ? (wx.y / gx.x) - 0.5f : 0.0f,
            gx.y > 1e-3f ? (wx.w / gx.y) + 0.5f : 0.0f);
        float2 oy = make_float2(
            gy.x > 1e-3f ? (wy.y / gy.x) - 0.5f : 0.0f,
            gy.y > 1e-3f ? (wy.w / gy.y) + 0.5f : 0.0f);

        float u0 = px + ox.x + 0.5f;
        float u1 = px + ox.y + 0.5f;
        float v0 = py + oy.x + 0.5f;
        float v1 = py + oy.y + 0.5f;

        float4 s0 = tex2D<float4>(tex, u0, v0);
        float4 s1 = tex2D<float4>(tex, u1, v0);
        float4 s2 = tex2D<float4>(tex, u0, v1);
        float4 s3 = tex2D<float4>(tex, u1, v1);

        float4 top = s0 * gx.x + s1 * gx.y;
        float4 bottom = s2 * gx.x + s3 * gx.y;
        float4 result = top * gy.x + bottom * gy.y;
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

        float4 sample = sample_bicubic_fast(src_tex, sx, sy);

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
        m.def("warp_face", &warp_face, "Fast bicubic warp of source ROI into destination frame");
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
