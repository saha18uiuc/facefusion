/**
 * Python bindings for FaceFusion CUDA kernels using pybind11.
 *
 * Exposes CUDA kernels to Python with NumPy array interface.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

namespace py = pybind11;

// Forward declarations of CUDA kernel launchers
// These will be defined in separate wrapper functions below

/**
 * Check CUDA error and throw Python exception if needed.
 */
inline void check_cuda(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error at " << file << ":" << line << ": "
           << cudaGetErrorString(code);
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUDA(x) check_cuda((x), __FILE__, __LINE__)

/**
 * RAII wrapper for CUDA device memory to prevent leaks.
 */
template<typename T>
class CudaDeviceMemory {
private:
    T* ptr_ = nullptr;

public:
    CudaDeviceMemory() = default;

    explicit CudaDeviceMemory(size_t size) {
        CHECK_CUDA(cudaMalloc(&ptr_, size));
    }

    ~CudaDeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);  // Don't check error in destructor
        }
    }

    // Disable copy
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    // Enable move
    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
};

/**
 * Warp bilinear (RGB) wrapper.
 */
void warp_bilinear_rgb(
    py::array_t<uint8_t> src,
    py::array_t<uint8_t> dst,
    py::array_t<float> affine_matrix,
    int roi_x = 0,
    int roi_y = 0
) {
    // Get buffer info
    auto src_buf = src.request();
    auto dst_buf = dst.request();
    auto M_buf = affine_matrix.request();

    // Validate shapes
    if (src_buf.ndim != 3 || dst_buf.ndim != 3 || M_buf.ndim != 1) {
        throw std::runtime_error("Invalid array dimensions");
    }
    if (src_buf.shape[2] != 3 || dst_buf.shape[2] != 3) {
        throw std::runtime_error("Expected RGB images (3 channels)");
    }
    if (M_buf.shape[0] != 6) {
        throw std::runtime_error("Affine matrix must be 6 elements [a,b,tx,c,d,ty]");
    }

    // Extract dimensions
    int src_h = src_buf.shape[0];
    int src_w = src_buf.shape[1];
    int dst_h = dst_buf.shape[0];
    int dst_w = dst_buf.shape[1];

    // Allocate device memory
    uchar3* d_src;
    uchar3* d_dst;
    float* d_M;

    size_t src_size = src_h * src_w * sizeof(uchar3);
    size_t dst_size = dst_h * dst_w * sizeof(uchar3);

    CHECK_CUDA(cudaMalloc(&d_src, src_size));
    CHECK_CUDA(cudaMalloc(&d_dst, dst_size));
    CHECK_CUDA(cudaMalloc(&d_M, 6 * sizeof(float)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_src, src_buf.ptr, src_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_M, M_buf.ptr, 6 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel (tile size 16x16)
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);

    // Declare kernel (must match .cu file)
    extern void launch_warp_bilinear_rgb_u8(
        const uchar3* src, int srcW, int srcH, int srcStride,
        uchar3* dst, int dstW, int dstH, int dstStride,
        const float* M, int roiX, int roiY,
        dim3 grid, dim3 block
    );

    launch_warp_bilinear_rgb_u8(
        d_src, src_w, src_h, src_w,
        d_dst, dst_w, dst_h, dst_w,
        d_M, roi_x, roi_y,
        grid, block
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(dst_buf.ptr, d_dst, dst_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_M));
}

/**
 * Composite RGB with alpha wrapper.
 */
void composite_rgb(
    py::array_t<uint8_t> fg,
    py::array_t<uint8_t> bg,
    py::array_t<uint8_t> alpha,
    py::array_t<uint8_t> out,
    int band_lo = 0,
    int band_hi = 0
) {
    auto fg_buf = fg.request();
    auto bg_buf = bg.request();
    auto alpha_buf = alpha.request();
    auto out_buf = out.request();

    if (fg_buf.ndim != 3 || bg_buf.ndim != 3 || alpha_buf.ndim != 2 || out_buf.ndim != 3) {
        throw std::runtime_error("Invalid array dimensions");
    }

    int h = fg_buf.shape[0];
    int w = fg_buf.shape[1];

    uchar3* d_fg;
    uchar3* d_bg;
    unsigned char* d_alpha;
    uchar3* d_out;

    size_t img_size = h * w * sizeof(uchar3);
    size_t alpha_size = h * w * sizeof(unsigned char);

    CHECK_CUDA(cudaMalloc(&d_fg, img_size));
    CHECK_CUDA(cudaMalloc(&d_bg, img_size));
    CHECK_CUDA(cudaMalloc(&d_alpha, alpha_size));
    CHECK_CUDA(cudaMalloc(&d_out, img_size));

    CHECK_CUDA(cudaMemcpy(d_fg, fg_buf.ptr, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bg, bg_buf.ptr, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alpha, alpha_buf.ptr, alpha_size, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    extern void launch_composite_rgb_u8(
        const uchar3* fg, const uchar3* bg, const unsigned char* alpha,
        uchar3* out, int W, int H, int stride, int bandLo, int bandHi,
        dim3 grid, dim3 block
    );

    launch_composite_rgb_u8(d_fg, d_bg, d_alpha, d_out, w, h, w, band_lo, band_hi, grid, block);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out_buf.ptr, d_out, img_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_fg));
    CHECK_CUDA(cudaFree(d_bg));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_out));
}

/**
 * Apply color gain in linear space wrapper.
 */
void apply_color_gain_linear(
    py::array_t<uint8_t> input,
    py::array_t<uint8_t> output,
    float gain_r,
    float gain_g,
    float gain_b
) {
    auto in_buf = input.request();
    auto out_buf = output.request();

    if (in_buf.ndim != 3 || out_buf.ndim != 3) {
        throw std::runtime_error("Expected 3D arrays");
    }

    int h = in_buf.shape[0];
    int w = in_buf.shape[1];

    uchar3* d_in;
    uchar3* d_out;

    size_t size = h * w * sizeof(uchar3);

    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_in, in_buf.ptr, size, cudaMemcpyHostToDevice));

    int total = h * w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    extern void launch_apply_color_gain_linear_rgb(
        const uchar3* in, uchar3* out, int W, int H, int stride,
        float gR, float gG, float gB, int grid_size, int block_size
    );

    launch_apply_color_gain_linear_rgb(d_in, d_out, w, h, w, gain_r, gain_g, gain_b, grid_size, block_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out_buf.ptr, d_out, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

/**
 * Mask edge hysteresis wrapper.
 */
void mask_edge_hysteresis(
    py::array_t<uint8_t> alpha,
    py::array_t<uint8_t> alpha_prev,
    py::array_t<uint8_t> alpha_out,
    int threshold_in = 4,
    int threshold_out = 6
) {
    auto alpha_buf = alpha.request();
    auto prev_buf = alpha_prev.request();
    auto out_buf = alpha_out.request();

    if (alpha_buf.ndim != 2 || prev_buf.ndim != 2 || out_buf.ndim != 2) {
        throw std::runtime_error("Expected 2D arrays (masks)");
    }

    int h = alpha_buf.shape[0];
    int w = alpha_buf.shape[1];

    unsigned char* d_alpha;
    unsigned char* d_prev;
    unsigned char* d_out;

    size_t size = h * w * sizeof(unsigned char);

    CHECK_CUDA(cudaMalloc(&d_alpha, size));
    CHECK_CUDA(cudaMalloc(&d_prev, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_alpha, alpha_buf.ptr, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_prev, prev_buf.ptr, size, cudaMemcpyHostToDevice));

    int total = h * w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    extern void launch_mask_edge_hysteresis_1d(
        const unsigned char* alpha, const unsigned char* prev, unsigned char* out,
        int W, int H, int stride, int Tin, int Tout, int grid_size, int block_size
    );

    launch_mask_edge_hysteresis_1d(d_alpha, d_prev, d_out, w, h, w, threshold_in, threshold_out, grid_size, block_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(out_buf.ptr, d_out, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_prev));
    CHECK_CUDA(cudaFree(d_out));
}

// Module definition
PYBIND11_MODULE(ff_cuda_ops, m) {
    m.doc() = "FaceFusion CUDA operations for high-performance video processing";

    m.def("warp_bilinear_rgb", &warp_bilinear_rgb,
          "Deterministic bilinear warp for RGB images",
          py::arg("src"), py::arg("dst"), py::arg("affine_matrix"),
          py::arg("roi_x") = 0, py::arg("roi_y") = 0);

    m.def("composite_rgb", &composite_rgb,
          "Alpha composite with seam-band feathering",
          py::arg("fg"), py::arg("bg"), py::arg("alpha"), py::arg("out"),
          py::arg("band_lo") = 0, py::arg("band_hi") = 0);

    m.def("apply_color_gain_linear", &apply_color_gain_linear,
          "Apply color gains in linear light space",
          py::arg("input"), py::arg("output"),
          py::arg("gain_r"), py::arg("gain_g"), py::arg("gain_b"));

    m.def("mask_edge_hysteresis", &mask_edge_hysteresis,
          "Apply hysteresis to mask edges to prevent flicker",
          py::arg("alpha"), py::arg("alpha_prev"), py::arg("alpha_out"),
          py::arg("threshold_in") = 4, py::arg("threshold_out") = 6);
}
