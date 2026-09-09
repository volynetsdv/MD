#include "cuda_slicer.cuh"

#include <cmath>

namespace {

// ---------------------------------------------------------------------------
// Kernel: one thread → one output pixel (ox, oy) for all 3 channels.
// Performs bilinear interpolation from the source tile region and writes
// normalised float values into a [3][target][target] tensor.
// ---------------------------------------------------------------------------
__global__ void extract_tile_kernel(
    const uint8_t* __restrict__ d_src,
    int src_stride_bytes,
    int rect_x, int rect_y, int rect_w, int rect_h,
    int target_size,
    float* __restrict__ d_dst)          // layout: [3][target_size][target_size]
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= target_size || oy >= target_size) return;

    // Map output pixel centre → source tile coordinates (float for bilinear).
    const float scale_x = static_cast<float>(rect_w) / static_cast<float>(target_size);
    const float scale_y = static_cast<float>(rect_h) / static_cast<float>(target_size);

    const float fx = rect_x + (static_cast<float>(ox) + 0.5f) * scale_x;
    const float fy = rect_y + (static_cast<float>(oy) + 0.5f) * scale_y;

    // Neighbouring integer pixels (clamped to valid tile bounds).
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image / tile boundaries.
    if (x0 < rect_x) x0 = rect_x;
    if (y0 < rect_y) y0 = rect_y;
    if (x1 > rect_x + rect_w - 1) x1 = rect_x + rect_w - 1;
    if (y1 > rect_y + rect_h - 1) y1 = rect_y + rect_h - 1;

    const float alpha = fx - static_cast<float>(x0);
    const float beta  = fy - static_cast<float>(y0);

    // Bilinear weights.
    const float w00 = (1.0f - alpha) * (1.0f - beta);
    const float w10 = alpha           * (1.0f - beta);
    const float w01 = (1.0f - alpha) * beta;
    const float w11 = alpha           * beta;

    // Pointer offsets for the four corners (BGR interleaved).
    const uint8_t* p00 = d_src + y0 * src_stride_bytes + x0 * 3;
    const uint8_t* p10 = d_src + y0 * src_stride_bytes + x1 * 3;
    const uint8_t* p01 = d_src + y1 * src_stride_bytes + x0 * 3;
    const uint8_t* p11 = d_src + y1 * src_stride_bytes + x1 * 3;

    for (int c = 0; c < 3; ++c) {
        float v00 = static_cast<float>(p00[c]);
        float v10 = static_cast<float>(p10[c]);
        float v01 = static_cast<float>(p01[c]);
        float v11 = static_cast<float>(p11[c]);

        float val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;

        // Normalise to [0, 1] and write into channel-plane layout.
        d_dst[c * target_size * target_size + oy * target_size + ox] = val / 255.0f;
    }
}

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

} // namespace

// ---------------------------------------------------------------------------
// Host wrapper — launches the kernel, no host↔device transfers.
// ---------------------------------------------------------------------------
cudaError_t extract_tile_gpu(const uint8_t* d_src_img,
                             int src_stride_bytes,
                             int src_w,
                             int src_h,
                             const TileRect& rect,
                             int model_target_size,
                             float* d_dst_tensor)
{
    if (!d_src_img || !d_dst_tensor) return cudaErrorInvalidValue;
    if (model_target_size <= 0)     return cudaErrorInvalidValue;

    // Validate that the tile fits inside the source image.
    if (rect.x < 0 || rect.y < 0 ||
        rect.x + rect.w > src_w || rect.y + rect.h > src_h) {
        return cudaErrorInvalidValue;
    }

    dim3 block(kBlockX, kBlockY);
    dim3 grid((model_target_size + kBlockX - 1) / kBlockX,
              (model_target_size + kBlockY - 1) / kBlockY);

    extract_tile_kernel<<<grid, block>>>(
        d_src_img, src_stride_bytes,
        rect.x, rect.y, rect.w, rect.h,
        model_target_size,
        d_dst_tensor);

    return cudaGetLastError();
}
