#include "cuda_slicer.cuh"

#include <cmath>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kMaxBatchSize = 64;

__constant__ TileRect c_batch_rects[kMaxBatchSize];

// ---------------------------------------------------------------------------
// Kernel: 3D grid (target_size_x, target_size_y, batch_size)
// Bilinear sampling from source image with on-the-fly BGR->RGB conversion and normalization.
// ---------------------------------------------------------------------------
__global__ void extract_batch_tiles_kernel(
    const uint8_t* __restrict__ d_src,
    int src_stride_bytes,
    int src_w, int src_h,
    int batch_size,
    int target_size,
    bool bgr_to_rgb,
    float* __restrict__ d_dst)
{
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int b  = blockIdx.z;

    if (ox >= target_size || oy >= target_size || b >= batch_size) return;

    TileRect rect = c_batch_rects[b];

    // Map output pixel centre -> source tile coordinates (OpenCV INTER_LINEAR align_corners=False).
    const float scale_x = static_cast<float>(rect.w) / static_cast<float>(target_size);
    const float scale_y = static_cast<float>(rect.h) / static_cast<float>(target_size);

    const float fx = rect.x + (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
    const float fy = rect.y + (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image / tile boundaries.
    if (x0 < rect.x) x0 = rect.x;
    if (y0 < rect.y) y0 = rect.y;
    if (x1 > rect.x + rect.w - 1) x1 = rect.x + rect.w - 1;
    if (y1 > rect.y + rect.h - 1) y1 = rect.y + rect.h - 1;

    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    const float alpha = fx - static_cast<float>(x0);
    const float beta  = fy - static_cast<float>(y0);

    // Bilinear weights.
    const float w00 = (1.0f - alpha) * (1.0f - beta);
    const float w10 = alpha           * (1.0f - beta);
    const float w01 = (1.0f - alpha) * beta;
    const float w11 = alpha           * beta;

    // Pointer offsets for the four corners.
    const uint8_t* p00 = d_src + y0 * src_stride_bytes + x0 * 3;
    const uint8_t* p10 = d_src + y0 * src_stride_bytes + x1 * 3;
    const uint8_t* p01 = d_src + y1 * src_stride_bytes + x0 * 3;
    const uint8_t* p11 = d_src + y1 * src_stride_bytes + x1 * 3;

    const size_t plane_size = static_cast<size_t>(target_size) * target_size;
    float* batch_dst = d_dst + static_cast<size_t>(b) * 3 * plane_size;
    const size_t pixel_offset = static_cast<size_t>(oy) * target_size + ox;

    for (int c = 0; c < 3; ++c) {
        int src_c = bgr_to_rgb ? (2 - c) : c;
        float v00 = static_cast<float>(p00[src_c]);
        float v10 = static_cast<float>(p10[src_c]);
        float v01 = static_cast<float>(p01[src_c]);
        float v11 = static_cast<float>(p11[src_c]);

        float val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;

        // Normalise to [0, 1] and write into planar CHW layout.
        batch_dst[c * plane_size + pixel_offset] = val / 255.0f;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
cudaError_t extract_tile_gpu(const uint8_t* d_src_img,
                             int src_stride_bytes,
                             int src_w,
                             int src_h,
                             const TileRect& rect,
                             int model_target_size,
                             float* d_dst_tensor,
                             bool bgr_to_rgb,
                             cudaStream_t stream)
{
    return extract_batch_tiles_gpu(d_src_img, src_stride_bytes, src_w, src_h,
                                   &rect, 1, model_target_size, d_dst_tensor,
                                   bgr_to_rgb, stream);
}

cudaError_t extract_batch_tiles_gpu(const uint8_t* d_src_img,
                                   int src_stride_bytes,
                                   int src_w,
                                   int src_h,
                                   const TileRect* rects,
                                   int batch_size,
                                   int model_target_size,
                                   float* d_dst_batch_tensor,
                                   bool bgr_to_rgb,
                                   cudaStream_t stream)
{
    if (!d_src_img || !d_dst_batch_tensor || !rects) return cudaErrorInvalidValue;
    if (batch_size <= 0 || model_target_size <= 0 || batch_size > kMaxBatchSize) return cudaErrorInvalidValue;

    for (int i = 0; i < batch_size; ++i) {
        if (rects[i].x < 0 || rects[i].y < 0 ||
            rects[i].x + rects[i].w > src_w || rects[i].y + rects[i].h > src_h) {
            return cudaErrorInvalidValue;
        }
    }

    cudaError_t err = cudaMemcpyToSymbolAsync(
        c_batch_rects, rects, batch_size * sizeof(TileRect), 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    dim3 block(kBlockX, kBlockY, 1);
    dim3 grid((model_target_size + kBlockX - 1) / kBlockX,
              (model_target_size + kBlockY - 1) / kBlockY,
              batch_size);

    extract_batch_tiles_kernel<<<grid, block, 0, stream>>>(
        d_src_img, src_stride_bytes, src_w, src_h,
        batch_size, model_target_size,
        bgr_to_rgb, d_dst_batch_tensor);

    return cudaGetLastError();
}
