/**
 * @file cuda_slicer.cu
 * @brief CUDA kernel and host wrappers for zero-copy tile extraction, bilinear resizing, and normalization.
 */

#include "cuda_slicer.cuh"

#include <cmath>

namespace
{

    constexpr int kBlockX = 32;
    constexpr int kBlockY = 8;
    constexpr int kMaxBatchSize = 64;

    __constant__ TileRect c_batch_rects[kMaxBatchSize];

    /**
     * @brief CUDA 3D grid kernel for batched bilinear tile extraction, optional BGR-to-RGB conversion,
     *        and [0, 1] float32 NCHW normalization.
     *
     * @param d_src            Device pointer to source BGR image (uint8).
     * @param src_stride_bytes Row stride of the source image in bytes.
     * @param src_w            Source image width in pixels.
     * @param src_h            Source image height in pixels.
     * @param batch_size       Number of tiles in the current batch.
     * @param target_size      Output spatial resolution per tile.
     * @param bgr_to_rgb       Whether to swap B and R channels during extraction.
     * @param d_dst            Device pointer to destination NCHW float32 batch tensor.
     */
    __global__ void extract_batch_tiles_kernel(
        const uint8_t *__restrict__ d_src,
        int src_stride_bytes,
        int src_w, int src_h,
        int batch_size,
        int target_size,
        bool bgr_to_rgb,
        float *__restrict__ d_dst)
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

std::pair<size_t, size_t> get_gpu_memory_info()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        // Clear any sticky error on systems without an active CUDA device/driver.
        (void)cudaGetLastError();
        return {0, 0};
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess)
    {
        (void)cudaGetLastError();
        return {0, 0};
    }
    return {free_bytes, total_bytes};
}

uintptr_t gpu_malloc(size_t bytes)
{
    if (bytes == 0)
    {
        return 0;
    }
    void *ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
    {
        (void)cudaGetLastError();
        return 0;
    }
    return reinterpret_cast<uintptr_t>(ptr);
}

cudaError_t gpu_free(uintptr_t dev_ptr)
{
    if (dev_ptr == 0)
    {
        return cudaSuccess;
    }
    return cudaFree(reinterpret_cast<void *>(dev_ptr));
}

cudaError_t gpu_memcpy_h2d(uintptr_t dst_dev_ptr, uintptr_t src_host_ptr, size_t bytes)
{
    if (bytes == 0)
    {
        return cudaSuccess;
    }
    if (dst_dev_ptr == 0 || src_host_ptr == 0)
    {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpy(
        reinterpret_cast<void *>(dst_dev_ptr),
        reinterpret_cast<const void *>(src_host_ptr),
        bytes,
        cudaMemcpyHostToDevice);
}

cudaError_t gpu_memcpy_d2h(uintptr_t dst_host_ptr, uintptr_t src_dev_ptr, size_t bytes)
{
    if (bytes == 0)
    {
        return cudaSuccess;
    }
    if (dst_host_ptr == 0 || src_dev_ptr == 0)
    {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpy(
        reinterpret_cast<void *>(dst_host_ptr),
        reinterpret_cast<const void *>(src_dev_ptr),
        bytes,
        cudaMemcpyDeviceToHost);
}

cudaError_t gpu_synchronize()
{
    return cudaDeviceSynchronize();
}
