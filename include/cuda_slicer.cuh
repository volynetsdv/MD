#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Rectangle describing a tile region in the source image (pixel coords).
 */
struct TileRect {
    int x;   // top-left corner x
    int y;   // top-left corner y
    int w;   // width  in pixels
    int h;   // height in pixels
};

/**
 * @brief Extract a tile from a GPU-resident BGR image, resize to model_target_size×model_target_size
 *        and normalise into a float32 tensor of shape [1, 3, target, target].
 *
 * All data stays on the device — no host copies, no disk I/O.
 *
 * @param d_src_img       Pointer to BGR image in VRAM (row-major, uint8).
 * @param src_stride_bytes Bytes per row in the source image (≥ src_w × 3).
 * @param src_w           Source image width  (pixels).
 * @param src_h           Source image height (pixels).
 * @param rect            Tile region to extract.
 * @param model_target_size  Output spatial size (e.g. 512).
 * @param d_dst_tensor    Pre-allocated float32 buffer in VRAM, size ≥ 3 × target² floats.
 * @return cudaError_t    CUDA status code.
 */
cudaError_t extract_tile_gpu(const uint8_t* d_src_img,
                             int src_stride_bytes,
                             int src_w,
                             int src_h,
                             const TileRect& rect,
                             int model_target_size,
                             float* d_dst_tensor);
