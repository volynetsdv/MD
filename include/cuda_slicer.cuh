/**
 * @file cuda_slicer.cuh
 * @brief Zero-copy GPU-resident image slicing, bilinear resizing, and NCHW normalization kernels.
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#ifndef TILE_RECT_DEFINED
#define TILE_RECT_DEFINED
/**
 * @brief Rectangle describing a tile region in the source image (pixel coordinates).
 */
struct TileRect
{
    int x{0}; ///< Top-left corner X coordinate (px).
    int y{0}; ///< Top-left corner Y coordinate (px).
    int w{0}; ///< Width in pixels (px).
    int h{0}; ///< Height in pixels (px).
};
#endif

/**
 * @brief Extract a tile from a GPU-resident BGR image, resize to model_target_size x model_target_size,
 *        and normalize into a float32 tensor of shape [1, 3, target, target].
 *
 * All data stays on the device — no host copies, no disk I/O.
 *
 * @param d_src_img          Pointer to BGR image in VRAM (row-major, uint8).
 * @param src_stride_bytes   Bytes per row in the source image (>= src_w * 3).
 * @param src_w              Source image width (pixels).
 * @param src_h              Source image height (pixels).
 * @param rect               Tile region to extract.
 * @param model_target_size  Output spatial size (e.g., 320, 416, 512, 640).
 * @param d_dst_tensor       Pre-allocated float32 buffer in VRAM, size >= 3 * target^2 floats.
 * @param bgr_to_rgb         If true, convert channel order from BGR to RGB on the fly.
 * @param stream             CUDA stream for asynchronous execution.
 * @return cudaError_t       CUDA status code.
 */
cudaError_t extract_tile_gpu(const uint8_t *d_src_img,
                             int src_stride_bytes,
                             int src_w,
                             int src_h,
                             const TileRect &rect,
                             int model_target_size,
                             float *d_dst_tensor,
                             bool bgr_to_rgb = false,
                             cudaStream_t stream = 0);

/**
 * @brief Extract a batch of B tiles from a GPU-resident BGR image, resize each to model_target_size,
 *        and normalize into a float32 tensor of shape [batch_size, 3, target, target].
 *
 * @param d_src_img           Pointer to BGR image in VRAM (row-major, uint8).
 * @param src_stride_bytes    Bytes per row in the source image (>= src_w * 3).
 * @param src_w               Source image width (pixels).
 * @param src_h               Source image height (pixels).
 * @param rects               Host pointer to an array of B tile rectangles.
 * @param batch_size          Number of tiles in the batch (1 <= B <= 64).
 * @param model_target_size   Output spatial size per tile.
 * @param d_dst_batch_tensor  Pre-allocated float32 buffer in VRAM of shape [B, 3, target, target].
 * @param bgr_to_rgb          If true, convert channel order from BGR to RGB on the fly.
 * @param stream              CUDA stream for asynchronous execution.
 * @return cudaError_t        CUDA status code.
 */
cudaError_t extract_batch_tiles_gpu(const uint8_t *d_src_img,
                                    int src_stride_bytes,
                                    int src_w,
                                    int src_h,
                                    const TileRect *rects,
                                    int batch_size,
                                    int model_target_size,
                                    float *d_dst_batch_tensor,
                                    bool bgr_to_rgb = false,
                                    cudaStream_t stream = 0);
