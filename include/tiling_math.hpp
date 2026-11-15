/**
 * @file tiling_math.hpp
 * @brief Dynamic hardware-aware spatial tiling mathematics and grid structures.
 */

#pragma once

#include <cstddef>
#include <vector>

/**
 * @brief Axis-aligned rectangular tile bounding box in pixel coordinates.
 */
struct Rect
{
    int x{0}; ///< Top-left X coordinate (px).
    int y{0}; ///< Top-left Y coordinate (px).
    int w{0}; ///< Width in pixels (px).
    int h{0}; ///< Height in pixels (px).
};

/**
 * @brief Complete hardware-aware tiling configuration and grid layout.
 */
struct TilingConfig
{
    int tile_size{0};         ///< Selected tile resolution in pixels.
    float overlap{0.0f};      ///< Dynamic spatial overlap ratio in [0.15, 0.20].
    int grid_cols{0};         ///< Number of columns in the tile grid.
    int grid_rows{0};         ///< Number of rows in the tile grid.
    std::vector<Rect> tiles;  ///< Ordered list of tile rectangles covering the frame.
};

/**
 * @brief Calculate dynamic hardware-aware tiling parameters.
 *
 * For ultra-large images (max(W, H) >= 5000px), adaptively scales the tile window
 * up to 736px for inference on 640px models.
 *
 * @param image_width        Source image width in pixels.
 * @param image_height       Source image height in pixels.
 * @param altitude           UAV flight altitude in meters.
 * @param vram_available_mb  Available GPU VRAM in megabytes.
 * @return TilingConfig      Computed tiling configuration and tile grid.
 */
TilingConfig calculate_tiling_params(int image_width,
                                     int image_height,
                                     float altitude,
                                     size_t vram_available_mb);
