#pragma once

#include <cstddef>
#include <vector>

struct Rect
{
    int x;
    int y;
    int w;
    int h;
};

struct TilingConfig
{
    int tile_size{0};
    float overlap{0.0f};
    int grid_cols{0};
    int grid_rows{0};
    std::vector<Rect> tiles;
};

/**
 * @brief Calculate dynamic hardware-aware tiling parameters.
 *
 * For ultra-large images (max(W, H) >= 5000px), adaptively scales the tile window
 * up to 736px for inference on 640px models.
 */
TilingConfig calculate_tiling_params(int image_width,
                                     int image_height,
                                     float altitude,
                                     size_t vram_available_mb);
