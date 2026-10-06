#include "tiling_math.hpp"

#include <algorithm>
#include <cmath>

namespace
{

    constexpr int kTileGrid[] = {320, 416, 512, 640};
    constexpr int kTileGridSize = sizeof(kTileGrid) / sizeof(kTileGrid[0]);

    int pick_tile_size(float t_calc)
    {
        for (int i = kTileGridSize - 1; i >= 0; --i)
        {
            if (kTileGrid[i] <= static_cast<int>(std::floor(t_calc)))
            {
                return kTileGrid[i];
            }
        }
        // Fallback: smallest tile if T_calc < 320
        return kTileGrid[0];
    }

    float clampf(float v, float lo, float hi)
    {
        return std::max(lo, std::min(hi, v));
    }

} // namespace

TilingConfig calculate_tiling_params(int image_width,
                                     int image_height,
                                     float altitude,
                                     size_t vram_available_mb)
{
    TilingConfig cfg;

    const bool is_ultra_large = (image_width >= 5000 || image_height >= 5000);
    const int effective_slice_size = is_ultra_large ? 736 : 640;

    // --- T_calc: larger VRAM & higher altitude → bigger tiles ---
    const float t_calc = static_cast<float>(vram_available_mb) / 4.0f + altitude * 10.0f;

    // --- Pick tile size from grid (largest ≤ T_calc) ---
    cfg.tile_size = pick_tile_size(t_calc);
    if (cfg.tile_size == 640 && is_ultra_large)
    {
        cfg.tile_size = effective_slice_size;
    }

    // --- Overlap: clamped to [0.15, 0.20] ---
    cfg.overlap = clampf(0.15f + altitude / 3000.0f, 0.15f, 0.20f);

    // --- Grid dimensions based on effective step ---
    const int step_x = std::max(1, static_cast<int>(std::round(
        static_cast<float>(cfg.tile_size) * (1.0f - cfg.overlap))));
    const int step_y = step_x; // square tiles

    if (image_width <= cfg.tile_size)
    {
        cfg.grid_cols = 1;
    }
    else
    {
        cfg.grid_cols = std::max(1, static_cast<int>(std::ceil(
            static_cast<float>(image_width - cfg.tile_size) / static_cast<float>(step_x))) + 1);
    }

    if (image_height <= cfg.tile_size)
    {
        cfg.grid_rows = 1;
    }
    else
    {
        cfg.grid_rows = std::max(1, static_cast<int>(std::ceil(
            static_cast<float>(image_height - cfg.tile_size) / static_cast<float>(step_y))) + 1);
    }

    // --- Safeguard against excessive tiling on medium frames (max(W, H) <= 1500px) ---
    if (std::max(image_width, image_height) <= 1500)
    {
        cfg.grid_cols = std::min(cfg.grid_cols, 3);
        cfg.grid_rows = std::min(cfg.grid_rows, 3);
        if (cfg.grid_cols * cfg.grid_rows > 6)
        {
            if (image_width <= image_height)
            {
                cfg.grid_cols = 2;
            }
            else
            {
                cfg.grid_rows = 2;
            }
        }
    }

    // --- Generate tile rectangles with boundary/edge alignment ---
    cfg.tiles.reserve(static_cast<size_t>(cfg.grid_cols * cfg.grid_rows));

    for (int r = 0; r < cfg.grid_rows; ++r)
    {
        int y = 0;
        if (cfg.grid_rows > 1)
        {
            y = static_cast<int>(std::round(
                static_cast<float>(r * (image_height - cfg.tile_size)) /
                static_cast<float>(cfg.grid_rows - 1)));
            y = std::max(0, std::min(y, image_height - cfg.tile_size));
        }
        const int h = std::min(cfg.tile_size, image_height - y);

        for (int c = 0; c < cfg.grid_cols; ++c)
        {
            int x = 0;
            if (cfg.grid_cols > 1)
            {
                x = static_cast<int>(std::round(
                    static_cast<float>(c * (image_width - cfg.tile_size)) /
                    static_cast<float>(cfg.grid_cols - 1)));
                x = std::max(0, std::min(x, image_width - cfg.tile_size));
            }
            const int w = std::min(cfg.tile_size, image_width - x);

            cfg.tiles.push_back({x, y, w, h});
        }
    }

    return cfg;
}
