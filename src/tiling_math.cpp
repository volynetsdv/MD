#include "tiling_math.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kTileGrid[] = {320, 416, 512, 640};
constexpr int kTileGridSize = sizeof(kTileGrid) / sizeof(kTileGrid[0]);

int pick_tile_size(float t_calc) {
    for (int i = kTileGridSize - 1; i >= 0; --i) {
        if (kTileGrid[i] <= static_cast<int>(std::floor(t_calc))) {
            return kTileGrid[i];
        }
    }
    // Fallback: smallest tile if T_calc < 320
    return kTileGrid[0];
}

float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

} // namespace

TilingConfig calculate_tiling_params(int image_width,
                                     int image_height,
                                     float altitude,
                                     size_t vram_available_mb) {
    TilingConfig cfg;

    // --- T_calc: larger VRAM & higher altitude → bigger tiles ---
    const float t_calc = static_cast<float>(vram_available_mb) / 4.0f
                         + altitude * 10.0f;

    // --- Pick tile size from grid (largest ≤ T_calc) ---
    cfg.tile_size = pick_tile_size(t_calc);

    // --- Overlap: more altitude → more overlap, clamped [0.1, 0.4] ---
    cfg.overlap = clampf(0.1f + altitude / 500.0f, 0.1f, 0.4f);

    // --- Grid dimensions ---
    const int step_x = static_cast<int>(std::round(
        static_cast<float>(cfg.tile_size) * (1.0f - cfg.overlap)));
    const int step_y = step_x; // square tiles

    if (step_x <= 0 || step_y <= 0) {
        return cfg; // degenerate guard
    }

    cfg.grid_cols = (image_width + step_x - 1) / step_x;   // ceil div
    cfg.grid_rows = (image_height + step_y - 1) / step_y;  // ceil div

    // --- Generate tile rectangles ---
    cfg.tiles.reserve(static_cast<size_t>(cfg.grid_cols * cfg.grid_rows));

    for (int r = 0; r < cfg.grid_rows; ++r) {
        const int y = r * step_y;
        if (y >= image_height) break;
        const int h = std::min(cfg.tile_size, image_height - y);

        for (int c = 0; c < cfg.grid_cols; ++c) {
            const int x = c * step_x;
            if (x >= image_width) break;
            const int w = std::min(cfg.tile_size, image_width - x);

            cfg.tiles.push_back({x, y, w, h});
        }
    }

    return cfg;
}
