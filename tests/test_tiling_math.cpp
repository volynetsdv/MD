#include "tiling_math.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // 8K frame: 7680 x 4320
    const int image_width = 7680;
    const int image_height = 4320;

    // Parameters chosen so that T_calc = 1000/4 + 30*10 = 550
    const float altitude = 30.0f;
    const size_t vram_mb = 1000;

    TilingConfig cfg = calculate_tiling_params(image_width, image_height,
                                               altitude, vram_mb);

    // Primary assertion: tile_size must be 512 (largest grid value ≤ 550)
    assert(cfg.tile_size == 512);

    // Overlap within valid range
    assert(cfg.overlap >= 0.1f && cfg.overlap <= 0.4f);

    // Grid covers the whole image
    const int step = static_cast<int>(std::round(
        static_cast<float>(cfg.tile_size) * (1.0f - cfg.overlap)));
    assert(step > 0);

    const int expected_cols = (image_width + step - 1) / step;
    const int expected_rows = (image_height + step - 1) / step;
    assert(cfg.grid_cols == expected_cols);
    assert(cfg.grid_rows == expected_rows);

    // Tile count matches grid
    assert(static_cast<int>(cfg.tiles.size()) == cfg.grid_cols * cfg.grid_rows);

    // First tile anchored at origin with full size
    const Rect& first = cfg.tiles.front();
    assert(first.x == 0 && first.y == 0);
    assert(first.w == 512 && first.h == 512);

    // Last tile must not exceed image bounds
    const Rect& last = cfg.tiles.back();
    assert(last.x + last.w <= image_width);
    assert(last.y + last.h <= image_height);

    std::printf("All tiling_math tests passed.\n");
    return 0;
}
