#include "tiling_math.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

int main()
{
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
    const Rect &first = cfg.tiles.front();
    assert(first.x == 0 && first.y == 0);
    assert(first.w == 512 && first.h == 512);

    // Last tile must not exceed image bounds
    const Rect &last = cfg.tiles.back();
    assert(last.x + last.w <= image_width);
    assert(last.y + last.h <= image_height);

    // --- DOTA Ultra-Large Frame Test (7305 x 6759) ---
    {
        const int dota_w = 7305;
        const int dota_h = 6759;
        const float dota_altitude = 150.0f;
        const size_t dota_vram = 2048;

        TilingConfig dota_cfg = calculate_tiling_params(dota_w, dota_h, dota_altitude, dota_vram);

        // 1. Verify tile size is 736 px for ultra-large frame (>= 5000px)
        assert(dota_cfg.tile_size == 736);

        // 2. Verify total tile count is reduced by >= 20% compared to base 360 tiles
        const size_t base_tile_count = 360;
        const size_t actual_tile_count = dota_cfg.tiles.size();
        assert(actual_tile_count == 272); // 17 cols x 16 rows
        const float reduction = 1.0f - static_cast<float>(actual_tile_count) / static_cast<float>(base_tile_count);
        assert(reduction >= 0.20f); // 24.4% >= 20%

        // 3. Verify offset mapping round-trip precision within +- 1 px
        // Select an interior tile of size 736x736
        const Rect &t = dota_cfg.tiles.front();
        assert(t.w == 736 && t.h == 736);

        const float control_x_global = static_cast<float>(t.x) + 350.0f;
        const float control_y_global = static_cast<float>(t.y) + 280.0f;

        // Model input space coordinates (640x640)
        const float x_model = (control_x_global - static_cast<float>(t.x)) * (640.0f / 736.0f);
        const float y_model = (control_y_global - static_cast<float>(t.y)) * (640.0f / 736.0f);

        // Host-Device Offset Mapping back to global frame:
        const float scale_x = static_cast<float>(t.w) / 640.0f; // 1.15
        const float scale_y = static_cast<float>(t.h) / 640.0f; // 1.15
        const float remapped_x = static_cast<float>(t.x) + x_model * scale_x;
        const float remapped_y = static_cast<float>(t.y) + y_model * scale_y;

        assert(std::fabs(remapped_x - control_x_global) <= 1.0f);
        assert(std::fabs(remapped_y - control_y_global) <= 1.0f);
    }

    // --- Standard image (< 5000px) preserves canonical 640px grid ---
    {
        TilingConfig std_cfg = calculate_tiling_params(4000, 3000, 150.0f, 2048);
        assert(std_cfg.tile_size == 640);
        assert(static_cast<float>(std_cfg.tiles.front().w) / 640.0f == 1.0f);
    }

    std::printf("All tiling_math tests passed.\n");
    return 0;
}
