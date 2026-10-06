import sys
from pathlib import Path

# Ensure build directory is in sys.path if not installed
project_root = Path(__file__).resolve().parent.parent
build_paths = [
    project_root / "build",
    project_root / "build" / "Release",
    project_root / "build" / "bindings",
]
for p in build_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytiling_core

config = pytiling_core.calculate_tiling_params(width=7680, height=4320, altitude=120.0, vram_mb=2048)
print(config.tile_size, len(config.tiles))

if __name__ == "__main__":
    assert config.tile_size in [320, 416, 512, 640, 736]
    assert len(config.tiles) > 0

    # Acceptance Criteria 1: DOTA ultra-large frame 7305 x 6759
    dota_cfg = pytiling_core.calculate_tiling_params(
        width=7305, height=6759, altitude=150.0, vram_mb=2048
    )
    assert dota_cfg.tile_size == 736
    assert len(dota_cfg.tiles) == 156
    reduction = (360 - len(dota_cfg.tiles)) / 360.0
    assert reduction >= 0.20  # >= 20% reduction compared to base 360 tiles

    # Acceptance Criteria 1: Reverse offset mapping precision within +- 1px
    t0 = dota_cfg.tiles[0]
    local_det = pytiling_core.Detection()
    # Point at global (350, 280), scaled into 640x640 model input
    local_det.x_local = 350.0 * (640.0 / 736.0)
    local_det.y_local = 280.0 * (640.0 / 736.0)
    local_det.w = 50.0 * (640.0 / 736.0)
    local_det.h = 40.0 * (640.0 / 736.0)
    local_det.conf = 0.95
    local_det.class_id = 0

    g = pytiling_core.remap_offsets(local_det, t0, 640, 0)
    assert abs(g.x - 350.0) <= 1.0
    assert abs(g.y - 280.0) <= 1.0

    # Acceptance Criteria: Medium frame 1382 x 1394 generates <= 6 tiles
    p0945_cfg = pytiling_core.calculate_tiling_params(
        width=1382, height=1394, altitude=150.0, vram_mb=2048
    )
    assert len(p0945_cfg.tiles) <= 6
    assert p0945_cfg.grid_cols <= 3 and p0945_cfg.grid_rows <= 3
    assert p0945_cfg.tile_size == 640

    # Acceptance Criteria: Containment / IoS Suppression (Intersection over Smaller >= 0.88)
    b1 = pytiling_core.GlobalDetection()
    b1.x, b1.y, b1.w, b1.h, b1.conf, b1.class_id, b1.tile_id = 100.0, 100.0, 150.0, 150.0, 0.85, 0, 0
    b2 = pytiling_core.GlobalDetection()
    b2.x, b2.y, b2.w, b2.h, b2.conf, b2.class_id, b2.tile_id = 110.0, 110.0, 50.0, 50.0, 0.92, 0, 1

    ios_val = pytiling_core.calculate_containment(b1, b2)
    assert ios_val >= 0.88

    res_nms = pytiling_core.cluster_diou_nms([b1, b2], [], 0.5, 0.0)
    assert len(res_nms) == 1
    assert abs(res_nms[0].x - 100.0) <= 0.01
    assert abs(res_nms[0].y - 100.0) <= 0.01
    assert abs(res_nms[0].w - 150.0) <= 0.01
    assert abs(res_nms[0].h - 150.0) <= 0.01
    assert abs(res_nms[0].conf - 0.92) <= 0.001

    print("test_binding.py passed successfully!")


