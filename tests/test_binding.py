import sys
from pathlib import Path

# Ensure build directory is in sys.path if not installed
project_root = Path(__file__).resolve().parent.parent
build_paths = [
    project_root / "build",
    project_root / "build" / "bindings",
]
for p in build_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytiling_core

config = pytiling_core.calculate_tiling_params(width=7680, height=4320, altitude=120.0, vram_mb=2048)
print(config.tile_size, len(config.tiles))

if __name__ == "__main__":
    assert config.tile_size in [320, 416, 512, 640]
    assert len(config.tiles) > 0
    print("test_binding.py passed successfully!")

