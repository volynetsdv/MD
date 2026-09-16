/**
 * @file bindings.cpp
 * @brief pybind11 bindings for pytiling_core — the Python interface to libtiling_core.
 *
 * Exports:
 *   Structs:
 *     pytiling_core.Rect              — tile bounding box (x, y, w, h)
 *     pytiling_core.TilingConfig      — full tiling plan with tile grid metadata
 *     pytiling_core.Detection         — single detection in tile-local coordinates
 *     pytiling_core.GlobalDetection   — detection remapped to global frame coordinates
 *
 *   Functions:
 *     pytiling_core.calculate_tiling_params(width, height, altitude, vram_mb)
 *         → TilingConfig
 *     pytiling_core.remap_offsets(detection, tile_rect, model_target_size, tile_id)
 *         → GlobalDetection
 *     pytiling_core.cluster_diou_nms(detections, tiles, diou_threshold, conf_threshold)
 *         → list[GlobalDetection]
 *     pytiling_core.to_json_string(detections, pretty)
 *         → str
 *
 * DRY Principle: all logic is delegated to the identical C++ implementations
 * used by the C++ Inference Engine — no duplicate logic exists.
 */

#include "postprocess.hpp"
#include "tiling_math.hpp"
#include "trt_detector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pytiling_core, m) {
    m.doc() =
        "pytiling_core — Python bindings for libtiling_core.\n\n"
        "Provides the same hardware-aware dynamic tiling, offset mapping, and\n"
        "Cluster-DIoU-NMS algorithms used by the C++ Inference Engine, ensuring\n"
        "full DRY compatibility between the training pipeline and the edge runtime.";

    // ========================================================================
    // Rect
    // ========================================================================
    py::class_<Rect>(m, "Rect",
        "Axis-aligned bounding box of a tile region in pixel coordinates.")
        .def(py::init<>())
        .def(py::init([](int x, int y, int w, int h) {
            Rect r; r.x = x; r.y = y; r.w = w; r.h = h; return r;
        }), py::arg("x") = 0, py::arg("y") = 0,
            py::arg("w") = 0, py::arg("h") = 0)
        .def_readwrite("x", &Rect::x, "Top-left X (pixels)")
        .def_readwrite("y", &Rect::y, "Top-left Y (pixels)")
        .def_readwrite("w", &Rect::w, "Width  (pixels)")
        .def_readwrite("h", &Rect::h, "Height (pixels)")
        .def("__repr__", [](const Rect& r) {
            return "<Rect x=" + std::to_string(r.x) +
                   " y=" + std::to_string(r.y) +
                   " w=" + std::to_string(r.w) +
                   " h=" + std::to_string(r.h) + ">";
        });

    // ========================================================================
    // TilingConfig
    // ========================================================================
    py::class_<TilingConfig>(m, "TilingConfig",
        "Complete hardware-aware tiling plan produced by calculate_tiling_params().\n\n"
        "Attributes\n"
        "----------\n"
        "tile_size : int\n"
        "    M_selected — chosen inference resolution in {320, 416, 512, 640}.\n"
        "overlap : float\n"
        "    Dynamic overlap ratio O_lap in [0.1, 0.4].\n"
        "grid_cols : int\n"
        "    Number of columns in the tile grid.\n"
        "grid_rows : int\n"
        "    Number of rows in the tile grid.\n"
        "tiles : list[Rect]\n"
        "    Ordered list of all tile bounding boxes covering the full frame.")
        .def(py::init<>())
        .def_readwrite("tile_size",  &TilingConfig::tile_size,
                       "M_selected ∈ {320, 416, 512, 640}")
        .def_readwrite("overlap",    &TilingConfig::overlap,
                       "O_lap ∈ [0.1, 0.4]")
        .def_readwrite("grid_cols",  &TilingConfig::grid_cols)
        .def_readwrite("grid_rows",  &TilingConfig::grid_rows)
        .def_readwrite("tiles",      &TilingConfig::tiles,
                       "List of Rect tile bounding boxes")
        .def("__repr__", [](const TilingConfig& c) {
            return "<TilingConfig tile_size=" + std::to_string(c.tile_size) +
                   " overlap=" + std::to_string(c.overlap) +
                   " grid=" + std::to_string(c.grid_cols) +
                   "x" + std::to_string(c.grid_rows) +
                   " tiles=" + std::to_string(c.tiles.size()) + ">";
        });

    // ========================================================================
    // Detection (tile-local coordinates)
    // ========================================================================
    py::class_<Detection>(m, "Detection",
        "Single detection in tile-local pixel coordinates.")
        .def(py::init<>())
        .def_readwrite("x_local",  &Detection::x_local,  "Top-left X within tile (px)")
        .def_readwrite("y_local",  &Detection::y_local,  "Top-left Y within tile (px)")
        .def_readwrite("w",        &Detection::w,        "Bounding-box width  (px)")
        .def_readwrite("h",        &Detection::h,        "Bounding-box height (px)")
        .def_readwrite("conf",     &Detection::conf,     "Confidence score [0, 1]")
        .def_readwrite("class_id", &Detection::class_id, "Class index")
        .def("__repr__", [](const Detection& d) {
            return "<Detection x_local=" + std::to_string(d.x_local) +
                   " y_local=" + std::to_string(d.y_local) +
                   " w=" + std::to_string(d.w) +
                   " h=" + std::to_string(d.h) +
                   " conf=" + std::to_string(d.conf) +
                   " class_id=" + std::to_string(d.class_id) + ">";
        });

    // ========================================================================
    // GlobalDetection (global frame coordinates)
    // ========================================================================
    py::class_<GlobalDetection>(m, "GlobalDetection",
        "Detection remapped to global 8K frame coordinates.")
        .def(py::init<>())
        .def_readwrite("x",        &GlobalDetection::x,        "Top-left X in global frame (px)")
        .def_readwrite("y",        &GlobalDetection::y,        "Top-left Y in global frame (px)")
        .def_readwrite("w",        &GlobalDetection::w,        "Bounding-box width  (px)")
        .def_readwrite("h",        &GlobalDetection::h,        "Bounding-box height (px)")
        .def_readwrite("conf",     &GlobalDetection::conf,     "Confidence score [0, 1]")
        .def_readwrite("class_id", &GlobalDetection::class_id, "Class index")
        .def_readwrite("tile_id",  &GlobalDetection::tile_id,  "Source tile index (-1 = unknown)")
        .def("__repr__", [](const GlobalDetection& d) {
            return "<GlobalDetection x=" + std::to_string(d.x) +
                   " y=" + std::to_string(d.y) +
                   " w=" + std::to_string(d.w) +
                   " h=" + std::to_string(d.h) +
                   " conf=" + std::to_string(d.conf) +
                   " class_id=" + std::to_string(d.class_id) + ">";
        });

    // ========================================================================
    // calculate_tiling_params
    // ========================================================================
    m.def("calculate_tiling_params",
        [](int width, int height, float altitude, size_t vram_mb) {
            return calculate_tiling_params(width, height, altitude, vram_mb);
        },
        py::arg("width"),
        py::arg("height"),
        py::arg("altitude"),
        py::arg("vram_mb"),
        R"doc(
Compute a hardware-aware tiling plan for an aerial image.

Parameters
----------
width : int
    Image width in pixels (e.g. 7680 for 8K).
height : int
    Image height in pixels (e.g. 4320 for 8K).
altitude : float
    UAV flight altitude in metres.  Higher altitude → larger tile size
    and more overlap.
vram_mb : int
    Available GPU VRAM in megabytes used to compute T_calc:
        T_calc = vram_mb / 4 + altitude * 10

Returns
-------
TilingConfig
    tile_size  : M_selected ∈ {320, 416, 512, 640}
    overlap    : O_lap ∈ [0.1, 0.4]
    grid_cols  : columns in the tile grid
    grid_rows  : rows    in the tile grid
    tiles      : list[Rect] — all tile bounding boxes

Examples
--------
>>> import pytiling_core
>>> cfg = pytiling_core.calculate_tiling_params(7680, 4320, 120.0, 2048)
>>> cfg.tile_size
640
>>> len(cfg.tiles)
180
)doc");

    // ========================================================================
    // remap_offsets  (single Detection, Rect overload)
    // ========================================================================
    m.def("remap_offsets",
        [](const Detection& det, const Rect& tile,
           int model_target_size, int tile_id) {
            return remap_offsets(det, tile, model_target_size, tile_id);
        },
        py::arg("detection"),
        py::arg("tile"),
        py::arg("model_target_size") = 0,
        py::arg("tile_id")           = -1,
        R"doc(
Remap a detection from tile-local pixel coordinates to global frame coordinates.

    x_global = tile.x + x_local * (tile.w / model_target_size)
    y_global = tile.y + y_local * (tile.h / model_target_size)
    w_global = w_local * (tile.w / model_target_size)
    h_global = h_local * (tile.h / model_target_size)

Parameters
----------
detection : Detection
    Bounding box in tile-local coordinates.
tile : Rect
    Tile region in global image space.
model_target_size : int, optional
    M_selected used during inference (0 = 1:1, no scaling).
tile_id : int, optional
    Source tile index to attach to the result.

Returns
-------
GlobalDetection
)doc");

    // ========================================================================
    // remap_offsets  (batch: list[Detection], Rect)
    // ========================================================================
    m.def("remap_offsets_batch",
        [](const std::vector<Detection>& dets, const Rect& tile,
           int model_target_size, int tile_id) {
            return remap_offsets(dets, tile, model_target_size, tile_id);
        },
        py::arg("detections"),
        py::arg("tile"),
        py::arg("model_target_size") = 0,
        py::arg("tile_id")           = -1,
        R"doc(
Remap a list of detections from tile-local to global frame coordinates.

Parameters
----------
detections : list[Detection]
tile : Rect
model_target_size : int, optional
tile_id : int, optional

Returns
-------
list[GlobalDetection]
)doc");

    // ========================================================================
    // cluster_diou_nms
    // ========================================================================
    m.def("cluster_diou_nms",
        [](const std::vector<GlobalDetection>& dets,
           const std::vector<Rect>& tiles,
           float diou_threshold, float conf_threshold) {
            return cluster_diou_nms(dets, tiles, diou_threshold, conf_threshold);
        },
        py::arg("detections"),
        py::arg("tiles")            = std::vector<Rect>{},
        py::arg("diou_threshold")   = 0.5f,
        py::arg("conf_threshold")   = 0.0f,
        R"doc(
Cluster-DIoU-NMS: merge duplicate detections at tile-overlap boundaries.

DIoU = IoU - d² / c²  where d = center distance, c = enclosing box diagonal.

Detections of the same class with DIoU > diou_threshold are merged using
confidence-weighted coordinate averaging:
    x_merged = Σ(conf_i * x_i) / Σ conf_i
    conf_merged = max(conf_i)

Parameters
----------
detections : list[GlobalDetection]
tiles : list[Rect], optional
    Tile grid used to restrict merging to overlap zones only.
diou_threshold : float, optional (default 0.5)
conf_threshold : float, optional (default 0.0)

Returns
-------
list[GlobalDetection]
)doc");

    // ========================================================================
    // to_json_string
    // ========================================================================
    m.def("to_json_string",
        [](const std::vector<GlobalDetection>& dets, bool pretty) {
            return to_json_string(dets, pretty);
        },
        py::arg("detections"),
        py::arg("pretty") = false,
        R"doc(
Serialize detections to a lightweight JSON string.

Format (compact):
    [{"x":450.5,"y":200.0,"w":60.0,"h":40.0,"conf":0.89,"class_id":0}]

Parameters
----------
detections : list[GlobalDetection]
pretty : bool, optional (default False)

Returns
-------
str
)doc");
}

