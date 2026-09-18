#pragma once

#include "tiling_math.hpp"
#include "trt_detector.hpp"

#include <cstddef>
#include <string>
#include <vector>

#ifndef TILE_RECT_DEFINED
#define TILE_RECT_DEFINED
/**
 * @brief Rectangle describing a tile region in the source image (pixel coords).
 */
struct TileRect
{
    int x{0}; ///< top-left corner x
    int y{0}; ///< top-left corner y
    int w{0}; ///< width in pixels
    int h{0}; ///< height in pixels
};
#endif

/**
 * @brief Detection in global frame coordinates (e.g. 8K image space).
 */
struct GlobalDetection
{
    float x{0.0f};    ///< Top-left X in global frame (px)
    float y{0.0f};    ///< Top-left Y in global frame (px)
    float w{0.0f};    ///< Bounding-box width (px)
    float h{0.0f};    ///< Bounding-box height (px)
    float conf{0.0f}; ///< Confidence score [0, 1]
    int class_id{0};  ///< Class index
    int tile_id{-1};  ///< Source tile index (-1 if unknown)
};

using BBox = GlobalDetection;

/**
 * @brief Remap a single detection from tile-local coordinates to global 8K frame coordinates.
 *
 * Formula:
 *   x_global = X_0 + x_local * (W_t / M_selected)
 *   y_global = Y_0 + y_local * (H_t / M_selected)
 *   w_global = w_local * (W_t / M_selected)
 *   h_global = h_local * (H_t / M_selected)
 *
 * If model_target_size <= 0, scale factor defaults to 1.0 (no scaling).
 *
 * @param local_det          Detection in tile-local coordinates.
 * @param tile               Tile rectangle in global space (X_0, Y_0, W_t, H_t).
 * @param model_target_size  Model input resolution M_selected (e.g. 320, 512, 640). 0 for 1:1.
 * @param tile_id            Optional identifier of source tile.
 * @return GlobalDetection   Detection with coordinates translated to global frame.
 */
GlobalDetection remap_offsets(const Detection &local_det,
                              const TileRect &tile,
                              int model_target_size = 0,
                              int tile_id = -1);

GlobalDetection remap_offsets(const Detection &local_det,
                              const Rect &tile,
                              int model_target_size = 0,
                              int tile_id = -1);

/**
 * @brief Remap a vector of detections from tile-local coordinates to global 8K coordinates.
 */
std::vector<GlobalDetection> remap_offsets(const std::vector<Detection> &local_dets,
                                           const TileRect &tile,
                                           int model_target_size = 0,
                                           int tile_id = -1);

std::vector<GlobalDetection> remap_offsets(const std::vector<Detection> &local_dets,
                                           const Rect &tile,
                                           int model_target_size = 0,
                                           int tile_id = -1);

/**
 * @brief Scalar coordinate translation from tile-local to global frame.
 */
void remap_offsets(float x_local, float y_local, float w_local, float h_local,
                   const TileRect &tile,
                   float &x_global, float &y_global, float &w_global, float &h_global,
                   int model_target_size = 0);

/**
 * @brief Calculate standard Intersection over Union (IoU) between two global detections.
 */
float calculate_iou(const GlobalDetection &a, const GlobalDetection &b);

/**
 * @brief Calculate Distance-IoU (DIoU) between two global detections.
 *
 * DIoU = IoU - (d^2 / c^2)
 * where d is Euclidean distance between box centers, and c is diagonal of the smallest enclosing box.
 */
float calculate_diou(const GlobalDetection &a, const GlobalDetection &b);

/**
 * @brief Cluster-DIoU-NMS: clusters and merges duplicate bounding boxes at tile overlap boundaries.
 *
 * When duplicate detections of the same object occur across overlapping tile seams, they are merged
 * into a single unified detection using confidence-weighted coordinate averaging:
 *   x_merged = sum(conf_i * x_i) / sum(conf_i)
 *   y_merged = sum(conf_i * y_i) / sum(conf_i)
 *   w_merged = sum(conf_i * w_i) / sum(conf_i)
 *   h_merged = sum(conf_i * h_i) / sum(conf_i)
 *   conf_merged = max(conf_i)
 *
 * Merging is performed only at overlap boundaries:
 * - If tiles list is provided, only boxes intersecting tile overlap zones are candidates for merging.
 * - If tile_id is present, detections within the same tile are preserved (not merged), while cross-tile
 *   overlaps are merged.
 *
 * @param detections      Global detections collected from all tiles.
 * @param diou_threshold  DIoU threshold above which boxes of the same class are clustered (default 0.5).
 * @param conf_threshold  Minimum confidence threshold to keep (default 0.0).
 * @return std::vector<GlobalDetection> Merged and filtered global detections.
 */
std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              float diou_threshold = 0.5f,
                                              float conf_threshold = 0.0f);

std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              const std::vector<TileRect> &tiles,
                                              float diou_threshold = 0.5f,
                                              float conf_threshold = 0.0f);

std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              const std::vector<Rect> &tiles,
                                              float diou_threshold = 0.5f,
                                              float conf_threshold = 0.0f);

/**
 * @brief Convert final detections into a lightweight JSON string for transmission to the UI.
 *
 * Format example:
 * [
 *   {"x":450.5,"y":200.0,"w":60.0,"h":40.0,"conf":0.89,"class_id":0}
 * ]
 *
 * @param detections  Vector of global detections.
 * @param pretty      If true, format with indentation; otherwise compact JSON.
 * @return std::string JSON representation.
 */
std::string to_json_string(const std::vector<GlobalDetection> &detections, bool pretty = false);
