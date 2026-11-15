/**
 * @file postprocess.cpp
 * @brief Implementation of coordinate remapping, spatial overlap metrics, Cluster-DIoU-NMS, and JSON export.
 */

#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <queue>
#include <sstream>

namespace
{

    /**
     * @brief Axis-aligned intersection rectangle representing the overlap seam between two tiles.
     */
    struct OverlapRect
    {
        float x1;
        float y1;
        float x2;
        float y2;
        int tile1;
        int tile2;
    };

    /**
     * @brief Check whether a global detection bounding box intersects a tile overlap region.
     */
    bool box_intersects_region(const GlobalDetection &box, const OverlapRect &r)
    {
        float bx1 = box.x;
        float by1 = box.y;
        float bx2 = box.x + box.w;
        float by2 = box.y + box.h;

        return !(bx2 <= r.x1 || bx1 >= r.x2 || by2 <= r.y1 || by1 >= r.y2);
    }

    std::vector<OverlapRect> compute_overlap_regions(const std::vector<TileRect> &tiles,
                                                     std::vector<std::vector<size_t>> *tile_to_overlap = nullptr)
    {
        std::vector<OverlapRect> overlaps;
        const size_t n = tiles.size();
        if (tile_to_overlap)
        {
            tile_to_overlap->assign(n, {});
        }
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                float x1 = static_cast<float>(std::max(tiles[i].x, tiles[j].x));
                float y1 = static_cast<float>(std::max(tiles[i].y, tiles[j].y));
                float x2 = static_cast<float>(std::min(tiles[i].x + tiles[i].w, tiles[j].x + tiles[j].w));
                float y2 = static_cast<float>(std::min(tiles[i].y + tiles[i].h, tiles[j].y + tiles[j].h));

                if (x2 > x1 && y2 > y1)
                {
                    size_t r_idx = overlaps.size();
                    overlaps.push_back({x1, y1, x2, y2, static_cast<int>(i), static_cast<int>(j)});
                    if (tile_to_overlap)
                    {
                        (*tile_to_overlap)[i].push_back(r_idx);
                        (*tile_to_overlap)[j].push_back(r_idx);
                    }
                }
            }
        }
        return overlaps;
    }

    bool is_box_in_overlap(const GlobalDetection &box,
                           const std::vector<OverlapRect> &overlaps,
                           const std::vector<std::vector<size_t>> &tile_to_overlap)
    {
        if (overlaps.empty())
            return false;

        if (box.tile_id >= 0 && static_cast<size_t>(box.tile_id) < tile_to_overlap.size())
        {
            for (size_t r_idx : tile_to_overlap[box.tile_id])
            {
                if (box_intersects_region(box, overlaps[r_idx]))
                    return true;
            }
            return false;
        }

        for (const auto &r : overlaps)
        {
            if (box_intersects_region(box, r))
                return true;
        }
        return false;
    }

    std::vector<TileRect> to_tile_rects(const std::vector<Rect> &rects)
    {
        std::vector<TileRect> tiles;
        tiles.reserve(rects.size());
        for (const auto &r : rects)
        {
            tiles.push_back({r.x, r.y, r.w, r.h});
        }
        return tiles;
    }

    struct TileSafeZone
    {
        float x1{-1e9f};
        float y1{-1e9f};
        float x2{1e9f};
        float y2{1e9f};
        bool has_overlap{false};
    };

    std::vector<TileSafeZone> compute_tile_safe_zones(const std::vector<TileRect> &tiles,
                                                      const std::vector<OverlapRect> &overlaps,
                                                      const std::vector<std::vector<size_t>> &tile_to_overlap)
    {
        const size_t n = tiles.size();
        std::vector<TileSafeZone> safe_zones(n);

        for (size_t i = 0; i < n; ++i)
        {
            if (i >= tile_to_overlap.size() || tile_to_overlap[i].empty())
            {
                safe_zones[i].has_overlap = false;
                continue;
            }

            safe_zones[i].has_overlap = true;
            const float t_x1 = static_cast<float>(tiles[i].x);
            const float t_y1 = static_cast<float>(tiles[i].y);
            const float t_x2 = static_cast<float>(tiles[i].x + tiles[i].w);
            const float t_y2 = static_cast<float>(tiles[i].y + tiles[i].h);

            float s_x1 = -1e9f;
            float s_y1 = -1e9f;
            float s_x2 = 1e9f;
            float s_y2 = 1e9f;

            for (size_t r_idx : tile_to_overlap[i])
            {
                const auto &r = overlaps[r_idx];
                const float delta_x = (r.x2 - r.x1) * 0.5f;
                const float delta_y = (r.y2 - r.y1) * 0.5f;

                // Right seam of tile i: neighbor is to the right
                if (std::fabs(r.x2 - t_x2) < 1.0f)
                {
                    s_x2 = std::min(s_x2, r.x1 - delta_x);
                }
                // Left seam of tile i: neighbor is to the left
                if (std::fabs(r.x1 - t_x1) < 1.0f)
                {
                    s_x1 = std::max(s_x1, r.x2 + delta_x);
                }
                // Bottom seam of tile i: neighbor is below
                if (std::fabs(r.y2 - t_y2) < 1.0f)
                {
                    s_y2 = std::min(s_y2, r.y1 - delta_y);
                }
                // Top seam of tile i: neighbor is above
                if (std::fabs(r.y1 - t_y1) < 1.0f)
                {
                    s_y1 = std::max(s_y1, r.y2 + delta_y);
                }
            }

            safe_zones[i].x1 = s_x1;
            safe_zones[i].y1 = s_y1;
            safe_zones[i].x2 = s_x2;
            safe_zones[i].y2 = s_y2;
        }

        return safe_zones;
    }

    bool is_box_border(const GlobalDetection &box, const std::vector<TileSafeZone> &safe_zones)
    {
        if (safe_zones.empty() || box.tile_id < 0 || static_cast<size_t>(box.tile_id) >= safe_zones.size())
        {
            return true;
        }
        const auto &sz = safe_zones[box.tile_id];
        if (!sz.has_overlap)
        {
            return false;
        }
        const float bx1 = box.x;
        const float by1 = box.y;
        const float bx2 = box.x + box.w;
        const float by2 = box.y + box.h;

        // Interior check: strictly inside safe zone
        if (bx1 > sz.x1 && bx2 < sz.x2 && by1 > sz.y1 && by2 < sz.y2)
        {
            return false; // B_interior
        }
        return true; // is_border = true
    }

} // namespace

GlobalDetection remap_offsets(const Detection &local_det,
                              const TileRect &tile,
                              int model_target_size,
                              int tile_id)
{
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    if (model_target_size > 0)
    {
        scale_x = static_cast<float>(tile.w) / static_cast<float>(model_target_size);
        scale_y = static_cast<float>(tile.h) / static_cast<float>(model_target_size);
    }

    GlobalDetection global_det;
    global_det.x = static_cast<float>(tile.x) + local_det.x_local * scale_x;
    global_det.y = static_cast<float>(tile.y) + local_det.y_local * scale_y;
    global_det.w = local_det.w * scale_x;
    global_det.h = local_det.h * scale_y;
    global_det.conf = local_det.conf;
    global_det.class_id = local_det.class_id;
    global_det.tile_id = tile_id;
    return global_det;
}

GlobalDetection remap_offsets(const Detection &local_det,
                              const Rect &tile,
                              int model_target_size,
                              int tile_id)
{
    TileRect tr{tile.x, tile.y, tile.w, tile.h};
    return remap_offsets(local_det, tr, model_target_size, tile_id);
}

std::vector<GlobalDetection> remap_offsets(const std::vector<Detection> &local_dets,
                                           const TileRect &tile,
                                           int model_target_size,
                                           int tile_id)
{
    std::vector<GlobalDetection> global_dets;
    global_dets.reserve(local_dets.size());
    for (const auto &det : local_dets)
    {
        global_dets.push_back(remap_offsets(det, tile, model_target_size, tile_id));
    }
    return global_dets;
}

std::vector<GlobalDetection> remap_offsets(const std::vector<Detection> &local_dets,
                                           const Rect &tile,
                                           int model_target_size,
                                           int tile_id)
{
    TileRect tr{tile.x, tile.y, tile.w, tile.h};
    return remap_offsets(local_dets, tr, model_target_size, tile_id);
}

void remap_offsets(float x_local, float y_local, float w_local, float h_local,
                   const TileRect &tile,
                   float &x_global, float &y_global, float &w_global, float &h_global,
                   int model_target_size)
{
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    if (model_target_size > 0)
    {
        scale_x = static_cast<float>(tile.w) / static_cast<float>(model_target_size);
        scale_y = static_cast<float>(tile.h) / static_cast<float>(model_target_size);
    }

    x_global = static_cast<float>(tile.x) + x_local * scale_x;
    y_global = static_cast<float>(tile.y) + y_local * scale_y;
    w_global = w_local * scale_x;
    h_global = h_local * scale_y;
}

float calculate_iou(const GlobalDetection &a, const GlobalDetection &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 1e-7f)
    {
        return 0.0f;
    }

    return inter_area / union_area;
}

float calculate_diou(const GlobalDetection &a, const GlobalDetection &b)
{
    float iou = calculate_iou(a, b);

    // Box centers
    float center_a_x = a.x + a.w * 0.5f;
    float center_a_y = a.y + a.h * 0.5f;
    float center_b_x = b.x + b.w * 0.5f;
    float center_b_y = b.y + b.h * 0.5f;

    float d2 = (center_a_x - center_b_x) * (center_a_x - center_b_x) + (center_a_y - center_b_y) * (center_a_y - center_b_y);

    // Smallest enclosing box
    float enc_x1 = std::min(a.x, b.x);
    float enc_y1 = std::min(a.y, b.y);
    float enc_x2 = std::max(a.x + a.w, b.x + b.w);
    float enc_y2 = std::max(a.y + a.h, b.y + b.h);

    float enc_w = enc_x2 - enc_x1;
    float enc_h = enc_y2 - enc_y1;
    float c2 = enc_w * enc_w + enc_h * enc_h;

    if (c2 <= 1e-7f)
    {
        return iou;
    }

    return iou - (d2 / c2);
}

float calculate_containment(const GlobalDetection &a, const GlobalDetection &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float min_area = std::min(area_a, area_b);

    if (min_area <= 1e-7f)
    {
        return 0.0f;
    }

    return inter_area / min_area;
}

std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              const std::vector<TileRect> &tiles,
                                              float diou_threshold,
                                              float conf_threshold)
{
    // 1. Filter by confidence
    std::vector<GlobalDetection> candidates;
    candidates.reserve(detections.size());
    for (const auto &d : detections)
    {
        if (d.conf >= conf_threshold && d.w > 0.0f && d.h > 0.0f)
        {
            candidates.push_back(d);
        }
    }

    if (candidates.empty())
    {
        return {};
    }

    // 2. Precompute tile overlap zones and interior safe zones if tiles are provided
    std::vector<OverlapRect> overlap_regions;
    std::vector<std::vector<size_t>> tile_to_overlap;
    std::vector<TileSafeZone> safe_zones;
    if (!tiles.empty())
    {
        overlap_regions = compute_overlap_regions(tiles, &tile_to_overlap);
        safe_zones = compute_tile_safe_zones(tiles, overlap_regions, tile_to_overlap);
    }

    // 3. Group by class
    std::vector<int> classes;
    for (const auto &d : candidates)
    {
        if (std::find(classes.begin(), classes.end(), d.class_id) == classes.end())
        {
            classes.push_back(d.class_id);
        }
    }

    std::vector<GlobalDetection> result;

    for (int cls : classes)
    {
        // Step 1: AABB-based Boundary Isolation
        // Segregate candidate detections into interior boxes (B_interior) and border candidates
        std::vector<GlobalDetection> border_boxes;
        for (const auto &d : candidates)
        {
            if (d.class_id != cls)
                continue;

            if (is_box_border(d, safe_zones))
            {
                border_boxes.push_back(d);
            }
            else
            {
                // Box completely within tile interior safe zone: preserve without building pairs
                result.push_back(d);
            }
        }

        const size_t num_border = border_boxes.size();
        if (num_border == 0)
        {
            continue;
        }
        if (num_border == 1)
        {
            result.push_back(border_boxes[0]);
            continue;
        }

        // Step 2: Sweep-and-Prune on border candidates
        std::sort(border_boxes.begin(), border_boxes.end(),
                  [](const GlobalDetection &a, const GlobalDetection &b)
                  { return a.x < b.x; });

        std::vector<std::vector<size_t>> adj(num_border);

        for (size_t i = 0; i < num_border; ++i)
        {
            const auto &box_a = border_boxes[i];
            const float a_x1 = box_a.x;
            const float a_x2 = box_a.x + box_a.w;
            const float a_y1 = box_a.y;
            const float a_y2 = box_a.y + box_a.h;
            const float area_a = box_a.w * box_a.h;

            for (size_t j = i + 1; j < num_border; ++j)
            {
                const auto &box_b = border_boxes[j];

                // 1. Sweep-and-prune: early break on X-axis separation
                if (box_b.x >= a_x2)
                {
                    break;
                }

                // 2. Boxes originating from the same tile are never merged
                if (box_a.tile_id >= 0 && box_b.tile_id >= 0 && box_a.tile_id == box_b.tile_id)
                {
                    continue;
                }

                const float b_y1 = box_b.y;
                const float b_y2 = box_b.y + box_b.h;

                // 3. Fast 1D Y-axis overlap check
                if (b_y2 <= a_y1 || b_y1 >= a_y2)
                {
                    continue;
                }

                // 4. Quick intersection area calculation (Step 2.1)
                const float inter_w = std::min(a_x2, box_b.x + box_b.w) - std::max(a_x1, box_b.x);
                const float inter_h = std::min(a_y2, b_y2) - std::max(a_y1, b_y1);
                if (inter_w <= 0.0f || inter_h <= 0.0f)
                {
                    continue;
                }
                const float inter_area = inter_w * inter_h;

                const float area_b = box_b.w * box_b.h;
                const float min_area = std::min(area_a, area_b);
                if (min_area <= 1e-7f)
                {
                    continue;
                }

                const float containment = inter_area / min_area;
                bool is_duplicate = false;

                // Step 2.2: Fast Cross-Tile Containment (containment >= 0.72)
                if (containment >= 0.72f)
                {
                    is_duplicate = true;
                }
                // Step 2.3: Asymmetric weak cut-off part suppression (containment >= 0.50 & conf_weak <= 0.55 * conf_strong)
                else if (containment >= 0.50f)
                {
                    const float conf_weak = std::min(box_a.conf, box_b.conf);
                    const float conf_strong = std::max(box_a.conf, box_b.conf);
                    if (conf_weak <= 0.55f * conf_strong)
                    {
                        is_duplicate = true;
                    }
                }

                // Step 2 criterion fallback: DIoU threshold for overlapping full targets across seams
                if (!is_duplicate && diou_threshold > 0.0f)
                {
                    const float diou = calculate_diou(box_a, box_b);
                    float eff_thresh = diou_threshold;

                    if (box_a.tile_id >= 0 && box_b.tile_id >= 0 && box_a.tile_id != box_b.tile_id)
                    {
                        const float c_ax = a_x1 + box_a.w * 0.5f;
                        const float c_ay = a_y1 + box_a.h * 0.5f;
                        const float c_bx = box_b.x + box_b.w * 0.5f;
                        const float c_by = b_y1 + box_b.h * 0.5f;
                        const float d2 = (c_ax - c_bx) * (c_ax - c_bx) + (c_ay - c_by) * (c_ay - c_by);

                        const float enc_x1 = std::min(a_x1, box_b.x);
                        const float enc_y1 = std::min(a_y1, b_y1);
                        const float enc_x2 = std::max(a_x2, box_b.x + box_b.w);
                        const float enc_y2 = std::max(a_y2, b_y2);
                        const float c2 = (enc_x2 - enc_x1) * (enc_x2 - enc_x1) + (enc_y2 - enc_y1) * (enc_y2 - enc_y1);

                        if (c2 > 1e-7f)
                        {
                            const float ratio = std::min(1.0f, d2 / c2);
                            eff_thresh = diou_threshold * (0.70f + 0.30f * ratio);
                        }
                    }

                    if (diou >= eff_thresh)
                    {
                        is_duplicate = true;
                    }
                }

                if (is_duplicate)
                {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }

        // Find connected components (clusters)
        std::vector<bool> visited(num_border, false);

        for (size_t i = 0; i < num_border; ++i)
        {
            if (visited[i])
                continue;

            if (adj[i].empty())
            {
                visited[i] = true;
                result.push_back(border_boxes[i]);
                continue;
            }

            std::vector<size_t> cluster_indices;
            std::queue<size_t> q;
            q.push(i);
            visited[i] = true;

            while (!q.empty())
            {
                size_t curr = q.front();
                q.pop();
                cluster_indices.push_back(curr);

                for (size_t neighbor : adj[curr])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }

            if (cluster_indices.size() == 1)
            {
                result.push_back(border_boxes[cluster_indices[0]]);
            }
            else
            {
                // Step 3: Envelope Union to reconstruct full target dimensions across seams
                float min_x = border_boxes[cluster_indices[0]].x;
                float min_y = border_boxes[cluster_indices[0]].y;
                float max_x = min_x + border_boxes[cluster_indices[0]].w;
                float max_y = min_y + border_boxes[cluster_indices[0]].h;
                float max_conf = border_boxes[cluster_indices[0]].conf;

                for (size_t idx : cluster_indices)
                {
                    const auto &b = border_boxes[idx];
                    min_x = std::min(min_x, b.x);
                    min_y = std::min(min_y, b.y);
                    max_x = std::max(max_x, b.x + b.w);
                    max_y = std::max(max_y, b.y + b.h);
                    max_conf = std::max(max_conf, b.conf);
                }

                GlobalDetection merged;
                merged.x = min_x;
                merged.y = min_y;
                merged.w = max_x - min_x;
                merged.h = max_y - min_y;
                merged.conf = max_conf;
                merged.class_id = cls;
                merged.tile_id = -1; // Unified across seams
                result.push_back(merged);
            }
        }
    }

    // Sort final results descending by confidence
    std::sort(result.begin(), result.end(), [](const GlobalDetection &a, const GlobalDetection &b)
              { return a.conf > b.conf; });

    return result;
}

std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              float diou_threshold,
                                              float conf_threshold)
{
    static const std::vector<TileRect> empty_tiles;
    return cluster_diou_nms(detections, empty_tiles, diou_threshold, conf_threshold);
}

std::vector<GlobalDetection> cluster_diou_nms(const std::vector<GlobalDetection> &detections,
                                              const std::vector<Rect> &tiles,
                                              float diou_threshold,
                                              float conf_threshold)
{
    return cluster_diou_nms(detections, to_tile_rects(tiles), diou_threshold, conf_threshold);
}

std::string to_json_string(const std::vector<GlobalDetection> &detections, bool pretty)
{
    if (detections.empty())
    {
        return "[]";
    }

    std::ostringstream oss;
    if (pretty)
    {
        oss << "[\n";
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto &d = detections[i];
            oss << "  {\n";
            oss << "    \"x\": " << std::fixed << std::setprecision(2) << d.x << ",\n";
            oss << "    \"y\": " << std::fixed << std::setprecision(2) << d.y << ",\n";
            oss << "    \"w\": " << std::fixed << std::setprecision(2) << d.w << ",\n";
            oss << "    \"h\": " << std::fixed << std::setprecision(2) << d.h << ",\n";
            oss << "    \"conf\": " << std::fixed << std::setprecision(4) << d.conf << ",\n";
            oss << "    \"class_id\": " << d.class_id << "\n";
            oss << "  }";
            if (i + 1 < detections.size())
            {
                oss << ",";
            }
            oss << "\n";
        }
        oss << "]";
    }
    else
    {
        oss << "[";
        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto &d = detections[i];
            oss << "{\"x\":" << std::fixed << std::setprecision(2) << d.x
                << ",\"y\":" << std::fixed << std::setprecision(2) << d.y
                << ",\"w\":" << std::fixed << std::setprecision(2) << d.w
                << ",\"h\":" << std::fixed << std::setprecision(2) << d.h
                << ",\"conf\":" << std::fixed << std::setprecision(4) << d.conf
                << ",\"class_id\":" << d.class_id << "}";
            if (i + 1 < detections.size())
            {
                oss << ",";
            }
        }
        oss << "]";
    }

    return oss.str();
}
