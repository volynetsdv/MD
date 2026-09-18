#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <queue>
#include <sstream>

namespace
{

    struct OverlapRect
    {
        float x1;
        float y1;
        float x2;
        float y2;
    };

    bool box_intersects_region(const GlobalDetection &box, const OverlapRect &r)
    {
        float bx1 = box.x;
        float by1 = box.y;
        float bx2 = box.x + box.w;
        float by2 = box.y + box.h;

        return !(bx2 <= r.x1 || bx1 >= r.x2 || by2 <= r.y1 || by1 >= r.y2);
    }

    std::vector<OverlapRect> compute_overlap_regions(const std::vector<TileRect> &tiles)
    {
        std::vector<OverlapRect> overlaps;
        const size_t n = tiles.size();
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
                    overlaps.push_back({x1, y1, x2, y2});
                }
            }
        }
        return overlaps;
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

    // 2. Precompute tile overlap zones if tiles are provided
    std::vector<OverlapRect> overlap_regions;
    if (!tiles.empty())
    {
        overlap_regions = compute_overlap_regions(tiles);
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
        std::vector<GlobalDetection> cls_boxes;
        for (const auto &d : candidates)
        {
            if (d.class_id == cls)
            {
                cls_boxes.push_back(d);
            }
        }

        // Sort descending by confidence
        std::sort(cls_boxes.begin(), cls_boxes.end(), [](const GlobalDetection &a, const GlobalDetection &b)
                  { return a.conf > b.conf; });

        const size_t num_boxes = cls_boxes.size();
        std::vector<std::vector<size_t>> adj(num_boxes);

        // Build adjacency graph for clustering
        for (size_t i = 0; i < num_boxes; ++i)
        {
            for (size_t j = i + 1; j < num_boxes; ++j)
            {
                const auto &bi = cls_boxes[i];
                const auto &bj = cls_boxes[j];

                // Check condition: merge only at overlap boundaries
                bool on_overlap_boundary = true;

                if (!overlap_regions.empty())
                {
                    // Must intersect at least one overlap zone between tiles
                    bool i_in_overlap = false;
                    bool j_in_overlap = false;
                    for (const auto &r : overlap_regions)
                    {
                        if (box_intersects_region(bi, r))
                            i_in_overlap = true;
                        if (box_intersects_region(bj, r))
                            j_in_overlap = true;
                        if (i_in_overlap || j_in_overlap)
                            break;
                    }
                    if (!i_in_overlap && !j_in_overlap)
                    {
                        on_overlap_boundary = false;
                    }
                }

                // If both boxes have valid tile_id, they must come from different tiles
                if (bi.tile_id >= 0 && bj.tile_id >= 0 && bi.tile_id == bj.tile_id)
                {
                    on_overlap_boundary = false;
                }

                if (!on_overlap_boundary)
                {
                    continue;
                }

                float diou = calculate_diou(bi, bj);
                if (diou >= diou_threshold)
                {
                    adj[i].push_back(j);
                    adj[j].push_back(i);
                }
            }
        }

        // Find connected components (clusters)
        std::vector<bool> visited(num_boxes, false);

        for (size_t i = 0; i < num_boxes; ++i)
        {
            if (visited[i])
                continue;

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
                result.push_back(cls_boxes[cluster_indices[0]]);
            }
            else
            {
                // Confidence-weighted merge
                float total_weight = 0.0f;
                float sum_x = 0.0f;
                float sum_y = 0.0f;
                float sum_w = 0.0f;
                float sum_h = 0.0f;
                float max_conf = 0.0f;

                for (size_t idx : cluster_indices)
                {
                    const auto &b = cls_boxes[idx];
                    float w = b.conf;
                    total_weight += w;
                    sum_x += b.x * w;
                    sum_y += b.y * w;
                    sum_w += b.w * w;
                    sum_h += b.h * w;
                    if (b.conf > max_conf)
                    {
                        max_conf = b.conf;
                    }
                }

                GlobalDetection merged;
                if (total_weight > 0.0f)
                {
                    merged.x = sum_x / total_weight;
                    merged.y = sum_y / total_weight;
                    merged.w = sum_w / total_weight;
                    merged.h = sum_h / total_weight;
                }
                else
                {
                    merged.x = cls_boxes[cluster_indices[0]].x;
                    merged.y = cls_boxes[cluster_indices[0]].y;
                    merged.w = cls_boxes[cluster_indices[0]].w;
                    merged.h = cls_boxes[cluster_indices[0]].h;
                }
                merged.conf = max_conf;
                merged.class_id = cls;
                merged.tile_id = -1; // Unified across tiles
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
