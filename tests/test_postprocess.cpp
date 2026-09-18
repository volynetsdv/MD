#include "postprocess.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace
{

    int g_failures = 0;

#define CHECK(cond, msg)                                                 \
    do                                                                   \
    {                                                                    \
        if (!(cond))                                                     \
        {                                                                \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

#define CHECK_NEAR(val, expected, eps, msg)                                           \
    do                                                                                \
    {                                                                                 \
        if (std::fabs((val) - (expected)) > (eps))                                    \
        {                                                                             \
            std::fprintf(stderr, "FAIL: %s (got %f, expected %f, eps %f, line %d)\n", \
                         msg, static_cast<double>(val),                               \
                         static_cast<double>(expected),                               \
                         static_cast<double>(eps), __LINE__);                         \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

} // namespace

// ---------------------------------------------------------------------------
// 1. Main required test:
//    Simulate two detections of one object at the seam/overlap of two adjacent tiles
//    and verify their merging into a single global vector.
// ---------------------------------------------------------------------------
void test_seam_detection_merging()
{
    std::printf("--- Running test_seam_detection_merging ---\n");

    // 8K frame grid (7680 x 4320)
    // Two adjacent tiles along X:
    // Tile 0 covers [0, 512]
    // Tile 1 covers [400, 912] (step = 400, overlap zone is [400, 512])
    TileRect tile0{0, 0, 512, 512};
    TileRect tile1{400, 0, 512, 512};
    std::vector<TileRect> tiles = {tile0, tile1};

    // Ground truth object at global (450, 200, 60, 40)
    // In Tile 0 local space: (450 - 0, 200 - 0) = (450, 200)
    Detection det0_local{
        .x_local = 450.0f,
        .y_local = 200.0f,
        .w = 60.0f,
        .h = 40.0f,
        .conf = 0.85f,
        .class_id = 0};

    // In Tile 1 local space: (450 - 400, 200 - 0) = (50, 200)
    // With realistic detector jitter (+1 px on x, -1 px on y, slightly different size & conf)
    Detection det1_local{
        .x_local = 51.0f,
        .y_local = 199.0f,
        .w = 59.0f,
        .h = 41.0f,
        .conf = 0.92f,
        .class_id = 0};

    // Step 1: remap local coordinates to 8K global space
    GlobalDetection det0_global = remap_offsets(det0_local, tile0, 512, /*tile_id=*/0);
    GlobalDetection det1_global = remap_offsets(det1_local, tile1, 512, /*tile_id=*/1);

    CHECK_NEAR(det0_global.x, 450.0f, 0.01f, "det0_global x matches");
    CHECK_NEAR(det0_global.y, 200.0f, 0.01f, "det0_global y matches");
    CHECK_NEAR(det1_global.x, 451.0f, 0.01f, "det1_global x matches (400 + 51)");
    CHECK_NEAR(det1_global.y, 199.0f, 0.01f, "det1_global y matches (0 + 199)");

    // Step 2: verify DIoU before NMS
    float diou = calculate_diou(det0_global, det1_global);
    CHECK(diou > 0.8f, "DIoU between the two seam detections should be high (> 0.8)");

    // Step 3: run Cluster-DIoU-NMS
    std::vector<GlobalDetection> input_dets = {det0_global, det1_global};
    std::vector<GlobalDetection> merged = cluster_diou_nms(input_dets, tiles, /*diou_threshold=*/0.5f);

    // Verify fusion into a SINGLE global detection
    CHECK(merged.size() == 1, "Duplicate seam detections must be merged into 1 global detection");

    if (merged.size() == 1)
    {
        const auto &m = merged[0];
        // Confidence should be max(0.85, 0.92) = 0.92
        CHECK_NEAR(m.conf, 0.92f, 0.001f, "Merged confidence is max of cluster");
        CHECK(m.class_id == 0, "Merged class_id is 0");

        // Merged box should be confidence-weighted average:
        // total_weight = 0.85 + 0.92 = 1.77
        // x = (450 * 0.85 + 451 * 0.92) / 1.77 = 450.52
        // y = (200 * 0.85 + 199 * 0.92) / 1.77 = 199.48
        CHECK_NEAR(m.x, 450.52f, 0.1f, "Merged X is weighted average");
        CHECK_NEAR(m.y, 199.48f, 0.1f, "Merged Y is weighted average");
    }

    // Step 4: JSON serialization
    std::string json = to_json_string(merged);
    CHECK(!json.empty(), "JSON output must not be empty");
    CHECK(json.front() == '[' && json.back() == ']', "JSON must be array format");
    CHECK(json.find("\"x\":") != std::string::npos, "JSON must contain x coordinate");
    CHECK(json.find("\"y\":") != std::string::npos, "JSON must contain y coordinate");
    CHECK(json.find("\"conf\":") != std::string::npos, "JSON must contain conf");
    std::printf("Generated JSON: %s\n", json.c_str());
}

// ---------------------------------------------------------------------------
// 2. Test: Distinct objects in non-overlapping regions must NOT be merged
// ---------------------------------------------------------------------------
void test_distinct_objects_no_merge()
{
    std::printf("--- Running test_distinct_objects_no_merge ---\n");

    TileRect tile0{0, 0, 512, 512};
    TileRect tile1{400, 0, 512, 512};
    std::vector<TileRect> tiles = {tile0, tile1};

    // Object A inside tile 0 interior (x=100)
    GlobalDetection det_a{
        .x = 100.0f, .y = 100.0f, .w = 40.0f, .h = 40.0f, .conf = 0.90f, .class_id = 0, .tile_id = 0};

    // Object B inside tile 1 interior (x=700)
    GlobalDetection det_b{
        .x = 700.0f, .y = 100.0f, .w = 40.0f, .h = 40.0f, .conf = 0.88f, .class_id = 0, .tile_id = 1};

    auto result = cluster_diou_nms({det_a, det_b}, tiles, 0.5f);
    CHECK(result.size() == 2, "Distinct distant objects must both be preserved");
}

// ---------------------------------------------------------------------------
// 3. Test: Overlapping detections of DIFFERENT classes must NOT be merged
// ---------------------------------------------------------------------------
void test_different_classes_no_merge()
{
    std::printf("--- Running test_different_classes_no_merge ---\n");

    TileRect tile0{0, 0, 512, 512};
    TileRect tile1{400, 0, 512, 512};
    std::vector<TileRect> tiles = {tile0, tile1};

    // Both at seam, but class 0 vs class 1
    GlobalDetection det_car{
        .x = 450.0f, .y = 200.0f, .w = 50.0f, .h = 30.0f, .conf = 0.89f, .class_id = 0, .tile_id = 0};
    GlobalDetection det_person{
        .x = 452.0f, .y = 201.0f, .w = 48.0f, .h = 28.0f, .conf = 0.82f, .class_id = 1, .tile_id = 1};

    auto result = cluster_diou_nms({det_car, det_person}, tiles, 0.5f);
    CHECK(result.size() == 2, "Detections of different classes must not be merged");
}

// ---------------------------------------------------------------------------
// 4. Test: Edge tile scaling with model_target_size (W_t != M_selected)
// ---------------------------------------------------------------------------
void test_remap_scaling()
{
    std::printf("--- Running test_remap_scaling ---\n");

    // Partial boundary tile: width 256, height 512, scaled from model_target_size = 512
    TileRect edge_tile{7000, 1000, 256, 512};
    Detection local_det{
        .x_local = 100.0f,
        .y_local = 200.0f,
        .w = 50.0f,
        .h = 60.0f,
        .conf = 0.95f,
        .class_id = 2};

    GlobalDetection g = remap_offsets(local_det, edge_tile, /*model_target_size=*/512);

    // scale_x = 256 / 512 = 0.5, scale_y = 512 / 512 = 1.0
    // x = 7000 + 100 * 0.5 = 7050
    // y = 1000 + 200 * 1.0 = 1200
    // w = 50 * 0.5 = 25
    // h = 60 * 1.0 = 60
    CHECK_NEAR(g.x, 7050.0f, 0.01f, "Scaled X coordinate");
    CHECK_NEAR(g.y, 1200.0f, 0.01f, "Scaled Y coordinate");
    CHECK_NEAR(g.w, 25.0f, 0.01f, "Scaled width");
    CHECK_NEAR(g.h, 60.0f, 0.01f, "Scaled height");
}

// ---------------------------------------------------------------------------
// 5. Test: JSON formatting
// ---------------------------------------------------------------------------
void test_json_formatting()
{
    std::printf("--- Running test_json_formatting ---\n");

    std::vector<GlobalDetection> empty;
    CHECK(to_json_string(empty) == "[]", "Empty vector must produce []");

    std::vector<GlobalDetection> dets = {
        {.x = 10.0f, .y = 20.0f, .w = 30.0f, .h = 40.0f, .conf = 0.9123f, .class_id = 1}};

    std::string compact = to_json_string(dets, false);
    CHECK(compact.find("{\"x\":10.00,\"y\":20.00,\"w\":30.00,\"h\":40.00,\"conf\":0.9123,\"class_id\":1}") != std::string::npos,
          "Compact JSON matches format");

    std::string pretty = to_json_string(dets, true);
    CHECK(pretty.find("\n") != std::string::npos, "Pretty JSON has newlines");
}

int main()
{
    test_seam_detection_merging();
    test_distinct_objects_no_merge();
    test_different_classes_no_merge();
    test_remap_scaling();
    test_json_formatting();

    if (g_failures == 0)
    {
        std::printf("\nALL POSTPROCESS TESTS PASSED!\n");
        return 0;
    }

    std::fprintf(stderr, "\nFAILED with %d error(s).\n", g_failures);
    return 1;
}
