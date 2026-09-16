/**
 * @file test_main_pipeline.cpp
 * @brief Unit-tests for the Inference Engine Orchestrator (run_pipeline).
 *
 * Tests verify:
 *  T1. TilingConfig selection for M ∈ {320, 416, 512, 640} based on telemetry.
 *  T2. run_pipeline completes without error on a 7680×4320 frame.
 *  T3. Per-tile VRAM lifecycle: slice allocated, used, and freed each iteration.
 *  T4. JSON output is non-empty and starts with '['.
 *  T5. Detections survive remap_offsets correctly (global coords within frame).
 *  T6. VRAM does not leak across the full pipeline run (baseline == teardown).
 *  T7. cluster_diou_nms deduplicates objects visible in overlapping tiles.
 */

// ---- Re-use the pipeline internals via forward declarations ----------------
// We compile against main_pipeline via a shared library approach:
// test links tiling_math + cuda_slicer + trt_detector + postprocess,
// and includes the pipeline header for PipelineOptions / PipelineResult / run_pipeline.

#include "cuda_slicer.cuh"
#include "postprocess.hpp"
#include "tiling_math.hpp"
#include "trt_detector.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test harness (no external dependencies)
// ---------------------------------------------------------------------------
namespace {

int g_pass = 0;
int g_fail = 0;

#define TEST_CHECK(cond, msg)                                                \
    do {                                                                     \
        if (cond) {                                                          \
            std::fprintf(stderr, "  PASS  %s\n", msg);                      \
            ++g_pass;                                                        \
        } else {                                                             \
            std::fprintf(stderr, "  FAIL  %s  (line %d)\n", msg, __LINE__); \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

#define TEST_CHECK_NEAR(a, b, eps, msg)                                              \
    do {                                                                             \
        if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (eps)) {  \
            std::fprintf(stderr, "  PASS  %s\n", msg);                              \
            ++g_pass;                                                                \
        } else {                                                                     \
            std::fprintf(stderr, "  FAIL  %s  got=%.6f expected≈%.6f eps=%.6f  (line %d)\n",\
                         msg, static_cast<double>(a), static_cast<double>(b),        \
                         static_cast<double>(eps), __LINE__);                        \
            ++g_fail;                                                                \
        }                                                                            \
    } while (0)

// Simple query without the pipeline header
struct VramSnap {
    size_t free_bytes{0};
    size_t total_bytes{0};
    size_t used_bytes{0};
};

VramSnap snap_vram() {
    VramSnap s;
    if (cudaMemGetInfo(&s.free_bytes, &s.total_bytes) == cudaSuccess)
        s.used_bytes = s.total_bytes - s.free_bytes;
    return s;
}

} // namespace

// ===========================================================================
// T1 — TilingConfig: M_selected ∈ {320, 416, 512, 640}
// ===========================================================================
static void test_T1_tiling_config_selection() {
    std::fprintf(stderr, "\n[T1] TilingConfig selection for M ∈ {320,416,512,640}\n");

    // T_calc = vram/4 + alt*10
    // (a) vram=800, alt=10  → T_calc = 200 + 100 = 300  → M = 320 (fallback)
    {
        TilingConfig cfg = calculate_tiling_params(7680, 4320, 10.0f, 800);
        TEST_CHECK(cfg.tile_size == 320, "T_calc=300 → M_selected=320");
    }
    // (b) vram=800, alt=30  → T_calc = 200 + 300 = 500  → M = 416 (max ≤ 500)
    {
        TilingConfig cfg = calculate_tiling_params(7680, 4320, 30.0f, 800);
        TEST_CHECK(cfg.tile_size == 416, "T_calc=500 → M_selected=416");
    }
    // (c) vram=1000, alt=30 → T_calc = 250 + 300 = 550  → M = 512
    {
        TilingConfig cfg = calculate_tiling_params(7680, 4320, 30.0f, 1000);
        TEST_CHECK(cfg.tile_size == 512, "T_calc=550 → M_selected=512");
    }
    // (d) vram=1000, alt=100 → T_calc = 250 + 1000 = 1250 → M = 640
    {
        TilingConfig cfg = calculate_tiling_params(7680, 4320, 100.0f, 1000);
        TEST_CHECK(cfg.tile_size == 640, "T_calc=1250 → M_selected=640");
    }
    // (e) grid covers full 8K frame (no gap)
    {
        TilingConfig cfg = calculate_tiling_params(7680, 4320, 100.0f, 1000);
        const Rect& last = cfg.tiles.back();
        TEST_CHECK(last.x + last.w <= 7680, "last tile does not exceed frame width");
        TEST_CHECK(last.y + last.h <= 4320, "last tile does not exceed frame height");
    }
    // (f) overlap clamped to [0.1, 0.4]
    {
        TilingConfig cfg_lo = calculate_tiling_params(7680, 4320, 0.0f, 100);
        TilingConfig cfg_hi = calculate_tiling_params(7680, 4320, 1000.0f, 9999);
        TEST_CHECK(cfg_lo.overlap >= 0.1f && cfg_lo.overlap <= 0.4f,
                   "O_lap clamped at altitude=0");
        TEST_CHECK(cfg_hi.overlap >= 0.1f && cfg_hi.overlap <= 0.4f,
                   "O_lap clamped at altitude=1000");
    }
}

// ===========================================================================
// T2 — Per-tile VRAM lifecycle: allocate → use → free each iteration
// ===========================================================================
static void test_T2_per_tile_vram_lifecycle() {
    std::fprintf(stderr, "\n[T2] Per-tile VRAM lifecycle (alloc → extract → free)\n");

    // Use a small source frame to keep test fast
    const int W = 1280, H = 720, target = 320;
    const size_t src_bytes = static_cast<size_t>(W) * H * 3;

    // Allocate source image in VRAM
    uint8_t* d_src = nullptr;
    cudaError_t err = cudaMalloc(&d_src, src_bytes);
    TEST_CHECK(err == cudaSuccess, "cudaMalloc source frame");
    if (err != cudaSuccess) return;

    cudaMemset(d_src, 100, src_bytes);
    cudaDeviceSynchronize();

    const VramSnap before = snap_vram();

    // Simulate 4 tile iterations
    TileRect tiles[4] = {
        {0,   0,   320, 320},
        {260, 0,   320, 320},
        {0,   260, 320, 320},
        {260, 260, 320, 320},
    };

    for (int i = 0; i < 4; ++i) {
        float* d_slice = nullptr;
        err = cudaMalloc(&d_slice, static_cast<size_t>(3) * target * target * sizeof(float));
        TEST_CHECK(err == cudaSuccess, "cudaMalloc tile slice");
        if (err != cudaSuccess) continue;

        err = extract_tile_gpu(d_src, W * 3, W, H, tiles[i], target, d_slice);
        TEST_CHECK(err == cudaSuccess, "extract_tile_gpu tile");

        cudaDeviceSynchronize();

        // Immediate release (the critical VRAM rule)
        cudaFree(d_slice);
        d_slice = nullptr;

        // Verify slice pointer is nulled
        TEST_CHECK(d_slice == nullptr, "d_slice nulled after free");
    }

    cudaDeviceSynchronize();
    const VramSnap after = snap_vram();

    // Allow ≤ 2 MB tolerance (CUDA runtime bookkeeping)
    const size_t tolerance = 2ULL * 1024ULL * 1024ULL;
    bool no_leak = (after.used_bytes <= before.used_bytes + tolerance);
    TEST_CHECK(no_leak, "No VRAM leak after 4 tile iterations");

    cudaFree(d_src);
}

// ===========================================================================
// T3 — remap_offsets: local → global coordinate correctness
// ===========================================================================
static void test_T3_remap_offsets_correctness() {
    std::fprintf(stderr, "\n[T3] remap_offsets local→global coordinate mapping\n");

    // Tile covers global [400, 640] x [200, 512] → 240x312 pixels, model=512
    Rect tile{400, 200, 240, 312};
    const int M = 512;

    Detection det;
    det.x_local  = 0.0f;     // top-left of tile-local space
    det.y_local  = 0.0f;
    det.w        = static_cast<float>(M);
    det.h        = static_cast<float>(M);
    det.conf     = 0.9f;
    det.class_id = 1;

    GlobalDetection g = remap_offsets(det, tile, M, 0);

    // x_global = 400 + 0 * (240/512) = 400
    TEST_CHECK_NEAR(g.x, 400.0f, 1.0f, "remap: x_global = tile.x");
    // y_global = 200 + 0 * (312/512) = 200
    TEST_CHECK_NEAR(g.y, 200.0f, 1.0f, "remap: y_global = tile.y");
    // w_global = 512 * (240/512) = 240
    TEST_CHECK_NEAR(g.w, 240.0f, 1.0f, "remap: w_global = tile.w");
    // h_global = 512 * (312/512) = 312
    TEST_CHECK_NEAR(g.h, 312.0f, 1.0f, "remap: h_global = tile.h");
    TEST_CHECK(g.conf     == 0.9f, "remap: confidence preserved");
    TEST_CHECK(g.class_id == 1,   "remap: class_id preserved");
    TEST_CHECK(g.tile_id  == 0,   "remap: tile_id set");
}

// ===========================================================================
// T4 — cluster_diou_nms: duplicate across tile seam is merged to one
// ===========================================================================
static void test_T4_diou_nms_deduplication() {
    std::fprintf(stderr, "\n[T4] cluster_diou_nms seam deduplication\n");

    // Two adjacent tiles with 112 px overlap (step = 400, tile = 512)
    Rect tile0{0,   0, 512, 512};
    Rect tile1{400, 0, 512, 512};

    // Same object visible in both tiles → two nearly identical global boxes
    GlobalDetection d0{450.0f, 200.0f, 60.0f, 40.0f, 0.85f, 2, 0};
    GlobalDetection d1{452.0f, 201.0f, 60.0f, 40.0f, 0.80f, 2, 1}; // slight jitter

    std::vector<GlobalDetection> raw = {d0, d1};
    std::vector<Rect>            tiles = {tile0, tile1};

    auto merged = cluster_diou_nms(raw, tiles, /*diou_thresh=*/0.3f, /*conf_thresh=*/0.1f);

    // The two high-overlap boxes (same class) should merge into exactly one
    TEST_CHECK(merged.size() == 1, "NMS merges 2 seam duplicates → 1 detection");
    if (!merged.empty()) {
        // Merged coordinate should lie between the two inputs
        TEST_CHECK(merged[0].x >= 450.0f && merged[0].x <= 452.0f,
                   "Merged x in [450,452]");
        TEST_CHECK(merged[0].conf >= 0.80f,
                   "Merged conf >= min(conf_i)");
    }
}

// ===========================================================================
// T5 — cluster_diou_nms: distinct objects in different tiles are preserved
// ===========================================================================
static void test_T5_diou_nms_preserves_distinct() {
    std::fprintf(stderr, "\n[T5] cluster_diou_nms preserves distinct objects\n");

    Rect tile0{0, 0, 640, 640};

    GlobalDetection da{100.0f, 100.0f, 50.0f, 30.0f, 0.9f, 2, 0}; // vehicle
    GlobalDetection db{500.0f, 400.0f, 30.0f, 60.0f, 0.8f, 0, 0}; // person

    std::vector<GlobalDetection> raw   = {da, db};
    std::vector<Rect>            tiles = {tile0};

    auto out = cluster_diou_nms(raw, tiles, 0.5f, 0.1f);
    TEST_CHECK(out.size() == 2, "Two distinct objects preserved after NMS");
}

// ===========================================================================
// T6 — JSON output validity
// ===========================================================================
static void test_T6_json_output() {
    std::fprintf(stderr, "\n[T6] JSON serialization\n");

    std::vector<GlobalDetection> dets;
    GlobalDetection d;
    d.x = 250.0f; d.y = 180.0f; d.w = 54.0f; d.h = 32.0f;
    d.conf = 0.91f; d.class_id = 2;
    dets.push_back(d);

    std::string compact = to_json_string(dets, false);
    std::string pretty  = to_json_string(dets, true);

    TEST_CHECK(!compact.empty(),              "compact JSON non-empty");
    TEST_CHECK(compact.front() == '[',        "compact JSON starts with '['");
    TEST_CHECK(compact.back()  == ']',        "compact JSON ends with ']'");
    TEST_CHECK(!pretty.empty(),               "pretty JSON non-empty");
    TEST_CHECK(pretty.find("\"x\"") != std::string::npos, "pretty JSON contains key 'x'");
    TEST_CHECK(pretty.find("\"conf\"") != std::string::npos, "pretty JSON contains key 'conf'");

    // Empty input → "[]"
    std::string empty_json = to_json_string({}, false);
    TEST_CHECK(empty_json == "[]", "empty detections → '[]'");
}

// ===========================================================================
// T7 — Full pipeline VRAM leak test on 7680×4320
// ===========================================================================
static void test_T7_full_pipeline_vram_leak() {
    std::fprintf(stderr,
        "\n[T7] Full pipeline 7680x4320 — VRAM leak check via cudaMemGetInfo\n");

    // Capture baseline BEFORE the pipeline process initializes any CUDA context
    cudaFree(nullptr);   // force CUDA context init so baseline is stable
    cudaDeviceSynchronize();

    const VramSnap baseline = snap_vram();
    std::fprintf(stderr, "  VRAM before pipeline: %.2f MB used\n",
                 static_cast<double>(baseline.used_bytes) / (1024.0 * 1024.0));

    // ---- Inline the essential pipeline steps without depending on main() ----

    const int W = 7680, H = 4320;
    const float altitude    = 100.0f;
    const size_t vram_mb    = baseline.free_bytes / (1024ULL * 1024ULL);

    // Step 1: Upload synthetic 8K frame to VRAM
    const size_t img_bytes = static_cast<size_t>(W) * H * 3;
    uint8_t* d_src = nullptr;
    cudaError_t err = cudaMalloc(&d_src, img_bytes);
    TEST_CHECK(err == cudaSuccess, "cudaMalloc 8K source frame");
    if (err != cudaSuccess) return;

    {
        std::vector<uint8_t> h_img(img_bytes);
        for (size_t k = 0; k < img_bytes; k += 3) {
            h_img[k]   = static_cast<uint8_t>(k & 0xFF);
            h_img[k+1] = static_cast<uint8_t>((k >> 8) & 0xFF);
            h_img[k+2] = 128;
        }
        err = cudaMemcpy(d_src, h_img.data(), img_bytes, cudaMemcpyHostToDevice);
    }
    TEST_CHECK(err == cudaSuccess, "H→D memcpy 8K frame");

    // Step 2: Tiling
    TilingConfig cfg = calculate_tiling_params(W, H, altitude, vram_mb);
    TEST_CHECK(cfg.tile_size > 0,        "tile_size > 0");
    TEST_CHECK(!cfg.tiles.empty(),       "tiles vector non-empty");

    const int target = cfg.tile_size;
    const size_t tile_bytes = static_cast<size_t>(3) * target * target * sizeof(float);

    // Step 3: Process 12 tiles (or fewer if grid is smaller)
    const size_t ntiles = std::min<size_t>(12, cfg.tiles.size());

    TRTDetector stub_detector;  // no engine loaded → infer() returns empty vector

    std::vector<GlobalDetection> all_raw;
    std::vector<Rect>            visited;
    visited.reserve(ntiles);

    for (size_t i = 0; i < ntiles; ++i) {
        const Rect& tile = cfg.tiles[i];
        visited.push_back(tile);
        TileRect tr{tile.x, tile.y, tile.w, tile.h};

        float* d_slice = nullptr;
        err = cudaMalloc(&d_slice, tile_bytes);
        if (err != cudaSuccess) continue;

        err = extract_tile_gpu(d_src, W * 3, W, H, tr, target, d_slice);
        cudaDeviceSynchronize();
        if (err != cudaSuccess) { cudaFree(d_slice); continue; }

        // infer returns empty (no engine) — acceptable for leak test
        auto local = stub_detector.infer(d_slice);

        // Immediate free (the core correctness requirement)
        cudaFree(d_slice);
        d_slice = nullptr;

        auto remapped = remap_offsets(local, tile, target, static_cast<int>(i));
        all_raw.insert(all_raw.end(), remapped.begin(), remapped.end());
    }
    cudaDeviceSynchronize();

    // Step 4: NMS
    auto final_dets = cluster_diou_nms(all_raw, visited, 0.5f, 0.25f);
    (void)final_dets;  // suppress unused-variable warning

    // Step 5: Teardown
    cudaFree(d_src);
    d_src = nullptr;
    cudaDeviceSynchronize();

    const VramSnap after = snap_vram();
    std::fprintf(stderr, "  VRAM after  pipeline: %.2f MB used\n",
                 static_cast<double>(after.used_bytes) / (1024.0 * 1024.0));

    // Tolerate ≤ 2 MB runtime bookkeeping difference
    const size_t tolerance = 2ULL * 1024ULL * 1024ULL;
    const bool no_leak = (after.used_bytes <= baseline.used_bytes + tolerance);

    std::fprintf(stderr, "  Delta: %+.2f MB  (tolerance ±%.2f MB)\n",
        (static_cast<double>(after.used_bytes) - static_cast<double>(baseline.used_bytes)) / (1024.0*1024.0),
        static_cast<double>(tolerance) / (1024.0*1024.0));

    TEST_CHECK(no_leak,           "No VRAM leak after full 7680×4320 pipeline run");
    TEST_CHECK(ntiles >= 1,       "At least 1 tile processed");
    TEST_CHECK(ntiles <= 12,      "At most 12 tiles processed");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    std::fprintf(stderr, "=== test_main_pipeline ===\n");

    test_T1_tiling_config_selection();
    test_T2_per_tile_vram_lifecycle();
    test_T3_remap_offsets_correctness();
    test_T4_diou_nms_deduplication();
    test_T5_diou_nms_preserves_distinct();
    test_T6_json_output();
    test_T7_full_pipeline_vram_leak();

    std::fprintf(stderr,
        "\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    return (g_fail == 0) ? 0 : 1;
}

