/**
 * @file main_pipeline.cpp
 * @brief Inference Engine Orchestrator — integrates all libtiling_core modules:
 *        tiling_math, cuda_slicer, trt_detector, postprocess.
 *
 * Flow:
 *  1. Ingest 8K frame (or synthetic frame) into VRAM.
 *  2. calculate_tiling_params()  → TilingConfig (M_selected ∈ {320,416,512,640})
 *  3. TRTDetectorPool selects the matching TRTDetector.
 *  4. For each tile (default 10–12):
 *     a. cudaMalloc + extract_tile_gpu  (Zero-Copy GPU slice)
 *     b. TRTDetector::infer()           (forward pass)
 *     c. cudaFree(d_slice)              (immediate VRAM release)
 *     d. remap_offsets()                (local → global coordinates)
 *  5. cluster_diou_nms()               (deduplication across tile seams)
 *  6. to_json_string()  → stdout       (lightweight metadata only)
 */

#include "cuda_slicer.cuh"
#include "postprocess.hpp"
#include "tiling_math.hpp"
#include "trt_detector.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// VRAM Telemetry
// ============================================================================

struct VramInfo
{
    size_t free_bytes{0};
    size_t total_bytes{0};
    size_t used_bytes{0};

    double free_mb() const { return static_cast<double>(free_bytes) / (1024.0 * 1024.0); }
    double total_mb() const { return static_cast<double>(total_bytes) / (1024.0 * 1024.0); }
    double used_mb() const { return static_cast<double>(used_bytes) / (1024.0 * 1024.0); }
};

VramInfo query_vram()
{
    VramInfo info;
    if (cudaMemGetInfo(&info.free_bytes, &info.total_bytes) == cudaSuccess)
        info.used_bytes = info.total_bytes - info.free_bytes;
    return info;
}

static void print_vram_status(const std::string &stage)
{
    VramInfo v = query_vram();
    std::cerr << "[VRAM] " << stage << ": Used = "
              << std::fixed << std::setprecision(2) << v.used_mb() << " MB  ("
              << v.free_mb() << " MB free / "
              << v.total_mb() << " MB total)\n";
}

// ============================================================================
// Synthetic Frame Generator (no disk I/O)
// ============================================================================

static void generate_synthetic_frame(uint8_t *h_img, int width, int height)
{
    std::cerr << "[Pipeline] Generating synthetic frame ("
              << width << "x" << height << ")...\n";
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uint8_t *px = h_img + (static_cast<size_t>(y) * width + x) * 3;
            px[0] = static_cast<uint8_t>((x * 255) / std::max(width - 1, 1));  // B
            px[1] = static_cast<uint8_t>((y * 255) / std::max(height - 1, 1)); // G
            px[2] = static_cast<uint8_t>(128 + ((x ^ y) & 63));                // R
        }
    }
}

// ============================================================================
// Synthetic Detections (validation fallback when no .engine is loaded)
// ============================================================================

struct SyntheticTarget
{
    float gx, gy, gw, gh;
    int class_id;
    float conf;
};

// Five objects spread across the 8K frame (global pixel coordinates).
static const SyntheticTarget kSyntheticTargets[] = {
    {250.0f, 180.0f, 54.0f, 32.0f, 2, 0.91f},   // vehicle
    {340.0f, 290.0f, 62.0f, 38.0f, 2, 0.88f},   // vehicle
    {390.0f, 370.0f, 48.0f, 28.0f, 2, 0.94f},   // vehicle (in tile-overlap zone)
    {720.0f, 410.0f, 22.0f, 44.0f, 0, 0.86f},   // person
    {1250.0f, 800.0f, 110.0f, 75.0f, 4, 0.95f}, // UAV
};
static constexpr int kNumSyntheticTargets =
    static_cast<int>(sizeof(kSyntheticTargets) / sizeof(kSyntheticTargets[0]));

static std::vector<Detection> generate_synthetic_tile_detections(
    const Rect &tile, int model_target_size)
{
    std::vector<Detection> dets;
    const float scale_x = static_cast<float>(model_target_size) / static_cast<float>(tile.w);
    const float scale_y = static_cast<float>(model_target_size) / static_cast<float>(tile.h);

    for (int k = 0; k < kNumSyntheticTargets; ++k)
    {
        const SyntheticTarget &t = kSyntheticTargets[k];
        // Intersection test (tile-local AABB vs global target AABB)
        if (t.gx + t.gw <= tile.x || t.gx >= tile.x + tile.w ||
            t.gy + t.gh <= tile.y || t.gy >= tile.y + tile.h)
            continue;

        Detection d;
        d.x_local = (t.gx - static_cast<float>(tile.x)) * scale_x;
        d.y_local = (t.gy - static_cast<float>(tile.y)) * scale_y;
        d.w = t.gw * scale_x;
        d.h = t.gh * scale_y;
        d.conf = t.conf;
        d.class_id = t.class_id;
        dets.push_back(d);
    }
    return dets;
}

// ============================================================================
// TRTDetector Pool (models: 320, 416, 512, 640)
// ============================================================================

class TRTDetectorPool
{
public:
    TRTDetectorPool()
    {
        for (int s : {320, 416, 512, 640})
            detectors_[s] = std::make_unique<TRTDetector>();
    }

    /// Try to load a .engine file for the given square resolution.
    bool loadModel(int size, const std::string &path)
    {
        auto it = detectors_.find(size);
        if (it == detectors_.end())
            return false;

        bool ok = it->second->loadEngine(path);
        std::cerr << "[DetectorPool] " << (ok ? "Loaded" : "FAILED to load")
                  << " engine " << size << "x" << size << " from " << path << "\n";
        return ok;
    }

    /// Scan a directory for engine files using canonical naming patterns.
    void scanDirectory(const std::string &dir)
    {
        if (dir.empty())
            return;
        for (int s : {320, 416, 512, 640})
        {
            const std::string ss = std::to_string(s);
            for (const std::string &name : {
                     "model_" + ss + ".engine",
                     "yolo_" + ss + ".engine",
                     ss + ".engine"})
            {
                std::ifstream f(dir + "/" + name, std::ios::binary);
                if (f.good())
                {
                    f.close();
                    loadModel(s, dir + "/" + name);
                    break;
                }
            }
        }
    }

    /// Returns the detector instance for the given resolution (always non-null
    /// for supported sizes; nullptr for unsupported sizes).
    TRTDetector *getDetector(int size)
    {
        auto it = detectors_.find(size);
        return (it != detectors_.end()) ? it->second.get() : nullptr;
    }

    /// True iff an engine was successfully loaded for this resolution.
    bool hasLoadedEngine(int size) const
    {
        auto it = detectors_.find(size);
        return (it != detectors_.end()) && (it->second->getDeviceMemoryUsage() > 0);
    }

private:
    std::unordered_map<int, std::unique_ptr<TRTDetector>> detectors_;
};

// ============================================================================
// Pipeline Options & CLI Parser
// ============================================================================

struct PipelineOptions
{
    int width{7680};
    int height{4320};
    float altitude{100.0f};
    size_t vram_mb_override{0}; ///< 0 = auto-detect from cudaMemGetInfo
    int max_tiles{12};          ///< 0 = process full grid
    std::string models_dir{"./models"};
    std::string engine_320{""};
    std::string engine_416{""};
    std::string engine_512{""};
    std::string engine_640{""};
    float diou_thresh{0.5f};
    float conf_thresh{0.25f};
    bool pretty_json{false};
    bool verbose{false};
};

static PipelineOptions parse_args(int argc, char *argv[])
{
    PipelineOptions opt;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> const char *
        { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--width")
            opt.width = std::atoi(next());
        else if (a == "--height")
            opt.height = std::atoi(next());
        else if (a == "--altitude")
            opt.altitude = static_cast<float>(std::atof(next()));
        else if (a == "--vram-mb")
            opt.vram_mb_override = static_cast<size_t>(std::atoll(next()));
        else if (a == "--max-tiles")
            opt.max_tiles = std::atoi(next());
        else if (a == "--models-dir")
            opt.models_dir = next();
        else if (a == "--engine-320")
            opt.engine_320 = next();
        else if (a == "--engine-416")
            opt.engine_416 = next();
        else if (a == "--engine-512")
            opt.engine_512 = next();
        else if (a == "--engine-640")
            opt.engine_640 = next();
        else if (a == "--diou-thresh")
            opt.diou_thresh = static_cast<float>(std::atof(next()));
        else if (a == "--conf-thresh")
            opt.conf_thresh = static_cast<float>(std::atof(next()));
        else if (a == "--pretty")
            opt.pretty_json = true;
        else if (a == "--verbose")
            opt.verbose = true;
        else if (a == "--help" || a == "-h")
        {
            std::cerr << "Usage: main_pipeline [options]\n"
                         "  --width       <int>    Image width        (default: 7680)\n"
                         "  --height      <int>    Image height       (default: 4320)\n"
                         "  --altitude    <float>  Flight altitude m  (default: 100.0)\n"
                         "  --vram-mb     <int>    Override VRAM MB   (default: auto)\n"
                         "  --max-tiles   <int>    Tiles to process   (default: 12, 0=all)\n"
                         "  --models-dir  <dir>    Engine search dir  (default: ./models)\n"
                         "  --engine-320  <path>   Path to 320 engine\n"
                         "  --engine-416  <path>   Path to 416 engine\n"
                         "  --engine-512  <path>   Path to 512 engine\n"
                         "  --engine-640  <path>   Path to 640 engine\n"
                         "  --diou-thresh <float>  DIoU NMS threshold (default: 0.5)\n"
                         "  --conf-thresh <float>  Confidence cutoff  (default: 0.25)\n"
                         "  --pretty               Pretty-print JSON\n"
                         "  --verbose              Verbose tile log\n";
            std::exit(0);
        }
    }
    return opt;
}

// ============================================================================
// Pipeline Result (returned by run_pipeline for testability)
// ============================================================================

struct PipelineResult
{
    int return_code{0}; ///< 0 = success
    size_t tiles_processed{0};
    size_t raw_detections{0};
    size_t final_detections{0};
    size_t vram_baseline_bytes{0};       ///< VRAM used before any allocation
    size_t vram_after_teardown_bytes{0}; ///< VRAM used after full cleanup
    double loop_ms{0.0};
    double nms_ms{0.0};
    double total_ms{0.0};
    std::string json_output{};
};

// ============================================================================
// run_pipeline() — the full orchestrator (callable from tests and main)
// ============================================================================

PipelineResult run_pipeline(const PipelineOptions &opt,
                            uint8_t *d_external_frame = nullptr)
{
    PipelineResult result;
    const auto t_start = std::chrono::high_resolution_clock::now();

    std::cerr << "=========================================================\n"
              << "  Inference Engine Orchestrator (Autonomous 8K Pipeline)\n"
              << "=========================================================\n";

    // ------------------------------------------------------------------
    // Step 0: VRAM baseline
    // ------------------------------------------------------------------
    print_vram_status("Baseline");
    result.vram_baseline_bytes = query_vram().used_bytes;

    size_t vram_available_mb = opt.vram_mb_override;
    if (vram_available_mb == 0)
        vram_available_mb = query_vram().free_bytes / (1024ULL * 1024ULL);

    std::cerr << "[Pipeline] VRAM available for tiling : " << vram_available_mb << " MB\n"
              << "[Pipeline] Altitude telemetry        : " << opt.altitude << " m\n"
              << "[Pipeline] Frame size                : " << opt.width << "x" << opt.height << "\n";

    // ------------------------------------------------------------------
    // Step 1: Upload 8K frame to VRAM (zero disk I/O)
    // ------------------------------------------------------------------
    const size_t img_bytes = static_cast<size_t>(opt.width) * opt.height * 3;
    std::cerr << "[Pipeline] Allocating source buffer ("
              << (img_bytes / (1024ULL * 1024ULL)) << " MB) in VRAM...\n";

    uint8_t *d_src_img = d_external_frame; // use caller-supplied frame if provided
    bool owns_frame = (d_external_frame == nullptr);

    if (owns_frame)
    {
        cudaError_t err = cudaMalloc(&d_src_img, img_bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "[Pipeline] FATAL: cudaMalloc source frame: "
                      << cudaGetErrorString(err) << "\n";
            result.return_code = 1;
            return result;
        }

        // Synthesize image on host, upload, then immediately free host buffer.
        {
            std::vector<uint8_t> h_img(img_bytes);
            generate_synthetic_frame(h_img.data(), opt.width, opt.height);
            err = cudaMemcpy(d_src_img, h_img.data(), img_bytes, cudaMemcpyHostToDevice);
        } // h_img freed here

        if (err != cudaSuccess)
        {
            std::cerr << "[Pipeline] FATAL: cudaMemcpy H->D: "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(d_src_img);
            result.return_code = 1;
            return result;
        }
    }
    print_vram_status("After frame upload");

    // ------------------------------------------------------------------
    // Step 2: Dynamic tiling via libtiling_core
    // ------------------------------------------------------------------
    TilingConfig tiling_cfg = calculate_tiling_params(
        opt.width, opt.height, opt.altitude, vram_available_mb);

    std::cerr << "[Pipeline] TilingConfig:\n"
              << "  Selected tile resolution : " << tiling_cfg.tile_size
              << "x" << tiling_cfg.tile_size << " (M_selected)\n"
              << "  Overlap (O_lap)          : " << tiling_cfg.overlap << "\n"
              << "  Grid                     : " << tiling_cfg.grid_cols
              << " cols x " << tiling_cfg.grid_rows << " rows\n"
              << "  Total tiles in grid      : " << tiling_cfg.tiles.size() << "\n";

    // ------------------------------------------------------------------
    // Step 3: Initialize model pool & select detector for M_selected
    // ------------------------------------------------------------------
    TRTDetectorPool pool;
    pool.scanDirectory(opt.models_dir);
    if (!opt.engine_320.empty())
        pool.loadModel(320, opt.engine_320);
    if (!opt.engine_416.empty())
        pool.loadModel(416, opt.engine_416);
    if (!opt.engine_512.empty())
        pool.loadModel(512, opt.engine_512);
    if (!opt.engine_640.empty())
        pool.loadModel(640, opt.engine_640);

    const int target_size = tiling_cfg.tile_size;
    TRTDetector *detector = pool.getDetector(target_size);
    if (!detector)
    {
        std::cerr << "[Pipeline] FATAL: unsupported tile size " << target_size << "\n";
        if (owns_frame)
            cudaFree(d_src_img);
        result.return_code = 1;
        return result;
    }

    const bool has_engine = pool.hasLoadedEngine(target_size);
    if (!has_engine)
    {
        std::cerr << "[Pipeline] Notice: no engine loaded — "
                     "using synthetic detections for validation.\n";
    }

    // ------------------------------------------------------------------
    // Step 4: Per-tile inference loop
    //  cuda_slicer → TRTDetector → free slice → remap_offsets
    // ------------------------------------------------------------------
    const size_t total_tiles = tiling_cfg.tiles.size();
    const size_t tile_count = (opt.max_tiles > 0)
                                  ? std::min(static_cast<size_t>(opt.max_tiles), total_tiles)
                                  : total_tiles;

    std::cerr << "[Pipeline] Processing " << tile_count
              << " tiles (max_tiles=" << opt.max_tiles << ", grid="
              << total_tiles << ")...\n";

    const size_t tile_floats = static_cast<size_t>(3) * target_size * target_size;
    const size_t tile_bytes = tile_floats * sizeof(float);
    const int src_stride = opt.width * 3;

    std::vector<GlobalDetection> all_raw;
    std::vector<Rect> visited_tiles;
    visited_tiles.reserve(tile_count);

    const auto t_loop_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tile_count; ++i)
    {
        const Rect &tile = tiling_cfg.tiles[i];
        visited_tiles.push_back(tile);
        TileRect trect{tile.x, tile.y, tile.w, tile.h};

        // 4a. Allocate VRAM slice for this tile
        float *d_slice = nullptr;
        cudaError_t err = cudaMalloc(&d_slice, tile_bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "[Pipeline] WARN: cudaMalloc tile " << i << ": "
                      << cudaGetErrorString(err) << " — skipping\n";
            continue; // d_slice is null; nothing to free
        }

        // 4b. Zero-copy bilinear extraction: src VRAM → slice VRAM
        err = extract_tile_gpu(d_src_img, src_stride, opt.width, opt.height,
                               trect, target_size, d_slice);
        if (err != cudaSuccess)
        {
            std::cerr << "[Pipeline] WARN: extract_tile_gpu tile " << i << ": "
                      << cudaGetErrorString(err) << " — skipping\n";
            cudaFree(d_slice); // ← must free before continue
            continue;
        }
        cudaDeviceSynchronize();

        // 4c. TRTDetector forward pass (device tensor → host detections)
        std::vector<Detection> local_dets = detector->infer(d_slice);

        // Validation fallback: inject synthetic objects visible in this tile
        if (local_dets.empty() && !has_engine)
            local_dets = generate_synthetic_tile_detections(tile, target_size);

        // 4d. CRITICAL: immediately free tile slice from VRAM
        cudaFree(d_slice);
        d_slice = nullptr;

        // 4e. Remap tile-local coordinates → global 8K frame coordinates
        std::vector<GlobalDetection> remapped =
            remap_offsets(local_dets, tile, target_size, static_cast<int>(i));
        all_raw.insert(all_raw.end(), remapped.begin(), remapped.end());

        if (opt.verbose)
        {
            std::cerr << "  [Tile " << std::setw(3) << i << "] ("
                      << tile.x << ", " << tile.y << ", "
                      << tile.w << "x" << tile.h << ") → "
                      << local_dets.size() << " det(s)\n";
        }

        ++result.tiles_processed;
    }

    cudaDeviceSynchronize();
    const auto t_loop_end = std::chrono::high_resolution_clock::now();
    result.loop_ms = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
    result.raw_detections = all_raw.size();

    const double ms_per_tile = (result.tiles_processed > 0)
                                   ? result.loop_ms / static_cast<double>(result.tiles_processed)
                                   : 0.0;

    std::cerr << "[Pipeline] Tile loop : " << std::fixed << std::setprecision(2)
              << result.loop_ms << " ms total  ("
              << ms_per_tile << " ms/tile)\n"
              << "[Pipeline] Raw detections before NMS : " << result.raw_detections << "\n";

    // ------------------------------------------------------------------
    // Step 5: Cluster-DIoU-NMS across tile-overlap boundaries
    // ------------------------------------------------------------------
    const auto t_nms_start = std::chrono::high_resolution_clock::now();
    std::vector<GlobalDetection> final_dets =
        cluster_diou_nms(all_raw, visited_tiles, opt.diou_thresh, opt.conf_thresh);
    const auto t_nms_end = std::chrono::high_resolution_clock::now();

    result.nms_ms = std::chrono::duration<double, std::milli>(t_nms_end - t_nms_start).count();
    result.final_detections = final_dets.size();

    std::cerr << "[Pipeline] Cluster-DIoU-NMS : " << result.nms_ms << " ms  →  "
              << result.final_detections << " detection(s) after deduplication\n";

    // ------------------------------------------------------------------
    // Step 6: Serialize JSON → stdout
    // ------------------------------------------------------------------
    result.json_output = to_json_string(final_dets, opt.pretty_json);
    std::cout << result.json_output << std::endl;

    // ------------------------------------------------------------------
    // Step 7: Teardown — free owned VRAM, verify no leak
    // ------------------------------------------------------------------
    std::cerr << "[Pipeline] Releasing source frame from VRAM...\n";
    if (owns_frame && d_src_img)
    {
        cudaFree(d_src_img);
        d_src_img = nullptr;
    }
    cudaDeviceSynchronize();

    print_vram_status("Teardown");
    result.vram_after_teardown_bytes = query_vram().used_bytes;

    const auto t_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cerr << "[Pipeline] Total time : " << result.total_ms << " ms\n"
              << "=========================================================\n";

    return result;
}

// ============================================================================
// CLI Entry Point
// ============================================================================
int main(int argc, char *argv[])
{
    PipelineOptions opt = parse_args(argc, argv);
    PipelineResult res = run_pipeline(opt);
    return res.return_code;
}
