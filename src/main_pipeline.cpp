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

namespace {

// ============================================================================
// VRAM & Diagnostic Telemetry Helpers
// ============================================================================
struct VramInfo {
    size_t free_bytes{0};
    size_t total_bytes{0};
    size_t used_bytes{0};

    double free_mb() const { return static_cast<double>(free_bytes) / (1024.0 * 1024.0); }
    double total_mb() const { return static_cast<double>(total_bytes) / (1024.0 * 1024.0); }
    double used_mb() const { return static_cast<double>(used_bytes) / (1024.0 * 1024.0); }
};

VramInfo query_vram() {
    VramInfo info;
    cudaError_t err = cudaMemGetInfo(&info.free_bytes, &info.total_bytes);
    if (err == cudaSuccess) {
        info.used_bytes = info.total_bytes - info.free_bytes;
    }
    return info;
}

void print_vram_status(const std::string& stage) {
    VramInfo info = query_vram();
    std::cerr << "[VRAM] " << stage << ": Used = "
              << std::fixed << std::setprecision(2) << info.used_mb() << " MB ("
              << info.free_mb() << " MB free / " << info.total_mb() << " MB total)\n";
}

// ============================================================================
// Synthetic 8K Image Generator
// ============================================================================
// Fills an 8K BGR image with simulated terrain gradients and synthetic targets
// across the frame without any disk I/O.
void generate_synthetic_8k(uint8_t* h_img, int width, int height) {
    std::cerr << "[Pipeline] Generating synthetic 8K frame (" << width << "x" << height << ")...\n";
    #pragma omp parallel for if(height > 1000)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t* px = h_img + (static_cast<size_t>(y) * width + x) * 3;
            px[0] = static_cast<uint8_t>((x * 255) / std::max(width - 1, 1));   // Blue channel
            px[1] = static_cast<uint8_t>((y * 255) / std::max(height - 1, 1));  // Green channel
            px[2] = static_cast<uint8_t>(128 + ((x ^ y) & 63));                 // Red texture
        }
    }
}

// ============================================================================
// Model Pool Manager
// ============================================================================
// Manages TRTDetector instances for supported resolutions: {320, 416, 512, 640}.
class TRTDetectorPool {
public:
    TRTDetectorPool() {
        detectors_[320] = std::make_unique<TRTDetector>();
        detectors_[416] = std::make_unique<TRTDetector>();
        detectors_[512] = std::make_unique<TRTDetector>();
        detectors_[640] = std::make_unique<TRTDetector>();
    }

    bool loadModel(int size, const std::string& path) {
        auto it = detectors_.find(size);
        if (it != detectors_.end()) {
            engine_paths_[size] = path;
            bool ok = it->second->loadEngine(path);
            if (ok) {
                std::cerr << "[DetectorPool] Successfully loaded engine for " << size
                          << "x" << size << " from " << path << "\n";
            } else {
                std::cerr << "[DetectorPool] Failed to load engine for " << size
                          << "x" << size << " from " << path << "\n";
            }
            return ok;
        }
        return false;
    }

    void scanDirectory(const std::string& dir) {
        const std::vector<int> sizes = {320, 416, 512, 640};
        for (int s : sizes) {
            std::vector<std::string> candidates = {
                dir + "/model_" + std::to_string(s) + ".engine",
                dir + "/yolo_" + std::to_string(s) + ".engine",
                dir + "/" + std::to_string(s) + ".engine"
            };
            for (const auto& path : candidates) {
                std::ifstream f(path, std::ios::binary);
                if (f.good()) {
                    f.close();
                    loadModel(s, path);
                    break;
                }
            }
        }
    }

    TRTDetector* getDetector(int target_size) {
        auto it = detectors_.find(target_size);
        if (it != detectors_.end()) {
            return it->second.get();
        }
        return nullptr;
    }

    bool hasLoadedEngine(int target_size) const {
        auto it = detectors_.find(target_size);
        if (it != detectors_.end()) {
            return it->second->getDeviceMemoryUsage() > 0;
        }
        return false;
    }

private:
    std::unordered_map<int, std::unique_ptr<TRTDetector>> detectors_;
    std::unordered_map<int, std::string> engine_paths_;
};

// ============================================================================
// Synthetic Test Objects (Ground Truth for Validation)
// ============================================================================
// Ground truth objects in global 8K frame to simulate detection when no TRT
// weights are supplied, testing the end-to-end geometry and DIoU clustering.
struct SyntheticTarget {
    float gx, gy, gw, gh;
    int class_id;
    float base_conf;
};

const std::vector<SyntheticTarget> kSyntheticTargets = {
    // Vehicles in first quadrant
    {250.0f, 180.0f, 54.0f, 32.0f, 2, 0.91f},
    {340.0f, 290.0f, 62.0f, 38.0f, 2, 0.88f},
    // Target placed in tile overlap region (~x=380..450, y=380..450)
    {390.0f, 370.0f, 48.0f, 28.0f, 2, 0.94f},
    // Person on border
    {720.0f, 410.0f, 22.0f, 44.0f, 0, 0.86f},
    // UAV / Aircraft
    {1250.0f, 800.0f, 110.0f, 75.0f, 4, 0.95f}
};

std::vector<Detection> generate_synthetic_tile_detections(
    const Rect& tile, int model_target_size)
{
    std::vector<Detection> dets;
    for (const auto& target : kSyntheticTargets) {
        // Check if target intersects tile
        float tx1 = target.gx;
        float ty1 = target.gy;
        float tx2 = target.gx + target.gw;
        float ty2 = target.gy + target.gh;

        float rx1 = static_cast<float>(tile.x);
        float ry1 = static_cast<float>(tile.y);
        float rx2 = static_cast<float>(tile.x + tile.w);
        float ry2 = static_cast<float>(tile.y + tile.h);

        if (tx2 > rx1 && tx1 < rx2 && ty2 > ry1 && ty1 < ry2) {
            // Target is visible in this tile -> compute local coordinates
            float scale_x = static_cast<float>(model_target_size) / static_cast<float>(tile.w);
            float scale_y = static_cast<float>(model_target_size) / static_cast<float>(tile.h);

            Detection d;
            d.x_local = (target.gx - rx1) * scale_x;
            d.y_local = (target.gy - ry1) * scale_y;
            d.w = target.gw * scale_x;
            d.h = target.gh * scale_y;
            d.conf = target.base_conf;
            d.class_id = target.class_id;

            dets.push_back(d);
        }
    }
    return dets;
}

// ============================================================================
// CLI Configuration
// ============================================================================
struct PipelineOptions {
    int width{7680};
    int height{4320};
    float altitude{100.0f};
    size_t vram_mb_override{0};
    int max_tiles{12};               // Process first 10-12 tiles by default, 0 for all
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

PipelineOptions parse_args(int argc, char* argv[]) {
    PipelineOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) {
            opt.width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            opt.height = std::atoi(argv[++i]);
        } else if (arg == "--altitude" && i + 1 < argc) {
            opt.altitude = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--vram-mb" && i + 1 < argc) {
            opt.vram_mb_override = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (arg == "--max-tiles" && i + 1 < argc) {
            opt.max_tiles = std::atoi(argv[++i]);
        } else if (arg == "--models-dir" && i + 1 < argc) {
            opt.models_dir = argv[++i];
        } else if (arg == "--engine-320" && i + 1 < argc) {
            opt.engine_320 = argv[++i];
        } else if (arg == "--engine-416" && i + 1 < argc) {
            opt.engine_416 = argv[++i];
        } else if (arg == "--engine-512" && i + 1 < argc) {
            opt.engine_512 = argv[++i];
        } else if (arg == "--engine-640" && i + 1 < argc) {
            opt.engine_640 = argv[++i];
        } else if (arg == "--diou-thresh" && i + 1 < argc) {
            opt.diou_thresh = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--conf-thresh" && i + 1 < argc) {
            opt.conf_thresh = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--pretty") {
            opt.pretty_json = true;
        } else if (arg == "--verbose") {
            opt.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: main_pipeline [options]\n"
                      << "  --width <int>         Image width (default: 7680)\n"
                      << "  --height <int>        Image height (default: 4320)\n"
                      << "  --altitude <float>    Flight altitude in meters (default: 100.0)\n"
                      << "  --vram-mb <int>       Override available VRAM in MB\n"
                      << "  --max-tiles <int>     Number of tiles to process (default: 12, 0=all)\n"
                      << "  --models-dir <dir>    Directory to search for engine files\n"
                      << "  --engine-320 <path>   Explicit path to 320x320 engine\n"
                      << "  --engine-416 <path>   Explicit path to 416x416 engine\n"
                      << "  --engine-512 <path>   Explicit path to 512x512 engine\n"
                      << "  --engine-640 <path>   Explicit path to 640x640 engine\n"
                      << "  --diou-thresh <float> DIoU NMS threshold (default: 0.5)\n"
                      << "  --conf-thresh <float> Confidence threshold (default: 0.25)\n"
                      << "  --pretty              Pretty-print JSON output\n"
                      << "  --verbose             Verbose logging\n";
            std::exit(0);
        }
    }
    return opt;
}

} // namespace

// ============================================================================
// Main Pipeline Entry Point
// ============================================================================
int main(int argc, char* argv[]) {
    const auto t_pipeline_start = std::chrono::high_resolution_clock::now();
    PipelineOptions opt = parse_args(argc, argv);

    std::cerr << "=========================================================\n";
    std::cerr << "  Inference Engine Orchestrator (Autonomous 8K Pipeline) \n";
    std::cerr << "=========================================================\n";

    // ------------------------------------------------------------------------
    // Step 0: Initial VRAM Baseline
    // ------------------------------------------------------------------------
    print_vram_status("Baseline (Program Start)");

    size_t vram_available_mb = opt.vram_mb_override;
    if (vram_available_mb == 0) {
        VramInfo vram = query_vram();
        vram_available_mb = vram.free_bytes / (1024 * 1024);
    }
    std::cerr << "[Pipeline] Using VRAM available: " << vram_available_mb << " MB\n";
    std::cerr << "[Pipeline] Telemetry altitude:  " << opt.altitude << " m\n";

    // ------------------------------------------------------------------------
    // Step 1: Ingest 8K Frame into VRAM (Zero-Copy GPU Buffer)
    // ------------------------------------------------------------------------
    const size_t img_bytes = static_cast<size_t>(opt.width) * opt.height * 3;
    std::cerr << "[Pipeline] Allocating 8K source buffer ("
              << (img_bytes / (1024 * 1024)) << " MB) in VRAM...\n";

    uint8_t* d_src_img = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_src_img, img_bytes);
    if (cuda_err != cudaSuccess) {
        std::cerr << "[Pipeline] FATAL: cudaMalloc for 8K source failed: "
                  << cudaGetErrorString(cuda_err) << "\n";
        return 1;
    }

    // Allocate host buffer to generate test image and copy to device
    std::vector<uint8_t> h_img(img_bytes);
    generate_synthetic_8k(h_img.data(), opt.width, opt.height);

    cuda_err = cudaMemcpy(d_src_img, h_img.data(), img_bytes, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        std::cerr << "[Pipeline] FATAL: cudaMemcpy HostToDevice failed: "
                  << cudaGetErrorString(cuda_err) << "\n";
        cudaFree(d_src_img);
        return 1;
    }
    // Immediately release host memory: all subsequent work is strictly GPU-resident
    h_img.clear();
    h_img.shrink_to_fit();

    print_vram_status("After 8K Ingestion in VRAM");

    // ------------------------------------------------------------------------
    // Step 2: Dynamic Tiling Calculation via libtiling_core
    // ------------------------------------------------------------------------
    TilingConfig tiling_cfg = calculate_tiling_params(
        opt.width, opt.height, opt.altitude, vram_available_mb);

    std::cerr << "[Pipeline] TilingConfig generated:\n"
              << "  - Selected Model Resolution : " << tiling_cfg.tile_size << "x" << tiling_cfg.tile_size << "\n"
              << "  - Dynamic Overlap (O_lap)   : " << tiling_cfg.overlap << "\n"
              << "  - Grid Dimensions           : " << tiling_cfg.grid_cols << " cols x "
              << tiling_cfg.grid_rows << " rows\n"
              << "  - Total Grid Tiles          : " << tiling_cfg.tiles.size() << "\n";

    // ------------------------------------------------------------------------
    // Step 3: Initialize & Select TRTDetector from Pool
    // ------------------------------------------------------------------------
    TRTDetectorPool model_pool;
    if (!opt.models_dir.empty()) {
        model_pool.scanDirectory(opt.models_dir);
    }
    if (!opt.engine_320.empty()) model_pool.loadModel(320, opt.engine_320);
    if (!opt.engine_416.empty()) model_pool.loadModel(416, opt.engine_416);
    if (!opt.engine_512.empty()) model_pool.loadModel(512, opt.engine_512);
    if (!opt.engine_640.empty()) model_pool.loadModel(640, opt.engine_640);

    const int target_size = tiling_cfg.tile_size;
    TRTDetector* detector = model_pool.getDetector(target_size);
    if (!detector) {
        std::cerr << "[Pipeline] FATAL: Unsupported tile size " << target_size << "\n";
        cudaFree(d_src_img);
        return 1;
    }

    const bool has_engine = model_pool.hasLoadedEngine(target_size);
    if (!has_engine) {
        std::cerr << "[Pipeline] Notice: Running in validation mode (simulated detections & zero-copy GPU slicing).\n";
    }

    // ------------------------------------------------------------------------
    // Step 4: Per-Tile Processing Loop
    // cuda_slicer -> TRTDetector -> immediate VRAM free -> remap_offsets
    // ------------------------------------------------------------------------
    const size_t tile_count = (opt.max_tiles > 0)
        ? std::min(static_cast<size_t>(opt.max_tiles), tiling_cfg.tiles.size())
        : tiling_cfg.tiles.size();

    std::cerr << "[Pipeline] Executing tile inference loop for " << tile_count << " tiles...\n";

    const size_t tile_floats = static_cast<size_t>(3) * target_size * target_size;
    const size_t tile_bytes = tile_floats * sizeof(float);
    const int src_stride = opt.width * 3;

    std::vector<GlobalDetection> all_raw_detections;
    std::vector<Rect> processed_tiles;
    processed_tiles.reserve(tile_count);

    size_t total_local_detections = 0;
    const auto t_loop_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tile_count; ++i) {
        const Rect& tile = tiling_cfg.tiles[i];
        processed_tiles.push_back(tile);
        TileRect trect{tile.x, tile.y, tile.w, tile.h};

        // 4a. cuda_slicer: Allocate sub-tensor slice in VRAM
        float* d_slice = nullptr;
        cuda_err = cudaMalloc(&d_slice, tile_bytes);
        if (cuda_err != cudaSuccess) {
            std::cerr << "[Pipeline] Error allocating VRAM for tile " << i << ": "
                      << cudaGetErrorString(cuda_err) << "\n";
            break;
        }

        // 4b. Zero-copy bilinear extraction into float32 NCHW tensor [1, 3, target, target]
        cuda_err = extract_tile_gpu(
            d_src_img, src_stride, opt.width, opt.height,
            trect, target_size, d_slice);
        if (cuda_err != cudaSuccess) {
            std::cerr << "[Pipeline] Error in extract_tile_gpu for tile " << i << ": "
                      << cudaGetErrorString(cuda_err) << "\n";
            cudaFree(d_slice);
            continue;
        }
        cudaDeviceSynchronize();

        // 4c. TRTDetector: Run inference pass on device tensor
        std::vector<Detection> local_dets = detector->infer(d_slice);

        // If no engine file loaded or output empty in validation run, supply simulated ground truth
        if (local_dets.empty() && !has_engine) {
            local_dets = generate_synthetic_tile_detections(tile, target_size);
        }

        total_local_detections += local_dets.size();

        // 4d. CRITICAL: Immediately delete slice from VRAM after forward pass
        cudaFree(d_slice);
        d_slice = nullptr;

        // 4e. Remap offsets: Tile-Local -> Global 8K Frame coordinates
        std::vector<GlobalDetection> remapped = remap_offsets(
            local_dets, tile, target_size, static_cast<int>(i));

        all_raw_detections.insert(
            all_raw_detections.end(), remapped.begin(), remapped.end());

        if (opt.verbose) {
            std::cerr << "  [Tile " << std::setw(2) << i << "] ("
                      << tile.x << ", " << tile.y << ", " << tile.w << ", " << tile.h
                      << ") -> " << local_dets.size() << " local dets\n";
        }
    }
    cudaDeviceSynchronize();

    const auto t_loop_end = std::chrono::high_resolution_clock::now();
    double loop_ms = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();

    std::cerr << "[Pipeline] Completed tile loop in " << std::fixed << std::setprecision(2)
              << loop_ms << " ms (" << (loop_ms / tile_count) << " ms/tile)\n";
    std::cerr << "[Pipeline] Total raw detections before NMS: " << all_raw_detections.size() << "\n";

    // ------------------------------------------------------------------------
    // Step 5: Postprocessing - Cluster-DIoU-NMS
    // ------------------------------------------------------------------------
    const auto t_nms_start = std::chrono::high_resolution_clock::now();
    std::vector<GlobalDetection> final_detections = cluster_diou_nms(
        all_raw_detections, processed_tiles, opt.diou_thresh, opt.conf_thresh);
    const auto t_nms_end = std::chrono::high_resolution_clock::now();

    double nms_ms = std::chrono::duration<double, std::milli>(t_nms_end - t_nms_start).count();
    std::cerr << "[Pipeline] Cluster-DIoU-NMS completed in " << nms_ms << " ms.\n";
    std::cerr << "[Pipeline] Detections after DIoU deduplication: " << final_detections.size() << "\n";

    // ------------------------------------------------------------------------
    // Step 6: Output Lightweight JSON Vector to stdout
    // ------------------------------------------------------------------------
    std::string json_result = to_json_string(final_detections, opt.pretty_json);
    std::cout << json_result << std::endl;

    // ------------------------------------------------------------------------
    // Step 7: Teardown & VRAM Leak Verification
    // ------------------------------------------------------------------------
    std::cerr << "[Pipeline] Freeing 8K source buffer from VRAM...\n";
    if (d_src_img) {
        cudaFree(d_src_img);
        d_src_img = nullptr;
    }
    cudaDeviceSynchronize();

    print_vram_status("Teardown (Post-Execution)");

    const auto t_pipeline_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_pipeline_end - t_pipeline_start).count();
    std::cerr << "[Pipeline] Total pipeline execution time: " << total_ms << " ms.\n";
    std::cerr << "=========================================================\n";

    return 0;
}

