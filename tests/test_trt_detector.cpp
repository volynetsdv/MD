// Standalone test: load a .engine, run one infer pass, verify VRAM cleanup.
// Usage: ./test_trt_detector <path/to/model.engine>
// If no path is given (or file missing), the test SKIPs with exit code 0.

#include "trt_detector.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
    int g_failures = 0;

#define CHECK(cond, msg)                             \
    do                                               \
    {                                                \
        if (!(cond))                                 \
        {                                            \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            ++g_failures;                            \
        }                                            \
    } while (0)
} // namespace

int main(int argc, char *argv[])
{
    // ------------------------------------------------------------------
    // Graceful skip if no engine provided or file missing.
    // ------------------------------------------------------------------
    if (argc < 2)
    {
        std::fprintf(stderr, "SKIP: no .engine path provided.\n");
        return 0;
    }
    const char *engine_path = argv[1];

    FILE *f = std::fopen(engine_path, "rb");
    if (!f)
    {
        std::fprintf(stderr, "SKIP: '%s' not found.\n", engine_path);
        return 0;
    }
    std::fclose(f);

    // ------------------------------------------------------------------
    // 1. loadEngine
    // ------------------------------------------------------------------
    TRTDetector det;
    CHECK(det.loadEngine(engine_path), "loadEngine should succeed");

    const size_t mem_after_load = det.getDeviceMemoryUsage();
    CHECK(mem_after_load > 0, "device memory > 0 after load");

    // ------------------------------------------------------------------
    // 2. Allocate a dummy NCHW input on the device (all zeros)
    // ------------------------------------------------------------------
    const int W = 640, H = 640, C = 3;
    const size_t in_bytes = static_cast<size_t>(C) * H * W * sizeof(float);

    float *d_input = nullptr;
    CHECK(cudaMalloc(&d_input, in_bytes) == cudaSuccess, "cudaMalloc input");
    CHECK(cudaMemset(d_input, 0, in_bytes) == cudaSuccess, "cudaMemset input");

    // ------------------------------------------------------------------
    // 3. Run infer – should return (possibly empty) vector without crash
    // ------------------------------------------------------------------
    const size_t mem_before_infer = det.getDeviceMemoryUsage();

    std::vector<Detection> dets = det.infer(d_input);

    const size_t mem_after_infer = det.getDeviceMemoryUsage();

    CHECK(mem_after_infer == mem_before_infer,
          "VRAM usage unchanged after infer (temp buffers freed)");

    std::fprintf(stderr, "INFO: %zu detections returned\n", dets.size());

    // ------------------------------------------------------------------
    // 4. Cleanup
    // ------------------------------------------------------------------
    cudaFree(d_input);

    if (g_failures == 0)
    {
        std::fprintf(stderr, "PASS: all checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: %d check(s) failed.\n", g_failures);
    return 1;
}
