// TensorRT YOLO detector – single fixed-resolution tile inference.

#include "trt_detector.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
namespace {
constexpr int kSupportedSizes[] = {320, 416, 512, 640};
constexpr int kNumSupported     = sizeof(kSupportedSizes) / sizeof(int);

/// Minimum confidence to keep a detection (pre-NMS threshold).
constexpr float kConfThreshold = 0.25f;
} // namespace

// ---------------------------------------------------------------------------
// Impl – holds all TensorRT / CUDA state
// ---------------------------------------------------------------------------
struct TRTDetector::Impl {
    nvinfer1::IRuntime*          runtime      = nullptr;
    nvinfer1::ICudaEngine*       engine       = nullptr;
    nvinfer1::IExecutionContext* context     = nullptr;

    cudaStream_t                stream       = 0;

    // Persistent device buffers (allocated once in loadEngine)
    float*                      d_input      = nullptr;
    void*                       d_output     = nullptr;
    size_t                      output_bytes = 0;
    void*                       workspace    = nullptr;
    size_t                      ws_size      = 0;

    // Geometry (filled by loadEngine)
    int                         input_w      = 0;
    int                         input_h      = 0;
    int                         num_classes  = 0;
    int                         max_det      = 0;

    size_t                      device_bytes = 0;   // tracked VRAM total

    ~Impl() {
        if (d_input)   cudaFree(d_input);
        if (d_output)  cudaFree(d_output);
        if (workspace) cudaFree(workspace);
        if (stream)    cudaStreamDestroy(stream);
        if (context)   context->destroy();
        if (engine)    engine->destroy();
        if (runtime)   runtime->destroy();
    }
};

// ---------------------------------------------------------------------------
// TRTDetector – public API
// ---------------------------------------------------------------------------
TRTDetector::TRTDetector() : pImpl_(new Impl()) {}
TRTDetector::~TRTDetector() { delete pImpl_; }

TRTDetector::TRTDetector(TRTDetector&& o) noexcept
    : pImpl_(std::exchange(o.pImpl_, nullptr)) {}

TRTDetector& TRTDetector::operator=(TRTDetector&& o) noexcept {
    if (this != &o) {
        delete pImpl_;
        pImpl_ = std::exchange(o.pImpl_, nullptr);
    }
    return *this;
}

// ---------------------------------------------------------------------------
bool TRTDetector::loadEngine(const std::string& engine_path) {
    // Release any previously loaded engine
    if (pImpl_->engine) {
        pImpl_->context->destroy();
        pImpl_->engine->destroy();
        pImpl_->runtime->destroy();
        cudaFree(pImpl_->d_input);
        cudaFree(pImpl_->d_output);
        cudaFree(pImpl_->workspace);
        cudaStreamDestroy(pImpl_->stream);
        pImpl_ = new Impl();   // fresh state
    }

    // Read engine file into host memory
    std::vector<char> blob;
    {
        FILE* f = std::fopen(engine_path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[TRTDetector] Cannot open '%s'\n",
                         engine_path.c_str());
            return false;
        }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        blob.resize(static_cast<size_t>(sz));
        std::fread(blob.data(), 1, blob.size(), f);
        std::fclose(f);
    }

    // Create runtime & deserialize engine
    pImpl_->runtime = nvinfer1::createInferRuntime(
        nvinfer1::getTRTLogger());
    if (!pImpl_->runtime) return false;

    pImpl_->engine = pImpl_->runtime->deserializeCudaEngine(
        blob.data(), blob.size());
    if (!pImpl_->engine) {
        std::fprintf(stderr, "[TRTDetector] deserializeCudaEngine failed\n");
        return false;
    }

    // Determine input / output geometry from the engine bindings
    const int nb = pImpl_->engine->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        auto dims  = pImpl_->engine->getBindingDimensions(i);
        bool is_in = pImpl_->engine->bindingIsInput(i);

        if (is_in) {
            // Expect NCHW: dims = [1, 3, H, W]
            pImpl_->input_h = dims.d[2];
            pImpl_->input_w = dims.d[3];
        } else {
            // Output shape varies by export; common layouts:
            //   [1, (4+num_classes), max_det]  or  [1, max_det*(4+C)]
            if (dims.nbDims == 3) {
                pImpl_->num_classes = dims.d[1] - 4;
                pImpl_->max_det     = static_cast<int>(dims.d[2]);
            } else if (dims.nbDims == 2) {
                const long flat = dims.d[1];
                pImpl_->num_classes = 80;  // COCO default
                pImpl_->max_det     = static_cast<int>(flat / (4 + pImpl_->num_classes));
            } else {
                std::fprintf(stderr,
                             "[TRTDetector] Unexpected output dims (%d)\n",
                             dims.nbDims);
                return false;
            }
        }
    }

    // Validate tile size is in the supported pool
    bool valid = false;
    for (int s : kSupportedSizes) {
        if (pImpl_->input_w == s && pImpl_->input_h == s) { valid = true; break; }
    }
    if (!valid) {
        std::fprintf(stderr,
                     "[TRTDetector] Input %dx%d not in supported pool\n",
                     pImpl_->input_w, pImpl_->input_h);
        return false;
    }

    // Allocate persistent device buffers
    const size_t input_bytes =
        static_cast<size_t>(3) * pImpl_->input_h * pImpl_->input_w * sizeof(float);
    if (cudaMalloc(&pImpl_->d_input, input_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[TRTDetector] cudaMalloc input failed\n");
        return false;
    }

    pImpl_->output_bytes =
        static_cast<size_t>(pImpl_->num_classes + 4) * pImpl_->max_det * sizeof(float);
    if (cudaMalloc(&pImpl_->d_output, pImpl_->output_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[TRTDetector] cudaMalloc output failed\n");
        return false;
    }

    // Workspace (required by TRT >= 8.5 for some engines)
    const size_t ws = static_cast<size_t>(pImpl_->engine->getDeviceMemorySize());
    if (ws > 0 && cudaMalloc(&pImpl_->workspace, ws) != cudaSuccess) {
        std::fprintf(stderr, "[TRTDetector] cudaMalloc workspace failed\n");
        return false;
    }
    pImpl_->ws_size = ws;

    // Execution context + stream
    pImpl_->context = pImpl_->engine->createExecutionContext();
    if (!pImpl_->context) {
        std::fprintf(stderr, "[TRTDetector] createExecutionContext failed\n");
        return false;
    }
    if (ws > 0) {
        pImpl_->context->setDeviceMemory(pImpl_->workspace);
    }

    if (cudaStreamCreate(&pImpl_->stream) != cudaSuccess) {
        std::fprintf(stderr, "[TRTDetector] cudaStreamCreate failed\n");
        return false;
    }

    // Track total VRAM held by this detector
    pImpl_->device_bytes = input_bytes + pImpl_->output_bytes + ws;

    return true;
}

// ---------------------------------------------------------------------------
std::vector<Detection> TRTDetector::infer(const float* d_input_tensor) {
    std::vector<Detection> results;
    if (!pImpl_->context || !d_input_tensor) return results;

    // ---- 1. Copy caller's tensor into our internal input buffer ----------
    const size_t in_bytes =
        static_cast<size_t>(3) * pImpl_->input_h * pImpl_->input_w * sizeof(float);
    cudaMemcpyAsync(pImpl_->d_input, d_input_tensor, in_bytes,
                    cudaMemcpyDeviceToDevice, pImpl_->stream);

    // ---- 2. Bind I/O and run enqueueV3 ----------------------------------
    const int nb = pImpl_->engine->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        if (pImpl_->engine->bindingIsInput(i)) {
            pImpl_->context->setBindingDimensions(
                pImpl_->engine->getBindingName(i),
                pImpl_->engine->getBindingDimensions(i));
            pImpl_->context->setBindingAddress(
                pImpl_->engine->getBindingName(i), pImpl_->d_input);
        } else {
            pImpl_->context->setBindingAddress(
                pImpl_->engine->getBindingName(i), pImpl_->d_output);
        }
    }

    if (!pImpl_->context->enqueueV3(pImpl_->stream)) {
        std::fprintf(stderr, "[TRTDetector] enqueueV3 failed\n");
        return results;
    }

    // ---- 3. Read output tensor to host for parsing ----------------------
    const int C = pImpl_->num_classes + 4;   // rows: xywh + class scores
    const int D = pImpl_->max_det;           // cols: anchors
    std::vector<float> h_out(static_cast<size_t>(C) * D);

    cudaMemcpyAsync(h_out.data(), pImpl_->d_output, pImpl_->output_bytes,
                    cudaMemcpyDeviceToHost, pImpl_->stream);
    cudaStreamSynchronize(pImpl_->stream);

    // ---- 4. Parse into Detection vectors (tile-local coords) ------------
    const float tile_w = static_cast<float>(pImpl_->input_w);
    const float tile_h = static_cast<float>(pImpl_->input_h);

    for (int d = 0; d < D; ++d) {
        // Find max class score
        int   best_cls = -1;
        float best_sc  = 0.0f;
        for (int c = 0; c < pImpl_->num_classes; ++c) {
            const float sc = h_out[static_cast<size_t>(4 + c) * D + d];
            if (sc > best_sc) { best_sc = sc; best_cls = c; }
        }
        if (best_sc < kConfThreshold || best_cls < 0) continue;

        // Centers -> top-left + w/h (output is normalized [0,1])
        const float cx = h_out[0 * D + d] * tile_w;
        const float cy = h_out[1 * D + d] * tile_h;
        const float bw = h_out[2 * D + d] * tile_w;
        const float bh = h_out[3 * D + d] * tile_h;

        Detection det{};
        det.x_local  = cx - bw * 0.5f;
        det.y_local  = cy - bh * 0.5f;
        det.w        = bw;
        det.h        = bh;
        det.conf     = best_sc;
        det.class_id = best_cls;

        // Clamp to tile boundaries
        det.x_local = std::max(0.0f, std::min(det.x_local, tile_w - 1.0f));
        det.y_local = std::max(0.0f, std::min(det.y_local, tile_h - 1.0f));

        results.push_back(det);
    }

    // ---- 5. Temporary host buffer (h_out) freed on scope exit -----------
    // Persistent VRAM buffers (d_input, d_output, workspace) remain owned by Impl.
    return results;
}

// ---------------------------------------------------------------------------
size_t TRTDetector::getDeviceMemoryUsage() const {
    return pImpl_ ? pImpl_->device_bytes : 0;
}
