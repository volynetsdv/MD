/**
 * @file trt_detector.cpp
 * @brief Implementation of the TensorRT inference wrapper using the modern TensorRT I/O Tensor API.
 */

#include "trt_detector.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                      \
    do                                                                                        \
    {                                                                                         \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess)                                                             \
        {                                                                                     \
            std::cerr << "[libtiling_core][ERROR] CUDA error in " << __FILE__ << ":"          \
                      << __LINE__ << " (" << cudaGetErrorString(err__) << ")" << std::endl;   \
        }                                                                                     \
    } while (0)
#endif

namespace
{

    /**
     * @brief File-scoped TensorRT logger forwarding warnings and errors with unified prefix.
     */
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity <= Severity::kERROR)
            {
                std::cerr << "[libtiling_core][ERROR] TensorRT: " << msg << std::endl;
            }
            else if (severity == Severity::kWARNING)
            {
                std::cout << "[libtiling_core][WARN] TensorRT: " << msg << std::endl;
            }
        }
    } g_logger;

    /**
     * @brief RAII deleter for TensorRT interface objects.
     */
    struct TrtDeleter
    {
        template <typename T>
        void operator()(T *obj) const
        {
            if (obj)
            {
                delete obj;
            }
        }
    };

} // namespace

struct TRTDetector::Impl
{
    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context_;

    void *d_input_{nullptr};
    void *d_output_{nullptr};
    size_t input_size_{0};
    size_t output_size_{0};

    ~Impl()
    {
        if (d_input_)
        {
            CUDA_CHECK(cudaFree(d_input_));
        }
        if (d_output_)
        {
            CUDA_CHECK(cudaFree(d_output_));
        }
    }
};

TRTDetector::TRTDetector() : p_impl_(new Impl()) {}

TRTDetector::~TRTDetector()
{
    delete p_impl_;
}

bool TRTDetector::loadEngine(const std::string &engine_path)
{
    // Read serialized .engine file in binary mode.
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "[libtiling_core][ERROR] Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), static_cast<std::streamsize>(size));
    file.close();

    // Initialize TensorRT runtime.
    p_impl_->runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!p_impl_->runtime_)
    {
        std::cerr << "[libtiling_core][ERROR] Failed to create TensorRT Runtime" << std::endl;
        return false;
    }

    // Deserialize CUDA engine.
    p_impl_->engine_.reset(p_impl_->runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!p_impl_->engine_)
    {
        std::cerr << "[libtiling_core][ERROR] Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    // Create execution context.
    p_impl_->context_.reset(p_impl_->engine_->createExecutionContext());
    if (!p_impl_->context_)
    {
        std::cerr << "[libtiling_core][ERROR] Failed to create execution context" << std::endl;
        return false;
    }

    // Compute VRAM input and output buffer dimensions via the I/O Tensor API.
    const char *input_name = p_impl_->engine_->getIOTensorName(0);
    const char *output_name = p_impl_->engine_->getIOTensorName(1);

    auto in_dims = p_impl_->engine_->getTensorShape(input_name);
    auto out_dims = p_impl_->engine_->getTensorShape(output_name);

    size_t in_elements = 1;
    for (int i = 0; i < in_dims.nbDims; ++i)
    {
        in_elements *= static_cast<size_t>(in_dims.d[i]);
    }
    p_impl_->input_size_ = in_elements * sizeof(float);

    size_t out_elements = 1;
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        out_elements *= static_cast<size_t>(out_dims.d[i]);
    }
    p_impl_->output_size_ = out_elements * sizeof(float);

    // Allocate device memory for the output buffer.
    if (p_impl_->d_output_)
    {
        CUDA_CHECK(cudaFree(p_impl_->d_output_));
    }
    CUDA_CHECK(cudaMalloc(&p_impl_->d_output_, p_impl_->output_size_));

    return true;
}

std::vector<Detection> TRTDetector::infer(const float *d_input_tensor)
{
    std::vector<Detection> detections;
    if (!p_impl_->engine_ || !p_impl_->context_)
    {
        return detections;
    }

    const char *input_name = p_impl_->engine_->getIOTensorName(0);
    const char *output_name = p_impl_->engine_->getIOTensorName(1);

    // Bind device tensor addresses and enqueue inference on the default CUDA stream.
    p_impl_->context_->setTensorAddress(input_name, const_cast<float *>(d_input_tensor));
    p_impl_->context_->setTensorAddress(output_name, p_impl_->d_output_);
    p_impl_->context_->enqueueV3(0);

    return detections;
}

std::vector<std::vector<Detection>> TRTDetector::infer_batch(const float *d_batch_tensor, int batch_size, int tile_size)
{
    std::vector<std::vector<Detection>> results;
    if (batch_size <= 0)
    {
        return results;
    }

    results.reserve(static_cast<size_t>(batch_size));
    const size_t tile_elements = static_cast<size_t>(3) * tile_size * tile_size;

    for (int i = 0; i < batch_size; ++i)
    {
        const float *d_slice = d_batch_tensor ? (d_batch_tensor + i * tile_elements) : nullptr;
        results.push_back(infer(d_slice));
    }
    return results;
}

size_t TRTDetector::getDeviceMemoryUsage() const
{
    if (!p_impl_)
    {
        return 0;
    }
    return p_impl_->input_size_ + p_impl_->output_size_;
}
