#include "trt_detector.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

// 1. Кастомний логер TensorRT (замкнений у цьому файлі)
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// RAII Deleter для об'єктів TensorRT замість застарілого .destroy()
struct TRTDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            delete obj; // Новий стандарт TensorRT API
        }
    }
};

struct TRTDetector::Impl
{
    std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context;

    void *d_input{nullptr};
    void *d_output{nullptr};
    size_t input_size{0};
    size_t output_size{0};

    ~Impl()
    {
        if (d_input)
            cudaFree(d_input);
        if (d_output)
            cudaFree(d_output);
    }
};

TRTDetector::TRTDetector() : pImpl_(new Impl()) {}
TRTDetector::~TRTDetector()
{
    delete pImpl_;
}

bool TRTDetector::loadEngine(const std::string &engine_path)
{
    // 1. Зчитування файлу .engine у бінарному режимі
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 2. Ініціалізація Runtime
    pImpl_->runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!pImpl_->runtime)
    {
        std::cerr << "Failed to create TensorRT Runtime" << std::endl;
        return false;
    }

    // 3. Десеріалізація CUDA Engine
    pImpl_->engine.reset(pImpl_->runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!pImpl_->engine)
    {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    // 4. Створення контексту виконання (Execution Context)
    pImpl_->context.reset(pImpl_->engine->createExecutionContext());
    if (!pImpl_->context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // 5. Автоматичне розрахування розмірів входів/виходів VRAM (через I/O Tensor API)
    const char *input_name = pImpl_->engine->getIOTensorName(0);
    const char *output_name = pImpl_->engine->getIOTensorName(1);

    auto in_dims = pImpl_->engine->getTensorShape(input_name);
    auto out_dims = pImpl_->engine->getTensorShape(output_name);

    // Розрахунок розміру тензора у байтах: batch * C * H * W * sizeof(float)
    size_t in_elements = 1;
    for (int i = 0; i < in_dims.nbDims; ++i)
        in_elements *= in_dims.d[i];
    pImpl_->input_size = in_elements * sizeof(float);

    size_t out_elements = 1;
    for (int i = 0; i < out_dims.nbDims; ++i)
        out_elements *= out_dims.d[i];
    pImpl_->output_size = out_elements * sizeof(float);

    // 6. Виділення VRAM під вихідний буфер
    if (pImpl_->d_output)
        cudaFree(pImpl_->d_output);
    cudaMalloc(&pImpl_->d_output, pImpl_->output_size);

    return true;
}

std::vector<Detection> TRTDetector::infer(const float *d_input_tensor)
{
    std::vector<Detection> detections;
    if (!pImpl_->engine || !pImpl_->context)
        return detections;

    // Оновлений TensorRT I/O Tensor API (Замість застарілих Bindings)
    const char *input_name = pImpl_->engine->getIOTensorName(0);
    const char *output_name = pImpl_->engine->getIOTensorName(1);

    // Встановлюємо адреси тензорів у VRAM
    pImpl_->context->setTensorAddress(input_name, const_cast<float *>(d_input_tensor));
    pImpl_->context->setTensorAddress(output_name, pImpl_->d_output);

    // Запуск асинхронного або синхронного інференсу
    pImpl_->context->enqueueV3(0);

    return detections;
}

// Повертає обсяг пам'яті у байтах, виділений під вхідний та вихідний буфери TensorRT
size_t TRTDetector::getDeviceMemoryUsage() const
{
    if (!pImpl_)
        return 0;
    return pImpl_->input_size + pImpl_->output_size;
}
