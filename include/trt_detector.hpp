/**
 * @file trt_detector.hpp
 * @brief TensorRT inference wrapper for fixed-resolution YOLO object detection tiles.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * @brief A single detection in tile-local coordinates (pixels).
 */
struct Detection
{
    float x_local{0.0f}; ///< Top-left X within the tile (px).
    float y_local{0.0f}; ///< Top-left Y within the tile (px).
    float w{0.0f};       ///< Bounding-box width (px).
    float h{0.0f};       ///< Bounding-box height (px).
    float conf{0.0f};    ///< Confidence score [0, 1].
    int class_id{0};     ///< Class index.
};

/**
 * @brief TensorRT-based YOLO detector for a single fixed-resolution tile.
 *
 * Thread-safety: NOT thread-safe. One instance per tile-size / engine file.
 */
class TRTDetector
{
public:
    TRTDetector();
    ~TRTDetector();

    // Non-copyable, movable.
    TRTDetector(const TRTDetector &) = delete;
    TRTDetector &operator=(const TRTDetector &) = delete;
    TRTDetector(TRTDetector &&) noexcept;
    TRTDetector &operator=(TRTDetector &&) noexcept;

    /**
     * @brief Load a serialized TensorRT engine (.engine) for a fixed input size.
     *
     * Supported resolutions: 320, 416, 512, 640 (square).
     *
     * @param engine_path  Filesystem path to the .engine file.
     * @return bool        True on success; false if the file is missing or incompatible.
     */
    bool loadEngine(const std::string &engine_path);

    /**
     * @brief Run one inference pass on a device-resident NCHW float32 tensor.
     *
     * The caller must guarantee that `d_input_tensor` points to valid VRAM
     * of size input_h * input_w * 3 floats (normalized [0, 1]).
     *
     * @param d_input_tensor          GPU pointer to NCHW float32 data.
     * @return std::vector<Detection> Detections in tile-local pixel coordinates.
     */
    std::vector<Detection> infer(const float *d_input_tensor);

    /**
     * @brief Run inference pass on a batch of B device-resident NCHW float32 tiles.
     *
     * @param d_batch_tensor                       GPU pointer to batched NCHW float32 tensor.
     * @param batch_size                           Number of tiles in the batch.
     * @param tile_size                            Spatial resolution of each tile (default 640).
     * @return std::vector<std::vector<Detection>> Detections per tile in the batch.
     */
    std::vector<std::vector<Detection>> infer_batch(const float *d_batch_tensor, int batch_size, int tile_size = 640);

    /**
     * @brief Query total bytes currently held by this detector on the device.
     *
     * @return size_t Device memory allocated in bytes.
     */
    size_t getDeviceMemoryUsage() const;

private:
    struct Impl;
    Impl *p_impl_{nullptr};
};
