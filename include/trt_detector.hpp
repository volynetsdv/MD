#pragma once

#include <cstddef>
#include <string>
#include <vector>

/// A single detection in tile-local coordinates (pixels).
struct Detection {
    float x_local;   ///< Top-left X within the tile
    float y_local;   ///< Top-left Y within the tile
    float w;         ///< Bounding-box width  (px)
    float h;         ///< Bounding-box height (px)
    float conf;      ///< Confidence score [0, 1]
    int   class_id;  ///< Class index
};

/// TensorRT-based YOLO detector for a single fixed-resolution tile.
///
/// Thread-safety: NOT thread-safe. One instance per tile-size / engine file.
class TRTDetector {
public:
    TRTDetector();
    ~TRTDetector();

    // Non-copyable, movable
    TRTDetector(const TRTDetector&)            = delete;
    TRTDetector& operator=(const TRTDetector&) = delete;
    TRTDetector(TRTDetector&&) noexcept;
    TRTDetector& operator=(TRTDetector&&) noexcept;

    /// Load a serialized TensorRT engine (.engine) for a fixed input size.
    /// Supported resolutions: 320, 416, 512, 640 (square).
    /// @param engine_path  Filesystem path to the .engine file.
    /// @return true on success; false if the file is missing or incompatible.
    bool loadEngine(const std::string& engine_path);

    /// Run one inference pass on a device-resident NCHW float32 tensor.
    ///
    /// The caller must guarantee that `d_input_tensor` points to valid VRAM
    /// of size  input_h * input_w * 3 floats (normalized [0,1]).
    ///
    /// All temporary VRAM allocations made during this call are freed
    /// before the function returns.
    /// @param d_input_tensor  GPU pointer to NCHW float32 data.
    /// @return Detections in tile-local pixel coordinates.
    std::vector<Detection> infer(const float* d_input_tensor);

    /// Total bytes currently held by this detector on the device.
    size_t getDeviceMemoryUsage() const;

private:
    struct Impl;
    Impl* pImpl_ = nullptr;
};
