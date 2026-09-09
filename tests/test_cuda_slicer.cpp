#include "cuda_slicer.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Helper: fill a GPU BGR image with a deterministic diagonal gradient so we
// can verify that the extracted tile contains plausible values in [0, 1].
// ---------------------------------------------------------------------------
static void fill_gradient_host(uint8_t* h_img, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t* px = h_img + (static_cast<size_t>(y) * w + x) * 3;
            px[0] = static_cast<uint8_t>((x * 255) / std::max(w - 1, 1));   // B
            px[1] = static_cast<uint8_t>((y * 255) / std::max(h - 1, 1));   // G
            px[2] = static_cast<uint8_t>(128);                              // R constant
        }
    }
}

int main() {
    const int src_w = 4000;
    const int src_h = 3000;
    const size_t img_bytes = static_cast<size_t>(src_w) * src_h * 3;

    // --- Allocate source image in VRAM and upload a gradient pattern ------
    uint8_t* d_src = nullptr;
    cudaError_t err = cudaMalloc(&d_src, img_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc src (%zu bytes): %s\n",
                     img_bytes, cudaGetErrorString(err));
        return 1;
    }

    uint8_t* h_img = static_cast<uint8_t*>(std::malloc(img_bytes));
    fill_gradient_host(h_img, src_w, src_h);
    err = cudaMemcpy(d_src, h_img, img_bytes, cudaMemcpyHostToDevice);
    std::free(h_img);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H2D: %s\n", cudaGetErrorString(err));
        cudaFree(d_src);
        return 1;
    }

    // --- Allocate output tensor [1, 3, 512, 512] float32 in VRAM --------
    const int target = 512;
    const size_t dst_floats = static_cast<size_t>(3) * target * target;
    float* d_dst = nullptr;
    err = cudaMalloc(&d_dst, dst_floats * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc dst (%zu floats): %s\n",
                     dst_floats, cudaGetErrorString(err));
        cudaFree(d_src);
        return 1;
    }

    // --- Extract tile (100, 100, 512, 512) → 512×512 --------------------
    TileRect rect{100, 100, 512, 512};
    err = extract_tile_gpu(d_src, src_w * 3, src_w, src_h, rect, target, d_dst);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: extract_tile_gpu: %s\n", cudaGetErrorString(err));
        cudaFree(d_src);
        cudaFree(d_dst);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
        cudaFree(d_src);
        cudaFree(d_dst);
        return 1;
    }

    // --- Validate output tensor on host (single D2H for verification) ----
    float* h_dst = static_cast<float*>(std::malloc(dst_floats * sizeof(float)));
    err = cudaMemcpy(h_dst, d_dst, dst_floats * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D2H: %s\n", cudaGetErrorString(err));
        std::free(h_dst);
        cudaFree(d_src);
        cudaFree(d_dst);
        return 1;
    }

    // Check that every value is in [0.0, 1.0] and not all-zero.
    bool valid = true;
    float sum = 0.0f;
    for (size_t i = 0; i < dst_floats; ++i) {
        if (!(h_dst[i] >= 0.0f && h_dst[i] <= 1.0f)) {
            std::fprintf(stderr, "FAIL: value out of range at index %zu: %f\n", i, h_dst[i]);
            valid = false;
            break;
        }
        sum += h_dst[i];
    }

    if (valid && sum == 0.0f) {
        std::fprintf(stderr, "FAIL: tensor is all zeros — extraction likely broken\n");
        valid = false;
    }

    // Spot-check: pixel at tile-local (0,0) should correspond to source (100,100).
    // B channel ≈ 100*255/3999 ≈ 6.4 → ~0.025; G ≈ 100*255/2999 ≈ 8.5 → ~0.033; R = 128/255 ≈ 0.502
    float b_val = h_dst[0];                          // channel 0, (0,0)
    float g_val = h_dst[target * target];            // channel 1, (0,0)
    float r_val = h_dst[2 * target * target];        // channel 2, (0,0)

    if (std::fabs(b_val - 6.4f / 255.0f) > 0.05f ||
        std::fabs(g_val - 8.5f / 255.0f) > 0.05f ||
        std::fabs(r_val - 128.0f / 255.0f) > 0.05f) {
        std::fprintf(stderr, "FAIL: spot-check mismatch B=%.4f G=%.4f R=%.4f\n", b_val, g_val, r_val);
        valid = false;
    }

    // --- Cleanup ----------------------------------------------------------
    std::free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    if (valid) {
        std::printf("PASS: extract_tile_gpu — tensor [1,3,%d,%d] valid in VRAM\n", target, target);
        return 0;
    }
    return 1;
}
