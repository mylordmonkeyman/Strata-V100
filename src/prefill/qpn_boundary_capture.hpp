#pragma once
// Default-off, private diagnostic snapshots at the narrowed layer 2/3 boundary.
// No device kernel, routing, slot, or arithmetic changes.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace strata::prefill {
class QpnBoundaryCapture {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
    std::FILE* file_ = nullptr;
    bool enabled_ = false, good_ = true;
#endif
public:
    explicit QpnBoundaryCapture(bool single_gpu) {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
        const char* path = std::getenv("STRATA_QPN_BOUNDARY_DUMP");
        enabled_ = single_gpu && path && path[0];
        if (enabled_) { file_ = std::fopen(path, "wb"); good_ = file_ != nullptr; }
#else
        (void) single_gpu;
#endif
    }
    ~QpnBoundaryCapture() {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
        if (file_) std::fclose(file_);
#endif
    }
    QpnBoundaryCapture(const QpnBoundaryCapture&) = delete;
    bool enabled() const {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
        return enabled_;
#else
        return false;
#endif
    }
    bool record(const void* data, int64_t chunk, int64_t layer, int64_t phase,
                int64_t rows, int64_t cols, int64_t encoding, cudaStream_t stream) {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
        if (!enabled_) return true;
        if (!good_ || !data || rows < 1 || rows > 1024 || cols < 1 || cols > 10240 ||
            (encoding != 1 && encoding != 2)) return false;
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
            return false;
        const int64_t count = rows * cols;
        const size_t bytes = (size_t) count * (encoding == 1 ? 4 : 2);
        std::vector<uint8_t> host(bytes);
        if (cudaMemcpyAsync(host.data(), data, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) return false;
        const int64_t header[8] = {1, chunk, layer, phase, rows, cols, encoding, count};
        good_ = std::fwrite(header, sizeof(header), 1, file_) == 1 &&
                std::fwrite(host.data(), 1, bytes, file_) == bytes && std::fflush(file_) == 0;
        return good_;
#else
        (void) data; (void) chunk; (void) layer; (void) phase;
        (void) rows; (void) cols; (void) encoding; (void) stream;
        return true;
#endif
    }
};
} // namespace strata::prefill
