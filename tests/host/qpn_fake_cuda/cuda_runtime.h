#pragma once
// Test-only CUDA API shim. Include solely with -I tests/host/qpn_fake_cuda.
// No real driver/runtime/device or GPU operations are involved.
#include <cstddef>
#include <cstdint>

struct FakeCudaEvent { double milliseconds = 0.0; bool recorded = false; };
using cudaEvent_t = FakeCudaEvent*;
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 1 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1 };

namespace fake_cuda {
inline double clock_ms = 0;
inline std::size_t events_created = 0;
inline std::size_t event_records = 0;
inline std::size_t copies = 0;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new FakeCudaEvent();
    ++fake_cuda::events_created;
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
    if (!e) return cudaErrorUnknown;
    e->milliseconds = (++fake_cuda::clock_ms) * 0.01;
    e->recorded = true;
    ++fake_cuda::event_records;
    return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float* elapsed, cudaEvent_t start, cudaEvent_t end) {
    if (!elapsed || !start || !end || !start->recorded || !end->recorded)
        return cudaErrorUnknown;
    *elapsed = float(end->milliseconds - start->milliseconds);
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
    if (!status) return cudaErrorUnknown;
    *status = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t,
                                   cudaMemcpyKind, cudaStream_t) {
    ++fake_cuda::copies;
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) {
    return cudaSuccess;
}
