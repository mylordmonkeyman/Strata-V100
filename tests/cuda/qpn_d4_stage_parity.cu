// Standalone, manually invoked synthetic MMQ D4 staging qualification.
// No model, expert weight, GGUF, cache, QPN dispatch, or benchmarking.
// Compile target only by default; run only under existing V100 gpu.lock.
#include "strata/kernels/nvfp4_volta_qpn.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace q = strata::kernels::qpn;
struct D4 {
    float d[4];
    std::int8_t qs[128];
};
static_assert(sizeof(D4) == 144, "ggml MMQ D4 block ABI");

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
};
struct Buffer {
    void* value = nullptr;
    explicit Buffer(std::size_t bytes) { ck(cudaMalloc(&value, bytes)); }
    ~Buffer() { if (value) cudaFree(value); }
    Buffer(const Buffer&) = delete;
};

std::uint16_t fp16_bits(float f) {
    const half h = __float2half_rn(f);
    std::uint16_t out = 0;
    static_assert(sizeof(h) == sizeof(out), "FP16 ABI");
    std::memcpy(&out, &h, sizeof(out));
    return out;
}
std::vector<D4> make_input(int rows, int cols, int pitch) {
    // MMQ producer pads columns to 512, even for logical 128 or 640.
    const int physical = ((cols + 511) / 512) * 512;
    std::vector<D4> blocks(std::size_t(physical / 128) * pitch);
    for (int block = 0; block < physical / 128; ++block)
        for (int row = 0; row < pitch; ++row) {
            auto& dst = blocks[std::size_t(block) * pitch + row];
            for (int k = 0; k < 4; ++k)
                dst.d[k] = row >= rows ? std::numeric_limits<float>::quiet_NaN()
                                      : (k + 1) * 0.125f + block * 0.03125f + row * 0.0625f;
            for (int k = 0; k < 128; ++k)
                dst.qs[k] = std::int8_t((k * 13 + row * 7 + block * 3) % 255 - 127);
        }
    return blocks;
}
std::vector<std::uint16_t> reference(const std::vector<D4>& blocks, int rows,
                                     int cols, int pitch) {
    std::vector<std::uint16_t> result(std::size_t(rows) * cols);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) {
            const auto& b = blocks[std::size_t(col / 128) * pitch + row];
            result[std::size_t(row) * cols + col] =
                fp16_bits(b.d[(col % 128) / 32] * float(b.qs[col % 128]));
        }
    return result;
}
void must_match(const std::vector<std::uint16_t>& got,
                const std::vector<std::uint16_t>& expected) {
    if (got != expected) throw std::runtime_error("D4 device/host FP16 parity failure");
}
void run_case(int rows, int cols, int pitch, bool graph_check) {
    const std::size_t count = std::size_t(rows) * cols;
    auto blocks = make_input(rows, cols, pitch);
    const std::size_t bytes = blocks.size() * sizeof(D4);
    constexpr std::size_t guard_bytes = 32;
    constexpr std::size_t mmq_tail_bytes = 128 * 144;
    Buffer input(bytes + mmq_tail_bytes), output(count * 2 + guard_bytes),
           error(sizeof(unsigned));
    Stream stream;
    auto* out = static_cast<std::uint16_t*>(output.value);
    auto* invalid = static_cast<unsigned*>(error.value);
    const auto stage = [&] {
        q::stage_mmq_d4_q8_1_to_fp16(input.value, bytes + mmq_tail_bytes,
                                      out, rows, cols, pitch,
                                      invalid, stream.value);
    };
    const auto transfer = [&] {
        // Actual mmq::q8_bytes supplies an unused 128-block overread guard.
        // Poison that tail to prove the stage reads only the producer extent.
        ck(cudaMemsetAsync(input.value, 0xa5, bytes + mmq_tail_bytes, stream.value));
        ck(cudaMemcpyAsync(input.value, blocks.data(), bytes,
                           cudaMemcpyHostToDevice, stream.value));
        ck(cudaMemsetAsync(error.value, 0, sizeof(unsigned), stream.value));
        ck(cudaMemsetAsync(output.value, 0xa5, count * 2 + guard_bytes, stream.value));
    };
    const auto read = [&] {
        ck(cudaStreamSynchronize(stream.value));
        unsigned invalid_host = 0;
        ck(cudaMemcpy(&invalid_host, error.value, sizeof(invalid_host), cudaMemcpyDeviceToHost));
        if (invalid_host) throw std::runtime_error("unexpected nonfinite/overflow D4 stage");
        std::vector<std::uint16_t> got(count + guard_bytes / 2);
        ck(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
        const auto expected = reference(blocks, rows, cols, pitch);
        must_match(std::vector<std::uint16_t>(got.begin(), got.begin() + count), expected);
        for (std::size_t i = count; i < got.size(); ++i)
            if (got[i] != 0xa5a5) throw std::runtime_error("D4 output tail guard overwritten");
        return expected;
    };

    transfer();
    stage();
    auto before = read();
    // Caller-side malformed shape/stride/buffer checks reject before launch.
    for (int bad = 0; bad < 4; ++bad) {
        bool rejected = false;
        try {
            q::stage_mmq_d4_q8_1_to_fp16(input.value, bad == 0 ? bytes - 1 : bytes,
                out, bad == 1 ? 0 : rows, bad == 2 ? cols + 1 : cols,
                bad == 3 ? rows - 1 : pitch, invalid, stream.value);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("D4 malformed input was accepted");
    }

    if (graph_check) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        stage();
        ck(cudaStreamEndCapture(stream.value, &graph));
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        blocks[0].d[0] *= 2.0f;
        transfer();
        ck(cudaGraphLaunch(exec, stream.value));
        const auto after = read();
        if (after == before) throw std::runtime_error("D4 Graph replay ignored changed input");
        ck(cudaGraphExecDestroy(exec));
        ck(cudaGraphDestroy(graph));
    }

    // Invalid scale must be counted and its output masked to zero; no consumer
    // may use the result when invalid_host is nonzero.
    blocks[0].d[0] = std::numeric_limits<float>::quiet_NaN();
    transfer();
    stage();
    ck(cudaStreamSynchronize(stream.value));
    unsigned invalid_host = 0;
    ck(cudaMemcpy(&invalid_host, error.value, sizeof(invalid_host), cudaMemcpyDeviceToHost));
    if (invalid_host != 32u) throw std::runtime_error("D4 nonfinite rejection count");
    std::uint16_t zero = 1;
    ck(cudaMemcpy(&zero, out, sizeof(zero), cudaMemcpyDeviceToHost));
    if (zero != 0) throw std::runtime_error("D4 invalid output must be zero");
}
} // namespace

int main() try {
    int id = 0;
    ck(cudaGetDevice(&id));
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, id));
    if (prop.major != 7 || prop.minor != 0) {
        std::fprintf(stderr, "D4 parity requires sm_70; no GPU test performed\n");
        return 77;
    }
    run_case(1, 128, 1, true);
    run_case(3, 256, 5, false);
    run_case(2, 640, 4, false);
    run_case(3, 2560, 5, false);
    std::puts("{\"schema\":1,\"test\":\"synthetic_mmq_d4_stage\",\"status\":\"passed\",\"cases\":4,\"graph_changed_input\":true,\"no_model_or_weights\":true}");
    return 0;
} catch (const std::exception& e) {
    // Error text is fixed code-controlled diagnostic, never from private data.
    std::fprintf(stderr, "D4 stage parity: %s\n", e.what());
    return 1;
}
