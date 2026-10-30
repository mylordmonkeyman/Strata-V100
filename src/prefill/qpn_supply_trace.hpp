#pragma once
// Experimental, aggregate-only diagnostics. No expert bytes or device arrays
// are read back. Every wrapper executes the original operation exactly once.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <atomic>

namespace strata::prefill {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
class QpnSupplyTrace {
    struct Span { int64_t layer; size_t bytes; cudaEvent_t begin = nullptr, end = nullptr; };
    struct Routes { int64_t layer, groups, resident, rows, resident_rows; };
    bool active_ = false;
    std::atomic<bool> failed_{false};
    cudaEvent_t origin_ = nullptr;
    // Separate producer and consumer containers: only inspect after issuer join.
    std::vector<Span> copies_, reuse_, waits_, gathers_, products_;
    std::vector<Routes> routes_;
    static bool sampled(int64_t layer) { return layer == 0 || layer == 24 || layer == 47; }
    bool ok(cudaError_t e) { if (e != cudaSuccess) failed_.store(true); return e == cudaSuccess; }
    template<class F> cudaError_t measure(std::vector<Span>& spans, int64_t l, size_t bytes,
                                          cudaStream_t s, F f) {
        if (!active_ || !sampled(l) || failed_.load()) return f();
        Span p{l, bytes};
        if (!ok(cudaEventCreate(&p.begin)) || !ok(cudaEventCreate(&p.end))) {
            if (p.begin) cudaEventDestroy(p.begin);
            if (p.end) cudaEventDestroy(p.end);
            return f();
        }
        ok(cudaEventRecord(p.begin, s));
        const cudaError_t e = f();
        ok(e); ok(cudaEventRecord(p.end, s));
        spans.push_back(p);
        return e;
    }
public:
    QpnSupplyTrace(bool single_gpu, cudaStream_t compute, cudaStream_t copy) {
        const char* e = std::getenv("STRATA_QPN_SUPPLY_TRACE");
        active_ = single_gpu && e && e[0] == '1';
        if (!active_) return;
        cudaStreamCaptureStatus a, b;
        if (!ok(cudaStreamIsCapturing(compute, &a)) || !ok(cudaStreamIsCapturing(copy, &b)) ||
            a != cudaStreamCaptureStatusNone || b != cudaStreamCaptureStatusNone) {
            failed_.store(true); return;
        }
        if (ok(cudaEventCreate(&origin_))) ok(cudaEventRecord(origin_, copy));
    }
    ~QpnSupplyTrace() {
        for (auto* v : {&copies_, &reuse_, &waits_, &gathers_, &products_})
            for (const auto& p : *v) { cudaEventDestroy(p.begin); cudaEventDestroy(p.end); }
        if (origin_) cudaEventDestroy(origin_);
    }
    QpnSupplyTrace(const QpnSupplyTrace&) = delete;
    bool enabled() const { return active_; }
    cudaError_t h2d(int64_t l, void* dst, const void* src, size_t bytes, cudaStream_t s) {
        return measure(copies_, l, bytes, s, [&] { return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s); });
    }
    cudaError_t reuse_wait(int64_t l, cudaStream_t s, cudaEvent_t e) {
        return measure(reuse_, l, 0, s, [&] { return cudaStreamWaitEvent(s, e, 0); });
    }
    cudaError_t ready_wait(int64_t l, cudaStream_t s, cudaEvent_t e) {
        return measure(waits_, l, 0, s, [&] { return cudaStreamWaitEvent(s, e, 0); });
    }
    template<class F> void gather(int64_t l, cudaStream_t s, F f) {
        measure(gathers_, l, 0, s, [&] { f(); return cudaSuccess; });
    }
    template<class F> void products(int64_t l, cudaStream_t s, F f) {
        measure(products_, l, 0, s, [&] { f(); return cudaSuccess; });
    }
    template<class Count, class IsResident> void routes(int64_t l, int experts, Count count, IsResident resident) {
        if (!active_ || !sampled(l)) return;
        Routes r{l, 0, 0, 0, 0};
        for (int e = 0; e < experts; ++e) if (count(e) > 0) {
            ++r.groups; r.rows += count(e);
            if (resident(e)) { ++r.resident; r.resident_rows += count(e); }
        }
        routes_.push_back(r);
    }
    // Diagnostic synchronization only, once the chunk's issuer has joined.
    // Event intervals include event/submission overhead; do not use instrumented
    // throughput as the performance result. Per-operation intervals can overlap.
    bool report(int64_t chunk, int64_t tokens, int ring, bool stream_all,
                cudaStream_t compute, cudaStream_t copy) {
        if (!active_) return true;
        const bool c = ok(cudaStreamSynchronize(copy));
        const bool d = ok(cudaStreamSynchronize(compute));
        if (!c || !d || failed_.load()) return false;
        // Prevalidate all event reads before emitting a complete JSON record.
        for (auto* v : {&copies_, &reuse_, &waits_, &gathers_, &products_}) for (auto& p : *v) {
            float a, b;
            ok(cudaEventElapsedTime(&a, origin_, p.begin));
            ok(cudaEventElapsedTime(&b, origin_, p.end));
        }
        if (failed_.load()) return false;
        std::fprintf(stderr, "strata qpn-supply: {\"schema\":1,\"chunk\":%lld,\"tokens\":%lld,\"ring\":%d,\"stream_all\":%s,\"instrumented\":true,\"layers\":[",
                     (long long) chunk, (long long) tokens, ring, stream_all ? "true" : "false");
        bool first_layer = true;
        for (int l : {0, 24, 47}) {
            bool seen = false;
            for (const auto& p : copies_) seen |= p.layer == l;
            for (const auto& r : routes_) seen |= r.layer == l;
            for (const auto& p : waits_) seen |= p.layer == l;
            if (!seen) continue;
            std::fprintf(stderr, "%s{\"layer\":%d,\"routed\":", first_layer ? "" : ",", l);
            first_layer = false;
            bool have_routes = false;
            for (const auto& r : routes_) if (r.layer == l) {
                std::fprintf(stderr, "{\"groups\":%lld,\"resident_groups\":%lld,\"rows\":%lld,\"resident_rows\":%lld}",
                    (long long) r.groups, (long long) r.resident, (long long) r.rows, (long long) r.resident_rows);
                have_routes = true; break;
            }
            if (!have_routes) std::fprintf(stderr, "null"); // fused GPU routing: deliberately no new readback
            const char* names[] = {"h2d", "slot_wait", "ready_wait", "gather", "products"};
            int kind = 0;
            for (auto* v : {&copies_, &reuse_, &waits_, &gathers_, &products_}) {
                std::fprintf(stderr, ",\"%s\":[", names[kind++]);
                bool first = true;
                for (const auto& p : *v) if (p.layer == l) {
                    float a = 0, b = 0;
                    cudaEventElapsedTime(&a, origin_, p.begin); cudaEventElapsedTime(&b, origin_, p.end);
                    std::fprintf(stderr, "%s[%.6f,%.6f,%llu]", first ? "" : ",", a, b, (unsigned long long) p.bytes);
                    first = false;
                }
                std::fprintf(stderr, "]");
            }
            std::fprintf(stderr, "}");
        }
        std::fprintf(stderr, "]}\n");
        return true;
    }
};
#else
// Compile-time off: preserve non-Volta/HIP builds without capture/timing API
// dependencies, containers, atomics, runtime checks or diagnostic allocations.
class QpnSupplyTrace {
public:
    QpnSupplyTrace(bool, cudaStream_t, cudaStream_t) {}
    cudaError_t h2d(int64_t, void* d, const void* p, size_t n, cudaStream_t s) {
        return cudaMemcpyAsync(d, p, n, cudaMemcpyHostToDevice, s);
    }
    cudaError_t reuse_wait(int64_t, cudaStream_t s, cudaEvent_t e) { return cudaStreamWaitEvent(s, e, 0); }
    cudaError_t ready_wait(int64_t, cudaStream_t s, cudaEvent_t e) { return cudaStreamWaitEvent(s, e, 0); }
    template<class F> void gather(int64_t, cudaStream_t, F f) { f(); }
    template<class F> void products(int64_t, cudaStream_t, F f) { f(); }
    template<class C, class R> void routes(int64_t, int, C, R) {}
    bool report(int64_t, int64_t, int, bool, cudaStream_t, cudaStream_t) { return true; }
};
#endif
} // namespace strata::prefill
