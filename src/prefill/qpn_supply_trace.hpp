#pragma once
// Experimental, aggregate-only diagnostics. No expert bytes or device arrays
// are read back. Every wrapper executes the original operation exactly once.
#include <cuda_runtime.h>
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL)
#include "qpn_r0_intervals.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#endif
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
    bool r0_aggregate_ = false;  // explicit opt-in: all instrumented layers, aggregate-only output
    // R0 reuses one event pair per category/layer rather than allocating
    // an event pair for every (potentially tens of thousands of) callbacks.
    // Its envelopes include gaps between operations and are NOT occupancy.
    std::array<std::array<Span, 48>, 5> r0_spans_{};
    std::array<std::uint64_t, 5> r0_operations_{};
    std::uint64_t r0_h2d_bytes_ = 0;
    std::atomic<std::size_t> r0_event_pairs_{0};
    std::atomic<bool> failed_{false};
    cudaEvent_t origin_ = nullptr;
    // Separate producer and consumer containers: only inspect after issuer join.
    std::vector<Span> copies_, reuse_, waits_, gathers_, products_;
    std::vector<Routes> routes_;
    bool sampled(int64_t layer) const {
        return r0_aggregate_ ? (layer >= 0 && layer < 48) :
               (layer == 0 || layer == 24 || layer == 47);
    }
    bool ok(cudaError_t e) { if (e != cudaSuccess) failed_.store(true); return e == cudaSuccess; }
    template<class F> cudaError_t measure_r0(int kind, int64_t l, size_t bytes,
                                             cudaStream_t s, F f) {
        // Existing supply callbacks use a single producer per operation kind.
        // The collector only inspects these containers after issuer.join().
        auto& ops = r0_operations_[std::size_t(kind)];
        if (ops == std::numeric_limits<std::uint64_t>::max() ||
            (kind == 0 && bytes > std::numeric_limits<std::uint64_t>::max() - r0_h2d_bytes_)) {
            failed_.store(true);
            return f();
        }
        ++ops;
        if (kind == 0) r0_h2d_bytes_ += bytes;
        Span& p = r0_spans_[std::size_t(kind)][std::size_t(l)];
        if (!p.begin) {
            if (r0_event_pairs_.fetch_add(1, std::memory_order_relaxed) >= 240 ||
                !ok(cudaEventCreate(&p.begin)) || !ok(cudaEventCreate(&p.end))) {
                failed_.store(true);
                return f();
            }
            p.layer = l;
            if (!ok(cudaEventRecord(p.begin, s))) return f();
        }
        // Re-recording the same end event retains the LAST completion on the
        // operation's stream; begin remains its FIRST occurrence in this layer.
        // This intentionally coalesces idle gaps into an upper envelope.
        const cudaError_t e = f();
        ok(e);
        ok(cudaEventRecord(p.end, s));
        return e;
    }
    template<class F> cudaError_t measure(std::vector<Span>& spans, int kind,
                                          int64_t l, size_t bytes,
                                          cudaStream_t s, F f) {
        if (!active_ || !sampled(l) || failed_.load()) return f();
        if (r0_aggregate_) return measure_r0(kind, l, bytes, s, f);
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
    // This branch emits only whole-chunk category envelopes and aggregate
    // metadata. Neither CUDA event envelopes nor their overlap establishes
    // exposed eligible compute, PCIe saturation, or request latency.
    bool report_r0_aggregate() {
        std::array<std::vector<r0::Window>, 5> categories;
        std::array<bool, 48> observed{};
        std::array<bool, 48> product_observed{};
        double earliest = std::numeric_limits<double>::infinity();
        double latest = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (int kind = 0; kind < 5; ++kind) {
            for (const auto& p : r0_spans_[std::size_t(kind)]) {
                if (!p.begin) continue;
                if (!p.end || p.layer < 0 || p.layer >= 48) return false;
                float a = 0, b = 0;
                if (!ok(cudaEventElapsedTime(&a, origin_, p.begin)) ||
                    !ok(cudaEventElapsedTime(&b, origin_, p.end)) ||
                    !std::isfinite(a) || !std::isfinite(b) || b < a)
                    return false;
                categories[std::size_t(kind)].push_back({a, b});
                observed[std::size_t(p.layer)] = true;
                if (kind == 4) product_observed[std::size_t(p.layer)] = true;
                earliest = std::min(earliest, double(a));
                latest = std::max(latest, double(b));
                any = true;
            }
        }
        const double h2d = r0::duration(categories[0]);
        const double products = r0::duration(categories[4]);
        auto waits = categories[1];
        waits.insert(waits.end(), categories[2].begin(), categories[2].end());
        const double wait = r0::duration(waits);
        const double gather = r0::duration(categories[3]);
        const double overlap = r0::overlap(categories[0], categories[4]);
        const double window = any ? latest - earliest : 0.0;
        const int observed_layers = int(std::count(observed.begin(), observed.end(), true));
        const int product_layers = int(std::count(product_observed.begin(), product_observed.end(), true));
        const std::size_t total_events = r0_event_pairs_.load(std::memory_order_relaxed);
        // No raw intervals, layer identities, route counts or private content.
        std::fprintf(stderr,
            "strata qpn-r0: {\"schema\":2,\"scope\":\"coalesced_layer_callback_envelopes_not_exposed_request\",\"instrumented\":true,"
            "\"coalesced_layer_envelopes\":true,\"callback_accounting_complete\":true,"
            "\"event_pairs\":%zu,\"observed_layers\":%d,\"product_layers\":%d,"
            "\"h2d_operations\":%zu,\"h2d_bytes\":%llu,"
            "\"trace_window_ms\":%.6f,\"h2d_envelope_ms\":%.6f,"
            "\"products_envelope_ms\":%.6f,\"wait_envelope_ms\":%.6f,"
            "\"gather_envelope_ms\":%.6f,\"h2d_products_overlap_ms\":%.6f}\n",
            total_events, observed_layers, product_layers,
            (std::size_t) r0_operations_[0], (unsigned long long) r0_h2d_bytes_,
            window, h2d, products, wait, gather, overlap);
        return true;
    }
public:
    QpnSupplyTrace(bool single_gpu, cudaStream_t compute, cudaStream_t copy) {
        const char* e = std::getenv("STRATA_QPN_SUPPLY_TRACE");
        const char* r0 = std::getenv("STRATA_QPN_SUPPLY_R0_AGGREGATE");
        r0_aggregate_ = single_gpu && r0 && r0[0] == '1';
        active_ = single_gpu && (r0_aggregate_ || (e && e[0] == '1'));
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
        for (auto& category : r0_spans_)
            for (auto& p : category) {
                if (p.begin) cudaEventDestroy(p.begin);
                if (p.end) cudaEventDestroy(p.end);
            }
        if (origin_) cudaEventDestroy(origin_);
    }
    QpnSupplyTrace(const QpnSupplyTrace&) = delete;
    bool enabled() const { return active_; }
    cudaError_t h2d(int64_t l, void* dst, const void* src, size_t bytes, cudaStream_t s) {
        return measure(copies_, 0, l, bytes, s, [&] { return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s); });
    }
    cudaError_t reuse_wait(int64_t l, cudaStream_t s, cudaEvent_t e) {
        return measure(reuse_, 1, l, 0, s, [&] { return cudaStreamWaitEvent(s, e, 0); });
    }
    cudaError_t ready_wait(int64_t l, cudaStream_t s, cudaEvent_t e) {
        return measure(waits_, 2, l, 0, s, [&] { return cudaStreamWaitEvent(s, e, 0); });
    }
    template<class F> void gather(int64_t l, cudaStream_t s, F f) {
        measure(gathers_, 3, l, 0, s, [&] { f(); return cudaSuccess; });
    }
    template<class F> void products(int64_t l, cudaStream_t s, F f) {
        measure(products_, 4, l, 0, s, [&] { f(); return cudaSuccess; });
    }
    template<class Count, class IsResident> void routes(int64_t l, int experts, Count count, IsResident resident) {
        // No route counts or identities in the R0 aggregate path.
        if (!active_ || r0_aggregate_ || !sampled(l)) return;
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
        if (r0_aggregate_) return report_r0_aggregate();
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
