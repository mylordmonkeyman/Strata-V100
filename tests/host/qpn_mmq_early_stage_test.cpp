// Host-only regression for pre-route publication and early MMQ staging.
#include "../../src/prefill/qpn_mmq_early_stage.hpp"
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <cstdio>

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
int main() {
    using strata::prefill::detail::qpn_mmq_early_stage_skip;
    std::atomic<int64_t> published{-1};
    bool unused0=false, used1=true, unused2=false;
    require(!qpn_mmq_early_stage_skip(published,0,&unused0),"unpublished skip");
    require(!qpn_mmq_early_stage_skip(published,0,&used1),"unpublished active skip");
    require(!qpn_mmq_early_stage_skip(published,-1,&unused0),"invalid layer skip");
    require(!qpn_mmq_early_stage_skip(published,0,nullptr),"null route skip");
    published.store(0,std::memory_order_release);
    require(qpn_mmq_early_stage_skip(published,0,&unused0),"published unused not skipped");
    require(!qpn_mmq_early_stage_skip(published,0,&used1),"published active incorrectly skipped");
    require(!qpn_mmq_early_stage_skip(published,1,&unused2),"future layer incorrectly skipped");
    // Fresh publication: route flag's ordinary store happens-before worker's
    // acquire-load, so no speculative reads of an unpublished layer occur.
    for (int trial=0;trial<512;++trial) {
        std::atomic<int64_t> ready{-1};
        bool routed=true;
        std::thread producer([&] { routed=false; ready.store(2,std::memory_order_release); });
        std::thread worker([&] {
            while (ready.load(std::memory_order_acquire)<2) std::this_thread::yield();
            require(qpn_mmq_early_stage_skip(ready,2,&routed),"published route unavailable");
        });
        producer.join();worker.join();
    }
    // Regression for the unsafe issuer/worker interleaving: the issuer
    // initially sees routing unpublished, then the worker sees an unused
    // published expert and omits staging. Issuer MUST wait for ready before
    // its final copy decision or it could upload unprepared buffer contents.
    {
        std::atomic<int64_t> pub{-1};
        std::atomic<bool> issuer_early_checked{false}, worker_ready{false};
        bool selected=false, actually_staged=true;
        std::thread worker([&] {
            while (pub.load(std::memory_order_acquire)<0) std::this_thread::yield();
            actually_staged = !qpn_mmq_early_stage_skip(pub,0,&selected);
            worker_ready.store(true,std::memory_order_release);
        });
        std::thread issuer([&] {
            require(!qpn_mmq_early_stage_skip(pub,0,&selected),"unpublished issuer should copy");
            issuer_early_checked.store(true,std::memory_order_release);
            while (!worker_ready.load(std::memory_order_acquire)) std::this_thread::yield();
            const bool needs_copy = !qpn_mmq_early_stage_skip(pub,0,&selected);
            require(!needs_copy && !actually_staged,"staging/issuer decision mismatch");
        });
        while (!issuer_early_checked.load(std::memory_order_acquire)) std::this_thread::yield();
        pub.store(0,std::memory_order_release);
        worker.join(); issuer.join();
    }
    std::puts("Early MMQ staging predicate: all host tests passed");
}
