// SPDX-License-Identifier: MIT
// Explicitly experimental host-only QPN routing permutation.
// Never called by the default engine. Keep each expert's genuine rows in
// token-major, rank-major order and retain one-to-one route ownership.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::prefill::detail {
inline bool qpn_opt_in(const char* flag) { return flag && std::strcmp(flag,"1")==0; }
inline bool qpn_candidate(int32_t count) { return count >= 17 && count <= 32; }

inline bool qpn_entire_group(const std::vector<int32_t>& order, std::size_t first,
                            std::size_t end, const std::vector<int32_t>& count) {
    if (first>=end || end>order.size()) return false;
    for (std::size_t j=first;j<end;j++) {
        int32_t e=order[j];
        if (e<0 || std::size_t(e)>=count.size() || !qpn_candidate(count[std::size_t(e)])) return false;
    }
    return true;
}

// An opt-in, single-GPU, single-schedule experiment ONLY: the caller must
// have ensured no CPU/peer share, no stream_all prefetch, and all active expert
// IDs appear in order once. As with MMQ resident sort, off[e] becomes the
// expert's START, not the previous sorted-ID prefix. No device writes here.
// Returns false without mutation if no all-candidate group would be formed.
// Throws on malformed plans rather than leaving a partially rewritten route.
inline bool qpn_partition_rows(const int32_t* ids, std::size_t routes, int32_t top_k,
                               const std::vector<int32_t>& count, std::vector<int32_t>& off,
                               std::vector<int32_t>& order, int32_t* slot, int32_t* src,
                               std::size_t group=16) {
    if (!ids || !slot || !src || !top_k || group!=16 ||
        count.empty() || off.size()!=count.size()+1 ||
        routes>std::size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("QPN partition invalid input geometry");
    std::vector<uint8_t> present(count.size(),0);
    std::size_t counted=0, chosen=0;
    for (int32_t e:order) {
        if(e<0 || std::size_t(e)>=count.size() || present[std::size_t(e)]++ ||
           count[std::size_t(e)]<1)
            throw std::invalid_argument("QPN partition duplicate/inactive expert");
        counted+=std::size_t(count[std::size_t(e)]);
        chosen+=qpn_candidate(count[std::size_t(e)]);
    }
    if(counted!=routes)
        throw std::invalid_argument("QPN partition missing routed entries");
    for(std::size_t i=0;i<routes;i++){
        int32_t e=ids[i];
        if(e<0 || std::size_t(e)>=count.size() || !present[std::size_t(e)])
            throw std::invalid_argument("QPN partition unowned route");
    }
    const std::size_t fallback=order.size()-chosen;
    const std::size_t first_group=(fallback+group-1)/group;
    if (!chosen || first_group*group>=order.size()) return false;
    std::vector<int32_t> candidate=order;
    std::stable_partition(candidate.begin(),candidate.end(),
        [&](int32_t e){return !qpn_candidate(count[std::size_t(e)]);});
    if(!qpn_entire_group(candidate,first_group*group,
                         std::min(candidate.size(),(first_group+1)*group),count))
        throw std::invalid_argument("QPN partition cannot form eligible group");
    std::vector<int32_t> starts(off.size(),0);
    int32_t next=0;
    for(int32_t e:candidate){
        starts[std::size_t(e)]=next;
        next+=count[std::size_t(e)];
    }
    if(std::size_t(next)!=routes)
        throw std::invalid_argument("QPN partition row accounting mismatch");
    starts.back()=next;
    std::vector<int32_t> fill(starts.begin(),starts.end()-1);
    std::vector<int32_t> new_slot(routes),new_src(routes,-1);
    for(std::size_t i=0;i<routes;i++) {
        const int32_t e=ids[i];
        const int32_t p=fill[std::size_t(e)]++;
        if(p<0 || std::size_t(p)>=routes || new_src[std::size_t(p)]!=-1)
            throw std::invalid_argument("QPN partition duplicate/out-of-range slot");
        new_slot[i]=p;
        new_src[std::size_t(p)]=int32_t(i/std::size_t(top_k));
    }
    for(int32_t e:candidate)
        if(fill[std::size_t(e)]!=starts[std::size_t(e)]+count[std::size_t(e)])
            throw std::invalid_argument("QPN partition missing expert row");
    if(std::find(new_src.begin(),new_src.end(),-1)!=new_src.end())
        throw std::invalid_argument("QPN partition missing mapped row");
    order.swap(candidate);off.swap(starts);
    std::copy(new_slot.begin(),new_slot.end(),slot);
    std::copy(new_src.begin(),new_src.end(),src);
    return true;
}
} // namespace strata::prefill::detail
