// SPDX-License-Identifier: MIT
#include "strata/prefill/qpn_experiment_policy.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace d=strata::prefill::detail;
static void check_plan(const std::vector<int32_t>& counts){
    constexpr int K=10;
    const int n=counts.size();
    std::vector<int32_t> order, ids;
    for(int e=0;e<n;e++) if(counts[e]) order.push_back(e);
    for(int e=0;e<n;e++) for(int t=0;t<counts[e];t++) ids.push_back(e);
    const auto original_ids=ids;
    std::vector<int32_t> off(n+1,0),slot(ids.size(),-1),src(ids.size(),-1);
    int32_t s=0;
    for(int e=0;e<n;e++){off[e]=s;s+=counts[e];}off[n]=s;
    // Preserve token-major/rank-major ordering while exercising all input
    // IDs. The exact original tokens are arbitrary; no device workload here.
    const auto old_order=order, old_off=off,old_slot=slot,old_src=src;
    const bool expected=[](const std::vector<int32_t>& c){
        int active=0,chosen=0;
        for(auto v:c)if(v>0){active++;chosen+=(v>=17&&v<=32);}
        return (active-chosen+15)/16*16<active;
    }(counts);
    const bool changed=d::qpn_partition_rows(ids.data(),ids.size(),K,counts,off,order,slot.data(),src.data());
    assert(changed==expected);
    if(!changed){assert(order==old_order&&off==old_off&&slot==old_slot&&src==old_src);return;}
    assert(ids==original_ids);
    std::vector<int32_t> seen(ids.size(),0);
    for(std::size_t i=0;i<ids.size();i++){
        const int32_t pos=slot[i];
        assert(pos>=0 && std::size_t(pos)<ids.size()&&!seen[pos]++);
        assert(src[pos]==int32_t(i/K));
        assert(pos>=off[ids[i]] && pos<off[ids[i]]+counts[ids[i]]);
    }
    for(auto v:seen)assert(v==1);
    int candidates=0;
    for(std::size_t j=0;j<order.size();j+=16)
        if(d::qpn_entire_group(order,j,std::min(order.size(),j+16),counts))candidates++;
    assert(candidates>0);
}
int main(){
    assert(!d::qpn_opt_in(nullptr));assert(!d::qpn_opt_in("0"));
    assert(!d::qpn_opt_in("true"));assert(d::qpn_opt_in("1"));
    assert(!d::qpn_candidate(16)&&d::qpn_candidate(17)&&d::qpn_candidate(32)&&!d::qpn_candidate(33));
    check_plan(std::vector<int32_t>(32,24));
    check_plan({1,2,3,8,16,33,200,64,7,4,10,17,18,19,20,21,22,23,24,
                25,26,27,28,29,30,31,32,17,18,19,20,21,22,23,24,25,26});
    check_plan({1,2,4,8,13,16,17,18}); // no pure group, must be unchanged
    check_plan({16,18,0,0,0}); // no all-candidate group
    {
        std::vector<int32_t> c={1,1};std::vector<int32_t> order={0,0};
        std::vector<int32_t> off(3),slot(2),src(2),ids={0,1};
        bool rejected=false;
        try { d::qpn_partition_rows(ids.data(),2,10,c,off,order,slot.data(),src.data());}
        catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    {
        // Host plan cannot omit an active expert.
        std::vector<int32_t> c={17,18};std::vector<int32_t> order={0};
        std::vector<int32_t> off(3),slot(35),src(35),ids(35,0);
        ids.back()=1;bool rejected=false;
        try{d::qpn_partition_rows(ids.data(),35,10,c,off,order,slot.data(),src.data());}
        catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    {
        // Packed eligibility must exclude canonical resident slots while preserving every route.
        std::vector<int32_t> c(32,64),order(32),off(33),ids,slot(32*64),src(32*64);
        std::vector<uint8_t> eligible(32,1);
        for(int e=0;e<32;e++) { order[e]=e;off[e]=e*64;if(e<16)eligible[e]=0;
            for(int r=0;r<64;r++)ids.push_back(e); }
        off[32]=32*64;
        assert(d::qpn_partition_rows(ids.data(),ids.size(),10,c,off,order,slot.data(),src.data(),16,33,512,&eligible));
        assert(!d::qpn_entire_group(order,0,16,c,33,512,&eligible));
        assert(d::qpn_entire_group(order,16,32,c,33,512,&eligible));
        std::vector<uint8_t> seen(ids.size(),0);
        for(size_t i=0;i<ids.size();i++) { assert(!seen[slot[i]]++);assert(src[slot[i]]==int32_t(i/10)); }
        for(auto v:seen)assert(v==1);
        eligible.pop_back();bool rejected=false;
        try{ d::qpn_partition_rows(ids.data(),ids.size(),10,c,off,order,slot.data(),src.data(),16,33,512,&eligible); }
        catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    }
    std::puts("QPN full-model opt-in policy ownership/eligibility PASS");
}
