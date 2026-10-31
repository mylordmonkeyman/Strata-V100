// Synthetic encoded QAD -> pinned ggml NVFP4 dequant -> Q4TC quantization fixture.
// Host-only; synthetic IPC is captured by Python and never printed in CI logs.
#include "ggml.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::vector<block_nvfp4> qad_bytes(int k, int seed) {
    if (k <= 0 || k % 64) throw std::runtime_error("invalid synthetic QAD shape");
    std::vector<block_nvfp4> q((size_t)k / 64);
    for (int i = 0; i < k; i += 16) {
        const int block = i / 64, sub = (i % 64) / 16;
        q[(size_t)block].d[sub] = (uint8_t)(0x28 + ((i / 16 + seed * 2) % 9) * 3);
        for (int j = 0; j < 8; ++j) {
            const int lo = ((i + j) * 7 + ((i + j) / 16) * 3 + seed * 5) & 15;
            const int hi = ((i + 8 + j) * 7 + ((i + 8 + j) / 16) * 3 + seed * 5) & 15;
            q[(size_t)block].qs[sub * 8 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
    return q;
}
std::string hex(const void* data,size_t n){
    constexpr char h[]="0123456789abcdef";
    const auto* p=(const uint8_t*)data;
    std::string s(2*n,'0');
    for(size_t i=0;i<n;++i){s[2*i]=h[p[i]>>4];s[2*i+1]=h[p[i]&15];}
    return s;
}
template<class B> void emit(const char* role,int k,int seed,float tail,ggml_type type,int block_k,
    void(*ref)(const float*,B*,int64_t),void(*deq)(const B*,float*,int64_t)){
    const auto qad = qad_bytes(k,seed);
    if (sizeof(block_nvfp4) != 36 || ggml_blck_size(GGML_TYPE_NVFP4) != 64 ||
        ggml_row_size(GGML_TYPE_NVFP4,k) != qad.size() * sizeof(block_nvfp4))
        throw std::runtime_error("pinned ggml QAD block ABI mismatch");
    std::vector<float> original((size_t)k);
    dequantize_row_nvfp4(qad.data(),original.data(),k);
    for(float v:original)if(!std::isfinite(v))throw std::runtime_error("nonfinite QAD source");
    for(int fold=0;fold<2;++fold){
        std::vector<float> src=original;
        if(fold)for(float& v:src)v*=tail;
        const size_t n=(size_t)k/block_k;
        std::vector<B> q(n),qr(n);
        ref(src.data(),qr.data(),k);
        const size_t bytes=ggml_quantize_chunk(type,src.data(),q.data(),0,1,k,nullptr);
        if(bytes!=n*sizeof(B)||ggml_row_size(type,k)!=bytes||
           ggml_blck_size(type)!=block_k||ggml_type_size(type)!=sizeof(B))
            throw std::runtime_error("pinned ggml ABI mismatch");
        std::vector<float> out((size_t)k);
        deq(q.data(),out.data(),k);
        for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite dequant");
        const bool same=std::memcmp(q.data(),qr.data(),bytes)==0;
        // Synthetic IPC only; CI captures stdout, no raw data in logs.
        std::cout<<role<<' '<<(fold?"folded":"tail")<<' '<<k<<' '<<(same?1:0)<<' '
                 <<hex(src.data(),src.size()*sizeof(float))<<' '
                 <<hex(q.data(),bytes)<<' '<<hex(qr.data(),bytes)<<' '
                 <<hex(out.data(),out.size()*sizeof(float))<<' '
                 <<hex(qad.data(),qad.size()*sizeof(block_nvfp4))<<' '
                 <<hex(original.data(),original.size()*sizeof(float))<<'\n';
    }
}
}
int main(){
    try{
        static_assert(sizeof(block_nvfp4)==36);
        static_assert(sizeof(block_q4_K)==144);
        static_assert(sizeof(block_iq4_nl)==18);
        emit("gate",2560,0,1.25f,GGML_TYPE_Q4_K,256,quantize_row_q4_K_ref,dequantize_row_q4_K);
        emit("up",2560,1,.5f,GGML_TYPE_Q4_K,256,quantize_row_q4_K_ref,dequantize_row_q4_K);
        emit("down",640,2,2.f,GGML_TYPE_IQ4_NL,32,quantize_row_iq4_nl_ref,dequantize_row_iq4_nl);
        return 0;
    }catch(const std::exception& e){std::cerr<<"Q4TC pinned fixture error: "<<e.what()<<'\n';return 1;}
}
