#pragma once
// Default-off, PRIVATE R1 route-count snapshots. No stdout or model weights.
// Linux-only experimental host-grouped path; zero allocations when not enabled.
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL) && defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::prefill {
class QpnR1PrefillCapture {
    bool requested_ = false;
    bool good_ = true;
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL) && defined(__linux__)
    std::FILE* file_ = nullptr;
    template<typename T> bool scalar(T v) {
        return std::fwrite(&v, sizeof(T), 1, file_) == 1;
    }
#endif
public:
    explicit QpnR1PrefillCapture(bool single_gpu) {
        const char* p = std::getenv("STRATA_QPN_R1_PRIVATE_PREFILL_CAPTURE");
        requested_ = p && *p;
        if (!requested_) return;
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL) && defined(__linux__)
        if (!single_gpu || p[0] != '/') { good_ = false; return; }
        int fd = ::open(p, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) { good_ = false; return; }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
            ::fchmod(fd, 0600) != 0) { ::close(fd); good_ = false; return; }
        file_ = ::fdopen(fd, "wb");
        if (!file_) { ::close(fd); good_ = false; }
#else
        (void)single_gpu;
        good_ = false; // refuse capture if the experimental Linux implementation was not compiled
#endif
    }
    ~QpnR1PrefillCapture() {
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL) && defined(__linux__)
        if (file_) std::fclose(file_);
#endif
    }
    QpnR1PrefillCapture(const QpnR1PrefillCapture&) = delete;
    bool good() const { return good_; }
    bool enabled() const { return requested_ && good_; }
    bool record(int64_t token_offset, int64_t tokens, int64_t layer, int64_t top_k,
                const std::vector<int32_t>& counts, const int32_t* residence,
                int32_t experts, bool stream_all) {
        if (!requested_) return true;
#if defined(STRATA_QPN_PREFILL_EXPERIMENTAL) && defined(__linux__)
        if (!good_ || !file_ || token_offset < 0 || tokens < 1 || tokens > 32768 ||
            layer < 0 || layer >= 48 || top_k < 1 || top_k > 512 ||
            experts != 512 || counts.size() != (size_t)experts) return false;
        int64_t sum = 0;
        for (const auto c : counts) { if (c < 0 || c > tokens) return false; sum += c; }
        if (sum != tokens * top_k) return false;
        const char magic[4] = {'Q','P','R','1'};
        const uint32_t version = 1;
        const int32_t l = (int32_t)layer, k = (int32_t)top_k;
        const int32_t flags = stream_all ? 1 : 0;
        std::vector<uint8_t> resident((size_t)experts, 0);
        for (int e = 0; e < experts; ++e)
            resident[(size_t)e] = residence && residence[(size_t)layer * experts + e] >= 0 ? 1 : 0;
        good_ = std::fwrite(magic, 1, 4, file_) == 4 && scalar(version) &&
            scalar(token_offset) && scalar(tokens) && scalar(l) && scalar(experts) &&
            scalar(k) && scalar(flags) &&
            std::fwrite(counts.data(), sizeof(int32_t), counts.size(), file_) == counts.size() &&
            std::fwrite(resident.data(), 1, resident.size(), file_) == resident.size() &&
            std::fflush(file_) == 0;
        return good_;
#else
        (void)token_offset; (void)tokens; (void)layer; (void)top_k;
        (void)counts; (void)residence; (void)experts; (void)stream_all;
        return false;
#endif
    }
};
}  // namespace strata::prefill
