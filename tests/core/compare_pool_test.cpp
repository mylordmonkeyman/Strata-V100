#include "strata/kernels/cpu/pool.hpp"
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>

namespace cpu = strata::kernels::cpu;
int main() {
    if (!cpu::cpu_features().usable()) return 77;
    std::mt19937 rng(9);
    std::vector<std::vector<uint8_t>> blobs(4, std::vector<uint8_t>(cpu::BLOB));
    for (auto& blob : blobs) {
        for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) blob[i] = static_cast<uint8_t>(rng());
        for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
            const uint16_t scale = static_cast<uint16_t>(0x1c00 + rng() % 0x0800);
            std::memcpy(blob.data() + i, &scale, 2);
        }
    }
    std::vector<float> x(cpu::H);
    for (auto& value : x) value = static_cast<float>(static_cast<int>(rng() % 257) - 128) / 128;
    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);
    std::vector<std::vector<float>> expected(4, std::vector<float>(cpu::H));
    cpu::ExpertScratch scratch;
    for (int e = 0; e < 4; ++e) cpu::s2_expert_vnni_q(blobs[e].data(), act, expected[e].data(), scratch);
    for (int workers : {1, 4}) for (bool host : {false, true}) {
        cpu::ExpertPool pool(workers, false, host);
        pool.set_telemetry_layer(0);
        std::optional<v100_compare::Round> round;
        if (v100_compare::level())
            round.emplace("strata", "pool:" + std::to_string(workers) + ":" + std::to_string(host), "verify");
        std::vector<std::vector<float>> output(4, std::vector<float>(cpu::H));
        std::vector<cpu::ExpertJob> jobs(4);
        for (int e = 0; e < 4; ++e) {
            jobs[e].blob = blobs[e].data(); jobs[e].act = &act; jobs[e].out = output[e].data();
        }
        for (int repeat = 0; repeat < 4; ++repeat) {
            pool.run(jobs.data(), 4);
            for (int e = 0; e < 4; ++e)
                if (std::memcmp(expected[e].data(), output[e].data(), cpu::H * sizeof(float)))
                    throw std::runtime_error("whole pool output differs from serial");
            // Wake sleepers into tiny batches to exercise the epoch/park boundary.
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::vector<cpu::ExpertJobMulti> multi(2);
        std::vector<std::vector<float>> rows(4, std::vector<float>(cpu::H));
        for (int e = 0; e < 2; ++e) {
            multi[e].blob = blobs[e].data(); multi[e].nt = 2;
            for (int token = 0; token < 2; ++token) {
                multi[e].act[token] = &act; multi[e].out[token] = rows[2 * e + token].data();
            }
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            pool.run_split_multi(multi.data(), 2);
            for (int e = 0; e < 2; ++e) for (int token = 0; token < 2; ++token)
                if (std::memcmp(expected[e].data(), rows[2 * e + token].data(), cpu::H * sizeof(float)))
                    throw std::runtime_error("multi-phase output differs from serial");
        }
    }
}
