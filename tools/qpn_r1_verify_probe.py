#!/usr/bin/env python3
"""One pinned MMQ-only private verification route probe, default-off QDV1.

Two complete original-MMQ runs compare first logits with trace disabled/enabled.
Raw prompts, logits, output text, router counts and logs remain in a private
0700 temporary directory. Only aggregate counts and hashes leave the runner.
This is NOT heldout R1, QPN decode execution, or model performance evidence.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from qpn_full_model_ab import benchmark_env, sha, logits, diff, read_summary
from qpn_r1_verify_convert import decode_private_verify

MODEL = "1e124ee94b7c921a7c895d2898bfa4ddc2d8111f3fd3e8b94207759c363291a5"
PROMPT = "df1ffef8677cccf2b8b8af36df04387edb747a186600e9801909b6f99fb6f414"
PREFILL_TOKENS = 2047


def aggregate_verify(sample):
    windows = sample["verify"]
    if not windows or sample["decode"] or len(windows) > 32:
        raise ValueError("missing or misclassified private verification windows")
    result = dict(verification_windows=len(windows), post_prefill_verification_windows=0,
                  verification_layer_records=0, verification_routed_entries=0,
                  verification_active_expert_uses=0, verification_distinct_layer_experts=0,
                  max_verification_window_tokens=0, max_per_expert_rows=0,
                  qpn_33_512_row_groups=0)
    distinct = set()
    for w in windows:
        if len(w["layers"]) != 48 or not 1 <= w["tokens"] <= 8:
            raise ValueError("incomplete private verification window")
        result["max_verification_window_tokens"] = max(result["max_verification_window_tokens"], w["tokens"])
        if w["position"] >= PREFILL_TOKENS:
            result["post_prefill_verification_windows"] += 1
        for layer in w["layers"]:
            rows = layer["counts"]
            if sum(c for _, c in rows) != w["tokens"] * 10:
                raise ValueError("invalid private verification row count")
            result["verification_layer_records"] += 1
            result["verification_routed_entries"] += sum(c for _, c in rows)
            result["verification_active_expert_uses"] += len(rows)
            for e, count in rows:
                result["max_per_expert_rows"] = max(result["max_per_expert_rows"], count)
                distinct.add((layer["layer"], e))
                if count >= 33:
                    raise ValueError("unexpected >=33 rows in native verification window")
    result["verification_distinct_layer_experts"] = len(distinct)
    if result["post_prefill_verification_windows"] == 0:
        raise ValueError("no post-prefill verification routes were captured")
    return result


def run(a):
    state, prepared = a.state, a.prepared
    tok, pack = state / "prompt-2048.txt", prepared / "pack/experts.bin"
    for f in (tok, pack, a.binary, prepared / "dense.gguf", prepared / "ple-nvfp4.gguf"):
        if not f.is_file():
            raise ValueError("missing pinned private model, prompt or executable")
    if sha(tok) != PROMPT or sha(pack) != MODEL:
        raise ValueError("pinned private model/prompt SHA mismatch")
    env_base = benchmark_env()
    env_base.update(STRATA_PREFILL_NVFP4="w4a8", STRATA_PREFILL_CPU_SHARE="0",
                    STRATA_QPN_PREFILL_EXPERIMENTAL="0",
                    STRATA_PREFILL_STREAM_MIN="4096", STRATA_PREFILL_GROUP_GATHER="1")
    command = [str(a.binary), "--pack", str(prepared / "pack"),
               "--native", str(prepared / "dense.gguf"),
               "--native-dense-gguf", str(prepared / "dense.gguf"),
               "--ple-gguf", str(prepared / "ple-nvfp4.gguf"),
               "--tokens-file", str(tok), "--max-context", "4096",
               "--max-new", "16", "--greedy", "--expert-cache", "auto",
               "--expert-profile", str(a.source / "data/expert-profile.bin"),
               "--resident-budget-gib", "24", "--pool-workers", "16",
               "--prefill", "1024", "--spec", "2", "--suffix-draft", "0",
               "--adapt-every", "0", "--vram-reserve-mib", "1536", "--stats"]
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qpn-private-verify-", dir=state) as d:
        os.chmod(d, 0o700)
        private = Path(d)
        logit_hashes = []
        capture_path = private / "routes.qdv1"
        for trace in (False, True):
            logfile = private / ("trace.log" if trace else "control.log")
            logit_path = private / ("trace.f32" if trace else "control.f32")
            env = env_base.copy()
            env["STRATA_DUMP_FIRST_LOGITS"] = str(logit_path)
            if trace:
                env["STRATA_QPN_R1_PRIVATE_VERIFY_CAPTURE"] = str(capture_path)
            with logfile.open("w") as f:
                p = subprocess.run(["numactl", "--cpunodebind=0", "--membind=0", *command],
                                   env=env, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=900, check=False)
            if p.returncode:
                raise RuntimeError("pinned MMQ verification arm failed; private logs destroyed on exit")
            raw = logfile.read_text(errors="replace")
            if re.search(r"non[- ]finite|illegal memory access|CUDA error|device-side assert", raw, re.I):
                raise RuntimeError("CUDA/nonfinite failure in private MMQ verification arm")
            summary = read_summary(logfile)
            if summary["prefill_tokens"] != PREFILL_TOKENS or summary["qpn_groups"] != 0:
                raise RuntimeError("pinned MMQ work mismatch or unexpected QPN dispatch")
            logit_hashes.append((sha(logit_path), logits(logit_path)))
        if logit_hashes[0][0] != logit_hashes[1][0]:
            raise RuntimeError("private verification trace changed first-logit bytes")
        cmp = diff(logit_hashes[0][1], logit_hashes[1][1])
        if cmp["relative_l2"] != 0 or cmp["kl_base_vs_candidate"] != 0:
            raise RuntimeError("private verification trace changed first-logit values")
        if not capture_path.is_file():
            raise RuntimeError("no private verification capture produced")
        sample = decode_private_verify(capture_path)
        metrics = aggregate_verify(sample)
        output = dict(schema=1,
            scope="single pinned MMQ-only post-prefill verification route capture; NOT heldout R1",
            source_commit=subprocess.check_output(["git", "-C", str(a.source), "rev-parse", "HEAD"],
                                                   text=True).strip(),
            model_sha256=MODEL, prompt_sha256=PROMPT, binary_sha256=sha(a.binary),
            private_capture_sha256=sample["private_capture_sha256"],
            control_flags={k: env_base[k] for k in
                           ("STRATA_MMQ_KQ", "STRATA_PREFILL_MMQ",
                            "STRATA_ARENA_LOCK", "STRATA_ARENA_PIN_GIB",
                            "STRATA_QPN_PREFILL_EXPERIMENTAL")},
            capture=metrics, first_logit_equal=True,
            first_logit_compared_elements=cmp["n_logits"],
            first_logit_relative_l2=cmp["relative_l2"],
            first_logit_kl=cmp["kl_base_vs_candidate"],
            qpn_groups=0, actual_native_verification_routes_collected=True,
            actual_qpn_decode_executed=False, heldout_prompts_collected=False,
            performance_qualified=False, model_quality_qualified=False,
            limitations=[
                "verification routes are genuine GPU ids_ snapshots; not CPU-miss doorbell h_ids_",
                "positions after prefill are verification computations, not necessarily accepted token routes",
                "capture only first 32 Verifier::run windows, excluding batch/pipeline",
                "one pinned prompt, no independent heldout train/test",
                "trace adds D2H graph copies; timing is NOT a performance comparison",
                "first-logit parity only; generated-token and model-quality parity not established",
                "no packed cache, layout churn, decode restoration or QPN dispatch implemented"])
        a.report.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "prepared", "state", "binary", "report"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    run(a)
    print("Private verification capture parity passed; aggregate-only report written")


if __name__ == "__main__":
    main()
