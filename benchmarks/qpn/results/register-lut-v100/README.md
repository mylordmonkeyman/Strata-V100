# Register-LUT decoder — numerical pass, single-expert performance gate fails

Source: `be323b70f656cf4359e281c2e1540eb42dc896f6`.
[Run](https://github.com/mylordmonkeyman/Strata-V100/actions/runs/40802984333).
Decode artifact was uploaded before the independent prefill phase.
FP4-Local checkpoint revision from the fixture audits:
`b9ce18a6887e18c089fdea8509a33e63429a3300`.

All 120 real-weight decoder/staging/T cases passed provisional numerical,
decoder-pair probe, graph replay and output-guard checks. Synthetic cases also
passed. Inputs remain Gaussian, not real captured model activations. Fixture
hashes match the previous run. The register-LUT decoder plus vector activation
loads improves QPN itself by about 2–3×, but still loses to the existing grouped
kernel for one expert. Median legacy/QPN ratios over the three fixtures with
FP32 input/hidden staging:

| T | Scalar decoder 0 | Register-LUT decoder 1 |
|---|---:|---:|
| 1 | 0.137 | 0.278 |
| 2 | 0.148 | 0.303 |
| 3 | 0.154 | 0.321 |
| 4 | 0.178 | 0.377 |
| 8 | 0.225 | 0.534 |
| 16 | 0.243 | 0.863 |
| 32 | 0.289 | 0.841 |

These are completed-expert CUDA-event timings, not engine speed/TTFT results.
No production integration or enablement is justified. Gate/up N=640 means
only 20 QPN CTAs per expert; the next uniform multi-expert experiment tests
the underfilled-grid hypothesis with distinct GPU addresses and >L2 working
sets. Repeated identical expert values do not simulate routed-group histograms
or differing scales, so that experiment still cannot qualify engine dispatch.

Raw timings, binary hash and model/fixture provenance are committed here.
Full sanitizer/build/telemetry logs remain in the Actions artifact.
