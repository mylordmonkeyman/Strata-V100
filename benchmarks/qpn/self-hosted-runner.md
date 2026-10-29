# Set up the local V100 GitHub Actions runner

**Docker is the preferred setup for this project:** use the
[Dockerfile, Compose configuration and startup commands](../../docker/qpn-runner/README.md).
The remaining instructions on this page describe the alternative native service.

Target: your Ubuntu 24.04 server, Tesla V100-PCIe-32GB on GPU 0, CUDA 12.8,
dual Xeon E5-2697A v4. The M2000 is excluded. Benchmarks bind to NUMA node 0.

Files:

- Workflow: [`.github/workflows/qpn-v100.yml`](../../.github/workflows/qpn-v100.yml)
- Job script: [`tools/qpn_v100_ci.sh`](../../tools/qpn_v100_ci.sh)
- Kernel/tooling details: [README](README.md)

The workflow builds the standalone QPN tools, runs host tests, synthetic and
real-expert parity/timings, CUDA Graph replay, and optional Compute Sanitizer.
It uploads logs, hashes, JSON measurements and telemetry. It does not upload
weights/activation files, install software during a job, stop services, kill
model processes, or change engine defaults. A green synthetic-only run does
not establish real-model qualification. Mixed-group dispatch, cache/PCIe
profiling, model KL/acceptance and actual prefill TTFT remain separate work.

## 1. Prepare software as your normal administrator account

Run on the V100 server, not your Windows desktop:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git curl ca-certificates python3 python3-venv numactl util-linux
nvidia-smi --id=0
/usr/local/cuda-12.8/bin/nvcc --version
/usr/local/cuda-12.8/bin/compute-sanitizer --version
cmake --version
numactl --hardware
```

Requirements: CMake >=3.24, CUDA **12.8**, a working V100 driver and toolkit
headers/libraries including cuBLAS, and CUDA 12.8 Compute Sanitizer. Your driver
may show "CUDA Version: 13.0" in `nvidia-smi`; the **compiler/toolkit path**
must still be 12.8. This workflow explicitly checks `nvcc`'s release.
Do not replace your working driver. If CUDA 12.8 lives elsewhere, set the
`QPN_CUDA_ROOT` repository variable in step 4.

No Docker installation is needed for this native workflow. Do not add the
runner account to the Docker group or give it sudo access. Allow outbound
HTTPS/443 to the GitHub Actions domains, GitHub release/checkout and artifact
services; no incoming port, public SSH exposure, or Tailscale change is needed.
The full domain list is in the GitHub reference linked at the end.

## 2. Create a dedicated unprivileged account and directories

If the account already exists, skip `useradd`:

```bash
sudo useradd --create-home --shell /bin/bash strata-runner
sudo install -d -o strata-runner -g strata-runner -m 0750 /opt/actions-runner-strata
sudo install -d -o strata-runner -g strata-runner -m 0750 /srv/strata-qpn
sudo install -d -o strata-runner -g strata-runner -m 0750 /srv/strata-qpn/fixtures
for gpu_group in video render; do
  if getent group "$gpu_group" >/dev/null; then
    sudo usermod -aG "$gpu_group" strata-runner
  fi
done
sudo -u strata-runner nvidia-smi --id=0
sudo -u strata-runner numactl --cpunodebind=0 --membind=0 true
df -h /opt/actions-runner-strata /srv/strata-qpn
```

Reserve approximately 20 GB of free space for runner software, source,
ggml fetch, build files and logs, plus the small fixtures. This is a planning
allowance, not a measured build size. The full model pack remains elsewhere
and does not need to be readable by this account. Your regular home directory,
SSH keys and service configuration should not be shared with the runner.

## 3. Prepare canonical NVFP4 fixtures before starting the service

Use your existing **FP4-Local NVFP4-QAD pack**, not the UD-Q4 GGUF you previously
benchmarked. If this pack has not been converted yet, prepare it using the
unchanged `tools/nvfp4_local_pack.py` pipeline first (see `docs/NVFP4.md`). No
full model download or conversion runs automatically in CI.

As your normal account, make a separate setup checkout:

```bash
mkdir -p "$HOME/ai"
git clone --branch qpn/direct-prototype https://github.com/mylordmonkeyman/Strata-V100.git "$HOME/ai/strata-qpn-local"
cd "$HOME/ai/strata-qpn-local"
QPN_PACK=/absolute/path/to/your/nvfp4-qad-pack
QPN_TEMP_FIXTURES=$(mktemp -d /tmp/strata-qpn-fixtures.XXXXXX)
for selection in 0:7 24:300 47:511; do
  qpn_layer=${selection%:*}
  qpn_expert=${selection#*:}
  python3 tools/qpn_audit.py "$QPN_PACK" --layer "$qpn_layer" --expert "$qpn_expert" \
    --extract "$QPN_TEMP_FIXTURES/l${qpn_layer}-e${qpn_expert}.bin" \
    > "$QPN_TEMP_FIXTURES/l${qpn_layer}-e${qpn_expert}-audit.json" || break
done
```

Replace `QPN_PACK` before running. These picks assume the expected 48-layer,
512-expert model; if your artifact differs, select valid low/middle/high layer
indices. Read the audit JSON and resolve any error before copying files. The
auditor checks blob sizes/types/tails and length; separately verify 2560/640
geometry, model identity and routing top-k in the source checkpoint/GGUF.

Once all selected audits succeeded, copy the fixtures:

```bash
sudo install -o strata-runner -g strata-runner -m 0640 "$QPN_TEMP_FIXTURES"/*.bin /srv/strata-qpn/fixtures/
sudo install -o strata-runner -g strata-runner -m 0640 "$QPN_TEMP_FIXTURES"/*-audit.json /srv/strata-qpn/fixtures/
sudo -u strata-runner find /srv/strata-qpn/fixtures -maxdepth 1 -type f
```

At least one `.bin` is needed for real-expert tests. No fixtures means the job
logs a skip and runs synthetic-only tests; it cannot qualify real weights.
Optionally install `activations.f32` in the same directory: little-endian raw
FP32 normalized expert-input rows, width 2560, at least 32 rows. That enables
an additional real-activation matrix. A Gaussian input run is labeled as such.
Do not commit these `.bin` or activation files to GitHub.

## 4. Configure the GitHub repository

1. Open [repository settings](https://github.com/mylordmonkeyman/Strata-V100/settings/actions).
   Enable Actions and permit the `actions/checkout` and `actions/upload-artifact`
   actions if you restrict allowed actions. Keep workflow token permissions read-only.
2. Under **Settings → Secrets and variables → Actions → Variables**, optionally
   add the following repository **variables**, not secrets:

| Variable | Default | When to change |
|---|---|---|
| `QPN_CUDA_ROOT` | `/usr/local/cuda-12.8` | Your CUDA 12.8 toolkit lives elsewhere |
| `QPN_GPU` | `0` | Prefer the V100's `GPU-...` UUID if GPU order can change |
| `QPN_FIXTURE_DIR` | `/srv/strata-qpn/fixtures` | You chose another runner-readable fixture directory |

Get the UUID with:

```bash
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

No Hugging Face token, PAT or SSH private key is needed in workflow secrets.
The runner registration token in step 5 is used only when registering the
service, never pasted into the workflow YAML or committed.

## 5. Download and register the runner

In GitHub, open **Settings → Actions → Runners → New self-hosted runner** for
**mylordmonkeyman/Strata-V100**. Choose **Linux / x64**. Use the current download,
checksum and extraction commands shown there; do not reuse an old ninfer
runner installation or hardcode a stale release version.

On the server, enter the dedicated account first:

```bash
sudo -iu strata-runner
cd /opt/actions-runner-strata
```

Run GitHub's **download, checksum verification and extraction** commands here.
Do not create another nested `actions-runner` directory if the UI example has
a `mkdir`/`cd` line; you are already in the intended installation directory.
After extraction, replace GitHub's config command with this one, entering
the short-lived registration token privately when prompted:

```bash
read -r -s -p 'GitHub runner registration token: ' QPN_REGISTRATION_TOKEN
printf '\n'
./config.sh --url https://github.com/mylordmonkeyman/Strata-V100 \
  --token "$QPN_REGISTRATION_TOKEN" --name strata-v100-qpn \
  --labels v100-sm70,strata-qpn --work _work --unattended
unset QPN_REGISTRATION_TOKEN
exit
```

Keep the automatic `self-hosted`, `linux`, and `x64` labels. The two custom
labels must be **exactly** `v100-sm70` and `strata-qpn`. If configuration reports
missing OS dependencies, run the extracted runner's
`sudo /opt/actions-runner-strata/bin/installdependencies.sh` as your administrator,
then retry registration as `strata-runner` with a fresh token if needed.

Registration tokens expire after one hour. Do not configure or run the runner
as root. Do not use a second runner service pointed at this same GPU.

## 6. Make the V100 idle, then install/start the service

Inspect compute clients:

```bash
nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv
```

Finish any benchmark or generation already running. If `llama-swap` owns the
loaded model, stop it during the CI session:

```bash
sudo systemctl stop llama-swap
```

Other independently launched GPU processes may need their own normal shutdown.
The workflow refuses a busy V100; it does not kill them. Keep the GPU reserved
for CI throughout the run, including against new manual generation requests.

As your normal administrator account:

```bash
cd /opt/actions-runner-strata
sudo ./svc.sh install strata-runner
sudo ./svc.sh start
sudo ./svc.sh status
```

The repository's runner page should show **Idle** before a job, then **Active**.
The runner service starts on reboot. Its checkout is under
`/opt/actions-runner-strata/_work`; GitHub cleans that checkout between jobs.
Keep all fixtures and the full model outside it.

## 7. Run the workflow and obtain results

Open [Actions](https://github.com/mylordmonkeyman/Strata-V100/actions) and select
**QPN V100 local runner**. The push that added this workflow creates an initial
queued run on `qpn/direct-prototype`; once registered, the matching runner can
pick it up. If it expired or failed before setup completed, open that run and
choose **Re-run all jobs**. Later pushes to this exact branch create new runs.
Only the `mylordmonkeyman` actor is allowed by this workflow's job condition.

There is a GitHub bootstrap limitation: **Run workflow** / `workflow_dispatch`
requires the YAML to exist on the default branch. While `main` is the untouched
upstream mirror, use the **push run/re-run** route above. Do not change `main`
or merge the prototype just to get a button. If you later intentionally adopt
this workflow on the default branch, manual dispatch becomes available and
lets you enable the full `sweep` or turn sanitizers off for timing-only runs.
The workflow checks out the trusted event commit, not an arbitrary input ref.

Download the artifact named `qpn-v100-<run_id>-<attempt>`. It contains:

- `job.log`, `configure.log`, `build.log`, host/audit test output.
- `synthetic/`, and `real/` / `real-activations/` when fixtures are present:
  raw per-candidate JSON, summaries, binary/fixture/activation hashes, commands,
  return codes, affinity and GPU state.
- Per-tool sanitizer logs at T=3, 8, 17; `exit-codes.tsv` for test failures.
- `telemetry.csv`, driver/device details, NUMA layout, compiler and commit ID.

Numerical/sanitizer failures make the job fail, while independent candidate
processes still finish and preserve their logs. Configure/build/preflight
failure stops the job; the artifact step still runs. Artifacts are retained
for 14 days. Download and review measurement logs before committing selected
data under `benchmarks/qpn/`; this workflow does not auto-commit results.

## 8. Full sweep before default-branch adoption (optional)

Keep the runner service stopped while doing a manual run, so a queued job
cannot start simultaneously. Use its own checkout, not your production engine:

```bash
cd /opt/actions-runner-strata
sudo ./svc.sh stop
sudo -iu strata-runner
git clone --branch qpn/direct-prototype https://github.com/mylordmonkeyman/Strata-V100.git "$HOME/strata-qpn-manual"
cd "$HOME/strata-qpn-manual"
QPN_SWEEP=1 QPN_SANITIZERS=1 bash tools/qpn_v100_ci.sh
```

For a later manual rerun, archive the existing `qpn-results` directory first;
the script refuses to mix logs. The shared `/srv/strata-qpn/gpu.lock` also
prevents two invocations of this job script from overlapping. It does not
lock out other inference software; keep those GPU clients stopped yourself.
Copy any desired results before deleting a manual checkout. Exit the account
and restart the service when ready to accept further CI jobs.

## 9. Routine operation and troubleshooting

- **Job queued indefinitely:** check `svc.sh status`, all five labels, the
  exact repository registration, actor condition and outbound connectivity.
- **Busy GPU:** stop its owning workload and re-run. No auto-kill is performed.
- **CUDA mismatch:** inspect `nvcc.txt`; point `QPN_CUDA_ROOT` at the full 12.8
  toolkit. Do not use a CUDA 13 Compute Sanitizer for Volta.
- **No real tests:** check `QPN_FIXTURE_DIR`, `.bin` files and permissions.
- **Build fails:** inspect uploaded configure/build logs; the full native CUDA
  harness build has not yet been verified on this server by the setup author.
- **Permission error on NUMA/GPU:** verify `sudo -u strata-runner nvidia-smi`
  and `numactl` from step 2. Restart the service after group changes.
- **Runner connection errors:** see `_diag` in the runner installation directory;
  keep runner diagnostics private because they may contain operational details.

To take the runner offline and restore inference after your CI session:

```bash
cd /opt/actions-runner-strata
sudo ./svc.sh stop
sudo systemctl start llama-swap
```

Do not start inference while leaving queued GPU jobs able to take the same V100.
To remove the runner, stop/uninstall its service with `svc.sh`, then use the
repository runner page's removal instructions and fresh removal token.

## Public repository boundary and official references

This is a persistent machine with access to your local hardware. GitHub warns
against public-repository self-hosted runners because untrusted PR code can
compromise the host. This workflow therefore has **no PR trigger**, restricts
the repository/actor, and uses a dedicated account without sudo/Docker rights.
Those are mitigations, not a sandbox: only push code you trust to this branch.
Do not add self-hosted jobs to the existing public PR workflow. Broader testing
of outside contributions should use a disposable, isolated runner instead.

- [GitHub runner registration](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [Linux service setup](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application)
- [Runner labels](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels)
- [Communication requirements](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [Manual dispatch/default branch requirement](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
- [Self-hosted runner security](https://docs.github.com/en/actions/reference/security/secure-use)
