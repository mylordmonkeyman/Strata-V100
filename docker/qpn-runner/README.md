# Docker V100 QPN runner

This replaces the native systemd runner setup. The existing GitHub workflow
uses the same `self-hosted,linux,x64,v100-sm70,strata-qpn` labels, so no workflow
change is required. Only trusted feature-branch pushes use this persistent host.

The image includes CUDA **12.8.1**, GCC, CMake, Ninja, Python, NUMA tools and
Compute Sanitizer. Runner v2.338.2 is checksum-verified at build time and can
auto-update in its persistent data volume. Container UID/GID is **10001**.
No host CUDA toolkit mount, Docker socket, host networking, privileged mode,
sudo in the container, or incoming ports are required.

## 1. Get the configuration

On your Ubuntu server as your normal account:

```bash
mkdir -p "$HOME/ai"
git clone --branch qpn/direct-prototype https://github.com/mylordmonkeyman/Strata-V100.git "$HOME/ai/strata-qpn-docker"
cd "$HOME/ai/strata-qpn-docker/docker/qpn-runner"
docker compose version
nvidia-smi --query-gpu=index,name,uuid --format=csv
```

If you already have this checkout, update its feature branch instead of
cloning into an existing directory. Commands below assume Docker Compose v2+
and NVIDIA Container Toolkit already work, as with your previous GPU runner.
If Docker needs elevated access on your host, prefix Docker commands with
`sudo`; `.env` still supplies the UUID when run from this directory.

If the previous **native QPN runner** was installed, stop it first:

```bash
if [ -x /opt/actions-runner-strata/svc.sh ]; then
  sudo /opt/actions-runner-strata/svc.sh stop
fi
```

Also stop any older Docker GPU runner using this same V100. Do not run two
runner agents against it simultaneously.

## 2. Select the V100 and check GPU passthrough

Your V100 is host GPU 0; select its UUID to avoid exposing the M2000:

```bash
QPN_V100_UUID=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
printf 'QPN_GPU_UUID=%s\n' "$QPN_V100_UUID" > .env
docker run --rm --gpus "device=$QPN_V100_UUID" nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If passthrough fails, install/configure NVIDIA Container Toolkit using
[NVIDIA's instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
On an already installed toolkit, `sudo nvidia-ctk runtime configure --runtime=docker`
configures Docker, followed by `sudo systemctl restart docker`. Restarting Docker
can affect your Open WebUI/game/database containers; schedule that only if the
passthrough test fails and the runtime needs reconfiguration. Do not replace
your working GPU driver.

The Compose file pins CPUs `0-15,32-47` (your NUMA node 0), with a 48 GiB RAM
limit and 8 GiB shared memory. Seccomp is unconfined specifically because the
job calls NUMA memory-policy syscalls; all Linux capabilities are dropped and
the process remains non-root. This is not a sandbox for untrusted workflows.

## 3. Fixtures and registration token

Prepare your canonical expert `.bin` fixtures as described in
[the fixture extraction guide](../../benchmarks/qpn/self-hosted-runner.md#3-prepare-canonical-nvfp4-fixtures-before-starting-the-service).
They reside on the host at `/srv/strata-qpn/fixtures` and are mounted read-only.
An empty directory is sufficient for synthetic-only smoke testing.

```bash
sudo apt-get install -y acl
sudo install -d -m 0755 /srv/strata-qpn
sudo mkdir -p /srv/strata-qpn/fixtures
sudo setfacl -R -m u:10001:rX /srv/strata-qpn/fixtures
```

Reapply the ACL when adding fixture files; optional real activations are named
`activations.f32` and also need read access. Do not mount the entire model pack.
No Hugging Face token or GitHub PAT is needed.

Open repository **Settings → Actions → Runners → New self-hosted runner →
Linux/x64**. Copy the **registration token** from its `./config.sh` command.
Skip the UI's software download instructions because the image handles those.
The token expires after one hour, so obtain it shortly before first startup.

Save it privately on the host without putting it into shell history or the image:

```bash
sudo -v
read -r -s -p 'GitHub runner registration token: ' QPN_REGISTRATION_TOKEN
printf '\n'
printf '%s' "$QPN_REGISTRATION_TOKEN" | sudo install -o 10001 -g 10001 -m 0600 /dev/stdin /srv/strata-qpn/registration-token
unset QPN_REGISTRATION_TOKEN
```

In repository Actions **variables**, use the default container paths:
`QPN_CUDA_ROOT=/usr/local/cuda-12.8`,
`QPN_FIXTURE_DIR=/srv/strata-qpn/fixtures`. Leave `QPN_GPU` unset (defaults to 0
on your current host) or set it to the V100 UUID. Remove any previous variable
values pointing to host-only paths. These are variables, not secrets.

## 4. Build, preflight, start

Build first (it does not need an idle GPU), then check the container environment:

```bash
docker compose build
docker compose run --rm --no-deps --entrypoint bash runner -lc 'id; nvcc --version; compute-sanitizer --version; nvidia-smi; numactl --cpunodebind=0 --membind=0 true'
```

Finish current GPU work. If llama-swap owns the loaded model:

```bash
sudo systemctl stop llama-swap
nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv
docker compose up -d
docker compose logs -f --tail=100
```

Wait for `Listening for Jobs`. The GitHub runner is named
**strata-v100-qpn-docker** with the workflow's matching labels. The queued
[QPN V100 local runner workflow](https://github.com/mylordmonkeyman/Strata-V100/actions)
can now start. Do not enable the public PR workflow on this GPU runner.

After successful registration, clear the short-lived bootstrap token:

```bash
sudo truncate -s 0 /srv/strata-qpn/registration-token
```

Keep the empty file present because Compose mounts it. Restart uses the saved
runner credentials, so it does not need a new registration token. The named
`runner-data` volume contains these credentials and must stay private.

## 5. Operation

```bash
docker compose ps
docker compose logs --tail=100
docker compose stop
docker compose start
```

Use `stop` only when idle or after cancelling/finishing a job. `restart:
unless-stopped` brings the container back after Docker/host restarts unless
you stopped it. GPU jobs still require an idle V100; this setup does not stop
inference software automatically.

The GitHub job uploads logs/measurements as artifacts. To inspect the current
checkout inside the container:

```bash
docker compose exec runner bash
```

The repository checkout/results live beneath `/runner/_work`. Fixtures are
not uploaded. `docker compose down` retains registration/work volumes;
**do not use `down -v`** unless intentionally deleting runner data. Replacing
those volumes requires removing the old GitHub runner registration and obtaining
a fresh registration token before starting again.

To return the GPU to your normal inference service, stop the runner first:

```bash
docker compose stop
sudo systemctl start llama-swap
```

This configuration is source/lint checked; Docker/GPU execution must be verified
on your host with the preflight command above. It does not itself qualify the
QPN kernel or integrate it into the engine.
