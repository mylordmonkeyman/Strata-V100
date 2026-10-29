FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git curl ca-certificates \
      python3 python3-venv numactl util-linux tini \
      libicu74 libssl3t64 zlib1g libkrb5-3 libunwind8 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 runner \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash runner \
    && mkdir -p /opt/runner-template /runner /srv/strata-qpn /srv/strata-qpn/fixtures \
    && curl -fsSL --retry 3 \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
      -o /tmp/runner.tar.gz \
    && echo "${RUNNER_SHA256}  /tmp/runner.tar.gz" | sha256sum --check - \
    && tar --no-same-owner -xzf /tmp/runner.tar.gz -C /opt/runner-template \
    && rm /tmp/runner.tar.gz \
    && chown -R runner:runner /runner /srv/strata-qpn /opt/runner-template
COPY --chmod=0755 entrypoint.sh /usr/local/bin/start-qpn-runner
ENV PATH="/usr/local/cuda-12.8/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}" \
    QPN_CUDA_ROOT=/usr/local/cuda-12.8
USER 10001:10001
WORKDIR /runner
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/start-qpn-runner"]
