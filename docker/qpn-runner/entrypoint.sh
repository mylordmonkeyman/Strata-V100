#!/usr/bin/env bash
set -euo pipefail
cd /runner
# The Docker named volume preserves registration, auto-updated runner binaries,
# diagnostics and the work directory. First start seeds an empty volume.
if [[ ! -x ./config.sh ]]; then cp -a /opt/runner-template/. /runner/; fi
if [[ ! -f .runner ]]; then
  token_file=/run/secrets/registration_token
  if [[ ! -r "$token_file" || ! -s "$token_file" ]]; then
    echo 'First start needs a readable, fresh repository runner registration token.' >&2
    exit 2
  fi
  registration_token=$(< "$token_file")
  ./config.sh --unattended \
    --url "${RUNNER_REPOSITORY_URL:-https://github.com/mylordmonkeyman/Strata-V100}" \
    --token "$registration_token" \
    --name "${RUNNER_NAME:-strata-v100-qpn-docker}" \
    --labels v100-sm70,strata-qpn --work _work
  unset registration_token
fi
exec ./run.sh
