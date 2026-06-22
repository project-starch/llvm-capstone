#!/usr/bin/env bash
set -euo pipefail

# Serial aggregate gate for all validated BEEBS wrappers. Individual
# run-beebs-*.sh scripts remain the authoritative diagnostic entry points.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

mapfile -t BEEBS_RUNNERS < <(
  find "$SCRIPT_DIR" -maxdepth 1 -name 'run-beebs-*.sh' \
    ! -name 'run-beebs-simple-common.sh' \
    | sort
)

if [[ ${#BEEBS_RUNNERS[@]} -eq 0 ]]; then
  echo "run-all-beebs.sh: no BEEBS runners found" >&2
  exit 1
fi

for runner in "${BEEBS_RUNNERS[@]}"; do
  echo "run-all-beebs.sh: running $runner"
  bash "$runner"
done

echo "run-all-beebs.sh: all BEEBS runners completed"
