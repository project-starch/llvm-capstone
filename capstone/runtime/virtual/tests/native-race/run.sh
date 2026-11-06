#!/usr/bin/env bash
# Copyable-only reproduction: shared staged runner, followed by the frozen RT oracle.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
exec python3 "$here/../native-integration/run-copyable.py" race "$@"
