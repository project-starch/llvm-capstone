#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
peer=${1:?page lifecycle source tree or snapshot}
out=${2:?output directory}
mkdir -p "$out"
sha256sum "$peer/capstone/runtime/virtual/unpinned/lifecycle/page_double.py" > "$out/peer.sha256"
PYTHONDONTWRITEBYTECODE=1 python3 "$here/tag-page-probe.py" "$peer" | tee "$out/probe.log"
