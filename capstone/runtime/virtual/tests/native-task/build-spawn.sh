#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../../tests/capstone-test-env.sh"
out=${1:?fresh output directory}
: "${CAPSTONE_NATIVE_CC:?built native SDK compiler}"
mkdir -p "$out/a"
"$CAPSTONE_NATIVE_CC" -O2 "$here/spawn.c" -o "$out/spawn.dom"
printf '#!/bin/sh\necho bad > script-ran\n' > "$out/script"
chmod +x "$out/script"
ln -s ../outside.dom "$out/escape"
sha256sum "$here/spawn.c" "$out/spawn.dom" > "$out/spawn-inputs.sha256"
# Guest gate must copy a scalar ELF (for example /bin/busybox) to ./scalar.
