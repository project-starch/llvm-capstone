#!/usr/bin/env bash
# Build the sole application runtime; callers own source fetching and compilation.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
[[ $# -ge 3 ]] || { echo "usage: build-sdk.sh OUT MUSL_SOURCE LIBC_ARCHIVE [CMAKE_OPTION...]" >&2; exit 2; }
SDK=$1 MUSL=$2 LIBC=$3
shift 3
PROFILE=${CAPSTONE_APPLICATION_PROFILE:-physical}
case "$PROFILE" in
  physical) VIRTUAL=OFF ;;
  virtual) VIRTUAL=ON ;;
  *) echo "invalid CAPSTONE_APPLICATION_PROFILE: $PROFILE" >&2; exit 2 ;;
esac
cmake --fresh -S "$CAPSTONE_REPO_ROOT/capstone/runtime/application" -B "$SDK" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$CAPSTONE_REPO_ROOT/capstone/ports/common/cmake/toolchains/capstone-domain.cmake" \
  -DCAPSTONE_LLVM_BUILD_DIR="$CAPSTONE_LLVM_BUILD_DIR" \
  -DPORT_HEADER_PROVIDER=musl -DPORT_C11_ATOMICS=ON -DPORT_MUSL_ROOT="$MUSL" \
  -DCAPSTONE_MUSL_ARCHIVE="$LIBC" -DCAPSTONE_APPLICATION_SDK=ON \
  -DCAPSTONE_APPLICATION_VIRTUAL="$VIRTUAL" \
  -DCAPSTONE_APPLICATION_DATA_BYTES=33554432 -DCAPSTONE_APPLICATION_ARENA_BYTES=67108864 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE=-O1 "$@"
cmake --build "$SDK" -j"${JOBS:-8}"
# CAPSTONE_SDK_EXTRA_CODEGEN appends words to the SDK's sdk.json codegen after the build, so every
# later `capstone-cc` compile carries them and the runtime archives are untouched. It is how the vp2
# Tier 3 campaign's t3-freeze-nb SDK got -fno-builtin (the C-81 axis) by hand: that SDK's sdk.json is
# its only file newer than its build. Unset, the SDK is unchanged. capstone-cc still validates the
# result against the profile, so an FP, ABI or gp option here is refused at the first compile.
if [[ -n ${CAPSTONE_SDK_EXTRA_CODEGEN:-} ]]; then
  python3 - "$SDK/sdk.json" $CAPSTONE_SDK_EXTRA_CODEGEN <<'APPEND'
import json, sys
path, extra = sys.argv[1], sys.argv[2:]
config = json.load(open(path))
config["codegen"] = list(config["codegen"]) + [w for w in extra if w not in config["codegen"]]
json.dump(config, open(path, "w"), indent=2)
APPEND
fi
