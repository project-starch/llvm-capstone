#!/usr/bin/env bash
# Build the sole application runtime; callers own source fetching and compilation.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
[[ $# -ge 3 ]] || { echo "usage: build-sdk.sh OUT MUSL_SOURCE LIBC_ARCHIVE [CMAKE_OPTION...]" >&2; exit 2; }
SDK=$1 MUSL=$2 LIBC=$3
shift 3
PROFILE=${CAPSTONE_APPLICATION_PROFILE:-virtual}
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
# CORPUS SDKs (CAPSTONE_SDK_CORPUS=1): the C-81 rule as a GATE, not a habit. A bug-corpus or kit image compiles with
# -fno-builtin. Without it LLVM treats malloc/realloc/free as library builtins and may elide an allocation whose
# result is only tested against NULL, assuming it succeeds (upstream LLVM semantics). A test that observes an
# allocation FAILURE then reads nothing: the v6 kit's malloc contract compiled its 512 MiB growth check to an
# unconditional failure, read as a runtime regression and retracted (2026-10-20). With CAPSTONE_SDK_CORPUS=1 this
# builder APPENDS -fno-builtin itself (the same post-build edit as CAPSTONE_SDK_EXTRA_CODEGEN above) and then VERIFIES
# it. A failed verification renames sdk.json to sdk.json.refused, so no capstone-cc compile can use it, and exits 3.
# CAPSTONE_SDK_ALLOW_BUILTINS=1 builds a corpus SDK without it deliberately and records that in sdk.json. Without
# CAPSTONE_SDK_CORPUS this builder behaves exactly as before: benchmark and port builds keep their codegen (moving it
# under the paper's performance numbers is the lead's call, not this gate's). The campaign's t3-freeze-nb SDK
# already carries -fno-builtin by hand, so a corpus SDK built this way matches it.
if [[ ${CAPSTONE_SDK_CORPUS:-0} == 1 ]]; then
  python3 - "$SDK/sdk.json" "${CAPSTONE_SDK_ALLOW_BUILTINS:-0}" <<'GATE' || exit 3
import json, os, sys
path, allow = sys.argv[1], sys.argv[2] == "1"
config = json.load(open(path))
if allow:
    config["builtins_allowed"] = ("CAPSTONE_SDK_ALLOW_BUILTINS=1 at build: allocation calls may be elided or "
                                  "assumed to succeed (C-81); a corpus SDK built without -fno-builtin")
    json.dump(config, open(path, "w"), indent=2)
    print("build-sdk.sh: corpus SDK with builtins ALLOWED by CAPSTONE_SDK_ALLOW_BUILTINS=1 (recorded)", file=sys.stderr)
    sys.exit(0)
def effective_no_builtin(codegen):
    # clang lets the LAST of -fbuiltin / -fno-builtin win, so presence alone is not the setting
    last = [w for w in codegen if w in ("-fbuiltin", "-fno-builtin")]
    return bool(last) and last[-1] == "-fno-builtin"
if "-fno-builtin" not in config["codegen"]:
    config["codegen"] = list(config["codegen"]) + ["-fno-builtin"]
    json.dump(config, open(path, "w"), indent=2)
config = json.load(open(path))           # verify what was written, not what was intended
if effective_no_builtin(config.get("codegen", [])):
    sys.exit(0)
os.replace(path, path + ".refused")
print("build-sdk.sh: REFUSED: a corpus SDK whose effective codegen is not -fno-builtin (C-81): "
      "a later -fbuiltin overrides it", file=sys.stderr)
sys.exit(1)
GATE
fi
