#!/usr/bin/env bash

# Source this from the repository root, or from any script under capstone/tests/.
# It centralizes the default paths used by Capstone tests and handoff docs.

# BASH_SOURCE does not exist in zsh, and the project shell IS zsh. `${BASH_SOURCE[0]}`
# then expanded to nothing, `dirname ""` gave ".", and two levels up from the repo root
# is /home -- so CAPSTONE_REPO_ROOT silently became "/home" and every path built from it
# pointed nowhere. Found 2026-08-19; it had been wrong for every zsh session, and wrong
# quietly, because most consumers then failed with "no such file" rather than naming the
# cause. Both branches parse in both shells (the untaken one is never expanded).
if [ -n "${ZSH_VERSION:-}" ]; then
  _CAPSTONE_TEST_ENV_SRC=${(%):-%N}
else
  _CAPSTONE_TEST_ENV_SRC=${BASH_SOURCE[0]}
fi
_CAPSTONE_TEST_ENV_DIR=$(cd -- "$(dirname -- "$_CAPSTONE_TEST_ENV_SRC")" && pwd)
_CAPSTONE_REPO_DEFAULT=$(cd -- "$_CAPSTONE_TEST_ENV_DIR/../.." && pwd)

export CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT:-$_CAPSTONE_REPO_DEFAULT}
export CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone}
export CAPSTONE_LLVM_BUILD_DIR=${CAPSTONE_LLVM_BUILD_DIR:-$CAPSTONE_REPO_ROOT/llvm/cmake-build-debug}
export CAPSTONE_LLVM_BIN=${CAPSTONE_LLVM_BIN:-$CAPSTONE_LLVM_BUILD_DIR/bin}
export CAPSTONE_LLVM_LIT=${CAPSTONE_LLVM_LIT:-$CAPSTONE_LLVM_BIN/llvm-lit}
export CAPSTONE_CLANG=${CAPSTONE_CLANG:-$CAPSTONE_LLVM_BIN/clang}
export CAPSTONE_LD_LLD=${CAPSTONE_LD_LLD:-$CAPSTONE_LLVM_BIN/ld.lld}
export CAPSTONE_LLVM_READOBJ=${CAPSTONE_LLVM_READOBJ:-$CAPSTONE_LLVM_BIN/llvm-readobj}
export CAPSTONE_BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:-$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot}
export CAPSTONE_QEMU_BINARY=${CAPSTONE_QEMU_BINARY:-$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/build/qemu-system-riscv64}
export CAPSTONE_HANDOFF_DIR=${CAPSTONE_HANDOFF_DIR:-$CAPSTONE_REPO_ROOT/capstone/agent-handoff}

# Self-check: a wrong root must be an ERROR, not a set of paths that point nowhere.
# This is what would have caught the zsh bug above on the day it was introduced.
if [ ! -d "$CAPSTONE_REPO_ROOT/capstone/tests" ] || [ ! -d "$CAPSTONE_REPO_ROOT/llvm" ]; then
  echo "capstone-test-env.sh: CAPSTONE_REPO_ROOT resolved to '$CAPSTONE_REPO_ROOT'," >&2
  echo "  which does not look like this repo (no capstone/tests and llvm under it)." >&2
  echo "  Set CAPSTONE_REPO_ROOT explicitly before sourcing." >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$CAPSTONE_TMP_ROOT"


