#!/usr/bin/env bash

# Source this from the repository root, or from any script under capstone/tests/.
# It centralizes the default paths used by Capstone tests and handoff docs.

_CAPSTONE_TEST_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_CAPSTONE_REPO_DEFAULT=$(cd -- "$_CAPSTONE_TEST_ENV_DIR/../.." && pwd)

# Sourced from a shell without BASH_SOURCE (zsh, dash), dirname sees an empty
# string and the root resolves two levels above the working directory -- and
# because every var below defaults with :-, that wrong value then wins in every
# child process. Check the result rather than the mechanism: it catches the
# stale-export case too.
#
# Validate a CANDIDATE and only export once it passes. Exporting first would
# leave the caller holding the bad value even though the guard failed, which is
# the stale-export mode this guard exists to prevent.
_CAPSTONE_REPO_CANDIDATE=${CAPSTONE_REPO_ROOT:-$_CAPSTONE_REPO_DEFAULT}
if [ ! -d "$_CAPSTONE_REPO_CANDIDATE/llvm" ] ||
   [ ! -d "$_CAPSTONE_REPO_CANDIDATE/capstone" ]; then
  echo "capstone-test-env.sh: $_CAPSTONE_REPO_CANDIDATE is not the fork root." >&2
  echo "  Source this from bash: bash -c 'source capstone/tests/capstone-test-env.sh; ...'" >&2
  echo "  If CAPSTONE_REPO_ROOT is already exported wrong, clear it: unset CAPSTONE_REPO_ROOT" >&2
  unset _CAPSTONE_REPO_CANDIDATE
  return 1 2>/dev/null || exit 1
fi
export CAPSTONE_REPO_ROOT=$_CAPSTONE_REPO_CANDIDATE
unset _CAPSTONE_REPO_CANDIDATE
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

mkdir -p "$CAPSTONE_TMP_ROOT"


