#!/usr/bin/env bash

# Source this from the repository root, or from any script under capstone/tests/.
# It centralizes the default paths used by Capstone tests and handoff docs.

_CAPSTONE_TEST_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
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

mkdir -p "$CAPSTONE_TMP_ROOT"


