#!/usr/bin/env bash
# Historical entry point; this branch has no external-module/launcher build.
set -euo pipefail
cat >&2 <<'MESSAGE'
The external Capstone module and capstone-vexec build has been retired on this branch.
Use capstone/runtime/virtual/install-native-task.sh with a prepared Linux tree,
then build Image with CONFIG_RISCV_CAPSTONE_TASK=y. Boot native applications
with the capstone_vm kernel arguments documented in tests/native-task/README.md.
Do not package or load an old capstone_vm.ko with the native kernel.
MESSAGE
exit 1
