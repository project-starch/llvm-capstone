#!/usr/bin/env bash
# build-adapter.sh OUT: the adapter directory of a native kit.
#
# The Capstone adapter is built into the kernel (install-native-task.sh, then
# Image with CONFIG_RISCV_CAPSTONE_TASK=y), so this directory holds no module.
# It holds the two programs the runners stage next to an image:
#   capstone-vexec  starts an image as a native task and prints the launcher's
#                   "domain fault" line from the kernel's record (../linux/native-exec.c)
#   capstone-job    records a remote command's wait status (../linux/job.c)
# Do not package or load an old capstone_vm.ko with the native kernel.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
mkdir -p "$out"

"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/native-exec.c" -o "$out/capstone-vexec"
"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/job.c" -o "$out/capstone-job"
