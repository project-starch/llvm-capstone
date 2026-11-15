#!/usr/bin/env bash
# build-adapter.sh OUT: the adapter directory of a trap-route kit.
#
# The kernel's Sublet code is built into the kernel (capstone/kernel/apply.sh), so this
# directory holds no module. It holds the two programs the runners stage next to an image:
#   capstone-vexec  starts an image and prints the launcher's "domain fault" line from the
#                   kernel's fault line (trap-vexec.sh, a shell script; ../linux/native-exec.c
#                   read the deleted route's record and is no longer built)
#   capstone-job    records a remote command's wait status (../linux/job.c)
# trap_platform.py make-kit writes a whole kit; this script remains for runners that take --adapter.
# Do not package or load an old capstone_vm.ko with the native kernel.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
mkdir -p "$out"

install -m 0755 "$here/trap-vexec.sh" "$out/capstone-vexec"
"${CROSS_COMPILE}gcc" -O2 -static -Wall -Wextra -Werror "$here/../linux/job.c" -o "$out/capstone-job"
