#!/usr/bin/env bash
# build.sh OUT: the first-program images (ok.dom, stale.dom, spin.dom, bounds.dom, map.dom, unmapped.dom, signal.dom, gup.dom, exec.dom, sockopt.dom, argv.dom, futex.dom, prlimit.dom, zap.dom, fp.dom, big.dom, kread.dom, cred.dom) into OUT.
#   CROSS_COMPILE  RISC-V Linux tool prefix
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out=${1:?output directory}
mkdir -p "$out"
for m in 0:ok 1:stale 2:spin 3:bounds 4:map 5:unmapped 6:signal 7:gup 8:exec 9:sockopt 10:argv 11:futex 12:prlimit 13:zap 14:fp 15:big 16:kread 17:cred; do
    "${CROSS_COMPILE:?}gcc" -nostdlib -nostartfiles -static -march=rv64gc -mabi=lp64d -DMODE=${m%%:*} \
        -Wl,-T,"$here/dom.ld" -Wl,--no-relax -o "$out/${m#*:}.dom" "$here/program.S"
    # e_machine 259, the Capstone image's machine number, which the .dom loader requires
    printf '\x03\x01' | dd of="$out/${m#*:}.dom" bs=1 seek=18 conv=notrunc status=none
done
