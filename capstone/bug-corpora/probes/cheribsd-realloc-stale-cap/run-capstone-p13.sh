#!/bin/bash
# Host side of the Capstone twin, as run on p13 on 2026-10-27 against the flat v7 gate platform
# (~/v6gate/platform-sdev: QEMU b290f565 v7 build, images-v7, gates-1018 adapter) with the dev
# control SDK (~/heap-v6/sdk-control). Usage: bash run-capstone-p13.sh <fresh outdir>
set -u
OUT=${1:?usage: run-capstone-p13.sh <fresh outdir>}
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
G=$HOME/v6gate; P=$G/platform-sdev; V=$G/gates-1018/tree-dev6/capstone/runtime/virtual
SDK=${SDK:-$HOME/heap-v6/sdk-control}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/stage" "$OUT/tmp"
export CAPSTONE_QEMU_LOCK=$G/qemu-realloc-probe.lock CAPSTONE_TMP_ROOT=$OUT/tmp
export CAPSTONE_QEMU_CPU_EXTRA=svadu=false,x-capstone-supervisor-slots=32
"$SDK/capstone-cc" -O1 "$HERE/probe-capstone.c" -o "$OUT/stage/probe.dom" || exit 75
cp "$P/adapter/capstone-vexec" "$P/adapter/module/capstone_vm.ko" "$OUT/stage/"
cp "$HERE/gate-capstone.sh" "$OUT/stage/gate.sh"; chmod 755 "$OUT/stage/gate.sh"
sha256sum "$P/qemu-system-riscv64" "$P"/images/* "$OUT"/stage/* "$SDK/libapplication-runtime.a" > "$OUT/inputs.sha256"
"$HOME/spec-kit/toolchain/build/bin/llvm-nm" "$OUT/stage/probe.dom" > "$OUT/probe.nm"
python3 "$V/run-staged.py" --qemu "$P/qemu-system-riscv64" --images "$P/images" \
  --stage "$OUT/stage" --work "$OUT/guest" --timeout 600 --disk-mib 64
rc=$?
echo "run-capstone-p13: run-staged exit $rc"
grep -a "@@PROBE\|@@CLEANUP\|@@INSMOD\|VIRTUAL_STAGED_DONE" "$OUT/guest/serial.log" | tr -d '\r' > "$OUT/result-lines.txt"
/usr/bin/cat "$OUT/result-lines.txt"
exit "$rc"
