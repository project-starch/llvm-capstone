#!/usr/bin/env bash
# Reproducers for the runtime defects the tshark port found, run through the port's QEMU runner
# (host/run-qemu.sh, oracle mode, with the program's native build as the reference):
#
#   run.sh c64   ctors.c: stdout MATCH, constructors, main and destructors in native order. Before
#                C-64 was fixed in the runtime (hostcall.c) it HALTED with cause 24 in musl's exit()
#                walking .fini_array through uintptr_t, and the constructors never ran.
#   run.sh c65   cond.c: HALTED, cause 24, in musl's __private_cond_signal (C-65, open)
#   run.sh i11   closefd1.c: the runtime's "capstone-domain: UNSERVED syscalls: 160x2" line is
#                present although the program closed fd 1. Before I-11 was fixed it was missing;
#                the port's exit hook reports unserved=160x2 stdout=closed either way.
#
# The core regression tests for C-64 and I-11 are tests/runtime-qemu/init-fini and unserved-report.
# Each builds a stand-in domain the way host/build-domain.sh builds tshark's (runtime from
# deps/env.sh, a 4 MiB level0 arena with stats, src/tsapp-heap.c as the exit hook, a declared stack).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); APP=$(cd -- "$HERE/../.." && pwd)
source "$APP/deps/env.sh" > /dev/null
CASE=${1:?c64 | c65 | i11}
case $CASE in c64) SRC=ctors.c ;; c65) SRC=cond.c ;; i11) SRC=closefd1.c ;; *) echo "unknown case" >&2; exit 2 ;; esac
W=$TS_WORK/runtime-gaps/$CASE; rm -rf "$W"; mkdir -p "$W"
RTF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -Xclang -target-feature -Xclang +a
     -ffreestanding -fno-builtin -fno-jump-tables -ffunction-sections -fdata-sections -std=c99 -O1 -w
     -Wno-int-conversion -D_XOPEN_SOURCE=700 -nostdinc
     -isystem "$TS_MUSL/arch/capstone64" -isystem "$TS_MUSL/arch/generic" -isystem "$TS_MUSL/obj/include"
     -isystem "$TS_MUSL/include" -I"$TS_MUSL/src/include" -I"$TS_MUSL/src/internal" -I"$TS_MUSL/obj/src/internal")
"$CAPSTONE_CLANG" "${RTF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES=$((4 << 20)) -DCAPSTONE_LEVEL0_STATS \
  -c "$CAPSTONE_REPO_ROOT/capstone/ports/musl-capstone/runtime/level0.c" -o "$W/level0.o"
"$CAPSTONE_CLANG" "${RTF[@]}" -c "$APP/src/tsapp-heap.c" -o "$W/tsapp-heap.o"
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -ffreestanding -O0 -DCAPSTONE_DOMREQ_DATA=$((1 << 20)) \
  -DCAPSTONE_DOMREQ_STACK=$((1 << 20)) -c "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/domreq.S" -o "$W/domreq.o"
"$CC" -O1 -c "$HERE/$SRC" -o "$W/prog.o"
RT=(); for o in "$TS_RUNTIME_DIR"/*.o; do [ "$(basename "$o")" = level0.o ] || RT+=("$o"); done
"$CAPSTONE_LD_LLD" --gc-sections -T "$TS_LINKER_SCRIPT" -o "$W/tshark_m5.dom" "${RT[@]}" "$W/level0.o" \
  "$W/tsapp-heap.o" "$W/domreq.o" "$W/prog.o" "$TS_LIBC_ARCHIVE"
cc -O1 -o "$W/native" "$HERE/$SRC" -lpthread
printf '#!/bin/sh\nexec %s\n' "$W/native" > "$W/stock.sh"; chmod +x "$W/stock.sh"
echo "native: $("$W/native" | tr '\n' '|')"
TSAPP_DOMAIN_DIR=$W TSAPP_STOCK=$W/stock.sh TSAPP_MINIMAL=none bash "$APP/host/run-qemu.sh" oracle dhcp | tee "$W/verdict.txt"
if [ "$CASE" = i11 ]; then
  L=$(grep -o "log /[^ ]*\.log" "$W/verdict.txt" | cut -d' ' -f2)
  echo "runtime UNSERVED lines in the domain's output: $(grep -ac '^capstone-domain: UNSERVED' "$L.out/out-dhcp.txt" || true)"
  grep -a '^TSAPP-HEAP' "$L.out/out-dhcp.txt"
fi
