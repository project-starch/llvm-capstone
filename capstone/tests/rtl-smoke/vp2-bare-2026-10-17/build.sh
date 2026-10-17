#!/bin/bash
# vp2's bare arms as M-mode board images (load at 0x80000000), for caplifive_vp2_bf3db3175.bit: item D's CSRUNV family,
# item A's flat lifetime table, and vp1's arms, each with the controls its header names. Wrapper, env and node-aware
# header as vp1-bare-2026-10-17 (inc/ copied from it). Sources are copies of capstone-ariane verif/tests/custom/capstone at
# virtual-profile-2-d c55e3ff31 (= bf3db3175's RTL; the four commits above it touch verif/ only; 1bcef8712 adds
# VB_TABLE to vb-trans.S and vb-trans-plain.S, c55e3ff31 fixes va-runv6), checked equal to the tree
# named by VP_SRC at build time when it is given.
# The recorder's three scratch GPRs per test are registers the test never names (pick-regs.py, recorded below), except
# va-csmint-mon, whose own code names all 32: its CAPPRINTs all sit in the M trap handler between the frame save of
# x1..x31 and m_return's restore, and a6, a7, s1 are unused in that handler (it uses sp, t0-t3, t6, a0, a3-a5, s3, s8,
# s11, ra), so they are dead at every CAPPRINT and restored from the frame afterwards.
# The board vector is NOT the header's simulation vector: in-domain CAPPRINTs and the recorder's own CAPCREATEs change
# it. The board vectors come from simulating THESE images on bf3db3175's RTL (the RTL lane).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
if [ -n "${VP_SRC:-}" ]; then
  for f in "$B"/tests/*.S; do cmp "$VP_SRC/$(basename "$f")" "$f" || { echo "tests/$(basename "$f") differs from $VP_SRC"; exit 1; }; done
  cmp "$VP_SRC/inc/capstone_csmint.S" "$B/tests/inc/capstone_csmint.S" || { echo "inc/capstone_csmint.S differs"; exit 1; }
fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char regs [extra -D]
  local n=$1 f=$2 ch=$3 r=$4; shift 4
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$B/inc" -I"$B/tests" -I"$B/tests/inc" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch $r "$@" -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-20s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
R16_8_4="-DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4"
build va-runv            va-runv.S        97  "$R16_8_4"
build va-runv-quantum    va-runv.S        98  "$R16_8_4" -DVA_QUANTUM
build va-runv2           va-runv2.S       99  "$R16_8_4"
build va-runv3           va-runv3.S      100  "$R16_8_4"
build va-runv4           va-runv4.S      101  "$R16_8_4"
build va-runv5           va-runv5.S      102  "$R16_8_4"
build va-runv6           va-runv6.S      103  "$R16_8_4"
build va-runv7           va-runv7.S      104  "$R16_8_4"
build va-barrier         va-barrier.S    105  "-DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8"
build va-q15             va-q15.S        106  "-DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x2"
build va-csmint-emu      va-csmint-emu.S 107  "-DBOARD_R_CAP=x8 -DBOARD_R_T1=x9 -DBOARD_R_T2=x18"
build va-csmint-mon      va-csmint-mon.S 108  "-DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x9"
build va-straddle        va-straddle.S   109  "-DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2"
build va-flat            va-flat.S       110  "$R16_8_4"
build va-flat-off        va-flat.S       111  "$R16_8_4" -DVA_FLAG_OFF
build va-alloc           va-alloc.S      112  "$R16_8_4"
build va-alloc-off       va-alloc.S      113  "$R16_8_4" -DVA_FLAG_OFF
build va-alloc-corrupt   va-alloc.S      114  "$R16_8_4" -DVA_CORRUPT
build va-alloc-exhaust   va-alloc.S      115  "$R16_8_4" -DVA_EXHAUST
build va-ns-collide      va-ns-collide.S 116  "$R16_8_4"
build va-ns-collide-off  va-ns-collide.S 117  "$R16_8_4" -DVA_FLAG_OFF
# vb-trans under item A needs a lifetime table at srevroot (VB_TABLE, 1bcef8712); the flag-off build of the edited file
# reproduces vp1-bare's vb-trans and vb-trans-off hashes exactly, so vp1's images are unchanged.
build vb-trans-vp2       vb-trans.S       121 "-DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2" -DVB_TABLE
build vb-trans-plain-vp2 vb-trans-plain.S 122 "-DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2" -DVB_TABLE
# q17-order (bare M-mode, no supervised domain) runs as vp1-bare-2026-10-17's image, byte for byte, so the three
# bitstreams' readings compare on one set of bytes. vb-trans and vb-trans-off as vp1 built them were held out of vp2: since item A a domain under csupctl.virt runs in the lifetime-TABLE namespace at srevroot,
# vb-trans sets no srevroot, and on bf3db3175 every arm reads csupstatus 5 / csupcause 25 (the RTL lane, 2026-10-17).
# On vp1 (no item A) vb-trans is correct as pre-registered and stays in vp1's batch unchanged. Their sources and this wrapper are identical to vp1-bare's, and this
# recipe with vp1's id characters (86, 87, 88) reproduces its three hashes exactly (a3fa54d0, 79e670cc, ee2adf9b),
# which also checks the toolchain behind every image above.
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
(cd "$B/out" && sha256sum *.elf > SHA256SUMS)
