#!/bin/bash
# Build the sup-capstl variants as bare M-mode board images (load at 0x80000000), with the supervised-CALL ladder's
# committed board harness (../sup-bare-2026-10-03-ladder: inc/ + env/).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
H=$B/../sup-bare-2026-10-03-ladder
BIN=${CAPSTONE_LLVM_BIN:-$(git -C "$B" rev-parse --show-toplevel)/llvm/cmake-build-debug/bin}
mkdir -p "$B/out"
build() {   # name id-char [extra -D]
  local n=$1 ch=$2; shift 2
  "$BIN/clang" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -fuse-ld=lld -I"$H/inc" -I"$B/tests" -I"$H/env/macros" -I"$H/env/p" \
    -DBOARD_TEST_FILE='"sup-capstl.S"' -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8 "$@" \
    -T "$H/inc/board.ld" "$H/inc/board_wrap.S" -o "$B/out/$n.elf"
  "$BIN/llvm-objcopy" -O binary "$B/out/$n.elf" "$B/out/$n.bin"
  printf '%-14s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/out/$n.bin")" \
    "$(sha256sum "$B/out/$n.bin" | cut -c1-16)" "$("$BIN/llvm-nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}')"
}
build intloop-q64 73 -DINTLOOP -DQUANTUM=64
build capstl-q64  67 -DQUANTUM=64
build capstl-q47  68 -DQUANTUM=47
build capstl-q16  69 -DQUANTUM=16
# cache-miss variants (after the hot-cache session completed every arm): ITER 2^20 for the domain-side sweep
build evict-noploop-q100k 78 -DEVICT -DNOPLOOP -DQUANTUM=100000 -DITER=1048576
build evict-intloop-q100k 79 -DEVICT -DINTLOOP -DQUANTUM=100000 -DITER=1048576
build evict-capstl-q100k  80 -DEVICT -DQUANTUM=100000 -DITER=1048576
build evict-capstl-q20k   81 -DEVICT -DQUANTUM=20000 -DITER=1048576
build mevict-noploop-q64  82 -DMEVICT -DNOPLOOP -DQUANTUM=64
build mevict-capstl-q64   83 -DMEVICT -DQUANTUM=64
# the FPGA monitor's __domcallsaves sequence around every CALL (CPMP/CSR swap, cscratch := sp, sp := 0)
build mswap-capstl-q64          84 -DMSWAP -DQUANTUM=64
build mswap-evict-noploop-q100k 85 -DMSWAP -DEVICT -DNOPLOOP -DQUANTUM=100000 -DITER=1048576
build mswap-evict-capstl-q100k  86 -DMSWAP -DEVICT -DQUANTUM=100000 -DITER=1048576
build mswap-mevict-capstl-q64   87 -DMSWAP -DMEVICT -DQUANTUM=64
# the eviction positive control: the first escape's seal-line load latency, hot vs swept
build latprobe-capstl-q64        88 -DLATPROBE -DQUANTUM=64
build latprobe-mevict-capstl-q64 89 -DLATPROBE -DMEVICT -DQUANTUM=64
build latprobe-evict-capstl-q100k 90 -DLATPROBE -DEVICT -DQUANTUM=100000 -DITER=1048576
# after every MSWAP arm hung with 0 dots: the swap code's own control (no arm) and the fine-grained trace
build mswapdbg-plain-noploop     91 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64
build mswapdbg-noploop-q64       92 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64
build mswapdbg-capstl-q64        93 -DMSWAP -DTRACE_CHARS -DQUANTUM=64
# after the RTL lane found the x26 = s10 clash in MSWAP_PLAIN: the checksum is x29 under MSWAP
build mswapfix-plain-noploop     94 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64
build mswapfix-noploop-q64       95 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64
# bisect the swap on the PLAIN control (it hangs on silicon with all parts): one part per image
build swappart1-plain 97 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=1
build swappart2-plain 98 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=2
build swappart4-plain 99 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=4
build swappart8-plain 100 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=8
build swappart0-plain 101 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=0
# every single part completed; three-part combinations (each omits one part)
build swappart7-plain  102 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=7
build swappart11-plain 103 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=11
build swappart13-plain 104 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=13
build swappart14-plain 105 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=14
# only 7 (no cscratch/sp) completed: part 8 is needed. Pairs with 8, plus the full macro, with 'K' after the CALL
build swappart9k-plain  106 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DTRACE_K -DSWAP_PARTS=9
build swappart10k-plain 107 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DTRACE_K -DSWAP_PARTS=10
build swappart12k-plain 108 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DTRACE_K -DSWAP_PARTS=12
build swappart15k-plain 109 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DTRACE_K -DSWAP_PARTS=15
# the K print fixed every pair: no print, a fence or 8 nops right after the CALL; pairs without K; the armed arm with fence
build swap15-plain-fence  110 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=1
build swap15-plain-nop8   111 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=2
build swappart9-plain     112 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=9
build swappart10-plain    113 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=10
build swappart12-plain    114 -DMSWAP -DMSWAP_PLAIN -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=12
build swap15-armed-fence  115 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=1 -DQUANTUM=64
# armed, many resumes: the real monitor's sp-dependent load (with / without fence), and two masks on the independent load
build armdep-q64        116 -DMSWAP -DSWAP_DEP -DNOPLOOP -DTRACE_CHARS -DQUANTUM=64 -DITER=65536
build armdep-fence-q64  117 -DMSWAP -DSWAP_DEP -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=1 -DQUANTUM=64 -DITER=65536
build arm-print-q64     118 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=3 -DQUANTUM=64 -DITER=65536
build arm-csrr-q64      119 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DPOSTCALL=4 -DQUANTUM=64 -DITER=65536
# the DYN-unit discriminator (LDC vs ld after the post-CALL ccsrrw, parts 8+4), and the dependent arm with a dot per 16
build arm12-ldc-q64     120 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=12 -DQUANTUM=64 -DITER=65536
build arm12-ld-q64      121 -DMSWAP -DNOPLOOP -DTRACE_CHARS -DSWAP_PARTS=12 -DLDTEST -DQUANTUM=64 -DITER=65536
build armdep-d16-q64    122 -DMSWAP -DSWAP_DEP -DNOPLOOP -DTRACE_CHARS -DDOTMASK=15 -DQUANTUM=64 -DITER=65536
build armdep-nt-d16-q64 123 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DQUANTUM=64 -DITER=65536
# S-16 bisect on the fast repro: part 8 (cscratch/sp, sp-dependent swap-in) alone and with each other part
build s16p8-nt  124 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=8  -DQUANTUM=64 -DITER=65536
build s16p9-nt  125 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=9  -DQUANTUM=64 -DITER=65536
build s16p10-nt 126 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=10 -DQUANTUM=64 -DITER=65536
build s16p12-nt 127 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=12 -DQUANTUM=64 -DITER=65536
# every pair completed: three-part combinations with part 8
build s16p11-nt 128 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=11 -DQUANTUM=64 -DITER=65536
build s16p13-nt 129 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=13 -DQUANTUM=64 -DITER=65536
build s16p14-nt 130 -DMSWAP -DSWAP_DEP -DNOPLOOP -DDOTMASK=15 -DSWAP_PARTS=14 -DQUANTUM=64 -DITER=65536
