#!/bin/bash
# B5b-2's next bitstream on silicon (8a + 8b + the CCSRRW deletion + step 9: capstone-ariane b5b2-step9 904cd9c6b): the
# B5b-2 tests as bare M-mode images, with no custom-3 encoding anywhere (step 9 makes the family illegal on silicon).
# Every test mints through CSMINT (-DCAP_MINT_CSMINT) and records and reports through plain M stores (inc/). The tests
# are capstone-ariane verif/tests/custom/capstone/*.S at 2bed0dfe7, unchanged (checked equal to VP_SRC when given).
# BOARD_R_CAP: the recorder's base, a register the test never names (pick-regs.py; b5b2-sref names all but tp).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
CROSS=${CAPSTONE_RV_CROSS:-riscv64-unknown-elf-}
TESTS="sublet-life b5b2-illegal b5b2-uprot b5b2-rule2 b5b2-sref b5b2-scratch b5b2-mwrite b5b2-raw b5b2-rawlcc b5b2-cptag b5b2-cptag2 b5b2-nsedge b5b2-ctxport"
if [ -n "${VP_SRC:-}" ]; then for f in $TESTS; do cmp "$VP_SRC/$f.S" "$B/tests/$f.S" || { echo "tests/$f.S differs from $VP_SRC"; exit 1; }; done; fi
mkdir -p "$B/out" "$B/images"
build() {   # name test-file id-char base-reg [extra -D]
  local n=$1 f=$2 ch=$3 r=$4; shift 4
  "${CROSS}gcc" -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch -DBOARD_R_CAP=$r -DCAP_MINT_CSMINT "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "${CROSS}objcopy" -O binary "$B/out/$n.elf" "$B/images/$n.bin"
  "${CROSS}nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/images/$n.rec"
  printf '%-22s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/images/$n.bin")" "$(sha256sum "$B/images/$n.bin" | cut -c1-16)" "$(cat "$B/images/$n.rec")"
}
build sublet-life          sublet-life.S   65 x8                  # the known-good control
build b5b2-illegal         b5b2-illegal.S  66 x16                 # step 1's deletions, CCSRRW's site 24 and step 9's sites 19-23, 25
build b5b2-illegal-runv    b5b2-illegal.S  67 x16 -DARM_RUNV      # step 8a: csrunv
build b5b2-illegal-csup    b5b2-illegal.S  68 x16 -DARM_CSUP      # step 8a: the supervisor CSRs
build b5b2-uprot           b5b2-uprot.S    69 x8
build b5b2-uprot-mis       b5b2-uprot.S    70 x8 -DUP_MIS
build b5b2-rule2           b5b2-rule2.S    71 x8
build b5b2-rule2-pair      b5b2-rule2.S    72 x8 -DR2_PAIR
build b5b2-sref            b5b2-sref.S     73 x4
build b5b2-sref-gen        b5b2-sref.S     74 x4 -DSR_GEN
build b5b2-scratch         b5b2-scratch.S  75 x17
build b5b2-mwrite          b5b2-mwrite.S   76 x16
build b5b2-raw             b5b2-raw.S      77 x8
build b5b2-rawlcc          b5b2-rawlcc.S   78 x8
build b5b2-cptag           b5b2-cptag.S    79 x8
build b5b2-cptag2          b5b2-cptag2.S   80 x16
build b5b2-nsedge          b5b2-nsedge.S   81 x8
build b5b2-ctxport-f4      b5b2-ctxport.S  82 x16 -DCP_F4
build b5b2-ctxport-f4ctl   b5b2-ctxport.S  83 x16 -DCP_F4CTL
build b5b2-ctxport-widelo  b5b2-ctxport.S  84 x16 -DCP_WIDE_LO
build b5b2-ctxport-widehi  b5b2-ctxport.S  85 x16 -DCP_WIDE_HI
build b5b2-ctxport-edgem   b5b2-ctxport.S  86 x16 -DCP_EDGEM
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
# simulation only, never booted: the key check's skew (the root assumed at slot 4), which must read the sentinel alone
mkdir -p "$B/out/skew"
build_skew() { local n=$1; shift; "${CROSS}gcc" -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" -DBOARD_TEST_CH=90 -DCAP_MINT_CSMINT -DKIT_ROOT_SLOT=4 "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/skew/$n.elf"
  "${CROSS}nm" "$B/out/skew/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/out/skew/$n.rec"; }
build_skew nsedge-skew  -DBOARD_TEST_FILE='"b5b2-nsedge.S"'  -DBOARD_R_CAP=x8
build_skew sref-skew    -DBOARD_TEST_FILE='"b5b2-sref.S"'    -DBOARD_R_CAP=x4
build_skew ctxport-skew -DBOARD_TEST_FILE='"b5b2-ctxport.S"' -DBOARD_R_CAP=x16 -DCP_F4
# no image may hold a custom-3 encoding (opcode 0x7b) except b5b2-illegal's six deletion arms
for e in "$B"/out/*.elf; do
  c=$(python3 - "$e" <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read(); shoff = struct.unpack_from('<Q', b, 0x28)[0]; es, n = struct.unpack_from('<HH', b, 0x3a); c = 0
for i in range(n):
    _, typ, fl, _, off, size = struct.unpack_from('<IIQQQQ', b, shoff + i * es)
    if typ != 1 or not fl & 4: continue
    p = off
    while p + 2 <= off + size:
        h = b[p] | b[p + 1] << 8
        if h & 3 != 3: p += 2; continue
        if (h & 0x7f) == 0x7b: c += 1
        p += 4
print(c)
PY
)
  want=0; [ "$(basename $e)" = b5b2-illegal.elf ] && want=6   # its deletion arms, sites 19-23 and 25: the scan's control
  [ "$c" = "$want" ] || { echo "CUSTOM-3 IN $(basename $e): $c, expected $want"; exit 1; }
done
echo "custom-3 encodings: b5b2-illegal 6 (its deletion arms, the control), every other image 0"
