#!/bin/bash
# B5b-2 on silicon: the B5b-2 directed tests as bare M-mode board images (load at 0x80000000), for B5b-2's full build
# (capstone-ariane b5b2-step6 0c0458149: steps 1-6 batched, CPTAG whole, and step 7's test). Wrapper and env as
# csmint-bare-2026-11-23 (inc/, = r79-pmp-bare-2026-11-23's), plus one kit-local recorder (BOARD_REC_PLAIN, for
# b5b2-sref, which names every GPR but tp). The tests are capstone-ariane verif/tests/custom/capstone/*.S at 0c0458149,
# unchanged (checked equal to the tree named by VP_SRC when given). Recorder scratch registers per test: pick-regs.py
# (satpstale-bare-2026-10-19), three GPRs the test never names.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
# riscv64-unknown-elf-gcc, as the tree builds these tests: clang expands macro names inside `#` comment lines
# (b5b2-sref.S:10, "LDC/STC (illegal, 2)"), gcc in assembler-with-cpp mode does not.
CROSS=${CAPSTONE_RV_CROSS:-riscv64-unknown-elf-}
TESTS="sublet-life b5b2-illegal b5b2-uprot b5b2-rule2 b5b2-sref b5b2-scratch b5b2-mwrite b5b2-raw b5b2-rawlcc b5b2-cptag b5b2-cptag2 b5b2-nsedge"
if [ -n "${VP_SRC:-}" ]; then for f in $TESTS; do cmp "$VP_SRC/$f.S" "$B/tests/$f.S" || { echo "tests/$f.S differs from $VP_SRC"; exit 1; }; done; fi
mkdir -p "$B/out" "$B/images"
build() {   # dir name test-file id-char [extra -D]
  local o=$1 n=$2 f=$3 ch=$4; shift 4
  "${CROSS}gcc" -march=rv64gc -mabi=lp64 -mcmodel=medany -static -nostdlib -nostartfiles \
    -I"$B/inc" -I"$B/tests" -I"$L/env/macros" -I"$L/env/p" \
    -DBOARD_TEST_FILE="\"$f\"" -DBOARD_TEST_CH=$ch "$@" \
    -T "$B/inc/board.ld" "$B/inc/board_wrap.S" -o "$B/out/$n.elf"
  "${CROSS}objcopy" -O binary "$B/out/$n.elf" "$B/$o/$n.bin"
  "${CROSS}nm" "$B/out/$n.elf" | awk '$3=="board_rec"{print $1}' | sed 's/^0*//' > "$B/$o/$n.rec"
  printf '%-18s %8d bytes  sha256 %s  board_rec %s\n' "$n" "$(stat -c %s "$B/$o/$n.bin")" "$(sha256sum "$B/$o/$n.bin" | cut -c1-16)" "$(cat "$B/$o/$n.rec")"
}
R="-DBOARD_R_CAP=x8 -DBOARD_R_T1=x4 -DBOARD_R_T2=x2"
build images sublet-life        sublet-life.S   65 $R                 # the known-good control (B5b-1's kit, on silicon)
build images b5b2-illegal       b5b2-illegal.S  66 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4
build images b5b2-illegal-runv  b5b2-illegal.S  67 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4 -DARM_RUNV
build images b5b2-uprot         b5b2-uprot.S    68 $R
build images b5b2-uprot-mis     b5b2-uprot.S    69 $R -DUP_MIS
build images b5b2-rule2         b5b2-rule2.S    70 $R
build images b5b2-rule2-pair    b5b2-rule2.S    71 $R -DR2_PAIR
build images b5b2-sref          b5b2-sref.S     72 -DBOARD_REC_PLAIN
build images b5b2-sref-gen      b5b2-sref.S     73 -DBOARD_REC_PLAIN -DSR_GEN
build images b5b2-scratch       b5b2-scratch.S  74 -DBOARD_R_CAP=x17 -DBOARD_R_T1=x8 -DBOARD_R_T2=x4
build images b5b2-mwrite        b5b2-mwrite.S   75 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build images b5b2-raw           b5b2-raw.S      76 $R
build images b5b2-rawlcc        b5b2-rawlcc.S   77 $R
build images b5b2-cptag         b5b2-cptag.S    78 $R
build images b5b2-cptag2        b5b2-cptag2.S   79 -DBOARD_R_CAP=x16 -DBOARD_R_T1=x17 -DBOARD_R_T2=x8
build images b5b2-nsedge        b5b2-nsedge.S   80 $R
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
# simulation only, never booted: the plain recorder's cross-check against the standard one on the same test
mkdir -p "$B/out/xcheck"
build out/xcheck b5b2-illegal-plain b5b2-illegal.S 66 -DBOARD_REC_PLAIN
