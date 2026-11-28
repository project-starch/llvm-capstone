#!/bin/bash
# The combined B5b-2 build on silicon (capstone-ariane b5b2-wedgeap-flu 28fcbb646: 4(b) + the drain + the guard + the
# wedge apertures + the FLU's untagged-x[rs1] rule): b5b2-untagged as a bare M-mode image, with no custom-3 encoding.
# It mints through CSMINT and records and reports through plain M stores (inc/, b5b2n's = crevoke4b's). The test is
# capstone-ariane verif/tests/custom/capstone/b5b2-untagged.S at 28fcbb646, unchanged (checked equal to VP_SRC when
# given); sublet-life.S is the kits' control.
# BOARD_R_CAP: the recorder's base, a register the test never names (pick-regs.py; b5b2-sref names all but tp).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
CROSS=${CAPSTONE_RV_CROSS:-riscv64-unknown-elf-}
TESTS="b5b2-untagged"
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
build b5b2-untagged        b5b2-untagged.S 66 x8                  # the CHERI rule on an untagged x[rs1], protected U / S / plain M
(cd "$B/images" && sha256sum *.bin > SHA256SUMS)
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
  want=0
  [ "$c" = "$want" ] || { echo "CUSTOM-3 IN $(basename $e): $c, expected $want"; exit 1; }
done
echo "custom-3 encodings: every image 0"
