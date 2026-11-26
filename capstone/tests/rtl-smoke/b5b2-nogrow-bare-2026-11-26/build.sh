#!/usr/bin/env bash
# NOGROW on Line B silicon: b5b2-nogrow (a full table that may not grow: CDERIVE's no-fault FAIL) and its growth-0 control
# (-DNG_CTL: 30), with sublet-life as the boot control, each with its own id char. No custom-3 encoding.
# capstone-ariane b5b2-hangmon-c's verif/tests/custom/capstone/b5b2-nogrow.S at da427ee88, unchanged (checked equal to VP_SRC when given).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
CROSS=${CAPSTONE_RV_CROSS:-riscv64-unknown-elf-}
TESTS="b5b2-nogrow"
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
build sublet-life          sublet-life.S   65 x8                   # the known-good control
build b5b2-nogrow-ctl      b5b2-nogrow.S        70 x8 -DNG_CTL     # the control: growth 0, the CDERIVE raises 30
build b5b2-nogrow          b5b2-nogrow.S        71 x8              # NOGROW: no fault, rd untagged, nothing allocated
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
