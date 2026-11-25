#!/bin/bash
# The final B5b-2 bitstream on silicon (4(b), interruptible CREVOKE, + the store drain + the residual-2 guard:
# capstone-ariane b5b2-crevoke4b 88db40027): the 4(b) and drain tests as bare M-mode images, with no custom-3
# encoding anywhere. Every test mints through CSMINT and records and reports through plain M stores (inc/, b5b2n's).
# The tests are capstone-ariane verif/tests/custom/capstone/*.S at 438a9c70b, unchanged (checked equal to VP_SRC when
# given); sublet-life.S is b5b2n's control.
# BOARD_R_CAP: the recorder's base, a register the test never names (pick-regs.py; b5b2-sref names all but tp).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); L=$B/../sup-bare-2026-10-03-ladder
CROSS=${CAPSTONE_RV_CROSS:-riscv64-unknown-elf-}
TESTS="b5b2-crevint b5b2-crevsv b5b2-sbdrain"
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
build b5b2-crevint         b5b2-crevint.S  66 x4                  # CREVOKE of a 200-record run, no interrupt
build b5b2-crevint-mt      b5b2-crevint.S  67 x4 -DCI_MTIMER      # the M timer during it
build b5b2-crevint-sh      b5b2-crevint.S  68 x4 -DCI_SHANDLER    # the no-Sstc shape, a CDERIVE between executions
build b5b2-crevint-lat     b5b2-crevint.S  69 x4 -DCI_LAT         # the stop latency, in mtime ticks, with its calibration
build b5b2-crevsv-none     b5b2-crevsv.S   70 x4 -DCS_NONE        # the saved position: nothing between restarts
build b5b2-crevsv-derive   b5b2-crevsv.S   71 x4 -DCS_DERIVE
build b5b2-crevsv-kill     b5b2-crevsv.S   72 x4 -DCS_KILL
build b5b2-crevsv-srev     b5b2-crevsv.S   73 x4 -DCS_SREV
build b5b2-crevsv-straw    b5b2-crevsv.S   74 x4 -DCS_STRAW
build b5b2-sbdrain-none    b5b2-sbdrain.S  75 x4 -DSB_NONE        # the store drain: growth stores, then a lifetime op
build b5b2-sbdrain-fill    b5b2-sbdrain.S  76 x4 -DSB_FILL
build b5b2-sbdrain-fence   b5b2-sbdrain.S  77 x4 -DSB_FENCE
build b5b2-sbdrain-sret    b5b2-sbdrain.S  78 x4 -DSB_SRET
build b5b2-sbdrain-mint    b5b2-sbdrain.S  79 x4 -DSB_MINT
build b5b2-sbdrain-cost    b5b2-sbdrain.S  80 x4 -DSB_COST
build b5b2-sbdrain-cost0   b5b2-sbdrain.S  81 x4 -DSB_COST0
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
