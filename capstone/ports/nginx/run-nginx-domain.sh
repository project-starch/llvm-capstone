#!/usr/bin/env bash
# Run the pool in a domain and say what the seven scenarios did. One region is shared, which the
# level below turns into its arena; the driver then runs and packs its result into the return
# value: checks in bits 8..15, failures in bits 0..7, so a run that did nothing cannot read as a
# pass.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$REPO/capstone/tests/capstone-test-env.sh"
DOM_NAME=${DOM_NAME:-ngx-pool}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/nginx-domain}
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share}; mkdir -p "$SHARE"
# The protected arm cannot work on a NONLIN arena, so it never has to be asked for twice.
if [ "${NGX_SUBLET:-0}" = 1 ]; then NGX_ARENA_LINEAR=1; fi
export NGX_SUBLET NGX_ARENA_LINEAR

# Fatal on purpose. Without it a failed build leaves the previous image in the share
# directory, the run starts THAT, and a stale pass is reported for code that never
# compiled. It happened once, with a toolchain path that did not exist.
bash "$SCRIPT_DIR/build-nginx-domain.sh" || exit 1
cp "$OUT_DIR/$DOM_NAME.dom" "$SHARE/"

GCC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
U=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace
"$GCC" -O2 -I "$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" -I "$U" \
    -o "$SHARE/ngx-guest" "$SCRIPT_DIR/tools/ngx-guest.c" "$U/lib/libcapstone.c"

# Removed, not truncated by the runner: if the run never starts, the grep below must find
# nothing rather than the previous run's answer. Both halves of that trap have bitten here.
rm -f "$OUT_DIR/boot.log"
"${PYTHON:-python3}" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$OUT_DIR/boot.log" --timeout-multiplier 2 \
  --guest-command "/mnt/host/ngx-guest /mnt/host/$DOM_NAME.dom${NGX_ARENA_LINEAR:+ --arena-linear}" \
  --success-marker "ngx retval" > "$OUT_DIR/smoke.txt" 2>&1 || true

line=$(grep -m1 'ngx retval' "$OUT_DIR/boot.log" 2>/dev/null || true)
fault=$(grep -m1 'domain halted' "$OUT_DIR/boot.log" 2>/dev/null || true)

# NGX_EXPECT_FAULT inverts the gate, for the one image whose result IS the fault. A return is then
# the failure, and a fault still has to be a capability fault and not a dead domain, so the line
# itself is printed rather than counted.
if [ "${NGX_EXPECT_FAULT:-0}" = 1 ]; then
  if [ -n "$line" ]; then
    printf "NO FAULT: %s\n" "$line" >&2
    exit 1
  fi
  if [ -z "$fault" ]; then
    echo "neither a return nor a fault; $OUT_DIR/boot.log says why" >&2
    tail -3 "$OUT_DIR/smoke.txt" >&2
    exit 1
  fi
  printf "FAULTED as required\n  %s\n" "$fault"
  exit 0
fi

if [ -z "$line" ]; then
  echo "no return; $OUT_DIR/boot.log says why" >&2
  printf '%s\n' "${fault:-$(tail -3 "$OUT_DIR/smoke.txt")}" >&2
  exit 1
fi
v=$(echo "$line" | grep -oE '[0-9]+$')
printf "%s\n" "$line"

# NGX_EXPECT_MARK is for an image whose return is a marker and not a check count. The expectation
# is exact, because "the domain came back" is not a result: the whole question in the
# use-after-destroy image is WHICH of three marks came back.
if [ -n "${NGX_EXPECT_MARK:-}" ]; then
  got=$(( v & 0xFFFFFF ))
  want=$(( NGX_EXPECT_MARK ))
  printf "  mark     0x%06X  expected 0x%06X\n" "$got" "$want"
  [ "$got" -eq "$want" ]
  exit $?
fi
printf "  steps    %d\n  checks   %d\n  failures %d\n" \
    $(( (v >> 16) & 0xFF )) $(( (v >> 8) & 0xFF )) $(( v & 0xFF ))
# A run that did nothing must not read as a pass, so a zero check count fails too.
[ $(( v & 0xFF )) -eq 0 ] && [ $(( (v >> 8) & 0xFF )) -gt 0 ]
