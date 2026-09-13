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

bash "$SCRIPT_DIR/build-nginx-domain.sh"
cp "$OUT_DIR/$DOM_NAME.dom" "$SHARE/"

GCC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
U=$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace
"$GCC" -O2 -I "$CAPSTONE_BUILDROOT_DIR/package/modcapstone/include" -I "$U" \
    -o "$SHARE/ngx-guest" "$SCRIPT_DIR/tools/ngx-guest.c" "$U/lib/libcapstone.c"

"${PYTHON:-python3}" "$REPO/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" --log-file "$OUT_DIR/boot.log" --timeout-multiplier 2 \
  --guest-command "/mnt/host/ngx-guest /mnt/host/$DOM_NAME.dom" \
  --success-marker "ngx retval" > "$OUT_DIR/smoke.txt" 2>&1 || true

line=$(grep -m1 'ngx retval' "$OUT_DIR/boot.log" 2>/dev/null || true)
if [ -z "$line" ]; then
  echo "no return; $OUT_DIR/boot.log says why" >&2
  grep -m1 'domain halted' "$OUT_DIR/boot.log" >&2 || true
  exit 1
fi
v=$(echo "$line" | grep -oE '[0-9]+$')
printf "%s\n" "$line"
printf "  checks  %d\n  failures %d\n" $(( (v >> 8) & 0xFF )) $(( v & 0xFF ))
[ $(( v & 0xFF )) -eq 0 ] && [ $(( (v >> 8) & 0xFF )) -gt 0 ]
