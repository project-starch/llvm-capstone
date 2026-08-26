#!/usr/bin/env bash
# The three matched pairs behind the nested-allocator finding. Each differs in ONE
# constant, and the pairing is the point: a quiet arm on its own cannot be told apart
# from a domain that never started.
#
#   ./run-nested-allocator-pairs.sh temporal   use-after-free inside one region
#   ./run-nested-allocator-pairs.sh spatial    overflow between two live blocks
#   ./run-nested-allocator-pairs.sh global     overflow between two static globals
#   ./run-nested-allocator-pairs.sh all        all three, in that order
#
# The arm expected to RETURN runs first in every pair: a fault takes the domain with
# it, so anything after it in a boot is lost.
#
# The three bounds widths ARE the finding:
#   allocator-carved object, model heap   0x400   =   1 KiB   untrapped
#   allocator-carved object, MicroPython  0x60000 = 384 KiB   untrapped   (separate run)
#   linker-assigned global                0x40    =    64 B   TRAPPED
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
export CAPSTONE_REPO_ROOT="$PWD"
source capstone/tests/capstone-test-env.sh >/dev/null 2>&1
L=capstone/tests/runtime-qemu/silicon-ladder
MODE=${1:-all}
ATTEMPTS=${REPRO_ATTEMPTS:-3}
rc=0

arm() {  # $1 label  $2 base  $3 rung  $4 -D or ""  $5 expected retval or FAULT
  local label=$1 base=$2 rung=$3 def=$4 want=$5 log=/tmp/capstone/$rung.log a
  for a in $(seq 1 "$ATTEMPTS"); do
    if [[ -n $def ]]; then
      DOMAIN_EXTRA_CFLAGS="$def" RUNG_NAME="$rung" bash "$L/run-ladder-qemu.sh" "$base" >"$log" 2>&1
    else
      bash "$L/run-ladder-qemu.sh" "$base" >"$log" 2>&1
    fi
    [[ $? -ne 75 ]] && break
    echo "     attempt $a: infra flake, retrying"
  done
  if [[ $want == FAULT ]]; then
    if grep -qa "capability fault" "$log"; then
      printf "   %-34s FAULT as recorded   %s\n" "$label" \
        "$(grep -ao 'bounds = ([^)]*)' "$log" | tail -1)"
    else
      printf "   %-34s DIFFERS -- no fault: %s\n" "$label" "$(grep -a retval "$log" | tail -1)"; rc=1
    fi
  else
    if grep -qa "PASSED__ (retval = $want)" "$log"; then
      printf "   %-34s retval %s as recorded\n" "$label" "$want"
    else
      printf "   %-34s DIFFERS\n" "$label"; grep -aiE "oracle|retval|FAILED" "$log" | tail -2 | sed 's/^/       /'; rc=1
    fi
  fi
}

[[ $MODE == temporal || $MODE == all ]] && {
  echo "== temporal pair: a use-after-free write inside one allocator region"
  arm "arm A, stale write in region"  nestalloc nestalloc ""                          170
  arm "arm B, past the region"        nestoob   nestoob   ""                          FAULT
}
[[ $MODE == spatial || $MODE == all ]] && {
  echo "== spatial pair: two LIVE blocks, nothing freed"
  arm "arm A, out of a into b"        nestspat  nestspat_in  "-DNEST_SPATIAL_OFFSET=64"   170
  arm "arm B, past the region"        nestspat  nestspat_oob "-DNEST_SPATIAL_OFFSET=1088" FAULT
}
[[ $MODE == global || $MODE == all ]] && {
  echo "== static-global pair: the control the other two needed"
  arm "arm A, in bounds"              nestglob  nestglob_in  "-DNEST_GLOBAL_OFFSET=0"   34
  arm "arm B, out of a into b"        nestglob  nestglob_oob "-DNEST_GLOBAL_OFFSET=64"  FAULT
}
exit $rc
