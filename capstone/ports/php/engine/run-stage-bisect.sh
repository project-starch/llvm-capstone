#!/usr/bin/env bash
# Stage-bisect zend_startup: run stages 1..N for each N and report which one first fails.
#   capstone/container/run.sh capstone/ports/php/engine/run-stage-bisect.sh [max]
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
R=$CAPSTONE_REPO_ROOT; P=${PHP_SRC:-/corpus/build/php-5.0.0}
O=$CAPSTONE_TMP_ROOT/php-engine
MAX=${1:-9}
# The domain runs in under a second; what times out is the GUEST BOOT under host
# load. Without this the bisect reports a spurious FAIL for a stage that passes.
TMO=${CAPSTONE_TIMEOUT_MULTIPLIER:-6}

# The engine objects must already exist (build-engine.sh builds them once).
[ -d "$O/obj" ] || { echo "run build-engine.sh first" >&2; exit 2; }

CF=(-target capstone64-unknown-elf -ffreestanding -fno-builtin -O0 -std=gnu89 -fno-common
    -gline-tables-only -nostdlibinc -isystem "$HERE/stubinc"
    -I"$P" -I"$P/Zend" -I"$P/main" -I"$P/TSRM" -I"$P/ext" -I"$P/ext/standard")

# TWO PHASES, deliberately.
#
# The build and the boot used to be interleaved, one clang compile and a 66-object lld
# link immediately before each guest boot. That made the boots flaky: runs died at
# random points in the GUEST BOOT (one in the OpenSBI banner, before Linux), and the
# same image booted by hand, or ten times in a loop without the builds in between,
# passed every time. Building everything first keeps the heavy work off the boots.
#
# And a failure is RETRIED once rather than trusted, because a spurious FAIL used to
# `exit 1` and hide every later stage -- which is how a passing stage 2 was reported as
# the first failure three separate times.

echo "  phase 1: building stages 1..$MAX"
BUILT=()
for n in $(seq 1 "$MAX"); do
  if ! "$CAPSTONE_CLANG" "${CF[@]}" -DRUNG_S_STAGE=$n \
       -c "$HERE/rung_S_domain.c" -o "$O/obj/domain.o" 2>/dev/null; then
    printf '    %-6s BUILD-FAIL\n' "$n"; continue
  fi
  ls "$O"/obj/*.o > "$O/objs.txt"
  if ! "$CAPSTONE_LD_LLD" --gc-sections -T "$R/capstone/my_first_domain/link.ld" \
       -o "$O/stage$n.dom" $(cat "$O/objs.txt") 2>/dev/null; then
    printf '    %-6s LINK-FAIL\n' "$n"; continue
  fi
  S="$O/share-s$n"; rm -rf "$S"; mkdir -p "$S"; cp "$O/stage$n.dom" "$S/"
  bash "$R/capstone/tests/runtime-qemu/build-capstone-test-user.sh" "$S/capstone-test.user" \
    >/dev/null 2>&1
  BUILT+=("$n")
done
echo "  built: ${BUILT[*]}"

# One domain per boot: a faulted domain poisons later create_dom in the same guest.
boot_one() {
  local n=$1 L=$2
  CAPSTONE_DEBUG_PRINT=1 flock "$CAPSTONE_QEMU_LOCK" python3 \
    "$R/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$O/share-s$n" --log-file "$L" \
    --timeout-multiplier "$TMO" \
    --guest-command "/mnt/host/capstone-test.user /mnt/host/stage$n.dom" \
    --success-marker "retval" >/dev/null 2>&1
  grep -a "retval = " "$L" 2>/dev/null | tail -1 | sed 's/.*retval = //'
}

echo "  phase 2: booting"
printf '  %-6s %-10s %s\n' stage result detail
FIRSTFAIL=
for n in "${BUILT[@]}"; do
  want=$((0x5000 + n))
  L="$CAPSTONE_TMP_ROOT/stage$n.log"
  v=$(boot_one "$n" "$L")
  note=
  if [ "${v:-}" != "$want" ]; then
    # Retry once: a lost boot is not a verdict.
    v=$(boot_one "$n" "$L"); note=" (after retry)"
  fi
  if [ "${v:-}" = "$want" ]; then
    printf '  %-6s %-10s %s\n' "$n" PASS "retval=$v$note"
  else
    oob=$(grep -a "Cap mem access OOB" "$L" 2>/dev/null | tail -1 | sed 's/.*OOB: //')
    cause=$(grep -aoE "cause = [0-9]+" "$L" 2>/dev/null | tail -1)
    printf '  %-6s %-10s %s\n' "$n" FAIL "retval=${v:-<none>} ${cause}"
    [ -n "$oob" ] && printf '         %s\n' "$oob"
    [ -z "$FIRSTFAIL" ] && FIRSTFAIL=$n
  fi
done
if [ -n "$FIRSTFAIL" ]; then
  echo "  -> first failing stage is $FIRSTFAIL"
  exit 1
fi
echo "  all $MAX stages passed"
