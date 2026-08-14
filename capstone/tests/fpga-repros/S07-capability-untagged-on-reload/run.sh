#!/usr/bin/env bash
# Rebuild the four exclusion rungs under QEMU, run each one's POSITIVE CONTROL, and stage them
# for the board.
#
# These are NEGATIVE results: each asks whether one simple property of a capability round trip is
# broken, and each says no. They are here so the hardware side can see what has already been
# excluded, and re-run it after any RTL change rather than taking the README's word for it.
#
# EVERY RUNG IS RUN TWICE, and the control arm is not optional: a rung returning 0xFFFF is
# indistinguishable from a rung whose check cannot fail. The selftest build feeds the same LCC
# query a value that is not a capability; it must return 0. If the control does not fire, the
# clean result means nothing and this script says so instead of printing a pass.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
LADDER="$ROOT/capstone/tests/runtime-qemu/silicon-ladder"
RUNNER="$LADDER/run-ladder-qemu.sh"
[[ -x "$RUNNER" || -f "$RUNNER" ]] || { echo "ERROR: runner not found at $RUNNER" >&2; exit 1; }

# Extract the domain's return value from the runner's output. Deliberately an ERROR, not a 0,
# when nothing matches -- a missing number that prints as a number is how a run that never
# happened gets read as a result.
retval_of() {
  local out=$1
  local v
  v=$(printf '%s\n' "$out" | grep -oE 'retval = [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  [[ -n "$v" ]] && printf '%s' "$v" || printf 'NO-RETVAL'
}

printf '%-10s %-52s %-8s %-8s %s\n' rung asks clean selftest verdict
printf '%-10s %-52s %-8s %-8s %s\n' --------- ---------------------------------------------- ------ -------- -------
fail=0
while read -r rung define question; do
  question=${question//_/ }
  clean=$(retval_of "$(RUNG_NAME=${rung}_clean bash "$RUNNER" "$rung" 2>&1)")
  st=$(retval_of "$(RUNG_NAME=${rung}_st DOMAIN_EXTRA_CFLAGS="-D$define" bash "$RUNNER" "$rung" 2>&1)")
  if [[ "$st" != "0" ]]; then
    verdict="CONTROL DID NOT FIRE -- clean result is meaningless"; fail=1
  elif [[ "$clean" == "65535" ]]; then
    verdict="excluded (property holds)"
  else
    verdict="RUNG FAILED -- zero bits name the bad slots"; fail=1
  fi
  printf '%-10s %-52s %-8s %-8s %s\n' "$rung" "$question" "$clean" "$st" "$verdict"
done <<'RUNGS'
s06spill S06SPILL_SELFTEST does_a_spilled_capability_come_back_TAGGED?
s06bnds S06BNDS_SELFTEST ...with_its_BOUNDS_intact?
s06wr S06WR_SELFTEST ...surviving_byte_stores_written_THROUGH_it?
s06pld S06PLD_SELFTEST ...surviving_a_scalar_load_of_its_own_granule?
RUNGS

echo
echo "To run them on the board, stage each and rebuild the initramfs (linux-rebuild BEFORE"
echo "opensbi-rebuild), then:  BAKED_RUNGS=\"s06spill s06bnds s06wr s06pld\" \\"
echo "  python3 -m fpga_driver.run_baked_rungs_fpga"
echo "The board runs the _fpga_app.c build, which is not the same bytes as the _app.c build QEMU"
echo "runs -- same kernel header and same -D flags, but a 24-byte result region instead of 8."

exit $fail
