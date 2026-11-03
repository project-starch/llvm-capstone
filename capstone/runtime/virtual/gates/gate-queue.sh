#!/usr/bin/env bash
# Install as ~/retire-gate/gate-queue.sh after qualification. Old callers remain exclusive.
set -uo pipefail
[[ $# -gt 0 ]] || { echo 'usage: gate-queue.sh COMMAND [ARG...]' >&2; exit 2; }
root=$HOME/retire-gate
mkdir -p "$root"
log=$root/gate-queue.log
stamp() { printf '%s %s pid=%s %s\n' "$(date -u +%FT%TZ)" "$1" "$$" "${2:-}" >> "$log"; }
stamp WAIT
slot=serial
if [[ ${GATE_PARALLEL:-0} == 1 ]]; then
  # Admission is opt-in and only for the driver that actually leases its resources.
  [[ $# == 3 && $1 == bash && -f $2 ]] &&
    grep -qx '# GATE_SLOT_PROTOCOL=1' "$2" || {
      echo 'gate-queue: parallel dispatch requires bash <slot-aware run-arm.sh> OUT' >&2; exit 2;
    }
  slots=${GATE_SLOTS:-1}
  [[ $slots == 1 || $slots == 2 ]] || { echo 'gate-queue: GATE_SLOTS must be 1 or 2' >&2; exit 2; }
  [[ -z ${GATE_SLOT+x} ]] || { echo 'gate-queue: dispatcher assigns GATE_SLOT; unset it' >&2; exit 2; }
  # Hold admission while waiting for a permit. Queued arms do not acquire the
  # legacy lock shared until they can run, so an exclusive arm waits for at most N.
  exec 7>>"$root/.gate-admission.lock"
  flock -x 7 || exit 75
  while :; do
    for ((slot=0; slot<slots; ++slot)); do
      exec 6>>"$root/.gate-dispatch-$slot.lock"
      if flock -n 6; then break 2; fi
      exec 6>&-
    done
    sleep 1
  done
  export GATE_SLOT=$slot
  mode=-s
else
  # A legacy driver still uses port 21299. It excludes every upgraded arm too.
  mode=-x
fi
exec 8>>"$root/.gate-queue.turnstile" 9>>"$root/.gate-queue.lock"
flock -x 8 || exit 75
flock "$mode" 9 || exit 75
flock -u 8
exec 8>&-
if [[ $slot != serial ]]; then flock -u 7; exec 7>&-; fi
stamp RUN "slot=$slot"
"$@"
rc=$?
stamp DONE "slot=$slot rc=$rc"
exit "$rc"
