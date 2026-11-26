#!/bin/bash
# run-b3-mcf.sh (nested lane, 2026-11-27): the window's check, then mcf. gcc's v3 run (runs/spec-gcc) passed its gate
# except the window: the first plugin build matched the UART by region name and counted 0 transmit stores, and the
# reader missed the pty's '\r\r\n'. B3 is a smoke boot with the fixed plugin: its transmit stores must equal the
# console's bytes (D <= 64, over 1000 stores), or mcf does not run. Then chain-dc.sh with mcf alone.
main() {
  set -u
  local R=<hdd>/capstone/nested/rectrace-dc G=<hdd>/capstone/nested/rectrace-dc/diag
  VARIANTS=B3 bash $G/diag-boot.sh >> $G/diag.log 2>&1
  python3 $R/dc-window.py $G/B3/boot.log $G/B3/snap.txt $G/B3/dcplugin.txt > $G/B3/window.out 2>&1
  local l; l=$(command grep -a '^console_chars' $G/B3/window.out)
  echo "B3 $l"
  python3 - "$l" <<'PY' || { echo "STOP B3 window check"; return 1; }
import sys
f = sys.argv[1].split()
c, t, d = int(f[1]), int(f[3]), int(f[5])
print('B3 check console %d thr %d D %d' % (c, t, d)); sys.exit(0 if (t > 1000 and d <= 64) else 1)
PY
  CHAIN_BENCH=mcf:505.mcf_r bash $R/chain-dc.sh
  tail -n 12 $R/chain-dc.log
}
main "$@"; exit
