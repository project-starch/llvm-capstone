#!/usr/bin/env bash
# C-56: the address of an undefined weak symbol must be null in a domain.
#
#   SDK=<application SDK> KIT=<QEMU kit> ./run.sh
#       builds src/ at -O0 and -O2 with the SDK's capstone-cc, runs both in one
#       boot of the kit (capstone-vexec, capstone_vm.ko, run-staged.py), VERDICT:
#       exit 0 = FIXED (every check ok in both images), 1 = PRESENT,
#       2 = no result (build or boot failed, or the control line is missing).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2
D=capstone/tests/compiler-repros/C56-undefined-weak-null
: "${SDK:?set SDK to an application SDK directory}" "${KIT:?set KIT to a QEMU kit directory}"
W=$(mktemp -d "${TMPDIR:-/tmp}/c56.XXXXXX") || exit 2
ST=$W/stage
mkdir -p "$ST" || exit 2
cp "$KIT/adapter/capstone-vexec" "$KIT/adapter/module/capstone_vm.ko" "$ST/" || exit 2
for o in O0 O2; do
  "$SDK/capstone-cc" -$o -o "$ST/weak-$o.dom" "$D/src/weak.c" "$D/src/weak2.c" || exit 2
done
cat > "$ST/gate.sh" <<'GATE'
#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
for v in weak-O0 weak-O2; do
  echo "C56_ARM:$v"
  ./capstone-vexec $v.dom
  echo "C56_EXIT:$v:$?"
done
rmmod capstone_vm
echo VIRTUAL_STAGED_DONE
GATE
chmod +x "$ST/gate.sh"
python3 capstone/runtime/virtual/run-staged.py --qemu "$KIT/qemu-system-riscv64" --images "$KIT/images" \
  --stage "$ST" --work "$W/guest" --timeout 600 > "$W/run.log" 2>&1 || { echo "  VERDICT: NO RESULT (boot failed, see $W)"; exit 2; }
python3 - "$W/guest/serial.log" <<'PY'
import re, sys
log = open(sys.argv[1], errors='replace').read().replace('\r', '')
if 'VIRTUAL_STAGED_DONE' not in log:
    print('  VERDICT: NO RESULT (the gate did not finish)'); sys.exit(2)
arms = re.split(r'^C56_ARM:', log, flags=re.M)[1:]
if len(arms) != 2:
    print('  VERDICT: NO RESULT (%d arms)' % len(arms)); sys.exit(2)
fixed = True
for arm in arms:
    name = arm.split('\n', 1)[0]
    lines = [l for l in arm.splitlines() if l.startswith(('ok ', 'FAIL ', 'capstone-exec: domain fault', 'C56-DONE'))]
    for l in lines: print('  %s  %s' % (name, l[:120]))
    if not any(l.startswith('ok   control') for l in lines):
        print('  VERDICT: NO RESULT (%s: control line missing)' % name); sys.exit(2)
    if 'C56-DONE failures=0' not in arm:
        fixed = False
print('  VERDICT: C-56 ' + ('ABSENT' if fixed else 'PRESENT'))
sys.exit(0 if fixed else 1)
PY
