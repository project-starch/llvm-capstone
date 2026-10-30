#!/bin/sh
# B (bitstream 4, capstone-ariane 325966ee4) first board boot, part 1 (spec-silicon PREREG Addendum 14; the predictions
# are module-ids55 PREREG section 2 + addendum 2, read against the twin's F0 arm). DIRECTED, hand-written:
#  1. the module's parameters and load lines (ring buffer, not the console);
#  2. Tier 1 contract: contract.dom `ok` (VIRTUAL_APPLICATION_OK, exit 0), then `stale` (cause 25, exit 139), as the
#     vp2 Tier 1 rung (board-tier12/rev8/vm-contract.sh) runs it, with run.py's environment;
#  3. the live smoke of the twin grid (f55/live-stage/run.sh, verbatim loop): N = 1000, 400000, 1100000. 1100000 is LAST:
#     it drives the flat table to exhaustion;
#  4. the module's allocation line and CMA lines from the ring buffer.
S=$(cd "$(dirname "$0")" && pwd)
V=${SPEC_VEXEC:?set SPEC_VEXEC}
echo "@@B4-STAGE t1a"
for p in ids index_bits node_flat_records node_max_pages; do echo "@@SMOKE-PARAM $p=$(cat /sys/module/capstone_vm/parameters/$p 2>&1)"; done
dmesg | grep -a 'capstone_vm:' | sed 's/^/@@KMSG-LOAD /'
mkdir -p /tmp/b4 && cd /tmp/b4 || { echo "@@B4-STAGE-ERR cd"; exit 1; }
export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1
echo "@@C-BEGIN ok"
"$V" "$S/contract.dom" ok argument > /tmp/b4/c-ok.out 2>&1; ok=$?
sed 's/^/@@C-OUT ok: /' /tmp/b4/c-ok.out; echo "VM_EXIT:normal:$ok"
echo "@@C-BEGIN stale"
"$V" "$S/contract.dom" stale > /tmp/b4/c-st.out 2>&1; st=$?
sed 's/^/@@C-OUT stale: /' /tmp/b4/c-st.out; echo "VM_EXIT:stale:$st"
r=0
[ $ok -eq 0 ] || r=922
[ $r -eq 0 ] && { grep -q "^VIRTUAL_APPLICATION_OK " /tmp/b4/c-ok.out || r=923; }
[ $r -eq 0 ] && { [ $st -eq 139 ] || r=924; }
[ $r -eq 0 ] && { grep -q "domain fault cause=25 " /tmp/b4/c-st.out || r=925; }
echo "RESULT vm-contract retval=$r"
unset CAPSTONE_VM_CONTRACT CAPSTONE_EXEC_DIAGNOSTICS
for n in 1000 400000 1100000; do
  echo "@@SMOKE-BEGIN live $n"
  "$V" "$S/bin/live-T3" $n > /tmp/b4/live.out 2>&1; rc=$?
  grep -a "LIVE" /tmp/b4/live.out
  grep -a "CAPSTONE_VM_STATS" /tmp/b4/live.out | sed 's/.*\(nodes=[0-9]*\).*\(node_capacity=[0-9]*\).*/STATS \1 \2/'
  grep -a -c "domain fault" /tmp/b4/live.out | sed 's/^/FAULT-LINES /'
  grep -a "domain fault" /tmp/b4/live.out | head -3 | sed 's/^/@@FAULT /'
  echo "@@SMOKE-END live $n rc=$rc"
done
dmesg | grep -a "capstone_vm" | grep -a -i -E "table|cma|dma|fail|refus|loaded" | tail -8
dmesg | grep -a -i "cma" | head -6 | sed 's/^/@@KMSG-CMA /'
cd / && rm -rf /tmp/b4
echo "@@SMOKE-DONE"
echo "@@SPEC-DONE b4-t1a directed"
