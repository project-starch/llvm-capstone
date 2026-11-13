#!/usr/bin/env python3
"""mkvm.py SRC DST: a copy of a native stage whose run.sh prints @@VM0/@@VM1 (vmstat page counters and the module's
fault-ahead counters) beside @@NATIVE-CNT begin/end. SRC is never modified; each insertion is asserted."""
import shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
shutil.copytree(src, dst, symlinks=False)
run = dst / 'stage' / 'run.sh'
t = run.read_text()
vm = ('vms() { echo "@@VM$1 $2 vm=$(grep -E \'^(pgfault|pgalloc_dma32|pgalloc_normal|pgfree|nr_anon_pages|nr_free_pages) \' /proc/vmstat | tr \' \\n\' \':,\') '
      'p=$(cd /sys/module/capstone_vm/parameters && for f in fault_ahead_pages fault_ahead_windows unpinned_faults; do printf "%s:%s," $f "$(cat $f 2>/dev/null)"; done)"; }\n')
anchor = 'cnt() {'
assert t.count(anchor) == 1
t = t.replace(anchor, vm + anchor)
b, e = 'echo "@@NATIVE-CNT $n begin $(cnt)"\n', 'echo "@@NATIVE-CNT $n end $(cnt)"\n'
nb, ne = t.count(b), t.count(e)
assert nb == ne and nb > 0, (nb, ne)
t = t.replace(b, b + 'vms 0 $n\n').replace(e, 'vms 1 $n\n' + e)
run.write_text(t)
print(dst, nb, 'commands')
