# Memory-behaviour driver, CPython 3.13.7 row of
# capstone/docs/plans/2026-10-11-protected-memory-behaviour.md.
#
# Run in the guest as
#   PYTHONHOME=/mnt/vm/cpy ./capstone-vexec ./cpython.dom -S /mnt/vm/mem_driver.py
# Lines 1-3 of the workload are the run-ports.py JSON/GC script, statement for
# statement. Nothing else allocates before sys._debugmallocstats(); every
# reading taken afterwards (VmHWM, bytecode check, module check) is printed
# after the allocator statistics and cannot change them.
import json,gc,sys
a=[{"n":n} for n in range(1000)]; assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")
sys.stdout.flush()
sys.stderr.write("MALLOCSTATS_BEGIN\n"); sys.stderr.flush()
sys._debugmallocstats()
sys.stderr.write("MALLOCSTATS_END\n"); sys.stderr.flush()

# Peak resident memory of this process, read just before exit. The domain runs
# inside the capstone-vexec process, so /proc/self is that Linux process.
try:
    with open('/proc/self/status') as f:
        for line in f:
            if line.split(':')[0] in ('VmPeak', 'VmSize', 'VmHWM', 'VmRSS'):
                print('PROC_STATUS', line.strip())
except OSError as e:
    print('PROC_STATUS unreadable', repr(e))

# The staged bytecode must be the one used: a pyc whose recorded source mtime
# and size do not match would be recompiled in memory (the disk is read-only).
import importlib.util, os
for mod in ('json', 'json.decoder', 'json.encoder', 'json.scanner', 'encodings'):
    m = sys.modules.get(mod)
    src = getattr(m, '__file__', None)
    if src is None:
        print('PYC', mod, 'not-loaded-from-file'); continue
    pyc = importlib.util.cache_from_source(src)
    try:
        with open(pyc, 'rb') as f:
            head = f.read(16)
        st = os.stat(src)
        ok = (int.from_bytes(head[4:8], 'little') == 0 and
              int.from_bytes(head[8:12], 'little') == int(st.st_mtime) & 0xFFFFFFFF and
              int.from_bytes(head[12:16], 'little') == st.st_size & 0xFFFFFFFF)
        print('PYC', mod, 'valid' if ok else 'STALE')
    except OSError as e:
        print('PYC', mod, 'missing', repr(e))
print('SYSCONFIGDATA_LOADED', any(k.startswith('_sysconfigdata') for k in sys.modules))
print('DRIVER_DONE')
