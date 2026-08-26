# MPY-T09 / upstream issue 18168, verbatim from the report.
# Stock behaviour at pin 2e3304a: no crash. The defect is present anyway:
# the view is left pointing at an orphaned buffer and stays writable.
# See stale-view-proof.py for the measurement that establishes that.
import gc
ba = bytearray(b"abcdefghij")
views = [memoryview(ba) for _ in range(4)]
ba[:] = ba + b"X"*256
gc.collect()
for i, mv in enumerate(views):
    mv[0:1] = b"Y"
print("SURVIVED")
