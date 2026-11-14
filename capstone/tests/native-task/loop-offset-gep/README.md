# Capability loop addressing, per-group byte form: directed test for `-lsr-capability-byte-iv-fallback`

The gate for the hidden LSR option `-lsr-capability-byte-iv-fallback` (lane branch capstone-loop-offset-gep; off by
default). LSR's `reduceCapabilityIndexGroups` gives a capability address group a capability recurrence when its
anchor is accessed on every iteration and otherwise leaves it unreduced. `-lsr-capability-byte-iv` gives every group
the shared integer byte IV instead. The fallback chooses per group: the recurrence where the proof holds, the byte
form where it fails (perl's S_regcppop: a conditional access, and one through a constant-offset GEP of the anchor).
See docs/design/capability-loop-addressing.md.

**The constraint it must keep:** a capability whose cursor leaves its representable window loses its bounds. On
silicon that happens at a bounds-deriving op or an access (R-60); on the QEMU twin, at the move itself. So no
rewrite may form a cursor the program did not form.

`lgep.c`: loops of the shapes the groups take, each run six times over a 24 MiB heap object, with the cursor at both
extremes: one past the end, and indices counting down to it. `shifted` indexes `a[i - K]` with K = 3,000,000: a
hoisted base there would be the object's base minus 72 MB. A parent spawns one child per case.

**What the twin catches.** qemu55 (compact bounds) writes a CINCOFFSET result UNTAGGED when its cursor is outside the
source's window (op_helper.c `capstone_set_moved_cap`), so any out-of-window intermediate faults at its next use. This
is not a preemption test, and does not need to be one. The twin is stricter than silicon, where a transit with no
access self-heals.

**The positive control:** `ctl-out` moves a capability 256 MiB below the object and back, then stores through it. It
must die with cause 24, or the run is VOID. Its pair, `ctl-in`, makes the same round trip by 12 MiB, inside the
object, and must return.

**The judge:** `judge-lgep.py OFF_RUN ON_RUN`, local to the build host (perf/lgep). It requires:
- every loop case to exit 0 in both builds, with equal checksums;
- ctl-in to return;
- ctl-out to die with cause 24 in both.
