# Results

5 non-nested defects in PostgreSQL 17.5's client code, each a C reduction
driving the real upstream function, measured on three arms on 2026-10-06.

| arm | detected | scored |
|---|---:|---:|
| `spatial` (base Capstone) | **5** | 5 |
| ~~`sublet`~~ (dropped: it was level0 under a Sublet label, see below) | -- | -- |
| `cheribsd-revocation` (purecap) | **3** | 5 |

**The two the purecap guest misses are the finding, and they were predicted
before the run.** That guest's malloc rounds capability bounds to a size
class, not to the request: 400 bytes granted 448, 4 bytes granted 16. Case 01
reads offset 400 of a 400-byte object and case 03 offset 4 of a 4-byte one, so
both stay inside the granted bound and neither faults. level0, which both
Capstone arms link, narrows a returned pointer to the bytes asked for, and
both are reported there. A malloc-granular machine sees an overrun when it
exceeds the ALLOCATOR'S ROUNDING, not when it exceeds the object.

**The `sublet` row is withdrawn** (the corpus README, "The arm recorded as `sublet` on
2026-10-06 was not the Sublet heap"). Its SDK kept the default level0 heap, so its five
catches, at the same pcs as `spatial`, were level0 bounds under a Sublet label. The run
stays in `sublet-20261006-161014` as the record of what was measured; no verdict is taken
from it. This paragraph used to call it "a measurement and not a relabelled copy" -- the
images did differ, but the configuration under test did not. The protected-allocator
column for these programs is `virtual-malloc` (5 of 5, `2026-10-10-virtual/`).

## What produced these numbers

| arm | runner | platform |
|---|---|---|
| `spatial`, `sublet` | `shared/run-arm.py` over `shared/build-domain.sh`'s images | Capstone application VM |
| `cheribsd-revocation` | `shared/run-cheribsd.sh` over `shared/build-cheri.sh`'s binaries | CheriBSD 15.0-CURRENT riscv64-purecap, QEMU |

The purecap run carries a real positive control: case 00 writes 4096 bytes
into an 8-byte malloc'd object and must raise SIGPROT on a guest that enforces
bounds. Its first attempt stopped there with exit 127 -- not a guest that had
stopped enforcing, but a runner passing ssh's `-n` to scp, so the binary never
arrived. The control refused to score five cases rather than record them
silent, which is what it is for.
