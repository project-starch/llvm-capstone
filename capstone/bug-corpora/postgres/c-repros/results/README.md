# Results

5 non-nested defects in PostgreSQL 17.5's client code, each a C reduction
driving the real upstream function, measured on three arms on 2026-10-06.

| arm | detected | scored |
|---|---:|---:|
| `spatial` (base Capstone) | **5** | 5 |
| `sublet` (Capstone + Sublet) | **5** | 5 |
| `cheribsd-revocation` (purecap) | **3** | 5 |

**The two the purecap guest misses are the finding, and they were predicted
before the run.** That guest's malloc rounds capability bounds to a size
class, not to the request: 400 bytes granted 448, 4 bytes granted 16. Case 01
reads offset 400 of a 400-byte object and case 03 offset 4 of a 4-byte one, so
both stay inside the granted bound and neither faults. level0, which both
Capstone arms link, narrows a returned pointer to the bytes asked for, and
both are reported there. A malloc-granular machine sees an overrun when it
exceeds the ALLOCATOR'S ROUNDING, not when it exceeds the object.

`sublet` matches `spatial` exactly, and that is the expected result rather
than a missing measurement: these five have no application allocator for
Sublet to protect, so its discipline has nothing extra to enforce. The two
arms' images differ, so this is a measurement and not a relabelled copy.

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
