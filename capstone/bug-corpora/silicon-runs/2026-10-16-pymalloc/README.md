# CPython's pymalloc corpus on silicon: pre-registration (2026-10-16, before the boot)

The board campaign's wave 4 (`docs/plans/2026-10-16-board-campaign-all.md`, §3, cpython/pymalloc-repros).
- Twenty consumer-side use-after-free defects in CPython 3.13.7, every reachable one at the pin. Their memory comes
  from pymalloc, so no `free()` reaches `malloc`.
- Each is the port's hosted replay (`ports/cpython/pymalloc/src/native/main.c`) around the case's `PYC_CASE` body,
  over CPython's own `Objects/obmalloc.c`.

Two arms, one boot on `caplifive_sublet_d93a8d66b`: private image 49c794e1fbd63081, firmware 5049ca43f7595b41, on the exact-regions
supervised-call line of sets 2-8.

| arm | obmalloc.c (the port's `cmake/prepare-source.py` variant) | predicted from |
|---|---|---|
| pym-silicon-pym-stock | `ported`: patches 0001 + 0002, pymalloc as released | `virtual-malloc` (MISSED, all 20) |
| pym-silicon-pym-sublet | `protected`: + patch 0003, every block a CDERIVE child of its arena, CREVOKEd when pymalloc takes it back | `virtual-nested-pools` (CAUGHT, all 20) |

## The build

It mirrors the port's `cmake/Replay.cmake` `defects` target, through the B0 pipeline:
- the replay's `main.c`, the case's `case.c`, the variant's `obmalloc.c` and the port's `src/shared/backing.c`;
- the port's defines (`PYMALLOC_PORT Py_BUILD_CORE WITH_PYMALLOC SIZEOF_SIZE_T=8 SIZEOF_INT=4 SIZEOF_VOID_P=16`),
  gnu11, `-O0` as the port's Debug preset;
- the archive is Python-3.13.7.tgz, sha256 6c9d8083…, as `upstream.json` pins.

**Two adaptations for the silicon runtime:**
- **`runners/silicon/aligned-alloc.c`.** The B0 heap is level0, which has no `aligned_alloc`. musl's own pulls
  oldmalloc into the link beside level0 and fails it on duplicate symbols.
  - The shim calls `malloc` and checks the alignment; a miss returns NULL, which the replay reports as exit 4.
  - The replay's two aligned requests (16 MiB and 64 MiB at 16 KiB) are placed by level0's lifetimes heap on their
    32 KiB and 128 KiB granules.
- **A 104 MiB data region over a 100 MiB level0 arena.**
  - The replay mallocs 88 MiB up front: an 8 MiB input buffer, 16 MiB of metadata and the 64 MiB pymalloc backing.
  - That is set 4's 88 / 84 MiB shape, one 128 MiB region.
  - The port's own sizes are kept (`PYM_ARENA_BYTES`, `PYM_META_BYTES` are not overridden).
  - `check-repr` reads every image OK at all 129 bases, OVERLAP 0.

**The construct is in the images.** Counted by encoding (`.insn r 0x5b, 1, 0x51/0x52`): 4 CDERIVE and 1 CREVOKE in a
stock image, which are level0's lifetimes heap. A Sublet image carries 6 and 2, the extra being patch 0003's derive and
revoke helpers.

**The protocol** (kind `pym`, `tools/run-silicon-cases.py`):
- a run is `<fixture> /tmp/pym-report.bin`;
- the fixture is the virtual runner's own one-event file naming the case (`runners/virtual/run-virtual.py`
  `fixture()`). It is staged beside the images and named by its baked path;
- completion is `PYM completed=1`, printed only after the case body's `read_probe`;
- a fault counts only in `read_probe` (label `pyc_defect_read`). No case declares other fault sites;
- the corpus control `controls/00_control_uaf_block` (one block freed and read through `read_probe`) must complete on
  stock and fault at the probe on Sublet. It gates the arm, beside the tool's four controls.

## The silicon-mimic QEMU reading (capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`)

- **pym-silicon-pym-stock:** all 20 complete (`PYM completed=1`, exit 0). The control completes.
- **pym-silicon-pym-sublet:** all 20 CAUGHT 24 at `read_probe`. The control is CAUGHT 24 at `read_probe`.
- Both arms' tool controls are as declared. Every row matches the recorded virtual verdicts.

## The board prediction

Every row as QEMU read it, by outcome and site, with each 24 reading 25.
- Each Sublet fault is the probe's own load through a block patch 0003 revoked. That is ISSUES Q-11's mechanism: QEMU
  untags the revoked child, and the RTL keeps it tagged and refuses the load with 25.
- The faulting instruction is the access itself, so the site carries over (the set 8 rule).
- The stock arm faults nowhere.

**New on silicon, and what a failure would say:**
- **The largest corpus domains yet:** 128 MiB regions, as set 4's memcached stock arm. A launch failure would read as
  a CMA or region error from capstone-exec, not as a case result.
- **A file read and a file write in every run.** A fixture that cannot be opened returns 2; a report that cannot be
  written returns 5. Either is NO-READING, not a verdict.
- **pymalloc's own arenas on silicon,** and patch 0003's per-block CDERIVE/CREVOKE inside them, for the first time.
