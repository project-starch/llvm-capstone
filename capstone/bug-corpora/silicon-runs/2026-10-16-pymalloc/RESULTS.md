# CPython's pymalloc corpus on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 9aa15f58c846, before the boot). One boot, 18:11-18:54:
- firmware 5049ca43f7595b41, Image 49c794e1fbd63081;
- two arms, 50 runs, every one ended by itself (each arm's rung returned 0).

**Every row and every control is as pre-registered,** by outcome, cause and site, on the pre-registered image
(sha256): 42 of 42.

| | pym-silicon-pym-stock (pymalloc as released) | pym-silicon-pym-sublet (patch 0003) |
|---|---|---|
| the 20 cases | 20 complete (`PYM completed=1`, exit 0): every stale access reads freed memory unseen | 20 CAUGHT 25 at `read_probe`, the labelled stale load |
| corpus control 00 (a block freed and read through `read_probe`) | completes | CAUGHT 25 at `read_probe` |
| the tool's four controls | as declared | as declared |

**What it shows:**
- **CPython's own small-object allocator runs on silicon** in a capstone-exec domain, in 128 MiB domains
  (104 MiB of data over a 100 MiB level0 arena). Its memory never reaches `free()`.
- **pymalloc as released misses all twenty defects** on silicon, as on every virtual arm.
- **Patch 0003 refuses all twenty, at the labelled stale load.** Every pymalloc block is a CDERIVE child of its
  arena, CREVOKEd when pymalloc takes it back. Ten shapes are covered:
  - eight free / reuse / stale reads;
  - interior pointers into freed blocks (2, 13);
  - a stale entry in a live array (4);
  - a pointer loaded out of a freed block (5);
  - payload buffers (6, 9);
  - a cursor across two API calls (7);
  - a block ended by a realloc that moved it (10);
  - a free and use on adjacent lines (11);
  - pointers parked with no bound on reuse (14, 18);
  - a bare `PyMem_Malloc` block (15).
- Each refusal is cause 25, where the silicon-mimic emulator reads 24 (ISSUES Q-11: the RTL keeps the revoked child
  tagged and refuses the load).

**Two things first exercised on silicon here:**
- **file-backed runs:** each run read its fixture from the image and wrote its report under `/tmp`;
- **CDERIVE/CREVOKE inside a second allocator nested in a malloc'd arena:** pymalloc's arenas sit inside level0's
  64 MiB block.

**Judged by** `tools/run-silicon-cases.py board --prereg` on the run-scoped transcript, then compared row by row with
the pre-registered outcome, cause and site, keyed by image hash.

**Not established:** N = 1 per case.
