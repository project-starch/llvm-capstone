# The three programs' bug corpus on the FPGA (memcached, FFmpeg, tshark)

*Filed 2026-10-15. The lead's request: the bug corpus for the three benchmarks, which we obtained on virtual
Capstone, is to be obtained on the FPGA. The task goes to the board lane (silicon builds, boots, records) and the
RTL lane (the hardware questions in section 5). This file is the handoff. Decisions marked LEAD are the lead's.*

## 1. What "on the FPGA" can mean: re-measure the corpus on hardware, not replay the virtual images

The virtual-profile images cannot run on this RTL. `virtual-capstone-rtl-feasibility.md`'s verdict is "Partly, and
not as a port". The RTL has no C-mode translation, lifetime tables, CSRUNV/srevroot or per-domain FP. Exact bounds
cannot be met inside the current capability format: "the software stack as landed (ABI v5, local mallocng, all
eight application ports) runs only under that profile".

The route the silicon work already follows is the B0 pipeline:
- port a program to the silicon ABI, as memcached was (B2/B3) and mruby (B5), in
  `b0-silicon-delegated-runtime.md`;
- that means full-LTO gp-captable musl images, soft float, and `capstone-exec` through the process-ABI module, with
  the FPGA monitor stepping the domain by supervised CALL.

So the corpus is re-measured on hardware: each case is rebuilt for the silicon ABI and run on the board. Verdicts are
predicted from the recorded verdicts of the arm with the same mechanism, corrected per case for the silicon
differences in section 3.

## 2. The corpus (dev, 2026-10-15)

There are 131 cases: 130 distinct defects, because FFmpeg plain-heap 13 is `duplicate_of` 02. There are also 11
control cases: memcached `allocator-repros/controls/*` 90-96 and FFmpeg `pool-repros/controls/*` 90-93. Every case
has a FIXED and a BUGGY build, and its verdict is read at a labelled probe (section 4).

| corpus | defects | kind | virtual-malloc | virtual-nested-pools |
|---|---:|---|---|---|
| memcached/plain-heap-repros | 9 | plain spatial | 9 caught | — |
| memcached/plain-temporal-repros | 3 | plain temporal | 3 caught | — |
| memcached/allocator-repros (slabs.c / cache.c) | 9 | nested (5 temporal, 4 spatial) | 8 missed, 1 caught | 8 caught, 05 no reading |
| ffmpeg/plain-heap-repros | 24 | plain spatial | 24 caught | — |
| ffmpeg/plain-temporal-repros | 13 | plain temporal | 13 caught | — |
| ffmpeg/subobject-repros | 9 | plain spatial, inside one struct | 9 missed | — |
| ffmpeg/carved-repros | 13 | nested spatial | 13 missed | 13 caught |
| ffmpeg/pool-repros (AVBufferPool) | 4 | nested temporal | 4 missed | 4 caught |
| ffmpeg/plane-repros | 1 | nested spatial | no virtual arm | — |
| wireshark/plain-heap-repros | 12 | plain spatial | 12 caught | — |
| wireshark/plain-temporal-repros | 10 | plain temporal | 10 caught | — |
| wireshark/wmem-repros | 23 | nested (14 temporal, 9 spatial) | 23 missed | 23 caught |

That is 80 plain and 50 nested defects. The tallies are counted from each `case.json`'s arm verdicts.
- Cite the bundle each corpus's `corpus.json` names in `verdict_bundles`, not a date.
- `tools/derive-verdicts.py --check` must stay current.

Two sets of results already exist.
- **Physical columns.** These are the paper's headline columns (`docs/ref/spatial-vs-temporal-three-programs.md`
  §0): temporal 49 (CHERI 26, Sublet in malloc 26, Sublet in nested 49), spatial 81 (41 / 46 / 71). They were
  measured on physical QEMU over the Sublet heap: per-object bounds and an MREV revoke on every free, with
  `tools/run-capstone-domain.py`, `CAPSTONE_REV_NODES=65536` and `CAPSTONE_GP_NONLIN=1`.
  - That heap and its arm declarations were retired on 2026-10-11 and removed in `7901153f10fa`. The last tree that
    builds them is `7901153f10fa^`. Their verdicts stay recorded (`SCHEMA.md`, arm table).
- **Virtual arms.** These are the table above: `virtual-malloc` and `virtual-nested-pools`, configurations in
  `tools/arms.json`.
  - They use mallocng with managed lifetimes, the ports' nested patches with CDERIVE/CREVOKE, and exact bounds.
  - Runner: `tools/run-virtual-cases.py`, plus wmem's own `shared/run-defects.py` on a persistent `capstone_vm`.

## 3. What differs on silicon (each must enter the per-case prediction)

- **Exact bounds are absent.** Bounds are representable-only, and every capability rounds OUTWARD. A probe that
  lands inside a child's slack reads MISSED on silicon where virtual read CAUGHT. The RTL lane's model gives both
  slacks per object (next bullet). Every object below 4 KiB at a 16-byte-aligned cursor is exact on this encoder, so
  the small-object cases carry over unchanged.
- **CDERIVE rounds outward.** `caplifive_sublet_d93a8d66b.bit` (sha256 89ba6d81…, CDERIVE/CREVOKE, the node parent
  link) was flashed on 2026-10-15 at 23:54, per the board lane. On it, CDERIVE rounds a child's bounds outward where
  physical QEMU refuses with 29 (ISSUES Q-15). A nested child can therefore cover more than its object.
  - The RTL does not refuse a child whose rounded bounds leave the parent.
  - The model is `bounds-model.py` in capstone-ariane `capstone-sublet` (`verif/tests/custom/capstone`). It takes
    `cderive_child(cursor, size, parent)`, or on the command line `python3 bounds-model.py cderive 0xCURSOR 0xSIZE
    [0xPBASE 0xPTOP]`, and returns base, top, slack_below, slack_above, granule, exact, pad_size, pad_align and
    exceeds_parent. It uses the RTL's own encoder.
  - Example: malloc(65535) at a +0x10 cursor decodes [cursor-16, end+113) with a 128-byte granule. It is exact at
    65536 B on 128-byte alignment.
- **A stale access traps with 25.** This is the RTL lane's directed test `stale-access.S` on `d93a8d66b`'s RTL
  (M-mode LSU path, 2026-10-16). A load and a store through a revoked capability each raise 25, in three paths:
  - a CDERIVE child after CREVOKE;
  - a LINEAR piece after REVOKE of its MREV handle;
  - a child parked with STC and reloaded with LDC after the kill. The LDC itself does not fault; the use does.

  Exactly the six probed accesses trapped, and no stale store landed. So R-35's fix holds, and the plain-temporal
  `case.json` disclaimer that "deployed silicon lets such a data access retire" was stale, in simulation for this
  bitstream and the FP respin alike.
  - **Read on silicon on 2026-10-16:** `caplifive_sublet_d93a8d66b`, bare image c0c635e5acf09bfc, the
    pre-registered vector exactly (`0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6`).
  - The 26 `case.json` sentences now say so. The cases themselves have since run on silicon (section 7): 26
    CAUGHT with cause 25 under per-object lifetimes, as predicted.
  - Prediction: where QEMU reads 24 (Q-11, the reload untags), silicon should read 25 (the reload keeps the tag,
    and the use faults).
- **Rounding, not truncation.** R-11 (a top truncated past a 2 MiB window) belonged to the cursorless scheme, which
  capstone-ariane `4a4261bb4` removed. On `d93a8d66b` no capability has a truncated top (RTL lane, 2026-10-16). The
  heap sees over-coverage instead, which is the CDERIVE model's slack: malloc(65535) at a 16-byte cursor covers 113
  bytes past its end.
  - The "3 bytes short at base" reading in `fpga-silicon-measurements-for-paper.md` belongs to `0568f93a9`.
  - ISSUES' R-11 heading predates this.
- **The node pool refuses with 30; it does not deadlock.** Since 2026-09-16 (`r12-pool-exhaust.S`), the 65,533rd
  allocation is refused with cause 30. The base stays valid, and the refusal is idempotent. ISSUES' R-12 heading,
  "DEADLOCKS the core", predates that.
  - Capacity: 65,532 mints from reset with nothing freed.
  - Accounting per instruction (RTL lane): +1 node per MREV, LINEAR SPLIT, CDERIVE and ALLOC. Zero for a NONLIN
    SPLIT, DELIN, TIGHTEN, SHRINK and CINCOFFSET.
  - DROP leaks one node.
  - CREVOKE frees the child. REVOKE frees the run below the handle and keeps the handle.
  - Freed nodes are reused LIFO, up to 16,384 reuses per index.
  - A nested case's peak is its live handles, plus live LINEAR siblings and live children, plus the DROPs so far.
- **Soft float only.** A domain starts with FS Off, and one F/D instruction wedges the board. Every image passes
  `runtime/silicon/no-fp-gate.py`.
- **MOVC (Q-04, C-32): silicon nulls an untagged source.** The silicon-mimic QEMU runs with `CAPSTONE_MOVC_NULL_SCALAR=1`.
- **R-55's carve policy.** Images for the full-scheme bitstreams build with `B0_GLUE_CARVE=pool`; see the board
  lane's current notes in `b0-silicon-delegated-runtime.md`.
- **The firmware image must stay under 32 MB.** The console rejects it after locking the board
  (`tests/preflight-board-run.sh`, C8).

## 4. The oracle on silicon

The oracle is the virtual one (`tools/verdicts.py`), with the fault read from the monitor.
- The FIXED run must print `VERDICT FIXED` and exit 0.
- The BUGGY run is CAUGHT only if its fault lies in the labelled probe (`(^|_)(read|write)_probe…`, `wm_probe`,
  `wm_write_probe`, or the case's `fault_sites`). The fault's cause, epc and tval are reported by the monitor's
  FAULT event (capstone-sbi `1f9aedd`: SUPC/SUPE/SUPT, unconditionally). The epc is resolved with llvm-nm on that
  exact image.
- It is MISSED only with `VERDICT DEFECT-REPRODUCED`, exit 0, and the arm's controls behaving as `arms.json`
  declares, in the same boot.
- A result is cited by image hash and bitstream (CLAUDE.md: "Cite a board result by its image hash").
- The arm gets a new name in `SCHEMA.md`'s arm table (for example `silicon-malloc` and `silicon-nested`), so a silicon
  verdict is never mistaken for a QEMU one.

## 5. Asks

**RTL lane.** All four asks were answered on 2026-10-16: stale access, the CDERIVE model, R-11 and R-12. The
answers are in section 3. The stale-access arm was confirmed on silicon on 2026-10-16.

**Board lane.**
1. The plain corpora first (80 defects). Each case is a small `case.c` + `shared/driver.c` + probe harness, not an
   application port. Build them through the B0 pipeline (`runtime/silicon/build-b0-hello.sh` as the template;
   gp-captable, full LTO, soft float, `check-repr`, `no-fp-gate.py`), with the allocator of LEAD decision (b).
2. Pass every image on the silicon-mimic physical QEMU first:
   - `CAPSTONE_GP_FABRICATE=0` and `CAPSTONE_MOVC_NULL_SCALAR=1`;
   - for CDERIVE/CREVOKE images, the physical decode now landing from capstone-qemu `physical-sublet-decode`.
3. Batch many cases per boot: ascending, a known-good control first, at most one expected wedge and that one last,
   firmware under 32 MB (CLAUDE.md, "BATCH VARIANTS").
4. The nested corpora next (50 defects). They need the ports' nested allocator patches built for the silicon ABI:
   memcached 0006, FFmpeg buffer-pool 0003, wmem 0001, and the carved `FFC_SUBLET_CARVE`.
   - That is compiler-lane territory where gp-captable or LTO breaks (I-8, C-75).

**Both lanes.** A pre-registered per-case prediction table is committed before the first boot. It is computed by a
tool from the recorded verdicts and the RTL lane's models in section 3 (`cderive_child`, the node accounting), never by hand.

**The plain corpora's table is committed (2026-10-16):** `bug-corpora/silicon-predictions/2026-10-16-plain/`, made by
`tools/predict-silicon.py` from geometry measured on the virtual profile (`tools/corpus-geometry.h`).
- **Spatial (45).** Every virtual catch carries over if the silicon allocator derives each child at the request size
  on a 16-byte placement; every object is exact under the encoder.
  - An allocator that rounds the child up to 16 misses five: memcached 00, FFmpeg 01, 04 and 05, and tshark 03.
    These are the same five CHERI recorded NOT CAUGHT. That is consistent, not independent: CheriBSD also rounds,
    measured at 17 and 24 B, inferred at 1 and 9 B.
  - On an 8-byte placement, tshark 00 (8192 B) also depends on placement.
- **Temporal (26).** CAUGHT with cause 25 under an allocator that revokes, and MISSED under level0.
- **Subobject (9).** MISSED.

So decision (b) decides the spatial column too: the request size against the rounded size. The nested corpora wait
for it. (What ran on the board, and what it read, is section 7.)

## 6. LEAD decisions

- **(a) Bounds exactness.** Exact bounds do not exist on this RTL. Either a predicted, pre-registered spatial
  difference is an acceptable outcome, or the silicon allocator must pad so that the cases stay comparable.
  - The RTL lane's input: padding to representable bounds (`pad_size` at `pad_align`) keeps the allocator's own
    bookkeeping exact. It does NOT restore a verdict whose probe falls inside the slack, because the padding is
    inside the child.
  - Its proposal: pad anyway, and pre-register MISSED where the probe offset lies within slack_below or slack_above,
    CAUGHT elsewhere.
- **(b) The silicon allocator for the malloc arms.** Two options: dev's managed mallocng (CDERIVE/CREVOKE, as
  virtual), ported to the silicon ABI; or the silicon Sublet runtime (`runtime/include/sublet/sublet.h`), whose
  `sublet_carve` the board lane is fixing for R-55. The silicon runtime's own allocator is level0 (per-object
  bounds, no temporal safety), so it cannot carry the temporal arms.
  - What ran (section 7): level0, and for the arms that revoke, level0 built with `CAPSTONE_LEVEL0_LIFETIMES`
    (dev `a9405e5bd711`; free reworked in `2c53d5641b72`): every object whose bounds are exact at its placement is a
    CDERIVE child of one group over the arena, and free CREVOKEs it.
- **(c) Order.** The plain corpora first, then the nested ones (proposed). Or both together, now that CDERIVE/CREVOKE
  are on silicon.

## 7. Read on silicon (sets 1-4, 2026-10-16)

The board lane ran four sets on `caplifive_sublet_d93a8d66b`, each pre-registered before its boot:
`bug-corpora/silicon-runs/2026-10-16-plain` (`ffd2a5173d1c`), `-subobject-pool` (`2c53d5641b72`), `-carved`
(`39d21c614605`) and `-nested` (`06f0b29f0463`). `tools/silicon-verdicts.py` re-reads each run from the board transcript, checks every image's
hash against the board matrix, and judges it as the QEMU arms are judged. The verdicts are in each corpus's
`case.json`, from the bundles under `results/2026-10-16-silicon/`. Every row read as pre-registered (set 4: every outcome and site; one cause differs, below).

| corpus | rows | `silicon-level0` (bounds only) | revoking arm | further arm |
|---|---:|---|---|---|
| plain-heap, 3 programs | 46 (45 + FFmpeg 13, a duplicate) | 46 CAUGHT, cause 28 | `silicon-lifetimes`: 46 CAUGHT, cause 28 | |
| plain-temporal, 3 programs | 26 | 26 MISSED | `silicon-lifetimes`: 26 CAUGHT, cause 25 | |
| ffmpeg/subobject-repros | 9 | 9 MISSED | `silicon-lifetimes`: 9 MISSED | `silicon-subobject` (member bounds): 7 CAUGHT, 2 MISSED |
| ffmpeg/pool-repros | 4 | | `silicon-pool-sublet` (protected pools): 4 CAUGHT, cause 25 | `silicon-pool-stock`: 4 MISSED |
| ffmpeg/carved-repros | 13 | 13 MISSED | `silicon-carve-sublet` (CDERIVE carve): 13 CAUGHT, cause 28 | `silicon-carve-bounds` (SHRINK carve): 13 CAUGHT, cause 28 |
| memcached/allocator-repros | 9 | | `silicon-mc-sublet` (patch 0006): 00-04 CAUGHT, cause 25; 06-07 CAUGHT, cause 28; 05 and 08 no reading | `silicon-mc-stock`: 00-07 MISSED; 08 CAUGHT, cause 28 at its declared site |
| ffmpeg/plane-repros | 1 | 1 MISSED | | `silicon-carve-bounds` (SHRINK per plane): CAUGHT, cause 28 |

- Cause 28 is the RTL's OUT_OF_BOUNDS, and 25 its stale access. QEMU reads 5/7 and 24 for the same cases.
- The plain spatial column is section 5's first column, the child at the request size. The five predicted misses
  under rounding to 16 remain predictions.
- **Set 4's heap.** The memcached arms run on level0 as changed in `06f0b29f0463`: under
  `CAPSTONE_LEVEL0_LIFETIMES` a request of 4 KiB or more is placed on the granule of its padded size and derived at
  that size, so its child is exact and base == cursor. Patch 0006's `slabs_take_back` computes
  `gen + (ptr - base(gen))` and needs that; on the earlier placement it faulted in `do_slabs_free` (set 3's QEMU
  stage). Sets 1-3 ran the earlier placement.
- **memcached 05 has no reading on any Sublet build.** Its sentinel byte (`memset(next, 0x5a, 1)`) lands on the next
  item's `sublet_parent`, the first field under patch 0006, so the FIXED run's next free hands CREVOKE an untagged
  operand (24 on silicon, where the rule-based mapping predicted 25: the operand was never a capability).
- **memcached 08 has no reading under patch 0006.** Its 16,400-byte cache.c child rounds up 16 bytes at its 32-byte
  granule, so the all-spaces scan stops in the slack and the case refuses its premise (CONTROL-FAILED 785). That is
  decision (a)'s pre-registered spatial difference. The plane case links 14 units of FFmpeg's libavutil.
- **Not read on silicon as of 2026-10-16:** wmem-repros (23), the board lane's set 5.
