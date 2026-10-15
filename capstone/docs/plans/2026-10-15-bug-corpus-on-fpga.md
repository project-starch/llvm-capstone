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

- **Exact bounds are absent.** Bounds are representable-only, and the allocator pads to the representable size
  (R-33's rule). A one-past or short-overflow case can read MISSED on silicon where virtual read CAUGHT.
- **CDERIVE rounds outward.** `caplifive_sublet_d93a8d66b.bit` (sha256 89ba6d81…, CDERIVE/CREVOKE, the node parent
  link) was flashed on 2026-10-15 at 23:54, per the board lane. On it, CDERIVE rounds a child's bounds outward where
  physical QEMU refuses with 29 (ISSUES Q-15). A nested child can therefore cover more than its object.
- **Whether a stale access traps.** R-35 (ISSUES-ARCHIVE) is FIXED on the M-mode LSU data path. But every
  plain-temporal `case.json` still says "deployed silicon lets such a data access retire" (citing Q-11). One of the
  two is stale, and every temporal verdict depends on which.
- **R-11, top truncation past a 2 MiB window.** It is worked around in the glue only, and "does not reach the heap":
  a malloc(65535) read 3 bytes short at its base (`fpga-silicon-measurements-for-paper.md`, the r54-bounds section).
- **R-12: the node pool is 65,536, and exhaustion deadlocks the core.** The nested arms derive per object.
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

**RTL lane.**
1. A directed test on `d93a8d66b`'s RTL: a data load and a data store through a capability whose node was revoked,
   on the M-mode LSU path, with the cause each raises. This settles R-35 against the case.json disclaimer before
   any temporal prediction.
2. CDERIVE's outward rounding (Q-15), as a function of size and alignment. Give it as a model the prediction
   tool can call, not as prose.
3. R-11 on the heap: which allocation sizes and placements in the corpus cross a truncating window.
4. R-12 against the nested arms: the node count each nested case reaches, and the margin to 65,536.

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
tool from the recorded verdicts and the models in 5.2-5.4, never by hand (memory: compute every predicted value).

## 6. LEAD decisions

- **(a) Bounds exactness.** Exact bounds do not exist on this RTL. Either a predicted, pre-registered spatial
  difference is an acceptable outcome, or the silicon allocator must pad so that the cases stay comparable.
- **(b) The silicon allocator for the malloc arms.** Two options: dev's managed mallocng (CDERIVE/CREVOKE, as
  virtual), ported to the silicon ABI; or the silicon Sublet runtime (`runtime/include/sublet/sublet.h`), whose
  `sublet_carve` the board lane is fixing for R-55. The silicon runtime's own allocator is level0 (per-object
  bounds, no temporal safety), so it cannot carry the temporal arms.
- **(c) Order.** The plain corpora first, then the nested ones (proposed). Or both together, now that CDERIVE/CREVOKE
  are on silicon.
