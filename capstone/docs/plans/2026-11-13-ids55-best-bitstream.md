# One Capstone bitstream with every optimisation and every needed fix (plan and pre-registration, 2026-11-13)

*RTL lane. The lead's instruction, relayed by the SPEC lane, was: "one Capstone bitstream that carries ALL best
practices and ALL optimisations; every experiment runs on it, never on a weaker one". Hard constraint: option B
(55-bit lifetime ids) + native support, so a descendant of the V1f line 0ef470559. The name is built from its ideas.*

## Content

capstone-ariane `perf/ids55-best`, on `caplifive_ids55-native-tagcache-mlp_b98f5a517`. Ancestry: 0ef470559 ->
79cf8de17 -> b98f5a517; C.5a 1fc2cd5aa and stage-B a481031ca are inside.

| commit | change | class | cycles |
|---|---|---|---|
| 5837b5465, 44fc5524c, d75be647e, 1853bf002 | c5a-cost.S control arms and HPM arms | tests only | none |
| bf2b7f7de | HPM event 23 = `replay_commit`, one pulse per COMMITTED R-43 replay (see the counter's meaning below) | instrument | none |
| 5f41c486e | R-64/R-65/R-66 divide/sqrt fixes, carried from r64-flags 319301b38 with their directed arms | correctness fix (confirmed on silicon at N=2, never synthesised) | none |

**Transplant check for 5f41c486e.**
- `core/cvfpu` is pinned at 3116391 on both e7af48457 and b98f5a517.
- `core/fpu_div_sqrt_fix/` and `core/Flist.cva6` are unchanged between those two commits.
- The three core files apply to b98f5a517 without conflict. Only `verif/tests/testlist_sup.yaml` needed a merge, and
  it kept both sides.

## Classification of every candidate branch (checked per file with `git apply --check -R` against b98f5a517)

| branch | class | in b98f5a517 | action |
|---|---|---|---|
| perf/tag-page-bitmap | optimisation (page bitmap, TWC) + NI-probe superpage fix | yes. The five adapter commits are 79cf8de17's port, and the adapter diff between them is 0 lines. The mmu, load_unit and config hunks are present | nothing |
| ni-superpage | correctness fix (R-68 NI probe PPN), plus the ObjBounds wrappers it branched from | the fix is present | nothing |
| v1f-ids55-native | the ids55 + native line itself | every file present | nothing |
| paircost-b3, hpm-probe-b3 | probe only: their core delta is the `CAPSTONE_NO_LIFETIME_CHECK` wrappers | no | excluded |
| objbounds-b, objb-eval, q04-board | measurement variant: ObjBounds removes the lifetime check (a weaker security level) | no | excluded |
| ovl-check | optimisation, refuted for LDC (0 gain); STC +43 % on bitstream 3, superseded on hits by C.5a | no (conflicts with C.5a's dyn_unit) | excluded |
| r64-flags | correctness fix | no | **folded in** (5f41c486e) |
| fp-fixes, fp-renorm-dual | compress_bounds timing variants: WNS only, no cycle change | no (pre-Sublet ariane_pkg) | excluded: WNS buys no cycles at a fixed clock |

**History sweep.** I searched docs/history, the plans, ISSUES, the measurements doc, and the lane branches
land/rtl-evariants, land/rtl-dyntrim and spec-silicon for built changes meant to cut cycles. Each one built is an
ancestor of b98f5a517 (`merge-base --is-ancestor`):
- c9c7952bd, the DYN handshake trim + LDC rs1 release (stage-B's content);
- 379248185, the R-12 revoke-walk splice (`capstone_rev_node.anvil:21`).

Never built: E1, E3, E4, store-buffer forwarding, lever (a) (STC rs1 claim, -1 cycle per STC), lever (b) (DYN
bypass) and pcc_hold-from-flops. E2 (L1D/DTLB size) is excluded by the lead's fairness rule. Two 2024 commits on a
superseded lineage carry no measurement and are recorded as unresolved: 91fd7e843 (domain switcher) and 7b0f6e45a
(skid buffer).

**E3 (larger verdict cache) is not included.** The mcf model (qemu-rvcm, 8 Nov) gives:
- 64x4 (today): 180.6 M misses;
- 256x4: 172.2 M misses, -4.7 %;
- 1024x4: 59.8 M misses.

Today the tags are 4 ways x 64 x 55 bits of 3-read-port LUTRAM, and the occupancy, node-valid and round-robin state
is 640 flops. By structure, 1024x4 costs about 14 k LUT and 10 k FF on a design at about 180 k LUT, which is at its
routability edge. A BRAM version needs a registered read, i.e. a redesign of the verdict path. That is too costly for
about 2 % of mcf. The replay counter will measure the real miss rate.

## Gates (filled in before the hand-over)

- **Replay counter.** Arms `c5a-cost-*-hpm`, 1,024 accesses each. Base 1853bf002 reads 0 on every arm (the event
  does not exist there). bf2b7f7de reads 1,021 / 1,024 / 1,024 on the miss arms (LDC, STC, plain ld through a
  capability) and 0 on the hit and integer arms. `mhpmevent3` reads back 23 on both. Cycle counts are identical
  between the two trees.
- **Lint: PASS, identical to the base on every counter.** `rtl-lint-gate.sh` on 5f41c486e reads LATCH 44,
  MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2,
  SELRANGE 100, ANVIL_UNOPTFLAT 0. Every counter equals b98f5a517's line (the mlp pre-registration's table) and the
  counter-only tree's paired run (1853bf002 / bf2b7f7de, identical on all eleven).
- **FP arms: identical to r64-flags' own tip.** 14 tests (fp-ieee x6, fp-rm, fp-of, fp-uf, fp-special, fp-carry,
  fp-box, fp-sqrtb, fp-edge) on 5f41c486e (`sim13-wt/.sup/best-arms`) and on r64-flags 319301b38
  (`sim11-wt/.sup/fpref-arms`). All 14 rc 0, 296 printed readings, 0 differences. The arms are known to separate
  the trees: the unfixed core reads them differently (R-64's overflow, R-65's carry, R-66's box; the fixed-tree and
  unfixed vectors in land/rtl-r6466-kit 5adf7a954946 differ). The five HPM arms on 5f41c486e read exactly as on
  bf2b7f7de.
- **95-test sweep: identical to b98f5a517's.**
  - Run `sim13-wt/.sup/swbest`: 5f41c486e, `+TAGMLP_PROBE`, `rtl_dirty=no`, SWEEP-RAN 95, 184 logs.
  - Reference `sim14-wt/.sup/swmlp-var`: 53276815f, synthesis-identical to b98f5a517, the same defines.
  - PROVENANCE is identical apart from date and commit, including every Anvil unit hash.
  - RVFI: 92 compared, 92 identical, 26 parked (prefix only). Exceptions, readings and retired counts: 92 identical.
    Cycles: 0 of 92 differ. The detector fired on neither side.
  - Missing on both sides: rand, perf_tcache_random, perf_riscv_random.
  - The sweep has no FP coverage; the FP arms above cover the FP change.
- **claim-auditor over the core diff: SUPPORTED, with one correction to the counter's wording.**
  - The fix copy is the one compiled. The auditor expanded the FPGA target's file list (276 files): the list has
    `core/fpu_div_sqrt_fix/fpnew_divsqrt_multi.sv` and `norm_div_sqrt_mvp.sv`, and no cvfpu copy of either. A scan
    finds 0 duplicate module definitions; its positive control, appending cvfpu's copy, reports the duplicate.
  - The R-62 swap of the same kind reads `fixnorm=1 cvfpunorm=0` in the add_sources of 17 earlier runs (the
    pre-R-62 run reads the reverse).
  - The copy differs from cvfpu 3116391 in exactly two hunks: the header and the R-66 NaN-box mux.
  - `perf_counters` is instantiated once (`PerfCounterEn = 1`).
  - `replay_commit` reaches only a counter's D input, never the CSR read cone.
- **What event 23 counts: COMMITTED replays, not every verdict miss.**
  - It counts plain load/store replays (`load_store_unit.sv:1419`) and LDC/STC replays (`ex_stage.sv:1524`)
    together.
  - It does not count wrong-path misses flushed before commit, or LDC/STC misses whose probe resolves before the DYN
    result returns.
  - An access that misses again counts twice.
  - The LDC/STC fail-closed path with no verdict record (`$error` under SUP_TRACE, `ex_stage.sv:1528`) did not fire
    in any of the five SUP_TRACE arms.
  - So read it as the replay rate: each replay costs about 12 cycles, and that is the quantity the performance
    question needs.

## Pre-registration of the synthesis (written before any build)

**Name:** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay_<sha>`.

**Synthesis-only gate**, against b98f5a517's synthesis checkpoint on the same host, under the 40 GB guard:
1. Technology-mapping peak in b98f5a517's class (5,114.6 MB). A blow-up fails the gate; it is read for its
   partition, with no retry.
2. `check_timing` loops 0; synthesis "Found timing loop" 0; Synth 8-326 0; LUTLP-1 0 (the S-10b control fires).
3. The fix file is listed once in add_sources, and cvfpu's `fpnew_divsqrt_multi.sv` 0 times.
4. Area, report only, with the floor first. Every module, core and TOP are 0 +/- 2,500 LUT (the floor; TOP is never keyed tighter than a module). The
   change's own estimate is a few hundred LUT at most (the divide/sqrt flag logic and one counter event). Graded post
   hoc; never a stopper.

**Full build** (on a gate pass):
- placed LUT within +/-2,500 of 179,661;
- congestion: placer max <= 5 and post-route router max effective <= 5 (b98f5a517 4 / 5; V1f's placer read 5);
- 0 routing errors;
- 0 loops (`check_timing`, LUTLP-1);
- WNS inside the null band of -13.738, reported as a value only;
- sha256 recorded.
- If routing fails: the readings are placed area, congestion and the census; there is no WNS.

**First boot (the SPEC lane's).**
- No cycle change is expected against b98f5a517 anywhere.
- The replay counter must read non-zero on a verdict-miss workload and 0 on the bare hit arm.
- The fp-r64-r66 bare kit (land/rtl-r6466-kit 5adf7a954946) must read the fixed-tree vectors.
