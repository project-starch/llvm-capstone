# Bitstream 1 (P1 + the R-76 ccsrrw term): full build, board lane pre-registration

Status: pre-registered 2026-11-20, before the hash is handed over and before any launch. Brief: post-submission
programme, hardware line A, row 1, and section 7.2 item 3 (spec-silicon 7b0e8cd4b52b).

## The tree, read at source

The candidate is capstone-ariane 9e801ab5e (origin/p1-ccsrrw-priv). Its parents back to P1 are fb2299b96 and 1c26498c1.
P1 d83d3a633 is an ancestor. `git diff d83d3a633 9e801ab5e`:
- `core/csr_regfile.sv`, +9 lines. This is the only RTL in the diff;
- three files under `verif/`: two directed tests (sup-ccsrrw-low.S, u-ccsrrw.S) and testlist_sup.yaml. None of them
  is synthesized.

The added condition is token-for-token the R-76 term of 7180a83b6 (B5a + R-76), which has already been synthesized,
routed and sealed:

    if (CVA6Cfg.CAPSTONE_EXT && ccsr_en && (priv_lvl_o != riscv::PRIV_LVL_M) && !sup_active_q
        && !(CVA6Cfg.DebugEn && debug_mode_q))
      privilege_violation = 1'b1;

Only its comment differs (9e801ab5e names the supervised-below-M case). Every input is already an input of
`privilege_violation`, so no new signal enters that cone, which is on the standing UNOPTFLAT list.

**The hash that is built** is the one the rtl lane hands over with its lint and sweep numbers. If it is not 9e801ab5e,
`git diff 9e801ab5e <hash> -- core corev_apu` is read before the launch, and any RTL change beyond comments sends this
pre-registration back for amendment before anything runs.

## What the term cost where it was built

7180a83b6 against 6e82d7bd2 (B5a alone), both full builds on the same flow:

| | B5a 6e82d7bd2 | B5a + R-76 7180a83b6 | Δ |
|---|---|---|---|
| placed LUT | 190,697 | 189,920 | -777 |
| placed FF | 77,373 | 77,368 | -5 |
| slices | 50,801 | 50,858 | +57 |

The term's cost is below the per-build redistribution floor, so it is read as none.

## Predictions (against P1's seal, d83d3a633: 187,368 LUT, 76,543 FF, 50,800 slices, WNS -13.173)

Graded with `graders/grade_full.py` against P1's seal:
- placer congestion ≤ 5; check_timing loops 0; routed clean; WNS inside P1's ±3.74 null band;
- reported: placed LUT 187,368 ± 2,000 (the top band); placed FF 76,543 ± 300; slices about 50,800 (99.7 %).

The in-flight loops-only gate runs against P1's own synthesis-only gate. The whole-run memory peak is inside the 40 GB
guard (P1's run peaked at 27.39 GB).

**Failure branch.** If placement or routing fails at P1's own utilisation, no retry and no flow or strategy change. The
failure is recorded with its phase and returned to the rtl and runtime lanes. The tree is P1 plus one term, so a failure
would point to the build's nondeterminism at 99.7 % slices (memory: routability at ~178k LUT is marginal), not to the term.

**Name.** The bitstream is named
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-r76_<hash9>.bit`.
It is sealed with sha256 and handed to the runtime lane, which flashes and runs it. The silicon predictions (u-ccsrrw
refused; gcc, mcf and perl within 0.2 % of P1) are the programme's and the runtime lane's, not this build's.
