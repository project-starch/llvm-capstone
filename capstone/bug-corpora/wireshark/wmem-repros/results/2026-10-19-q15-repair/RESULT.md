# wmem patch 0001, the Q-15 repair: results (2026-10-19)

Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md), pushed before any of these runs, with its addendum on what
"pad-masked" means. Comparator `q15-cmp.py`: per cell, verdict, cause, symbol, controls, notes and infra.

Images: the unrepaired ones are wm1's (t3-images-nb, the images of the wm1 prediction and of tree A). The repaired ones
were built from origin/dev a81857dcd6d0 plus this patch: t3-freeze-nb SDK, Debug, the same flags.

Kits:
- `qkit7`: capstone-qemu b290f5656600, module 5bfc38f808e1effd, exact bounds stripped, svadu=false;
- `qkit7-eb`: identical, with exact bounds left on.

| run | images | kit | reading | against | result |
|---|---|---|---|---|---|
| probe | wm1 controls 00, 01 | qkit7-eb | uaf-jumbo prints `case=1 ready`, then cause 25 at `wm_defect_probe` (vaddr 0x215fc) | wm1 on qkit7: 29 at the CDERIVE | the window test passes; the compact 29 is the representability test |
| A | wm1 (unrepaired) | qkit7-eb | CAUGHT 22, NO-READING:unattributed 1 (case 22); controls fault 23/23 each, uaf-jumbo "fault in wm_probe" | pre-registered CAUGHT 23 | MISS on case 22: see the census below |
| B | repaired | qkit7-eb | CAUGHT 22, unattributed 1 | A | 23/23 identical; pad-masked 0 |
| C | repaired | qkit7 | CAUGHT 22, unattributed 1; 0 `cause=29` in the console | B | 23/23 identical |
| D | repaired cases, planted controls | qkit7 | NO-READING:control-failed 23, uaf-jumbo "fault in capstone_cap_derive" 23/23; one 29 line | pre-registered | as predicted: the planted negative fires |
| E | malloc arm (OFF rebuilt, `.text` identical 30/30) | qkit7 | MISSED 23 | t3-pred-wm0 | 23/23 identical |

- The vp2-nested-pools arm reads on the compact profile (run C) for the first time, cell for cell as with exact bounds.
- **Pad-masked count: 0.** The pad report is unmeasured: `Q15_PAD` prints only after a completed case, and every
  nested-pools case faults first.
- **Case 22** faults where it always did: cause 25 at `capstone_cap_revoke_child+88`, the second free's revoke. The
  helper is out of line in these Debug images. Its callers include `wmem_sublet_block_renew`
  ([`case22-caller-census.txt`](case22-caller-census.txt)), so pc attribution cannot reach the declared free. The
  semantics owner ruled no judge change for this repair.
- The board reading of the repaired arm is the next step. Its prediction is run C, cell by cell.

Each run's `verdicts.jsonl`, `inputs.json` and `SOURCE` (the run directory) are in `A-eb-unrepaired/` to
`E-compact-malloc/`.

**C' (addendum 2: the board module's QEMU twin).** The repaired images on qkit7-rev2 (module a5dec6a32c49b3d3, which
never collects, as the board's 39d8492d never does) read 23/23 identical to C. "collect skipped" 0, `cause=29` 0. So C is
the board's prediction for its own module. Bundle `Cprime-compact-repaired-a5dec6a3/`.
