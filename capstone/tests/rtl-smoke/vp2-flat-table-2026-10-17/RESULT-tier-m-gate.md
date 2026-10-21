# The tier-m gate: result, read against PREREGISTRATION-tier-m-gate.md

Pre-registered at lane tier-m-gate f83d781f285d before any run. Tree: integration/pr-tier-m 5fae9ed2848e. Kits
qkit7-tm (one slot) and qkit7-tms2 (two slots): QEMU 7a4f3a64776f0c88, module 7a2283ac195bfd2f, launcher
ab7880387be9eb6a. Images: tm-kit, built with the corpus SDK. Result lines only; logs stay outside the repo.

| step | reading | prediction |
|---|---|---|
| 1 counters | baseline (0,0,0); no context (0,0,0); contract `ok` (426,536,0) against steps=537, faults=426 | holds |
| 2 frame tests (two slots) | multipage -22; twice first=1, second -16, child_ran=1; copy cause 25; overlap PASS; 0 halts | holds |
| 3 step hooks (tmtest c5ccd49b5a6bc0ae) | 210 PASS, 0 FAIL: no_contract_failure, malloc16_ok_exit and malloc64_ok_exit pass; launcher_first reported | holds |
| 4 Tier 1 | 30/45 (see below) | holds |
| 5 chain, 15 arms | every arm's images equal the v6 chain's (verify, 6 to 15 per arm); 15 of 15 arms equal to gate6-t3-v6 case by case (compare-gate.py: verdicts, faults, signatures); refused lines 0 | holds |
| 6 mpt, two-sided | qkit7-v6 and qkit7-tm on the same three images (2126edc5837f, c6c3b4490abe, f76b089115a6): CAUGHT 3 both, verdicts, faults and signatures equal; the runner's SDK-BUILTINS gate read OK | holds |

Tier 1 in detail:
- FALSE: exactly gate6-t1-v6's 15.
- The exits are the v6 gate's 22 in order, with vm-large:0 and vm-mappings:0 after sparse-stale.
- vm_large_contract and vm_mappings_contract pass. These are #242's purpose: growth above 256 MiB, VA above 1 GiB,
  and 1,200 mappings.
- Console banner 2, 'wrote no event record' 0, 'collect skipped' 2.

**Flips:** none. #242's new checks pass, and no existing reading moved.

**compare-gate.py's positive control:** pb12 compared against pb10 reads DIFFER (verdicts, faults, signatures). The
comparison can therefore report a difference, and its equalities above are readings, not defaults.

pb13 exits 75 here as in the v6 gate, because its 9 cases are NO-READING on both, the same 9.
