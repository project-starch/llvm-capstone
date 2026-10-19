# srevroot-zero: the QEMU gate (pre-registered in PREREG-srevroot-zero.md), 2026-10-20

- **Tier 1, two-sided in one session** (capstone-qemu 7a4f3a64776f0c88, svadu=false; the v9-kit images):
  - control module a3c788ff13d77cca (code-identical to a5dec6a32c49b3d3; banner "diag arena-pin-1");
  - srz module 2fcef16d09e81269 (banner "diag srevroot-zero-1");
  - both read PASS 26/41 with the same 15 FALSE checks, the same 22 exits in order, banner 2, 'wrote no event record' 0
    and 'collect skipped' 2.
  - The comparison script's nonzero status came from a diff of the run-path `file=` line only.
- **The 15-arm chain and mpt** on kit qkit7-srz (qkit7-rev2 with only the module swapped) against gate6-t3-rev2: 16
  arms, 0 differ, 0 no-data (verdicts, faults, signatures, refused lines). All 15 chain consoles carry the srz banner.
- QEMU does not model the trackers, so equality is the prediction. The board pair (sql0, the old module against srz,
  DRT1 17 down to 0) is the test of the mechanism.
