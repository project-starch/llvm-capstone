# srevroot-zero on dev's module, unconditional: the QEMU gate, read against PREREG-srevroot-zero-dev.md

- **Tier 1, two-sided in one session** (tm-kit images, QEMU 7a4f3a64776f0c88, svadu=false):
  - dev's module 7a2283ac195bfd2f (banner `diag arena-pin-1`) and this change 057846f74a993748 (banner
    `diag srevroot-zero-2`) both read 30/45;
  - the same 15 FALSE and the same 24 exits in order;
  - banner 2, 'wrote no event record' 0, 'collect skipped' 2.
- **The 15-arm chain** on kit qkit7-srzdev (qkit7-tm with only the module swapped) against gate6-t3-tm:
  - every arm's images equal;
  - 15 arms, 0 differ, 0 no-data (compare-gate.py: verdicts, faults, signatures), refused lines 0;
  - all 16 chain and mpt consoles carry the srevroot-zero-2 banner, and none the old one.
- **mpt** against gate6-t3-tm's t3-mpt-tm: equal, CAUGHT 3.

Every prediction holds. QEMU does not model the trackers, so equality is all it can show. The mechanism's test is the
board's: sql0-srz-treeB read DRT1 0 against 16, with the parameter version of this change. Board module for this
version's own Tier 1 boot: 90bf036f59da1231. No corpus kit uses it before that boot.
