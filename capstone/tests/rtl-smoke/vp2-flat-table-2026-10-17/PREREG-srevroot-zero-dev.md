# srevroot-zero on dev's module, unconditional: QEMU gate, pre-registered 2026-10-20 before it runs

The runtime lane ruled it unconditional (2026-10-20). This ports lane module-srevroot-zero ce25d6de7280 onto dev
2ca4dab92724 (the tier-m module, source 3f34f80acc54c9a7). Its first version was a parameter defaulting to on; this
has no parameter. The board test of the mechanism is already read: sql0-srz-treeB, DRT1 0 against 16, N=1.

## Builds (buildroot cross-gcc, rv64g, flat)

| | dev (control, tm) | this change |
|---|---|---|
| source | 3f34f80acc54c9a7 | 528bc48bba1f6315 |
| QEMU kit | 7a2283ac195bfd2f | 057846f74a993748 |
| board | deb6dcf834e335ff | 90bf036f59da1231 |

Board module census, this change:
- 4,666 instructions, 0 RVC, vermagic `6.4.14 SMP riscv`, 0 QEMU-only CSR accesses;
- vp2-silicon-gate (trusted_S) 0 refused / 0 candidates / 0 attributed / 4 conditional / 0 unimp, as dev's;
- srz-check.py: CSRUNV 3, each followed by `csrw 0x5c1, zero`; 9 such writes in all, including restore_root's
  inlined sites.

Control: dev's build reads CSRUNV 3, followed 0, writes 0, so the check can read either way.

## Predictions

QEMU's switcher already restores the caller's srevroot, so the write is a no-op there, and equality is the
prediction.

1. **Tier 1,** two-sided in one session (tm-kit images, QEMU 7a4f3a64776f0c88, svadu=false): both modules read
   30/45 with the same 15 FALSE and the same 24 exits in order; banner 2, 'wrote no event record' 0, 'collect
   skipped' 2. The banner reads `diag srevroot-zero-2` on this module, and `diag arena-pin-1` on dev's.
2. **The 15-arm chain and mpt** on kit qkit7-srzdev (qkit7-tm with only the module swapped) against gate6-t3-tm:
   - every arm's images equal;
   - every case equal (compare-gate.py), refused lines 0;
   - mpt equal to gate6-t3-tm's t3-mpt-tm.
