# sqlite/engine-repros on vp2 silicon, tree A (caplifive_vp2c_8522d72fe, board 1), 2026-10-20

The board bundles of arm q15: vp2-nested-pools, memsys5 with the Q-15 repair (dev f1d522d7fff1 images, SDK t3-freeze-nb).
They are scored cell by cell against the QEMU run C, results/2026-10-20-q15-repair/run-C. The board records, the
pre-registration and every amendment are in tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/README.md (arm q15).

- `vp2-nested-pools/`: subsets d, e and f merged (cases 18-32), giving CAUGHT 14 and MISSED 1. Each subset's own bundle
  sits beside it, as `vp2-nested-pools-q15{d,e,f}/`.
- `vp2-nested-pools-q15f-repeat/`: subset f booted again on the same kit, which reads the same three cells.
- Tree B's subsets a-c are under ../2026-10-20-vp2c-treeB/. The merge refuses mixed platforms, so the two bitstreams are
  two bundles, and combining them is the fold's decision.

## Per-platform note: case 31 (31_fz10_dbstat_decodepage_overflow) is MISSED on vp2 and CAUGHT on QEMU, by R-60

This is a measured hardware deviation, not a model error (docs/ref/ISSUES.md R-60; the ruling is in
docs/design/virtual-capstone/isa.md).
- The faulting access is statDecodePage's `lbu` through a capability whose cursor was moved by a fuzzed offset to
  50,976 bytes past the end of a 20,160-byte buffer, outside its representable window.
- vp2 checks the access against the bounds decoded at that cursor. Those are the true bounds shifted by one 64 KiB span,
  and they contain the cursor, so the load passes and the case completes.
- The diag31 boot printed those shifted bounds. MISSED on three boots, against CAUGHT 28 on QEMU in three runs.
- On QEMU, which keeps exact bounds in registers, the case is CAUGHT with cause 28 at statDecodePage+2024.
- A cell difference of this shape (a far, data-dependent overrun reading MISSED on vp2 only) is R-60's, and is not
  scored as a silicon or model discrepancy.
