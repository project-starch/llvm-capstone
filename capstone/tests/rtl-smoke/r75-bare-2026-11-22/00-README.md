# r75-bare-2026-11-22: a silicon reproducer for R-75 (bitstream 1 against R-75 option (d))

Bare M-mode images (load at 0x80000000), the silchk-b1-2026-11-21 kit's wrapper. R-75: after a trap into M from S while
the lifetime namespace changes, the LSU's DEAD record for the caller's physical PC id is consumed in the cycle before
the registered flush, and the M handler re-faults at its own entry for ever (ISSUES R-75). `PREREG.md` holds the
vectors, the grading and the board procedure. `build.sh` builds every image; `active-lines.py` is the #if evaluator
its register choice uses. Sibling records: the fix's record docs/plans/2026-11-22-r75-option-d-on-bitstream-1.md.
