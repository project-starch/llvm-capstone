# memcached fixture 18's precondition is no longer met; fixture 26 (fx18b) catches the stale write, 2026-11-25

Fixture 26 was pre-registered at 66dcaa6e43f9 (ports/memcached-fx18b), before its first boot. Run: run-arm.sh's
memcached gate, `--mode off`, on the dev-pinned ABI-7 kit of capstone/tests/c5-driver-defaults (QEMU ea57497908,
32 supervisor slots). The images were built from dev e12fee7a561b plus that commit: safety image
c1b705adf97e4d03, server image 95790a3016cdb695.

| fixture | prediction (sublet rows) | read | checks |
|---|---|---|---|
| 17 | FAULT temporal | FAULT temporal | 3/3 |
| 18 | FAULT temporal | RETURN 12e0018, "storage NOT reissued: triggering condition not created" | 1/3 |
| 26 | FAULT temporal | FAULT temporal: `same-address=1 attempts=7`, then `domain fault cause=25 ... address=0x3f804812a0`, the printed target (watcher 0x3f80481280 + 32) | 3/3 |

The gate's FAIL set is fx18_prediction, fx18_trigger and fx24_trigger. Fixture 24 is unchanged from its records.

## Reading

- **Fixture 18 depends on placement.** It needs the first calloc(1, 64) after the free to take the freed storage.
  mallocng cycles slot offsets and does not promise that; fixture 17 has searched for the reissue since adc891bb920c
  (2026-10-08) for that reason. Fixture 18 got immediate reuse on 2026-10-13, but not on the 2026-11-11 record's
  base revision or since. So its precondition is no longer met; what moved the slot cycle for this class is not
  shown.
- **This write is caught.** Once the reissue is created (fixture 26, the seventh calloc), the write through the dead
  pointer faults temporal at its exact target. That is shown for this fixture's shape only.
- **Whether fixture 18's change hides a defect is NOT shown.** "No defect" was written here first (9a328c4e1e85) and
  is withdrawn: it was a ruling, not a reading. Fixture 18 stays byte-identical and keeps its prediction and reading.
- **The cause is not bisected**, by the semantics owner's ruling, because fixture 26 passes. Six candidate commits in
  that range change placement:
  - 0d29c79b85e7 (reuse a released group's mapping for the next group of the same length);
  - d43089b43aa8 (one record lookup per call, calloc skips zeroing a clean slot);
  - d8e7fc417637 (slot records out of the mallocng meta);
  - 3702f61342ee (64 KiB regions shared among small groups);
  - de7cdacda737 (a slot's header is the granule of its whole stride);
  - b36cb221a3a2 (minimal mallocng, #301).
  Any of them can legitimately move the cycle.
