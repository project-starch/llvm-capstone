# pool-repros 0-2: the `backing` arm measured, 2026-10-09

The `backing` arm (revocation only when the buffer-pool port's payload goes back to its backing
allocation, mode 1) had a run for case 3 (probe 39) and only a prose claim for cases 0-2. Probes 36,
37 and 38 were run in mode 1 by `ports/ffmpeg/buffer-pool/security-tests/qemu/run.sh --cases
36,37,38 --modes 1`: **all three complete, as pre-registered (f7adf03a1009)** -- the pool never releases
the buffer to its backing before the stale access.
