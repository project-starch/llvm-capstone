# cheri-perf — CHERI-side temporal-safety overhead microbenchmark

The **CHERI arm** of the QEMU-to-QEMU CHERI-vs-Capstone performance comparison
(`agent-handoff/plans/perf-cheri-vs-capstone-qemu.md`, PI 2026-07-14). It measures
what temporal safety costs on CHERI-RISC-V purecap and puts it opposite the
Capstone arm (`tests/runtime-qemu/revoke-cost-probe/`), which runs the identical
`malloc(64) → touch → free` loop.

See `RESULTS.md` for the numbers and the cross-system reading.

## Files

| file | role |
|------|------|
| `revoke_cost_cheri.c` | the microbench: malloc/touch/free loop, `rdinstret`-bracketed, structured `RCPERF` output. One binary, three revocation configs chosen by the caller. |
| `compile.sh` | build `rc_instret` + `rc_cycle` (counter fallback) + `cheri_status` purecap into the rootfs overlay. |
| `run-in-guest.sh` | runs inside CheriBSD: sets each revocation policy via `sysctl`, runs the bench, prints `RCPERF` lines. |
| `perf-run.py` | boots CheriBSD under CHERI-QEMU (pexpect) and drives the guest script. |
| `parse.py` | parses `RCPERF` lines → per-op instr + overhead vs the spatial baseline. |
| `run.sh` | end-to-end: compile → bake disk image → boot → parse. |

## Method (why it is shaped this way)

- **Same workload as the Capstone arm**, so the marginal cost per `free` is
  comparable across the two QEMU vehicles.
- **`rdinstret`, counting user+kernel.** CHERI revocation is a kernel quarantine
  sweep, so the temporal cost is only visible if kernel retirements are counted.
  `instret` counts all privilege modes; the bracket captures the sweep.
- **Three policies** via the cheri-baseline `sysctl` knobs: spatial (off) /
  temporal (async, the deployed default) / eager (revoke-on-every-free, the config
  that matches our security). See `RESULTS.md`.
- **No `-icount`.** Eager is ~10¹² retired instructions/trial — untractable to
  count one-by-one. Plain TCG + `rdinstret` is reproducible enough (see the trial
  spread in `RESULTS.md`).

## Run

```
bash capstone/tests/cheri-perf/run.sh
# tunables: RC_ITERS (200000), RC_BLOCK (64), RC_TRIALS (3), ICOUNT (0)
```

Requires the CHERI stack at `~/cheri` (SDK, rootfs, `qemu-system-riscv64xcheri`)
and `~/cheri-ws/cheribuild`. Uses CHERI's **own** disk image — no rootfs-lock
contention with the Capstone QEMU suites. `run.sh` auto-creates a dummy `makeinfo`
shim so the bake skips rebuilding the already-built gdb (alternative:
`apt install texinfo`).
