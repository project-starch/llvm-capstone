# Temporal-safety overhead — CHERI side (revoke-cost microbenchmark)

**The CHERI arm of the PI's CHERI-vs-Capstone performance comparison**
(`plans/perf-cheri-vs-capstone-qemu.md`; PI 2026-07-14: "test the perf.
distinction on QEMU-Capstone and QEMU-cheri"). Companion to the Capstone arm
(`tests/runtime-qemu/revoke-cost-probe/RESULTS.md`), which runs the **same**
`malloc(64) → touch → free` loop so the per-op instruction counts are comparable.

**Vehicle:** `qemu-system-riscv64xcheri` 7.1.0 (CTSRD-CHERI), CheriBSD purecap
(`CHERI-PURECAP-QEMU`, INVARIANTS+WITNESS+`CHERI_CAPREVOKE`) — the exact stack the
cheri-baseline security run used (task-015). Ordinary purecap userspace program
(`-O2`), not bare-metal.

**Counting:** `rdinstret`, bracketing the loop. `instret` retires in **all**
privilege modes, so the delta **includes the kernel-side revocation sweep** —
which is the whole point: CHERI pays its temporal cost in a kernel quarantine
sweep, not at the `free`. Confirmed user-readable in CheriBSD (no SIGILL;
`RCPERF-COUNTER ./rc_instret`).

## Workload

`malloc(64) → touch one byte → free()`, `ITERS = 200000`, three revocation
policies set via `sysctl` before each run (the cheri-baseline knobs):

| config | `runtime_revocation_default` | `..._every_free_default` | meaning |
|---|:--:|:--:|---|
| **spatial**  | 0 | 0 | revocation OFF — CHERI spatial safety only (baseline) |
| **temporal** | 1 | 0 | async quarantine — the **realistic deployed default** |
| **eager**    | 1 | 1 | revoke on **every** free — the config that **matches our security** |

Per-op = `(allocfree − empty)/ITERS`; empty is an identical allocation-free
calibration loop (~2.7 instr/iter, subtracted). 3 trials/config, median reported.

## Result (dynamic instruction count, user+kernel — NOT silicon timing)

| config | per-op (instr) | overhead vs spatial | trials (per-op) |
|--------|---------------:|--------------------:|-----------------|
| **spatial** (baseline) | **3,760** | — | 3702 / 3760 / 3768 |
| **temporal** (async)   | **23,977** | **+20,217  (6.38×)** | 24073 / 23977 / 23773 |
| **eager** (every-free) | **14.03 M** | **+14,025,640  (3,731×)** | 14.23M / 13.83M  *(n=2)* |

Trials are tight (spatial/temporal within ~1%; the two eager trials within ~3%),
so despite the kernel-time inclusion the numbers are stable. *Eager is n=2: the
3rd eager trial exceeded the 60-min pexpect boot window — a single eager trial is
~2.8×10¹² retired instructions (200 000 frees × ~14 M/free) under TCG. The two
completed trials agree to 3%, so the median is well-determined.*

## Reading — the money comparison (put opposite the Capstone arm)

The marginal cost of making **one `free` temporally safe**, each system measured on
its own QEMU vehicle:

| system | temporal mechanism | Δ instr / free | character |
|---|---|---:|---|
| **Ours (Capstone)** | revoke-at-free, inline capability op | **+5** | O(1), at the contract point |
| **CHERI async** (default) | quarantine + amortized sweep | **+20,217** | *does not* catch the corpus's temporal bugs at the contract point (see `tab:cheri`) |
| **CHERI eager** | revoke-on-every-free sweep | **+14,000,000** | matches our security; a per-free stop-the-world GC sweep |

- **CHERI eager — the only CHERI config that matches our security — costs ~14 M
  instructions per free**, versus our **+5**. That is the PI's headline
  quantified: at equal temporal security the mechanisms differ by roughly **six
  orders of magnitude per operation**. Eager revocation scans the address space
  for capabilities on every free; our revoke is a single O(1) capability op at the
  point the object dies.
- **CHERI async is ~6.4× over its own baseline** and amortizes the sweep across
  the frees between quarantine flushes (~700× cheaper than eager here) — but by
  its capability mechanism it blocks **0/11** use-after-free rows at the contract
  point (`tab:cheri`); its temporal protection is deferred, not synchronous.
- The two baselines are **not** directly comparable in absolute instr/op (Capstone
  bare-metal domain, 7 instr/op bump vs CheriBSD jemalloc-under-purecap,
  3,760 instr/op) — different vehicles, different allocators, kernel vs none. The
  transferable, apples-to-apples quantity is the **marginal Δ per free** above and
  the **within-vehicle overhead ratio**.

## Caveats / honesty

- **Proxy, not timing.** Dynamic instruction count; QEMU has no pipeline/cache/
  cycle model. The cycle-accurate Capstone number is the FPGA follow-on
  (`tests/rtl-smoke/`). CHERI has no comparable silicon here — that asymmetry is
  why the PI's rule is QEMU-to-QEMU for the comparison and RTL only for the
  Capstone absolute.
- **Includes kernel time — deliberately.** CHERI revocation *is* kernel work;
  counting user+kernel is the only faithful way to price it. `rdinstret` also
  counts timer-interrupt handlers landing in the bracket — a minor contaminant the
  ~1% trial spread shows is small relative to the signal (and negligible against
  eager's 14 M).
- **`-icount` is infeasible for eager.** Eager is ~2.8×10¹² retired instructions
  per trial; counting them one-by-one under `-icount` would take many hours. Plain
  TCG + `rdinstret` is the only tractable way to measure eager, and the trial
  stability shows it is reproducible enough. (spatial/temporal *could* be redone
  under `-icount` for determinism; the shape does not change.)
- **Naive workload.** A tight malloc/free of one size; a real allocator's
  quarantine tuning and a real workload's live-set change absolute sweep cost. The
  qualitative gap (inline O(1) op vs address-space sweep) is structural, not an
  artifact of this microbench.

## Reproduce

```
bash capstone/tests/cheri-perf/run.sh            # compile + bake + boot + parse
RC_ITERS=200000 RC_TRIALS=3 ICOUNT=0 bash .../run.sh
```

Needs the CHERI stack under `~/cheri` (SDK, rootfs, `qemu-system-riscv64xcheri`)
and `~/cheri-ws/cheribuild`. `run.sh` auto-creates a dummy `makeinfo` shim so the
disk-image bake skips rebuilding the already-built gdb (else install `texinfo`).
CHERI uses its **own** `cheribsd-riscv64-purecap.img`, so there is no rootfs-lock
contention with the Capstone arm.
