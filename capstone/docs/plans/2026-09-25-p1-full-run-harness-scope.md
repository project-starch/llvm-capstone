# P1 full-run harness: scope (2026-09-25)

**What this is:** the requirements for the harness P1 needs on the new platform of record
(`caplifive_r42_6cbdaeeb4.bit`, once acceptance passes). It is written by the paper lane for the
board lane. **It is not a design and not an implementation.** The source rules are
`paper-nested-allocators/experiments/protocols/hardware/P1-application-cost.md` (steps 5-7 and
Procedure 4-6) and `METHODS.md`. Where this file and the protocol disagree, the protocol wins.

## What exists (read from source on `origin/dev`)

| Piece | Where | What it covers |
|---|---|---|
| Inner timer | `ports/sqlite/speedtest1_measure.c`, the `c0 = rd_mcycle(); rc = main(argc, argv); c1 = rd_mcycle();` bracket | speedtest1's `main()` only. `sqlite3_config(SQLITE_CONFIG_HEAP, ...)` and `sqlite3_initialize()` run BEFORE `c0`. The line is `SPEEDTEST1-CYCLES`. |
| Pool/table release | `ports/sqlite/sqlite_host.c`, `release_region(pool_region)` / `release_region(tables_region)`, reported as `SQ: released pool rc=` | Host-side and untimed. This is where the monitor revokes the Sublet pool, so the deferred drain happens here, outside every current bracket. |
| Reclaim evidence | `RCLM` report, `sqlite_host.c` (the two-share comment block) | A count, not a time. |
| Arms on silicon | measurements doc §7s: ⑤ custom-spatial (memsys5 + lookaside, 2 MiB **static** heap), ⑥ custom-sublet (`--arena` grant from a region), native warm baseline | -O0 and -O2 at size 1 only. The old E2 images are gone (board lane, 2026-09-25). |
| Counter access | domain: `mcycle` (M-mode). Linux userspace: `cycle` (0xC00) is readable; the native baseline measures with it on the board (§7s `BASELINE-WARM CYCLES`) | So the host can time the release itself. |
| CMA on the board | `caplifive-buildroot/configs/caplifive.dts`, `linux,cma@ac000000`, 256 MiB, `linux,cma-default` (`8c530f7`, 2026-09-11) | Capability **regions** above 4 MiB. **Not** the domain block: `modcapstone` still takes that from `__get_free_pages`, so it is capped at 4 MiB (`capstone.c`, the `dom_pages_log2` allocation). |

## What P1 requires that does not exist yet

1. **Outer timer (T_run).**
   - Starts before the allocator is configured, i.e. before `sqlite3_config(SQLITE_CONFIG_HEAP, ...)`.
   - Ends after `sqlite3_shutdown()` and the domain's own cleanup.
   - Emitted as its own line. `SPEEDTEST1-CYCLES` stays as the phase figure for continuity with §7s.
2. **Deferred drain (T_drain), reported separately AND inside the charged total** (P1 Procedure 5).
   - On this port the drain is the host's `release_region` of the pool and tables.
   - Bracket both calls with the userspace `cycle` counter, in every arm. The spatial arm also
     releases regions, so the ioctl transport cancels in the ratio and does not need excluding.
   - Charged total: T = T_run + T_drain, reported as two numbers plus their sum.
3. **Completion condition for the drain.** The drain counts as complete only when:
   - `released pool rc=1` and `released tables rc=1` are both seen;
   - the RCLM delta is the one pre-registered for that arm.

   A run missing either one is a failed run, not a short one.
4. **Five arms** (P1 Design): custom-sublet ⑥, custom-spatial ⑤, system-sublet, system-spatial,
   and custom-plain as a reference only.
   - No system-sublet or system-spatial adapter exists: a case-insensitive search of `capstone/`
     on `origin/dev` finds the terms only in `docs/plans/2026-09-14-sublet-paper-board-experiments.md`.
   - Without system-sublet, the protected-advantage headline is dropped and only `protection_cost` is reported (P1 "Completion").
5. **Capacity pilot at -O2** over sizes 100, 30, 10, 3, 1, starting at a 128 MiB arena. Every rejected candidate is logged.
   Two limits the pilot will hit:
   - **⑥'s tables grow with the pool.** They take about 41 B per 64 B atom (§7s formula), so a
     128 MiB pool needs about 86 MB of tables: 214 MB of regions against a 256 MiB CMA area.
   - **⑤'s heap is static, so it lives in the domain block, which is capped at 4 MiB.** That caps
     the matched arena at about 2 MiB, and so the workload at about size 1, unless either:
     - ⑤ takes its heap from a region, as ⑥ does; or
     - the domain block moves to CMA. That module change is already planned by the helper lane
       for tshark, as its own `dev` commit.

     This is inferred from the image layout, not measured. **It decides whether any size above 1 can be matched.**
6. **Schedule:** 5 independent runs per arm over at least 3 boots, in the shared seeded order, with
   no workload warm-up (P1 step 7).
   - A fresh boot for each Sublet repetition applies only to an unreclaimed configuration.
   - R-42 carries the reclaimer, so confirm from its acceptance run before dropping that rule.
7. **QEMU validation of every bracket before any board run** (P1 step 6). Predictions for each arm's
   tokens (hash, HEAP, sublet counts, rc, RCLM delta) are pushed before the first boot.
8. **R1 on R-42 in the same boots:** the warm and cold fixture, the byte positive control and the
   four nulls. The current R1 numbers are on `caplifive_m1_054cea69b.bit`, and H1 names
   `r30r31_1bfff7776`.

## Open for the lead

- **Replay or complete run?** The paper's text (`sections/evaluation/04-hardware-cost.tex`,
  "Complete allocator replays") says trace replays of the allocator binaries. The P1 protocol says
  the complete speedtest1 run. This scope follows the protocol. Only the lead can reconcile the two.
- **`H1-platform.md` still names `r30r31_1bfff7776`.** Updating it is a paper-repo edit.
