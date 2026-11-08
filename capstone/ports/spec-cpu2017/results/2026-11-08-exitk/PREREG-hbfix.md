# unify gcc train with the kallsyms dump moved past the command: the heartbeat's cost (DRAFT pre-registration)

DRAFT, 8 Nov, before any board use; nothing runs before Wednesday. Why: UNIFY-440S.md beside this file. The board
harness's heartbeat runs `wc -c` and `wc -l` over /tmp/step.out every 30 s; on out-nd3-b3 its children took 704.7 s of
CPU because the stage had printed a 762,587-byte kallsyms dump into that file first.

## What changes: the stage only

- **Stage** perf/exitk/stages/gtr-unify-hbfix = perf/nodiag's `502.gcc_r-train01-capstone-lifetimes-unify-rgz-ndcpu`
  (the stage of out-nd3-b3, with its @@CPU/@@PST snapshots and the 30 s tick probe) with two lines changed: the dump
  goes to /tmp/ksyms.txt before the stage (`@@KSYMS-DEFERRED <n> lines`) and is printed after `@@SPEC-DONE`, between
  the same @@KSYMS-BEGIN/END markers. The command, its parameters and every other line are unchanged.
- **Kit** bkit30-unify, unchanged (module 13a5ad66752a4e0b, launcher a8a4dbe637ac758a). Driver
  perf/exitk/board-hb/un-run-hbfix.sh (results/2026-11-06-unify/un-run.sh with its own list and output root, and the
  pre-registration branch as a parameter); list perf/exitk/board-hb/un-b3.list; dry run perf/exitk/board-hb/dry.
- A side effect, registered: the tick probe no longer runs beside a heartbeat reading 762 KB. Its earlier runs
  (2026-11-06-unify, three boards' readings 0.324-0.325, 54 gaps/s) were VOID because HB 1's `wc` ran during the
  probe (HB 1's tail ends in `@@TICKPROBE-BEGIN`); the earlier record's stated cause, console drain, was wrong
  (UNIFY-440S.md, last section).

**Reference:** out-nd3-b3 (board 3, same kit and stage but the dump): 4,529.6 s; inside 85.80 G cycles; `kcyc - rcyc`
4.874 G; `ucyc` 19.18 G (767.3 s) over 5,863 steps; the heartbeat subshell's children 44,567 + 25,905 ticks.

## Board 3, C.5a, one boot

| # | Prediction |
|---|---|
| H0 | rc 0, specdiff 1/1 PASS, the `.s` and `.out` hashes equal out-nd3-b3's; control rungs pass; `@@KSYMS-DEFERRED` about 19,983 lines; the first heartbeat reads `out=` under 4,096 B |
| H1 | the heartbeat subshell's children (the `sh` whose parent is the batch's gate.sh and whose own start precedes the step's; its cutime + cstime over @@PST0..1) ≤ 150 ticks per 10 heartbeats, i.e. ≤ 1.5 s each (nd3: ~6.7 s each). The value is the reading: what every board stage pays per 30 s |
| H2 | `ucyc` ≤ 13.5 G cycles (540 s); point 2.5-6 G |
| H3 | run (up1 - up0) ≤ 4,250 s; point 3,850-4,100 s |
| H4 | inside cycles within ±0.5 % of 85.80 G; `kcyc - rcyc` within ±5 % of 4.874 G (the kernel window is not this change) |
| H5 | tick probe T0: gaps per second in 90-110 (else VOID, as before). T1, the gap share s, graded with 2026-11-06-unify's thresholds: s ≥ 4 %, Plain pays a comparable per-tick cost; s ≤ 2 %, the masked-time term is a Capstone excess; in between, unresolved |

**Readings.** H2/H3 failing with H1 holding: the heartbeat was not the excess, and UNIFY-440S.md is withdrawn. H1
failing: the small-file heartbeat is itself expensive, and every board stage, Plain included, carries it.

N = 1. If the coordinator prefers the harness fix itself (heartbeat sizes from `stat`), this stage still serves as its
control.
