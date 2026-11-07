# The unified stack on silicon: dev's runtime and module with the board's switches (pre-registration, 6 Nov 20:32)

## Why

- gcc train c3f reads 1.210× Plain (results/2026-11-06-arena, last section). Its remaining gap is inside the program:
  +649 s over Plain user time, from 1.26× the instructions.
- dev's heap runtime (in-domain mapping reuse + regions + the null-capability memset, PR #306 and its region branch)
  cuts gcc train's in-context instructions on QEMU 13.735 → 13.421 G (−2.3 %), and entries 11,875 → 3,899.
- For perlbench it cuts run1 96.74 → 89.86 G (−7.1 %) and entries 353,780 → 146,348 (−59 %), specdiff 2/2.
- The boards could not run it: their module (perf/exit-timing-vp2) and dev's had diverged both ways.

## What runs

- **Module:** perf/board-on-dev 08d74206e5df = origin/dev c9611344eeb7 plus the board module's eight commits, cherry-picked
  (counters, retire_exit_sweep, image_rootfs, step_budget_ns, exit timing, resume_quantum, hpm_snapshot).
  - The one conflict, retire_exit_sweep against dev's batched unpin, keeps dev's loop; modes 1 and 2 are applied inside it.
  - dev's fault-ahead and bench counters were already on dev; their board ports are not taken.
  - Built for the board kernel -DCAPSTONE_VM_BENCH_HPM -DCAPSTONE_VM_INDEX_BITS=18: 13a5ad66752a4e0b, 0 two-byte
    instructions (the count's control fires: 2 on a c.nop/c.addi object).
- **Launcher:** dev's capstone-vexec a8a4dbe637ac758a (static, rv64g). It rounds the image window up to a power of two;
  the SPEC kits' launcher granted the exact image (memory: launcher-window-is-exact-on-spec-kits).
- **Kit bkit30-unify** = bkit27-gtq with those two files; params unchanged (fault_ahead_max=64, image_rootfs=1,
  retire_exit_sweep=3, step_budget_ns=80000000). On dev's module mode 3's store sweep is what every retire already does.
- **Images:** gcc cpugcc_r.c3devrgz a961f56a5e383661 (llvm-c3 objects, dev SDK devrgz); later perlbench c3devrg
  89398ba2be3127ab.
- **The stage's parameter block,** read back as @@BP: step_budget_ns 10 s, resume_quantum 1, retire_exit_sweep 2,
  image_rootfs 1, fault_ahead_max 64, arena_pin 1.

**QEMU gates on the same module (kit-unq, its QEMU build be4c814f5daabb4f):**
- gcc train c3devrgz with default parameters: 13,425,112,966 instructions, 3,910 entries, specdiff PASS. kit-devq read
  13,424,956,115 and 3,903.
- With the board's bundle: 13,420,874,486 instructions, 3,899 entries, specdiff PASS, every parameter read back.

## The tick probe (first, Plain conditions, before the context starts)

`tickprobe 30` (source beside this file) spins on rdcycle for 30 s in a Plain process. It sums every step longer than
2,000 cycles: time the kernel took from a user thread.
- **Why it matters.** The kernel's time in CV_STEP outside the context, minus fault resolution, is 5.7–6.2 % of inside
  time on every graded run. Whether Plain pays the same inside its user time decides whether that is a lever.
- **Interrupt counts cannot decide it.** On this kernel one interrupt after a masked quantum runs tick_periodic once per
  missed period (kernel/time/tick-common.c, oneshot mode), so Capstone's 17–19 interrupts/s are not fewer ticks.
- **T0 (control):** gaps per second in 90–110 (HZ=100). Outside that the probe is void.
- **T1:** the gap share s.
  - s ≥ 4 %: Plain pays a comparable per-tick cost, and k−r is not a Capstone excess.
  - s ≤ 2 %: k−r is an excess of about 4 % of inside time (mcf about 500 s), and a firmware or kernel lever.
  - In between: unresolved. The point estimate is not registered; either reading is informative.

## Predictions

| # | Prediction |
|---|---|
| U0 | rc 0 and specdiff PASS; the kit's control rungs pass; the module banner and @@BP read back as above |
| U1 | gcc train ≤ 4,253.3 s (c3f, same board). Point 4,170–4,240 s: −0.316 G at 80–223 s per G removed, outside about unchanged. 1.20× Plain is 4,218.6 s. |

perlbench joins the list only after its own full QEMU gate on kit-unq passes (specdiff 2/2, every sub-test). Its
prediction is written then, against perlbench c3g's board run (board 2), whose own test.out must be clean.

## Limits

- N = 1.
- Plain gcc train is on board 1, this run is on board 3.
- The module, launcher and runtime change together: this is a stack comparison against c3f, not one variable.

## Addendum 1 (6 Nov 21:14, before any board call): perlbench joins board 3's list after gcc train

**QEMU gate on kit-unq** (perlbench c3devrg 89398ba2be3127ab, the stage's parameter block read back as @@MX-SET):
- 2/2 commands rc 0, **specdiff 2/2 PASS**;
- run1 89,859,365,946 instructions, 146,026 entries;
- test.out holds 317 "ok" lines and 0 "FAILED". The FAILED count fires: it reads 3 on the VOID -ac board run.
- The three sub-tests that failed there pass: t/japh/abigail, t/re/pat_rt_report, t/re/reg_email.

The board stage `500.perlbench_r-test-capstone-lifetimes-unify-rg` is that stage, with plan.json's executable hash
corrected. The copied plan named c3f's e42f4649eccf10c0; the runner does not read that field.

**Prediction (U2)**, against the -ac board run (inside 25,364 s, outside ~9,500 s, 34,887 s; VOID for its failed
sub-tests, but a valid time measurement of that image):
- inside ×0.826 (in-context instructions 93.55 G against -ac's 113.2 G on the twin) ≈ 20,950 s at the same CPI;
- outside 6,500–8,000 s: entries −68 % on the twin, while launch and teardown per context and the rate components stay;
- **run 27,000–30,000 s (1.16–1.29× Plain's 23,311 s); bound ≤ 32,000 s;**
- rc 0, specdiff 2/2, and test.out with 0 FAILED. A FAILED sub-test voids the run.
- perlbench c3g's board run (board 2) is read beside it when it ends.

## Result, gcc train (7 Nov 03:38): 4,611.0 s, U0 holds, U1 FAILS; the tick probe is VOID

The stage ran on board 3 02:07-03:38 on C.5a (1fc2cd5aa), kit bkit30-unify. Graded with `un-grade.py` (this folder)
against c3f (out-ga2-b3, the same board and bitstream).

| | c3f (bkit27-gtq) | unified (bkit30-unify) | change |
|---|---|---|---|
| run | 4,253.3 s | 4,611.0 s | +357.7 s |
| inside | 3,513.8 s, 13.737 G instructions, IPC 0.156 | 3,432.1 s, 13.422 G, IPC 0.156 | −81.7 s |
| outside | 739.5 s | 1,179.0 s | +439.5 s |
| steps (CAPSTONE_VM_STATS); exits by kind [1, 3, 4] | 18,730; 8,327 / 7,528 / 2,875 | 11,772; 10,487 / 527 / 758 | |
| module time per CV_STEP: kcyc / rcyc / ucyc / ocyc (G cycles) | 7.65 / 2.42 / 9.84 / 0.35 | 6.68 / 1.77 / 19.26 / 0.53 | ucyc +9.42 G |
| machine-wide instructions (hpm snapshots) | 16.53 G | 22.92 G | +6.39 G |
| /proc/stat intr, ctxt | 71,561, 60,960 | 111,868, 61,163 | |

- **U0 holds:** rc 0, specdiff 1/1 PASS, outputs equal the twin's, control passes, @@BP reads back every parameter.
- **The inside reading is the one predicted:** −0.315 G instructions at an unchanged IPC, −82 s.
- **U1 FAILS:** 4,611.0 s against ≤ 4,253.3 s; 1.312× Plain. The outside time rose 440 s while exits fell.
- **Where the 440 s is:** in ucyc, the module's time from one CV_STEP's return to the next CV_STEP's start: the launcher
  and every call it makes there, plus whatever the kernel runs meanwhile. It is 3.24 M cycles per step against 0.87 M.
  The machine executed 6.39 G more instructions over the run while the context executed 0.32 G fewer, so the time is
  executed work, about 1.1 M instructions per step, not stalls. Kernel time inside CV_STEP (kcyc) fell.
- **What it is not:** the grant lines c3f's quiet launcher removed (the .err holds no CAPSTONE_VM_ADD_OK line). The QEMU
  twin does not show it: there ucyc is 0.09 M cycles per step over 623 steps.
- **What it is:** UNRESOLVED. The unified kit changes the launcher, runtime and module together: dev's launcher with
  CAPSTONE_VM_MAPCACHE=64M (235 misses, 0 hits), against c3f's quiet-grant launcher. A per-step cost of that size scales
  like a walk over the 65,065 pinned pages or the mapping registry, which is a hypothesis to test, not a finding.
- **The tick probe is VOID (T0 FAILS):** 54 gaps/s, outside the 90-110 control band; the gap share read 32.4 %. The
  console was still draining the stage's kallsyms dump (20,000 lines, printed just before) while it ran: the driver
  received `@@KSYMS-END`, `@@TICKPROBE-BEGIN` and the probe's RESULT line, printed 30 s later, in one chunk. T1 is not
  read.

## The 440 s is found (7 Nov, 17:5x): a SHA-256 of the whole image at every context's exit

- **Where.** A QEMU capmix profile of the same stage (kit-unq, gcc train) at block resolution: in the run window the
  launcher executes 1.352 G instructions, 1,336.9 M of them in `capstone_application_hash`. c3f's launcher executes
  about 0.29 G.
- **Why.** The batch harness exports `CAPSTONE_EXEC_DIAGNOSTICS=1` for every step (bug-corpora/tools/virtualvm.py:130,
  boardvm.py:193), so that a fault prints its cause line. dev's launcher (exec.c:1380) also prints the image's digest
  when a context exits under that switch, and computes it there: SHA-256 of the 12.9 MB gcc image, about 104
  instructions per byte. c3f's quiet launcher predates the exit digest: its .err holds no `exit image` line.
- **Check.** The stage with `unset CAPSTONE_EXEC_DIAGNOSTICS` before the arm (perf/nodiag, one line, `@@NODIAG unset`):
  launcher instructions in the run window 1.352 G → 0.002 G; rc 0, specdiff 1/1 PASS, every output hash equal to the
  earlier twin's.
- **The earlier reading is withdrawn:** "a per-step cost … like a walk over the pinned pages" (the Result above) was a
  hypothesis, and it is wrong: the cost is per context exit, not per step. perlbench runs 318 contexts with a 3.2 MB
  image each, which is the shape of its +5,431 s.

## RETRACTION (7 Nov, 19:3x): the digest is not the 440 s

The section above ("The 440 s is found") is withdrawn for gcc. Board 2 ran the stage without the digest
(results/2026-11-07-nodiag): 4,518.6 s, outside 1,083.5 s, ucyc 19.18 G cycles (unify 19.26 G, c3f 9.84 G). The digest
was removed (`@@NODIAG unset`, no `exit image` line) and took 92 s with it, about what 1.35 G instructions cost on the
board. It was never in ucyc: the digest runs once, after the last CV_STEP, and ucyc only counts time between two
steps. The reasoning matched the QEMU launcher count to the board's per-step window without checking that they were
the same window. The per-step excess, 3.2 M cycles against 0.87 M, is UNRESOLVED again; QEMU does not show it (its
ucyc per step is 0.1 M for both stacks).
