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
