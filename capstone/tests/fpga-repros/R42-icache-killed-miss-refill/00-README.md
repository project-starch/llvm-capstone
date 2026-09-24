# R-42 — a speculative I-cache miss killed by a taken-branch redirect costs +1 cycle every loop iteration

**Performance, not correctness.** No value is ever wrong. A loop whose predicted-taken backward
branch is the **last word of a 16-byte I-cache line** (address mod 16 = 12) runs **7** cycles per
iteration instead of **6**. Which loops are hit depends on nothing but code layout.

**Siblings, so a reader with the wrong symptom is redirected:**
- **R-35** (`../R35-revoked-reference-retains-authority/`) is a correctness defect in the LSU, not here.
  It shares only the lineage: this fix is one commit on top of R-35's.
- The ladder's layout-dependent control ratio (`ctrsanity` reading **1.167×** at one start address) is
  where R-42 was found: `../../rtl-smoke/ladder-revival-2026-09-22/`, phases 6–10, and
  `docs/ref/fpga-silicon-measurements-for-paper.md` §2. **Any cycle comparison across a rebuild or a
  reflash must hold loop layout fixed** until this fix is on silicon.
- Registry entry: `docs/ref/ISSUES.md` R-42.

**Record, by date:**
- **2026-09-24:** mechanism confirmed in the RTL source and by waveform (board lane, ladder revival).
- **2026-09-24:** fix `capstone-ariane 6cbdaeeb4`, branch `r42-icache-killed-miss`, one commit on
  `4ad0df694`. Validated in simulation, then SYNTHESIZED: no new loop, bitstream written,
  sha256 `0cd45bb0…2b8c05`.
- **2026-09-25:** reflash authorized by the project lead and handed to the board lane.

Board results, once run, land as `results/board-*.result-lines.txt`. The pre-registered acceptance is
below.

---

## The problem, in pictures

The front end fetches **PC+4 speculatively every cycle**. Fetches are 4 bytes; an I-cache line is
**16 bytes** (`CVA6ConfigIcacheLineWidth = 128`). The test loop is five instructions,
`srai / xor / addi / add / bne`, placed so that `bne` is the last word of its line:

```
          I-cache line A (cached)                 I-cache line B (cached)          line C (never executed)
   +---------+---------+---------+---------+ +---------+---------+---------+---------+ +---------+----
   |  ...    |  ...    |  ...    |  srai   | |  xor    |  addi   |  add    |  bne  --+ | fall-   |
   +---------+---------+---------+----^----+ +---------+---------+---------+-------|-+ | through |
     +0        +4        +8        +12 |        +0        +4        +8        +12  |     +0 ...
                                       |                                           |      ^
                                       +-------------- predicted taken ------------+      |
                                                                                          |
                           the speculative PC+4 fetch after `bne` lands HERE, in line C --+
                           C is never executed, so it is never cached: it MISSES, every time
```

The miss then meets the branch predictor's redirect (`kill_s2 = kill_s1 | bp_valid`,
`frontend.sv:345`), and the I-cache FSM throws the miss away **and wastes a cycle doing it**:

```
                     cycle n                        cycle n+1                    cycle n+2
                +---------------------------+  +-------------------------+  +------------------+
  BEFORE        | READ: lookup of line C    |  | IDLE: now accept the    |  | READ: fetch the  |
  (4ad0df694)   |   miss + kill_s2          |  |   redirected request    |  |   loop head      |
                |   -> go to IDLE, ready=0  |  |   (ready=1)             |  |                  |
                +---------------------------+  +-------------------------+  +------------------+
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^
                                                 the +1 cycle, EVERY iteration

                +---------------------------+  +-------------------------+
  AFTER         | READ: lookup of line C    |  | READ: fetch the         |
  (6cbdaeeb4)   |   miss + kill_s2          |  |   loop head             |
                |   -> ready=1, accept the  |  |                         |
                |   redirect, stay in READ  |  |                         |
                +---------------------------+  +-------------------------+
```

A **hit** in the same position was always accepted in the same cycle (`cva6_icache.sv`, the hit arm of
READ). Only the killed **miss** detoured through IDLE. That asymmetry is the whole defect.

## The fix, as a state diagram

Only one branch of the READ state changes. Everything else — the hit path, a normal miss's refill,
translation, flush — is byte-identical:

```
                                 READ  (lookup of the speculatively fetched address)
                                   |
                  +----------------+-----------------------------+
                  |                                              |
              line HIT                                       line MISS
          (unchanged, already                                    |
           same-cycle accept)                      +-------------+--------------+
                  |                                |                            |
     ready=1 (if !mem_rtrn_vld_i)             not killed                   killed (kill_s2)
     req  ? -> READ : -> IDLE                      |                            |
     kill_s1 -> IDLE                  request the refill -> MISS     BEFORE:  -> IDLE, ready=0
                                           (unchanged)                          (the wasted cycle)
                                                                     AFTER:   ready=1 if !inv_q && !mem_rtrn_vld_i
                                                                              req    ? -> READ : -> IDLE
                                                                              kill_s1 -> IDLE      <- correctness guard
                                                                              the miss is STILL abandoned:
                                                                              no refill, no wrong-path fetch
```

Why each guard is there:
- **`kill_s1 → IDLE` carries the correctness.** In a `kill_s1` cycle the front end presents a stale
  fetch address (a mispredict's target goes only to `npc_d`), so that request must not be taken. This
  is exactly what the hit path does. An audit mutant without this guard delivered the wrong
  instruction stream on 200 of 200 seeds.
- **`!inv_q`, `!mem_rtrn_vld_i`** mirror the hit path's accept. On this target neither can fire here.
  `inv_q` cannot, because the AXI adapter never generates invalidations (`wt_axi_adapter.sv:685`
  returns only `ICACHE_IFILL_ACK`). `mem_rtrn_vld_i` cannot, because fill acks arrive only in
  MISS/KILL_MISS: the FSM has one fill outstanding at most and leaves those states only on its ack.
  They are kept for parity.

**Deliberately NOT the shape first noted in the registry** ("let a right-path miss complete its
refill"). That sentence was wrong: the killed fetch is the **fall-through**, which the loop never
executes, so it is a *wrong*-path fetch. Completing its refill would be a wrong-path prefetch — it
would block the cache for a miss latency and could evict useful lines.

**The one real risk, and how it was closed:** `dreq_o.ready` on the killed-miss path used to be the
constant 0. It now depends on `kill_s2`, which the front end computes. `cva6_icache` and `frontend`
already share a struct-level combinational cycle (UNOPTFLAT) that lint cannot see through, so a new
real loop would be invisible to every check except synthesis:

```
        frontend                                        cva6_icache
   +------------------+   kill_s2 = kill_s1|bp_valid   +------------------+
   |  bp_valid ------ +------------------------------->|  READ, killed    |
   |                  |                                |  miss: ready =   |
   |  if_ready  <-----+--------------------------------+  f(kill_s2)      |
   |     |            |      dreq_o.ready              +------------------+
   |     v            |
   |  npc_d  (a REGISTER input)  <- the loop would close here if if_ready fed bp_valid
   +------------------+             in the same cycle. It does not: npc_d is registered.
```

A bit-level trace predicted no loop. **Synthesis confirmed it:** one combinational loop in both
builds, and it is the same loop (the TIMING-23 arc `ex_stage_i/lsu_i/state_q[3]_i_19`), with no I-cache
or front-end signal in it.

## Evidence

| what | result | file |
|---|---|---|
| matched pair, slow layout (offset mod 16 = 12), both modes | 28035 → **24036** cycles = −3,999 = exactly one per taken branch; 7.009 → 6.009 cyc/iter | `results/sim-loopalign-pair.result-lines.txt` |
| matched pair, fast layouts (mod 16 = 0, 4, 8) | unchanged to the cycle | same |
| warm discriminator (next line cold / warmed / fast) | 477 → **415** / 413 / 410 | same |
| base arm vs the board lane's table | reproduces all of it exactly | same |
| neutrality sweep, 92 tests | **0** trap-count differences; the counter fires (48 tests trap, 639,786 traps) | same |
| `rtl-lint-gate` | PASS, every counter at baseline (LATCH 52, MULTIDRIVEN 3, UNOPTFLAT 40, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 736) | — |
| synthesis | no new loop, LUTLP-1 = 0, bitgen OK, area +3 LUT / +2 FF | `results/synth-6cbdaeeb4.result-lines.txt` |
| synthesis timing | WNS **−10.615** vs −8.341 — the ±0.5 prediction MISSED; worst path touches neither the I-cache nor the front end (0 of the worst 500) | same |

The timing loss is a trade the project lead accepted when authorizing the reflash: about 2.3 ns of WNS,
on paths the change does not touch, in exchange for one cycle per affected loop iteration. The board
has previously run a flashed bitstream at −12.425.

## Board acceptance — pre-registered before the reflash

One boot, ordered by the `board-run` skill:

1. **Control first:** `k800` returns 4.
2. **The ladder, including the layout case.** `ctrsanity` at its slow start address (the `…1ac` layout,
   **1.167×** on `054cea69b`) must read **≈ 1.000×**. Every rung's retval must be unchanged against
   `../../rtl-smoke/ladder-revival-2026-09-22/phase9-aligned-full-table.result-lines.txt`. Cycle counts
   move ONLY for loops whose taken branch sits at mod 16 = 12.
3. **Last, because it wedges:** R-35's stale probe (image `35fb3fec3196841b`) must STILL trap with cause
   25 at +0x4354. That is the regression check that this bitstream kept R-35's fix. It also identifies the
   image by behaviour, which the label cannot.

**Refuted if** the slow layout stays at ≈1.167×, any retval changes, or the R-35 probe reads data.

## Reproduce in simulation

```bash
# in a capstone-ariane worktree at 4ad0df694 (base) or 6cbdaeeb4 (fix):
cp src/r42-loopalign-*.S   verif/tests/custom/capstone/
cp src/testlist_r42.yaml   verif/tests/
# inside the cva6-build-rv container (docs/ONBOARDING.md), with S12_MEM_DELAY=12 and --sv_seed pinned:
python3 cva6.py --testlist=../tests/testlist_r42.yaml --test r42-loopalign-ctrsanity \
  --iss_yaml cva6.yaml --target capstone_cv64a6_imafdc_sv39 --iss=veri-testharness \
  --isscomp_opts=+define+S12_MEM_DELAY=12 --sv_seed 1
# cycles per arm = the difference of the two `csrr mcycle` reads around each loop (x5/x6 in the trace)
```

`src/cva6_icache-r42.patch` is the complete fix (`git apply` on `4ad0df694`).

## Files

```
00-README.md                                   this report
src/r42-loopalign-ctrsanity.S                  10-arm layout sweep, M-mode and capability mode
src/r42-loopalign-warm.S                       the cold/warm/fast discriminator
src/testlist_r42.yaml                          cva6.py testlist for both
src/cva6_icache-r42.patch                      the fix, 6cbdaeeb4 against 4ad0df694
results/sim-loopalign-pair.result-lines.txt    matched pair + neutrality sweep
results/synth-6cbdaeeb4.result-lines.txt       synthesis readings
SHA256SUMS
```
