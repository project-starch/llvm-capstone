# RESULTS: the first MEASURED silicon overhead of pervasive spatial safety

**Date:** 2026-07-26 · **Lane:** B · Two board runs (plus two spent on a bug and a
transfer flake). Board powered off + unlocked in `finally` both times.

## What was missing, and why this run existed

The paper's silicon numbers were all *primitive* costs (`tab:primcost-rtl`: load 2,
shrink 1, mrev 50, delin+revoke 121). Every **application-level** figure was
analytic: `tab:appoverhead`'s ~1% / ≤6% SQLite boundary overhead is a QEMU-measured
borrow rate times the silicon per-borrow cost divided by instructions, at an
**assumed CPI of 1** (the footnote concedes it, and quotes 0.53% / 3.98% at CPI 1.5).

And the paper claims spatial safety is *pervasive* — "every pointer is a bounded
capability, always on" — shows it is **correct**, and never says what it **costs**.
RV8/CoreMark/BEEBS appear in exactly one sentence, as a compatibility claim.

The capability half of that measurement already existed (the 25-07 ladder sweep).
The **denominator had never been measured**: the same kernels as plain RISC-V on the
same board. That is what this run produced.

## The baseline vehicle

`rtl-smoke/ladder_base_{ctl,kern}.c` + `build-ladder-base-fpga.sh`. Each rung's
`<rung>_kernel.h` — the very header the capability domain runs — compiled by the
**same clang at the same `-O`**, for `-target riscv64` instead of `-target
capstone64`, with no capability flags. Building the baseline with buildroot gcc
instead would have made the ratio a comparison of two compilers; a static gate fails
the build if any capability instruction reaches the baseline.

**One boot, not one per rung.** The capability sweep power-cycles per rung (~2.5 min)
because every domain must be the first of a clean boot. The baseline creates no
domain, so all rungs run back to back: ~5 minutes total.

Both halves bracket the **same thing** — `ladder_perf_domain.h` reads `mcycle`
*inside* `domain_main` around the compute only, and the baseline brackets `r->fn()`
identically — so domain entry/exit is excluded from both and needs no correction.

## Run 1 produced an impossible number

| rung | capability | baseline (single pass) | ratio |
|---|---:|---:|---:|
| rv8_primes | 17,283,292 | 16,909,131 | 1.022 |
| **beebs_prime** | **47,804** | **87,845** | **0.544** |

`beebs_prime` — one of the only two rungs whose retval is correct on *both* sides —
said the capability domain was **1.84× faster** than plain RISC-V. That cannot be a
capability effect.

**The confound was mine.** The design held board, clock, DRAM, compiler and `-O`
fixed, and said so, but did not hold the **execution environment** fixed: the
capability rung runs as a bare-metal domain with no paging, the baseline as a Linux
process paying demand-paging faults on first touch of its own `.bss`. On an 88k-cycle
kernel that dominates. The tell was already in the table — the ratio sat near 1.0
only for `rv8_primes` at 17M cycles, where a fixed cost amortises away.

## Run 2: null control + cold/warm passes + instret

| rung | cold cyc | warm cyc | Δ | cold ins | warm ins | clean |
|---|---:|---:|---:|---:|---:|---|
| null | 48,944 | 376 | 48,568 | 3,632 | 23 | — |
| matmult_int | 136,400 | 71,860 | 64,540 | 33,050 | 29,661 | |
| coremark_matrix | 58,314 | 55,975 | 2,339 | 27,788 | 27,788 | **YES** |
| rv8_primes | 16,705,892 | 16,459,057 | 246,835 | 7,976,634 | 7,960,829 | |
| beebs_crc32 | 263,247 | 272,819 | −9,572 | 101,707 | 104,878 | stateful |
| beebs_insertsort | 25,119 | 8,398 | 16,721 | 5,532 | 3,410 | |
| beebs_prime | 46,733 | 46,306 | 427 | 14,680 | 14,680 | **YES** |
| beebs_recursion | 20,855 | 36,654 | −15,799 | 4,759 | 6,881 | |

All eight retvals correct. `beebs_crc32` self-identified as stateful (pass-2 retval
differs), exactly as the harness was designed to detect.

### The null control falsified my stated prediction

I predicted null would come back at "tens to hundreds of cycles" and wrote that a
~41k reading would mean "my whole model of where the time goes is wrong." **It came
back at 48,944.**

The reasoning was wrong in its specifics: I claimed the null rung "touches no
arrays," but `base_null` reads a `volatile` in `.bss` and executes from a fresh text
page, so it takes page faults like anything else — and being the *first* rung, it
absorbs the process-wide first touch. Warm null = **376 cycles** confirms the
instrument itself is negligible; cold null = 48,944 shows the paging cost is real and
large. So the substance of the hypothesis (a large constant cold cost, predicted at
~41,200 from the two-point fit) is confirmed, while my account of *which control
would show it* was not. **Second time in this investigation that a control fired and
my mechanism story for it was wrong — state predictions, but do not narrate the
mechanism as if it were settled.**

### Two noise sources, and a cleanliness certificate

- **Demand paging** inflates cold passes (cold > warm, cold instret > warm instret).
  It is also **layout-sensitive**: adding `null_sink` to `.bss` shifted the layout
  enough that `beebs_prime`'s cold pass stopped paying a fault at all (87,845 in run
  1 → 46,733 cold in run 2). **Cold numbers are not reproducible; warm numbers are.**
- **Interrupts are counted inside the bracket.** `beebs_recursion` is deterministic
  and idempotent, yet its warm pass retired **2,122 more instructions** than its cold
  pass and cost 15,799 more cycles. A timer tick landed in the bracket. Retval
  idempotency does **not** imply timing idempotency.
- ⇒ **identical instret across the two passes certifies a clean measurement** (no
  fault, no interrupt counted). `coremark_matrix` and `beebs_prime` carry it.

## THE RESULT

Using the **warm** baseline — the column comparable to a domain that has no paging —
and only the two rungs whose retval is correct on **both** sides:

| rung | capability | baseline (warm) | **overhead** | basis |
|---|---:|---:|---:|---|
| beebs_prime | 47,804 | 46,306 | **1.032×** | instret identical across passes ⇒ certified clean |
| rv8_primes | 17,283,292 | 16,459,057 | **1.050×** | 16.5M cycles ⇒ noise negligible |

**Pervasive spatial safety costs ≈3–5% on this CVA6.** The impossible 0.544 is gone;
both surviving points are above 1 and mutually consistent.

## Measured CPI — replaces the paper's assumption

| rung | warm cycles | warm instret | **CPI** |
|---|---:|---:|---:|
| coremark_matrix | 55,975 | 27,788 | 2.01 |
| rv8_primes | 16,459,057 | 7,960,829 | 2.07 |
| matmult_int | 71,860 | 29,661 | 2.42 |
| beebs_insertsort | 8,398 | 3,410 | 2.46 |
| beebs_prime | 46,306 | 14,680 | 3.15 |

`tab:appoverhead` assumes **CPI = 1** and calls it "the conservative upper bound",
with a 1.5 sensitivity case. Measured CPI on this silicon is **2.0–3.2**. Since a
larger CPI enlarges the baseline and shrinks the overhead, the SQLite boundary
figures should be roughly **halved**: ~1% → **≈0.5%**, ≤6% → **≈3%**. An assumption
in the paper becomes a measurement.

## What is NOT established

- **The 3–5% is not yet decomposed.** It could be the gp-captable ABI retiring *more
  instructions* (every global goes through `ldc rd, i*16(gp)`), or the same
  instructions costing *more cycles* under capability enforcement. Only the baseline
  side has `instret`. **Next: add `minstret` to `ladder_perf_domain.h` and re-run the
  two good rungs** — that splits the number cleanly and is what a reviewer will ask.
- **Two kernels only.** The other four rungs miscompile (open gp-captable bug), so
  their ratios measure wrong executions; `coremark_matrix` still has no capability
  verdict (transfer). Each rung the miscompute investigation recovers widens this.
- **The domain is built in the gp-captable *silicon workaround* config** (shrink off,
  `-fno-jump-tables`, gp cap-table), chosen for this RTL's constraints — not
  necessarily what a tuned Capstone ABI would emit. This figure plausibly
  **overstates** pervasive spatial safety in general. Do not present it as canonical.
- **No error bars.** One measurement per rung per pass. Repetition with a **minimum**
  statistic would be robust to interrupts (which only ever add).

## Process notes

- The QEMU parity leg (`run-ladder-base-qemu.sh`) validated the **binary** but both
  legs then invoked it with **different command lines**; the board rejected a report
  label passed as a counter name and a boot was wasted. **Parity must cover the
  invocation, not just the artifact.** Both now issue the identical `/tmp/lbc all`.
- A `fast_put` wedge cost a second boot. Boot+transfer is now one retryable unit in
  `run_ladder_base_fpga.py` (cold boot is idempotent).
- Idempotency was checked on the host before relying on the warm pass: 6/7 kernels
  idempotent, `beebs_crc32` stateful. Cheap, and it made the warm column defensible.

Artifacts: `rtl-smoke/ladder_base_{ctl,kern}.c`, `ladder_base_null_kernel.h`,
`build-ladder-base-fpga.sh`, `run-ladder-base-qemu.sh`,
`fpga_driver/run_ladder_base_fpga.py`. Results `/tmp/capstone/ladder-base-results.txt`.
Capability half: `25-07-2026_03-58-47_fpga-ladder-perf-sweep-results.md` (UPDATE 3).
