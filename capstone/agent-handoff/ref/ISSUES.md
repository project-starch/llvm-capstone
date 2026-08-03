# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-08-02.

---

## RTL / FPGA

### R-1 — A load through one capability register misses a store through another `CHARACTERISED`
**The blocker for several of the 13 benchmark rungs.** An intervening store through one capability register
causes a later load through a *different* capability register to miss an earlier store to its own
address — though the addresses are distinct and both capabilities are in-bounds derivations of the
same object. Not loop-specific. QEMU executes every probe correctly.

- **Repro:** `tests/fpga-repros/R01-lsu-hazard/`; sources
  `tests/runtime-qemu/silicon-ladder/rawhazard{_kernel.h,5,6,7}_fpga_app.c`
- **Evidence:** `history/27-07-2026_17-05-00_RESULTS-culprit-found-register-indexed-load-misses-pending-stores.md`
- **Mitigations tried (7, all failed):** fence before load, fence after every store, register
  hoisting, other store register-indexed, 64 B cache-line separation, constant-offset pointer
  walk, both accesses via pointers. **No general software workaround** — a dynamic array index
  cannot have a compile-time-constant base.
- **Impact:** `matmult_int`, `coremark_matrix`, `beebs_crc32`, `beebs_insertsort` unmeasurable.
- **Confidence it is hardware:** high, not certain. Residual doubt is whether our non-standard
  gp-captable ABI provokes it. **Open question for the board owner.**
- **Predictive record — see the SCORED entry below for the final tally (2 hits, 3 misses,
  1 partial). R-1 is NOT a complete account of the board's behaviour, but its own scope is
  confirmed.** Rungs were written specifically to test its predictions:
  - `beebs_bs` — **predicted PASS, PASSED** (887447230 = oracle, 2264 cyc). This is the
    load-bearing confirmation: `bs_data[mid]` is a genuine register-indexed load through a
    derived capability — the exact addressing form in every failing rung — and it is correct
    because nothing is ever *stored* to the table. **The intervening store is a necessary
    ingredient**, not incidental.
  - **SCORED 2026-07-27 (board): 2 hits, 3 misses, 1 partial — and the same-object clause is
    CONFIRMED.** `beebs_cnt` passes on silicon (oracle exactly), and it is the sharpest
    cross-object control available: its seeding loop keeps stores outstanding to `Array` and
    to `Seed` through two capability registers naming two *different* globals. R-1 predicted
    PASS and it passed. **The "same object" clause in this entry and in the repro README is
    therefore tested, not merely inferred, and needs no correction.**
    `beebs_bs` passed again (2,258 cyc, reproducing 2,264 from the prior session).
    `beebs_fac` and `beebs_duff` HANG; `beebs_fibcall` miscomputes while retiring ~94 % of the
    baseline's instructions (166,539 vs 177,855) — a third signature, distinct from both the
    hangs and from "the compute never ran". R-1 speaks to memory-shape failures and does not
    explain hangs, consistent with the standing ≥2-independent-faults position.
    > **⚠ A mid-run report that "R-1's same-object clause is REFUTED" was WRONG and is
    > withdrawn.** It came from a sweep accidentally run at −O0 (see I-1); at the intended
    > −O1 the cross-object control passes. Nothing in the repro package needs changing.
  - **Four predictions registered 2026-07-27 before the board ran.** Built, QEMU-green through
    the identical controller, oracles fixed, `-O1` to match `beebs_bs`. Written down *before*
    the board speaks so they are tests and not stories:

    | rung | predicted | what it discriminates |
    |---|---|---|
    | `beebs_fibcall` | PASS | no arrays at all — a failure would mean R-1 is not the whole story |
    | `beebs_fac` | PASS | same, plus a 2nd recursion point for the 1.801× headline |
    | `beebs_cnt` | PASS | **the same-object clause.** Stores to `Array` and to `Seed` are outstanding through two capability registers naming two *different* globals |
    | `beebs_duff` | PASS | **pointer-walk into two distinct objects** — the form that failed in rawhazard7 *within one object* |

    `cnt` and `duff` are the load-bearing pair. Every failing rung to date is same-object
    (`C[i*N+j] += …` reads and writes ONE array through two derived registers); no
    cross-object control has ever been run. If they pass, R-1 stays narrow and most of a
    benchmark suite remains measurable. **If either fails, R-1 is wider than written — any two
    derived capability registers — and this file plus the repro README must be corrected
    before the package goes to the board owner.**
  - `beebs_janne` — **predicted PASS, HANGS** (see R-6). Now bisected: the failing loop nest
    contains **no memory operations at all**, so R-1 cannot explain it and the two must not be
    conflated. R-1's scope is unchanged by it; its completeness as an explanation of the whole
    board's behaviour is not.

### R-2 — `delin` in domain code wedges the board `EXPLAINED 2026-07-29 by C-13 — not an RTL defect`

**This is not a hardware fault and not specific to domain code.** It is the C-13 root
cause seen from the other end: the RTL's `DELIN` accepts `CAP_TYPE_LINEAR` only, and a
capability **loaded from the gp cap-table is already `NONLIN`** — cap-table storage caps
are produced by `SPLIT` from an `sp` the entry glue already delin'd, and `SPLIT` preserves
`cap_type`. So the delin in the repro was a *second* delin on a non-linear capability,
which the RTL correctly rejects. QEMU's `helper_csdelin` returns early in that case, which
is why the repro looked like an RTL-only defect. The description below ("a delin on a
capability loaded from the gp cap-table") states the precondition exactly.

Correct rule: **never `delin` a capability obtained from the gp cap-table.** It is already
non-linear, so the `delin` is redundant as well as fatal. See C-13, and
`history/29-07-2026_C-13-root-cause-double-delin.md`.

The original text follows; the observation was sound, the "RTL wedges on delin"
interpretation was not.

A `delin` executed in domain code on a capability loaded from the gp cap-table wedges the board
(power-cycle to recover). Proven against a size-matched `addi x0,x0,0` control at the same address,
so it is the instruction and not code layout.

- **Repro:** `tests/fpga-repros/R02-delin/` (superseded — now a secondary item in the
  R-1 package); probe knob `LADDER_CM_WITH_DELIN`
- **Evidence:** `history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`
- **Workaround:** the `delin` was ours and unnecessary — removed from the default build, which
  also returns `coremark_matrix` to being a faithful copy of upstream.
- **Probably our bug**, not the platform's: the glue already delins every cap-table entry before
  storing it, and our QEMU was patched to tolerate the redundant case *"rather than faulting"*.
  Only the failure *mode* (full wedge vs catchable trap) is worth the board owner's attention.

### R-7 — `rv8_sha512` hangs on silicon: an INSTANCE OF R-1, not a new fault `CLOSED into R-1`
Measured 2026-07-28. The rung builds with the C-5 window + copy-path bypass, passes the
QEMU parity leg with its full 640 B table (oracle 1390718314), and then **hangs the
`cscall` on the board**, both attempts.

- **Its BASELINE half is clean and measured:** 540,073 cyc / 462,646 instret, 15/15 passes
  tied at min instret, spread 0, correct oracle. So only the capability half fails.
- **R-1 predicted PASS**: `sha512_k[i]` is a read-only indexed load with nothing ever
  stored to that table — the `beebs_bs` shape, which passes. But `sha_w[i&15]` **is** both
  read and written inside the compression loop, with `sha_chain[]` stored in the same
  region, so the same-object load/store pattern R-1 describes *is* present after all. This
  rung is therefore consistent with R-1 rather than a counter-example — unlike R-6.
- **CONFOUND ELIMINATED — the C-5 workaround is EXONERATED.** The control
  (`rv8_sha512s`: identical compression loop, 16-entry table, **default 4 KiB window,
  default unrolled path, no bypass**, QEMU-green at oracle 2842840124) **hangs on silicon
  too**. So neither the 32 KiB window nor the ~8 KB prologue is implicated: the fault is
  the kernel's memory shape. **R-7 is an instance of R-1**, and the `DOMAIN_WINDOW=32k` /
  `LADDER_NO_RO_COPY=1` machinery is sound and reusable for other rungs.
- **Which also means my PASS prediction was simply a misread of my own kernel:**
  `sha_w[i&15]` is read *and written* in the compression loop while `sha_chain[]` is stored
  in the same region — the same-object load-with-intervening-store pattern R-1 describes.
  Only `sha512_k` is read-only, and that was the part I looked at.
- **Control kept in the tree** (`rv8_sha512s_*`) as the cheapest R-1 reproducer that is not
  a synthetic probe: a real crypto kernel, 4 KiB, no special flags.
- **Repro:** `DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1 DOMAIN_OPT_LEVEL=-O1`, artifacts in the
  ladder dir; capability half must be run with `LADDER_REBUILD=0` (see below).

**Tooling gap found while running this — FIXED 2026-07-28.** The runner's rebuild path did
not know about `DOMAIN_WINDOW` / `LADDER_NO_RO_COPY`, so a default run would silently rebuild
this rung at 4 KiB with the broken copy path and measure the wrong binary; `LADDER_REBUILD=0`
with a pre-built dir was the workaround. The knobs now live in **`ladder-rungs.spec` field 5**
and travel with the rung through `build-ladder-fpga.sh`, so a plain sweep builds it correctly
and `LADDER_REBUILD=0` is no longer needed. Same fix shape as I-1: put the per-rung build
property in the one file both halves read, rather than relying on an env var set by hand.
(The baseline half discards field 5 explicitly — it is plain riscv64 with no glue to affect.)

**Re-reproduced AGAIN 2026-07-28 after C-4b was fixed**, now via the copy path at the
DEFAULT 4 KiB window with no knobs (transfer `sha 1e159a9fa415a763 OK`, first attempt):
still no END marker in 120 s, both attempts. Expected — R-7 is an R-1 instance and the
4 KiB control `rv8_sha512s` hangs too — but it costs nothing to confirm alongside `beebs_ns`
and being wrong in that direction would have been worth knowing.

**Re-reproduced 2026-07-28** on the burst-transfer path with the knobs coming from the spec:
transfer clean (`sha a88b9760f76b5741 OK`, first attempt), `rv8_sha512 domain ID = 0` prints,
then no END marker in 120 s, twice. Same hang, now on a build the runner produced itself.

### C-10 — capability-spill lead: REFUTED `CLOSED`
Proposed and killed the same evening, by the falsification checks written into the entry
before acting on it.

**The lead:** `accum_probe`'s slot stores are emitted but never land, and nearby sits
`sd a0, 0x40(sp)` — a 128-bit capability apparently spilled with a plain 8-byte store,
which would drop the tag and corrupt `res` on reload.

**Refuted by the control:** `expint_diag`, which writes the same slots **successfully**,
contains the **identical instruction** (`100b8: sd a0, 0x40(sp)`). Present in both the
working and the failing probe, so it cannot be the cause. A follow-up check also killed the
register-reuse variant: **both** probes use `a0` as the base for their slot stores
(`sd _, 0x18(a0)`, `0x20(a0)`, …) over the same offset range.

**So the two probes are structurally identical in every respect hypothesised, and
`accum_probe`'s delivery failure is UNEXPLAINED.** Both spill `a0` the same way, both store
through `a0`, both write `res[0]`/`res[2]` last — and only one delivers. Something outside
this comparison differs. Do not re-run either on the board until it reproduces off-board;
the QEMU ladder harness gives an 8-byte `res` region and so cannot exercise the debug-slot
path at all, which is why two boots were spent learning nothing.

**Value of the entry:** it is kept because the *method* worked. The falsification checks
were written down before the fix was attempted, and they killed the theory in one command
instead of after a codegen change. That is the practice to repeat.

### I-4 — some probes return ALL ZEROS on the board while correct under QEMU `OPEN — blocks R-6/R-8 work`
2026-07-28. Two probes (`accum_probe`, `accum2_probe`) fail to deliver results **on the
board** while the **identical binaries** are correct under QEMU via the new diag loader.
`expint_diag` and the `rawhazard*` family deliver fine on the board, so the mechanism is
not "debug slots don't work".

| probe | QEMU (diag loader) | board (`ladder_perf_ctl`) |
|---|---|---|
| `expint_diag` | — | **slots delivered** (`dbg0=0 dbg1=2 … dbg7=2`) |
| `accum_probe` | **9/9 correct** | retval **100 correct**, all slots **0** |
| `accum2_probe` | **9/9 correct** (`3883 0 3883 100 3881 3883 49 100 3883`) | retval **0**, all slots **0** |

`accum2_probe` is the sharper case: on the board **even `res[0]` is zero**, i.e. the region
reads back entirely unwritten, yet the `cscall` returned normally (no hang, no fault, the
runner reported a result). Under QEMU the same binary writes everything correctly.

**Why this blocks the R-6/R-8 hunt:** every bisect designed to find that fault is delivered
through exactly this path, and two of three such probes now come back empty. Until it is
understood, a board "all zeros" cannot be distinguished from "the fault under
investigation".

**Leads, none checked:** `expint_diag` (works) writes `res[3+0]` **early**, before its main
loop, while both failing probes write only after several loops; `accum2_probe` uses a
`volatile unsigned long *out` alias where `expint_diag` writes `res[...]` directly; the
failing probes are also the largest. > **⚠ CORRECTION: this is NOT an off-board investigation.** An earlier note here claimed it
> was. It cannot be: these probes are **correct under QEMU** and fail only on the board, so
> emulation cannot reproduce the failure. What QEMU buys is that a probe can be proven
> *well-formed* before spending a boot — not that this fault can be chased there.
>
> Static comparison of the three domains found **no discriminator**: identical `.text`
> section size (0x1000, the padded window) and no visible frame-size difference. So the
> difference is not code size or stack depth as guessed.
>
> **This therefore costs board time to resolve, and each attempt is one boot.** Budget
> accordingly, and prefer adding slots to a probe that ALREADY delivers on the board
> (`expint_diag` is the known-good vehicle) over debugging why a new one does not.

### R-8 — pure-scalar miscompute; the "accumulator" characterisation is TOO BROAD `OPEN`
Measured 2026-07-28 on `beebs_expint`, and it is the cleanest instance of this class yet.

| | capability | baseline (bare-metal) |
|---|---:|---:|
| retval | **2,223,116,741** ✗ | 2,021,290,181 ✓ |
| cycles | 110,988 | 110,844 |
| instret | **71,243** | **71,248** |

**The instruction counts differ by 5 out of 71,000.** The domain ran the whole
computation — this is not a hang, and not the "compute never ran" signature
(`beebs_insertsort`'s 560 instructions) — and produced a different number.

**Why R-1 cannot explain it:** `beebs_expint` has **no arrays at all**. Every value is a
scalar local; the only global is a `volatile long` accumulator. There is no same-object
load/store pair for a memory hazard to act on. The rung was in fact *selected* against
R-1's shape for exactly this reason.

**Why it is not a compile-time difference:** the identical binary is **QEMU-correct**
(`__CAPSTONE_LADDER_BEEBS_EXPINT_PASSED__`, oracle 2,021,290,181). So constant folding,
the `2e6`/`3e7` double-to-long literals, and shift-amount UB are all ruled out — those
would fail under emulation too.

**So: same instructions, same count, different arithmetic result, on silicon only.**

- **Companion case:** `beebs_fibcall` is also pure scalar and also miscomputes on silicon
  (at −O1 it retired 166,539 against a baseline 177,855 — ~94 % of the work, wrong answer).
  Two independent pure-scalar miscomputes make this a class, not a one-off.
- **Relation to R-6:** `beebs_janne`'s failing nest is likewise pure register arithmetic.
  R-6, R-8 and the `fibcall` miscompute plausibly share one mechanism that is **not** R-1.
- **Value:** this is the strongest evidence yet that **R-1 is not the whole story**, which
  the registry has flagged since R-6 but could not previously support with a clean case.
- **Repro:** `tests/runtime-qemu/silicon-ladder/beebs_expint_*`, `-O1`, oracle
  2,021,290,181, QEMU-green, baseline half clean (15/15 tied, spread 0).
#### BISECTED 2026-07-28 (`expint_diag`) — one slot diverges, and it names the fault

| slot | board | expected | |
|---|---:|---:|---|
| dbg0 branch / dbg1 init | 0 / 2 | 0 / 2 | ✓ |
| dbg2,3 `fact` (signed div) | 0 / 0 | 0 / 0 | ✓ |
| dbg4 `psi` (nested loop) | **3881** | 3881 | ✓ |
| dbg5 `ei_foo` (the shift) | 0 | 0 | ✓ |
| **dbg6 `del` at i==nm1** | **3881** | 3881 | ✓ **the addend is correct** |
| dbg8 trip count | **100** | 100 | ✓ **the loop ran fully** |
| dbg9 `sum(ii)` | 1225 | 1225 | ✓ |
| **dbg7 final `ans`** | **2** | **3883** | ✗ |

**`ans` is frozen at its INITIAL value.** The loop ran all 100 iterations, `del` was
computed correctly as 3881, and `ans += del` did not accumulate. Nothing else diverges —
division, shifts, the nested loop and control flow are all correct.

#### This is R-6's mechanism, and the two issues unify

`beebs_janne` (R-6) showed *exactly* this: `a` frozen at **2** after 200 iterations of
`a = a + 2`, with the loop counters self-consistent. Both cases are:
- **pure register arithmetic**, no arrays, no memory in the loop
- the loop **runs its full trip count**
- the per-iteration value is **computed correctly**
- the **accumulator retains its initial value**

> **Proposed statement (NOW KNOWN TOO BROAD): a scalar accumulated across loop
> iterations retains its initial value.**
>
> **⚠ REFUTED as stated, 2026-07-28.** A minimal probe --
> `long a = 0; for (i = 0; i < 100; i++) a += 1;` in a domain, returned as the retval --
> **comes back as 100, correct**, on the same board and toolchain. So plain accumulation
> is NOT broken, and whatever breaks `expint` and `janne` needs more than a loop and a
> `+=`. Candidate extra ingredients, none yet tested: a branch inside the loop body,
> register pressure, a nested loop, or the specific accumulate-inside-an-if shape both
> failing kernels share.

R-1 cannot explain either (no memory involved), and the identical binaries are
QEMU-correct. `beebs_fibcall`'s pure-scalar miscompute is very likely the same thing.

**Why this matters more than a benchmark row:** R-1 plus this account for essentially every
silicon failure seen — R-1 for the array kernels, this for the scalar ones. Two mechanisms,
not a fog. It is also a far better bug report: a five-line loop whose accumulator does not
accumulate, with a QEMU-correct binary and every neighbouring operation proven good.

**Probe status: TWO versions run, both INCONCLUSIVE — the blocker is our harness, not the
board.** v1 pinned accumulators to named registers (suspected of corrupting `res` in `a0`);
v2 removed all pinning, used a `volatile` store pointer and wrote each slot immediately
after its loop. **Both behaved identically**: `res[0]` and `res[2]` land (retval 100 and the
`0xD09E` marker both arrive) while `res[3..11]` all read zero, so the controller suppresses
the DEBUG line.

**The discriminating fact: `expint_diag` writes the SAME slots successfully** (it returned
`dbg0=0 dbg1=2 ... dbg7=2`). So slot delivery works in one probe and not another, and the
difference is in our two `domain_main` implementations, not in silicon. **Diff them before
running anything else** — `expint_diag_fpga_app.c` vs `accum_probe_fpga_app.c`. Do not
spend another boot on this probe until a QEMU-visible reproduction exists; note the QEMU
ladder harness gives only an 8-byte `res` region, so the debug-slot path is currently
board-only, which is itself worth fixing.

**Original probe status note (superseded, kept for the reasoning):**
It was designed to discriminate the important question (see below) across 9 debug slots. On
the board `res[0]` returned **100** — the plain accumulate, correct — but **all nine
`res[3..11]` slots read zero**, so the controller suppressed the DEBUG line and eight of
nine probes produced no data. The `res[3..]` writes did not land even though `res[0]` did;
the QEMU harness separately rejects this probe because its shared region is only 8 bytes.
**Fix the probe's use of the debug slots, then re-run** — the discrimination is still the
right experiment.

**The question the probe must answer, and why it matters more than the benchmark:**
"an accumulator does not accumulate" is an extraordinary claim about an ALU. An ordinary
explanation fits every observation equally well — the value lives in a **register that
something clobbers on silicon**: our entry glue, the `cscall` path, or a trap handler that
saves less than our QEMU fork models. That would present identically (right addend, right
trip count, value reverting to its initial state) and would be **our bug, not the board's**.
Reading: memory-form correct + register-forms wrong ⇒ ours; one register class failing ⇒
names the culprit; all forms failing ⇒ the hardware claim survives; short loop passing and
long failing ⇒ something periodic, i.e. a trap.

**Confidence, stated plainly:** R-1 is well supported (five-line repro, controls both
sides, 7 failed mitigations, a correct advance prediction). **R-6/R-8 are NOT** — calling
them hardware is currently an assumption, and the minimal probe passing makes a
software-side explanation *more* likely, not less.
- **Repro:** `tests/runtime-qemu/silicon-ladder/expint_diag_fpga_app.c`, `-O1`,
  expected `dbg7=3883`, board returns 2.

### R-9 — `beebs_ns` hangs although its tables are never written `LIKELY EXPLAINED 2026-07-29 by C-13 — re-test required`

**Leading explanation, not yet confirmed on hardware: the copy-path double delin.**
`beebs_ns` takes the large-RO **copy path**, and the C-4b fix prepends `delin(sp)` to the
generated glue *only for copy-path rungs* — which made that glue's later `delin(gp)`,
`delin(t2)` and trailing `delin(sp)` faults on silicon, since `SPLIT` preserves `cap_type`
and the RTL's `DELIN` is `LINEAR`-only. Copy-path rungs are exactly the set that hangs on
the board while passing under QEMU, which is R-9's signature.

Fixed in `39f652b6e704`: `beebs_ns` and `beebs_crc32big` drop from 5+ delins to 1;
non-copy-path rungs verified byte-identical. QEMU still green (it cannot see this bug).
**Re-run `beebs_ns` on the board** — if it passes, R-9 closes and may yield a 9th measured
row. Note the earlier "all four variants hang" boot used `interp` and is void regardless.

Original entry follows.

Measured 2026-07-28, first silicon attempt, reproduced across two independent board runs.

`beebs_ns` (BEEBS `ns`, four nested loops linearly scanning a 4-D lookup table) passes the
QEMU parity leg at −O1 (oracle 1184999093, `cjalr=0 ldc-gp=2`) and its **baseline half is
clean and measured** — 88,451 cyc / 62,097 instret, 15/15 passes tied at min instret,
spread 0, correct oracle. Only the capability half fails: `beebs_ns domain ID = 0` prints,
then no END marker in 120 s, both attempts.

- **Not a transfer artefact.** The domain arrived intact — `sha b911a58bd6d7dac0 OK` on the
  first attempt in run 2, matching the locally computed sha of the decompressed binary. The
  controller then started it and it never returned.
- **R-1 predicts PASS and is wrong here.** Neither `ns_keys` nor `ns_answer` is ever written
  by the kernel: `ns_foo` only compares and returns. The same-object load-with-intervening-
  store shape R-1 describes is **absent from the kernel proper**. That puts this rung with
  **R-6** (`beebs_janne`) rather than with R-7 — two hangs R-1 does not account for.
- **"It is the 32 KiB window" is NOT available as an explanation.** That confound was already
  eliminated under R-7: the `rv8_sha512s` control (identical kernel, 16-entry table, default
  4 KiB window, default unrolled path, no bypass) hangs on silicon too, and C-5 is recorded
  as silicon-validated at 32 KiB. Do not re-run that experiment; it has been done.
- **What is actually distinctive is SCALE of the glue prologue.** The passing read-only rung,
  `beebs_bs`, also has initialized tables materialised by the same unrolled `li`/`sd` path —
  but 120 B / 15 entries plus 72 B / 18, against ns's **2 x 2,000 B / 500 entries**. So the
  glue writes ~500 words per table through its carving capability and the kernel then reads
  them through `ldc gp[i]`, a *different* capability register. That is R-1's shape at
  prologue scale rather than loop scale. **This is a hypothesis, not a finding** — the only
  evidence for it is that bs (small, passes) and ns (large, hangs) differ in that dimension,
  and shape-based prediction has been measured non-predictive on this platform.
- **PROLOGUE SCALE REFUTED 2026-07-28 (the pre-registered falsification fired).** C-4b was
  fixed the same day, so `beebs_ns` now takes the large-RO **copy path** at the DEFAULT
  4 KiB window with no knobs: the ~500-store unrolled prologue is replaced by a
  6-instruction loop, and the transferred domain shrank from **3,676 to 2,024** b64 chars.
  Re-run on the board: transfer clean (`sha eac91ea38af6da9a OK`, first attempt, burst=16),
  and it **hangs identically** — no END marker in 120 s, both attempts. So the prologue is
  not the variable, and neither is the 32 KiB window (this build used 4 KiB). Per the plan
  written before the experiment: **stop shrinking.** The difference between `beebs_bs`
  (passes) and `beebs_ns` (hangs) is somewhere else entirely, and R-9 stays open with its
  leading hypothesis dead rather than with a hypothesis that was never tested.
- *Superseded plan, kept to show what was pre-registered:* shrink the tables to
  `[1][5][5][5]` (125 entries, 500 B, still over the 256 B threshold so the same code path,
  still inside the offset limit). If it PASSES, prologue scale is implicated and the
  bisection continues by doubling. If it still HANGS at bs-comparable size, prologue scale is
  refuted and the difference is elsewhere — do not keep shrinking.
- **THREE MORE HYPOTHESES ELIMINATED 2026-07-28, in ONE boot.** Rather than test one
  theory per board session, three variants were built that each change exactly ONE
  property, with data byte-identical to `beebs_ns` where present, and run in a single
  boot with `beebs_ns` itself as the in-boot control. All four hang:

  | variant | changed vs `ns` | silicon |
  |---|---|---|
  | `beebs_ns` | — (control) | hangs |
  | `beebs_nskeys` | reads ONE table, never a second | hangs |
  | `beebs_nsflat` | same 500 elements FLAT, one index level | hangs |
  | `beebs_nssmall` | 125 entries instead of 500 | hangs |

  So it is **not** two cap-table globals in one loop, **not** 4-level nested address
  arithmetic, and **not** table size. `nssmall`'s tables are 500 B — *smaller than
  `beebs_bs`'s* 120 B + 72 B combined data is not, but its per-table 500 B is within
  the same order, and `bs` passes — so a size threshold between them is not credible.

  All three are QEMU-green at −O1 (oracles 3914083333 / 1184999093 / 2711842293) and
  are kept in `ladder-rungs.spec` as a ready-made discriminator set: whatever the next
  hypothesis is, it has to explain why all four of these hang while `bs` and `cover`
  pass.

  **Copy-path hypothesis REFUTED 2026-07-28, without a board session.** The obvious
  remaining variable was the delivery mechanism: `beebs_ns` takes the large-RO COPY
  path (monitor blob) while `beebs_bs` takes the unrolled `li`/`sd` path, and that
  would have explained R-9 and the SQLite board hang with one cause. It does not.
  Checking the generated glue rather than booting:

      beebs_ns        copy-path = yes    hangs
      beebs_nssmall   copy-path = NO     hangs      <- unrolled, still hangs
      beebs_bs        copy-path = no     passes

  `nssmall`'s tables are 500 B and 500 % 8 == 4, so they are not copy-eligible and
  fall to the unrolled path -- the same path `bs` uses successfully. Delivery is not
  the variable. Reading the build output first is what made this free.

  **What is left, and it is now a short list.** The kernel is a linear scan comparing a
  loaded value against a loop-invariant, with an early `return` out of a nest. `bs`
  (passes) is a binary search — same read-only indexed load, but a *computed* index and
  no early exit from a nest. Candidate remaining differences: the early return itself,
  the loop-invariant compare operand, or the fact that ns's index advances by 1 while
  bs's jumps. Test those next, again as a one-boot discriminator set.

- **Repro:** rung `beebs_ns` in `ladder-rungs.spec` carries its own knobs
  (`DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1`); a plain
  `LADDER_RUNGS=beebs_ns LADDER_ONE_BOOT=1 LADDER_DISTINCT_VA=1 run_ladder_perf_fpga.py`
  reproduces it.

### R-6 — `beebs_janne` hangs although R-1 predicts it should pass `OPEN`
BEEBS `janne_complex`: nested data-dependent loops whose conditions are computed **entirely from
locals**, with one `.bss` counter (`jc_iters++`) touched through a single capability register.
R-1 requires a load through one capability register with an intervening store through *another*;
that never occurs here, so R-1 predicts PASS. **The board hangs it.**

- **Repro:** `tests/runtime-qemu/silicon-ladder/beebs_janne_{kernel.h,fpga_app.c,host.c}`,
  `-O1`, oracle 484656629, QEMU-correct through the identical controller.
- **BISECTED 2026-07-27 (`janne_diag`), and the result does NOT fit R-1.** Safety bounds turned
  the hang into a returned diagnostic:

  | slot | board | correct |
  |---|---|---|
  | outer trips | **200** (its safety bound) | 9 |
  | inner trips | **500** (its safety bound) | 12 |
  | final `a` | **2** | 31 |
  | final `b` | **-339** | 27 |
  | `jc_iters` | 700 (= 200+500, self-consistent) | 21 |

  Neither loop terminates, and `a` is frozen at 2 — after 200 outer iterations of `a = a + 2` it
  should be ≥ 400. The board state is internally consistent (`a`=2 and `b`=−339 keep both
  conditions true forever), so the loops behaved *exactly* as if `a` stopped accumulating.

  **The damning part: the loop nest is pure register arithmetic.** Verified in both the emitted
  assembly and the shipped `.dom` — `a`=`a3`, `b`=`a2`, the counter accumulates in `a6`, and
  `jd_iters` is stored **once after** the loops. There are **no memory operations inside the
  nest**. R-1 is a memory hazard and therefore cannot explain this.

- **Status: mechanism UNKNOWN. Do not fold this into R-1.** Candidate explanations, none tested:
  a control-flow/branch-resolution issue on this RTL (the nest is unusually branch-dense); an
  interrupt landing inside the measured bracket (the measurements doc notes ~16k cycles when one
  does; this rung ran 11,167); or the emitted code differing from what actually executes.
- **Next probe RUN (`regloop_diag`, 2026-07-27) — and it PASSES, which deepens the puzzle.**
  A staircase of register-pure loops, no memory in any body:

  | probe | board | correct |
  |---|---|---|
  | simple counted loop, 100 trips | 100 | 100 ✅ |
  | nested 10x10 | 100 | 100 ✅ |
  | data-dependent branch in body | 100 | 100 ✅ |
  | multiply in body | 100 | 100 ✅ |
  | **janne's EXACT nest, bounded** | **21** | 21 ✅ |

  So pure-register looping is fine, and **janne's algorithm itself runs correctly on this board**.

- **The open puzzle: two register-pure implementations of the same nest, one fails, one works.**
  Verified by counting memory ops in the loop *region* (not the whole function): `janne_diag`'s
  nest has **0**, and it fails; `regloop_diag`'s dbg4 nest also has 0, and it passes. The
  differences are incidental — three counters vs two, bounds 200/500 vs 400, and dbg4 executes
  after four other loops. Nothing algorithmic.
- **Most likely reading: this is the known code-layout / perturbation sensitivity**, the same
  phenomenon as the 2026-07-26 controlled A/B where **four added instructions flipped a passing
  rung from correct to wrong**. That makes R-6 a *symptom class* rather than a distinct fault, and
  means **a passing rung is not stable ground** — already the standing caveat in
  `ref/fpga-silicon-measurements-for-paper.md` §5.
- **Do not merge R-6 into R-1** (R-1 is a memory hazard; these nests touch no memory), and do not
  claim it is understood. The honest status is: janne's algorithm works, one particular build of
  it does not, and the discriminator is not algorithmic.

### R-3 — Second domain at the same entry VA hangs within one boot `WORKED AROUND`
A domain reused at entry VA `0x10000` within a single boot silently hangs its `cscall` —
a missing icache invalidate on the domain switch. This forced **one full power-cycle +
JTAG firmware reload per rung** (~2.5 min), the dominant cost of every board sweep.

- **RESOLVED IN PRACTICE 2026-07-28: the fault is ADDRESS-KEYED.** Domains linked at
  *different* entry VAs run back to back in one boot. `beebs_bs` @`0x10000` then
  `beebs_prime` @`0x20000`, no power-cycle between them, both returned their oracles.
  Nobody had tested this; the per-rung power-cycle was an assumption, not a measurement.
- **Validated as measurement-safe, not merely correct.** The obvious risk was that a
  second domain runs with an icache warmed by the first, so cycle counts would not be
  comparable to the published first-domain numbers. A reversed-order control says no:

  | rung | as 1st domain | as 2nd domain | spread |
  |---|---:|---:|---|
  | `beebs_bs` | 2,258 / 2,246 | 2,263 | 0.75 % |
  | `beebs_prime` (−O1) | 9,746 | 9,749 | **0.03 %** |

  `instret` was byte-identical in both positions (875, 2,708).
- **A wedged rung poisons the rest of the sweep unless recovery is enabled.** On
  2026-07-28 `rv8_primes` hung and the runner kept "reusing" the dead boot, losing the
  **four** rungs after it — all of which had worked minutes earlier. Fixed: a rung that
  times out clears the boot flag so the next one power-cycles. One failure now stays one
  failure. Anyone re-implementing one-boot mode must include this.
- **How to use it:** `LADDER_DISTINCT_VA=1` on the build (assigns `0x10000`, `0x20000`, …
  64 KiB apart) **and** `LADDER_ONE_BOOT=1` on the runner. Both are opt-in: if the
  address-keying assumption ever fails the symptom is a silent hang that looks like a
  rung result, so this must not become a default without a control rung in the sweep.
- **Impact:** a 13-rung sweep goes from ~13 boots (~35 min) to **1** (~5 min).
- **Not a root fix.** The monitor still lacks the icache invalidate on domain switch, so
  same-VA reuse still hangs. Sidestepped, not repaired — the fix sketch remains in
  `plans/curried-crunching-gizmo.md`.
- **Mechanism note:** the domain-boundary `fence.i` was long suspected to fix R-1 as well;
  board test #63 disproved that. It remains the right fix for **this** issue only.

### R-4 — A shared-region word is silently corrupted `OPEN`
`rv8_primes` returned the *correct* result while a word of its shared region held a stray DRAM
address. Passing rungs were only ever clean where someone looked.
- **Evidence:** `ref/fpga-silicon-measurements-for-paper.md` §5

### R-5 — Illegal/meaningless capability ops wedge rather than trap `OPEN`
M-mode appears to spin (`capstone_error` = `while(1)`); only a power-cycle recovers. Seen for
`C_GEN_CAP` (QEMU-only op), for the R-2 `delin`, and for an `scc`-derived load.
- **Evidence:** `history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md`

---

### C-13 MECHANISM FULLY CHARACTERISED 2026-07-29 — the glue reads the blob 96 bytes too low

**The copy WORKS. The blob is present. The glue looks in the wrong place.**

Board-measured with a probe rung (`blobpeek`, INTERP_DIAG_STAGE=11) that publishes one
word of the blob as the domain's retval:

    domain offset +96  -> 0   == descriptor built_flag
    domain offset +104 -> 1   == descriptor count
    domain offset +0/+8/+32/+48 -> all 0 (the zeroed dom_seal tail)

So the descriptor's byte 0 sits at domain offset **+96**, not 0. 96 is exactly
`code_size - gpoff` (4192 - 4096) — the blob size.

Corroborated by three independent measurements that all agree:
  - `sp` region size          125,440  (source model says 125,344)
  - `sp.base mod 128 KiB`     5,632 = gpoff + DOMAIN_DATA_SIZE
  - monitor's dom_data.base   5,728 = code_size + DOMAIN_DATA_SIZE  (from source)
A monitor-written gradient at dom_data[0..255] was invisible at domain +0 — and the
words at +96/+104 hold the real descriptor, i.e. the copy overwrote the gradient. Both
facts confirm the copy executes and lands at dom_data.base.

**Consequence, and why every earlier symptom followed:** the glue reads `count` from its
own base+8, which is 88 bytes below the descriptor, inside the zeroed seal region. It
gets 0, takes `beqz s4, 99f`, skips the whole table build, never establishes gp, and
domain_main faults on its first `ldc gp[i]`. Stage 7 "passed" only because built_flag's
correct value is also 0 — zeroed memory is indistinguishable from a correct read there.

**NOT YET EXPLAINED: why sp.base != dom_data.base.** create_domain computes
dom_data = __split(dom_seal, base + code_size + DOMAIN_DATA_SIZE) and seals it into slot
2; code_size is 4192 (verified: image_size 4184 for every probe domain, controller passes
it through, module forwards untouched, dispatch is arg0..arg4 in order, no reassignment
after the 16-byte rounding). Yet the domain's sp starts at base + gpoff + DOMAIN_DATA_SIZE.
The discrepancy is exactly code_size - gpoff, i.e. ONE TERM uses gpoff where the other
uses code_size. Source reading has not located it; three separate attempts derived the
wrong answer, which is why this entry records measurements rather than a mechanism.

**FIX OPTIONS**
1. Make sp == dom_data (correct fix). Requires finding the term above; look at the
   seal->cscratch handoff and the domain switcher, not at create_domain's arithmetic.
2. Make the glue LOCATE the descriptor instead of assuming offset 0: put a magic word
   first in .capstone_gp_initdesc and have the glue scan the first few words for it.
   Robust to any displacement, contained to compiler + glue, and unblocks SQLite without
   resolving (1). Pragmatic given the deadline.

### R-10 — a 16-byte capability copy MANGLES plain scalar data in its high half `ROOT CAUSE of C-13, board-confirmed 2026-07-29`

**THE MECHANISM, complete.** A capability's two halves are stored differently:

    low  8 bytes = cursor   -> written RAW      (wt_axi_adapter.sv:201, axi_wr_data[0] = dcache_data.data)
    high 8 bytes = metadata -> written ENCODED  (stored as compress_cap(...), ariane_pkg.sv:809)

`compress_bounds` (`ariane_pkg.sv`) is a genuine LOSSY encoder: leading-zero count, an
exponent E, and truncation to 21/14/12-bit fields. It is closed over real capabilities
and **not** over arbitrary bit patterns.

So the monitor's copy `dom_data[ci] = dom_code[gpoff_c + ci]` (`sbi_capstone.c:400-404`,
both `__linear void *`, i.e. one `ldc` + one `stc` per 16 bytes) does
decompress-then-recompress on the high half of every granule. Plain scalar data does not
survive it. The low half, being the raw cursor, does.

**BOARD-CONFIRMED, 4 rungs in one boot:**

    stage 7  reads blob +0 and USES it                    PASS  (582955588)
    stage 10 reads blob +8 and DISCARDS the value         PASS  x2
    stage 8  reads blob +8 and USES it as `count`         FAIL

The load does NOT fault -- stage 10 performs the identical access and passes twice. The
VALUE is wrong. A mangled `count` then makes `slli`/`sub`/`split` carve at a wild bound,
which is the wedge. The image descriptor is verified correct (built_flag=0, count=1), so
the corruption happens in the copy, not the compiler.

The monitor's own comment -- "the image bytes here are const initializer data with no
capability tags, so the 128 bits round-trip unchanged" -- is FALSE on real silicon.

**Secondary defect, same root.** `is_cap_req = |dcache_data.user`
(`wt_axi_adapter.sv:196`) and `st_wr_cap = |wr_user_i` (`wt_dcache_mem.sv:138`) decide
"holds a capability" by OR-reducing the metadata word; neither file references `cap_type`
(0 occurrences in each), so `cap_type == NOT_CAP` is never consulted. A consequence worth
noting separately: when the high half is ZERO, `is_cap_req = 0` sets `axi_wr_blen = 0`
(`:209`), so only ONE beat is written and the high 8 bytes are left at whatever was in
DRAM. That also means `dom_seal[i] = 0` zeroes only half of each granule.

**QEMU cannot reproduce any of this.** It stores exact fat structs with a discrete tag
(`cap.h:93`, `cap_mem_map`); there is no lossy codec and no content-derived tag. Third
RTL/QEMU divergence to cause a multi-session blocker, after DELIN and this.

**FIX IMPLEMENTED 2026-07-29, AND IT DID NOT UNBLOCK THE REAL PATH.** Root cause of the
16-byte copy turned out to be a capstone-c DECLARATOR BUG, not a design choice:
`__linear void *mem_l, *dom_code, *dom_data, *mem_r;` accumulates the `*` across
declarators (dag_builder mutates the shared decl type and never resets it), so only
`mem_l` got `void *` -- `dom_code` became `void **` and `dom_data` `void ***`.
Dereferencing them therefore yielded a POINTER (16 B), which is why the copy emitted
`ldc`/`stc` at all. Fix = one declarator per declaration, plus `>> 4` -> `>> 3`.
Verified by regenerating the monitor: exactly 6 instructions change in 4,653, and the
two that matter become scalar `ld`/`sd`. Confirmed present in the shipped firmware
(size 17,466,376, create_domain labels 30, `ld a4, 0(a3)` / `sd a4, 0(a7)` in the loop).

**Board result with that firmware: real interp STILL FAILS.** `beebs_primer1` and
`gpstress`, both real interp, both no END marker. The primer domain was byte-identical
to the one that failed before (sha 3e3980cd), so the monitor was the only variable.

So the copy corruption was REAL and board-confirmed (stage 8 fails / stage 10 passes on
the identical load), and fixing it is NECESSARY -- but it is NOT SUFFICIENT. Something
else also breaks the real path. **Do not record C-13 as fixed.**

Next experiment, one variable: re-run the stage ladder against the FIXED firmware.
Stage 8 (reads `count` from blob+8, the previously-mangled half) is the discriminator --
  stage 8 now PASSES -> the copy fix repaired the read; the remaining fault is downstream
                        in the record reads / gp-park / cap-init, all of which have knobs
  stage 8 still FAILS -> the copy fix did not repair the read and the mechanism story
                        above is incomplete despite being individually verified

**FIX DIRECTIONS (not yet implemented):**
1. *Monitor copies scalars with scalar accesses.* The correct general fix -- it also fixes
   the bulk initializer data, which matters at SQLite scale (1,059 globals). Open question
   is whether capstone-c can express a non-`__linear` view of the same span so the
   compiler emits `ld`/`sd` instead of `ldc`/`stc`. `sbi_capstone.c` has no `memcpy` and
   no scalar-pointer cast today. UNVERIFIED.
2. *Descriptor avoids metadata halves.* Lay the descriptor out so every 8-byte scalar sits
   in the LOW half of its own 16-byte granule. Purely a compiler+glue change, no monitor
   change. Fixes the descriptor but NOT the bulk initializer data, so it unblocks the
   glue and not SQLite's globals.

Both may be needed: (2) to unblock quickly, (1) for correctness at scale.

**Confirmed by direct quote, verified independently:**

    core/cache_subsystem/wt_axi_adapter.sv:196   assign is_cap_req = |dcache_data.user;
    core/cache_subsystem/wt_dcache_mem.sv:138    assign st_wr_cap  = |wr_user_i;

Both decide whether a 16-byte granule "contains a capability" by **OR-reducing the whole
64-bit metadata word**. Neither file references `cap_type` even once (0 occurrences in
each, checked). So the architectural notion of a capability — `cap_type != NOT_CAP`,
bits [30:28] of the metadata word (`ariane_pkg.sv:646`) — is **never consulted by the
memory subsystem**. The shadow tag is set from raw bit content.

**Consequence.** Any 16-byte capability-width store (`stc`) whose HIGH 8 bytes are
nonzero marks its destination granule as holding a capability, even when the value is
plainly not one. Copying ordinary scalar data with `ldc`/`stc` therefore poisons the
shadow tag across the whole copied region.

**Where this bites us.** The monitor copies the entire globals blob into `dom_data` with
16-byte capability accesses (`sbi_capstone.c:400-404`, `dom_data[ci] = dom_code[...]`,
both `__linear void *`). Its own comment asserts the bytes "round-trip unchanged" — the
BYTES do; the shadow tag does not. For the descriptor, `count = 1` sits in the high half
of granule 0, so `|1 = 1` and that granule is mis-tagged. For SQLite (1,059 globals, most
initialized) it would be most of the blob.

**QEMU cannot reproduce this class at all.** Its capability tag is a discrete per-register
boolean plus a side table (`cap.h:93`, `cap_mem_map`), content-independent, and
`helper_compress_cap` returns 0 for an untagged source (`op_helper.c:1155-1164`), so the
destination is never marked. Same shape of blind spot as the DELIN divergence (C-13).

**NOT yet established:** that this mis-tagging is what *wedges* the board. The data plane
reads symmetrically (`wt_dcache_mem.sv:261`, banks muxed by address bit 3) with no fault
found tied to the tag, and no explicit fault condition was located in `wt_dcache_ctrl.sv`
or `wt_dcache_missunit.sv`. The board experiment that separates "the load faults" from
"the value is wrong" is INTERP_DIAG_STAGE=10 (see C-13). Do not write this up as C-13's
cause until that lands.

**Unread, and needed to close the mechanism:** `capstone_dyn_unit.anvil` /
`capstone_unit.anvilh` for the `_load_ep_res` vs `_load_ep_normal_res` handshake —
`ex_stage.sv:791` decompresses EVERY load's result, not just `ldc`, and forwards it to
the DYN unit on a channel whose ack is left dangling (`ex_stage.sv:910`).

**Fix direction (unresolved):** M-mode must copy plain scalar data with scalar stores
rather than `ldc`/`stc`. Whether capstone-c can express a non-`__linear` view of the same
span is a compiler/ABI question, not an RTL one, and is unverified — `sbi_capstone.c` has
no `memcpy` and no scalar-pointer cast anywhere today.

### R-14 — struct-array init wedges `OPEN — REPRODUCED ON SILICON 2026-08-03; title is now WRONG`

> **2026-08-03 — read `ref/SILICON-BLOCKER.md` first.** Reproduced with BOTH controls passing
> in one boot (`f10ctl=0 | :0=0 | :144=WEDGE`, 2/2). The wedge is a **capability
> OUT_OF_BOUNDS fault (mcause=28) taken into M-mode**, where the M-mode side hangs — NOT a
> revocation-node stall (`wrev=0`, `serving_idx=0`, rev head 602/1023, `overflow=0`).
>
> **This heading no longer describes the fault.** Refuted since it was written:
> * *"distinct string constants"* — `:143` stores the SAME literal 8x and still wedges, and the
>   standalone `r14b` fails with string merging OFF (`cl::init(false)`, never set by
>   `build-ladder-domain.sh`). Merging is not necessary for the fault.
> * *"straight-line"* — `r14b_app.c` records the opposite: its four STRAIGHT-LINE entries pass
>   and its twelve LOOP-ASSIGNED ones fail.
> * *offset, and store count* — `:147` (2 stores at high offsets) and `:148` (3 stores) both
>   return correctly.
>
> **The fault is NONDETERMINISTIC**: the same source arm `:141` returned 1 (3 boots), wedged,
> and returned 0 across images whose frames are byte-identical. Any statement of the form
> "N stores wedge" is unsafe — one such boundary was already retracted.
>
> Current reading (INFERRED): a capability stored to the stack array is not reliably usable on
> read-back — sometimes correct, sometimes null (returns 0), sometimes right-address/wrong-bounds
> (dereference => the measured OUT_OF_BOUNDS). Next probe is `:150`, still unmeasured after 5
> images because R-16 blocks it. Prefer the **standalone** repro (`r14b.dom`, 10896 B, 10
> carves) over the SQLite-derived images (1624128 B, 181 carves).

**MINIMAL CASE, control-validated on silicon 2026-08-02.** Four straight-line assignments of
distinct string literals into a two-capability struct array. No SQLite, no allocator, ~10 lines:

```c
struct kv { const char *z; const char *y; };      /* 2 capabilities, 32 B, no tail padding */
struct kv a[64];
a[0].z="ltrim"; a[0].y="aaa0";   a[1].z="rtrim"; a[1].y="aaa1";
a[2].z="trim";  a[2].y="aaa2";   a[3].z="max";   a[3].y="aaa3";
for (i=4;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
for (i=0;i<16;i++) if (a[i].z && a[i].y && strlen(a[i].z)>0 && strlen(a[i].y)>0) ok++;
return ok;                                        /* expect 16; silicon: WEDGES */
```

* **Control-validated**: in the SAME boot and image, a trivial `return 0` (selector `:0`)
  RETURNED `rc=0` immediately before this wedged. So the wedge belongs to the construct, not
  to the image or the boot.
* **N as low as 4** is enough — so clamping the count is not a workaround.
* **QEMU-clean** at `-O0` and `-O1` (returns 16) with the C-16 fix in place, so this is NOT the
  untagged-capability-arithmetic class that QEMU asserts on.
* **Rungs**: `tests/runtime-qemu/silicon-ladder/r14a_app.c` (16 straight-line) and
  `r14b_app.c` (4 straight-line), each with a native host oracle; board equivalents are
  selectors `:110` / `:111` of any staged SQLite probe image.
* **Same fault reaches SQLite**: `f10:0` and `f10:9` returned `rc=0` while `f10:10`
  (`sqlite3MallocInit` + `sqlite3RegisterBuiltinFunctions`) wedged, in one boot.
  `sqlite3RegisterBuiltinFunctions` builds exactly this shape.

**Not established**: attribution. QEMU executes it correctly and silicon does not, which is
consistent with hardware but is precisely the pattern C-16 showed before turning out to be a
compiler bug of ours. Do not present as a hardware defect without further evidence.


**Read C-16 first.** The *SQLite* blocker behind this entry is now root-caused and FIXED: it was
a compiler bug (`memset` destination typed in AS0, stripping the capability tag), not hardware.
Stage 10 and the full SQLite QEMU gate now pass with no workaround.

**But R-14 is NOT simply closed by that**, and the difference matters:

* C-16 needs a struct with **tail padding**, because the trigger is the initialiser's
  padding-zeroing `memset`. Variant A below is `struct{2 ptr}` = 32 bytes with **no tail
  padding**, so no `memset` is emitted and C-16 does not explain it.
* Variant D (flat `const char*[64]`, also no padding) is correct, so "struct vs flat" is still
  an unexplained axis.

**UPDATE 2026-08-02 (post-fix, measured):** the re-run happened. Variants A and B still fail on
silicon with the C-16 fix in place, and both are QEMU-clean:

    QEMU (fixed compiler)   r14a -O0/-O1 -> 16      r14b -O0/-O1 -> 16
                            stages 110/111/112/113 from one image -> 16 each
    BOARD (fixed compiler)  variant A (r110) -> IN-DOMAIN WEDGE after SQ: G/enter
                            variant B (r111) -> IN-DOMAIN WEDGE after SQ: G/enter

So **R-14 does not close as a duplicate of C-16** — it is a separate, silicon-only fault that
QEMU cannot see. Note variant B previously *returned 4*; as a staged probe it wedges instead,
but those are different binaries (standalone fpga-repro vs. the same shape inside the SQLite
amalgamation), so that is NOT evidence the fix made anything worse.

The C and D **controls are still unmeasured post-fix** — every attempt to run them was killed by
the R-16 entry stall before the domain executed. Until they run, "both ingredients required"
rests on pre-fix data.

New QEMU-gated rungs, one source building both a QEMU domain and a board domain:
`tests/runtime-qemu/silicon-ladder/r14a_app.c`, `r14b_app.c`, `r14d_app.c` (+ `_host.c` oracles,
all 16).

**Required next step:** re-run all four variants below with the fixed compiler. If A and B now
pass, R-14 closes as a duplicate of C-16 and the "confidence it is hardware" note was right to
stay unconvinced. If A still wedges, R-14 is a genuinely separate defect and everything below
still applies. Until that re-run, treat the variant table as PRE-FIX data.

**CORRECTIONS 2026-07-31 (wide audit, all verified against source):**

* **The candidate mechanism is REFUTED by our own capture.** The load-syncer arming leak
  (`capstone_dyn_unit.anvil:302-307`, commit `3a59ac52c485`) requires `req_set == 1` to
  persist. `board-regs.log` decoded and printed `load_syncer_req=0` and `store_syncer_req=0`
  on the wedged core. It was read and not noticed. The asymmetry at `:306` is still a real
  one-line difference from `STC:369-370`, but it is NOT this failure.
* **"The core stops retiring" was never measured.** `cva6.sv:500` — `ex_commit` is
  `// exception from commit stage`, wired to `.exception_o`. `ex_commit.valid = 0` means no
  exception is committing, nothing about retirement. The bit that does report retirement,
  `commit_instr_id_commit[0].valid`, is in bank `debug_byte_sel = 3'b110` and has never been
  sampled.
* **`stall_issue = 1` is not evidence of a hang.** `issue_read_operands.sv:390` —
  `stall_issue_o = stall_raw[0]`, a RAW hazard. `strlen`'s loop is four mutually dependent
  instructions, so `stall_issue = 1` is its steady state while RUNNING.
* **The evidence was double-counted.** The register capture attributed here to an
  independent "20-line synthetic" is `sqlite_silicon.dom` built as stage 18 — a SQLite
  staged build, not a separate artifact. The two lines of evidence are one.

Consequence: the failure class may be a LIVELOCK IN DOMAIN CODE rather than a core
deadlock, and no experiment run so far distinguishes them — every probe either returned or
produced silence. Sampling `debug_byte_sel = 3'b110 / reg_sel = 0` (retirement) on a wedged
core is the measurement that would.

A 20-line C function with no SQLite in it wedges the core: no return, no output, no reported
trap. It is the blocker behind `sqlite3RegisterBuiltinFunctions`, which is where the SQLite
domain stops on silicon.

Four variants differing by exactly one variable each (board-measured 2026-07-31):

| variant | shape | result |
|---|---|---|
| A | 16 distinct literals, **straight-line**, `struct{2 ptr}[64]` | **WEDGE** |
| B | 4 distinct straight-line + loop filler, same struct | **returns 4**, expected 16 |
| C | 16 distinct via **loop from a static table**, same struct | returns 16 (correct) |
| D | 16 distinct **straight-line**, flat `const char*[64]` | returns 16 (correct) |

So it needs **both** straight-line materialisation **and** the struct element type; either
alone is fine. **Variant B is the important one** — it returns a WRONG VALUE instead of
hanging, i.e. the same construct corrupts silently at smaller scale, with the twelve
loop-assigned entries failing and the four straight-line ones passing.

- **Repro:** `tests/fpga-repros/R14-strline-struct/` (source, run recipe, and the rebuild
  commands for the four domains — the `.dom` files themselves are ~1.5 MB each and are not
  tracked). Put variant A last in any batch — a wedged domain takes the core with it.
- **Wedged-core state:** `privM=1`, `flu_ready=dyn_ready=lsu_ready=1`, `ex_commit.valid=0`,
  `stall_issue=1`, all other status bits 0; commit pc = image VA `0x14c71c`, the `bnez`
  closing `strlen`'s loop. Selectors verified against `cva6.sv:1090-1215`.
- **Candidate mechanism, NOT established:**
  `history/31-07-2026_18-30-00_ldc-load-syncer-arming-leak.md`. `capstone_dyn_unit.anvil:306`
  arms the load syncer and never disarms it on the `NOT_CAP` path, while `STC:369-370` does.
  A stale arming on a 3-bit `trans_id` would make a later unrelated load be consumed instead
  of forwarded — which matches "stalled at issue, every unit ready, nothing committing"
  exactly. **The asymmetry is verified by quote; its role here is not.** That arm raises
  cause 24, which would have overwritten the latched cause-9 in the trap log, and did not;
  and variant B's selective corruption fits a swallowed load poorly.
- **Confidence it is hardware:** NOT established. It could equally be our codegen for
  straight-line capability materialisation into adjacent struct fields. Do not present it as
  a hardware defect until the trigger is settled.
- **Open question, not answerable from this tree:** does a pipeline flush reset `req_set` /
  `cap_trans_id` in the load/store syncers (`capstone_dyn_unit.anvil:521-522`)? Only the
  `.anvil` is present here, no generated Verilog. If it does not, any capability access
  abandoned between `send cap_load_ri.init(...)` (`:302`) and its `req`/`res` pair
  (`:343-345`) leaves an 8-value comparator armed that will match and consume an unrelated
  later load.
- **Workaround, board-validated:** variant C passes. Building the array **in a loop from a
  static table** instead of straight-line avoids it. Applying that shape to the patched
  `capstoneBuiltinFunc[]` is the obvious next move and needs no RTL change.
- **Impact:** SQLite cannot complete `sqlite3_initialize()` on silicon.


### C-16 — `memset` destination typed in AS0 strips the capability tag `FIXED 2026-08-02`

**This was the SQLite blocker.** `SelectionDAG::getMemset`
(`llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp:9380`) built the destination argument type with
`PointerType::getUnqual(Ctx)` — an **addrspace(0)** pointer. AS0 here is a 64-bit integer
address while the real destination is an AS200 128-bit capability, so the declared argument type
is narrower than the value and call lowering inserts a `TRUNCATE` of the pointer.

    %8:gpr  = PseudoTRUNC_CAP %5      ; truncate the array base -- TAG GONE
    %9:gpr  = ADDI killed %8, 49      ; tail-padding address
    $x10    = COPY %9                 ; passed as memset's destination
    %13:gpr = CIncOffsetImm %5, 64    ; next element -- CORRECT, tag preserved

`memset`'s own `p++` is then `cincoffsetimm` on an untagged base. **QEMU asserts on that; the
RTL does not check a `cincoffset` base at all** (`SPLIT`, `LDC`, `STC` all validate their
operands, `cincoffset`/`cincoffsetimm` do not) — so on silicon the untagged pointer is used and
`memset` writes through a garbage address while execution continues. Silent memory corruption,
once per array element.

Triggered by any **struct with tail padding in an aggregate initialiser**: the initialiser
zero-fills the padding via `memset`. `sqlite3RegisterBuiltinFunctions`' `FuncDef` array is
exactly that shape.

- **Fix:** take the address space from `DstPtrInfo` (already in scope, already used for
  `checkAddrSpaceIsValidForLibcall`). No-op for AS0 targets.
- **Repro / regression test:** `tests/runtime-qemu/silicon-ladder/strarray_app.c` +
  `strarray_host.c`, oracle 420. `DOMAIN_OPT_LEVEL=-O0 bash run-ladder-qemu.sh strarray`.
  ~1 minute, no board.
- **Verified:** codegen `addi ...,49` x8 -> 0, replaced by `cincoffsetimm`; reproducer PASS
  (retval 420); **stage 10 non-static returns rc=0x00**; **full SQLite QEMU gate passes with
  `SQLITE_STATIC_BUILTINS` unset**.
- **Why it hid so long:** the staged probes were built and shipped to the board for four
  sessions without ever being run under QEMU — the one tool that would have asserted on it.

---

### R-16 — domain never returns from its FIRST entry (`SHA5` stall) `OPEN — still unexplained 2026-08-03`

> **2026-08-03.** Now separated from board health for the first time: run a KNOWN-ENTERING
> image (`f10.dom:0`) as the FIRST domain of every boot. Measured `f10ctl=0` while the image
> under test stalled in the same boot, on the same firmware — so R-16 is a property of the
> IMAGE, not of the board or firmware. Every stall verdict must carry such a control; a boot
> whose control fails is VOID (the control itself wedges ~1 in 5).
>
> **Not strictly per-image, either:** `q145` entered for `:0` and hung at `:146`, and `c142`
> entered for `:0` twice and stalled on `:150` twice — same binary, same boot. So it tracks
> which invocation runs too, and "deterministic per image" overstates it. Retrying the same
> binary is still futile; REDRAW instead.
>
> `SHA5` last does NOT by itself mean an entry stall — a domain that enters and wedges
> immediately leaves `SHA5` last too. Distinguish on `SQ: G/enter`: present => it ran.
>
> Still unexplained: carve count, `.text` size, merged-string bytes, dom_data geometry, and
> "carries the ladder block" all fail to separate entering from stalling images. It has blocked
> 2 of 3 minimisation arms all night, so it **biases which constructs are measurable at all**.

**UPDATE 2026-08-02 23:1x — two corrections, both narrowing this entry.**

1. **Do not count any "entry stall" from 21:00-22:33 as an R-16 instance.** In that window
   `board-watchdog.sh` matched a `SHA5` from the console's replayed previous-boot scrollback and
   killed runners seconds after `load_image`, before the board booted. 13 of 13 checked runs in
   that window have ZERO `SHA` markers after their own `load_image` and 50 before it. The
   watchdog is fixed (run-scoped scan + `load_image` gate); the affected runs are
   `waa/wab/wac`, `tsp/tsq/tsr`, `kg1/kg2`, `sllog-*`, `rflog-*`, `pzlog-*`.
   In particular the conclusion "the board stopped accepting any image" is **refuted**.

2. **R-16 is not currently blocking.** At 22:50, on a freshly reflashed board with the fixed
   watchdog, a three-domain ladder ran: `f10:0` returned `rc=0`, `f10:9` returned `rc=0`, and
   `f10:10` wedged in-domain. So domains enter fine right now, and the SQLite blocker (R-14
   shape) reproduces cleanly with two controls returning in the same boot.

Also: the bullet below saying `r110`/`r111` "each entered **1/1** only" understates the entering
side — `r110` entered **3/3** in the 19:05-19:20 repeat test (control returned each time). The
"per-BOOT coin toss not excluded" caveat still stands, but it is a weaker doubt than written.

**Now the primary blocker for the whole measurement campaign**, ahead of R-14 and ahead of
SQLite itself. The monitor completes a region share and hands off; the domain never comes back.
Last UART line is `SHA5:xxxx`.

`SHA5` = "about to leave M-mode for the domain", `SHA6` = "the domain returned from the share
entry" (`sbi_capstone.c:111`, `:1020-1026`). A stop between them means the monitor is exonerated
and the domain died on its FIRST entry — which is where the glue builds the cap table (one
`split` per global) and runs `__capstone_cap_init`. **The domain's own code never runs**, so such
a run carries NO information about the domain under test and must never be recorded as one.

- **Not QEMU-visible.** Every image that stalls on the board runs clean under QEMU.
- **Per-image repetition, but the "entering" side is thin:** `x101` stalled 6/6, `r112` 3/3,
  `r113` 1/1, `v110` 1/1, `st10` 1/1; `r110`/`r111` each entered **1/1** only. So it is
  reproducible for stalling images and merely assumed for entering ones — a per-BOOT coin toss
  is not excluded.
- **Ruled out as discriminators (all MEASURED):** dom_data geometry — `r110` (entered) and
  `r112`/`r113`/`v110` (stalled) have byte-identical blob/cap-table/storage/stack/globals-offset,
  as do `r111` (entered) and `st10` (stalled); also carve count, `.text` size and merged-string
  bytes.
- **It defeats the runtime-selector workaround.** One image carrying all probes dispatches
  correctly under QEMU, but if that image stalls, selection never gets a chance — and any rebuild
  is a fresh draw, so "it enters" cannot be carried across builds.
- **Retrying the same binary is futile** (three boots spent on `r112`). Retry is correct for an
  `__CAPSTONE_INFRA_FLAKE__`; for an entry stall, change the binary or the order.
- **Position:** slot 2 stalls ~10x more often than slot 1 (32% vs 2.8% over 274 launches), but
  those are pooled figures across many binaries and should not be used as a per-image probability.

**Next step:** it is board-only and not reproducible offline, so it needs instrumentation rather
than a reproducer. Every board session should run `tests/rtl-smoke/board-watchdog.sh` alongside
the runner so a stall is distinguishable from a dead runner and from normal work while it happens.

---

## Infrastructure / procedure

### I-1 — A sweep silently rebuilds at −O0 and discards your pre-built set `FIXED`
`run_ladder_perf_fpga.py` **rebuilds every artifact by default** (the 25-07 anti-stale fix),
shelling out to `build-ladder-fpga.sh` with the inherited environment. Setting `LADDER_OPT`
on a *pre-build* and omitting it from the *sweep* means the runner rebuilds everything at its
`-O0` default and measures that — against baselines specified at another level.

- **Cost when it fired (2026-07-27):** five rungs reported as silicon failures, including one
  that had passed before; a false conclusion that **R-1's same-object clause was refuted**,
  which would have gone to the board owner as a correction to the bug report; and a nearly
  published §5 claim that *an ordinary rebuild flips a passing rung*. All three withdrawn.
- **Caught only by the in-sweep control.** `beebs_bs` was included purely as a stability
  check; its failure is what made the sweep suspect instead of informative.
- **Rules:**
  1. Set `LADDER_OPT` on the **runner** invocation, not just the pre-build.
  2. Keep a **known-good rung in every sweep**. It is the only thing distinguishing
     informative failures from a misconfigured harness.
  3. `LADDER_REBUILD=0` is **required** to run a specific pre-built binary — pointing
     `LADDER_FPGA_DIR` at it does not stop the rebuild from overwriting it.
  4. Compare the static shape (`.text` size, `ldc gp[i]` count) against the known-good build
     before believing a flipped result.
- **Static signature of the mistake** (`beebs_bs`): −O0 → 2,100 B text, 4 `ldc gp[i]`, 2
  cap-table globals, FAILS; −O1 → 1,408 B, 2, 1 global, PASSES. The function-local
  `static const int probes[18]` becomes a delivered cap-table global at −O0 — the C-4
  boundary moving under an optimization flag.
- **Evidence:** `history/27-07-2026_22-40-00_RESULTS-two-new-silicon-rungs-and-an-O-level-procedure-bug.md`

### I-2 — Linux baseline served interrupts inside the bracket `FIXED`
**Fixed 2026-07-28 by removing the OS**, not by modelling the error. The baseline now runs
as a bare-metal S-mode OpenSBI payload (`build-ladder-base-bare.sh`,
`fpga_driver/run_base_bare_fpga.py`).

- **Proof:** the `ctrsanity` control -- identical 5-instruction loop on both sides -- reads
  **600,041 cyc bare vs 600,309 cyc capability, ratio 1.000** (Linux was 728,727, 1.21x).
  Quality went from 1/15 passes tied at min instret to **15/15 with spread 0**.
- **Consequence: every published overhead ratio rose.** `beebs_prime` 1.032x -> **1.683x**,
  `rv8_primes` 1.050x -> **1.263x**, `beebs_recursion` 1.801x -> **1.955x**,
  `beebs_bs` 1.274x -> **1.530x**. Pervasive spatial safety costs **26-96 %**, not 3-5 %.
- **And it refuted a second claim:** with a clean baseline, `rv8_primes` cycles grow
  (1.263x) FASTER than instructions (1.130x) and CPI RISES 1.762 -> 1.970, inverting the
  "overhead is ABI, not enforcement" section.
- **Side benefit:** firmware 15.4 MB -> 2.1 MB, so the JTAG reload that dominates every
  boot is much faster.
- **Bring-up trail (3 silent board sessions):** legacy SBI console absent; DBCN impossible
  (board reports SBI 1.0, DBCN needs 2.0) and the probe read `a0` instead of `a1` anyway;
  fixed by direct ns16550a MMIO with parameters taken from the firmware's **device tree**
  (`/soc/uart@10000000`, `reg-shift=2`). **The FDT had the answer on disk the whole time.**
- **Evidence:** `history/28-07-2026_02-30-00_RESULTS-bare-metal-baseline-works-*.md`

---

### I-3 — diagnostic probes could not run under QEMU `FIXED 2026-07-28`
Diagnostic rungs write raw values into `res[3..47]`. Under QEMU a domain saw only an
8-byte return slot, so every `*_diag` / `rawhazard*` probe was **board-only** — each
iteration cost a full boot and a broken probe could not be caught before spending one.
Two boots on 2026-07-28 produced one data point between them for exactly this reason.

**Root cause, after four failed attempts: `the share IS the entry`.**
`ladder_perf_ctl` says so in its own comment, and it is the whole difference. An
**annotated** region share *invokes* the domain with the REGION as its argument. The QEMU
loader shared a region and then called `call_dom()`, which enters through the plain call
path whose first argument is the 8-byte return slot — so `res[3]` faulted every time.

Attempts that failed first, recorded so nobody repeats them: plain `share_region`;
`shared_region_annotated` (with `REV_SHARED` wrongly passed as `0x0` — it is `0x2`);
adding `map_region` + zeroing. **None of them mattered: the bug was the trailing
`call_dom`, not the share.**

**Fix:** `package/modcapstone/userspace/capstone-diag.c` → `capstone-diag.user`, a
**separate** loader that maps a 4096-byte region, shares it annotated
(`ANNOT_PERM_INOUT`, `REV_SHARED`) — which enters the domain — then reads `res[0]` and
prints `res[3..47]` as a `DEBUG` line.

**Deliberately separate from `capstone-test.c`**, which loads the entire QEMU corpus (82
BEEBS, RV8, CoreMark, SQLite, authority). Changing that file's entry model would move where
every existing domain finds its result. Zero regression surface this way.

**No guest image rebuild needed** — build with the buildroot cross-compiler, drop it in the
9p share:
```
run-domain-smoke.py --domain-loader /mnt/host/capstone-diag.user <rung>.dom
```

**Verified:** `accum_probe` returns all nine slots under QEMU —
`dbg0..dbg6=100, dbg7=3, dbg8=1000`, **9/9 correct** — the probe that produced nothing on
two board boots.

**Consequence:** probe iteration drops from ~2.5 min of a shared physical resource to
seconds of emulation, and R-1's diagnostic family can finally be developed off-board.

## Compiler / toolchain (ours)

### C-2 — `Cannot select: i128 = or` / `= xor`, mixed extends `OPEN (partially widened)`
Blocks `rv8_qsort` and `rv8_miniz` at −O1/−O2 (both still fail 2026-07-28; −O0 passes).

**The semantics question was malformed, and the answer is now settled.** It was framed as
"do the high 64 bits mean capability metadata or a genuine 128-bit integer?" — neither.
`lowerScalarI128Logical` computes the op in XLen and re-extends, which is exact **only while
the i128 carrier's high half is an extension of its low half.** Matching extends preserve that
invariant. Mixed extends break it: for `sext(a) OR zext(b)` the true 128-bit high half is
`sign(a)`, which is **not a function of the low-half result**, so re-extending the narrow
result under *either* rule is a **miscompile**. **The bail is correct. Do not "fix" it by
picking an extension rule.**

- **Widened safely 2026-07-28** (`CapstoneISelLowering.cpp`): when the sign-extended operand is
  **known non-negative** (`DAG.SignBitIsZero`), its sign extension and a zero extension are the
  same bits, so both operands agree and the invariant holds. Covers indices/sizes the optimizer
  has already proven `>= 0`, without assuming anything about meaning.
  Lit `i128-logical-mixed-extend.ll`; **Capstone lit 43/43**.
- **Does NOT unblock rv8.** Re-verified with exit codes: `qsort` −O1/−O2 still
  `Cannot select: i128 = xor`, `miniz` still `i128 = or`. Their signed operand is not provably
  non-negative, so they are the genuinely unrepresentable case.
  > ⚠ An intermediate report that both benchmarks "now build" was **wrong** — that check
  > grepped output for error strings without testing the exit code, so a failing build read as
  > success. Always gate on exit status.
- **What the real fix needs, and why it is not a lowering patch:** the remaining case cannot be
  represented while i128 is carried in a single capability register. Either (a) genuine
  128-bit integers get a register-pair representation distinct from the capability carrier, or
  (b) find why a **64-bit** `or`/`xor` is being widened to i128 at all — if the source only does
  64-bit logic, the i128 node is an artifact upstream of this lowering and should be prevented
  rather than lowered. **(b) is the cheaper investigation and should come first.**

### C-3 — RV8 fails at runtime at −O1/−O2 `OPEN`
**Now also reaches the ladder (2026-07-28):** the `rv8_primes` *rung* runs at −O0 and
**HANGS at −O1** on silicon, so it is the one row in the overhead table that cannot be
measured at the uniform level. Same family as the RV8 −O1/−O2 failures below.
Five RV8 benchmarks now *build* at −O1/−O2 but fail 10/10 at runtime: `primes`/`aes`/`dhrystone`
hang silently; `sha512`/`norx` take deterministic capability faults (cause 5 OOB / cause 24, same
PC at both levels). −O0 controls all pass. **Not regressions** — code that never compiled cannot
regress.
- **Evidence:** `history/27-07-2026_12-59-35_three-codegen-fixes-*.md`
- **Leads:** `sha512` faults with bounds visibly too small; `norx` with an untagged capability
  reaching a load. Both smell like a bounds/provenance codegen bug at −O1+.

### C-4 — split into a FIXED half and a remaining domain-creation bug
Renamed from "large read-only data cannot be delivered": size was never the variable.

#### C-4a — constant pools are unreachable in a domain `FIXED 2026-07-28`
**Root cause, with the emitted sequence:**
```
.LCPI0_0: .quad 81985529216486895        ; .rodata.cst8 -- a CONSTANT POOL entry
  auipc a2, %pcrel_hi(.LCPI0_0)
  addi  a1, a2, %pcrel_lo(...)
  scc   a1, gp, a1     ; set gp's cursor to a .rodata address
  ld    s6, 0(a1)      ; FAULTS
```
A pool entry is **not** a `GlobalVariable`, so it gets no cap-table slot (correctly);
`lowerConstantPool` then falls back to `LGA` → `scc gp`. Under gp-captable `gp` is bounded
to the **cap table itself**, so the cursor lands out of bounds. The tell in the fault line
is that the reported bounds are exactly the table:
`cursor = 0x101561000, bounds = (0x10157ffd0, 0x101580000)`.

**Fix:** `CapstoneSubtarget::useConstantPoolForLargeInts()` returns **false** whenever the
gp-free/gp-captable ABI is active, so the constant is materialised inline instead. Forming
a pool in a domain is always a miscompile, never an optimisation — the same reason
`-fno-jump-tables` is already mandatory (a jump table is `.rodata` too).

**Validated:** the previously-faulting `rv8_sha512` configuration now returns its oracle
(`__CAPSTONE_LADDER_RV8_SHA512_PASSED__`); 0 `.LCPI` entries remain in the emitted asm;
Capstone lit **43/43**; `beebs_bs`, `beebs_prime`, `beebs_cnt` still pass QEMU parity.

> **Two wrong turns on the way, both worth remembering.** First this was called a
> *large-data delivery* problem, because bigger constants are the ones that get pooled.
> Then, on seeing that all named globals DID have cap-table slots, the constant-pool
> explanation was **retracted as refuted** — but the faulting object was never a global,
> so the descriptors could not have refuted it. The lesson is to identify the faulting
> OBJECT before reasoning about the mechanism: a symbolised `-S` listing settled in one
> step what two rounds of inference got wrong.

#### C-4b — the large-RO COPY PATH in the generated glue is broken `FIXED 2026-07-28`

**FIXED 2026-07-28. Root cause: `cincoffset` CONSUMES a linear `rs1`.**

`op_helper.c:635-640` — `helper_cscincoffset` with `rd != rs1` does
`*rd_v = *rs1_v; if(!captype_is_copyable(rs1_v->val.cap.type)) *rs1_v = CAPREGVAL_NULL;`
and `cap.h:122` defines `captype_is_copyable(ty) { return ty == CAP_TYPE_NONLIN; }`.
`sp` arrives from cscratch as `CAP_TYPE_LIN`, and the builder's only `delin(sp)` was its
LAST line — so the copy path's `cincoffset(t4, sp, t5)` **nulled `sp` outright**, and the
next `split(t2, sp, t1)` tripped `helper_cssplit`'s `assert(rs1_v->tag && !rs2_v->tag)`
with `tag == 0`.

That accounts for every observed symptom: it fired only AFTER `Created domain ID = 0`,
only when `COPY_THRESHOLD` selected the copy path, and never in the zero-init path (which
`cincoffset`s `t2`, already delinearized) or the unrolled path (which never `cincoffset`s
`sp`). It is also why five careful static readings of the generated assembly missed it —
**the assembly is correct as written; the defect is in the ISA semantics of one operand.**

**Fix:** emit `delin(sp)` at the top of `BUILD_GP_CAPTABLE`. Minimal and correct rather
than a workaround — `helper_cssplit` asserts `type == LIN || NONLIN` so every split still
works, and `split` (unlike `cincoffset`) never consumes `rs1`. `sp` was delinearized by the
builder's last line anyway, so this only moves that transition earlier; the capability
handed to compiled code is unchanged.

Emitted **only when a global actually took the copy path**, so every currently-measured
rung stays byte-identical — verified by diffing generated glue against the previous
generator (`beebs_aha_mont64`: 0 differing lines; `beebs_crc32big`: gains exactly the
`delin` and a comment). The condition is derived from the emitted body, not by re-testing
the eligibility predicate, so the two cannot drift.

**Validated:** `beebs_crc32big` (2,048 B `const crc_32_tab`) returns oracle **1703161001**
through the copy path — the first time that path has worked end to end. Standing ladder
regression 6/6 green (`matmult_int` 774662735, `beebs_prime` 582955588, `beebs_bs`
887447230, `beebs_cover` 1993178309, `ctrsanity` 43260934, `beebs_aha_mont64` 2185097489).

*Previous status, kept for provenance:* the MONITOR half working, and the failure moving.
 C-11 (the
monitor could not be rebuilt) is fixed, so the monitor-side copy specified in
`plans/sqlite-on-silicon-scoping.md` is now implemented, built and running:
`create_domain` copies the image's initialized-globals bytes
`[base+GPFREE_GLOBALS_OFFSET, base+code_size)` into the front of `dom_data`, guarded so it
is skipped rather than overrunning when the image is large relative to the data region.
Source is uncommitted submodule state, mirrored at
`tests/vendor-patches/opensbi-capstone-sbi.patch`.

Evidence it works: `beebs_crc32big` (2,048 B `const crc_32_tab`, external linkage, the
rung built specifically for this path) previously **failed at domain CREATION**; it now
prints `Created domain ID = 0` and proceeds. The regression rung `beebs_aha_mont64` still
passes with the copy live (`retval = 2185097489`).

**What remains: the same `helper_cssplit` assertion (`rs1_v->tag && !rs2_v->tag`), but
later in the sequence** — no longer at creation, now after the domain exists. Static
reading of the generated glue does NOT explain it: every `split` there takes `sp` (tagged)
as rs1 and an `lcc`-derived integer as rs2, and the registers that do hold capabilities
(`t3`, `t4` in the copy loop) are re-loaded with `li` before any later split. So the next
step is to LOCATE the faulting `cssplit` rather than reason about it — QEMU aborts on the
assertion, so add a print of `rs1`/`rs2` provenance in `helper_cssplit`, or break there
under gdb, and find out whether it is in the glue at all or in the monitor's
`create_region`/`share_region` path that runs immediately after.

**One implementation trap already paid for, recorded so it is not repeated:** the copy
must index in **16-byte** units. `__linear void *` subscripting steps one CAPABILITY and
generates a 16-byte `ldc`/`stc` — `dom_seal`'s own zeroing loop uses the same convention
(`DOMAIN_DATA_SIZE = 16 * DOMAIN_DATA_N`). An earlier draft used `>> 3`, walked twice the
intended distance and stored past `dom_data`:
`Cap mem access OOB: cursor = 101562000, size = 16, bounds = (101560000, 101561020)`.

*Original entry, still accurate for the glue half:*

**Not a domain-creation bug, and not about size.** Earlier notes here (now corrected) chased
image geometry through the loader and kernel module. That was the wrong component:

> `Created domain ID = 0` appears **before** the assertion in the serial log. Domain
> creation **succeeds**; `helper_cssplit: rs1_v->tag && !rs2_v->tag` fires afterwards, in
> the **entry glue**.

**The actual trigger is a threshold in the glue generator, not a size limit.**
`gen-gp-captable-glue.py` has `COPY_THRESHOLD = 256` and picks between two paths:

| initializer size | glue path | result |
|---|---|---|
| 640 B (`sha512_k[80]`) | **large-RO copy loop** (`stor > 256`) | **FAILS** |
| 128 B (`sha512_k[16]`) | unrolled `li`/`sd` immediates (`stor <= 256`) | **passes** |

So every "size-dependent" symptom was just this threshold selecting a different code path.
The large-RO copy path is the thing that is broken; it is emitted for exactly one global in
the ladder today, which is why nothing else has hit it.

**The suspect sequence** (from the generated `.inc`):
```
lla t4, sha512_k
lla t5, __gpfree_globals_base
sub t5, t4, t5               /* blob offset = sym - base */
cincoffset(t4, sp, t5)       /* src */
cincoffset(t3, t2, x0)       /* dst */
```
`lla` on a Capstone target may not yield a plain integer, so `sub` of two such values --
and hence the operand feeding a later `split` -- is where a stray tag most plausibly comes
from. **Verify by dumping tags, not by reading:** that inference is exactly the kind that
has been wrong three times on this issue.

**Refuted along the way, recorded so nobody repeats them:** (a) `tot_size` invariant --
both images give `tot_size` 8192 and satisfy `tot_size > code_size + 1536`; (b) `code_len`
carrying the exec segment -- it is `image_size`, the whole loadable image
(`libcapstone.c:197`); (c) `dom_pages_log2` rounding -- it rounds **up** correctly
(`dom_pages == 1 ? 0 : ilog2(dom_pages - 1) + 1`).

**Experiment RUN (2026-07-28): the unrolled path is not a viable stopgap, and C-4b is
entangled with C-5.** Raising `COPY_THRESHOLD` above 640 so the big table takes the
unrolled `li`/`sd` path fails at link time:

```
ld.lld: error: unable to place section .text at file offset [0x1000, 0x2E77]
```

`.text` reaches **11,895 B** against the 4 KiB window — 640 B of data costs ~8 KB of
immediate-materialisation code, exactly the reason the copy path exists. So:
- The copy path is **necessary**, not an optimisation — it cannot simply be disabled.
- **C-4b cannot be worked around without first lifting C-5** (the 4 KiB window), or by
  fixing the copy path itself.
- Threshold reverted to 256; no code change kept from this experiment.

**The `lla`-produces-a-tag hypothesis is REFUTED (disassembly, 2026-07-28).** The emitted
glue uses plain integer addressing exactly as intended:
```
auipc t4, 0x1 ; addi t4, t4, -0x108     ; integer address of sha512_k
auipc t5, 0x1 ; addi t5, t5, -0x150     ; integer address of __gpfree_globals_base
sub   t5, t4, t5                        ; plain integer offset
<cincoffset t4, sp, t5> ; <cincoffset t3, t2, x0> ; li t6, 0x280 ; ld/sd loop
```
No capability reaches an operand that must be untagged in this sequence. That is the
**fourth** hypothesis refuted on C-4b (after the `tot_size` invariant, `code_len`, and
`dom_pages_log2`).

**New observation, unexplained:** `li t6, 0x280` (640) appears **TWICE** in the domain, at
`0x10164` and `0x10324` — two identical 640-byte copy loops, where only one global is
640 bytes. Either the glue is emitted twice, or the generator emits a duplicate descriptor.
A second copy loop would carve/copy storage a second time and could plausibly leave the
register state that the next `split` chokes on.

**Counted, and the GENERATOR IS CORRECT.** The emitted `.inc` contains exactly
**1** copy loop, **3** global headers, **4** `split`s (cap table + 3 globals) and **3**
`stc`s to the table — all as intended.

**The duplicate is BY DESIGN — this lead is refuted too.**
`start-gp-captable-generic.S` has two entry points and each expands the macro:
```
__test_reentry:  ccsrrw(sp, cscratch, x0) ; BUILD_GP_CAPTABLE  /* reentry */
_start:          ccsrrw(sp, cscratch, x0) ; BUILD_GP_CAPTABLE  /* normal entry */
```
Two copies in the image, exactly one executed per entry. Nothing wrong with it.

**Status: FIVE hypotheses proposed, FIVE refuted by measurement.** In order: the
`tot_size` invariant; `code_len` carrying the exec segment; `dom_pages_log2` rounding;
`lla` yielding a tagged value; a duplicated copy loop. Each looked sound on paper and each
died on contact with a dump, a count or a disassembly.

**What is solidly established, and is the whole of what a successor should trust:**
- Domain creation **succeeds** (`Created domain ID = 0` precedes the assertion) — the fault
  is in the **entry glue**, not `create_domain`, not the loader, not the kernel module.
- The discriminator is `COPY_THRESHOLD = 256` selecting the **large-RO copy path**, not
  image size: 640 B takes it and fails, 128 B takes the unrolled path and passes.
- The copy path is **not optional** — forcing the unrolled path for 640 B blows `.text` to
  11,895 B against the 4 KiB window, so **C-4b is entangled with C-5**.
- The generated glue is **correct by count** (1 copy loop, 3 globals, 4 splits, 3 `stc`),
  and the two copies in the image are the two entry points, by design.

**BYPASSED 2026-07-28 — C-5 dissolves C-4b.** The copy path exists only because the
unrolled `li`/`sd` alternative does not fit a 4 KiB window. Give it a **32 KiB** window and
it does, so the broken path can simply not be taken:

```
DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1 DOMAIN_OPT_LEVEL=-O1 run-ladder-qemu.sh rv8_sha512
  -> __CAPSTONE_LADDER_RV8_SHA512_PASSED__ (retval = 1390718314)
```

`rv8_sha512` now runs with its **full 640 B table** — the crypto/bitwise rung the ladder
lacked. Both knobs are **opt-in per rung, not defaults**: changing the window changes image
layout and this project has documented layout sensitivity (2026-07-26: four added
instructions flipped a passing rung), so every measured rung stays at 4 KiB and its
published number stands. `beebs_bs` and `beebs_prime` re-verified unchanged.

**C-4b remains open and still matters**: the copy path is still broken, and any initializer
needing more than ~32 KiB of unrolled materialisation will still hit it (SQLite is the
likely first). But it no longer blocks a benchmark. When someone does fix it: **instrument,
do not reason** — dump the capability tag at each `split` in the copy path. Five paper
hypotheses have failed here; the sixth should not be one.

**Related hazard — CHECKED 2026-07-28, NOT a bug.** `getGpCaptableIndex` derives its index
from a global's *position* in `M.globals()`, and GlobalMerge mutates that list (it merged
`sha_chain` + `sha_w` into one 192 B entry here), which raised the possibility of an access
lowered against the pre-merge order loading the **wrong capability slot** — silent wrong
data rather than a fault. It cannot happen: **GlobalMerge runs in `addPreISel`**
(`CapstoneTargetMachine.cpp`), i.e. before instruction selection, so `lowerGlobalAddress`
during ISel and `emitGpCaptableTable` in the AsmPrinter both see the same post-merge list.
Confirmed empirically as well — the merged-global `rv8_sha512` build and the 6-global
`beebs_cnt` both return their exact oracles, which mismatched indices would break.
**Recorded because the reasoning is the useful part:** any future pass that adds or removes
globals *after* ISel would silently break this positional scheme.

### C-9 — Redundant `mv rd, rd` around inline-asm register constraints `OPEN`
The Capstone backend emits **no-op self-moves** around an `asm volatile("" : "+r"(x))`
tie. A 5-instruction loop body became 7 — `srai / xor / add / **mv a4,a4** / addi /
**mv a4,a4** / bne` — where plain riscv64 emits 5 for the same source.

- **Found:** 2026-07-27, while building the I-2 counter-sanity probe. It is logged because
  it **silently defeated that probe**: the measurement depends on both targets retiring the
  same instruction count, and the compiler manufactured a 1.4× difference out of nothing.
- **Repro:** `tests/runtime-qemu/silicon-ladder/ctrsanity_kernel.h` with the inner
  `__asm__ volatile("" : "+r"(acc))` restored; disassemble
  `--triple=riscv64 --mattr=+m` and compare against `ladder-base/obj/base_ctrsanity.o`.
- **Impact:** small in isolation (two wasted instructions per tie), but the register-pinning
  idiom is used throughout the ladder kernels to defeat constant folding, so it inflates
  the capability instruction count of **any** rung that uses it — i.e. it can bias an
  overhead ratio upward. Worth a look before the next measurement round.
- **Workaround:** keep inline-asm ties out of measured loops; use an opaque trip count and
  a consumed result instead.

### C-11 — the monitor cannot be rebuilt: boot-hangs with zero serial `FIXED 2026-07-28`
**FIXED 2026-07-28. Root cause: a stale object file, not the compiler.**

`build/build/opensbi-custom/build/platform/generic/firmware/fw_jump.o` was compiled
2026-07-22 **for the FPGA firmware, where embedding a device tree is mandatory**.
`make A=opensbi-rebuild` only **relinks** and never recompiles it, so every QEMU monitor
rebuild silently linked in an **FPGA device tree**; `firmware/fw_base.S:217`
(`#ifdef FW_FDT_PATH` → `lla a1, fw_fdt_bin`) then makes OpenSBI **discard the DTB QEMU
passes in `a1`**. Wrong memory map, wrong UART, console never initialised → hang with zero
serial, before any banner.

**Fix — make it part of the rebuild recipe, not a troubleshooting step:**
```bash
D=build/build/opensbi-custom/build/platform/generic/firmware
rm -f $D/fw_jump.o $D/fw_jump.elf $D/fw_jump.bin $D/fw_dynamic.o $D/fw_payload.o
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
```
**Verify before trusting any rebuilt monitor:**
`readelf -sW build/images/fw_jump.elf | grep -c fw_fdt_bin` must be **0**, and
`.rodata` must be `002de8` (an FDT-contaminated build reads `003a10`).
Validated: rebuilt monitor md5 `9cbf5068` boots and `beebs_aha_mont64` returns its oracle.

**The trap RE-ARMS every time the FPGA firmware is built in this tree**, because the same
build dir serves both and the FPGA side *requires* `FW_FDT_PATH`. Separating them (a
distinct `O=` build dir) is the durable fix and is not done yet.

**What was wrong before.** The recorded cause was compiler drift (good monitor `s0–s6`/
frame −368 vs regen `s0–s11`/−464). That difference is real but confined to `create_domain`,
which does not run at boot. The decisive experiment was to hold every generated input fixed
— install the known-good `.c.S`, block regeneration, rebuild — and it **still hung**, which
exonerated capstone-c outright. Then a section-by-section ELF diff showed `.rodata` alone
grew by 3,112 B, the symbol diff showed exactly one new symbol (`fw_fdt_bin`), and dumping
the first bytes of `.rodata` gave `d00dfeed` — FDT magic. Full trail:
`history/28-07-2026_16-10-00_monitor-regen-SOLVED-stale-fdt-object.md`.

**Unblocks:** large-`.rodata` delivery (C-4b) → SQLite on silicon; the `fence.i`
domain-boundary fix (the real fix for R-3, i.e. the per-rung power-cycle that dominates
board time); and any future monitor change.

---
*Historical detail below, kept because it is still the best record of what was ruled out.*


**Why it blocks SQLite.** SQLite's static tables need the large-`.rodata` **copy** path,
because the unrolled `li`/`sd` path has a hard ceiling: a single initialized global must
be `size % 8 == 0` and fit a 12-bit store offset (~2 KB). Verbatim from the generator when
`beebs_ns` hit it:
`2512 B of *initialized* data overflows the 12-bit store offset and is not copy-eligible (sym='ns_keys', size%8=4)`.
The copy path needs one monitor change (C-4b), the monitor cannot be rebuilt, so SQLite on
silicon has no path today. **This is the single gate, and it is not a compiler problem.**

- **The recorded cause CANNOT be the cause.** `plans/large-ro-delivery-completion-task-A.md`
  §1-STATUS v3 blames compiler drift: good monitor `s0–s6`/frame −368, every regen
  `s0–s11`/−464. That difference is real and reproduces. But attributing a boot hang to it
  requires the differing code to run at boot. Attributed every differing line of a fresh
  regen against the known-good `.c.S` to its enclosing label: **100 % of the real
  differences are inside `create_domain`** (the only other hit is the trailing `.align`
  line; `cap_env_init` is byte-identical). `create_domain` is an SBI handler invoked from
  userspace, and §1-STATUS v2 itself records that it "isn't even called at boot" against a
  hang with **zero serial**. v2 and v3 contradict each other; v2 has the direct observation.
- **The one untried candidate is REFUTED.** `caplifive-system` pins `sw/capstone-c` at
  `bugfix@508342a`; the isolation had used `master@8cda52c` and the merge-base `4899cf9`.
  Built `508342a` in a throwaway worktree (submodule tree untouched) and ran the regen
  command from `caplifive-buildroot/Makefile:26`: output differs from the current tree's by
  **two lines, both `.align 4` vs `.align 16`**, and in the direction *away* from the good
  monitor. No board time, no firmware risk. `ref/HOW-TO-LAUNCH-ON-FPGA.md` still records
  `508342a` as the "known fix" — that may hold for **caplifive-system's own** monitor, a
  different tree, but it is not a fix for the buildroot one.
- **Next steps, cheapest first.** (a) **Splice, don't regenerate** — apply the large-RO copy
  hunk directly to the known-good `.c.S` and rebuild; if it boots, SQLite is unblocked and
  the hang can stay open indefinitely. **This needs no board time — the QEMU leg is the
  gate.** (b) `capstone_int_handler.c.S` is regenerated too and is **unexamined** (no
  known-good backup was found), and unlike `create_domain` it *is* live early. (c) Localise
  with the board's gdb (halt, read `pc`).
- **HAZARD — the checked-in `.c.S` IS the broken regen.** `components/opensbi/lib/sbi/
  sbi_capstone_dom.c.S` is md5 `6dfe662a` (the `s0–s11`/−464 build); only `fw_jump.elf` was
  restored on 2026-07-24. It has no `%.c.S: %.c` rule, so **any buildroot rebuild from this
  tree silently links the broken monitor**, for both lanes. Known-good copies existed only
  in temp dirs and are now preserved at
  `~/capstone-b-artifacts/monitor-known-good/` (`sbi_capstone_dom.c.S.good-b7baff6f`,
  `fw_jump.elf.good` = `6724bcb3`).
- Full trail: `history/28-07-2026_14-30-00_monitor-regen-boot-hang-cause-not-established.md`.

### R-12 — rev-node exhaustion is SILENT CORRUPTION, not a fault `OPEN, will bite at call_dom`

The revocation-node allocator's `head` is 10 bits (`capstone-ariane/core/anvil_build/capstone_rev_node.anvil:168`), so allocation
**#1025 wraps to node id 0 and reuses live ids**. `overflow_flag` reaches only a debug LED
(`cva6.sv:1185`) -- nothing traps, nothing prints. Only `SPLIT` and `MREV` allocate
(`capstone_dyn_unit.anvil:136, :91`); `ldc`/`stc`/`cincoffset` allocate nothing
(`:330-332, :399`, `capstone_flu_unit.anvil:29-44`).

`create_domain` does **5** splits, so this is NOT the current SQLite blocker. But SQLite's
entry glue does **1,060** splits (1 table + 1,059 globals) and will be the first domain to
cross 1,024 -- at `call_dom`, i.e. the moment after the present wedge is cleared. No
ladder rung approaches it (bigmany: 65).

### R-13 — `CINCOFFSET` duplicates a linear capability, untracked `OPEN`

It writes the unmodified `rs1` back alongside `rd` with the same `revnode_id` and
`CAP_TYPE_LINEAR` (`capstone_flu_unit.anvil:29-44`, `commit_stage.sv:278`), so one linear
capability becomes two with no bookkeeping. Sits directly next to C-14 in kind: an
instruction whose source-register behaviour diverges from what the compiler assumes.

### I-4 — every monitor error is invisible on the FPGA `OPEN, cheap fix identified`

`capstone_error` is `C_PRINT(...)` + `while(1)`, and `C_PRINT` is `csrw 0x800` -- the RTL
trace, NOT the UART. So all five silent-spin sites look identical to a hang on the board:
`handle_interrupt` default (`sbi_capstone.c:898-900`), `handle_exception` default
(`:973-977`), illegal-instruction-not-`time` (`:959-963`), `swap_cpmp` -> `capstone_error`
(`:917-923`), and two in `split_out_cap` (`:236, :246`).

**Fix, zero board cost to develop:** give `capstone_error` a real UART putchar via
`split_out_cap(0x10000000, 0x100, 0)` -- the same mechanism the monitor already uses for
`mtime` (`sbi_capstone_dom.c:32-36`). Every future wedge would then name its own site
instead of presenting as silence. This is the highest-leverage change available for board
debugging and should be done before more board sessions are spent guessing.

### C-14 — the COMPILER uses `movc` (a MOVE) for scalar register copies `ROOT-CAUSED 2026-07-30`

> **ATTRIBUTION WAS REVISED TWICE ON 2026-07-30. Read this box before the rest.**
>
> v1 "the RTL is buggy" -> v2 "the spec mandates it, the RTL is conforming, QEMU deviates"
> -> **v3 (current): the spec is UNDER-SPECIFIED here; the weight of evidence favours
> scalars being EXEMPT, so the RTL's MOVC is probably an oversight -- but this must be put
> to the board owner as a QUESTION, not an accusation.**
>
> What killed v2 (all verified in-tree):
> * `parts/mem-access-insn.adoc:45` glosses the very parenthetical v2 relied on --
>   "not **a scalar or** a non-linear capability (i.e., `type != 1`)". So in the spec's own
>   usage `type != 1` is shorthand for "scalar or non-linear", which EXEMPTS scalars.
> * `parts/mem-access-insn.adoc:105`, the one other place the consumption rule meets a
>   possibly-scalar operand (STC), writes the guard explicitly: "If `x[rs2]` **is a
>   capability and** `x[rs2].type` is not `1`". That is literally QEMU's `tag &&`.
> * `parts/prog-model.adoc:219-222`: a register holds "either a capability **or** a raw
>   `XLEN`-bit integer", so `type` is undefined for an integer and the MOVC clause's test
>   does not cleanly apply to one.
> * Spec commit `a1db3c2` ("MOVC now works with non-capabilities without generating
>   faults") removed the not-a-capability exception but never revised the consumption
>   clause -- so that clause was written when `rs1` was guaranteed to be a capability.
> * QEMU's guard is deliberate, not an accident: commit `b9c53f0d09`, subject
>   "[Capstone] movc allows scalars", is the change that added `rs1_v->tag &&`.
> * The RTL contradicts ITSELF: its STC exempts scalars
>   (`capstone_dyn_unit.anvil:408`, `if(rs2_v.metadata.cap_type != NOT_CAP)`) while its
>   MOVC does not (`capstone_flu_unit.anvil:14-25`). Internal inconsistency is the usual
>   signature of an oversight rather than a design choice.
>
> **What is NOT in doubt, through all three versions:** the mechanism (MOVC zeroes a scalar
> source on this silicon), the numeric proof, and that LLVM is emitting the wrong
> instruction. Only blame moved.

**What the spec says.** `capstone-spec/parts/cap-man-insn.adoc:33-37`, MOVC:

    * If `rs1 = rd`, the instruction is a no-op.
    * Otherwise
    . Write `x[rs1]` to `x[rd]`.
    . If `x[rs1]` is not a non-linear capability (i.e., `type != 1`),
      write `cnull` to `x[rs1]`.

Type encoding: `0` linear, `1` non-linear, `3` uninitialised, `5` sealed-return
(`parts/existing-insn.adoc:60-65`). A plain scalar is not a non-linear capability, so
`type != 1` holds and **the spec mandates zeroing the source.** `parts/intro.adoc:59-61`
states the design intent plainly: instructions "can only **move**, but not copy, linear
capabilities between general-purpose registers."

**So MOVC is a MOVE, by design.** It is the wrong instruction for an ordinary
register-to-register copy of a scalar, on any conforming implementation.

**Who is wrong, precisely:**

| component | behaviour | verdict |
|---|---|---|
| RTL (`capstone_flu_unit.anvil:13-21`) | zeroes source unless `type == NONLIN` | **spec-compliant** |
| QEMU (`op_helper.c:580-584`) | adds `rs1_v->tag &&`, exempting scalars | **deviates from spec** -- and this is what hid the bug from every QEMU test |
| LLVM (`CapstoneInstrInfo.cpp:520-523`) | emits MOVC for *every* GPR-to-GPR copy | **the actual bug** |

**Correct rule for the compiler:**
* scalar copy -> `addi rd, rs, 0` (`mv`). MOVC is simply wrong here.
* non-linear capability copy -> MOVC is correct and preserves the source (`type == 1`).
* linear capability -> cannot be copied at all, by design. MOVC moves it, which is the
  only legal semantics; the IR should never ask for a duplicate.

**STILL DO NOT PATCH THE RTL, but for a different reason than v2 gave.** Not because the
RTL is conforming -- it probably is not -- but because a reflash invalidates every silicon
measurement taken so far, is a hard stop needing approval, and the fix we control (the
compiler) is free and lossless. Ask the board owner which behaviour is normative; do not
assert that theirs is wrong.

**The LLVM bug is bigger than the scalar case.** `CapstoneInstrInfo.td:2455-2460` declares
MOVC with `hasSideEffects = 0` and `$rs1` as a pure USE with no def. LLVM therefore
believes MOVC never clobbers its source -- which is wrong for LINEAR capabilities on ANY
implementation, since every reading of the spec agrees those are consumed. Fixing only the
scalar path leaves that hole open.

**The fix is cheaper than first estimated:** `PseudoSCALAR_COPY_I128`
(`CapstoneInstrInfo.td:2446-2447`) already exists and expands to `ADDI`. The scalar-copy
machinery is in the backend; what is missing is routing scalar GPR copies through it
instead of through MOVC.

---

**Original mechanism analysis (unchanged and still correct as to WHAT happens):**

`capstone_flu_unit.anvil:13-21`, MOVC with `rs1 != rd`:

    if(data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN){
        let rs1 = data.cap_rs1;          // source preserved
        let rd  = rs1;
    } else {
        let rs1 = call create_cnull();   // SOURCE ZEROED
        let rd  = data.cap_rs1;
    }

A plain scalar is `NOT_CAP`, so it takes the else branch and the source register is
nulled (`create_cnull` zeroes cursor and metadata, `capstone_unit.anvilh:383-384`).

QEMU (`op_helper.c:580-584`) guards the same zeroing with `rs1_v->tag &&
!captype_is_copyable(...)`. A scalar has `tag == false`, so **QEMU preserves what silicon
destroys.** DIVERGENT, and invisible to every QEMU test.

**Delivery mechanism.** `copyPhysReg` emits MOVC for every GPR-to-GPR copy
(`CapstoneInstrInfo.cpp:520-523`), so ordinary register moves inherit it. The write
reaches the register file through an rs1 write-back port gated only by
`cap_result.valid` (`commit_stage.sv:278-281`), i.e. for EVERY op in `check_cap_op`.
A narrower set was evidently intended: `check_fwd_rs1` lists
`{SPLIT, MOVC, CJALR, CCSRRW, STC}` (`ariane_pkg.sv:925-931`) and is **dead code** --
defined and referenced nowhere in the tree, verified by grep. The broad gate is harmless
for ops that echo rs1 faithfully (CINCOFFSET does, `capstone_flu_unit.anvil:37-44`) and
fatal for MOVC, which writes a null.

**Both failure modes follow mechanically.** In gpn2:

    203c0: movc a4, a6       ; a4 := a6, and on silicon a6 := 0
    203c4: bne  a6, a5, back ; a6 is 0, a5 is 4 -> always taken -> INFINITE LOOP

That is the wedge: the domain never faults, it spins, which is why no capture ever showed
an `mcause`, `mepc` or `badaddr`.

**NUMERIC PROOF** of the other mode. `gpw2` ends its loop with `beq a6, a4` rather than
`bne`. With `a6` zeroed, `0 != 1`, the loop exits one iteration early and `g[1]` is never
written. Predicted checksum for `g = {1, 0}`: **3950255460**. The board returned exactly
**3950255460**. Derived before inspection, bit-for-bit.

**Scope.** Every measured rung sorts correctly: the four that pass have no `movc` whose
source is read afterwards; the nine that fail do. SQLite has 444 occurrences of the
pattern. `gpstress` has none and does NOT wedge -- it returns wrong data, so it stays a
separate defect.

**Fix is a design decision, not a one-liner.** No single instruction copies both scalars
and capabilities while preserving the source -- and per the spec, none should: copying a
linear capability is deliberately impossible. What the compiler needs is to pick the right
instruction per type:

| candidate | scalars | capabilities |
|---|---|---|
| `addi rd, rs, 0` | correct, preserves source | drops capability metadata |
| `movc rd, rs` | DESTROYS source | correct for NONLIN only |
| `cincoffset rd, rs, x0` | RTL preserves rs1; QEMU nulls it (C-4b) | same divergence |
| `cincoffsetimm rd, rs, 0` | traps UNEXPECTED_OPERAND on NOT_CAP (`:49-52`) | -- |

`copyPhysReg` cannot tell them apart -- scalars and capabilities share the GPR class. A
correct fix needs the type distinction (separate register classes, or a copy pseudo
selected by type at ISel). See `plans/c14-movc-source-destruction-fix.md`.

This is a CORRECTNESS fix, not a workaround for a hardware defect: emitting a move where a
copy was meant is wrong against the spec regardless of which core runs it.

**Retracted on the way here** (four hypotheses, all mine): more-than-one-global,
exactly-16-byte globals, unrepresentable capability bases, and stale shadow-RF metadata
poisoning cincoffset's offset. The last was refuted by the same RTL read that found the
real cause: ordinary ALU writes DO invalidate the metadata shadow entry, because the
metadata regfile shares its write-enable with the integer regfile
(`issue_read_operands.sv:1695-1709`, `commit_stage.sv:271-279`).

### C-14 (superseded framing) — "a domain with MORE THAN ONE global fails" `RETRACTED`

**The split is exact.** Sorting every silicon result by the domain's global count:

| count | rungs | silicon |
|------:|-------|---------|
| 1 | beebs_primer1, bigwin, gpsz, gpcp, gptl, gpbg, gppv | all PASS |
| 2 | gpn2 | HANG |
| 4, 8, 16, 32, 64 | gpn4, gpn8, gpn16, gpn32, gpn64, bigmany | all HANG |
| 6 | gpstress | wrong value (444323487) |
| 1059 | SQLite | HANG |

**Two globals is the minimal reproducer**, established with a control in the SAME boot
(`LADDER_ONE_BOOT=1`, both transfers sha-verified, no reboot between them):
`beebs_primer1` returned 582955588 at 9775 cycles, then `gpn2` produced no END marker in
75 s. This is what the SQLite "hang" actually is; SQLite is not special.

**This supersedes the reading that the five initializer paths were validated.** `gpsz`,
`gpcp`, `gptl`, `gpbg` and `gppv` each have exactly ONE global, so none of them ever ran
the carve loop's second iteration. The paths are fine; the loop is not.

**Symptom.** `domain ID = 0` prints, then nothing — no `mcause`, `mepc`, `badaddr` or
`panic` anywhere in the capture. On silicon a monitor fault is `C_PRINT` + `while(1)` and
C_PRINT goes to the RTL trace, so a wedge and a hang are indistinguishable on the console.

**QEMU cannot see it, structurally.** gpn2, gpn4, gpn8 and SQLite are all green under
QEMU with `DOMAIN_GLUE=interp`. `helper_cssplit` keeps full 64-bit `{cursor, base, end}`
and never calls `cap_compress` (`op_helper.c:848-870`), and a tagged load overwrites the
decompressed bounds with exact ones from an out-of-band shadow map
(`op_helper.c:1128-1140`); the RTL round-trips EVERY capability write-back through
`compress_bounds` (`ex_stage.sv:1080-1098`) because the compressed form IS the
architectural register state. **A QEMU-green interp result is not evidence about
silicon.** Same shape as the DELIN divergence.

**Refuted, both without board time:**
- *Descriptor record order != cap-table index order.* `emitGpCaptableTable` and
  `emitGpCaptableInitDesc` both walk `M.globals()` with the same filter
  (`CapstoneAsmPrinter.cpp:857, 938`) and `getGpCaptableIndex` assigns indices in that
  order (`CapstoneISelDAGToDAG.cpp:134-138`). Record i IS slot i. Would have been a
  perfect no-op at count 1, hence worth checking.
- *`ldc rd, 16(gp)` is mis-decoded.* RTL uses the standard sign-extended 12-bit
  immediate added raw to the cursor, with the same address for the bounds check and the
  access and a trap on 16-byte misalignment — identical to QEMU
  (`decoder.sv:1300-1315, 1767-1770`; `capstone_dyn_unit.anvil:296-297, 318-328`).
- *Unrepresentable capability bases.* `split` sets cursor == base, selecting the
  cursorless branch where the base is exact at any alignment (see R-11).
- *Capability stack spills.* `beebs_primer1` already spills a capability
  (`stc 16(sp)` / `ldc 16(sp)` in `domain_main`) and passes.

**In flight.** `gpn2use0` / `gpn2use1` — both build a 2-entry table and run the carve
loop twice, but each reads only ONE slot (verified by disassembly): use0 reads slot 0,
use1 reads slot 1. Both pass => the fault needs two live slots. use0 fails alone => slot
0 was corrupted after being written, which points at the second store. Both fail =>
building a 2-entry table is itself fatal, and `INTERP_BUILD_LIMIT=1` then separates the
second split/store from the table split.

### R-11 — RTL truncates a capability TOP past a 2 MiB window; QEMU never does `OPEN, not yet hit`

`compress_bounds` has two branches selected by `bounds.start == cursor`
(`ariane_pkg.sv:749`). `split` sets cursor == base on both outputs
(`capstone_dyn_unit.anvil:139-144`), so carved capabilities take the **cursorless**
branch: the base is returned as `start: cursor` verbatim (`ariane_pkg.sv:662-665`),
exact at any alignment, while the TOP is truncated DOWN to a multiple of 2**E with E set
by the highest bit at which base and top differ, floored at bit 20. E is 0 — and the
capability exact — only while base and top share one 2 MiB window.

Domains are exact **by construction** today: the module rounds the allocation to a
power-of-two page count (`capstone.c:83-84`) and the allocator returns it aligned, so
everything sits in one window. Past 2 MiB, interior splits straddle a boundary and
globals silently get SHORT capabilities. `check-repr.py` fails a build at that cliff.

The other branch (`ariane_pkg.sv:769-806`, reached once cursor != base) is the
`granule(L) = 1 << (max(0, floor(log2 L) - 12) + 3)` rule with the base truncated down —
that one is C-13, caused by the monitor's `C_SET_CURSOR`. Applying it to the glue's carve
instead was a wrong fix (765da7f8, reverted in 91685f14); do not re-derive it.

### C-15 — `getGpCaptableIndex` gives `llvm.compiler.used` a cap-table slot `FIX WRITTEN, NOT YET BUILT`

Any TU using `__attribute__((used))` fails to link:
`ld.lld: error: undefined symbol: llvm.compiler.used, referenced by
.capstone_gp_table+0x48`. LLVM-reserved appending-linkage globals are markers, not data.
Found while building the gpn2use1 rung. Fix factors the predicate into a single
`isGpCaptableGlobal` so the early-out and the index-assigning enumeration cannot drift —
they define the ABI order the glue depends on.

### C-13 — interp glue fails on silicon `SUPERSEDED BY C-14 2026-07-30 — real interp PASSES at count=1`

**STATUS, stated precisely.** A real defect was found and fixed (below), and it fully
accounts for the stage-1 vs stage-2 difference. It does **not** yet account for C-13:
with the fix in place, the **real** interp path (no `INTERP_FAKE_COUNT`) still produced
no END marker on hardware — `beebs_primer1`, 2 attempts, 2026-07-29. So either the fix is
insufficient, or there is a SECOND independent failure.

The prime suspect for the remainder is the one thing real interp does that stage 2 does
not: **read the descriptor out of the monitor-copied blob in `dom_data`**. The glue's own
comment flags it as "the one assumption in this design never checked on hardware" — the
monitor's WRITE is proven, the domain's READ back is not. Next isolation step is stage 2
(fix, no descriptor read) x4: if stage 2 now passes, the delin fix works and the
descriptor read is the second bug; if stage 2 still fails, the delin fix is not the
answer.

**Do not record C-13 as closed on the strength of the delin fix alone.**

**Defect found and fixed: `delin` is not idempotent on silicon, and the glue delin'd four times.**
Full write-up: `history/29-07-2026_C-13-root-cause-double-delin.md`. Commits
`7e83841b5113` (glue) and `39f652b6e704` (generator + domain code).

The RTL's `DELIN` (`capstone-ariane/core/anvil_build/capstone_dyn_unit.anvil`) accepts
`CAP_TYPE_LINEAR` **only** and raises `UNEXPECTED_CAP_TYPE` otherwise. Our QEMU
`helper_csdelin` (`op_helper.c:900`) was patched to return early when the capability is
already `NONLIN`, so a double `delin` is a **silent no-op under emulation and a hard
fault on hardware**. `SPLIT` preserves `cap_type`, so once `sp` is delin'd at entry every
capability split from it is already `NONLIN`. The glue delin'd `sp`, then `gp`, `t2` and
`sp` again — three fatal. `delin(gp)` faults first. The generated glue never delins `sp`
early, which is exactly why it passes and `interp` does not.

Evidence — one fixed configuration repeated, not a single sample:

    stage 1 (no entry delin, sp stays LIN):  4/4 PASS  retval 582955588 == oracle, ~9722 cyc
    stage 2 (entry delin present):           3/3 FAIL
    real interp, WITH the fix:               FAILS    <-- the fix did not close C-13

The first two lines are what the delin finding explains. The third is why C-13 stays open.

**Two further instances of the same bug, found by audit** (see `39f652b6e704`):
- **Generated glue, copy path only.** The C-4b fix prepends `delin(sp)`, which turned
  that glue's `delin(gp)`/`delin(t2)`/tail `delin(sp)` into faults. Copy-path rungs are
  exactly the ones that hang on the board while passing on QEMU → **likely root cause of
  R-9**. Non-copy-path rungs verified byte-identical; `beebs_ns`/`beebs_crc32big` drop
  from 5+ delins to 1.
- **`output_text()` in `sqlite_capstone_domain.c`.** Delin'd `text`, which under
  gp-captable is a cap-table storage capability and therefore already `NONLIN`. On
  SQLite's critical path — it prints every success marker, so the domain would have
  wedged before emitting one. Compiled out under `-DCAPSTONE_GP_CAPTABLE_ABI`.

**CORRECTION (2026-07-29, same day):** an earlier version of this entry claimed `lcc
zimm=1` is non-portable because the RTL returns `cap_type - 1` and QEMU returns
`cap_type`. **That was wrong.** The RTL enum starts at `NOT_CAP = 0`
(`capstone_unit.anvilh`), so it is offset by one from QEMU's, where `CAP_TYPE_LIN = 0`
(`cap.h`) — and the `- 1` is precisely that conversion: `LINEAR(1) - 1 == LIN(0)`,
`NONLIN(2) - 1 == NONLIN(1)`, through `SEALEDRET(6) - 1 == 5`. **`lcc zimm=1` MATCHES
across QEMU and silicon, and a runtime cap-type test IS portable.** The `delin` fixes use
compile-time gating because it is free, not because a runtime test would be unsound.

What genuinely is not portable is the **raw enumeration** wherever it appears outside
`lcc` — compressed capability metadata, the `captype` debug instruction, any hand-written
type constant. Those are offset by one between the two targets.

**QEMU cannot detect any of this** — its `delin` is idempotent. QEMU runs prove
no-regression only. Recommended follow-up: make QEMU's `delin` strict (or put the
leniency behind an off-by-default flag) so this class becomes emulator-visible.

---

**Original entry (retained for the record).**
Found 2026-07-29 by a one-variable control, after it had already cost several board
sessions and a firmware rebuild.

    same rung (beebs_prime), same known-good firmware, same everything else:
      DOMAIN_GLUE=interp      FAILS  (no END marker, twice)
      DOMAIN_GLUE=generated   PASSES (582955588, 9,751 cycles)

`start-gp-captable-interp.S` is green on QEMU for every rung it has been tried on
(`aha_mont64`, `prime`, `crc32big`, `ns`, `statictbl`, `strtab`, `reentry`, plus the
6/6 regression) and fails on the board. It was never once run on silicon against a
known-good rung before everything else was built on top of it.

**What this RETRACTS — all of these were measured with `interp` and are now void:**
- **R-9's "all four variants hang"** (`ns`, `nskeys`, `nsflat`, `nssmall`). That whole
  boot used `interp`, so it measured the glue, not the kernels. The three hypotheses
  recorded as eliminated are **un-eliminated**; the variants may be fine.
- **The SQLite board hang** is most likely this rather than a 1.3 MB PCC limit -- the
  SQLite domain is built with `interp`.
- **The window climb** result, which never got past its control.
- **"My rebuilt FPGA firmware is broken"** -- it is not; the firmware was never the
  variable. (The `capstone_error` fix and the caplifive-system monitor port stand on
  their own merits and should be kept.)

**Why it went unnoticed:** the rule "test the default path after every change" was
applied to QEMU and not to silicon. `interp` was introduced, gated on QEMU, and then
used for every subsequent board run *including the controls*, so nothing in the setup
could reveal it.

**THE BISECTION BELOW IS INVALID. The failure is NOT REPRODUCIBLE run to run.**

    stage 1   PASS
    stage 2   PASS   -> FAIL on repeat, same build, same firmware, same rung
    stage 3   FAIL
    stage 4   FAIL
    stage 5   FAIL

Stage 2 was re-run with no change of any kind and flipped. So every attribution made
from single runs is reading noise: first "it is RUN_CAP_INIT's jalr" (wrong -- the rung's
cap-init table is empty and the jalr never executed), then "it is lla/auipc" (wrong --
stage 5 removed the added lla and still failed, and the passing stages already contain
six auipc).

**The methodological error, which is the useful part:** I bisected without first
establishing that the failure was DETERMINISTIC. One run per stage is only evidence if
the same configuration reproduces. It does not here. Roughly six board sessions were
spent building a causal story on single samples.

**What must happen before any further bisection:** measure the failure RATE. Run one
fixed configuration (interp, stage 2, `beebs_prime`) N times and count. Until that
number exists, no single-run pass or fail can attribute anything, and the same applies
retroactively to R-9's discriminator boot -- those four "hangs" are also single samples.

**What still stands**, because it rests on repeated or structural evidence:
- `generated` glue passes on silicon; `interp` has never yet passed twice.
- Firmware is not the variable (generated passes on both the prebuilt and the rebuild).
- SQLite's QEMU results are unaffected -- they are deterministic and re-run many times.

*Superseded reasoning follows, kept only to show what was tried.* Isolated to ONE instruction
pair by staged bisection on `beebs_prime`, one variable per boot, every build
QEMU-gated first:

    stage 1  minimal carve loop only                    PASS
    stage 2  + early delin(sp) + s1 blob view           PASS
    stage 4  + ONE `lla`, nothing else                  FAIL   <-- one instruction
    stage 3  + full RUN_CAP_INIT                        FAIL

**The earlier "it is the indirect call" conclusion was wrong**, and the reason is worth
keeping: `beebs_prime`'s cap-init table is EMPTY, so in stage 3 the only instructions
that ever executed were two `lla`s and a `bgeu` -- the `jalr` never ran. Blaming the
call was an inference from "cap-init is the block that differs" without checking which
instructions inside it actually execute for this rung.

**Scope is much wider than the glue, and this is the important part:**
- **R-9 is very likely THIS.** The large-RO copy path emits `lla <sym>` and
  `lla __gpfree_globals_base`; the zero-init and unrolled paths emit none. That splits
  the ladder exactly along the observed line -- `ns`/`crc32big` (copy path, `lla`) fail;
  `bs`/`cover`/`prime`/`mont64`/`ctrsanity` (no `lla`) pass. Every "kernel shape"
  hypothesis under R-9 was untestable, because the variants all kept the `lla`.
- **The `selectLGA` function-pointer change is implicated.** Code symbols now lower to a
  raw `PseudoLLA` -- i.e. `auipc` -- which is green on QEMU and untested on silicon.
  SQLite's method tables depend on it.
- **SQLite is hit twice**: copy path and function pointers.

**This looks like a platform constraint, not a bug in our glue**, and is worth a
board-owner question: is `auipc` expected to work in C-mode with a bounded PCC? A
plausible mechanism is that `auipc` computes from a PC that is PCC-cursor-relative in a
way the RTL does not implement as QEMU does. **Do not report it as fact until asked** --
what is measured is that one `lla` turns a passing rung into a hang.

**Workaround direction:** avoid `auipc` in domain code entirely. Offsets that today come
from `lla A - lla B` are link-time constants and can be baked as immediates by the
generator or the compiler; that is the same move that fixed the private-symbol problem
in C-4b.

*Superseded reasoning follows.* Bisected on hardware with `beebs_prime` (known-good,
3 KB, one boot each), one variable per stage, each build QEMU-gated first:

    stage 1  minimal carve loop only                    PASS
    stage 2  + early delin(sp) + s1 blob view           PASS
    stage 3  + RUN_CAP_INIT                             FAIL

So the interpreter's core is fine on silicon -- the carve loop, the splits, the `stc`
into the cap table, the s-registers, the early `delin(sp)` (R-2 does NOT bite here) and
the `sp`-derived blob view all work. Only cap-init fails.

**Why it is the culprit.** `RUN_CAP_INIT` calls each initializer with `jalr` on a PLAIN
INTEGER computed from `lla` differences. The reference implementation
(`my_first_domain/start.S:58-68`) instead derives a real CODE CAPABILITY with
`cincoffset gp, off` and calls it with `cjalr` -- which is valid there because in that
ABI `gp` spans the whole image. Under gp-captable `gp` is bounded to the cap table, so a
bare `jalr` was substituted. QEMU accepts an integer jump target; the RTL does not.

**Fix:** derive the code capability from **PCC**, which covers the code region by
construction, instead of from `gp` or an integer. Contained to one macro.

**Verify on BOTH:** `beebs_prime` has an EMPTY cap-init table, so it exercises only the
two `lla`s and the guard branch -- it proves the mechanism, not the calls. SQLite has 54
real pointer-valued initializers and is what proves the scale. Gate on both.

**Descriptor READ eliminated 2026-07-29.** The leading suspect was the runtime read of
the monitor-copied blob -- the one assumption in the design never checked on hardware.
Built `interp` with `INTERP_FAKE_COUNT=1`, which replaces the read with immediates
(`li s4,1` / `li t3,16` / `li t5,-1`) so no descriptor field is touched, QEMU-gated it
(`beebs_prime` returns 582955588), and ran it on the board: **still hangs.** So the blob
read is not the cause, and the monitor's copy is not implicated either.

Firmware is also eliminated, by the full matrix on one rung:

    generated + known-good prebuilt   PASS
    generated + rebuilt firmware      PASS   <- my firmware is fine
    interp    + known-good prebuilt   FAIL
    interp    + rebuilt firmware      FAIL

**What is left in `interp` that the generated prologue does not do**, for a rung with a
single `.bss` global:
1. `delin(sp)` at the TOP (generated delins `sp` last). **R-2 is literally "`delin` in
   domain code wedges the board"** -- this is the strongest remaining candidate.
2. `cincoffset(s1, sp, x0)` + `scc(s1, s1, t3)` to make a second view of `sp`.
3. `RUN_CAP_INIT`, which runs even when the table is empty (two `lla`s and a `bgeu`).
4. s-register use (`s1`-`s5`) across the builder.

Test them in that order, one variable per build, `beebs_prime` as the rung -- and note
that (1) and (3) can each be removed independently without touching the rest.

**Next:** bisect the glue against the generated prologue on hardware. The two differ in
that `interp` reads the descriptor from the blob at runtime, uses `s1`/`s2`/`s3`/`s4`
across the builder, and calls `RUN_CAP_INIT`. The first suspect is the runtime
descriptor READ from `dom_data` -- the whole design rests on the claim that the blob is
data-authority-readable by the domain, which is proven for the monitor's WRITE but has
never been proven for the domain's READ on silicon.

### C-12 — a NON-DEFAULT globals offset does not work `FIXED 2026-07-28`
**FIXED. Two capstone-c miscompiles in the monitor, both found by printing values.**

`DOMAIN_WINDOW=32k` (globals at image offset 0x8000) now returns oracle **1703161001**,
and the default window stays 6/6 green on both glue paths. This unblocks SQLite, which
needs `globals_off ~= 0x230000` for its 2.2 MB `.text` -- the same mechanism at a larger
value.

**Miscompile 1 -- `x >> 32` evaluates at 32 bits.** The monitor received
`entry_offset = 0x800000000000` intact (printed), but `entry_offset >> 32` produced 0, so
the packed offset was lost and `gpoff` fell back to 0x1000. Workaround:
`(entry_offset >> 16) >> 16`, which yields 0x8000.

**Miscompile 2 -- a nested ternary does not select the branch its condition implies.**
With `gpoff = packed_gpoff ? packed_gpoff : (globals_off ? globals_off : DEFAULT)` the
monitor computed `gpoff = 0x1000` while `packed_gpoff` printed as **0x8000** on the line
immediately above. Replaced with plain `if` statements and it takes the right branch.

**Both are capstone-c bugs, not ours**, and both are silent -- no diagnostic, just a wrong
value. Anything nontrivial written in the monitor should be checked by printing the
computed value, not by reading the C. Worth reporting upstream with these two reductions.

**Two self-inflicted diagnostic errors on the way, recorded because they cost more time
than the bugs did:**
- *A stale log read as evidence.* `run-domain-smoke.py`'s log is not cleared between runs,
  so I read prints from an earlier firmware and concluded that "only the later of two
  `C_PRINT` markers executes" -- an anomaly that never existed. `rm` the log first.
- *An `&&` chain broken by a relative path.* Running `make` from `caplifive-buildroot` and
  then `source capstone/tests/...` short-circuited the whole test, and the log I then read
  was again stale. `EXIT=` printing empty was the tell.

Confirmed properly by disassembling the LINKED `fw_jump.elf`: `_create_domain.0` at
`0x80020d9e` is `lui t0, 0xc12a; addiw t0, t0, 0x63`, i.e. the marker is on the executed
path immediately after a five-argument prologue. Checking the linked artifact rather than
the generated `.c.S` is what settled it -- the same check that resolved C-11.

### C-5 — 4 KiB code window `OPEN`
`link-gpfree.ld` forces globals to image offset `0x1000`, capping `.text` at 4096 B. One
hardcoded number, QEMU-validated at 16 KiB and 32 KiB and silicon-validated at 32 KiB. Lifting it
is what full CoreMark and Dhrystone need. Task #62.

---

## Archive — fixed, kept for provenance

**Move an entry here as soon as it is fixed**, with the fix and how it was validated.
Keep the id so older notes that cite it still resolve.

### Fixed 2026-07-27 (evening)

| id | issue | fix | validated by |
|---|---|---|---|
| **C-1** | `Cannot select: i128 = sign_extend_inreg` — an `int` index feeding capability address arithmetic crashed the backend at −O1+. The `Custom` action only runs during Legalize, and `performSIGN_EXTEND_INREGCombine` deliberately handles **only** the `any_extend(i64)` shape because expanding the general case in a combine ping-pongs against `visitSIGN_EXTEND` forever. Every other shape reached ISel unselectable. | Selected directly in `CapstoneDAGToDAGISel::Select` (`CapstoneISelDAGToDAG.cpp`), where there is no combiner to fight: `PseudoTRUNC_CAP` to XLen → `SLLI`/`SRAI` pair to sign-extend the source field → `PseudoSCALAR_COPY_I128` to widen. | repro clean at −O0/−O1/−O2/−O3; new lit `i128-sext-inreg-int-index.ll`; **Capstone lit 42/42** |
| **I-1** | A sweep silently rebuilt at −O0 and discarded the pre-built set, running capability halves at a different −O than their baselines. Cost five bogus "silicon failures", a false refutation of R-1, and a nearly published claim that a plain rebuild flips a passing rung. | Both build scripts now record the per-rung level to `<OUT_DIR>/optlevels.txt`; `run_ladder_perf_fpga.py` logs the effective levels and **hard-fails** on any capability/baseline mismatch, naming the rungs and telling you to set `LADDER_OPT` on the runner. | mismatch path exercised; runner parses; levels appear in the run log |

### Fixed 2026-07-27 (daytime)

| id | issue | fix |
|---|---|---|
| C-6 | CodeGenPrepare zero-extended a **negative** address offset into the 128-bit pointer carrier (`AddrMode.BaseOffs` is `int64_t`, `ConstantInt::get` defaults to `IsSigned=false`). Produced a **wrong address**; latent on any wide-pointer target. | `/*IsSigned=*/true` at 3 sites |
| C-7 | `APInt::getSExtValue()` asserted on an i128 constant in `SelectionDAGAddressAnalysis::matchLSNode` | `fitsInOffset` guard at 3 sites |
| C-8 | `Cannot select: i128 = and` — the dispatch returned the constant-mask helper unconditionally, so its bail left the node unlowered | fall through to `lowerScalarI128Logical` |

Validated: Capstone lit 41/41, BEEBS 82/82, CoreMark, authority 32/32, RV8 −O0 5/5, full X86 +
RISCV lit (6 `emutls*` failures **verified pre-existing** by stash-rebuild-reproduce).

---

## How to add an entry

One heading per issue with: a one-line statement of the behaviour, a **runnable repro**, the
evidence note, what has been tried, and the impact. Board reproducers go in
`tests/fpga-repros/R<nn>-<slug>/` — **committed, never `/tmp`**, which loses them on reboot
and makes them unreviewable. Keep frozen `.dom` images with the package when they are small
(the exact binary that reproduced is the point); when they are megabytes, ship the source
plus the rebuild command instead. An issue without a reproducer is
a rumour — write the probe first. Every probe must be **QEMU-verified before the board** so a
board deviation is unambiguous, and must **return a diagnostic rather than hang** (a hung domain
reports nothing at all).

### R-15 — a domain with a 9216-byte capability-bearing global wedges `OPEN — ATTRIBUTION RETRACTED 2026-07-31`

**Read the retraction before using this entry.** The observable is real; the mechanism
originally recorded here was wrong and has been withdrawn after an adversarial audit.

**What is actually observed.** With `SQLITE_STATIC_BUILTINS=1` (the R-14 workaround, which
moves `sqlite3RegisterBuiltinFunctions.aBuiltinFunc` from a stack array to a 9216-byte
compile-time-initialised global), a domain that does nothing but return WEDGES. Without it,
the same domain returns `rc=0`.

**What was withdrawn, and why:**

* **"Six domains in one boot."** Three ran. `run_sqlite_stages_fpga.py:120-123` breaks on the
  first wedge, so `ci-450`, `ci-550` and `ci-full` were never executed. The bracket rests on
  ctl / 200 / 350 only.
* **"The wedge is in cap-init."** NOT SHOWN, and the evidence points the other way. `ci_350`'s
  last output is `SHA5:00000002`, mid-way through the FIRST share entry — it never printed
  `SHA6`, `ECSZ`, `SQ: F/share2` or `SQ: G/enter`. `__capstone_cap_init` runs *after*
  `call_dom`, so on this run **it never executed at all**. The two earlier wedges of the same
  workaround build both printed `SQ: G/enter` before dying, so the console does deliver that
  marker at an entry wedge — `ci_350` has a materially different signature.
* **"aBuiltinFunc is implicated."** The 200→350 window admits **ten** holders, not one:
  `pragmaFunclistLine.azEnc`, `sqlite3Attach.attach_func`, `sqlite3ParseUri.aCacheMode`,
  `sqlite3ParseUri.aOpenMode`, `sqlite3Detach.detach_func`, `openStatTable.aTable`,
  `statInitFuncdef`, `statPushFuncdef`, `statGetFuncdef`, and `aBuiltinFunc`. Nothing in the
  data separates them.
* **"Control passes at 406 stores."** 403. The 406 count included three callee-save `stc`
  spills to `sp`.
* **"It is not the store count."** The comparison is confounded. `ctl` and the workaround
  build differ in far more than store count: `.data` +9216, `.bss` −10240,
  `aBuiltinFunc` moves `.bss`→`.data`, and **descriptor record 150 flips `blob_off` from the
  `-1` zero-init sentinel to `52240`** — so the entry glue goes from *zero-filling* a
  9216-byte carve to *copying* 9216 bytes into it, before cap-init is reached.
* **n=1**, one fixed order (ctl→200→350), no repeat, no order swap. The rev-node pool is a
  bump allocator with no reclamation and `ci_350` ran third.

**REFUTED 2026-07-31 (synthetic probe, board):** neither leaves-per-holder nor total
cap-init store count explains it. Five synthetic single-holder domains, one boot, no SQLite:

| leaves in ONE holder | total cap-init stores | returned |
|---|---|---|
| 40 | 446 | 40 — correct |
| 100 | 506 | 100 — correct |
| 160 | 577 | **0 — mismatch, non-monotonic, see below** |
| 300 | 733 | 255 (capped) — correct |
| 580 | 1017 | 255 (capped) — correct |

A holder with **580 leaves and 1017 total stores returns correctly**, against `aBuiltinFunc`'s
159 leaves and 596 total that wedge. So the size/count hypothesis is dead in both forms.

The 160-leaf mismatch is **non-monotonic** (160 fails, 300 and 580 pass), which points at the
probe rather than the platform — a genuine threshold cannot be crossed and then uncrossed.
n=1, not re-run, do not build on it.

**What is left of R-15:** only the bare observable — `SQLITE_STATIC_BUILTINS=1` makes a
do-nothing domain wedge, and without it the same domain returns rc=0. Every proposed
mechanism has now been refuted. The most likely remaining difference is the one the audit
surfaced and nobody has tested: descriptor record 150 flips `blob_off` from the `-1`
zero-init sentinel to `52240`, so the entry glue goes from *zero-filling* a 9216-byte carve
to *copying* 9216 bytes into it — before cap-init runs at all.

**What survives:** `-capstone-cap-init-limit` truncates in the same order
`-capstone-cap-init-print` prints (`CapstoneCapGlobalInit.cpp:213-236`; confirmed empirically
— each build's store sequence is an exact prefix of the next). All six `.dom` hashes differ,
so the flag took effect. `limit=200` returns and `limit=350` wedges — as an observation.

**Next experiments, in order:** (1) re-run `ci_350` alone, first in a fresh boot, ×3 — one
pass voids the bracket; (2) build `limit=223` vs `limit=224` and run them adjacently, the
only pair that separates `aBuiltinFunc` from the nine co-entering holders; (3) only then
bisect inside 223–381.

**Repro:** `CAPSTONE_SQLITE_STAGE=30..34` — **NOT YET RUN SUCCESSFULLY.** Its first attempt
built nothing (i128 `SELECT_CC`) and the harness reported a false pass; both are now fixed.

