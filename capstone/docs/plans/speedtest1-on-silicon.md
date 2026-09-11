# speedtest1 on the board: a comparison vehicle, not a one-off run

*Approved by the lead 2026-09-10. The first draft was audited by the board lane and by a
claim-auditor; both found real errors, and this one is rewritten around their findings with every
refutation re-verified against the sources. Phase 0 has since been RUN and its results are below —
they withdrew the plan's central prerequisite, so read that section before the phases.*

## Context

The lead asked for speedtest1 on the FPGA and was explicit about purpose: capability versus a matched
baseline first, later a CHERI column and allocator comparisons inside SQLite. So this is a **vehicle
with swappable arms**, shaped so the second and third comparison are new arms rather than new
projects. **Every board action belongs to the board lane**; this lane builds, validates under
emulation, stages, hands over a branch with oracle values and predicted readings, and does not drive
the board.

**What already exists.** speedtest1.c is on disk in a full SQLite 3.53.3 tree (3,487 lines) and needs
pinning like the amalgamation. Both build scripts honour `DOMAIN_SRC`. A domain can read `mcycle`,
and `minstret` too (`ladder_perf_domain.h:41-46`). The paired-baseline method is defined
(`fpga-silicon-measurements-for-paper.md:55-71`). `sqlite_boundary_cost_domain.c` is a 214-line
precedent. And one image runs **many stages per boot** — boot sw50 ran one image hash across four
selectors including a non-corpus one, which is also the precedent for the non-SLT oracle this needs.

### What the first draft got wrong, corrected

**The floating-point failure is LOUD, not silent.** I claimed `sqlite3_bind_double` takes an integer
while a caller's `double` stays real, so speedtest1 would link and silently truncate. That is refuted
for the build this targets: the silicon path compiles everything into **one translation unit**
(`sqlite_silicon_amalgam.c` includes `sqlite3-capstone.c` then the `DOMAIN_SRC` slot), so the domain
source sits after `#define double sqlite_int64`, which is never undone. No mismatch, no truncation.
The real failure is already recorded in our own tree (`build-slt-native.sh:11-16`): under
`SQLITE_OMIT_FLOATING_POINT` **the tokenizer rejects a decimal point at all** — `INSERT INTO t1
VALUES(3,'ccc',1.5)` dies with `near ".": syntax error`. speedtest1's SQL is full of decimal
literals, every SQL error reaches `fatal_error`, and that ends in `exit(1)`. It stops; it does not
lie. That is a better failure than I described, and the go/no-go must be argued on its real terms.

**The default testset cannot run at all, for two reasons that have nothing to do with floating
point.** `--testset` defaults to `mix1`, which expands to `main,orm,cte,json,fp,parsenumber,rtree,
star,app`. `json` is excised by `SQLITE_OMIT_JSON`, which is in the always-active define list. `rtree`
calls `fatal_error("compile with -DSQLITE_ENABLE_RTREE…")` unless that define is set, and it is set
nowhere. Naming `--testset` explicitly is therefore not a refinement, it is a precondition.

**I cited a line that never compiles.** My `fflush` citation was inside `#ifdef
SQLITE_SPEEDTEST1_WASM`. That is the project's recurring "blaming code that does not execute" shape,
appearing in my own plan.

**Two inferences deleted rather than defended.** "C-5 is closed so a larger image is not capped" is a
non-sequitur: the SQLite path never used that knob and computes its own globals offset, so closing
C-5 licenses nothing. And "the largest heap that fits is 2 MiB" was the largest class *tried* for a
smaller image, not a ceiling; the budget is non-linear in image size, so a bigger image can shrink
the available heap. The `domdata-budget.py` gate stays; the number goes.

## Phase 0 — RUN 2026-09-10. The go/no-go dissolved: no floating point needed

**speedtest1 runs today, unmodified, with our exact define set, in three of its testsets.** Built
natively against the patched amalgamation (`/tmp/capstone/sqlite-build/sqlite3-capstone.c`) with the
21 harvested `deployed` defines, `:memory:`:

| testset | result |
|---|---|
| `main` | **runs clean**, 21 sub-tests |
| `orm` | **runs clean** |
| `parsenumber` | **runs clean** (its decimals are string literals, so the tokenizer never sees a bare `.`) |
| `cte` | `SQL error: near ".": syntax error` — the Mandelbrot decimal literals |
| `star` | same, `0.125` |
| `fp` | `no such function: round` — a math function, so it needs `SQLITE_ENABLE_MATH_FUNCTIONS`, not merely floating point |
| `json` | `no such function: jsonb_object` — `SQLITE_OMIT_JSON` |
| `app` | `no such table: config` — depends on state another testset builds |

So the prerequisite argument is **withdrawn**: `--testset main,orm,parsenumber` needs no floating
point, no new defines and no source changes. Floating point remains the gate for `cte`, `star` and
(with math functions) `fp`, and that is now a scope choice rather than a blocker. Note for whoever
revisits it: enabling floating point alone would **not** fix `fp`, which fails on `round()`.

**Sizing, measured rather than estimated.** Native totals for the three working testsets, seconds:

| `--size` | main | orm | parsenumber | total |
|---|---|---|---|---|
| 1 | 0.017 | 0.004 | 0.002 | 0.023 |
| 5 | 0.065 | 0.013 | 0.005 | 0.083 |
| 20 | 0.264 | 0.049 | 0.010 | 0.323 |
| 100 | 1.876 | 0.253 | 0.041 | 2.170 |

The board-to-native factor comes from the one workload whose board time is recorded: `select4` took
**0.444 s natively** against **~78 minutes in the capability domain**, so **≈10,500×**. That is one
workload, includes both the 25 MHz clock and the capability overhead, and is an estimate — but it is
measured on the same engine with the same defines rather than assumed. Applying it:

| `--size` | projected board time, three testsets |
|---|---|
| 1 | ~4 min |
| 5 | ~15 min |
| 20 | ~57 min |
| 100 | ~6.4 h |

Against the board lane's target of one arm under ~2×10¹⁰ cycles (~13 min): **`--size` 1 to 5 is the
workable range**, with `main` alone at size 5 landing at ~11 min. `--size 100` is out by two orders
of magnitude, which is what the first draft failed to quantify.

**The unguarded include is real but not fatal.** `<unistd.h>` under the capstone triple with our libc
header preincluded compiles clean and pulls in 97 host-glibc header references. So it is a silent
hole rather than a build failure; add `_UNISTD_H` to the guard list rather than discover a conflict
later.

**Done in Phase 1:** the trial link of the amalgam translation unit with speedtest1 added, and the
real heap headroom that came out of it.

## Phase 0 as originally written — three cheap experiments (no board, hours)

Each settles something the plan currently assumes, and all three are cheaper than the work they would
otherwise misdirect.

1. **Does `<unistd.h>` resolve to host glibc?** speedtest1 includes it unconditionally, our libc
   header guards `_STDIO_H` and friends but not `unistd.h`, and neither build passes `-nostdinc`.
   One `clang -E` against the capstone triple settles it in seconds.
2. **Build speedtest1 natively with the harvested define set and run `--testset main --size 1`.**
   No cross-compilation. This settles, together: which testsets survive the omissions, what the
   decimal-literal failure actually looks like, and the shape of the smallest useful size.
3. **Trial-link the amalgam TU with speedtest1.c added and run `domdata-budget.py` on it.** That
   gives the real heap headroom for *this* image rather than a number borrowed from a smaller one.

## The decision this plan asks for

**Floating point, Part 2 of the merged stock-ness plan, is still the prerequisite — but for the
reason the audit established.** Without it, four of the nine default testsets abort on decimal
literals, and `testset_fp` (engine-side `sum()` over REAL columns, `round()`, `printf('%g')`) is
meaningless whatever the caller does. With `--testset main` alone it may be avoidable, and Phase 0
step 2 is what decides that. So the go/no-go is: **do we want speedtest1's arithmetic testsets, or a
narrower run that avoids floating point entirely?** Phase 0 answers it with evidence rather than
argument.

## Phase 1 as originally written — the vehicle (no board time)

1. **Pin the source.** Add `fetch-sqlite-src.sh` beside `fetch-sqlite.sh`, same version as the
   amalgamation, SHA3-pinned the same way; take `test/speedtest1.c` from it. `/tmp` is not storage.
2. **The stdio shim**, `adapted/capstone_sqlite_stdio.c`, over the existing `output_text` channel.
   The surface, measured rather than guessed: `printf`, `fprintf`, `vfprintf`, `fflush`, `exit`,
   `atoi`, **`fopen`, `fclose`, `fwrite`, `unlink`**, plus the **`stdout`/`stderr` objects** and
   **synthetic `argc`/`argv`**, since a domain has none and the option loop is the only way to set
   `--size` and `--testset`. Conversions actually reaching these calls: `d s u x`, the `ll` length
   modifier, width, zero-padding, left-justification and `.*` star-precision. **No float formatting
   is needed** — `%f` goes through SQLite's own `sqlite3_vmprintf`. Do not implement a general
   printf. The channel truncates silently, so the shim counts dropped bytes and the domain reports
   the count.
3. **The clock.** Both VFS clock methods are stubs returning zero. speedtest1 uses the 64-bit one, so
   back that with `mcycle` by overriding it after bootstrap inside the ten-line
   `adapted/capstone_sqlite_os.c`, not in the shared skeleton. Check whether SQLite internals use the
   other method before assuming one override suffices. Scale at **25 MHz**, and see the clock note
   below.
4. **The domain source**, `ports/sqlite/speedtest1_domain.c`, on the boundary-cost model,
   selected by a new magic opcode beside the existing two. **The host needs its own result branch**:
   the `0x4EB0` family is accepted only inside the feature-probe and corpus branches, and the
   terminal arm rejects anything that is not `DONE`. "Unchanged" was false in the first draft.

## Phase 1 — RUN 2026-09-10. The vehicle builds; the SIZE SWEEP is refuted on memory

The domain compiles, links and passes every build gate. Four things had to be settled on the way,
and one of them changes what Phase 3 can ask for.

**The exit path is a deliberate capability fault, not a longjmp.** `__builtin_setjmp` and
`__builtin_longjmp` are rejected outright by the target ("not supported for the current target"), so
the unwind the first draft assumed does not exist, and hand-rolling one would put untested
capability-ABI code in the middle of the measurement vehicle. Instead `exit()` writes its report and
then executes `ldc` through a NOT_CAP register. The monitor routes a synchronous cause it does not
handle to `fault_return_from_domain`, which terminates the domain and RETURNS to the host
(`sbi_capstone.c:1526-1530`), on both targets. An illegal instruction would be wrong: on FPGA that
lands at the ILLX site, which still ends in `while(1)`. Because a deliberate abort and a genuine
capability defect produce the same retval and the same monitor tags, **the discriminator is the
payload marker**, so the host must dump the payload on a fault result and not only on a clean one.

**Three build conflicts, all found by the trial compile rather than by reasoning.** speedtest1 and
the amalgamation each define a static `randomFunc`, which is a redefinition error in one translation
unit; the staged copy is renamed under a counted gate. `struct capstone_sqlite_file` is already
completed by the VFS header, so the shim's own completion is now conditional and applies only to the
standalone gate harness. And `testset_cte` computes `5.0/g.szTest`, a genuine double division,
because `SQLITE_OMIT_FLOATING_POINT` redefines the `double` KEYWORD but not the type of a floating
literal — so `__divdf3` has to link even though `cte` is not a runnable testset. It is added only
when speedtest1 is staged, because those builtin objects are linked individually and every board
result on record describes an image built without it.

**The heap is the binding constraint, and it rules out every size above 1.** Measured natively on
the same engine through the same `sqlite3_config(SQLITE_CONFIG_HEAP, ...)` path, the smallest
memsys5 arena in which each configuration COMPLETES:

| testset | `--size 1` | `--size 5` | `--size 20` |
|---|---|---|---|
| main | 1.5 MiB | 6 MiB | 32 MiB |
| orm | 2 MiB | 8 MiB | 32 MiB |
| parsenumber | under 256 KiB | under 256 KiB | under 256 KiB |

A domain's `dom_data` cannot exceed 4,194,304 bytes: `MAX_ALLOC_ORDER` is 10 and one
`__get_free_pages` call is where the region comes from. The heap is carved from that alongside the
globals blob, the cap table and the stack. So the ceiling is a little under 3 MiB of arena, and
**`--size 5` does not fit and `--size 20` is out by an order of magnitude** — on memory, before any
argument about cycles. The plan's earlier sizing table chose between 1 and 5 on RUN TIME; that
choice was never available.

**The budget tool's ceiling is necessary and NOT sufficient — corrected in Phase 2.** A 2.5 MiB
arena with a 1 MiB stack declaration built, passed every gate including `domdata-budget.py`, and
then faulted at `SQ: E/share1` under QEMU, before the domain was ever entered. Do not raise the heap
on the strength of the budget passing. The configuration that actually runs is 2 MiB with the stack
declaration at 1 MiB, and it carries `main`, `orm` and `parsenumber` alike — which matters because
one SQLite image carries one compiled-in heap and only one may be staged per boot.

**The reason given above was incomplete, and the conclusion survives anyway.** "The heap is carved
from `dom_data`, which caps at 4 MiB" is true of THIS build, where the arena is a domain global. It
is not the only way to deliver an arena: the revoke-on-free allocator takes its arena from a THIRD
SHARED REGION granted by the host (`sqlite_row11_domain.c:179`), which is host memory and is not
charged to `dom_data` at all. So "size 1 is the ceiling" needed checking against that route before it
was relied on.

Checked, 2026-09-10: two 1 MiB regions run clean, two 4 MiB regions run clean, two 8 MiB regions fail
at `map_region` after `SQ: D/mapped` and before the domain is entered.

**I then drew the wrong conclusion from that, and it is corrected here.** I wrote "so the region
route caps around 4 MiB as well". The module SOURCE allocates regions with `dma_alloc_pages`, which
draws on a reserved CMA area precisely to get past the buddy allocator's 4 MiB block — but **both
built module trees contain zero calls to it and three to `__get_free_pages`**. The image that
produced that bracket allocates regions from the buddy allocator, so the 8 MiB failure is the
order-10 cap and says nothing about regions.

Reserving CMA was then tried and changed nothing, as it must not: `cma: Reserved 64 MiB` appears in
the boot log and `map_region` still fails, because the running module never asks CMA for anything.
So **the ceiling on a CMA-backed region is UNKNOWN**, and settling it needs the module rebuilt from
the package source rather than a kernel argument.

That does not change the board ask or the recommendation to buy breadth before depth — ten benchmark testsets
at size 1 exercises more of SQLite than three at size 5 either way — but "the sweep is dead under
both routes" was overstated, and the door is not closed.

What the region route WOULD buy is not capacity but simplicity: an arena delivered as a region would
free the current geometry contortion, where the declared stack is cut to 1 MiB to make room for a
2 MiB arena inside `dom_data`. That is a rebuild of the image the board request is about, so it is
recorded here rather than done now.

So Phase 3 asks for `main --size 1`, `orm --size 1`, and `parsenumber` at whatever size the cycle
budget allows. This is not a defeat for the vehicle: it makes the ALLOCATOR comparison the lead
asked for substantive rather than decorative, since memsys5 is a buddy allocator and its
fragmentation is exactly what the ceiling is being spent on.

**The clock plumbing is confirmed rather than assumed.** speedtest1 takes the `xCurrentTimeInt64`
branch (`speedtest1.c:309` requires `iVersion>=2` and a non-null pointer; our VFS sets `iVersion` to
3), which is the method the mcycle override replaces. Its v1 fallback carries a real type bug under
these defines — a local `double` against a method that writes an int64 — but is dead on both paths.

**Done later in Phase 1:** the host's `--speedtest1` argument form with its own result branch, and
the run script that sets the region size for both halves.

## Phase 2 as originally written — QEMU, with a native oracle (no board time)

The same speedtest1 three ways: native x86, QEMU capability domain, QEMU baseline. Per-testset counts
and checksums must match; only timings differ. **Guard against the gate that fires and still
under-determines**: the native oracle is built from the same harvested define set, so all three
dying identically on a missing testset would "agree". Assert the expected work happened, not merely
that the three agree.

## Phase 2 — RUN 2026-09-10. Green, with a content oracle and two retractions

**speedtest1 runs end to end in a capability domain under QEMU, and computes byte-identical
results to a native build.** Six arms plus a control.

**The control first.** The ordinary SQLite domain passes in this worktree, toolchain and
environment. Without it, "the speedtest1 image faults" would have been equally consistent with
"nothing built here runs" — and the first arm did fault, so the control was load-bearing rather
than ceremonial.

**The content oracle.** Matching phase lists prove almost nothing: speedtest1 prints those names
from a table whether or not the SQL did anything. `--verify` hashes every result row instead, and
the domain's hash equals the native build's exactly:

| arm | verification hash |
|---|---|
| `main --size 1`, native x86 | `112006 38bb59fd…3925d8518` |
| `main --size 1`, capability domain | `112006 38bb59fd…3925d8518` |
| `orm --size 1`, native x86 | `465769 f3699caa…fd29c01c` |
| `orm --size 1`, capability domain | `465769 f3699caa…fd29c01c` |

**And the check is shown to fire.** Native `main` at sizes 1, 2 and 3 gives three different hashes,
so the oracle is sensitive to the work actually done and not merely to the testset's name.

**Readings.** With `-icount shift=0,sleep=off`, so `mcycle` is a deterministic INSTRUCTION count.
Each arm was run twice and the counts are byte-identical across passes, which is the control that
says the flag took — a plausible-looking number proves nothing, which is how the first set of
readings got published.

| arm | heap | instructions | verification hash |
|---|---|---|---|
| `main --size 1` | 2 MiB | 691,509,892 | `112006 38bb59fd…3925d8518` |
| `orm --size 1` | 2 MiB | 232,136,202 | `465769 f3699caa…fd29c01c` |
| `parsenumber --size 1` | 2 MiB | 45,957,678 | `0 0e12171d…ed18f85c` |

`DROPPED` is 0 in every arm, so no report was truncated. All three hashes equal the native x86
build's exactly, and all three were re-measured after the shared toolchain was relinked mid-session
and are unchanged.

### The matched baseline, and the first overhead numbers

**The comparison the lead asked for needs a denominator, and Phase 2 originally had none.** The
baseline is built on the ladder's method (`build-ladder-base-fpga.sh`): the SAME amalgam translation
unit, the same clang at the same `-O`, the same 27 defines, compiled `-target riscv64-unknown-elf`
with no capability flags, linked with a harness built in buildroot gcc that sits outside the counter
brackets. A static gate fails the build if any capability instruction reaches the binary; it reads 0.

**Instruction counts under `-icount`, and the ratio — CORRECTED 2026-09-10; see RETRACTION 5.**

| testset | capability domain | baseline, matched | ratio | as first published |
|---|---|---|---|---|
| `main --size 1` | 691,509,892 | 544,374,531 | **1.270×** | ~~1.502×~~ |
| `orm --size 1` | 232,136,202 | 174,683,787 | **1.329×** | ~~1.672×~~ |
| `parsenumber --size 1` | 45,957,678 | 29,724,522 | **1.546×** | ~~1.901×~~ |

**All three arms produce the same verification hash as the native x86 build**, so the three-way
comparison is between three runs of the same computation and not three different ones.

**These are INSTRUCTION ratios, not cycle ratios.** Under `-icount` QEMU's cycle counter and its
instruction counter are the same quantity, so this prices the extra instructions the capability ABI
emits and says nothing about what they cost in cycles on real hardware. The cycle ratio is what the
board is for, and it is the number the paper wants.

**The warm-up, and why it is not two runs of the benchmark.** The method requires the baseline's
warm pass, because first-touch page faults inside the bracket once made capabilities look 1.8×
faster. Running speedtest1 twice in one process does not work: it is not re-entrant, and a second
call to its `main` dies after a clean first pass. `sqlite3_shutdown` between them dies sooner. So the
warm arm runs a separate small workload first — SQLite init, a create/insert/index/select/update/
delete/drop cycle, and one byte written per page across the whole arena — and then measures the
benchmark once. What it warms is the SQLite code pages, the arena and the stack; what it cannot warm
is speedtest1's own code. Cold minus warm is reported rather than assumed, and it behaves as it
should: 1.4 % on the longest run and 21.7 % on the shortest, which is a roughly fixed cost against a
shrinking denominator.

**One defect found on the way.** The baseline's first run died in M-mode after both counter probes
had succeeded, which reads like a monitor bug. It was the VFS clock: `capstone_sqlite_os.c` reads
`mcycle` (0xB00), and Linux userspace cannot. The U-mode mirror `cycle` (0xC00) is now used in
baseline builds.

**And one latent issue in the existing build, recorded rather than fixed.** The domain build never
puts the amalgamation's directory on its include path, so `#include "sqlite3.h"` from
`capstone_sqlite_vfs.h` resolves to the HOST's `/usr/include/sqlite3.h` — **version 3.45.1, against
the 3.53.3 we build**. It is inert in the amalgam, because `sqlite3-capstone.c` is included first and
defines `SQLITE3_H` at its line 355, so the host header expands to nothing. But any translation unit
that includes the VFS header without the amalgamation ahead of it compiles against a different
SQLite's declarations. The baseline build names the correct directory explicitly.

### RETRACTION 5: about half of the published overhead was string-primitive workarounds

The first ratios — 1.502×, 1.672×, 1.901× — were **wrong in the direction that flatters the
finding**, and the baseline script's own header says why it should not have happened: "a baseline
built at a different level measures the optimiser, not the ABI."

`build-sqlite-silicon.sh:2811-2823` compiles the two support objects with three defines the baseline
did not pass: `BEEBS_STRING_LINEAR_SAFE`, `BEEBS_MEMCPY_OPTNONE` and `BEEBS_STRING_WRITERS_OPTNONE`.
They make `memcpy`, `memset`, `memmove` and `strcpy` `optnone, noinline` and switch
`strlen`/`strcmp`/`strcpy` to an indexing form. That script states plainly that these two objects
"hold the string primitives and are therefore where the domain spends its tight loops". So the
capability arm ran an un-inlinable `-O0` `memcpy` against the baseline's inlinable `-O1` one, and the
difference was charged to the capability ABI.

Measured, by building the baseline both ways:

| testset | baseline with the workarounds | baseline without | share of the "overhead" that was workarounds |
|---|---|---|---|
| `main` | 544,374,531 | 460,252,971 | **46 %** |
| `orm` | 174,683,787 | 138,795,555 | **51 %** |
| `parsenumber` | 29,724,522 | 24,159,173 | **39 %** |

Both numbers are meaningful and they answer different questions. The corrected ratio, where the arms
differ only in `-target`, prices **the capability ABI**. The original, against a plain baseline,
prices **what it costs to run SQLite safely on this silicon today, workarounds included** — which is
a legitimate figure, and is not what it was labelled.

`SPEEDTEST1_BASELINE_WORKAROUNDS` now selects between them and defaults to the matched arm. Two
smaller repairs went with it: the baseline read `SQLITE_SUPPORT_OPT` where the domain reads
`SQLITE_SUPPORT_OPT_LEVEL`, so the two agreed only by their defaults coinciding at `-O1`.

**One asymmetry that remains, measured and left in place deliberately.** The baseline is compiled
`-march=rv64imac` and is 44 % compressed encodings; the capability target has no compressed
extension, so the domain emits none. That is not something this work introduced — it follows the
convention `build-ladder-base-fpga.sh` already uses, so changing it would make these numbers
incomparable with every overhead figure the project has published.

Measured both ways on `main --size 1`, so the size of the effect is known rather than argued:

| baseline | instructions | ratio | text |
|---|---|---|---|
| `rv64imac` (default, compressed) | 544,372,609 | 1.270× | 926,248 B |
| `rv64ima` (plain) | 541,944,018 | 1.276× | 1,128,496 B |

**The instruction ratio barely moves: 0.45 %.** A compressed instruction still retires as one
instruction, which is why. **The code size moves 18 %,** and that is an icache effect which cannot
appear in an emulated instruction count and WILL appear in a board cycle count, in the baseline's
favour. So the QEMU ratio above is safe, and any board CYCLE ratio has to name this alongside the
granule guard. `SPEEDTEST1_BASELINE_MARCH` exists to re-measure it, not to hide it.

**Still inside the corrected number, and named rather than removed:** the granule guard (`W-12`,
`SQLITE_GRANULE_GUARD`, on by default) is an R-29 workaround that exists only in the capability arm
and costs "+33660 bytes of .text and a branch per granule", in the same primitive. Any published
ratio has to name its state.

### RETRACTION 1: the abort path does not return to the host

Phase 1 recorded that the deliberate capability fault "terminates the domain and returns to the
host, so the core survives and the boot continues". **That is wrong for QEMU and unverified for the
board.** A domain installs no `ctvec` — only the monitor does — so the fault cannot be delivered
horizontally, and `capstone-qemu`'s `cpu_helper.c:1866-1887` says exactly that: it prints
"domain halted by capability fault" and exits, noting that returning control to the host launcher
"requires a monitor-side fault-return path that does not exist yet". The monitor's
`fault_return_from_domain` is never reached on this path. Measured rather than argued: two arms
reached the abort path, because their heap was below the testset's minimum, and the emulator halted
with no host-side return.

The fault stays, because the alternatives are worse — a spin hangs the emulator until timeout and
wedges the core on the board, while the fault at least ends QEMU immediately and flushes. But
**nothing in the vehicle now depends on it working.** What keeps a run alive is not reaching it: the
measured heap minimum, the host's refusal of an invocation that does not name a testset, and putting
any stage that might abort last in a boot.

### RETRACTION 2: `HIGHWATER` was a zero that read like a finding

The report printed `HIGHWATER 0` in every run, because the define set carries
`-DSQLITE_DEFAULT_MEMSTATUS=0` and `sqlite3_memory_highwater()` is therefore never updated. A zero
there reads exactly like "the run used no heap". It now prints `n/a` unless
`CAPSTONE_SPEEDTEST1_MEMSTATUS` is set, and that knob is documented as perturbing the very
measurement it would report. The heap question is answered natively instead.

### RETRACTION 3: the first cycle figures were the host machine's clock

Phase 2 originally reported 1.3e10 cycles for `main --size 1` and derived "about eight and a half
minutes at 25 MHz" from it. **Those were host x86 timestamp-counter deltas, not guest cycles.**
`capstone-qemu`'s `target/riscv/csr.c:750-763` reads `cpu_get_host_ticks()` unless icount is
enabled, and `run-domain-smoke.py` never enables it. The evidence was already in hand and ignored:
four runs of the same arm gave 1.248e10 to 1.316e10, a spread no deterministic counter produces.

Worse, the "internal consistency check" claimed for those numbers — cycles ÷ 25 MHz agreeing with
speedtest1's own `TOTAL` — **is circular**. Both come from the same `mcycle` read, because
speedtest1's per-phase times go through the mcycle-backed VFS clock. Their agreement shows the clock
is PLUMBED and says nothing about whether it is scaled to anything real. The run script now carries
that warning at the assertion, and enables `-icount` by default so that forgetting it cannot
silently produce a plausible number again.

The corrected board prediction follows from instructions, not from QEMU time: board cycles ≈ CPI ×
instructions, and at the board lane's measured CPI range of 2.0–3.2 that puts `main --size 1` at
1.4e9–2.2e9 cycles, i.e. 55–89 s at 25 MHz, rather than the 8.5 minutes first reported.

### RETRACTION 4: the verification hash cannot fire for `parsenumber`

The hash was handed over as the per-arm verdict on the strength of it changing across three sizes —
which was checked for `main` and generalised to all three testsets. It does not hold. `parsenumber`
returns no result rows, so there is nothing to hash: its verification hash is
`0 0e12171d…ed18f85c` at sizes 1, 5 and 20 alike. For that arm the hash is a constant a broken run
would also produce, and the verdict has to be the instruction count and the four phase lines
instead. For `main` and `orm` the hash stands.

### RETRACTION 6: the within-pair placement asymmetry does not exist

Boot sw56 put the domain's capability regions in the CMA pool at `0xAC000000` where sw52's had sat
in ordinary buddy memory at `0x81xx`. I argued from that to a **within-pair** asymmetry: that the
capability arm's memory and the baseline arm's now sit ~700 MiB apart inside a single pair, an
asymmetry sw52 did not have, invisible to instruction counts and visible in board cycles.

The premise is true and the conclusion is wrong, because the regions are not where the benchmark
lives. There are two allocations and only one of them is a region:

    capstone.c:136   domain code+heap+stack   __get_free_pages(GFP_HIGHUSER | __GFP_ZERO, order)
    capstone.c:236   the 64 KiB share region  dma_alloc_pages(...)  -> CMA at 0xAC000000

The domain block is not merely *unlikely* to come from CMA, it is **ineligible**. Linux 6.4.14, the
board's own tree:

    gfp_types.h:334    GFP_HIGHUSER = GFP_USER | __GFP_HIGHMEM        <- no __GFP_MOVABLE
    gfp_types.h:335    GFP_HIGHUSER_MOVABLE adds it; this call site does not use it
    page_alloc.c:3374  ALLOC_CMA is set ONLY if gfp_migratetype(gfp) == MIGRATE_MOVABLE

and both CMA doors are gated on `ALLOC_CMA` — the balance rule at `page_alloc.c:2294` (CMA preferred
only when `NR_FREE_CMA_PAGES > NR_FREE_PAGES / 2`) and the empty-list fallback at `:2305`. An
UNMOVABLE order-10 allocation reaches neither, in any boot, at any free-memory level.

And that block is where the arena is: `speedtest1_domain.c:41` declares
`static unsigned char sqlite_heap[SQLITE_HEAP_SIZE]` and `:485` hands it to
`sqlite3_config(SQLITE_CONFIG_HEAP, …)`. memsys5 never allocates a page. `domdata-budget.py` on the
delivered image confirms the layout — the 2 MiB arena sits inside dom_data's 2,212,656 bytes of
globals storage — and sw56's own run marker reports `HEAP 2097152`, which is the static array and
not a mapped region.

So the domain's working set and the baseline's userspace pages come from the same regular free
lists, in sw52 and sw56 alike. What moved between the boots is the 64 KiB argv/output buffer,
written once on the way in and once on the way out. The pairs never carried this asymmetry, and the
cross-boot difference it belongs to is that buffer rather than the working set. The board lane
withdrew the confound on the same evidence, having re-derived it independently.

**CLOSED OBSERVATIONALLY, 2026-09-11, and the reading was already in the transcripts.** The
derivation above had no unverified link, but the direct confirmation never needed a probe: the
monitor traces each domain's load base as `DBAS` on every `create_domain` (`sbi_capstone.c:964`).
Scoped to each run's own `load_image`, sixteen domain creations across two boots:

    sw56  DBAS  81D60000 82400000 82800000 82C00000 83000000 83400000 83800000 83C00000
    sw57  DBAS  81D60000 82400000 82800000 82C00000 83000000 83400000 83800000 83C00000
    CMA area                              AC000000 .. BBFFFFFF

Every one in the 0x81–0x83 range, none within a hundred megabytes of the CMA area, exactly as
`GFP_HIGHUSER` being ineligible for CMA requires. This is stronger than the QEMU check that was
planned in its place, because it is the actual target. **The `dmesg` arm is struck from the next
boot.**

**Two consequences beyond the withdrawal.** A non-CMA control arm is not worth building: there is
nothing in the pairs to control for, and `linux,cma-default` is boot-global — strip it and *every*
domain arm in that boot loses CMA, so it could never have been an eighth pair beside seven CMA arms.
And the placement question, if it is ever wanted as a number, is now available for free from the
region-arena arm described under "What remains" — same boot, same image family, one difference.

**What would have caught it:** reading the allocation site of the thing that is actually hot before
claiming a placement asymmetry — which is the existing "read the caller" habit applied to memory
rather than to control flow, so no new rule is proposed. What made it easy to get wrong is worth
recording instead: `capstone.c:144` prints the buddy block as
`Domain memory region vaddr = …, paddr = …`, and the word "region" there names the one thing it is
not. Both lanes read that word the same wrong way.

### Repository layout

The SQLite port moved from `capstone/benchmarks/sqlite/` to `capstone/ports/sqlite/` while this
work was in flight. Git's rename detection carried the edited files across but placed four of the
new ones in `capstone/bug-corpora/sqlite/`, following the majority of the old directory rather than
the port; they are now beside their siblings. All three gates — the shim self-test, the native
baseline build, and a full domain build plus QEMU run — were re-run from the new location and the
verification hash is unchanged.

## Phase 3 — the board handover, with the readings written down first

**The board lane executes. This lane hands over a branch, a staged pair, the oracle values and the
predicted readings; it does not drive the board.**

### One image, three stages

Every SQLite image links at entry VA `0x10000` and there is no `DOMAIN_BASE_VA` knob in this build,
so exactly one may be staged per boot — but that one image runs many stages with different
arguments, which is what makes a three-testset boot possible. The geometry below is the one that
carries all three, and it is not a free choice in either direction:

| | |
|---|---|
| heap | 2,097,152 B — `orm --size 1` needs 2 MiB and `main --size 1` needs 1.5 MiB |
| declared stack | 1,048,576 B (`SQLITE_SILICON_STACK`) — 2 MiB does not leave room for the arena |
| region | 65,536 B, set for BOTH halves by the run script |
| build | `DOMAIN_SRC=ports/sqlite/speedtest1_domain.c`, `SQLITE_SPEEDTEST1_SRC` from `fetch-sqlite-src.sh` |

**Do not raise the heap on the strength of `domdata-budget.py` passing.** A 2.5 MiB arena passed
that gate and then faulted at `SQ: E/share1` under QEMU, before the domain was ever entered.

### Stage order, and why

`k800` control first — a boot whose control fails is void. Then, ascending by cost, so that the
cheapest arms are banked before anything expensive can take the core:

1. `parsenumber --size 1 --verify`
2. `orm --size 1 --verify`
3. `main --size 1 --verify`

The stage spec form is `<dom>:--speedtest1 --testset main --size 1 --verify`. The host accepts the
arguments split across argv precisely so the driver's `"{host} {host_args}"` shell string does not
have to preserve quotes; that split form is exercised under QEMU, not assumed.

**The baseline runs in the same boot and is not a domain.** `speedtest1_baseline` is an ordinary
static Linux binary, so the one-image-per-boot rule does not bind it and it needs no power cycle
between invocations. Per testset, run it three times:

    speedtest1_baseline probe cycle          # each probe in its OWN invocation: a gated CSR traps,
    speedtest1_baseline probe instret        # there is no handler, and only that invocation dies
    speedtest1_baseline warm --testset main --size 1 --verify

**Probe before measuring.** The board is recorded as gating the unprivileged counter for domains,
and whether it does so for ordinary Linux userspace is unknown — that is exactly what the probes
settle, and they cost nothing. `warm` is the denominator; `run` gives the cold number if the
difference is wanted.

### The predicted readings, before the boot

From INSTRUCTION counts, not from QEMU time — see Retraction 3 for why the first version of this
table was meaningless. Board cycles ≈ CPI × instructions; the CPI range is the board lane's, measured
on the ladder's kernels, and SQLite's cache behaviour is not obviously inside it. A board reading
ABOVE the range is a finding about the memory system, not a failed arm; a reading at or BELOW the
instruction count means the arm did not do the work.

| stage | instructions | board cycles at CPI 1.13–6.44 | at 25 MHz |
|---|---|---|---|
| `parsenumber --size 1` | 4.60e7 | 5.2e7 – 3.0e8 | 2 – 12 s |
| `orm --size 1` | 2.32e8 | 2.6e8 – 1.5e9 | 10 – 60 s |
| `main --size 1` | 6.92e8 | 7.8e8 – 4.5e9 | 31 – 178 s |

The CPI span is the board lane's, recomputed from the measurements doc's own table after its glossary
line (2.0–3.2, "never near 1") was found to contradict the data below it. **The timeout decision does
not depend on resolving it**: a 900 s stage at 25 MHz is 2.25e10 cycles, which tolerates CPI up to
32.5 against `main`'s instruction count — five times the highest ratio ever measured on this board.

### What counts as a result

- **The verification hash is the verdict for `main` and `orm`, and only for them.** `main --size 1`
  must read `112006 38bb59fd…3925d8518` and `orm --size 1` must read `465769 f3699caa…fd29c01c`.
  Those are the native x86 values and the QEMU capability domain reproduces them exactly. A board run
  that reports cycles and a *different* hash has measured the wrong computation. **For
  `parsenumber` the hash cannot fire** (Retraction 4) — judge that arm on its instruction count and
  its four phase lines.
- **`DROPPED` must be 0.** Non-zero means the report was truncated and the text cannot be trusted,
  though the run still can.
- **`SPEEDTEST1-CYCLES` must be present.** The `__CAPSTONE_SPEEDTEST1_RAN__` marker alone would also
  appear on a run whose report was truncated away.
- Read results from the run's own transcript segment, and **cite each row by its image hash, never
  by its label.**

### What can go wrong, and what it will look like

- **A fatal error inside speedtest1 ends the stage with no report and probably takes the boot.** The
  abort path does not return to the host (see Retraction 1). This is why the heap is sized above the
  measured minimum and why `main` is last.
- **`SQ: E/share1` with cause 24** was seen once, at a 2.5 MiB arena. The same signature was
  attributed earlier this month to an entirely different trigger, a restored `-USQLITE_OMIT_EXPLAIN`,
  and no mechanism was established either time. Do NOT read it as benign: it is one observation and a
  coincidence of signature, not a classification.
- **An entry stall (no `SQ: G/enter`)** says nothing about this code; redraw rather than retry.

## Phase 5 — RUN 2026-09-10. Seven of ten testsets, up from three

`SQLITE_FLOAT=on` removes `SQLITE_OMIT_FLOATING_POINT`; `SQLITE_FULL=on` adds rtree on top. Both are
off by default and are read identically by the domain build, the native oracle and the heap sweep,
so every recorded board result keeps its exact define set.

**Floating point is the keystone and it is one define.** Measured natively with the deployed set as a
control: it takes speedtest1 from three runnable testsets to seven. `cte` and `star` stop failing on
decimal literals, `fp` gets `round()`, and `app` gets `unixepoch` — because the omission
force-defines `SQLITE_OMIT_DATETIME_FUNCS` as a side effect.

**Two things this plan previously stated were wrong.** `fp` does **not** need
`SQLITE_ENABLE_MATH_FUNCTIONS`; `round()` is gated on the floating-point omission, and that define
covers `ceil`, `floor`, `ln`, `pow` and `sqrt`, none of which `testset_fp` calls. And `app` does
**not** depend on state another testset builds; it creates every table it uses. The
`no such table: config` recorded in the Phase 0 survey was an artefact of passing `:memory:`
positionally, which makes speedtest1 reopen an empty database in test 110.

**In the domain, under emulation, each testset as its own invocation — every hash equal to native:**

| testset | hash | | testset | hash |
|---|---|---|---|---|
| `parsenumber` | `0 0e12171d…` | | `star` | `0 0e12171d…` |
| `cte` | `186 9849fb27…` | | `rtree` | `12913 f8cd0f1d…` |
| `fp` | `8 6e798592…` | | `main` | `111130 1e792c9d…` |
| `orm` | `408505 35f60ec9…` | | | |

Note four of the nine share the no-rows hash `0 0e12171d…` — `parsenumber`, `star`, `json` and
`app`. **Their hashes cannot discriminate**, exactly as `parsenumber`'s could not before. Judge those
arms on instruction counts and phase lines.

### The two that do not run, and they fail differently

**`json` is excluded deliberately, for two independent reasons.** It needs a 6 MiB arena, measured,
against a domain ceiling near 3 MiB — so it cannot run in a domain whatever the defines say. And
including it pushes the image into the S-14 pre-entry fault: the domain halts at `SQ: E/share1` with
cause 24 on every testset, before it is entered. `SQLITE_JSON=on` adds it back for anyone measuring
that fault rather than trying to run json.

**That is a third trigger for S-14, and the progression is the useful part** — `ISSUES.md` asks for
exactly this comparison:

| build | `code_len` | globals offset | globals | result |
|---|---|---|---|---|
| baseline | 1,483,608 | 0x150000 | 208 | runs |
| + floating point | 1,551,080 | 0x160000 | 211 | runs |
| + rtree | 1,621,624 | 0x170000 | 218 | runs |
| + json | 1,760,872 | 0x190000 | 234 | **faults before entry** |

**`app` fails differently and is left open.** It ENTERS and then takes cause 4, a misaligned load,
inside the datetime code the floating-point omission had been disabling. Not the S-14 signature, and
not memory — its arena need is 1 MiB.

### Two defects found on the way

**In my own heap sweep.** It passed `:memory:` positionally, the argument shape that makes
speedtest1 reopen an empty database, and reported "app DOES NOT COMPLETE at any arena up to
33,554,432 bytes". That reads as a fact about memory and was a fact about my command line. It now
omits the positional name, which also makes it match the invocation it is measuring.

**In speedtest1 with our defines.** A multi-testset invocation segfaults at test 999, "Reset the
database" — natively, at any arena size, for any pair. So `mix1` is not the route. Each testset is
its own stage, which is how the board runs them anyway.

## What remains, 2026-09-11 (after boot sw56)

Boot sw56 ran the seven-testset image on `caplifive_r25r26r27_66c4e7517`, the same bitstream as
sw52. Six domain arms are in, none failed, the `k800` control passed at 4,508 cycles against sw52's
4,430, so the boot carries a verdict. Both placement confounds raised against it — the board lane's
cross-boot one and mine within the pair — have been withdrawn on the evidence in RETRACTION 6, by
both lanes independently. What is left is one instrument gap, one open fault, and one build that
lifts a ceiling.

### 1. Record sw56 (this lane, when the baseline half lands)

Domain arms, all seven, on `caplifive_r25r26r27_66c4e7517`:

| testset | board cycles | icount prediction | cycles ÷ predicted instr |
|---|---:|---:|---:|
| `star` | 228,836,250 | 59,443,585 | 3.850 |
| `parsenumber` | 230,682,632 | 60,938,746 | 3.785 |
| `orm` | 937,346,376 | 274,563,789 | 3.414 |
| `main` | 2,675,472,428 | 696,764,754 | 3.840 |
| `fp` | 4,165,418,923 | 933,826,108 | 4.461 |
| `cte` | 6,048,111,965 | 2,019,055,552 | 2.996 |
| `rtree` | 9,084,912,260 | 2,420,439,736 | 3.753 |

**RETRACTED, within an hour of being written here: "every board instruction count equals its
prediction exactly".** It cannot, because **sw56 contains no domain instruction count at all.** The
domain marker is `SPEEDTEST1-CYCLES <n> HIGHWATER n/a HEAP 2097152 DROPPED 0 RC 0` — cycles only,
no instret field — on all seven arms; `instret` appears once in the whole boot and it is the `k800`
control rung. The column I read as a board measurement was the board lane's table carrying *my own
icount predictions* through so that a CPI could be computed. It agreed with my predictions exactly
because it *was* my predictions. The `speedtest1_instret.dom` image is still waiting on a boot of
its own, and that boot is what would make the check real.

What the check would have needed is not a new rule: a number that matches a prediction to the digit,
on seven workloads, is the shape that should have prompted "where was this measured?" before it was
called the strongest evidence in the exercise. The one habit worth naming is narrower and specific
to working with a peer lane — **a column in someone else's table is not a measurement until you ask
which instrument produced it**, and this one arrived already merged with mine.

So the third column above is `cycles ÷ predicted instructions`, not a measured CPI, and it is
labelled that way here because it is still the useful figure — it prices a board cycle against an
instruction count established under emulation — but it is not two independent readings.

The CPI spread, 2.996 on `cte` to 4.461 on `fp`, orders the same way as the instruction-ratio column
does at its two ends. It goes into `docs/ref/fpga-silicon-measurements-for-paper.md` §7f **once the
baseline halves land**, as one table, with three caveats named beside it rather than folded into it:

- the **RVC asymmetry** already recorded in §7f — the baseline is `-march=rv64imac_zicsr` and carries
  compressed encodings the capability build does not, which makes the published ratios *overstate*
  capability cost;
- the **optimisation asymmetry**, likewise already recorded;
- and **new: the soft-float implementation asymmetry.** Both arms are soft-float, so there is no
  hardware-FP confound — the baseline's `-march` has no `f`/`d`, and its disassembly holds zero FP
  instructions against a positive control that finds 6 of 6 in an `-march=rv64imafdc` object. But
  the *implementations* differ: the baseline resolves `__divdf3` and its siblings from buildroot's
  libgcc, the domain from our compiler-rt builtins at `SQLITE_SUPPORT_OPT_LEVEL`. On `fp`, `cte` and
  `rtree` those routines may be a large share of the instructions, and `fp` has the highest CPI of
  the six. Named beside the ratios, in the same class as the other two, and quantified under item 6.

- and **the periodic tick, which points the OTHER way.** The baseline's board instruction count runs
  ~6 % above its icount figure, and the excess tracks *cycles* rather than instructions -- 3,745 to
  3,787 instructions per tick across seven arms spanning a 44-fold range of durations, where the
  per-instruction figure ranges over 4.39 % to 7.45 %. The domain does not pay it, and the monitor
  says why: `handle_interrupt` at `sbi_capstone.c:1856-1861` answers `IRQ_M_TIMER` by clearing `MTIP`
  in `mie` and setting `STIP` in `mip`, and `mie.MTIP` is re-armed only by an
  `SBI_EXT_TIME_SET_TIMER` call from S-mode (`:1772-1775`), which Linux cannot issue while it is not
  running. So a domain arm takes at most one machine timer interrupt and then runs tick-free, while
  the baseline is ticked at 100 Hz throughout. That inflates the baseline's cycles and therefore
  **understates** the ratio, where the RVC and optimisation asymmetries overstate it. They do not
  cancel to anything in particular and are named separately rather than netted.

  **`board_instret - icount` is a difference of TWO ticked machines**, not one ticked minus one
  unticked, and reading it as the board's tick cost is safe only while the subtracted term is
  negligible. The two guests differ in rate *and* kernel, both read from the built `.config`:

      QEMU guest   build/build/linux-6.1.26   CONFIG_HZ=250   NO_HZ_IDLE
      board        build/build/linux-6.4.14   CONFIG_HZ=100   HZ_PERIODIC

  Under `-icount shift=0` QEMU advances virtual time 1 ns per instruction, so its guest ticks every
  4,000,000 instructions: `cte` carries 442 subtracted ticks where `star` carries 11. That is why
  `cte` alone read as an outlier before correction.

      testset       naive T   qemu ticks   corrected T
      star            3,762         11.2         3,832
      parsenumber     3,756         10.6         3,822
      orm             3,773         50.9         3,846
      main            3,787        137.2         3,858
      fp              3,784        170.2         3,841
      cte             3,745        442.5         3,842
      rtree           3,763        172.3         3,827

  Corrected, `cte` is fifth of seven rather than last, and `parsenumber` becomes the low arm. The
  spread hardly moves -- 1.10 % to 0.92 % -- so this is an argument about the outlier, not about
  precision.

  **The QEMU per-tick cost is measured, not assumed equal to the board's.** Sweeping `-icount shift`
  over 0, 3 and 6 on the same `star` baseline gives 4,353 / 4,535 / 4,516 instructions per tick from
  the three pairings, against the board's ~3,780. **That sweep needed rescuing before it meant
  anything:** `instret` under `-icount` returns virtual time in nanoseconds, so raw INSTRS scaled by
  2^shift and the "flat versus grows" reading it was designed for could not have discriminated.
  Dividing by 2^shift recovers the instruction count, and the CYCLES column agrees independently
  (ratios 8.062 and 68.91 against 8 and 64). What makes the correction robust is that the fit
  determines the PRODUCT of tick count and per-tick cost and the correction consumes the same
  product, so an error in the assumed rate cancels exactly; a correction using the *board's* per-tick
  cost against a QEMU tick count would not have that property.

  **What is still not measured:** how much of the baseline's CYCLES the tick costs. Converting
  excess instructions into a share of cycles assumes kernel CPI equals user CPI, and an interrupt
  handler runs cold-cache with a different mix. And the excess is *consistent with* the tick rather
  than attributed to it -- nothing rules out another wall-time-proportional cost inside the
  baseline's bracket.

### 2. The next boot (board lane executes) — led by `speedtest1_instret.dom`

**The instrumented domain image is the ask, and it is worth a boot on its own merits.** Three
separate questions dead-ended on the same missing measurement on 2026-09-11, arriving independently
rather than as one complaint, so the request states what it DECIDES rather than asking for a
capability nobody has to weigh:

1. **Does the board's domain instruction count equal the emulated one?** Currently unknown for the
   domain arm in either direction. The claim that it did was retracted the same day it was written
   (see item 1) because sw56 carries cycles only. Every ratio in §7h pairs a measured cycle count
   against a *predicted* instruction count; one boot converts that into two measured columns.
2. **Is `cte` a real effect or a bad prediction?** It is the only pair whose cycle ratio exceeds its
   instruction ratio, 1.166 against 1.141. The tick correction makes it worse rather than better, and
   the CPI route is closed algebraically — domain CPI over native CPI *is* cycle ratio over
   instruction ratio, so it cannot be independent evidence about the same discrepancy. A real effect
   and an error in that one testset's predicted domain count are indistinguishable without a
   board-side domain instret, and distinguishable immediately with one.
3. **Which denominator the paper may use.** §7h states two defensible ratios and refuses to mix them,
   because a board native instruction count under an emulated capability numerator compares a ticked
   machine with an unticked one. With both arms measured on the board that question stops being a
   framing choice forced by missing data and becomes a straightforward reporting decision.

**DECIDED with the board lane: its own boot, and the `DOMAIN_BASE_VA` knob does NOT go in.** Two
SQLite images cannot share a boot because every one links at `0x10000`, and the obvious fix — three
lines of relink — is the option with a documented failure rate. R-17 (`ref/ISSUES.md:3307`, **OPEN —
NOT ROOT-CAUSED**) is "a ~1.6 MB domain hangs after ANY perturbation of its image": two images
differing by one dead, never-called, empty function, one returning on the board and one never
returning, silently, with no trap and no marker, while QEMU runs both identically. **Nine
structurally different perturbations were built and every one hangs; only unmodified builds return.**
A base-VA relink is a perturbation of that class and it is not on R-17's tested-and-excluded list —
that list excludes the address of the *executed code* being the same in both, which is the opposite
condition. So the knob would make the instret image a coin toss whose tails side is a silent hang
indistinguishable from a result, on the one arm the boot exists for.

**The evidence is not all one way and the entry should say so.** The seven-testset image is itself a
large perturbation of the three-testset one — +176,760 bytes, ten more globals, a different define
set — and it returned on the board twice, in sw52's family and again across all seven arms of sw56.
So R-17 does not govern this image family in the literal form its title states, and the argument for
a separate boot rests on the cost asymmetry rather than on a high probability of hanging: a wedge on
a dedicated boot costs instret arms only, and a wedge on a shared one costs the pairs as well.

**The instret image must be REBUILT from the seven-testset source.** The one in
`~/capstone-artifacts/speedtest1-boot1/` is 1,684,344 bytes against that set's plain 1,684,232 — the
sw52-era three-testset workload, 176 KB smaller than the image sw56 actually measured. Booting it
would answer an instruction-count question about a program we no longer run. (Its own build log
opens `== minstret bracket ON -- separate image, stage it LAST`, so the image has always been
expected to be the risky one; the 112-byte delta against its own plain sibling is R-17's exact
shape, and that risk is present however it is staged.)

**Boot shape**, needing no build change beyond the rebuild: control `k800` first as always, then the
seven instret arms ascending — `star`, `parsenumber`, `orm`, `main`, `fp`, `cte`, `rtree`. Ascending
order means an R-17 hang costs the expensive end and nothing cheaper, and a minstret read that
faults outright fails on `star` within seconds, costing one boot rather than a measurement anyone
wants.

**RUN AS BOOT sw57 ON 2026-09-11 — all three questions answered; results in
`docs/ref/fpga-silicon-measurements-for-paper.md` §7i.** 8 arms, control passed, zero failures. The
board's domain instruction count equals the emulated one to between 5.2e-08 and 1.4e-06 on all seven
testsets, which is the check retracted that morning now actually made; `cte`'s prediction was right
to +105 instructions, so its anomalous ratio is a real effect rather than a bad number; and the
denominator gap is measured at ~6 %, the tick and nothing else. Domain CPI is directly measurable for
the first time. What follows is the pre-boot record, kept because the predictions were written before
the run and that is what makes them falsifiers.

**BUILT AND VERIFIED, 2026-09-11**, at `~/capstone-artifacts/speedtest1-instret7/`:

    speedtest1_instret7.dom   1,861,136   c8766ec3331089ee…
    sqlite_host.user             23,392   00237b7856f854ba…

**The predictions, measured on the INSTRUMENTED image itself** rather than carried over from the
plain one — it is a different binary, and the boot exists precisely because a real effect and a
wrong prediction are currently indistinguishable for one testset:

| testset | predicted instret | predicted mcycle | hash |
|---|---:|---:|---|
| `star` | 59,443,559 | 59,443,520 | `0 0e12171d` |
| `parsenumber` | 60,938,716 | 60,938,677 | `0 0e12171d` |
| `orm` | 274,563,761 | 274,563,722 | `408505 35f60ec9` |
| `main` | 696,764,734 | 696,764,695 | `111130 1e792c9d` |
| `fp` | 933,826,083 | 933,826,044 | `8 6e798592` |
| `cte` | 2,019,055,526 | 2,019,055,487 | `186 9849fb27` |
| `rtree` | 2,420,439,710 | 2,420,439,671 | `12913 f8cd0f1d` |

**Every hash equals the plain seven-testset image's on the same testset**, so the minstret bracket
did not perturb the computation — which is a different question from equalling the native oracle,
and it is the one the build script's own note warns about, instrumentation having flipped a rung to
a deterministic miscompute on this silicon before.

**Two constants that are the positive control on the instrumentation.** `instret − mcycle` is
**exactly 39 on all seven arms**, across a 41-fold range of workload size — the bracket asymmetry,
`i0` read before `c0` and `c1` before `i1`. And the instrumented image's count sits 20 to 30
instructions *below* the plain image's, also without scaling. A bracket whose offset tracked the
workload would be measuring something other than what it brackets; these do not.

**What the boot falsifies.** Each arm's board instret should equal its row above. An arm landing
within a few tens of instructions confirms the emulated and silicon counts agree and settles
question 1. `cte` is the arm to read first: if its board instret matches 2,019,055,526 then its
prediction was right and its anomalous ratio is a real effect, and if it does not, the prediction
was wrong and §7h's `cte` row needs redoing. Either answer is worth the boot; only the absence of
the measurement is not.

Three more things fit the same boot and none justifies one alone:

- **The position control.** `main`'s domain arm, run twice off one image in one boot: once pinned to
  its sw52 position and once again last. Every cross-boot delta so far moves image, position and
  region provenance together, so a small delta is as consistent with cancellation as with no effect.
  This is the only comparison in any of this with exactly one difference in it.
- **~~`dmesg | grep 'Domain memory region'`~~ — STRUCK, already answered.** The paddr that closes
  RETRACTION 6 was in the sw56 and sw57 transcripts all along, as the monitor's own `DBAS` trace
  rather than as module `pr_info` output: sixteen domain creations, every one at 0x81–0x83, against a
  CMA area at 0xAC000000. No arm needed.
- **`app` is no longer a candidate arm.** It is root-caused off-board (item 4) and needs no board
  time at all.

### 3. Move the arena into a region — the ceiling that closed `json` and depth

`json` and bigger `--size` were both written off on a 4 MiB arena ceiling. The ceiling is real and I
have now re-verified it at the source rather than carrying it: the board kernel has no
`CONFIG_ARCH_FORCE_MAX_ORDER`, and 6.4.14 defines `MAX_ORDER 10` *inclusive*
(`mmzone.h:28-32`, checked as `order > MAX_ORDER` at `page_alloc.c:4744`), so `__get_free_pages`
tops out at order 10 — 4 MiB — and the arena, being a static array in the domain's globals storage,
lives under it.

But the arena does not have to live there. `domdata-budget.py` on the delivered image:

    declared        dom_data>=   3379024  (stack 1048576)
    image           code_len=    1621624
    allocation      TWO regions: code order=9 (2097152), data order=10 (4194304)
    dom_data           4192768
      - blob            114304   (globals_off=0x170000 .. code_size)
      - cap table         3488   (218 globals)
      - storage        2212656          <- the 2 MiB arena is 2097152 of this
      = STACK          1862320

Take the arena out and storage falls to ~115,504, the declared `dom_data` need falls from 3,379,024
to ~1,281,872, and the data allocation drops from order 10 to order 9. The arena's ceiling stops
being the buddy allocator's and becomes the region's — **130 MiB, demonstrated on 2026-09-11**, in a
256 MiB CMA area.

The mechanism is already in the domain and costs one branch. `domain_main` receives each region as a
plain pointer keyed on `shared_region_count`: 0 is the hostcall metadata, 1 the payload. A third
region becomes the arena, and `sqlite3_config(SQLITE_CONFIG_HEAP, arena, len, 64)` takes it exactly
as it takes the static array today — memsys5 wants a byte buffer and does not care where it came
from. The host creates it and hands it over; it never needs to `mmap` it, so `MAP_SIZE_LIMIT` does
not bind.

Then, in one boot: `json` at size 1 with a 6 MiB arena, and `main`/`orm` at `--size 5`.

**Re-derive, do not assume.** An arena drawn from a region is a different geometry, and geometry is
S-14's suspected common factor — the fault is a *pre-entry* one at `SQ: E/share1`, and this change
adds a third region *and* moves the image from order 10 to order 9. Both directions are live: it may
trip S-14 on an image that runs today, or it may be the lever that gets `json` past it. The heap
sweep, the budget gate and the pre-entry check all get re-established on the new footing before any
number from it is quoted, and the arm is ordered last in its boot.

**It also settles placement for free, which is why it is worth building even if depth were not
wanted.** The region-arena arm puts the working set in CMA at `0xAC000000` while the `.bss`-arena arm
keeps it in buddy memory — same boot, same image family, one difference. Run both on `main --size 1`:
if the instruction counts match, the cycle difference is the placement effect and the caveat becomes
a number; if they differ (a region pointer may not codegen identically to a `.bss` address), compare
CPI instead and say so. This is the measurement the board lane wanted a device-tree change for, and
unlike that change it is not boot-global.

### 4. `app` — ROOT-CAUSED off-board: SQLite carves Expr nodes at 8-byte granularity

`app` reproduces under QEMU with the delivered artifacts, so the board is not needed for it. It
enters (`SQ: G/enter` present) and then:

    [CAPSTONE] Unaligned cap access (addr = 0x101fb8168)
    [CAPSTONE] domain halted by capability fault: cause = 4, pc = 0x101cd5d9c,
               tval = 0x1015b0000, badaddr = 0x1015b0000

**CORRECTION, made the same evening: I first wrote that the `pc` on that line was stale and that the
instruction it names was not established. That was wrong.** `riscv_raise_exception(env, excp,
GETPC())` reaches `cpu_loop_exit_restore`, which calls `cpu_restore_state(cpu, pc)` whenever `pc` is
non-zero (`accel/tcg/cpu-exec-common.c:75-81`), so the state IS restored and `mepc` is the faulting
instruction. The explicit `cpu_restore_state` in the bounds path twenty lines below exists so that
*its own debug print* can read `env->pc`, not to fix the trap. I inferred a defect from the
asymmetry between the two paths without reading what the raise already does.

So the pc is good and the chain closes:

    pc 0x101cd5d9c, DBAS 0x101c00000  ->  image VA 0xe5d9c
    0xe5d9c:  5b 48 b5 00     stc  a1, 0x10(a0)          in exprDup+0x3b0
    store address 0x101fb8168  ->  a0.cursor = 0x101fb8158, which is 8 mod 16

**The mechanism is in SQLite's source and it is not a capability defect.** `exprDup` sub-allocates
`Expr` nodes out of one byte buffer and advances the cursor in 8-byte steps —
`sEdupBuf.zAlloc += ROUND8(nNewSize)` (amalgamation `:114522`), with
`ROUND8(x) = ((x)+7)&~7` (`:16088`) and `dupedExprNodeSize` returning `ROUND8(nByte)`. The next node
starts at `pNew = (Expr *)sEdupBuf.zAlloc` (`:114473`), so after any odd-multiple-of-8 predecessor it
is 8 mod 16, and the first capability-typed field stored into it faults. Same family as the R-29
granule guard: a structure packed for an 8-byte world, addressed by a capability that needs 16.

`app` reaches it and the other seven testsets do not because the omission of floating point was also
disabling the datetime code that builds these expression trees.

**FIXED AND VERIFIED 2026-09-11**, in `capstone-qemu` `cabc953e58`, on a rebuilt canonical binary.
Re-running `app` on the instrumented image now reports:

    [CAPSTONE] Unaligned cap access: insn = 00b5485b, pc = 101cd5d8c, pcc_base = 101c00000,
               va = d5d8c, rs1 = x10, cursor = 101fb8138, imm = 16, addr = 101fb8148, is_store = 1
    [CAPSTONE] domain halted: cause = 6, pc = 0x101cd5d8c, tval = 0x101fb8148, badaddr = 0x101fb8148

`cause = 6` rather than 4, and `tval`/`badaddr` now equal the real faulting address instead of a
stale `0x1015b0000`. **The instruction word `00b5485b` is the same `stc` the hand-derived analysis
named on a different build of the program**, and `cursor 0x101fb8138` is 8 mod 16, so the root cause
is confirmed on a second binary by an instrument that is now reporting rather than inferred from one
that was not.

**Two genuine instrument defects found on the way, both worth filing separately from `app`:**

- **No capability fault path assigns `env->badaddr`**, so it retains whatever the last ordinary
  fault left there — here `0x1015b0000`, unrelated to the real faulting address `0x101fb8168`. That
  field has been readable-looking and wrong on every capability fault line ever printed. **Word the
  filing that way and not as "badaddr does not appear":** `grep -c badaddr target/riscv/op_helper.c`
  returns **2**, at `:656` and `:1002`, and a reviewer who runs the obvious grep and sees 2 stops
  reading. Both are prose inside comments and neither assigns; every real `env->badaddr =` in the
  tree is in `cpu_helper.c` (`:1217`, `:1262`, `:1287`, `:1990`), all on ordinary MMU paths.
- **A misaligned capability STORE raises `RISCV_EXCP_LOAD_ADDR_MIS`**, cause 4, rather than
  `STORE_AMO_ADDR_MIS`, cause 6. Unconditional on `is_store`.

**AND THE CANONICAL EMULATOR WAS TWO DAYS STALE, which the rebuild also retired.**
`build/qemu-system-riscv64` dated 2026-09-08 and predated Q-07 (`72fb56be86`, 2026-09-10), so every
QEMU result taken between those dates came from a binary missing the revoke-polarity and STC
cursor-advance fixes — including all of tonight's speedtest1 verification. **It did not affect these
numbers, and that is checked rather than assumed:** `star` returns 59,443,559 instructions and hash
`0 0e12171d` on the rebuilt emulator, identical to the stale one, and sw57 had already matched the
stale emulator's counts against silicon to 1.2e-06. speedtest1's measured path performs no
revocation and creates no UNINIT capability, which is all Q-07 touches. Only `op_helper.c` needed
recompiling, so the binary was stale for want of a `ninja` rather than for any deeper reason.

### 4a. CLASSIFY BY THE DIAGNOSTIC, NOT BY THE MARKER SEQUENCE

Written after getting it wrong on 2026-09-11 and having the claim refuted by an auditor before it
reached the registry.

A 2.5 MiB-arena image was reported as an S-14 instance on the strength of `SQ: E/share1` present and
`SQ: G/enter` absent. **That condition is satisfied by a pre-entry capability fault AND by an
emulator abort**, so it could not separate the two hypotheses actually on the table — and the image
had in fact aborted QEMU on a `CSSPLIT` assert (now **Q-10**), never taking a capability fault at
all. `cause =` appears **zero** times in all five logs from that experiment, while firing in eleven
others in the same directory, so the detector worked and the images simply never produced the
signature.

**The rule:** a classifier reads the diagnostic line. Marker presence and absence describes how far a
run got, not what happened to it. Concretely, in order:

    helper_cssplit: Assertion   -> QEMU abort (Q-10), NOT a domain fault
    cause = 24                  -> the S-14 signature
    domain halted ... cause = N -> a capability fault, and N is not 24, so it is not S-14
    result markers present      -> RUNS

**Second failure in the same script, same event.** The section-size extraction used
`awk '$2==".text"'`, but `llvm-readelf -S` renders `  [ 1] .text`, so `$2` is `1]` and it never
matched. It printed `?`, the `?` was reported rather than chased, and that is why `.bss` moving by
exactly the same 524,288 bytes as the carve went unseen — which would have shown the two could not be
separated anyway. **A `?` from an instrument is a result about the instrument.**

### 4b. The denominator was wrong, and `trigger` is broken upstream

**Ten benchmark testsets, not nine.** `speedtest1.c` defines eleven `testset_*` functions; `debug1`
is self-described as a self-test, leaving ten. The nine came from an early blocked-testset table that
omitted `trigger` and was never checked against the source. Corrected here and in §7k, where the
denominator was implied by an omission rather than stated.

**`trigger` cannot run on any platform.** Tried for the first time on 2026-09-11: it fails identically
in the domain and on a stock native build of the same source, `SQL error: no such table: t1`.
`testset_trigger` creates `z1`, `z2`, `t3` and then inserts into `t%d` for 1..3; `CREATE TABLE t1`
occurs nowhere in the file. An upstream defect in 3.53.3, not a capability limitation, so the
reachable maximum is nine of ten.

The check that found it is worth naming because it is cheap and nobody had run it: **enumerate the
testsets from the source rather than from our own notes.** A denominator that no one derived from the
artifact is exactly the kind of number that survives into a paper.

### 5. Two instruments, one latent and one deferred

- **`domdata-budget.py` reads the wrong `MAX_ORDER` convention.** It computes
  `CONFIG_ARCH_FORCE_MAX_ORDER - 1`, which was right when `MAX_ORDER` was exclusive and is off by one
  on 6.4, where it is inclusive. The symbol is currently unset, so the fallback returns 10 and the
  tool is right by coincidence — the moment anyone sets it, the tool reports half the real ceiling
  and "DOES NOT FIT" for images that load. Fix the branch, and negative-test it by setting the symbol
  in a scratch config.
- **`capstone.c:144` calls the buddy block a "region".** It is the wording both lanes read wrong on
  the way to RETRACTION 6. **The two copies have now converged** (`959cdba456a3` in both
  `caplifive-system` and `caplifive-buildroot`), so the original blocker is gone — but the board lane
  has just fast-forwarded, rebuilt and verified `fw_payload 805cde707751`, and touching module source
  now makes the built `.ko` stale against it for zero functional gain. It goes in with the next
  module change, when a rebuild is happening anyway, and that rebuild needs `A=modcapstone-rebuild`
  FIRST: `A=linux-rebuild` and `A=opensbi-rebuild` both return 0 and silently carry the previous
  `.ko`, because buildroot has already stamped `.stamp_rsynced` for the package.

### 6. Quantify the soft-float implementation share

The caveat under item 1 is currently a direction, not a size. What settles it is the share of
instructions spent inside the soft-float routines on each arm of `fp`, `cte` and `rtree` — libgcc's
on the baseline, compiler-rt's in the domain. Off-board, and after the pairs are recorded, not
before: it refines a caveat that is already named, and naming it is what the paper needs first.

## Phase 4 — the other comparisons

- **Allocator arms** (memsys5, umm_malloc, the existing revoke-on-free allocator) are the cheapest
  extension, but they are **three SQLite `.dom` files**, so the one-image rule *does* bite here. The
  fix is a three-line `DOMAIN_BASE_VA` sed copied from `build-ladder-domain.sh`; without it they cost
  three boots. Say which before scheduling.

  **The vendored umm_malloc cannot host these testsets at its current block size, and that is
  arithmetic rather than an experiment.** Its free-list indices are 15 bits
  (`UMM_MAXBLOCKS = UMM_BLOCKNO_MASK = 0x7FFF`, `umm_malloc.c:95-96`), and the block-count check
  divides the arena by `UMM_BLOCKSIZE`, which is `sizeof(umm_block)` and **not**
  `UMM_BLOCK_BODY_SIZE` — a distinction worth keeping, because it is the size the ceiling actually
  depends on. Measured rather than assumed, by compiling the struct for the domain's own target:
  `sizeof(umm_block)` is 32 (body 32, `UMM_HEADER_SIZE` 16, no packing attributes in this
  configuration). So the largest addressable arena is 32,767 × 32 = **1,048,544 bytes**.

  `main --size 1` needs 1.5 MiB under memsys5 and `orm --size 1` needs 2 MiB, both above that.
  **The conclusion does not depend on pinning the struct size**: at 36 bytes the ceiling would be
  1.12 MiB and at 40 bytes 1.25 MiB, so every plausible value stays below the smallest arm's
  requirement. It is a stronger "no" than exact arithmetic would make it look.

  `umm_init_heap` refuses an oversized arena rather than wrapping the index
  (`umm_malloc.c:308-312`), which is the good half. The bad half is that it returns `void` and leaves
  the allocator uninitialised, so the caller learns about it by crashing rather than from an error.

  Raising `UMM_BLOCK_BODY_SIZE` to 96 would reach ~3 MiB, which is the whole usable budget, at the
  cost of up to 96 bytes of internal fragmentation per allocation on a workload made largely of small
  objects. So the allocator arm is a question about *overhead at a size all three can run*, not a
  route to a bigger workload — `parsenumber` is the testset that fits every candidate, and it is
  where this comparison should start.

  **This does not weaken the case for the arm; it sharpens it.** The heap ceiling found in Phase 1 is
  what makes allocator choice load-bearing rather than decorative: memsys5 is a buddy allocator, its
  fragmentation is what the 4 MiB `dom_data` cap is being spent on, and the revoke-on-free allocator
  never coalesces at all, which is a deliberate cost its own header names.
  **MEASURED 2026-09-10, before building the arm: the revoke-on-free allocator can run
  `parsenumber` and nothing else.** Its arena is consumed by one-way `SPLIT` and `xFree` can never
  return space, so what it needs is not the run's PEAK live bytes but the SUM of every allocation
  the run ever makes. An opt-in counting wrapper around memsys5
  (`CAPSTONE_SPEEDTEST1_ALLOCSTATS`) measures both:

  | testset, size 1 | carved (sum of all) | peak (live at once) | allocations | ratio |
  |---|---|---|---|---|
  | `parsenumber` | 3.03 MiB | 0.04 MiB | 9,704 | 78× |
  | `orm` | 7.41 MiB | 1.48 MiB | 31,749 | 5.0× |
  | `main` | 17.08 MiB | 1.23 MiB | 32,639 | 13.9× |

  Against a shared-region ceiling of about 4 MiB, only `parsenumber` fits, at 76 % occupancy.
  `orm` needs nearly twice the ceiling and `main` more than four times it. **So the arm is one
  testset, and that is known for the cost of a ten-line wrapper rather than a third shared region,
  a host argument form and a second SQLite image.**

  The 78× on `parsenumber` is the generality cost stated as a number, and it is the interesting
  result rather than a disappointment: that testset holds 40 KiB live and churns 3 MiB through the
  allocator, which is exactly the shape a never-coalescing design punishes hardest.

  **An unplanned cross-check falls out of it.** The peak figures come from a completely different
  instrument than the heap-minimum sweep, and they agree: `main` peaks at 1.23 MiB and needs a
  1.5 MiB memsys5 arena, `orm` peaks at 1.48 MiB and needs 2 MiB. Buddy-allocator overhead of 1.22×
  and 1.35× over live bytes is plausible, and two independent measurements landing consistently is
  worth more than either alone.

  The census costs a few per cent of run time, so its runs are diagnostics and the reported cycle
  numbers come from builds without it.

- **CHERI is not a swappable arm.** It needs a different bitstream flashed, which evicts the Capstone
  one, so it is a separate session with a reflash on each side, and a reflash is ask-first and the
  lead's. There is also no CHERI Linux or purecap userspace for that SoC in the tree; **ask the cheri
  lane for the shortest path** rather than sizing it here.

## Verification

- Phase 0's three answers are recorded before Phase 1 starts.
- The dropped-byte counter reads zero in the reported run, and is shown able to read non-zero.
- The clock is shown to move: stub versus `mcycle`-backed, zeros versus not.
- **The size knob is shown to bite**: two sizes must produce visibly different row counts and
  materially different cycles, or a mis-plumbed size reads as a valid measurement of the wrong thing.
- Board readings from the run's own transcript segment, `k800 = 4` first, into
  `tests/board-results/*.tsv`.
- The overhead ratio uses the baseline's warm pass only, per the existing definitions.

## Risks

- The floating-point decision gates the arithmetic testsets, and Phase 0 decides whether a narrower
  run avoids it. This is the plan's largest dependency.
- The stdio shim is new code in the measurement path; keep it outside the cycle brackets, as the
  ladder does for its harness.
- R-17's axis is **perturbation, not size**: its own excluded list names image size, with three
  same-size images that hang anyway. A new program is neither perturbed nor unmodified-and-known, so
  size is not the thing to watch. If it hangs, the discriminator is whether an unperturbed rebuild of
  the same source also hangs.
- The one-TU rule means speedtest1's file-scope statics must live in the amalgam TU; compiling it
  separately is the silent-wrong-data failure that rule exists to prevent.
