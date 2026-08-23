# Proposal: what it would take to claim SQLite "passes its regression suite"

**Status: proposal for review. Nothing here is started.**

## The gap, stated plainly

What we have today is `benchmarks/sqlite/sqlite_capstone_domain.c` — **written by this project**,
first landed `3c5815ee45d8` (2026-06-30). It drives the genuine upstream **SQLite 3.53.3
amalgamation**, but the assertions are ours: **10 `exec_ok` + 8 `query_scalar_eq`**, ~18 checks.

**No upstream SQLite test harness is vendored anywhere in the tree** — no TCL `testfixture`, no
SQLLogicTest, no TH3. Verified by search.

So:

| claim | supported today? |
|---|---|
| "SQLite 3.53.3 runs end-to-end in a pure-capability domain on silicon, with results verified against expected values" | **YES** — 3/3, control green |
| "SQLite passes its regression suite" | **NO** |
| "SQLite fully works" | **NO** |

18 assertions is a good integration test. It is not a regression suite, and a reviewer who checks
will find that out.

## Options

**A. SQLite's TCL suite (`testfixture` + `tester.tcl`).** The canonical one. Requires a TCL
interpreter, a filesystem and `exec` **inside** the domain. The domain budget is ~1.55 MB today and
**must stay under 2 MiB** or the loader cannot even allocate it (see Q-01). Assessed **infeasible
in-domain**. A variant — testfixture on the host, SQLite core in the domain, one boundary crossing
per API call — is possible in principle because the host/domain split already exists, but it is a
large lift and it changes the performance story the paper also depends on.

**B. SQLLogicTest.** Text-driven: statements plus queries plus expected result hashes. The runner is
small C; the corpus is *data*. **This is the realistic option** and it is what the rest of this plan
assumes.

**C. Broaden our own workload.** Cheapest. Never supports "passes its regression suite" — it moves
"18 assertions" to "60 assertions". Worth doing anyway as a by-product, not as the answer.

## ~~THE CENTRAL RISK~~ — ~~B0b RAN, AND THE RISK IS VOID~~ — **PARTLY RETRACTED 2026-08-21**

**The feared risk was:** the `SQLITE_OMIT_*` set that makes SQLite fit in a capability domain is
the same set that makes an upstream suite fail, so the claim would have to be scoped to a crippled
configuration.

**It does not apply. The shipped build carries NO `SQLITE_OMIT_*` flags at all.**
`build-sqlite-silicon.sh` defines a 14-flag `SILICON_TRIM` array at `:877-897` and then, at `:911`:

    [[ "${SQLITE_TRIM:-0}" == "1" ]] || SILICON_TRIM=()

`SQLITE_TRIM` defaults to `0`, so the array is emptied. The comment above it is explicit — the trim
was **measured to break SQLite** (2026-07-31: compiles and links clean, then faults at the domain's
first entry; the same tree passes end-to-end without it), because SQLite supports `SQLITE_OMIT_*`
only when building from canonical sources, not against the prebuilt amalgamation. *"Opt in with
`SQLITE_TRIM=1` only to re-measure the carve count; never for a correctness run."*

~~**So the SQLite that passed 3/3 on silicon is a feature-complete amalgamation build**, and an
upstream suite would not be bounded by an omission set.~~ (The 15th flag, `SQLITE_OMIT_SELECT`, was
commentary about SQLite's own source — not ours, as suspected.)

### RETRACTION, 2026-08-21: the build is NOT omission-free

**`SILICON_TRIM` is not the only `SQLITE_OMIT_*` list, and checking only it was the error.**
`build-sqlite-silicon.sh:848` harvests a *second* list out of `build-sqlite-capstone.sh` —
`SQLITE_DEFINES`, which is **always active** and carries **seventeen more omissions**, among them:

    -DSQLITE_OMIT_FLOATING_POINT=1   -DSQLITE_OMIT_JSON=1      -DSQLITE_OMIT_FOREIGN_KEY=1
    -DSQLITE_OMIT_UTF16=1            -DSQLITE_OMIT_EXPLAIN=1   -DSQLITE_DQS=0

Verified by capturing the actual `clang` invocation, not by reading the script — the earlier
claim was made by reading one variable and stopping.

**How it surfaced, and why that matters more than the error itself.** The first domain run of the
negative control reported five statement failures the native baseline did not have:
`INSERT INTO t1 VALUES(3,'ccc',1.5)` → `near ".": syntax error`. With `SQLITE_OMIT_FLOATING_POINT`
the tokenizer does not accept a decimal point at all. **An ordinary configuration difference was
presenting as a capability defect** — which is exactly the failure the shared-runner design exists
to prevent, and it was caught only because the baseline is built from the same source. The native
baseline now harvests the same define list (`build-slt-native.sh`), and both sides agree.

Consequences for the plan, none of them fatal:

* **The R (real) column type is unreachable in this configuration.** Only one file in all 622 uses
  it, so the subset loses nothing measurable — but the runner's `%.3f` path is dead code here and
  must not be described as exercised.
* **`SQLITE_OMIT_FLOATING_POINT` does not compile as shipped** in 3.53.3: the `#else` arm of
  `sqlite3AtoF` refers to a `z` declared only in the `#ifndef` arm. `build-sqlite-capstone.sh:66`
  already patches it; the native baseline applies the identical one-line fix.
* **The claim must name its configuration.** "Passes N% of SQLLogicTest" is not available;
  "agrees with its own native build, configuration enumerated" is, and is what stage 4 writes.

## THE REAL CEILING, found in the same place: capability carves, not features

`build-sqlite-silicon.sh:913-919` — one capability carve per global costs one revocation node, and
**the board's rev-node allocator wraps after 1021**. Untrimmed SQLite needs **1059 carves and
overflowed the pool on silicon** (measured 2026-07-31, head=74 with the overflow flag set). String
merging of private read-only literals takes it to **179 carves, ~215 allocations**.

**This is the budget an SLT runner spends against, and it replaces the OMIT risk as B1's principal
constraint.** A runner plus its buffers adds globals, and globals become carves. Current headroom
is roughly 179 → 1021. Track carve count, not just image size — and note the two ceilings are
independent: the 2 MiB image limit (Q-01) and the 1021 rev-node pool.

## Staged plan — each stage says what it licenses

| # | Work | Board? | What it licenses |
|---|---|---|---|
| 0 | **Fix Q-01** — rebuild the QEMU arm at silicon config so `code_len <= 2 MiB` | no | A working reference. Everything below is developed against QEMU; silicon only confirms. |
| ~~0b~~ | **DONE 2026-08-20 — the live `SQLITE_OMIT_*` set is EMPTY.** `SILICON_TRIM` is gated off at `build-sqlite-silicon.sh:911` and was measured to break SQLite. Superseded by: **track capability carves against the 1021 rev-node pool** (179 today, 1059 untrimmed). | no | No corpus section is excluded by feature omission. The binding budget is carves, not features. |
| 1 | **One SLT file end-to-end in-domain** — plumbing only: stream the file through the existing shared region, execute, hash, report | no | That the mechanism works. Licenses nothing about SQLite. |
| 2 | **A subset corpus under QEMU**, pass rate measured | no | The real number. Every failure here is ours and needs no board. |
| 3 | **Same corpus on silicon**, compared against the stage-2 baseline | yes, a few boots | Silicon-vs-QEMU divergence — the thing we currently cannot detect at all. |
| 4 | **Write the claim from the measured rate** | no | Whatever stages 2-3 actually support, and not more. |

**Why stage 0 is first and not optional:** with silicon now *passing*, we have no reference model to
attribute a future silicon failure against. Stage 3 is meaningless without stage 0.

**Why the corpus must stream:** it cannot be baked into the domain image — the 2 MiB ceiling is the
same one that produced Q-01. The shared region (`SQLITE_HC_REGION_SIZE`, already used for output) is
the only channel.

## Effort, honestly

Stage 0 and 0b: hours. Stage 1: the real engineering — days, and the risk sits here. Stages 2-4:
mostly running and reading. **This is not an afternoon**, and it should not be started on the
assumption that it is.

## Decisions for the lead

1. **Is the scoped claim enough for the paper?** "Passes N% of SQLLogicTest for the tested
   configuration, omissions enumerated" — if yes, this plan is right-sized. If the paper needs the
   unqualified claim, option A and its cost need discussing first, because B will not get there.
2. **Is stage 3 worth the board time**, given the S-10 reflash and the rate ladder are competing for
   the same hardware?
3. **Should stage C (broaden our own workload) run in parallel?** It is cheap and improves the
   fallback claim if B stalls.

---

# Stage 1 design, settled 2026-08-21 — and the plan's assumption was wrong

**The plan said "stream the file through the existing shared region". That mechanism does not
exist.** Three findings, in the order they were established:

**1. The shared region is OUTPUT-ONLY and ONE-SHOT.** `sqlite_hostcall.h` defines a
`{phase, opcode, offset, length, result, error}` block that *looks* like a request/response
protocol, but the host never dispatches on `opcode` — it is written once as a probe-stage marker
(`sqlite_host.c:134`), and the payload is read exactly once, **after the domain returns**
(`:147`). There is no way for the domain to ask the host for data mid-run.

**2. RE-ENTRY DESTROYS DOMAIN STATE.** `ioctl_call_dom` forwards straight to the SBI call with no
teardown, so a domain *can* be called repeatedly — but the entry glue rebuilds the cap-table
**"on reentry"** (`start-gp-captable-generic.S:30`), and `BUILD_GP_CAPTABLE` re-runs every
global's initialiser stores. **An in-memory SQLite database would not survive a second call.** So
"host feeds chunk N, domain accumulates" is not available either.

**3. But the region size is a PARAMETER, not the 4 KB the header implies.** The host calls
`create_region(SQLITE_HC_REGION_SIZE)` (`sqlite_host.c:116-125`) and `SQLITE_HC_REGION_SIZE` is
our own `#define`. The module allocates it with `__get_free_pages(order)`, so it is bounded by the
same order-10 buddy limit that produced Q-01: **up to ~4 MB**.

## The design that follows

**One large shared region, one `call_dom` per SLT file.** Raise the payload region to the megabyte
range, have the host write a whole test file into it before the call, let the domain execute that
file and write results back, and return. Repeat per file.

This sidesteps both problems rather than solving them: no streaming protocol is needed because the
file arrives whole, and no state preservation is needed because **SLT files are self-contained** —
each creates its own tables. It uses only existing primitives (`create_region`, `map_region`,
`shared_region_annotated`, `call_dom`), so stage 1 adds a *runner*, not a *mechanism*.

**Budgets to watch, and they are independent:**
- the region: ≤ ~4 MB (order-10), so a file larger than that must be split at a test boundary;
- the domain image: ≤ 2 MiB of code (Q-01's ceiling);
- capability carves: ≤ 1021, 179 used today — the runner's globals count against it.

## Stage 3 caveat, from the RTL lane 2026-08-21

**8 of 16 legs of the write-buffer residual are LIVE on the flashed `caplifive_s07fix.bit`** (its
own committed sweep, `s07-strip.txt`: test 9, control 17). A load can hand back a dereferenceable
capability over memory the program already scrubbed. **If SQLite behaves oddly on this bitstream,
that is a candidate cause and it is not the runner's bug.** S-10 alone is now measured to close
that residual (9 → 17 with the control pinned at 17, plus a model-identity control), so stage 3 is
worth more after the S-10 reflash than before it.

## The large-region assumption is now MEASURED, not inferred (2026-08-21)

The design above rests on "the region size is a free parameter up to ~4 MB", which was inferred
from `create_region(len)` and the allocator's order-10 limit. Inference of exactly that shape has
been wrong repeatedly, so it was tested by changing the one `#define` both halves share and
running the real workload end to end:

| `SQLITE_HC_REGION_SIZE` | result |
|---|---|
| 4 KiB (as committed) | works |
| **1 MiB** | **works — all five markers, zero failure signals** |
| 64 MiB | **FAILS** — `SQ: X/fail`, `map_region failed`, zero markers |

**The 64 MiB arm is the control and it is what makes the 1 MiB arm mean anything.** Without it a
pass at 1 MiB is equally consistent with the constant not reaching the build at all. It fails at
`map_region` rather than `create_region`, which is worth knowing: the ceiling bites at map time.

**So a megabyte-scale region is available and proven, and that is the mechanism stage 1 needs.**
The header is left at 4096 deliberately — raising it belongs with the runner that consumes it,
not as a change with no consumer.

## Corpus: obtainable, canonical layout

The sqlite.org tarball is behind a login (302 to `/login`), but the GitHub mirror
`gregrahn/sqllogictest` carries the standard tree — `select1.test` … `select5.test`, plus
`evidence/`, `index/`, `random/`. Fetch-with-a-SHA, do not vendor: same policy as
`fetch-sqlite.sh` and `fetch-musl.sh`.

## Format, from the sqlite.org wiki rather than memory

Line-oriented ASCII. Records separated by blank lines; `#` starts a comment; comments do not
separate records. Two record kinds matter:

    statement ok            statement error
    <one SQL command, no trailing semicolon>

    query <type-string> <sort-mode> <label>
    <SQL>
    ----
    <expected values, one per line>

Type string is per column (`I` integer, `R` real, `T` text); sort mode is `nosort`, `rowsort` or
`valuesort`; large result sets are compared by hash. Omitting `----` means an empty result is
expected. This is a few hundred lines of C, not a project — which is the main reason stage 1 is
smaller than the proposal originally assumed.

---

# STAGES 1 AND 2 ARE DONE — measured 2026-08-21, under QEMU

**The runner exists, it runs inside a capability domain, and it agrees with its own native
build over 9,371 records.** What follows is the measurement, what it licenses, and the one
file that does not agree.

## The method, and why it is a DIFFERENCE and not a rate

`slt/slt_runner.h` compiles unchanged for the host and for the domain, and
`slt/slt_native.c` links it against the same SQLite 3.53.3 amalgamation with the same
semantic define set. **The result is the difference between the two sides.** An absolute
pass rate would be contaminated by corpus-versus-engine artifacts that have nothing to do
with capabilities, and this corpus contains several: `evidence/slt_lang_aggfunc.test` alone
produces eleven, on any machine. With one runner they appear identically on both sides and
cancel. With two they would be indistinguishable from a capability defect — which is not a
hypothetical, it is what the `SQLITE_OMIT_FLOATING_POINT` discovery above looked like for
the first hour.

## Results — QEMU, silicon configuration, 4 MiB region, 1 MiB arena

Every field of every summary is EQUAL between domain and native unless the row says otherwise.

| file | records | queries | verdict |
|---|---|---|---|
| `slt/negative-control.test` | 21 | 10 | **identical**, including 2 `stmt_fail`, 4 `query_fail`, 2 `skip_cond`, 1 `parse_err` |
| `select1.test` | 1031 | 1000 | **identical**, 0 failures |
| `select2.test` | 1031 | 1000 | **identical**, 0 failures |
| `select3.test` | 3351 | 3320 | **identical**, 0 failures |
| `select4.test` | 3857 | 2617 | **identical**, 0 failures, 215 skipped for size |
| `evidence/slt_lang_aggfunc.test` | 80 | 67 | **identical**, including 11 shared corpus artifacts |
| `select5.test` | — | — | **DOMAIN DIES — see below. Not attributed.** |

**9,371 records agree.** 7,393 of the query records state their expectation as an MD5 of the
entire result set, so this is agreement over hashed result sets, not over scalars.

**The negative-control row is the load-bearing one.** Six of its arms are wrong on purpose
and two more must be skipped rather than passed. The domain reproduces every one. Without
it, "0 failures" on select1-5 would be equally consistent with a comparator that cannot
fail — which is the single most expensive mistake available on this project.

## select5.test — a real divergence, and NOT yet a capability defect

The domain enters, loads all 702,577 bytes, and dies inside the workload:

    qemu-system-riscv64: op_helper.c:627: helper_cscincoffset: Assertion `rs1_v->tag' failed

A `cincoffset` on an **untagged** capability — the S-07 family signature, and the point at
which silicon would raise `UNEXPECTED_OPERAND` rather than assert. The native baseline runs
the same file clean: 1436 records, 732 queries, 0 failures.

**Reading 3 — "QEMU is stricter than the hardware" — is REFUTED, from the RTL and not from
a comment.** `capstone-ariane/core/anvil_build/capstone_flu_unit.anvil`, `func CINCOFFSET`:

    if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)||
       (data.cap_rs2.metadata.cap_type!=cap_type_t::NOT_CAP)){
        call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)

Silicon traps on exactly this condition; only the failure mode differs. So whatever this
is, it fires on the board too.

## What is ESTABLISHED about select5 — audited 2026-08-21

**The fault site, and it is not the runner's code.** The emulator now prints the guest pc
and the untagged value before it aborts (capstone-qemu `29e90c40f8`):

    capstone-qemu: cincoffset with an UNTAGGED rs1 -- pc=0x101c544d4 rd=x11 rs1=x10 val=0x0 priv=3

* Image VA **0x644d4**, `cincoffset a1, a0, a1`, inside **`sqlite3VdbeExec`**
  (`0x5436c` + `0x10168`, function size `0x107bc`). SQLite's own bytecode interpreter —
  `slt_runner.h` never executes there, so **the runner is exonerated**.
* mmap load base **0x101c00000**.
* The operand is an **all-zero untagged word**, loaded by the `ldc a0, 0x10(a0)` immediately
  before it, from inside a validly-bounded, non-revoked capability's memory.
* **Deterministic, N=2**, across two QEMU builds and a domain rebuild.
* At a **2 MiB arena the same file passes**: `records=1436 stmt_pass=704 query_pass=732
  query_fail=0`, matching native exactly. The *record count* is what proves the file was
  consumed — `completed=1` does not, since a `halt` also sets it (select5 has no `halt`).

**This EXCLUDES the two capability families it looked like.** Not an S-07 tag strip: an
untagged `ldc` result carries the raw memory low word, and a revoked or tag-stripped
capability would have printed the compressed capability's non-zero low word. Not a
linear move-out either — those zero the whole register, but `a0`'s producer here is a
straight-line `ldc` two instructions earlier, and `0x644d4` is the target of no direct
branch in `.text` (checked with a decoder validated against objdump on 45,988 targets).

### RETRACTED, same day it was written: "an allocation failure reaching a dereference"

**I recorded a causal root cause the evidence does not carry, and an auditor refuted it.**
"More arena makes it pass" does not establish "an allocation failed", for three reasons,
each fatal on its own:

1. **The 1 MiB and 2 MiB arms are not a matched pair.** The arena is a static `.bss` array
   charged against `dom_data`, so changing it rebuilt the image: `.text` differs by 4 bytes
   and an address-normalised diff of the two disassemblies shows **91,931 differing lines**.
   "More arena fixes it" and "a different layout hides it" are not separated by that
   experiment.
2. **There is no control.** The native baseline deliberately excludes `SQLITE_ENABLE_MEMSYS5`
   (`build-slt-native.sh:45`), so it runs on system malloc with an unbounded heap. **No
   non-capability build has ever run select5 under a constrained memsys5 arena.**
3. **`oom=0` across the entire passing 1300-record prefix** at 1 MiB. Under the retracted
   claim SQLite would go from never once failing an allocation straight to a fatal unchecked
   NULL — the opposite of the select4 signature I cited in its support, where the clean OOM
   bucket recorded 2,772.

**Two slips corrected while I was at it**, both of the kind this project keeps paying for:
`sqlite3VdbeExec`'s size is 67516 (`0x107bc`) — I had reported 423190, which is that hex
value read as decimal; and the load base is `0x101c00000`, not the `0x101bf0000` I derived
by forgetting the segment's own `0x10000` vaddr. **My "the implied base is page-aligned, so
the mapping is corroborated" was circular** — any candidate passing a low-12-bit filter
yields a page-aligned difference by construction. The real corroboration is that 3,864 of
4,096 low-12 buckets contain **no** `cincoffset a1, a0, *` site at all, so a misattributed
pc would most likely have matched nothing (p ≈ 0.057 of a spurious single hit).

**Four hypotheses survive, and nothing measured so far separates them:**

* **H1** an allocation failed and SQLite dereferenced the NULL without checking;
* **H2** the slot was never written — an uninitialised read that only reads as zero because
  QEMU's fresh memory is zero (a blind spot documented at `build-sqlite-silicon.sh:1660`);
* **H3** a stray plain integer store zeroed a live capability slot — and plain stores are
  unchecked in this configuration, so nothing would catch it;
* **H4** a layout-dependent bug the 91,931-line rebuild reshuffles away.

### H2 IS DEAD — the poisoned-arena arm ran, 2026-08-21

`CAPSTONE_POISON_ARENA=1` fills the memsys5 arena with `0xA5` before `SQLITE_CONFIG_HEAP`,
and carries its own witness gate (a run that failed to arm returns `0xBADA5000` instead of
executing). The gate did not fire, so the arena really was poisoned. The fault still
reported:

    capstone-qemu: cincoffset with an UNTAGGED rs1 -- pc=0x101c5458c rd=x11 rs1=x10 val=0x0 priv=3

**`0x0`, not `0xa5a5a5a5a5a5a5a5`. Something WROTE that zero; the slot was not left
uninitialised.** H2 is refuted.

**And the fault survived a third distinct image.** The poison build shifts the site by
`0xb8` — image VA `0x6458c`, still inside `sqlite3VdbeExec`, and the seven instructions
around it are byte-identical to the 1 MiB build's:

    6458c: db 15 b5 18   cincoffset a1, a0, a1

Three separate builds (1 MiB, 1 MiB + pc/value printing, 1 MiB + poison) all fault at the
same logical site. **H4 — "a layout-dependent bug the rebuild reshuffles away" — is
substantially weakened**, though not formally excluded.

**The one caveat, stated rather than glossed:** the poison fills `sqlite_heap` only. The
argument above holds if the faulting slot lies in the arena, which is where SQLite allocates
its structures — but the load address was not captured, so "in the arena" is inferred, not
measured. If the slot were in `.bss`, on the stack or in the shared region, the poison says
nothing about it.

**H1 and H3 both survive.** The experiment that separates them is the remaining one: emit a
marker on memsys5's NULL-return path (the patch site is already scripted at
`build-sqlite-silicon.sh:1128`) and re-run at 1 MiB. A NULL return before the fault confirms
H1; none refutes it and leaves H3, a stray plain integer store zeroing a live capability
slot — which nothing in this configuration would catch, since plain stores are unchecked.

**NOT A BOARD BLOCKER, and this is why it is being left here.** The board runs a 256 KiB
arena, at which select4 already returns a CLEAN `oom` bucket (2,772 records, zero failures)
rather than faulting. select5 at 256 KiB would exhaust the arena long before reaching this
site. The finding is real and worth finishing, but it does not gate stage 3.

**Bisect, each slice verified clean natively before use, 4 MiB region / 1 MiB arena:**

| slice | records | queries | domain |
|---|---|---|---|
| `s5_800` | 800 | 96 | passes, matches native |
| `s5_1200` | 1200 | 496 | passes, matches native |
| `s5_1300` | 1300 | 596 | passes, matches native |
| `s5_1400` | 1400 | 696 | **INCONCLUSIVE** — entered and returned, no assertion, harness hit its 60 s prompt timeout |
| full | 1436 | 732 | **untagged cincoffset**, deterministic |

The trigger lies in records 1300–1436. Note the `s5_1400` row is neither a pass nor a
failure and must not be read as either.

## What this does and does not license

**Supported now:** *"SQLite 3.53.3 executes 9,371 SQLLogicTest records inside a pure-capability
domain and produces results identical to the same SQLite built natively with the same
configuration, including MD5 hashes of full result sets, with zero divergences."*

**NOT supported, and none of these should be glossed:**

* **This is QEMU, not silicon.** Stage 3 is the board and it has not run.
* **The corpus is a subset** — 6 files of 622. `select5.test` is excluded by a live failure.
* **The configuration omits 17 features**, floating point among them, so the R column type
  is unreachable and no float is ever compared.
* **Result sets above 4096 values are not compared** — 215 records in select4, all rowsort.
* **`select4` needs a 1 MiB arena**; at the silicon 256 KiB it reports `oom=2772` and
  evaluates 1,085 records. select1, select2 and the negative control fit in 256 KiB.

## Stage 3 — the board. Prerequisites, all board-free and all now met

* **The `.test` files must be baked into the initramfs.** There is no 9p mount on the board;
  they are staged into `overlay/test-domains` like any `.dom`.
* **No driver change is needed.** `run_sqlite_stages_fpga.py` passes its `path:selector`
  suffix through to the host verbatim, so `sqlite_slt.dom:--slt /test-domains/s1_81.test`
  invokes the runner directly.
* **A slice ladder bounds the runtime**, which is the real unknown on a 25 MHz core:
  `s1_81`, `s1_231`, `s1_531`, `s1_1031` — 50, 200, 500 and 1000 queries, each keeping all
  31 setup statements and each verified clean natively before it may be used.
* **Silence is expected and must not be read as a wedge.** The domain emits nothing between
  `SQ: G/enter` and `SQ: H/return` — the report accumulates in the shared region and the
  host prints it only after the domain returns. `SQLITE_IDLE_S` defaults to 30 s and WILL
  abort a healthy long run; set it from the measured cost of the smallest slice.
* **Region 1 MiB is enough** for every slice (largest is 258 KiB against a 512 KiB input
  half) and 1 MiB is already board-relevant, whereas 4 MiB is not yet measured on silicon.

**Board session shape** — one boot, one `.dom`, only the `.test` argument varies, so a single
firmware rebuild covers every arm:

1. the known-good basic SQLite workload — **control; a boot whose control fails is void**;
2. `negative-control.test` — 21 records with **known non-zero failures**, so silicon is
   proven to be running the comparator rather than returning a clean void;
3. `s1_81`, then `s1_231`, then `s1_531`, then `s1_1031` — ascending, every one expected to
   RETURN, and the first that does not is the bisection point;
4. **the MINIMAL select5 reproducer** last — never the 700 KB file. Until the crash is
   attributed, a select5 slot is either wasted (if it is a runner bug) or spent on a
   needlessly large arm (if it is a port defect). Reduce the prefix while the crash
   persists, then bake that; note the reduction may not go to a single record if the
   trigger is allocation-history-dependent. Per the ordering rule there may be at most one
   expected-to-wedge arm and it goes at the end.

**Two holes in the session's own failure classes, both cheap to close:**

* **The freshness gate will not check the shared `.dom`.** With `dom:--slt <file>` specs the
  gate's candidate picker takes the LAST half that names a staged artifact — the `.test`
  file — so the one artifact every arm shares goes unverified. Add one selector-free spec,
  or hash the `.dom` bytes inside the decompressed initramfs by hand, before booting.
* **R-16 is per-image, and the SLT domain is a NEW image.** The control arm proves the
  boot; it does not prove this image enters. One baked `QR_DRAW` redraw variant costs about
  eleven seconds of JTAG and insures the session against drawing a 100%-stalling image.

**Timeouts must both be raised, from measurement.** `SQLITE_IDLE_S` (30 s) and
`SQLITE_STAGE_TIMEOUT` (90 s) will each independently kill a healthy `s1_1031`; set them
from `s1_81`'s measured wall time, not from a guess.

**The QEMU reference for these arms must be re-taken at BOARD configuration.** The matrix
above ran at 1 MiB arena / 4 MiB region; the board runs 256 KiB / 1 MiB. Re-running the
exact arm set at board config removes every "the configuration differed" escape hatch when
a board number diverges, and costs minutes.

Read every board result against the **8 of 16 legs of the write-buffer residual that are
live on the flashed `caplifive_s07fix.bit`**: a load there can hand back a dereferenceable
capability over scrubbed memory. If SQLite diverges on silicon but not under QEMU, that is
a candidate cause before the runner is.

---

# STAGE 3 RUNBOOK — the board. Everything below is BAKED and VERIFIED; only the boot remains.

## What is in the image (initramfs 9.1 MB, every artifact hash-checked inside `rootfs.cpio`)

| artifact | sha256 (16) | what it is |
|---|---|---|
| `sqbase.dom` | `f1214600d0dac351` | **the control** — the plain SQLite silicon workload, five known markers. Byte-identical to the build made from pre-SLT sources, so the control is provably unperturbed by any of this work. |
| `sqslt.dom` | `b6d1cb1da795f291` | the SLT runner at board configuration: 1 MiB region, 256 KiB arena |
| `sqsltr.dom` | `8a0edd2753011386` | identical but for `QR_DRAW=64` — **R-16 redraw insurance** |
| `sqlite_host_slt.user` | `086f136227af0efa` | host built with the MATCHING region size, under its own name so the existing 4 KiB-region SQLite arms keep working |
| `slt_neg.test` | `a3c824fb8c95d12e` | 21 records, **known non-zero failures** |
| `s1_81.test` | `fdeed2acc1d04303` | 50 queries |
| `s1_231.test` | `2bde6de55edc1435` | 200 queries |
| `s1_531.test` | `430ec636fce58083` | 500 queries |

## The invocation — ONE boot, six arms, ascending

```bash
cd capstone/tests/rtl-smoke
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"   # credential; never commit or echo
export FPGA_FW=<caplifive-system>/.../opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
H=/test-domains/sqlite_host_slt.user
SQLITE_STAGE_DOMS="\
/test-domains/sqbase.dom,\
$H|/test-domains/sqsltr.dom:--slt /test-domains/slt_neg.test,\
$H|/test-domains/sqslt.dom:--slt /test-domains/slt_neg.test,\
$H|/test-domains/sqslt.dom:--slt /test-domains/s1_81.test,\
$H|/test-domains/sqslt.dom:--slt /test-domains/s1_231.test,\
$H|/test-domains/sqslt.dom:--slt /test-domains/s1_531.test" \
SQLITE_STAGE_TIMEOUT=2400 SQLITE_IDLE_S=2400 \
python3 -m fpga_driver.run_sqlite_stages_fpga
```

No driver change was needed: the `path:selector` suffix is passed to the host verbatim, and
the `host|path` form supplies the matching host per entry.

## Expected values — the QEMU reference at THE SAME configuration

Taken at 256 KiB arena / 1 MiB region, not the 1 MiB/4 MiB the stage-2 matrix used, so a
board divergence has no "the configuration differed" escape hatch. Each also equals the
native baseline field for field.

| arm | expected `SLT-SUMMARY` |
|---|---|
| `slt_neg` | `records=21 stmt_pass=9 stmt_fail=2 query_pass=6 query_fail=4 skip_big=0 oom=0 skip_cond=2 parse_err=1 completed=1` |
| `s1_81` | `records=81 stmt_pass=31 stmt_fail=0 query_pass=50 query_fail=0 skip_big=0 oom=0 skip_cond=0 parse_err=0 completed=1` |
| `s1_231` | `records=231 stmt_pass=31 stmt_fail=0 query_pass=200 query_fail=0 skip_big=0 oom=0 skip_cond=0 parse_err=0 completed=1` |
| `s1_531` | `records=531 stmt_pass=31 stmt_fail=0 query_pass=500 query_fail=0 skip_big=0 oom=0 skip_cond=0 parse_err=0 completed=1` |

**`slt_neg` is the arm that matters most, and it is second and third on purpose.** Its two
statement failures, four query failures, two skips and one parse error are all deliberate.
An arm that comes back all-zero has not "passed" — it has failed to run the comparator, and
without this arm every clean result behind it would be unfalsifiable.

## Ordering, and why it is this order

1. **`sqbase` — the control.** A boot whose control fails is VOID and carries no verdict
   about anything; the control itself fails roughly one time in five.
2. **`sqsltr` + `slt_neg`** — the redraw variant, on the smallest case. R-16 is **per-image**,
   so the control proves the boot, not that the SLT image class enters. This arm costs
   seconds and insures the whole session against a bad draw.
3. **`sqslt` + `slt_neg`** — the real image, same tiny case, known non-zero failures.
4.–6. **`s1_81` → `s1_231` → `s1_531`**, ascending. Every arm is expected to RETURN, and the
   first that does not IS the bisection point.

## Two things that will otherwise produce a wrong verdict

**SILENCE IS EXPECTED, AND IDLE DETECTION CANNOT BE USED HERE.** The domain emits nothing
between `SQ: G/enter` and `SQ: H/return` — the report accumulates in the shared region and
the host prints it only after the domain returns. `SQLITE_IDLE_S` therefore has to be set
equal to the stage budget rather than to its 30 s default, and the ladder ordering is the
only wedge-bounding mechanism left. Measured wall clock under QEMU: 50 queries ≈ 10 s of
work, 200 ≈ 25 s, **500 ≈ 370 s**. CVA6 at 25 MHz will be slower than emulation, not faster,
so `s1_531` may run to tens of minutes and `s1_1031` was deliberately not baked — it exceeded
even the QEMU harness's 360 s budget and remains UNMEASURED, which is neither a pass nor a
failure.

**THE FRESHNESS GATE DOES NOT CHECK THE SHARED `.dom`.** With `dom:--slt <file>` specs the
gate's candidate picker takes the last half naming a staged artifact — the `.test` file — so
the one artifact every arm shares goes unverified by it. The first spec (`sqbase.dom`) is
selector-free and is checked; `sqslt.dom` is covered instead by the bake's own cpio hash
verification, recorded in the table above. Re-run the bake if in any doubt: it fails loudly.

## Reading the result

Classify per the `board-run` skill, on `SQ: G/enter`, and read no further than the first
failure. A wrong VALUE is a result and a good one — it is bisectable where a wedge is not.

## THE BITSTREAM CHANGED UNDER THIS PLAN — 2026-08-21, and two caveats above are now wrong

**The board owner flashed `caplifive_s10fix_80843404c` (name to be read off the board, not
taken from any message).** Everything below is from the RTL lane, whose lineage checks I have
not re-derived; treat the attributions as theirs.

**RETRACTED, FORWARD ONLY: "8 of 16 legs of the write-buffer residual are live."** That was
true of `caplifive_s07fix.bit` and remains true of every result taken on it — **including our
own 3/3 SQLite baseline**. It is NOT true of this image. The `core/` delta from `f231b5af0`
(what `s07fix` was built from) to `80843404c` is exactly one file, `wt_dcache_mem.sv`, byte
-identical to the tree the closure was measured on:

    S-07 only   test  9 exceptions   control 17
    with S-10   test 17 exceptions   control 17     <- 17 is the ceiling: every leg traps

Control pinned at 17 across both, plus a model-identity control, so the single variable is
S-10. **Do not carry the residual caveat into a result taken on this image.**

**S-10b is NOT in this bitstream** — dead on `DRC LUTLP-1`, a 69-LUT loop across `rev_node`,
`load_unit` and `csr_regfile`, reproduced in two builds. Do not describe the image as
containing it.

### THE CAVEAT THAT REPLACES IT, and it is worse for us than the one it replaces

**The baseline was measured on a −10.629 ns part; this is a −16.400 ns part.** S-10 did not
improve the timing — it was exonerated by attribution (its own comparator nets sit ~10 ns
clear of the critical path; essentially all failing paths launch from two single-bit
registers in `dom_switcher` and the LSU bypass, neither belonging to S-10; and the design
already failed timing at −10.629 before it). The WNS is still −16.400.

`corev_apu/fpga/scripts/run.tcl:93-99` is explicit that a timing-failing bitstream *"behaves
intermittently and data-dependently — the exact signature of the S-07 defect under
investigation, with no way to separate the two afterwards."*

**So a NEW INTERMITTENT OR DATA-DEPENDENT SQLite failure on this image is a candidate timing
artefact and not necessarily an SLT finding.** That is a fresh confound relative to the
baseline, it is not separable after the fact, and it is the single most important thing to
carry into reading this run. A *deterministic* divergence — the same wrong value on repeat —
is much better evidence than a flaky one.

**THIS BITSTREAM HAS NEVER EXECUTED ON SILICON.** The RTL lane recommended holding — the
acceptance arms (`wb0-4`, `wf1/wf5`, `wr6/wr7`) were lost when `/tmp` cleared and `wr8` has
never run, and `wr8`'s carve cost against the 1021-entry pool is uncounted. The owner flashed
anyway, which is their call, but **this boot may be its first**. Consequence for the session,
and it overrides the ordinary control rule: **if `sqbase` diverges at all from its known
pre-SLT behaviour, STOP and treat the boot as VOID.** On a first-run bitstream that is far
more likely to be the silicon than the harness.

**S-07, S-06 and S-08 all hold** — `f231b5af0` is an ancestor of `80843404c` and the three
fixes (`5c5f4e3a7`, `25035c4c0`, `9fd5507be`) are all in. The 3/3 workload baseline stays a
valid comparison point **for functionality**; treat it with care for anything timing-sensitive
or rate-based, per the paragraph above.

**Read the resident bitstream name OFF THE BOARD.** The RTL lane named only its own local
copy and cannot confirm what the owner named the flashed artifact; the driver default still
says `caplifive_s07debug_18august.bit`. Guessing between two recollections is how a launch
gets burned.

---

# STAGE 3, FIRST BOOT — VOID, and what it does and does not show (2026-08-21)

**One boot on `caplifive_s10fix_80843404c.bit` (name read off the board from
`flash_state.nv_bitstream_name`, not from anyone's recollection — two agents had two
different strings and neither matched the driver default). Four arms, plain-SQLite control
first. THE CONTROL NEVER RAN, so the boot is VOID and carries no verdict about SQLite.**

Sequence, scoped to this run's own `load_image` (the console replays ~548 KB of the previous
boot on connect, and the unscoped log shows four `SQ: A/dom-ok` and three `EXTENDED_PASSED`
that all belong to that replay — none to this run):

    power on -> SBI banner -> Linux 6.4.14 -> shell prompt -> device check DEVOK / DN_0
    -> driver dispatches TEST 1/4 /test-domains/sqbase.dom
    -> NOTHING. Then binary garbage on the console.

**The control is not a new artifact.** `sqbase.dom` is `f1214600d0dac351`, byte-identical to
the plain SQLite build from pre-SLT sources — the domain that previously passed 3/3 — run
under its matching 4 KiB-region host. The firmware is new but booted to a shell and completed
a command, so it is not simply broken.

## It is NOT the S-08 "cannot run domains" class — and there is a clean discriminator

S-08 (`fpga-repros/S08-s06fullfix-bitstream-cannot-run-domains/`) is the obvious precedent:
that bitstream booted perfectly and domains would not run, 4 of 4 boots, because `medeleg`
came back 0 and **U-mode ecalls stopped being delegated, so every syscall died**. Our symptom
looked like a superset of it: `Ok, good file` is libcapstone's first line and reaches the
console through `write(2)`, and shell echo is syscall-mediated too, so both would go silent
together under exactly that mechanism.

**The discriminator, which needs no extra tooling: the MONITOR runs in M-mode and writes the
UART directly, so its output does not depend on ecall delegation.** Under the S-08 mechanism
M-mode markers keep printing while everything user-side goes quiet.

Measured over this run's post-dispatch window, with the same detector run over the replayed
previous boot as a POSITIVE CONTROL so a zero cannot be an instrument failure:

| | monitor-side (M-mode: `DBAS:`, `ECSA:`, `SHA0-6:`, `EXCX:`, `MCAU:` …) | host-side (U-mode: `Ok, good file`, `LC:`, `SQ:`) |
|---|---|---|
| after the control was dispatched | **0** | **0** |
| replayed previous boot (control) | 164 | 70 |

**Zero M-mode output as well as zero U-mode output.** The core stopped executing altogether —
it did not even echo the typed command — which is a wedge or a reset, not a delegation
failure. **So this is not the S-08 signature**, and the S-08 fix (`9fd5507be`) is in this
lineage anyway.

## What that leaves, in order

1. **TIMING.** This is a −16.400 ns part; the baseline was −10.629. `run.tcl:93-99` says a
   timing-failing bitstream "behaves intermittently and data-dependently", which fits "boots,
   completes a shell command, then dies at the first domain dispatch" well.
2. **Infrastructure.** The control fails roughly 1 in 5 for infra reasons, and `user_count`
   was 3 — the board owner and at least one other session were on the console — so external
   interaction is not excluded. **Removing that is the cheapest next step: retry with the
   console not shared.**
3. **Something about this image specifically.** It had never executed on silicon before this
   boot; the RTL lane recommended against flashing it and its acceptance arms do not exist.

**S-10 as the cause is DOWN but not out**, on the RTL lane's source analysis and stated as
theirs: S-10's only behavioural addition is that a write-buffer granule-mate with `ctag == 0`
forces a read's tag to zero, and on the reachable path a capability context row never shares
a granule with a scalar row (rows 0-2 are 16-byte `metadata_en=1`, rows 3-7 are 8-byte
scalars that pair only with each other, and those are `ctag=0` regardless). That is layout
analysis, not a proof; it assumes a granule-aligned base and says nothing about the LSU path
or about timing.

**N=1, AND IT STAYS N=1 UNTIL RETRIED.** One void boot convicts nothing. The retry is: same
image, same arms, no changes, console not shared, plus a `create_dom`-scoped trap-register
read (`EXCX:0000E002`, `MCAU:00000008`, `MSTA` with MPP=0 are the S-08 constants to compare
against). If even monitor-side output stays absent, that is the finding.

## THE TIMING CAVEAT IS RETIRED FOR BOOT 1 ONLY — IT IS NOT RETIRED

Boot 2 was control-green and boot 1 did not reproduce, so **boot 1** is attributed to ordinary
infrastructure (the control fails ~1 in 5, and the console was shared for that boot). That is
a statement about boot 1 and nothing else.

**This image still misses setup by 5.8 ns more than the one every prior board result came
from** (−16.400 against −10.629), and `run.tcl:93-99` still says a timing-failing bitstream
behaves intermittently and data-dependently with no way to separate it from a real defect
afterwards. **That caveat stands for every future anomaly on this image**, including anything
the S-10 acceptance arms turn up. "Explained once by infrastructure" is not "explained".

Concretely, for reading any run on this bitstream: a **deterministic** divergence — the same
wrong value on repeat — is worth far more than a flaky one, and a one-off should be repeated
before it is attributed to anything at all.

**A NOTE ON THE IDLE BUDGET, because it cuts both ways.** `SQLITE_IDLE_S=1800` on boot 1 meant
the driver would have waited thirty minutes before declaring a silent domain idle and taking
its wedge read — the run was killed during that wait, which is why boot 1 produced no trap
registers. Boot 2 used 240 s and got its readings. But the SLT arms are **silent by
construction** between `SQ: G/enter` and `SQ: H/return`, so a long arm can trip a 240 s idle
and be reported as wedged when it is merely working. Measured QEMU work: 50 queries ≈ 10 s,
200 queries ≈ 25 s, 500 queries ≈ 370 s; silicon is slower. **An idle-triggered verdict on an
SLT arm is an instrument artifact until the arm is re-run with a longer budget.**

**The S-10 acceptance arms are on hold** (`tests/rtl-smoke/wbuf-arms/`, built and
distinctness-checked). Staging them on an image that may not run domains would produce a
ladder of void arms that reads like an S-10 result.

---

# STAGE 3 RESULT — SQLLogicTest RUNS ON SILICON, and one arm WEDGES (2026-08-21, boot 2)

**Control-green boot, so these verdicts count.** `caplifive_s10fix_80843404c.bit`, four arms,
one boot.

| # | arm | result |
|---|---|---|
| 1 | `sqbase.dom` — plain SQLite workload, **the control** | **PASS, returned in 6 s** — all five markers: `alpha=11`, `beta=22`, `gamma=33`, `EXTENDED_PASSED`, `MEMORY_PASSED`, `rc=0` |
| 2 | `sqslt.dom --slt slt_neg.test` — 21 records | **PASS, returned in 12 s** — summary IDENTICAL to the QEMU reference, field for field |
| 3 | `sqslt.dom --slt s1_81.test` — 50 queries | **WEDGED. No return within 1200 s.** |
| 4 | `s1_231.test` | not run — a wedge takes the core, so everything after arm 3 is collateral |

## What arm 2 establishes, and it is the larger half of this result

    BOARD:    records=21 stmt_pass=9 stmt_fail=2 query_pass=6 query_fail=4 skip_big=0 oom=0 skip_cond=2 parse_err=1 completed=1
    QEMU ref: records=21 stmt_pass=9 stmt_fail=2 query_pass=6 query_fail=4 skip_big=0 oom=0 skip_cond=2 parse_err=1 completed=1

**SQLLogicTest runs inside a capability domain on real silicon**, and the comparator is proven
to DISCRIMINATE there rather than return a clean void: all six deliberately-wrong arms fired
on hardware for the right reasons, the unparseable record was counted rather than skipped, and
**the MD5 the board computed over a 500-value result set is bit-identical to the host's**. The
whole 3,577-byte test file crossed into the domain through the shared region (`SQ: slt=3577`),
and for arm 3 the 16,774-byte file did too, with `BASE:83000000 ALEN:00100000` confirming the
1 MiB region was created and shared correctly.

## Arm 3 — a REAL wedge, and my first explanation for it was wrong

**I proposed that my own `SQLITE_IDLE_S=240` had cut a legitimately-slow arm short. The log
refutes that:** the driver reports `NO RETURN within 1200s (ActionTimeout)`, so the binding
limit was `SQLITE_STAGE_TIMEOUT=1200`, not the idle budget. Against 6 s for the control and
12 s for 21 records, twenty minutes of silence is two orders of magnitude past any slowness
explanation.

The domain **entered cleanly** — `A/dom-ok`, both regions created and shared with full monitor
output, `SQ: slt=16774`, `G/enter` — and never returned.

**Two debug-mux readings must NOT be reported as findings:**

* `TRAP LOG sw=255 = 0x89` is `seen=1, mcause=9` — **ECALL from S-mode, which is not a
  capability fault.** This is the documented 2026-08-01 near-miss where a `0x89` was nearly
  written up as an untrapped capability fault and was a stale boot ecall. Capability faults
  are cause 23+code (24–28).
* `rev-node head = 65312 (0xff20), delta = -223, "went BACKWARDS"` is a **RUNNING** read, and a
  running mux read ORs ~42 ms of execution together — anything carrying a VALUE needs a halted
  read. The halted read at the wedge gives `rev_node_head = 0x0268 = 616`, plausible against a
  1021-entry pool. Several sibling readings in the same sweep returned VOID / INSTRUMENT FAULT.

What the halted read does say: `privM=1 flush=1 ex_commit.valid=0` — the core is in M-mode with
nothing committing.

## What separates arm 2 from arm 3 — the bisection target

Arm 2 exercises the runner end to end: parsing, execution, rendering, all three sort modes,
MD5 over 500 values, and failure reporting. It passes in 12 s. So **the mechanism is not what
wedges.** Arm 3 differs by running select1.test's actual queries — aggregates, `CASE`,
correlated subqueries over a 30-row table — 50 of them.

**Next: finer slices** (`s1_31` = the 31 setup statements and no queries, then 5 / 10 / 25
queries), ascending in one control-led boot. The first that fails to return IS the bisection
point. Board-free to build and native-verify first, as with every slice so far.

## Caveats that travel with this result

* **N=1 for arm 3.** It has not been reproduced.
* **This part misses setup by 5.8 ns more than every part our prior results came from**
  (−16.400 vs −10.629), and `run.tcl:93-99` says such a bitstream behaves intermittently and
  data-dependently with no way to separate that from a real defect afterwards. **A wedge that
  does not reproduce is a timing candidate; one that reproduces deterministically at the same
  record is a real finding.** Reproducing it is therefore the first thing the bisect buys.
* The image had never executed on silicon before today, and the RTL lane advised against
  flashing it.

---

# BISECT BOOT 3 — the setup is exonerated, the wedge REPRODUCES, and it is deterministic

Control-green boot (all five markers, 17 s), so these verdicts count.

| slot | arm | queries | result |
|---|---|---|---|
| 1 | `sqbase` — control | — | **PASS**, 17 s, five markers |
| 2 | `s1_31` | **0** | **PASS**, 8 s — `records=31 stmt_pass=31 …` **identical to native** |
| 3 | `s1_56` | **25** | **WEDGE** — no return within 400 s |
| 4 | `s1_81` | 50 | not a result: collateral after a wedge |

**Three things established:**

1. **The table setup is NOT the trigger.** `CREATE TABLE` plus 30 `INSERT`s execute correctly
   in a capability domain on silicon and return a summary matching native field for field.
2. **The trigger is inside the first 25 queries of `select1.test`.**
3. **IT REPRODUCES.** Boot 2 wedged on 50 queries; boot 3 wedges on 25 with 0 queries clean in
   between. Two different inputs, same failure — much harder to explain as a timing flake than a
   single hang would be.

**And it is not "queries" in general.** The negative control passed on silicon in 12 s with ten
queries, including a 500-value result set hashed to MD5, all three sort modes and six failure
paths. The runner executes queries on this hardware. Something specific to `select1`'s query set
does this — aggregates, `CASE`, correlated subqueries over a 30-row table.

## The wedge state is BYTE-IDENTICAL across both boots

    TRAP LOG sw=255      0x89        seen=1, mcause=9
    sw=224               0x7f        privM=1 flush=1 ex_commit.valid=0
    sw=225               0xd5
    rev_node_head        0x0268      = 616, against a 1021-entry pool
    commit pc                        0x0000000082d7fffc
    trap mepc (LATCHED)              0xffffffff800072cc

Identical across two different workloads is consistent with both wedging at the SAME query
(they share their first 25 queries, so allocation history to that point is identical), i.e. a
deterministic failure rather than a flake.

**`trap mepc = 0xffffffff800072cc` is a LINUX KERNEL virtual address**, and with `mcause=9`
(ECALL from S-mode) it is a routine Linux→OpenSBI ecall. So the trap latch is **stale and not a
capability fault** — the documented 2026-08-01 near-miss, avoided. It doubles as a **positive
control for the mux readout**: `0xffffffff…` is exactly the right shape for a kernel VA, so the
readout is returning real data rather than error-slave junk.

## The commit PC is NOT yet trustworthy, and the next boot tests it for free

`commit pc = 0x82d7fffc` minus arm 3's base `0x82C00000` is offset **`0x17FFFC`** — **past the
1,444,104-byte loadable image**, inside the domain's data region. That is either control-flow
corruption or a readout artifact, and **`0x82d7fffc` sitting exactly 4 bytes below a 512 KB
boundary is a reason to suspect the instrument**.

Both wedges so far were at **slot 3**, hence the same base, so a constant reading and a real one
are indistinguishable. The next bisect boot puts the wedging arm at whatever slot fails first,
and the slots have different bases:

    slot 2 -> 0x82800000   a real, base-relative PC would read 0x8297fffc
    slot 3 -> 0x82C00000                                       0x82d7fffc
    slot 4 -> 0x83000000                                       0x8317fffc

**If the PC tracks the base it is real and `0x17FFFC` is a genuine location; if it stays
`0x82d7fffc` regardless, the value is a constant artifact and nothing may be inferred from it.**
Free, because that boot is being run for the bisection anyway.

## Next: control + `s1_36` (5 q) + `s1_43` (12 q) + `s1_50` (19 q), ascending

All three verified clean natively before use. First to fail is the bisection point; all clean
narrows the trigger to queries 20–25.

---

# BISECT BOOT 4 — down to FIVE queries, and the commit PC is now TRUSTED

Control-green (5/5 markers, 6 s), so the verdicts count. Fourth consecutive clean plain-SQLite
run on this bitstream.

| slot | arm | queries | base | result |
|---|---|---|---|---|
| 1 | `sqbase` control | — | `0x82400000` | **PASS**, 6 s |
| 2 | `s1_36` | **5** | `0x82800000` | **WEDGE**, no return in 400 s |
| 3–4 | `s1_43`, `s1_50` | 12, 19 | — | not results: collateral |

**The trigger is inside the FIRST FIVE queries of `select1.test`**, all five of which pass
natively under the identical configuration (`query_pass=5`), so this is silicon-specific and
not a configuration artifact.

## THE COMMIT-PC READING IS REAL — the control fired

Boots 2 and 3 wedged at slot 3 (base `0x82C00000`) and both reported `commit pc = 0x82d7fffc`.
A constant artifact and a real base-relative address are indistinguishable from that alone. This
boot wedged at **slot 2**, base `0x82800000`, and reported:

    commit pc = 0x000000008297fffc        predicted if real: 0x82800000 + 0x17FFFC = 0x8297FFFC

**It tracks the base exactly.** So the reading is genuine, and **three wedges across two
different bases all stop at the same domain offset `0x17FFFC`** — a deterministic failure at a
fixed location.

## Where `0x17FFFC` is, and it is not code

From the domain's own build report (`sqslt.dom`, 256 KiB arena):

    code            0x000000 .. 0x160910
    blob            0x160910 .. 0x171220
    cap table       0x171220 .. 0x171de0
    global storage  0x171de0 .. 0x1c2070     <== 0x17fffc lands here, 57,884 bytes in
    stack           0x1c2070 .. 0x400000

`.text` ends at VA `0x153bc8`. **The last committed instruction is ~57 KB inside the domain's
global-variable storage, far past any executable content in the image** — the signature of
control flow leaving the code and entering data. Under `-capstone-gp-captable` that storage is
exactly where every carved global lives.

**Two things stop this being a root cause, and both must be resolved before it is called one:**

1. ~~**`privM=1` is not reconciled.**~~ **WITHDRAWN — this was my misconception, corrected by
   the RTL lane and verified here against the RTL rather than taken on trust. CAPABILITY
   DOMAINS RUN AT M PRIVILEGE WITH `capmode` SET**; `CAPENTER` sets capmode and does NOT drop
   privilege (`core/csr_regfile.sv:295` `capmode_d = capmode_q | capmode_set_i; // set by
   CAPENTER, sticky`, driven from `capenter_commit` at `core/cva6.sv:2053`;
   `core/commit_stage.sv:208` gates on `priv_lvl_i == PRIV_LVL_M && capmode_i`;
   `core/cva6.sv:85` "valid when capmode && priv_lvl==M"). Containment comes from the PC
   capability and the CPMP entries, NOT from the privilege level. So `privM=1` with a PC inside
   the domain's own allocation is the normal expected state — and it is **not** evidence that
   execution ended up in the monitor, which STRENGTHENS the reading that the PC is genuinely in
   the domain's data. Anyone assuming domains are U-mode will chase a non-existent anomaly.
2. **The `DBAS` ↔ allocation-offset-0 mapping is the natural reading, not a verified one.**
   The base-relative control supports the scheme as a whole but does not pin the origin.

## Cumulative silicon status

**Passing on hardware, matching native field for field:** the negative control (21 records, ten
queries, a 500-value MD5, all three sort modes, six deliberate failures) and `s1_31` (31 setup
records). Plus four clean plain-SQLite control runs.

**Wedging on hardware, deterministically, at a fixed address:** any slice containing the first
five `select1.test` queries — reproduced at 5, 25 and 50 queries.

## Next, and it is one boot

`control + s1_32 (1 q) + s1_33 (2 q) + s1_34 (3 q)`, ascending. Either it names a single query
outright, or all three pass and the trigger is query 4 or 5. Slices to be built and
native-verified first, as every slice so far has been.

**The timing caveat still stands** and is not retired by any of this: −16.400 ns against a
baseline part's −10.629. What weighs against timing here is that the failure is *deterministic
at a fixed address across three runs and two bases*, which is not the profile `run.tcl` warns
about ("intermittently and data-dependently").


## The `0x180000` lead — SUGGESTIVE, and the tidy version of it does NOT survive

The wedge PC sits exactly four bytes below `0x180000`, which is `code_size` (1,444,112)
rounded up to a 512 KiB granule. The RTL lane's reading: a last-committed instruction one word
short of a boundary, at the same offset across two bases, is the shape of execution running
**forward through** data and halting at the edge of a mapping — rather than a single jump to a
computed address.

**Attractive, and the obvious mechanism for it fails on inspection.** The monitor asks for a
code bound far below that: `sbi_capstone.c:298` rounds `code_size` to a **16-byte** granule and
`dom_code` is split at `base + code_size` = `0x160910`. For the hardware bound to sit at
`0x180000` instead, capability-bounds compression would have to round `0x160910` up by 128 KiB,
which requires a **4-bit** bounds mantissa:

    mantissa  4 bits -> granule 0x20000 -> rounds up to 0x180000   <-- only this reaches it
    mantissa  8 bits -> granule 0x2000  -> rounds up to 0x162000
    mantissa 14 bits -> granule 0x80    -> rounds up to 0x160980

Four bits is implausibly small for a 128-bit format (CHERI-class formats use ~14). **So the
"halted at the PC capability's rounded bound" story is NOT adopted.** It is recorded because
the coincidence is exact and worth one cheap check, not because the evidence supports it.

### The check was run: THE LEAD IS DEAD

Answered from the RTL, which is what the board runs, rather than from QEMU's `cap_compress.c`.
`core/include/ariane_pkg.sv:674` in `decompress_bounds` declares `logic[13:0] B, T;` with
`B[13:3] = bounds_full.b` — **a 14-bit mantissa.** The granule is `2^E`, and E is set by the
region length: `0x160910 / 2^E < 2^14` needs `E >= 7`, so the granule is **`0x80`** and the
bound rounds to `0x160980`. Reaching `0x180000` would require the 4-bit mantissa this format
does not have and no 128-bit capability format would.

**So bounds compression cannot produce `0x180000` from `0x160910`, and the arithmetic
coincidence is just a coincidence.** `0x17FFFC` is therefore **UNEXPLAINED** — which is the
honest state. What survives is that the offset is REAL: it reproduces across two different
domain bases, so it is a genuine fixed location and simply not the code bound.
**The discriminator that remains** — walked forward through data, or a single control transfer
into `0x17FFFC` — needs RVFI, which is a Verilator instrument. Getting SQLLogicTest running in a
domain under simulation crosses exactly the fidelity gap the `rtl-sim` skill warns about (bare
M-mode with no monitor, a `.data` buffer instead of a monitor-carved stack, different cache
warmth), so a clean simulation would not exonerate the silicon and the wedge might not reproduce
there at all.

**Cheaper, and it is the next step: name the single offending query on the board.** If one query
wedges and its neighbours do not, the difference between them is a far smaller search space than
an RVFI trace — and it comes from hardware that already works. Simulation is the fallback if the
query does not localise it.


---

# DEBUGGING THE SILICON WEDGE — instruments calibrated, 2026-08-22

**Framing correction first: SQLLogicTest DOES NOT WORK ON SILICON.** The harness runs there (the
21-record negative control matches native field for field) but the real corpus wedges inside
`select1.test`'s first five queries. Earlier text in this document led with the half that worked;
stage 3 is open.

## 1. Prefix ladders cannot answer the question — isolated queries can

Every slice run so far is a PREFIX (setup + queries 1..N). That bounds where execution stops. It
**cannot** distinguish:

* query 5 is individually pathological, from
* queries 1-5 **cumulatively** exhaust something — a leak, an allocator watermark, a rev-node count

Different causes, and the monotone 0/5/25/50 result is void for that question while looking
conclusive. Same shape as the RTL lane's `wr6`/`wr7` note: two arms agreeing means the variable
you intended was not the variable you varied.

`q1_only … q5_only` are setup + exactly ONE query, all five verified natively (`query_pass=1`).
A wedge on one names it; all five clean while the 5-query prefix wedges moves the search to what
accumulates.

## 2. The VDBE clamp, armed PER QUERY — and calibrated

A wedged core takes the host with it, so nothing in the shared region survives: only a build that
RETURNS can report. `CAPSTONE_VDBE_CLAMP=n` stops `sqlite3VdbeExec` after n opcodes, returns
`SQLITE_DONE`, and records the opcode that was about to run.

**Arming it once at entry measures the WRONG STATEMENT, and that mistake was made and caught
here.** SQLite's opcode counter is cumulative across every `sqlite3VdbeExec`, so a clamp of 20
armed at domain entry stopped `CREATE TABLE` and reported `no such table: t1` for every
statement — exactly what `sqlite_capstone_domain.c` already warns about. Found by a QEMU positive
control, not by a board slot. Now armed and reset per query through `SLT_VDBE_ARM/DISARM`, no-ops
by default so the runner stays capability-agnostic and the native baseline is untouched.

**BOTH OUTCOMES DEMONSTRATED, which is what makes a clean result readable:**

| clamp | result |
|---|---|
| 20 | **fires** — 31 setup statements pass, query truncated, `SLT-VDBE ops=20 lastop=40` (`OP_Next`) |
| 1000000 | **does not fire** — `query_pass=1 completed=1`, `SLT-VDBE ops=574 lastop=0` |

**Query 1 of `select1.test` executes 574 VDBE opcodes.** That sizes the ladder instead of guessing
it.

**The default SLT image stays byte-identical** (`b6d1cb1da795f291`, the image whose wedge has
reproduced three times) because the arming is gated on `CAPSTONE_VDBE_CLAMP`. An instrument that
moves the failure it is measuring is worse than no instrument.

## Plan, two boots, control-led

    Boot A   control, q1_only, q2_only, q3_only      -> names the query, or shows it is cumulative
    Boot B   control, q4_only, q5_only, + first clamp arm

Then a clamp ladder on whichever query wedges — ascending over its opcode count (574 for query 1),
so the first n that fails to return brackets the offending opcode. Every arm before it RETURNS, so
the bisection converges instead of guessing.

## Housekeeping

A discarded QEMU run showed `EXT4-fs error (device vda): doubly allocated?`, the shared
`rootfs.ext2` signature. It produced no result and is discarded rather than interpreted; the same
run repeated cleanly with **zero** EXT4 errors, so the image is sound. Earlier QEMU results stand:
they were self-consistent and positive-controlled.


---

# TWO HYPOTHESES KILLED FOR THE SILICON WEDGE (2026-08-22) — both with proven instruments

## 1. HOSTILE UNINITIALISED MEMORY — REFUTED

`CAPSTONE_POISON_ARENA=1` fills the memsys5 arena with `0xA5` before `SQLITE_CONFIG_HEAP` and
carries its own witness gate. Run on the 5-query case under QEMU: **`query_pass=5 completed=1`**,
clean. So the wedge is not explained by silicon's fresh memory being hostile where emulation's is
zero — the standard first suspect for a silicon-only effect here, and documented as an emulation
blind spot at `build-sqlite-silicon.sh:1660`. **A ruled-out class is a result.**

## 2. REVOCATION-NODE POOL EXHAUSTION — REFUTED, after TWO broken instruments

The hypothesis was strong and documented (`build-sqlite-silicon.sh:851-860`): the RTL allocator's
head is 10 bits from 3, so allocation ~#1022 wraps to id 0 and reuses LIVE ids; REVOKE_NODE has no
visit bound or cycle detection and walks a spliced chain forever; and every `stc` blocks on a
revocation query **with no timeout**, so the next capability store **hangs with no trap**. That is
our exact signature — hang, no trap, `ex_commit.valid=0`. It also explained the silicon-only part:
QEMU's pool is `CAP_REV_TREE_SIZE 10000` **with reuse**, so emulation structurally cannot reach
the wrap.

**Measured: `s1_36` (5 queries) reaches cumulative allocation 128 and not 256 — 128-255 total,
against a pool of ~1021.** Not close. And that total is dominated by the domain's 188 startup
carves; the five queries themselves add very little. **Exhaustion is not the mechanism.**

### Three instrument failures on the way, all of the same family

1. **Counted `alloced_n`** — that is QEMU's PEAK CONCURRENT usage, because QEMU reclaims via a
   free list. Silicon's bump allocator never reclaims, so it spends CUMULATIVE allocations. The
   check fired correctly and could not have detected the thing it was built to test, at any
   magnitude. It returned a clean "refutation" that meant nothing.
2. **Fired only at 1022** — so "allocated 900" and "the counter never incremented" produced
   identical silence. Replaced with a report at every power of two, which is **self-proving**:
   any run that allocates at all MUST print 1, 2, 4..., so their absence indicts the instrument
   rather than exonerating the workload.
3. **Read the wrong log.** QEMU's stderr goes to `$OUT_DIR/sqlite-slt.log`, not the wrapper's
   stdout — and I grepped the wrapper's. The milestones were there the whole time.

**Failure 3 was caught only because of the fix for failure 2.** A single-threshold watermark would
have shown nothing and I would have recorded "does not cross 1022" as a finding. The self-proving
ladder made the absence of milestone 1 impossible to read as a result about the subject.

## Where that leaves the wedge

Still unexplained, and now with two classes eliminated rather than one guess replaced by another.
The board experiment that has NOT yet run is the one that matters most:

**`q1_only` … `q5_only` — setup plus exactly ONE query.** Every board slice so far has been a
PREFIX, which cannot separate "one query is pathological" from "the queries cumulatively exhaust
something". Both refutations above weaken the cumulative reading — neither poison nor rev-node
pressure accumulates fast enough — but they do not settle it, and the isolated arms do.


---

# THE WEDGE ADDRESS IS LOCALIZED: it is INSIDE `sqlite_heap` (2026-08-22)

`0x17FFFC` is no longer unexplained. Decoding the domain's own `.capstone_gp_table` descriptor
(188 records of `{size, align, init_off}` in cap-table index order, per
`gen-gp-captable-glue.py`) and walking the storage allocation:

    storage base            0x171de0   (code_size + blob + cap table)
    wedge PC offset         0x17fffc   = 57,884 bytes into storage
    -> cap-table SLOT 176, a 262,144-byte ZERO-INIT global, 14,508 bytes in

**262,144 is `SQLITE_HEAP_SIZE`, and slot 176 is the only global of that size in the image.**
That is `sqlite_heap`, the memsys5 arena.

**Corroboration, not assertion:** the same layout model accounts for 327,599 storage bytes
against the build report's own `storage 328336` — 0.2% apart, which is alignment rounding. If the
model were wrong about ordering or base, it would not land that close.

## What this means

    dom_code bound (sbi_capstone.c:304, split at base + code_size)   0x160910
    wedge PC                                                        0x17fffc
    -> 0x1F6EC = 128,748 bytes BEYOND the code capability's bound

**The core committed an instruction ~128 KB outside its code capability, inside the SQLite
heap.** Control flow left the code region and entered the memsys5 arena; the resulting
out-of-bounds instruction fetch then **hung rather than trapping**, which is the wedge.

## It joins up with the select5 finding

select5 (a different file, under QEMU) faulted at a `cincoffset` whose operand was an **all-zero
untagged word loaded by `ldc a0, 0x10(a0)`** — a pointer field inside a heap structure reading as
NULL, inside `sqlite3VdbeExec`. This one is a control transfer to an address inside the heap.

**Both are pointer-shaped values living in the memsys5 arena coming back wrong.** That is a single
theme, and it points at capability round-trips through `sqlite_heap` rather than at SQLite logic.

## The minimal repro this implies — and it does not need SQLite

If capabilities stored into `sqlite_heap` do not survive a store/reload on silicon, then the
minimal reproducer is **a domain that stores a capability into a 256 KiB carved global, reloads
it, and compares** — no SQLite, no SQLLogicTest, no VDBE. That is orders of magnitude smaller
than the current 5-query case and is the right next artifact.

**Why the existing ladder rungs do not already cover it:** they round-trip capabilities through
small stack and `.data` buffers. `sqlite_heap` is a 256 KiB **carved global** under
`-capstone-gp-captable`, reached through the cap table, and no rung exercises that shape at that
size.

**NOT established:** what diverts control flow there. A corrupted function pointer in a heap
structure is the obvious candidate (SQLite keeps them in `FuncDef`, VDBE and cursor structures,
and the failing queries are exactly the ones using aggregates and subqueries), but that is a
hypothesis with a mechanism, not a root cause. The stack is at `0x1c2070+`, so this is not a
smashed return address landing in the stack.


---

# MINIMAL REPRO BUILT AND CALIBRATED — `CAPSTONE_HEAPCAP_PROBE` (2026-08-22)

**No SQLite, no VDBE, no SQLLogicTest.** Store a capability into `sqlite_heap`, load it back, ask
whether it is still a capability. Nine offsets spanning the arena, chosen around the wedge site
(14,508) which the descriptor decode placed inside that global.

**It always RETURNS.** Tag loss is detected with the LCC TOTAL type query, which answers 7 for
NOT_CAP instead of raising — so a lost tag is reported rather than wedging the core. A probe that
hangs yields one bit; a probe that returns yields a count.

**It carries its own positive control.** One slot is deliberately written with a plain integer and
must be flagged. The marker sentinel distinguishes the two cases, so a broken instrument can never
read as a passing one:

    0x4EA0_TTBB   control fired -> the counts are readable
    0x4EB0_TTBB   CONTROL DID NOT FIRE -> the run is VOID, whatever BB says

**QEMU result: `0x4EA00A01` — sentinel 4EA0, tested 10, bad 1.** The one failure is the
deliberate control; all nine real offsets round-trip correctly under emulation. So the instrument
works and emulation shows no arena problem — exactly the setup needed for the silicon arm to mean
something.

## Why this global and not a synthetic buffer

The wedge PC decodes to cap-table slot 176, uniquely the 262,144-byte `sqlite_heap`. Under
`-capstone-gp-captable` that is a CARVED GLOBAL reached through the cap table, and no existing
ladder rung round-trips capabilities through that shape at that size — the rungs use small stack
and `.data` buffers. A synthetic buffer would test a different thing.

## OPEN ANOMALY, recorded rather than explained: the default build MOVED

After these edits, a default SLT build (`CAPSTONE_HEAPCAP_PROBE` **not** defined — verified, zero
occurrences in the compiler invocation) hashes `577f9f9a0aa64f3c` where before it was
`b6d1cb1da795f291`. Same total size, but **4 more instructions** and a 16-byte data-layout shift
that moves ~20,000 disassembly lines' offsets.

**I have not explained it, and it matters**, because `b6d1cb1da795f291` is the image the wedge
reproduced on three times, and image perturbation is a documented hazard on this platform (S01).

**Consequence for the next board session, and it is manageable:** the wedge must be RE-ESTABLISHED
on whatever image is built, not assumed to carry over. The boot therefore needs a known-wedging
arm (`s1_36`) alongside the isolated single-query arms, so that a clean `q1_only`..`q3_only` result
cannot be confused with "the wedge went away when the image moved".

**Also noted:** the staged `overlay/test-domains/sqslt.dom` was replaced at 13:05 on 2026-08-22
(`5d2a56e85a316deb`) by something other than my bake, whose last run was the previous evening.
Not a problem in itself — the next bake rebuilds it — but board artifacts are shared state and
worth checking rather than assuming.


---

# BOARD SESSION 5 — the minimal repro RAN, and its simplest form is CLEAN (2026-08-22)

Control-green (5/5, 6 s). **`sqheap` — the minimal repro, no SQLite in the frame — returned
`0x4EA00A01`:** sentinel `4EA0` (its own positive control fired), 10 slots tested, **1 bad, and
that one is the deliberately-written plain integer.** The monitor echoed the value back as
`ENT2:4EA00A01`, so the domain demonstrably ran.

**So capabilities DO survive a store-and-immediate-reload through `sqlite_heap` on silicon**, at
nine offsets spanning the arena including the wedge site. The hypothesis is refuted in its
simplest form.

**That is exactly why pass A is not enough, and pass B exists.** An immediate reload is the
easiest case the hardware can be given: the store may still sit in the write buffer and be
forwarded straight back, so the capability need never reach DRAM. A clean pass A shows a
capability survives FORWARDING, not MEMORY. The wedge involves heap data written, displaced by
other work, and read back later — which is pass B's shape (walk the whole 256 KiB arena at
cache-line stride, then reload). Probe v2 reports the two separately, because "survives
forwarding but not DRAM" is the interesting answer and folding them would hide it.

## Two instrument defects, and arms 3-4 lost to one of them

**1. A HARD STOP that was wrong, and it cost two arms of a control-validated boot.** The probe
returns its own marker instead of `SQLITE_HC_SLT_RAN`, so the host exited 1; the driver's "no
`RESULT retval=` marker plus a non-zero exit" heuristic then concluded *"the domain almost
certainly was not staged"* and abandoned the rest of the session. It WAS staged and it DID run.
Fixed in `sqlite_host.c`: a `0x4EA0`/`0x4EB0` marker is now reported as `SQ: probe=` and exits
cleanly, so probe arms are first-class rather than tripping a staging heuristic.

**2. MY UART REASSEMBLY NEARLY VOIDED A VALID BOOT.** The control read 4 of 5 markers — which by
the rule means VOID and stop. The missing one was `row name=alpha value=11`, and the cause was
that I concatenate console chunks as Python `repr` strings **without stripping their quotes**, so
every chunk boundary injects `''` into the text:

    row name''=alpha value=11

The board was fine; the extraction was not. Now fixed in one shared helper (`/tmp/capstone/uart.py`
pattern, folded into the analysis path) rather than re-derived per script.

**The pattern across this investigation, worth more than any single fix:** every instrument defect
so far — `alloced_n` vs cumulative, the single-threshold watermark, the wrong log file, and now
the quote-injecting reassembly — is the same family. **The check could not distinguish its own
failure from the subject's.** Two of them produced clean-looking results that would have been
published as findings.


---

# THE MINIMAL SQL REPRO IS **ONE QUERY** (2026-08-22, board session 6)

Control-green boot (5/5 with the corrected UART reassembly; the old buggy one read 4/5 and would
have thrown this boot away).

| arm | result |
|---|---|
| `sqbase` control | PASS, 6 s, five markers |
| `sqheap` — the minimal repro, probe v2 | **`0x4EA0000A` — passA 0, passB 0, tested 10, control fired** |
| `q1_only` — setup + **exactly ONE query** | **WEDGE**, no return in 400 s |
| `s1_36` | collateral — not needed, arm 3 already re-established the wedge |

## 1. THE FAILURE IS NOT CUMULATIVE

`q1_only` is the 31 setup statements plus **one** query. It wedges. So there is no accumulation,
no exhaustion, no watermark — which is what the prefix ladder could never have told us and is why
the isolated arms were built. **The minimal SQL reproducer is:**

    SELECT CASE WHEN c > (SELECT avg(c) FROM t1) THEN a*2 ELSE b*10 END
      FROM t1
     ORDER BY 1

against a 30-row table, in a capability domain, on silicon. Down from 10,807 records to one query.

**It also re-establishes the wedge on the new image** (`577f9f9a`), so the unexplained hash change
did not matter — arm 4 was unnecessary.

## 2. THE ARENA IS NOT CORRUPTING CAPABILITIES — in the strong form either

Probe v2 on silicon: **pass A 0 failures, pass B 0 failures**, ten slots, control fired. Pass B
reloads *after* walking the entire 256 KiB arena at cache-line stride, so the stored granules are
displaced and must come back from DRAM rather than from store-to-load forwarding.

**So capabilities survive both forwarding AND a DRAM round trip through `sqlite_heap`.** The
"capabilities stored in the arena get corrupted" hypothesis is refuted in its strong form, not
just its weak one. Whatever diverts control flow into the heap, it is not the arena failing to
hold a capability.

## 3. THE WEDGE OFFSET IS INVARIANT

    boot 2   s1_81  (50 queries)   base 0x82C00000   commit pc 0x82d7fffc
    boot 3   s1_56  (25 queries)   base 0x82C00000   commit pc 0x82d7fffc
    boot 4   s1_36  ( 5 queries)   base 0x82800000   commit pc 0x8297fffc
    boot 6   q1_only ( 1 query)    base 0x83000000   commit pc 0x8317fffc

**Four wedges, three bases, two images, workloads from 1 to 50 queries — always offset
`0x17FFFC`.** A data-dependent corruption would move with the workload. An invariant offset says
the same code path computes the same target every time.

**Three mechanisms for why `0x180000` might be a boundary have now been tested and all are dead:**
capability-bounds rounding (the RTL mantissa is 14 bits → `0x80` granule, so `0x160910` rounds to
`0x160980`); the code capability's end (`base + code_size` = `0x160910`); and the `dom_seal`
region end (`DOMAIN_DATA_N = 96`, so `DOMAIN_DATA_SIZE` is 1536 bytes). The offset stays
**recorded and unexplained** rather than fitted to a fourth story.

## Next: name the CONSTRUCT, not the address

`p3_avgonly` (`SELECT avg(c) FROM t1`), `p2_nosubq` (the same CASE with a constant instead of the
subquery), `p1_scalarsub` (the full query), all verified to RETURN natively. Whichever wedges
names the construct, which narrows the code path — more actionable than the address.


---

# RETRACTED SAME DAY: "the scalar subquery is the trigger" (board session 7)

Control-green boot. `p3_avgonly` returned in 8 s, `p2_nosubq` returned in 8 s, and
`p1_scalarsub` did not return — which looked like a clean matched pair naming the scalar
subquery as the trigger.

**It is not, because `p1_scalarsub` NEVER RAN ITS QUERY.** Its UART ends at:

    SQ: A/dom-ok   SQ: B/mkregion1   SQ: C/mkregion2
    SPLB:0000E010 RGID:0000000A RGNN:00000016 BASE:83300000 ALEN:00100000
    <garbage>

No `SQ: D/mapped`, no `SQ: G/enter`. It wedged **inside the monitor**, during the second
`create_region`, before executing any SQL. The commit pc confirms it: `0x80020fbe`, which the
firmware symbol table places inside **`_split_out_cap`** — whose region path contains two silent
`while(1)`s (`sbi_capstone.c:236` and `:246`).

**So the construct question is UNRESOLVED.** `p2` and `p3` returning still stands; the failing
half of the pair does not exist.

## What DOES survive, verified rather than assumed

Every earlier wedge reached `SQ: G/enter`, i.e. entered the domain and wedged inside the workload:

    boot 3  s1_56    A/dom-ok D/mapped slt= G/enter
    boot 4  s1_36    A/dom-ok D/mapped slt= G/enter
    boot 5  sqheap   A/dom-ok D/mapped slt= G/enter H/return   (returned, 0x4EA0000A)
    boot 6  q1_only  A/dom-ok D/mapped slt= G/enter

**So "one query wedges inside the domain" HOLDS** — `q1_only` entered and then wedged. Only
session 7's arm 4 failed before entry, and it is the only arm that did.

## AND A PRACTICAL RULE THIS BUYS: SLOT 4 IS NOT SAFE FOR THESE DOMAINS

`preflight-board-run.sh` allows four domains, on the basis that the monitor's middle exact-fit
case spins at roughly the fifth `create_dom`. **But each SLT arm performs one `create_dom` AND
TWO `create_region` calls**, so the monitor's region path is exercised three times per arm and
exhausts earlier than the domain count implies. Session 7 lost its fourth arm to exactly that.

**Consequence: for region-creating domains, treat slot 4 as unreliable and put the arm you most
need to read no later than slot 3.** The invariant-offset observation recorded earlier also needs
this qualification: it holds for the four wedges that entered the domain, and session 7's arm 4 is
NOT one of them — its commit pc is in the monitor, not at `base + 0x17FFFC`.

## Re-run, with `p1_scalarsub` in slot 2

    control, p1_scalarsub, p4_subq_where, p5_exists

Same question, asked where the answer can be read.


---

# THE CONSTRUCT IS NAMED: a SCALAR SUBQUERY in an expression (board session 8)

Control-green boot, and this time the failing arm ran in **slot 2**, where it can be read.

| arm | slot | reached | result |
|---|---|---|---|
| `sqbase` control | 1 | `A/dom-ok D/mapped G/enter H/return` | PASS, 6 s, 5/5 markers |
| **`p1_scalarsub`** | **2** | `A/dom-ok D/mapped G/enter` | **ENTERED, then WEDGED** |

## The matched pair, with the failing half verified to have executed

    p3_avgonly   SELECT avg(c) FROM t1                      RETURNS (8 s)
    p2_nosubq    CASE WHEN c>100      THEN a*2 ELSE b*10    RETURNS (8 s)
    p1_scalarsub CASE WHEN c>(SELECT avg(c) FROM t1) ...    ENTERS, WEDGES

`p2` differs from `p1` by **exactly one thing**: a constant where `p1` has the subquery. `p3` is
the subquery's own content in isolation. **So the trigger is a scalar subquery in an expression,
and both of its components are individually exonerated.** This is the pair the retracted session
lacked, where the "failing" arm had never executed its SQL.

## The offset is invariant, and now across DIFFERENT DATA

    boot 3  s1_56       25 queries  base 0x82C00000  pc 0x82d7fffc
    boot 4  s1_36        5 queries  base 0x82800000  pc 0x8297fffc
    boot 6  q1_only       1 query   base 0x83000000  pc 0x8317fffc
    boot 8  p1_scalarsub  1 query   base 0x82800000  pc 0x8297fffc

**Five in-domain wedges, four bases, two images, always offset `0x17FFFC`** — and `p1_scalarsub`
runs on synthetic 30-row data of my own, NOT select1's contents. **Different data, same offset.**

## What `p1` adds over `p2`, from the VDBE (39 vs 26 opcodes)

    BeginSubrtn  Return  Once        <- a VDBE-internal control transfer
    AggStep  AggFinal  Copy  Null  DecrJumpZero
    OpenRead 1->2  Rewind 1->2  Next 1->2   <- a SECOND concurrent cursor

`p3_avgonly` has the aggregate and a cursor and returns, so the aggregate is out. **Two candidates
remain and `p1` confounds them:** the VDBE subroutine mechanism, or two concurrent cursors.

## Next boot separates them — probes built and native-verified

    p8_selfjoin    SELECT t1.a FROM t1, t1 AS y WHERE ...  OpenRead=2  BeginSubrtn=0
    p9_subq_other  ... (SELECT max(x) FROM t2) ...         OpenRead=2  BeginSubrtn=1

* `p8` wedges -> **concurrent cursors**
* `p8` returns, `p9` wedges -> **the VDBE subroutine**, an internal control transfer matching a
  control-flow diversion
* both return -> something specific to `p1` neither factor explains

Order: control, `p8` (2), `p9` (3), `p1` (4 — known-wedging, so the slot-4 monitor hazard costs
nothing).


---

# BREAKTHROUGH: TWO CONCURRENT CURSORS, and a REAL CAPABILITY FAULT (board session 9)

Control-green boot. **`p8_selfjoin` — a plain self-join, no subquery, no aggregate, no `CASE` —
ENTERED and WEDGED.**

    SELECT t1.a FROM t1, t1 AS y WHERE y.b < t1.b ORDER BY 1     OpenRead=2  BeginSubrtn=0

## 1. The VDBE subroutine is NOT required

    p2_nosubq     1 cursor,  0 subroutines   RETURNS
    p3_avgonly    1 cursor,  0 subroutines   RETURNS
    p8_selfjoin   2 cursors, 0 subroutines   WEDGES     <-- no subquery at all
    p1_scalarsub  2 cursors, 1 subroutine    WEDGES

**The common factor in every wedging case is TWO CONCURRENT CURSORS OVER THE SAME TABLE.** The
scalar subquery named earlier is a *sufficient* trigger, not the cause — it happens to open a
second cursor. `p8` reaches the same failure with none of the SQL machinery.

## 2. THE SIGNATURES DIFFER, and `p8`'s is a genuine capability fault

|  | `p1_scalarsub` | `p8_selfjoin` |
|---|---|---|
| TRAP LOG | `0x89` -> mcause **9** (stale S-mode ecall) | `0x99` -> mcause **25** |
| `ex_commit.valid` | **0** | **1** |
| sw=225 | `0xd5` tbe,wstore,wrev,stall,memwait | `0x80` tbe only |
| commit pc | base + `0x17FFFC` | `0x2` |

**`mcause 25` = `UNEXPECTED_OPERAND`** (`core/anvil_build/capstone_unit.anvilh:303`), which is
what `CINCOFFSET` raises when `cap_rs1` is `NOT_CAP`. That is **the same condition QEMU asserted
on for select5** — `cincoffset` with an untagged `rs1`, `val=0x0`, inside `sqlite3VdbeExec`.

**So emulation and silicon now show the SAME fault class, reached by the same kind of workload:**
select5's 64-table joins under QEMU (many cursors) and a 2-cursor self-join on hardware.

**CAVEAT ON THE NUMBERING, stated because the source states it:** the enum's own comment warns
that a nearby block "disagrees with both encoders and with riscv_pkg.sv and looks like an
off-by-one in its own right". So mcause 25 is `UNEXPECTED_OPERAND`, or under that off-by-one
`INVALID_CAPABILITY`. **Either way it is a capability fault on an operand**, which is the part
the argument rests on.

## 3. The likely chain, stated as a chain and not a proof

    two concurrent cursors on one table
      -> a pointer-shaped value comes back untagged (NOT_CAP)
      -> a capability op on it raises UNEXPECTED_OPERAND (mcause 25)
      -> the fault path then wedges (commit pc = 2, the documented M-mode wedge shape)

**`p1_scalarsub` and `p8_selfjoin` are therefore probably NOT the same failure.** `p1` shows no
capability trap and stops at `0x17FFFC`; `p8` traps with mcause 25 and stops at pc 2. They may be
two faults reachable from the same cursor condition, or two different bugs. **Not folded together.**

## What this does NOT establish

* WHY two cursors produce an untagged value — the arena probe showed capabilities survive
  store/reload/evict cycles through `sqlite_heap` cleanly (passA 0, passB 0).
* Whether `same table` matters or merely `two cursors`. `p9_subq_other` (two cursors on
  DIFFERENT tables) was collateral when `p8` wedged in slot 2 and still needs a boot.
* Why `p1`'s wedge has no capability trap at all.


---

# QUALIFIED: "two concurrent cursors are the trigger" rests on a CONFOUNDED comparison

**`p8_selfjoin` differs from `p2_nosubq` in TWO ways, not one:**

    p2_nosubq     1 cursor,   30 output rows through ORDER BY   RETURNS
    p8_selfjoin   2 cursors, 435 output rows through ORDER BY   WEDGES

A second cursor **and** ~14.5x the rows through the sorter. I attributed the wedge to the cursor
alone. That is the same single-variable discipline I applied correctly to `p1` vs `p2` and failed
to apply here — and it is the exact shape this project warns about: a ladder measuring whichever
difference you did not intend.

**The linearity hypothesis is separately DEAD.** Two cursors alias the same pager page, and
Capstone capabilities are linear — a move invalidates the source — so aliasing looked like a
mechanism. Measured instead of assumed: an aliasing pass (store, copy to a second slot, re-check
the ORIGINAL) reports **0 failures**, and the LCC TOTAL type query says why:

    HEAPCAP types: heapslot=1 src=1 heapbase=1   (1 = NONLIN, 7 = NOT_CAP)

**Heap capabilities are NON-LINEAR**, so a move cannot invalidate them and aliasing is safe by
construction. Pass C's zero is explained, not merely observed.

## The separating probes, built and native-verified

    p11_smalljoin   2 cursors,  30 rows   SELECT t1.a FROM t1, t1 AS y WHERE y.a = t1.a ORDER BY 1
    p10_bigsort     1 cursor,  435 rows   SELECT a FROM big ORDER BY 1

* `p11` wedges -> **cursor count** is the factor
* `p10` wedges -> **output-row count / the sorter** is, and "two cursors" is retracted outright
* both wedge -> two independent triggers
* neither -> it is the **combination**, which is a narrower answer than either alone

Order: control, `p11` (2), `p10` (3), `p8` (4 — known-wedging, so the slot-4 monitor hazard costs
nothing).


---

# NEITHER FACTOR ALONE — it is the COMBINATION, and it looks like ARENA PRESSURE (session 10)

Control-green. All three returning arms verified to have reached `G/enter`.

| probe | cursors | output rows | slot | result |
|---|---|---|---|---|
| `p2_nosubq` | 1 | 30 | (s8) | returns |
| `p11_smalljoin` | **2** | 30 | 2 | **returns**, 8 s |
| `p10_bigsort` | 1 | **435** | 3 | **returns**, 9 s |
| `p8_selfjoin` | **2** | **435** | 2 (s9) | **WEDGES**, mcause 25 |

**"Two concurrent cursors are the trigger" is REFUTED by its own control.** `p11` has two cursors
and returns. **"Output-row count" is refuted too** — `p10` has 435 rows and returns. Both
single-factor claims are dead; **the wedge needs both together.**

**And "both together" has a physical reading: `p8` is the most HEAP-HUNGRY of the four** — two
cursors *and* a 435-row sorter inside a 256 KiB arena. That unifies with a result measured
earlier and not connected at the time: **select5 crashed at a 1 MiB arena and PASSED at 2 MiB.**
It also explains why the wedge PC of the `p1`-class failures sits *inside* `sqlite_heap`.

**Next test, and it is a true matched pair:** the same `p8_selfjoin` case, same slot, differing in
**one** parameter — `SQLITE_HEAP_SIZE` 256 KiB versus 1 MiB, as two domains in one boot. If the
larger arena returns while the smaller wedges, this is an arena-capacity threshold and neither
cursors nor SQL constructs are causal.

## SLOT 4 FAILS IN THE MONITOR — now N=2, reproducible, and not about SQL

Session 10's arm 4 reached only `A/dom-ok, B/mkregion1, C/mkregion2` and wedged at commit pc
`0x80020fbe`, which the firmware symbol table places in **`_split_out_cap`**. That is byte-for-byte
the same failure as session 7's arm 4, with a different domain and different SQL.

**So the 4th domain of a boot fails in the monitor's region path regardless of what it runs.**
`preflight-board-run.sh` permits four domains on the basis that the monitor spins near the *fifth*
`create_dom` — but each SLT arm performs one `create_dom` **and two `create_region`s**, so the
region path is exercised three times per arm. **The effective budget for region-creating domains
is THREE readable slots, not four.** Session 7 lost an arm to this and session 10 lost another;
the rule is now evidenced twice rather than inferred once.


---

# THE ARENA-CAPACITY HYPOTHESIS IS REFUTED (sessions 11-12)

A one-variable matched pair: the same `p8_selfjoin.test`, the same slot order, two domains
differing only in `SQLITE_HEAP_SIZE`.

    sqbig  1 MiB arena, slot 2   ENTERED, WEDGED   trap 0x99 (mcause 25), commit pc 0x2
    sqslt  256 KiB arena (s9)    ENTERED, WEDGED   trap 0x99 (mcause 25), commit pc 0x2

**Identical signatures at both sizes. Heap capacity is not the variable.** I had assembled a tidy
story around it — select5 passing at 2 MiB, the wedge PC inside `sqlite_heap`, mcause 25 being
what a NULL raises — and the test says no.

**In hindsight the arithmetic never supported it:** `p8` is a 30-row self-join producing 435 small
rows, which was never going to exhaust 256 KiB, and `p10_bigsort` pushes the same 435 rows through
the same sorter at 256 KiB and returns in 9 s. The correlation with select5's arena size was real
but not causal for this case.

**Session 11 was VOID** — its control produced no output at all, not even libcapstone's first
line, the same total-silence signature as the very first boot of the investigation. Re-run
unchanged, the control returned 5/5. Two such failures in ~12 boots is consistent with the
documented ~1-in-5 infrastructure rate, and the rule earned its keep: the void boot cost eight
minutes and produced no wrong answer.

## Where the root cause stands: six hypotheses dead, each by its own control

| hypothesis | how it died |
|---|---|
| hostile uninitialised DRAM | poisoned arena passes, `query_pass=5` |
| revocation-node pool exhaustion | 128-255 cumulative allocations against a ~1021 pool |
| arena corrupts capabilities | passA 0 **and** passB 0 (incl. a DRAM round trip) |
| linear-capability aliasing | heap capabilities read type **1 = NONLIN**; a move cannot invalidate them |
| two concurrent cursors alone | `p11_smalljoin` (2 cursors, 30 rows) **returns** |
| output-row count alone | `p10_bigsort` (1 cursor, 435 rows) **returns** |
| **arena capacity** | **`p8` wedges identically at 256 KiB and 1 MiB** |

## What survives, stated exactly

**`p8_selfjoin` needs BOTH two cursors AND ~435 sorted output rows, and heap size is irrelevant.**
The fault is `mcause 25` — `UNEXPECTED_OPERAND`, raised when a capability operation gets a
`NOT_CAP` operand — reproduced twice with identical machine state. Under QEMU the same fault class
appears for select5 as a `cincoffset` on an all-zero untagged operand inside `sqlite3VdbeExec`.

**So something in a 2-cursor, many-row join produces an untagged pointer, deterministically, and
it is not memory pressure.**

## Next measurement, and it is quantitative rather than another guess

Instrument memsys5 to count `malloc`, `free` and the OOM return, then run the three cases that
differ — `p8` (wedges), `p11` (2 cursors, few rows, returns), `p10` (1 cursor, many rows,
returns). **What differs numerically between a wedging and a passing case is data, where another
mechanism story would be speculation.** The OOM half of that counter is already built and
calibrated: it reports 1 at a 64 KiB arena and 0 at 1 MiB, so it discriminates.


---

# THE FAULT IS IN THE FIRST 2,000 VDBE OPCODES (session 13)

Control-green boot. `p8_selfjoin` executes **6,715** VDBE opcodes in total. Clamped to **2,000**
it still **WEDGED**, so the faulting opcode is at or before 2,000 — the window fell by 3.4x on one
arm.

## Why a clamp, and why it took this long to reach for it

**A wedged domain never returns, so it can never report anything.** Twelve boots of measurement
have all been *outside* the fault, inferring from what did not happen. The clamp stops
`sqlite3VdbeExec` after N opcodes and returns `SQLITE_DONE`, so the arm comes back carrying state:
`lastop` (the opcode it declined to run) and the allocator counters at that moment. That is the
project's own rule — prefer a diagnostic that converts a hang into a wrong answer over one that
observes the hang — and it is what finally gets inside the failure.

## The clamp is now RUNTIME-selectable, which changes the economics

As a compile-time constant, each clamp value cost a firmware rebuild plus a boot — about eleven of
each to reach a single opcode. It now reads from the hostcall block's unused `phase` field
(`--clamp N` on the host), so **one image answers any value** and a bisection step is a boot
alone. Verified that the knob moves: with the built-in constant at 99,999,999, `--clamp 100`
reports `ops=100 lastop=40` (`OP_Next`) and `--clamp 5000` reports `ops=5000 lastop=96`
(`OP_Column`).

## Allocator traffic, measured at a 256 KiB arena under QEMU

    p11_smalljoin (returns)   malloc=1157  free=1157  oom=0
    p10_bigsort   (returns)   malloc=1206  free=1206  oom=0
    p8_selfjoin   (WEDGES)    malloc=1516  free=1516  oom=0

**Balanced, so no leak; `oom=0`, so no allocation failure under emulation.** What differs between
the wedging and passing cases is traffic volume, not failure. The same counters on silicon are the
first-link evidence for the NULL chain — the gap that got an earlier root-cause claim retracted.

## Next

Bisect 0-2,000 with `--clamp 500 / 1000 / 1500` in one boot, one image. The last clamp that
RETURNS names the faulting opcode in `lastop` and reports whether the allocator had already
failed.


---

# THE FAULT IS IN THE FIRST ~125 VDBE OPCODES — accumulation is dead (sessions 15-17)

Clamp bisection of `p8_selfjoin`, every wedging arm verified to have reached `G/enter` and to
carry the same signature (trap `0x99` = mcause 25):

    total opcodes   6,715
    clamp 2000      WEDGE
    clamp  500      WEDGE
    clamp  125      WEDGE      <-- current bound

**`p8`'s entire VDBE program is 23 opcodes**, so 125 executions is the setup plus roughly three
or four inner-loop iterations. **The fault happens almost immediately.**

That retires every remaining accumulation story — rows piling into the sorter, allocator traffic
building, pool growth — all of which need hundreds or thousands of opcodes. Something goes wrong
in the **first few passes of the inner scan**.

## AND `p11_smalljoin` WAS NEVER A VALID CONTROL

I claimed "two cursors alone returns" on the strength of `p11`. Reading the plans instead of the
SQL:

    p8  (y.b < t1.b)   SCAN t1, SCAN y                              true nested loop, 900 steps
    p11 (y.a = t1.a)   SCAN t1, SEARCH y USING AUTOMATIC COVERING
                       INDEX (a=?) + BLOOM FILTER                   indexed lookup, ~30 steps

**SQLite silently built an automatic covering index for the equijoin**, so `p11` has two cursors
but performs no repeated inner scan. It is not the control I described, and the 2x2 built on it is
confounded a second time — in a way the row counts concealed.

**The lesson is specific and general: on a query engine, "same shape of SQL" is not "same
execution plan". The plan must be READ, not inferred from the text.** Two of my controls have now
turned out to execute something structurally different from what I assumed.

`p12_countjoin` is the repaired control: `SELECT count(*) FROM t1, t1 AS y WHERE y.b < t1.b` —
plan `SCAN t1, SCAN y`, identical 900-step nested scan, **one output row, no sorter**, verified to
return natively. If it wedges, the nested scan is the factor and output rows are irrelevant, which
would also explain why `p10_bigsort` (435 rows, no nested loop) returns.

## Localization update — instrument now PROVEN, and the localization moved again

Earlier framing (inner loop / two iterating cursors / accumulated rows) is **withdrawn**. The
clamp ladder that supported it was monotone for a trivial reason: every clamp in it
(2000/500/125/30) sits far past a fault that happens in the first seven opcodes.

Measured, boot #18 (control `sqbase.dom` passed → boot valid):

| clamp | opcodes executed | QEMU | silicon |
|---|---|---|---|
| 8 | 1–7 | returns, `SLT-VDBE ops=8 lastop=36` | **WEDGE** (`G/enter`, `ENT0`, `ENT1`, then nothing) |

Clamp semantics: the injected test is `++ops >= clamp_n`, so clamp N executes **N-1** opcodes
and `lastop` is the Nth (the one about to run). Verified, not assumed: QEMU clamp 8 reports
`lastop=36` = `OP_Rewind`, exactly the 8th entry of the execution order below.

`p8_selfjoin.test` query program, in execution order:

    1 Init  2 Transaction  3 Goto  4 SorterOpen  5 OpenRead(c0)  6 OpenRead(c1)  7 Rewind(c0)
    8 Rewind(c1)  9 Column 10 Column 11 Ge  [12 Column 13 MakeRecord 14 SorterInsert] 15 Next

So the silicon fault is inside opcodes 1-7 — **setup: opening the sorter and the two cursors**.
`Column`/`Ge`/`MakeRecord`/`SorterInsert` never run. This retires the join-body,
row-count and accumulation hypotheses directly rather than by inference.

Two instrument facts established while getting here, both of which would have produced a wrong
reading:

- **The clamp is query-scoped, despite a file-wide pre-arm.** `sqlite_capstone_domain.c:6106`
  sets `capstone_vdbe_armed = 1` before `slt_run`, and the 31 setup statements never DISARM, so
  the counter looked like it should include `CREATE TABLE` + 30 `INSERT`s. Measurement says
  otherwise: `stmt_pass=31 stmt_fail=0`, and the query reached `OP_Rewind`, which requires
  `OpenRead` to have found a real table. The setup statements are not truncated and the opcode
  map is against the right program.
- **`M5 oom=/malloc=/free=` are inert unless `CAPSTONE_MEMSYS5_OOM=1` is compiled in.** The QEMU
  arm printed `malloc=0`, impossible for a working SQLite — the knob was simply off. The baked
  board image *does* carry it (`bake16.log:3`), so board `oom=` is meaningful and QEMU's is not.
  Do not read `oom=0` from a build without the knob as "no allocation failure".

### THE LOCALIZATION ABOVE IS NOT YET SUPPORTED. Read this first.

The runtime clamp has **never been observed to work on silicon**. Every arm of the ladder
(2000, 500, 125, 30, 8) WEDGED, and a wedged arm prints no `ops=`/`lastop=` -- so on the board
there is not one instance of the clamp demonstrably firing. Two hypotheses fit every silicon
datapoint equally well:

* **(a)** the clamp applies, and the fault is inside query opcodes 1-7 (the table above);
* **(b)** the clamp never applies on silicon -- `metadata->phase` does not survive, or the read
  at `sqlite_capstone_domain.c:6070` gets something else -- so every arm ran effectively
  UNCLAMPED and wedged at the real fault, wherever that is. The ladder would then localize
  nothing at all, and its monotonicity would be an artifact.

QEMU proves the clamp works *in QEMU*. It says nothing about the silicon path, which is the
one under test. The plumbing (`sqlite_host.c:152` publishes `clamp_n` in `phase`; the domain
reads it at 6070) is correct by inspection, and inspection is not what is in doubt.

**Positive control required before any clamp result is believed:** clamp a test that is KNOWN
to return on silicon (`negative-control.test`), at a value that must change its
`SLT-SUMMARY` in a predicted way, and confirm `ops=` equals the clamp and `lastop=` matches
the QEMU reference. Until that arm returns, treat the whole ladder -- including the table
above -- as UNRESOLVED.

Next discriminator: clamp 1 (query executes zero opcodes — isolates setup statements from the
query) and clamp 6 (splits the 1-7 window). Arm 3 of boot #18 was collateral after the wedge and
carries no verdict.


## Boot #19: the clamp is proven, and the fault is in PREPARE, not execution

Control `sqbase.dom` passed, so the boot is valid.

**Arm 2 -- the instrument is proven.** `negative-control.test --clamp 5` RETURNED on silicon
with a tally bit-identical to the QEMU reference, and clearly different from its own unclamped
tally:

    silicon clamp 5 : records=21 stmt_pass=9 stmt_fail=2 query_pass=0 query_fail=9 skip_big=1
                      oom=0 skip_cond=2 parse_err=1 completed=1
                      M5 oom=0 malloc=1315 free=1315   ops=5 lastop=114
    QEMU    clamp 5 : IDENTICAL, every field
    unclamped (both): query_pass=6 query_fail=4 skip_big=0

So the runtime clamp DOES fire on silicon; the ladder is valid. The doubt recorded above is
resolved in the ladder's favour -- but see the next arm, which makes it moot.

**Arm 3 -- and the ladder was measuring the wrong thing anyway.** `p8_selfjoin --clamp 1`
entered and WEDGED. Clamp 1 fires on the very first opcode, so the query executes ZERO VDBE
opcodes -- and it still wedges. The fault is therefore not in the query's bytecode at all.

The mechanism is specific and explains the whole ladder: `slt_runner.h` calls
`sqlite3_prepare_v2` at line ~440 and only arms the clamp at line 467, AFTER the prepare
succeeds. **Prepare is entirely unclamped.** The parser, query planner and code generator for
the self-join run in full for every clamp value, which is exactly why 2000, 500, 125, 30, 8 and
1 all wedged identically. A monotone ladder, and every rung was reporting the same upstream
failure -- the failure mode CLAUDE.md warns about, arrived at a second time by a different road.

Confound checked before believing it: p8 wedged in SLOT 2 (boot #18) and SLOT 3 (boot #19),
while `negative-control` PASSED in slot 2 of boot #19. The wedge follows the workload, not the
slot position.

**Retired by this result:** every hypothesis about VDBE execution -- join body, two cursors
iterating, output-row count, accumulation, sorter inserts, and the "arena exhausted -> NULL ->
UNEXPECTED_OPERAND" chain insofar as it was pinned to execution. `M5 oom=0 malloc=1315
free=1315` on the returning arm also shows the allocator is healthy and its counters live.

**Next, the minimal repro.** Two new files, QEMU-referenced before they go near the board:

* `slt/p8_trivial.test` -- the identical 31 setup statements, trivial query. Expected to RETURN;
  a wedge would mean the INSERTs are the trigger.
* `slt/p8_empty.test --clamp 1` -- the self-join over an EMPTY table, zero query opcodes. A
  wedge here has no data and no execution left to blame and localizes the fault to
  `sqlite3_prepare_v2` outright.

## Boot #20: the fault is in sqlite3_prepare_v2. Setup statements are exonerated.

Control passed, boot valid.

**Arm 2 -- `p8_trivial.test` (identical 31 setup statements, trivial query) RETURNED**, with a
tally bit-identical to the QEMU reference:

    records=32 stmt_pass=31 stmt_fail=0 query_pass=0 query_fail=1 skip_big=0 oom=0
    skip_cond=0 parse_err=0 completed=1     M5 oom=0 malloc=1049 free=1049   ops=10 lastop=0

So `CREATE TABLE` and all 30 `INSERT`s execute correctly on silicon. The setup statements are
exonerated, and with them the last remaining "data volume" explanation.

**Arm 3 -- `p8_empty.test --clamp 1` WEDGED.** That file is ONE `CREATE TABLE` plus the
self-join, over an EMPTY table, with the clamp firing on the first opcode so ZERO query opcodes
execute. No INSERTs, no rows, no bytecode. The only remaining code is `sqlite3_prepare_v2` --
parser, query planner, code generator.

**Localization: the wedge is at PREPARE time, not execution time.** Reproducer is 465 bytes.

Two exonerations available offline, from workloads already known to pass on silicon:

* **`ORDER BY` is exonerated.** `negative-control.test` contains `SELECT a FROM t1 ORDER BY a`
  and `SELECT x FROM big ORDER BY x`, and it passes on silicon -- matching native field-for-field
  unclamped, and matching QEMU field-for-field at clamp 5. The sorter path works.
* **Joins have NEVER been exercised successfully on silicon.** select1/select2/select3 -- the
  slices that passed, 5320 SELECTs between them -- contain ZERO multi-table `FROM` clauses.
  Every silicon success to date is single-table.

That makes the planner's JOIN path the prime suspect. For this query `EXPLAIN QUERY PLAN`
previously showed `SEARCH y USING AUTOMATIC COVERING INDEX (a=?)` plus a BLOOM FILTER, so
automatic-index construction at prepare time is the specific mechanism to attack next.

**Next: a matched pair differing by EXACTLY one thing** (`slt/q_one.test`, `slt/q_two.test`):

    q_one:  SELECT t1.a FROM t1
    q_two:  SELECT t1.a FROM t1, t1 AS y

Same empty table, same sort mode, same return path, no ORDER BY on either side. `diff` between
the two files is one line. If q_one returns and q_two wedges, the join alone is the trigger.

## Boot #21: MINIMAL REPRODUCER -- one extra FROM term wedges the core

Control passed, boot valid. The matched pair separates:

| arm | query (table is EMPTY in both) | silicon |
|---|---|---|
| `q_one.test` | `SELECT t1.a FROM t1` | RETURNS -- bit-identical to QEMU: `ops=6`, `M5 oom=0 malloc=236 free=236`, `records=2 stmt_pass=1 stmt_fail=0 query_pass=0 query_fail=1 completed=1` |
| `q_two.test` | `SELECT t1.a FROM t1, t1 AS y` | **WEDGE** |

`diff q_one.test q_two.test` is ONE line. Same empty table, same sort mode, same return path,
no ORDER BY on either side, no WHERE on either side, no rows anywhere. On QEMU the pair differs
by 1 opcode and 5 allocations (6/236 vs 7/241).

**A single additional table reference in the FROM clause is the whole trigger.** 588 bytes.

### What is established, and what is NOT

ESTABLISHED:

* `q_one` vs `q_two`: the trigger is the second FROM term and nothing else. The arms differ in
  exactly one respect, so the difference between them IS the variable.
* From boot #20, for the self-join WITH `WHERE` and `ORDER BY`: `p8_empty --clamp 1` wedges with
  ZERO VDBE opcodes executed, so THAT query's wedge is inside `sqlite3_prepare_v2`.
* Setup statements, data volume, ORDER BY/the sorter, and the whole VDBE-execution family are
  retired, each by its own control.

NOT ESTABLISHED -- do not write this down as though it were:

* **Whether `q_two` wedges in prepare or in its 7 opcodes.** `q_two` ran UNCLAMPED. The
  prepare-time attribution is proven for `p8_empty`, not for `q_two`. `q_two --clamp 1` is the
  arm that settles it and has not been run.
* **Whether this is a silicon defect or a codegen bug.** QEMU runs both arms fine, but that is
  exactly what the gp/`__split` investigation looked like too, where QEMU was permissive and the
  RTL enforced. "QEMU passes, silicon wedges" is equally consistent with our compiler emitting
  something the hardware legitimately rejects. Nothing here yet distinguishes the two, and it
  must not be handed to the hardware side as a silicon defect until something does.

Note the automatic-index theory from boot #20 does NOT survive contact with this result:
`q_two` has no `WHERE` clause, so the planner builds no automatic index and no bloom filter,
and it wedges anyway. The trigger is simpler than that.

### Next

1. `q_two --clamp 1` -- prepare vs execution for the minimal case.
2. `SELECT t1.a FROM t1, t2` (two DISTINCT tables) -- is it a self-join or any second FROM term?

## ROOT LOCALIZATION: the faulting instruction, named

The per-arm `PROBE_SCOPED_OUT` files carry a full wedge signature and had not been read. They do.
Boots #20 and #21 -- two independently constructed reproducers -- have IDENTICAL signatures:

    sw=255 TRAP LOG {seen,mcause[6:0]}   0x99  -> seen=1, mcause 0x19 = 25 UNEXPECTED_OPERAND
    sw=196..203 trap mepc (LATCHED)      0x0000000082cf499c
    sw=249/250  rev_node_head            0x0276 = 630   (pool is ~1021: NOT exhaustion)
    sw=224      privM=1                  (expected: domains run at M with capmode)

`q_two` loaded at `DBAS=0x82C00000`; the loadable segment is VMA 0x10000 size 0x1609E8 and
`DENT=0`, so VA = 0x10000 + (mepc - DBAS) = **0x10499C**. The domain is an unstripped ELF
(hash `277b73f08742a71b`, unchanged across four bakes), so the symbol resolves directly:

    0x104910  sqlite3WhereCodeOneLoopStart      <- fault is 0x8C into this function

That is SQLite's **query-planner loop code generator**, which runs at PREPARE time and is called
once per FROM term. It independently confirms the prepare-time attribution for `q_two`, which
boot #21 alone could not establish.

The instruction:

    104998: ldc            a4, 0x0(a0)      ; reload a capability from a stack slot
    10499c: cincoffsetimm  a4, a4, 0xb0     ; <-- TRAPS

From the prologue the incoming arguments spill as `a0->[s0-0x60]`, `a1->[s0-0x50]`,
`a2->[s0-0x70]`, `a3->[s0-0x74]` (`sw`, an int), `a4->[s0-0x90]`, `a5->[s0-0x98]` (`sd`, a
64-bit scalar). At 0x104948 `a0 = s0-0x70` and is not reassigned before 0x104998, so the `ldc`
reloads **the third argument**. The signature confirms the mapping exactly -- arg 4 is
`int iLevel` (hence `sw`) and arg 6 is `Bitmask notReady` (hence `sd`):

    Bitmask sqlite3WhereCodeOneLoopStart(Parse *pParse, Vdbe *v, WhereInfo *pWInfo,
                                         int iLevel, WhereLevel *pLevel, Bitmask notReady)

**So: reloading `pWInfo` from its spill slot yields a value that is not a valid capability, and
`cincoffsetimm` on it raises UNEXPECTED_OPERAND.** mcause 25, not 29 -- this is a tag/type
failure, NOT a bounds failure. A tag was lost between the `stc` that spilled `pWInfo` and the
`ldc` that reloaded it, or `pWInfo` was already untagged on entry.

### What this does and does not settle

* It settles prepare-vs-execute for the minimal case: `sqlite3WhereCodeOneLoopStart` is code
  generation, so `q_two` wedges at prepare, like `p8_empty`.
* It does NOT yet settle silicon-vs-codegen. QEMU executes the identical path with a tagged
  value (QEMU asserts on untagged `ldc` via `capstone_report_untagged` and does not fire).
* A HYPOTHESIS worth testing, not a conclusion: this is the shape of **S-06** (untagged
  `ldc`/`stc` mishandled in RTL). Whether the flashed bitstream
  `caplifive_s10fix_80843404c.bit` carries the S-06 fix is unknown here and is a question for
  the RTL lane. Do not report this to the hardware side as a defect before that is answered --
  if the fix is absent, this may be a known bug rather than a new one.
* Still open: whether `pWInfo` is already untagged on ENTRY (pointing upstream to
  `sqlite3WhereBegin`, which allocates the variable-sized `WhereInfo` -- and its size DOES
  depend on the number of FROM terms, which is the one thing that differs between q_one and
  q_two), or whether the tag is lost in the spill slot itself.

## S-06 REFUTED as the cause; S-10b is the candidate. Static analysis of both frames.

The RTL lane answered the bitstream question, and the answer kills my own hypothesis:

* **S-06 IS in `caplifive_s10fix_80843404c.bit`.** Verified two ways on their side: `25035c4c0`
  is an ancestor of `80843404c`, and the whole `core/` delta from the S-07 bitstream's base to
  `80843404c` is exactly one file, `wt_dcache_mem.sv`. So S-06/S-07/S-08 are all in and S-10 is
  the only addition. **This is not S-06** -- my hypothesis in the previous commit is REFUTED.
* **The AMO I-4 residual is not in play.** It is confined to atomics, and its polarity is the
  OPPOSITE of this symptom: it makes non-capability data read back as TAGGED. Ours is a tag
  going missing.
* **S-10b is the candidate and is NOT in this bitstream** (`c867dfcbb` is not an ancestor;
  `store_buffer.sv` still compares `page_offset_i[11:3]`). The RAW interlock between a load and
  a not-yet-drained store compares at 64-bit WORD granularity while a tag is a per-16-byte
  GRANULE property. It has no shippable fix: widening the compares to `[11:4]` failed synthesis
  twice with `DRC LUTLP-1`, a 69-LUT combinational loop across `rev_node`/`load_unit`/
  `csr_regfile`, and `write_bitstream` refused.

### Their check #1 -- does a scalar store share the pWInfo granule? NO, in either frame.

Checked statically, verifying each `cincoffsetimm`/store pair inside its own basic block rather
than by linear scan (a whole-function linear scan is control-flow-blind and gave a WRONG answer
first time -- it paired definitions with stores thousands of instructions away).

CALLER `sqlite3WhereBegin` (0xef8ac):

* pWInfo lives at caller `s0-0xc0`, written ONCE by `stc a1, 0x0(a2)` at `0xefa80`, BEFORE the loop.
* The calling loop is `0xf10a0..0xf1398` (back-edge `j 0xf10a0` at `0xf1398`), reloading pWInfo
  each iteration at `0xf1304`/`0xf130c`.
* **Inside that loop nothing writes into `[s0-0xc0, s0-0xb0)`** -- no scalar store, no capability
  store.
* Scalar stores DO hit that granule (`-0xc0`, `-0xbc`, `-0xb8` = base+8, `-0xb4`) at `0xf16c4`,
  `0xf16cc`, `0xf1d54`, `0xf2920`, `0xf3538`, `0xf3544` -- but every one is AFTER the loop ends.
  That is ordinary slot reuse once pWInfo is dead. **So this is NOT a codegen live-range
  overlap**, which was the other way this could have gone and would have made it our bug.

CALLEE `sqlite3WhereCodeOneLoopStart` (0x104910):

* pWInfo spills to callee `s0-0x70`, 16-byte aligned, granule `[s0-0x70, s0-0x60)`.
* The only nearby scalar, `sw a3, 0x0(a2)` at `0x10495c` with `a2 = s0-0x74`, is in the PREVIOUS
  granule `[s0-0x80, s0-0x70)`. No overlap.

### The better candidate: the callee's own tight stc->ldc pair

    104950: stc a2, 0x0(a0)      ; a0 = s0-0x70, spill pWInfo
    ...9 unrelated stores in between...
    104998: ldc a4, 0x0(a0)      ; SAME address, 18 instructions later
    10499c: cincoffsetimm a4, a4, 0xb0    ; traps

Same address, so a `[11:3]` compare should match on word 0 and stall. The symptom needs correct
DATA with a STALE (clear) TAG, i.e. the reverse polarity of the S-10b legs described to me.
Whether the word-vs-granule mismatch can run that way is with the RTL lane.

### Call counts, measured natively (counter injected into the amalgamation)

    SELECT t1.a FROM t1            -> 1 call  (iLevel=0)
    SELECT t1.a FROM t1, t1 AS y   -> 2 calls (iLevel=0 then 1), SAME pWInfo pointer

**Correction to the previous commit's reasoning:** I argued the fault must be on call #2, since
0x10499C is unconditional prologue code and q_one's single call passes it. That assumed q_two's
FIRST call behaves like q_one's, which is not guaranteed -- the preceding planning code differs,
so store-buffer residency can differ. Call #2 is likely but NOT proven, and is not claimed.

## The constraint that any hypothesis must satisfy: 6537 identical pairs work

Counted over the whole 331795-instruction text of the same binary: same-address
`stc rS, D(rB)` -> `ldc rD, D(rB)` pairs with no redefinition of `rB` between them.

    within 40 instructions : 7807
    within 18 instructions : 6537     <- 18 is the distance of the FAULTING pair

This binary runs 5320 single-table SELECTs, 10807 SLT records under QEMU, and on SILICON runs
sqbase, negative-control, p8_trivial and q_one cleanly -- bit-identical to QEMU wherever there
is a reference. **So a generic "tight same-address stc->ldc reads a stale tag" is refuted by the
software's own behaviour**: it would fire continuously and nothing would ever run.

Whatever the mechanism is, it must explain why it does NOT fire on 6537 near-identical pairs.
Candidate distinguishing features of the failing window, ranked:

1. the **two ctag=0 capability entries** (`movc a4, zero` then `stc`, to `s0-0x5a0` and
   `s0-0x120`) -- most spill sequences have none;
2. **9 intervening stores**, possibly exceeding a buffer depth and reaching a drain/eviction
   path shorter windows never touch;
3. the **mix** (6 tagged stc, 2 ctag=0 stc, 3 scalar) rather than the distance.

### Alignment verified, not assumed

The domain `sp` comes from the monitor's cscratch region and every adjustment is a multiple of
16 (`-96` in the entry glue, then `cincoffsetimm sp,sp,-0x7f0`, `-0x450`). `s0 = sp` at function
entry. So `s0-0x70` IS granule-aligned, which CONFIRMS the RTL side's "granule-aligned, one
write-buffer entry" premise rather than breaking it. Worth stating because if it had been false,
a capability spill would straddle two granules and several arguments would change.

### Hypothesis ledger

| hypothesis | status | killed by |
|---|---|---|
| VDBE execution (join body, cursors, rows, sorter) | DEAD | `p8_selfjoin --clamp 1`, zero opcodes |
| setup statements / data volume | DEAD | `p8_trivial` returns; `p8_empty` is empty |
| ORDER BY / sorter | DEAD | negative-control uses it and passes on silicon |
| rev-node pool exhaustion | DEAD | latched head = 630 of ~1021 |
| codegen live-range overlap on the spill slot | DEAD | the granule's scalar stores are all AFTER the loop |
| S-06 untagged ldc/stc | DEAD | `25035c4c0` IS an ancestor of the flashed bitstream |
| AMO I-4 residual | DEAD | atomics only, and opposite polarity |
| S-10b store-buffer word-vs-granule | WITHDRAWN | same-address pair matches at `[11:3]` on word 0 |
| S-10 `wbuffer_gran_clr` | WEAK | entry class DOES exist (ctag=0 `stc`), but addresses do not overlap and the `paddr[55:4]` compare reads as correct |
| store buffer upstream of the tag path | **SURVIVING** | tag is sourced from write buffer or L1, never the store buffer -- so a store-buffer-resident `stc` leaves `wbuffer_be`=0 and the tag comes from an L1 array the `stc` has not reached |

The RTL lane is building a directed Verilator test of the exact window (offsets above, both
ctag=0 entries included) with a granule-distance sweep whose same-granule arm is a positive
control, A/B across S-10 present and absent. Board-free and lock-free.

### Open, needs the project lead

`caplifive_s07fix.bit` differs from the current image by exactly one `core/` file
(`wt_dcache_mem.sv`, i.e. S-10). Running the 588-byte repro on the S-07 image would say directly
whether S-10 introduced this. That needs a **reflash, which is ask-first** and the lead's call.
The earlier 3/3 plain-SQLite baseline on that image is NOT evidence here: it was single-table
throughout, so it never built a self-join spill layout.

## RESOLVED DIRECTION: tval = 0. It is a NULL DEREFERENCE, in software. Not silicon.

An adversarial audit of the previous localization confirmed the address, function, instruction
and operand slot (it also found corroboration I had missed: `cincoffsetimm a4, a4, 0xb0` is
literally `pWC = &pWInfo->sWC;` at `sqlite3-capstone.c:165463`, the function's FIRST statement,
with the next three source lines mapping 1:1 onto `0x1049a8..0x1049cc`). But it REFUTED the
semantic step, on two grounds:

1. **mcause 25 has TWO live producers on this bitstream.** `ex_stage.sv:479` builds
   `64'd24 + code` (ordinal 1 = UNEXPECTED_OPERAND), but `commit_stage.sv:226` at `80843404c`
   emits `64'd25` from the **PC-capability** check using base **23**, into the same latch.
   `capstone_unit.anvilh:299-301` documents the collision -- I had quoted only the first half of
   that note.
2. **"25 not 29, so tag not bounds" was VACUOUS.** `capstone_flu_unit.anvil:57-90` gives
   `CINCOFFSETIMM` no bounds arm at all; 29 was unreachable, so excluding it excluded nothing.
   An exclusion that could not have gone the other way. (Also: 25 is specifically NOT_CAP;
   a genuine *type* failure would be 27, so "tag/type failure" blurred the distinction.)

**The discriminator existed on this bitstream and had never been sampled.** `tval` carries the
rs1 CURSOR for a capability cause (`ex_stage.sv:487`) but the PC for the PC-cap cause
(`commit_stage.sv:604`). Added switches 210/211/213-218 to the wedge readout -- note aperture
212 is SKIPPED in the mux, verified against `git show 80843404c:core/cva6.sv:1355-1362` -- with
assembly, an all-or-nothing rule, and all three verdict branches positive-controlled offline
before spending the boot.

Boot #22, control passed, `q_one` returned bit-identically to QEMU and to boot #21:

    trap_seen = 1        (sw=255 -> 0x99, bit 7)
    mcause    = 25
    mepc      = 0x0000000082cf499c   <- MATCHES the faulting instruction, so the latch is NOT stale
    tval      = 0x0000000000000000

The staleness guard matters and it passes: the latch is last-writer-wins on commit-stage
exceptions, so a tval whose mepc did not match would have belonged to some earlier trap.

**tval = 0 means the operand was a NULL/integer.** The reasoning is sharper than the RTL comment
alone: tval carries the rs1 **cursor**, so a capability that had merely lost its TAG would still
read pointer-like. Zero means the value is genuinely zero. **`pWInfo` is NULL.**

So `cincoffsetimm a4, a4, 0xb0` is `&pWInfo->sWC` on a NULL pointer. On a conventional machine
that computes 0xb0 and hurts nobody until the load; on Capstone the offset computation itself
traps. **This is a null dereference in our software, not a lost tag and not a silicon defect.**

### Everything the tag-loss reading supported is withdrawn

Withdrawn: the S-10b store-buffer route, the S-10 `wbuffer_gran_clr` route, the write-buffer
capacity chain (nine distinct granules against `WtDcacheWbufDepth = 8`), and my own "a tag went
missing across an stc/ldc spill pair". None of them is what is happening. The 6537-pair
constraint and the alignment verification remain true and simply no longer have anything to
explain. The RTL lane's directed sweep was held before it was built.

### What is NOT settled

Where the NULL comes from. `sqlite3WhereBegin` allocates `pWInfo` with `sqlite3DbMallocRawNN`
and checks `db->mallocFailed`, so a NULL should never reach the loop at all.

Ruled out already: **the heap is not the difference.** `build-sqlite-silicon.sh:44` defaults
`SQLITE_HEAP_SIZE` to `256*1024`, exactly what the board bake passes explicitly, so QEMU and
silicon run the same 256 KiB arena. And silicon's own allocator behaviour matches QEMU exactly
on the passing arm -- `q_one` reports `M5 oom=0 malloc=236 free=236` on BOTH.

Next: convert the hang into a returning answer. A build that checks `pWInfo == NULL` at entry to
`sqlite3WhereCodeOneLoopStart`, records a marker and returns, so the run survives to print
`SLT-SUMMARY` and `M5 oom=/malloc=/free=`. Non-zero `oom` means an allocation failed and the
question becomes why it failed on silicon and not under QEMU with an identical heap; zero `oom`
means the NULL came from somewhere else.

## RETRACTION: arm position is a confound, and `tval` is an UNFIRED instrument

An adversarial audit killed the width/traffic reversal AND undermined the NULL reading it
reversed. **Both readings are unsupported. Do not act on either.**

### 1. Arm position, 5 for 5

Every "matched pair" this session compared a probed/passing arm at **slot 2** against a
wedging arm at **slot 3**. Those arms differ in slot, domain id, DBAS, rgid pair, and all
allocator/rev-node state carried over from the previous arm.

| boot | arm 2 | arm 3 |
|---|---|---|
| 19 | negative-control `rc=0` | p8_selfjoin WEDGE |
| 20 | p8_trivial `rc=0` | p8_empty WEDGE |
| 21 | q_one `rc=0` | q_two WEDGE |
| 22 | q_one `rc=0` | q_two WEDGE |
| 24 | q_two (PROBED) `rc=0` | q_two WEDGE |

**Arm 3 has never completed; arm 2 has always completed.** Boots 21-22 are decisive: the
UN-PROBED binary completes at arm 2. Nothing on record has run the probed build at arm 3, or
un-probed `q_two` at arm 2 -- and boot 24's preflight warned a slot was free.

The wedge is also **input-independent**: `p8_empty` (one CREATE TABLE, no rows) wedges at the
same instruction as the self-join. Hard to square with "this query's pWInfo", easy to square
with position.

**So the q_one/q_two "matched pair" was never matched.** The one-line `diff` was real; the arm
slot was not held constant. That invalidates the minimal-reproducer reasoning built on it.

### 2. `tval` has never been shown to fire on the FLU path on this silicon

Every latched `tval` at a capability wedge in `slt-board-out*.txt` reads 0x00 in all eight
bytes. The one non-zero `tval` on record came from **mcause 15**, a store page fault, whose tval
comes from the LSU, not from `ex_stage.sv:488`. So the FLU capability path has never produced a
non-zero tval here. **A zero from an unfired instrument reads exactly like a finding.**

What I called a "positive control" was offline simulation of my own PARSING logic. That is not
the hardware path, and conflating them is the same error this file warns about.

Directed control that settles it, one arm:

    li a0, 0xBEEF ; cincoffsetimm a0, a0, 8    -> must trap mcause 25 with tval == 0xBEEF

**Until that fires, every `tval=0` is NO DATA**, and both the "software NULL" reading and the
"manufactured zero" reversal rest on it.

### 3. The probe cannot see what the fault checks

`CINCOFFSETIMM` raises on `cap_type == NOT_CAP` and never consults the cursor
(`capstone_flu_unit.anvil`). A granule with a good non-zero cursor and a clobbered `cap_type`
reads NON-ZERO to a scalar `ld` and still faults the `ldc`+`cincoffsetimm` -- the S-07 signature
verbatim. So the 8-byte read cannot distinguish tag-loss from value-loss, which are the two live
hypotheses. The width claim was unearned.

Also, probed vs un-probed differ in `stc`->`ldc` distance (19 vs 44 instructions) AND the probed
build pre-touches the granule. So even if the probe mattered, naming the LOAD path specifically
was unsupported; a store-side/commit-latency effect fits equally.

### What SURVIVES the audit

* Probe mechanics: guard runs, polarity correct, counter incremented before the test, and the
  `ldc`+`cincoffsetimm` pair still EXISTS and executed 5 times in the probed build -- nothing was
  optimised away.
* `caller_arg` is plausible and cross-checks against QEMU at the SAME allocation offset
  (`0x3E5CE0`) from a different base.
* Spill layout is identical between builds (`s0-0x50/-0x60/-0x70/-0x74/-0x90/-0x98`), so
  "different spill layout" is dead as an alternative.
* The fault is deterministic: same instruction, function+0x8C, 5 boots, 2 binaries, 6 inputs.

### Artifact hygiene problem to fix

`/tmp/capstone/slt-qemu-clamp/` is a REUSED build directory -- its `sqlite_silicon.dom` and
`sqlite-slt.log` are overwritten by every QEMU run, so QEMU reference numbers cited from it are
not reproducible afterwards. Worse: the "probe does not perturb behaviour" check compared against
`277b73f08742a71b`, a DIFFERENT un-probed binary from the one that wedged on boot 24
(`c7eff8412`). **The exact wedging binary has never been observed to complete anywhere.**
Future QEMU references need a per-run output directory.

### Next boot: break the confound, and fire the instrument

Four arms: `sqbase` control, un-probed `q_two` at **arm 2**, probed `q_two` at **arm 3**, and the
`li a0,0xBEEF; cincoffsetimm` tval control. If position predicts the outcome, the probe is
irrelevant and the real variable is domain slot / DBAS / carried-over allocator state.

## POSITION CONFOUND REFUTED. Reproducer restored on controlled evidence.

The audit's arm-position confound was correct to raise and is now settled by measurement, not
argument. Boots #26 and #28, both with a passing control:

| binary | input | arm | result |
|---|---|---|---|
| un-probed | q_one | **2** | completes (#28, `M5 malloc=236`) |
| un-probed | q_two | **2** | **WEDGES** (#26) |
| un-probed | q_two | 3 | WEDGES (#21, #22, #24, #25) |
| probed | q_two | **2** | completes (#25) |
| probed | q_two | **3** | completes (#28, `WIDTH type=1 notcap=0`) |

Three conclusions, each with position held constant:

* **POSITION IS NOT THE VARIABLE.** The un-probed build wedges at BOTH arm 2 and arm 3; the
  probed build completes at BOTH. The 5-for-5 correlation was real and is now broken in both
  directions.
* **THE INPUT IS A VARIABLE.** `q_one` completes and `q_two` wedges, same binary, same arm 2.
  The one-line reproducer (`SELECT t1.a FROM t1` vs `SELECT t1.a FROM t1, t1 AS y`, empty table)
  stands on clean evidence.
* **THE PROBE IS A VARIABLE.** Probed completes, un-probed wedges, same input, same arm.

The retraction two entries above is therefore itself partly withdrawn: the confound did not hold.
What does NOT come back is the *interpretation* -- the width story remains unsupported, because
the 8-byte read cannot see `cap_type`, which is what `CINCOFFSETIMM` actually checks, and because
probed vs un-probed differ in `stc`->`ldc` distance (19 vs 44 instructions) AND in pre-touching
the granule. "The probe changes something" is established; "the load path manufactures a zero" is
not.

**Healthy silicon `ldc` baseline, first ever measured on this bitstream** (probed arm, 3 boots):

    #25 arm2  WIDTH type=1 lo=2197710272 hi=7660733082902  calls=5 notcap=0
    #28 arm3  WIDTH type=1 lo=2201904576 hi=11337225088278 calls=5 notcap=0

`type=1` both times, `notcap=0`, `M5 oom=0 malloc=241 free=241` matching QEMU exactly.

**Still unvalidated and still blocking any fault interpretation:** the `tval` instrument. Every
`tval=0` remains no-data until `li a0,0xBEEF; cincoffsetimm a0,a0,8` traps with `tval==0xBEEF`
AND `mepc` pointing at that control's own instruction.

## Producer settled RETROACTIVELY: the FLU, not commit_stage. `cap_type == NOT_CAP`.

`mcause 25` has two producers on this bitstream, and they are different faults with the same
number. They are separated by ONE comparison that needs `tval` only to be LATCHED, not
interpretable:

| producer | 25 means | tval holds |
|---|---|---|
| FLU, `ex_stage.sv:488` (base 24) | UNEXPECTED_OPERAND -- rs1 is `NOT_CAP` | rs1 **cursor** |
| commit_stage `:604`+`:226` (base 23) | INVALID_CAPABILITY -- PC cap's revnode invalid | the faulting **PC** |

`tval == mepc` -> commit_stage. `tval != mepc` -> FLU.

**The precondition checked first**, because "latched" is doing the work:
`git show 80843404c:core/cva6.sv:1126-1136` assigns `recent_nontrivial_mcause_log_q`,
`..._mepc_log_q` and `..._tval_log_q` **in one `if` block from one event**
(`ex_commit.valid && cause != 0 && cause != 2`). `mepc` is demonstrably correct in every boot, so
`tval` was captured from the same trap and `tval=0` is a latched value, not reset state.

    boot 21  mcause 0x99  mepc 0x82cf499c   (tval not yet in the readout)
    boot 22  mcause 0x99  mepc 0x82cf499c   tval 0
    boot 24  mcause 0x99  mepc 0x830f4aa8   tval 0
    boot 25  mcause 0x99  mepc 0x830f4a54   tval 0
    boot 26  mcause 0x99  mepc 0x82cf4a54   tval 0

`tval != mepc` in all -> **commit_stage EXCLUDED**. The producer is the FLU, and
`capstone_flu_unit.anvil:57-70` raises there **only** on `cap_type == NOT_CAP`, never consulting
the cursor.

**ESTABLISHED, with no interpretation of the tval VALUE: the reloaded `pWInfo` had
`cap_type == NOT_CAP`.** Value-loss with an intact type cannot raise 25 at the FLU. The
"code capability revoked out from under the domain" line is dead.

This also downgrades the `0xBEEF` control from blocker to nice-to-have: it would tell us whether
`ex_commit.tval` populates for FLU capability causes, which now only bears on whether the CURSOR
was also zero. The type finding does not depend on it.

### Two corrections that came out of this

* **DBAS is NOT fixed per arm.** Boot 26's arms took `0x82800000`/`0x82C00000`, not the
  `0x824/0x828/0x82C` of earlier boots. Read DBAS per arm; assuming it nearly mis-mapped boot 26
  by 4 MiB. Read correctly, boots 25 and 26 give the IDENTICAL VA `0x104A54` from different
  bases -> `sqlite3WhereCodeOneLoopStart + 0x8c`, the same source line across **three** binaries
  and six boots.
* **`lcc` selector 1 returns `cap_type - 1` in THREE bits, so it WRAPS**
  (`capstone_dyn_unit.anvil:208`, surviving generation at `capstone_dyn_unit.anvil.sv:2631-2634`):
  NOT_CAP->7, LINEAR->0, NONLIN->1, REVOKE->2, UNINIT->3, SEALED->4, SEALEDRET->5, EXIT->6.
  So the healthy baseline `type=1` is **NONLIN, not linear**, and **0 is not an empty field** --
  it is a healthy LINEAR capability. Both ends are traps for a reader.

### Type-query positive control added

`notcap=0` was an unproven instrument: two healthy reads of 1 are equally consistent with a
working query and a constant. The probe now issues `lcc` selector 1 against a plain integer and
reports `CTL(must be 7)`, riding in the returning arm at no cost. If CTL is not 7, `notcap=0`
means nothing.

### The NOT_CAP step needed one more check, and it passes

"mcause 25 at the FLU" does NOT by itself mean an operand lost its tag. Of the 21
`UNEXPECTED_OPERAND` raise sites, **seven are disjunctions whose other arm is `!= NOT_CAP`** --
firing when an operand that should be a plain integer is unexpectedly a CAPABILITY. The split
runs through one mnemonic family: `CINCOFFSETIMM` (`capstone_flu_unit.anvil:58`) is the single
condition `cap_rs1.cap_type == NOT_CAP`, but the register-form `CINCOFFSET` (`:30`) is
`(rs1 == NOT_CAP) || (rs2 != NOT_CAP)` and under that a 25 has two readings.

So the conclusion required knowing WHICH instruction faulted. Disassembled at the mepc VA in both
surviving binaries:

    adb59241 (boots 25,26)  0x104a50 ldc a4,0x0(a0) / 0x104a54 cincoffsetimm a4,a4,0xb0
    277b73f0 (boots 21,22)  0x104998 ldc a4,0x0(a0) / 0x10499c cincoffsetimm a4,a4,0xb0

Identical encoding `5b 27 07 0b`. **The IMMEDIATE form** -- and structurally it cannot be
otherwise, since it has only one capability operand and therefore no rs2 to be unexpectedly a
capability. **`cap_type == NOT_CAP` stands.**

Gap labelled rather than glossed: boot 24's binary `c7eff841` has been overwritten (the reused
bake directory again). Its mepc mapped to function+0x8C and the instruction there is
byte-identical in both surviving binaries, so it is almost certainly the same, but for that boot
it is an inference from neighbours, not a disassembly. Boots 21, 22, 25, 26 are direct.

Also recorded: **only `mcause[6:0]` survives the debug bank** (`cva6.sv:1341-1342` packs
`{trap_seen, mcause[6:0]}`), so `0x99` = seen + 25. Fine for causes 24-30; a cause >= 128 would
alias silently.

### Where the fault now stands

A capability spilled by `stc` and reloaded by `ldc` 18 instructions later comes back with
`cap_type == NOT_CAP` -- deterministically, at the same source line, across three binaries and
six boots -- and inserting instrumentation between the spill and the reload makes it stop.

Not established: which half of the instrumentation does that. The probe changes `stc`->`ldc`
distance (19 -> 44 instructions) AND pre-touches the granule with a scalar `ld`. A matched pair
separating those is the next experiment. Also, the `q_one` (passing) row is N=1, whereas the
`q_two` wedge is solid at both arm slots.

## DISTANCE IS THE VARIABLE. 600 nops of pure delay make the fault vanish.

Boot #29, control returned in 6s (boot valid). Two arms differing ONLY in how many `nop`s sit
between the `stc` that spills `pWInfo` and the `ldc` that reloads it. The pads touch NO memory --
no traffic, no granule access, nothing but fetch slots.

| pad | `stc`->`ldc` gap | outcome |
|---|---|---|
| 10 nops | 19 -> 29 instructions | **WEDGES**, `cincoffsetimm a4,a4,0xb0` at VA 0x104A9C, mcause 25 |
| 600 nops | 19 -> 619 instructions | **COMPLETES**, fully correct: `records=2 completed=1`, `M5 oom=0 malloc=241 free=241`, `ops=7` -- identical to QEMU |

So a deterministic fault that reproduced across 7 boots and 3 binaries is removed by **delay
alone**. Nothing is stored, nothing is loaded, nothing touches the granule.

Combined with the peer lane's reading of `wt_dcache_wbuffer.sv:291`
(`miss_req_o = (|dirty) && free_tx_slots`), drain is **autonomous** -- it needs neither a
subsequent store nor memory pressure, only time and a free tx slot. That makes this a
**store-to-load drain-latency window on capability data**, and specifically NOT a capacity or
pressure effect. The capacity chain predicted the opposite sign anyway: more traffic means FEWER
free tx slots and a SLOWER drain.

### Verification done before believing it

* **Gate**: the bake refused unless exactly 600 nops sat between the spill and the faulting
  instruction. It passed, and the baked `sqpad600.dom` (`068681485a217f57`) is byte-identical to
  the QEMU-validated binary.
* **Uncompressed nops** (`13 00 00 00`), so 600 are 600 fetch slots, not ~300 `c.nop`s.
* **A no-return is a wedge, not a timeout**: 5 calls x 600 nops is ~60 us against a 400 s
  timeout, bounded BEFORE the boot rather than argued after.
* **The wedging arm faults at the same instruction**: pad10 VA 0x104A9C is
  `ldc a4,0x0(a0)` / `cincoffsetimm a4,a4,0xb0`, verified in that exact binary.

### Caveats, stated not buried

* **The pair is not perfectly matched on arm slot** -- pad600 ran at arm 2, pad10 at arm 3.
  Position was independently shown irrelevant in boot #26 (the un-probed build wedges at arm 2
  AND at arm 3), so this is acceptable, but pad600 has not been run at arm 3.
* **I-cache alignment is uncontrolled.** A 600-nop pad shifts everything after it. The asymmetry
  matters: alignment cannot explain a fault SURVIVING a 619-instruction gap, so a wedging 600 arm
  would have been clean -- but since 600 COMPLETED, this result reads as "distance/time OR
  alignment", not distance/time alone. A nop-count bisection with alignment held constant is the
  way to separate them.
* **The threshold is bracketed only as 10 < T <= 600.**

### Driver text corrected

The tval verdict line still printed "the operand was NULL, therefore a SOFTWARE bug" -- a reading
that was retracted. It now says tval==0 is NO DATA until the `0xBEEF` control fires, and points at
the cause code, which gives `cap_type == NOT_CAP` for `cincoffsetimm` without needing tval at all.
A retracted claim left printing itself into every future run is worse than no message.

### CORRECTION: the pad600 result establishes DELAY-dependence, not DRAIN-latency

The entry above called the mechanism a "store-to-load drain-latency window". **That over-claims.**
What the experiment establishes is that the fault is **delay-dependent**. Other time-dependent
mechanisms scale with a 619-instruction gap just as well:

* DRAM refresh phase,
* a periodic interrupt,
* another AXI master's traffic.

`miss_req_o = (|dirty) && free_tx_slots` makes drain-latency the **leading** hypothesis and a good
one, but the pad600 arm alone cannot separate it from "something periodic got a chance to happen".

**Simulation separates them for free**, because it has no refresh and no competing masters: if the
effect survives in sim, drain latency is essentially the only candidate left; if it vanishes, the
periodic-external family moves to the front.

### Two design fixes for the next round

* **`CAPSTONE_PAD_LOOP=<n>` supersedes `CAPSTONE_PAD_NOPS`.** A bounded register-only loop has
  **constant code size and constant alignment for every n**, so a sweep varies CYCLES ONLY. That
  removes the I-cache-alignment confound outright instead of bisecting around it -- and removes a
  subtler one too: a 600-nop pad spans many cache lines, so a bisection over nop count would also
  be a bisection over footprint and the extracted threshold would blend the two. Register-only and
  volatile, so it stays a pure delay arm and does not reintroduce traffic.
* **Next round goes to SIMULATION, not the board.** The shape is now a plain `stc` / delay / `ldc`
  at a fixed address -- exactly what a directed assembly test expresses. RVFI gives the load's
  RETURNED VALUE and the store's WRITTEN DATA directly, which no board instrument can. A run is
  ~14 s, so bracketing `10 < T <= 600` is a ten-run binary search in minutes rather than five
  boots.

  **Asymmetry, stated up front:** if sim REPRODUCES, everything downstream is cheap and the trace
  is available. If sim does NOT reproduce, that is **not exoneration** -- the known fidelity gaps
  are bare M-mode vs a real capability domain, a register-resident capability vs one loaded from
  the cap table, a `.data` buffer vs a monitor-carved stack, and cache warmth. A non-reproduction
  says the gap matters, having cost minutes instead of boots.

Also still open: pad600 has never run at arm 3. Position is dead on the evidence (boot #26), but
this is the pair the whole result rests on, and it is the one asymmetry left in it -- worth one
cheap arm whenever a slot is spare.

## Simulation does NOT reproduce -- and the gap that matters is not on the standard list

The peer lane built and ran the directed test (`verif/tests/custom/capstone/stc-delay-ldc.S`,
submodule commit `2d0c26b27` on `timing-multicycle`, unpushed -- not on the allowlist). Six arms,
register-only counted loop, three instructions wide for every N so size and alignment are constant:

    N=1     reload @cyc 401    Type 2, cursor 0x10000, bounds 0x10000..0x40000
    N=4     reload @cyc 445    identical
    N=16    reload @cyc 533    identical
    N=64    reload @cyc 770    identical
    N=256   reload @cyc 1835   identical
    N=1024  reload @cyc 4945   identical
    control @cyc 4992          NOT_CAP, lcc = 7

**Zero exceptions at any delay.** Two positive controls fired, which is the only reason the clean
result is readable rather than void: the DETECTOR can report a loss (an `ldc` from a granule that
never held a capability reports NOT_CAP), and the PAD genuinely scales (~3 cycles/iteration, gaps
34/46/81/223/1065/3106) so the arms differ from one another.

**The decisive gap is that the test never creates the triggering condition.** `stc` and `ldc` to
the same address with nothing else in flight means ONE write-buffer entry draining into an idle
memory system. SQLite has a working set. If the effect needs a co-resident entry, a competing
miss, or a partially-drained buffer, this test cannot see it at ANY delay -- and would come back
exactly this clean.

**That reconciles the two results rather than leaving them in tension: delay DRAINS co-resident
entries.** "600 nops fixed it on silicon" and "the mechanism needs contention" are the same story
from opposite ends.

### The contention arm, with the REAL pattern rather than generic traffic

Nine stores sit between the subject `stc` and the subject `ldc` in the actual binary, touching
**9 distinct granules against `WtDcacheWbufDepth = 8` -- over by one**:

    stc  -> s0-0x70    SUBJECT, granule-aligned, tagged
    stc  -> s0-0x5d0   cap, tagged
    sw   -> s0-0x74    scalar 4B  (granule s0-0x80)
    stc  -> s0-0x5b0   cap, tagged
    stc  -> s0-0x90    cap, tagged
    sd   -> s0-0x98    scalar 8B  (granule s0-0xa0)
    stc  -> s0-0x5a0   cap, ctag=0   (movc rX, zero ; stc)
    sw   -> s0-0x10c   scalar 4B \_ same granule s0-0x110; may merge to ONE entry or take two
    sw   -> s0-0x110   scalar 4B /
    stc  -> s0-0x120   cap, ctag=0

Three features generic traffic would miss, each a candidate trigger: the **two ctag=0 capability
entries** (the class `wbuffer_gran_clr` keys on, needing no plain store to create); the
**9-vs-8 over-by-one**, which predicts that dropping any granule-OWNING store stops it while
dropping one of the two merging `sw`s does not; and the **mix** (6 tagged stc, 2 ctag=0 stc, 3
scalar) rather than uniform traffic.

### Two instrument traps recorded

* **Two type encodings.** `CAPPRINT` shows the RAW `cap_type` (healthy = 2, NONLIN) while `lcc`
  selector 1 shows type-minus-one (healthy = 1). Same capability, different numbers. Our probe
  uses the query form, so its healthy baseline is 1.
* **`SUCCESS (tohost = 0)` is a STRING LITERAL** in the harness success branch
  (`corev_apu/tb/ariane_tb.cpp:397`), printed by every passing test and carrying no value. Read
  the pass/fail word, never that number. Same family as `tval=0`: a number that renders like data
  and carries none.
