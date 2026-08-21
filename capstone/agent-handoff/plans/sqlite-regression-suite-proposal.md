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
Any board-only divergence must be read against the **8 of 16 legs of the write-buffer
residual live on the flashed `caplifive_s07fix.bit`**: a load there can return a
dereferenceable capability over memory the program already scrubbed. That is a candidate
cause before the runner is.

**No reflash is proposed for this.** S-10 costs −16.4 ns WNS against a design −10.629 and
buys SQLite nothing; S-10b is dead on a DRC combinational loop.
