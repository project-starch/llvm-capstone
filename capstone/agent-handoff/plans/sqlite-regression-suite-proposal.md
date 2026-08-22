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
