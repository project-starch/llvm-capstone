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

## ~~THE CENTRAL RISK~~ — B0b RAN, AND THE RISK IS VOID

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

**So the SQLite that passed 3/3 on silicon is a feature-complete amalgamation build**, and an
upstream suite would not be bounded by an omission set. That is a materially better starting
position than this proposal originally assumed. (The 15th flag, `SQLITE_OMIT_SELECT`, was
commentary about SQLite's own source — not ours, as suspected.)

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
