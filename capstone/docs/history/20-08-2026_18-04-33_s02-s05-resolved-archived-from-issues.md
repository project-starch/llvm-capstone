# ARCHIVED — S-02 and S-05, resolved 2026-08-20

Moved out of `ref/ISSUES.md` on 2026-08-20 because both symptoms are gone and the entries were
the two longest open items in a file agents are told to read before re-investigating anything.
Nothing here is deleted; this is the full investigation trail as it stood at archival.

**Why they were closed.** The workload that passed 3/3 on `caplifive_s07fix.bit` IS their failure
site (`benchmarks/sqlite/sqlite_capstone_domain.c:2014-2027`):

    sqlite3_initialize();                                            <- S-02's wedge site
    sqlite3_open(":memory:", &db);
    CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);  <- S-05's exact statement
    INSERT INTO items VALUES('alpha',11),('beta',22),('gamma',33);

The run printed `alpha 11`, `beta 22`, `gamma 33`, the loop's own `row != 3` count assert passed,
then `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and `rc=0`. Direct, not inferred. Re-verification
commit: `c7def143e473`.

**Attribution, kept separate from observation.** S-05's recorded root cause is S-06, fixed in
silicon and verified 2026-08-14 — consistent. **S-02's origin was never attributed and still is
not**; plausibly S-06 and/or S-07, both now fixed, but that is NOT demonstrated.

**What the re-verification does NOT establish.** S-02's own `rn1`/`rn2` `RUNSTOP` probe builds
were not re-run, so the symptom is gone from the workload that defines it rather than from those
specific images. n=3. Taken on a bitstream that misses setup (WNS -10.629; the S-07 fix is
exonerated as the cause and the failing cone is domain-switch machinery).

**If either symptom reappears, start here** — the ruled-out lists below cost many board sessions
and should not be re-derived.

---

## S-02 — SQLite wedges inside `sqlite3_initialize()` in a pure-capability domain · `RESOLVED as observed 2026-08-20 -- the workload now runs end to end; origin never attributed`

> **RE-VERIFIED 2026-08-20 — NO LONGER MANIFESTS. Directly, not by inference.**
> `XU` ran the base workload 3/3 on `caplifive_s07fix.bit`, control green, and that workload is
> literally the failure site of both issues (`sqlite_capstone_domain.c:2014-2027`):
>
>     sqlite3_initialize();                                              <- S-02's wedge site
>     sqlite3_open(":memory:", &db);
>     CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);    <- S-05's exact statement
>     INSERT INTO items VALUES('alpha',11),('beta',22),('gamma',33);
>
> The run emitted `row name=alpha value=11`, `beta 22`, `gamma 33`, with the loop's own
> `row != 3` count assert passing, then `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and `rc=0`.
>
> **Attribution.** S-05's recorded root cause is **S-06**, fixed in silicon and verified
> 2026-08-14 — consistent, and this is the workload-level confirmation that fix was waiting for.
> S-02's origin was never attributed and still is not; it is plausibly S-06 and/or S-07, both now
> fixed, but that is **not demonstrated** and should not be written down as though it were.
>
> **What this does NOT establish:** S-02's own `rn1`/`rn2` `RUNSTOP` probe builds were not re-run,
> so this says the symptom is gone from the workload that defines it, not that those specific
> images would now pass. n=3. And like every silicon number this week it was taken on a bitstream
> that misses setup (WNS -10.629) — the S-07 fix is exonerated as the cause of that, and the
> failing cone is domain-switch machinery, but the caveat is recorded project-wide.


**The project's headline blocker.** SQLite must RUN on the FPGA; it has never produced a row there.

**Status 2026-08-09: LOCALIZED to `sqlite3_initialize()`, origin unattributed.** Two boots on
`caplifive_65536_r18_fix.bit`, control green in both:

| build | returns after | result |
|---|---|---|
| `rn1` (`RUNSTOP=1`) | `sqlite3_config(SQLITE_CONFIG_HEAP)` | **RETURNED `rc=0` in 4 s** |
| `rn2` (`RUNSTOP=2`) | `sqlite3_initialize()` | **WEDGED**, entered, no return in 240 s |

Both images are byte-size and cap-init identical to a build that entered 2/2 (`stc=558`,
1551336 bytes), differing only by one inserted early return — so size, layout and cap-init load are
held constant. `rn1` returning is the positive control.

**Why S and not R:** it is NOT established as a hardware defect. It survives on silicon that has
R-18/R-19 fixed, but nothing yet rules out our compiler or our software. Do not hand it to the
hardware side as an RTL issue, and do not open an `fpga-repros/` package for it, until the origin
is demonstrated.

### Ruled out, each measured on FIXED silicon with a valid control

* **R-18 / R-19** — the full domain still wedges on the fixed bitstream (it enters cleanly first).
* **The builtin-function array** — `BUILTIN_LIMIT=0` (loop body never executes) still wedges.
* **The whole interior of `sqlite3RegisterBuiltinFunctions`** — `rs0`, an early return at its FIRST
  instruction, still wedges. **This REFUTES the 2026-08-06 localization** recorded further down
  this file; that probe was reading a downstream symptom.
* Earlier and on the old bitstream: `auipc` asymmetry, PCC/code-window overrun, stale-metadata
  leakage through integer ops, malformed cap-init descriptors, and cap-init itself (it completes —
  the `0x9E11` precall sentinel is emitted after `RUN_CAP_INIT`).

### Confirmed, still true on fixed silicon

* **2026-07-31 "stage 2 wedges (after `sqlite3_initialize`)"** — reproduced, now on repaired hardware.

### Vehicle note, hard-won

Heavyweight `CAPSTONE_SQLITE_STAGE` builds **do not ENTER** on this bitstream (0/4). Lightweight
early-return builds enter 4/4. Use `RUNSTOP` / `INITSTOP` / `REGBUILTIN_STOP`, never the staged
block. Two explanations for the staged stall were raised and **refuted**: layout (a REDRAW at
`CAPSTONE_TEXT_PAD=4096` still stalled) and the `holder[580]` default (fixing it cut cap-init
1257→608 `stc`; still stalled). The holder fix is kept — 580 unused capability leaves per staged
build is a real defect — but it is not the cause. **That stall is itself unexplained and may deserve
its own S-number if it blocks anything else.**

### 2026-08-09 SPLIT INTO TWO SITES. The call + prologue are PROVEN INNOCENT.

The decisive pair. Identical gutted `sqlite3_initialize` (body replaced by a bare return); the
ONLY difference is what `run_sqlite` does next:

| build | gutted `initialize` returns | `run_sqlite` then | result |
|---|---|---|---|
| `nk` | non-zero | takes `return fail("initialize", rc, 0);` | **WEDGED** |
| `n8` | zero | falls through to a direct `return` | **RETURNED, 4 s** |

Control green in both. Therefore:

1. **The call into `sqlite3_initialize` and its prologue WORK.** `n8` executes
   `auipc`/`addi`/`jalr` → `sqlite3_initialize` → prologue → return, and comes back. The
   "eight instruction" localization recorded earlier is **withdrawn** — those instructions are fine.
2. **`fail()` WEDGES.** It is the only thing on `nk`'s path that `n8` does not execute.
3. **`sqlite3_initialize`'s BODY also wedges, independently** — `rn2` runs the full body and
   returns DIRECTLY (no `fail()`), and wedges. So there are **two separate wedge sites**, and any
   arm that reaches `fail()` measures site 2, not site 1.

**This retracts `nz0`, `nk`, `F0`, `in1`, `in2`, `in4`** as evidence about `initialize`: every one
of them returned non-zero (or fell through) into `fail()`. Only `rn1` (returns), `rn2` (wedges) and
`n8` (returns) are clean.

### Site A — ROOT CAUSE FOUND (2026-08-09): the UNGUARDED `CAPSTONE_DELIN(payload)` in `output_text()`

    FS1  return at fail() entry, before any output_text   RETURNED (4 s)
    FS2  return after output_text("SQLITE ERROR stage=")  WEDGED

Control green in both. So the FIRST `output_text()` call wedges.

`output_text()` contains **two** `delin`s. Only one is guarded:

    #ifndef CAPSTONE_GP_CAPTABLE_ABI
      CAPSTONE_DELIN(text);      <- guarded, compiled OUT on the silicon ABI
    #endif
      ...
      CAPSTONE_DELIN(payload);   <- UNGUARDED, always compiled in

The compiled artifact confirms it: `output_text @ 0x13addc`, 71 instructions, **`delin` x1** — the
`payload` one, since the `text` one is compiled out.

**Mechanism, quoted from the guard's own comment** (which was written for `text` and applies
identically to `payload`): *"the RTL's DELIN accepts CAP_TYPE_LINEAR only and raises
UNEXPECTED_CAP_TYPE otherwise; our QEMU helper_csdelin returns early instead, which is why this
never showed up under emulation."* Illegal capability ops wedge rather than trap on this silicon
(R-5). **This is why SQLite is green under QEMU and dead on hardware.**

**NOT YET CONFIRMED — the one experiment that closes it:** rebuild with `CAPSTONE_DELIN(payload)`
removed (or guarded the same way) and re-run `FS2`. If it returns, Site A is proven and fixed in the
same stroke. That build is trivial; do it first.

**Origin: OURS, not the hardware.** If confirmed this is a `C-n`, not an `R-n` — domain code
delin'ing a capability that is already non-linear. The guard for `text` was added when C-13 hit the
identical hazard in the entry glue; `payload` was missed.

### Site A background — `fail()`, how it was reached

    static int fail(const char *stage, int rc, sqlite3 *db) {
      output_text("SQLITE ERROR stage="); output_text(stage); ...

`stage` is a **string literal** — a capability — and `output_text` writes into the shared payload
region. That is a tiny operation, and unlike the rest of SQLite it is small enough to rebuild
inside the 13 KB fdreg model, which would take Site A off the 1.5 MB image entirely.

**Next cut for Site A:** clamp inside `fail()` — before the first `output_text`, between the
literal and the `stage` argument, and after — to separate "writing to the shared region" from
"dereferencing a string-literal capability" from "dereferencing a *passed* string capability".

### Site A — FIXED AND CONFIRMED, but SQLite still does not run (2026-08-09)

Full unclamped SQLite, rebuilt with the guard, `output_text` verified `delin x0` in the
artifact: **still WEDGED**, control green (2 s). So Site A was real and is fixed, and **Site B
is a genuinely independent second defect** — not the same one reached by another path.

That is the honest state: the root cause of Site A is proven (`FS2` wedged / `fx2b` returned,
one variable), and it did not unblock the benchmark.


---

## S-05 — SQLite fails building the schema · `RESOLVED as observed 2026-08-20 -- the exact failing CREATE now succeeds; root cause S-06 is fixed`

> **RE-VERIFIED 2026-08-20 — NO LONGER MANIFESTS. Directly, not by inference.**
> `XU` ran the base workload 3/3 on `caplifive_s07fix.bit`, control green, and that workload is
> literally the failure site of both issues (`sqlite_capstone_domain.c:2014-2027`):
>
>     sqlite3_initialize();                                              <- S-02's wedge site
>     sqlite3_open(":memory:", &db);
>     CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);    <- S-05's exact statement
>     INSERT INTO items VALUES('alpha',11),('beta',22),('gamma',33);
>
> The run emitted `row name=alpha value=11`, `beta 22`, `gamma 33`, with the loop's own
> `row != 3` count assert passing, then `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and `rc=0`.
>
> **Attribution.** S-05's recorded root cause is **S-06**, fixed in silicon and verified
> 2026-08-14 — consistent, and this is the workload-level confirmation that fix was waiting for.
> S-02's origin was never attributed and still is not; it is plausibly S-06 and/or S-07, both now
> fixed, but that is **not demonstrated** and should not be written down as though it were.
>
> **What this does NOT establish:** S-02's own `rn1`/`rn2` `RUNSTOP` probe builds were not re-run,
> so this says the symptom is gone from the workload that defines it, not that those specific
> images would now pass. n=3. And like every silicon number this week it was taken on a bitstream
> that misses setup (WNS -10.629) — the S-07 fix is exonerated as the cause of that, and the
> failing cone is domain-switch machinery, but the caveat is recorded project-wide.


### 2026-08-10 — ROOT CAUSE: untagged `ldc`/`stc` loses the high 64 bits (new issue S-06)

S-05 is a consequence of **S-06** below: a 16-byte `ldc`/`stc` block copy of PLAIN data keeps
only the low 8 bytes. SQLite's schema text is copied through exactly that loop, so half of every
16-byte chunk is destroyed, which is why the schema will not re-parse. The error message is the
tell and it was in plain sight: `malforme` is the first **8** bytes of "malformed database schema
(items)", with byte 8 gone so the string ends there. It stayed 8 bytes even when emitted into an
empty output region, so it was never output truncation.

It also explains why a SHORT `CREATE TABLE t(a INTEGER, b TEXT)` **succeeds** on silicon
(stage 168 = rc 0, reproduced twice) while the workload's longer
`CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);` fails: the damage is
length-dependent.

**Status: S-05 needs the S-06 silicon fix.** The software workaround is correct at the primitive
level and still wedges the workload — see S-06.

### 2026-08-10 — the `-O0` strlen theory is REFUTED, and this is now a clean single-defect measurement

This section previously named the `-O0` `strlen` defect as the live suspect, on the grounds that
the S-04 workaround reintroduced it. **That is refuted.** With `BEEBS_MEMCPY_OPTNONE=1` the build
carries the correct `-O1` `strlen` (zero `ldc` in its loop, verified in the linked domain) AND a
working `memcpy` (S-04 board-cleared, `0x74` -> `0x70`), and the full run on `caplifive_r20.bit`
still ends at:

    SQ: G/enter ... SQ: H/return
    SQLITE ERROR stage=create rc=11 message=malforme

Identical symptom, so it was never the `strlen` form. Control green in the same boot (`k800` = 4);
the domain ENTERS and RETURNS, so this is a real result and not an entry stall.

**What this run establishes positively — SQLite gets further on silicon than ever before:**
`sqlite3_config` -> `sqlite3_initialize` -> `sqlite3_open(:memory:)` **succeeds** -> statements
prepare -> `sqlite3_step` runs -> failure only at CREATE TABLE, reading back a schema written
moments earlier on an in-memory database.

**Next, and no longer confounded:** the remaining wrong-data shape is "something written is read
back wrong", the same family as S-04 but on a different path. Since S-04 turned out to be a store
that does not commit rather than anything about SQLite, the first question is whether the schema
write is another non-committing store — bisect the CREATE path the way stages 160-166 bisected
`openDatabase`, reading the written bytes straight back.

### Earlier history (symptom moved twice)

Originally reached only with the S-04 memcpy workaround (`SQLITE_SUPPORT_OPT_LEVEL=-O0`). The
symptom moved when the R-20 bitstream landed, which is itself evidence the two were entangled:

| bitstream | failure |
|---|---|
| `caplifive_65536_r18_fix.bit` | `stage=step rc=21` (SQLITE_MISUSE) |
| `caplifive_r20.bit` | **`stage=create rc=11`** (SQLITE_CORRUPT, "malformed database schema") |

SQLite now configures, initialises, OPENS the database, prepares and steps statements, and fails
while creating a table. A corrupt schema on a `:memory:` database created moments earlier means
schema text is read back wrong -- the same shape as S-04, and a live suspect for the documented
`-O0` `strlen` defect that this very workaround reintroduces.

**First task attempted, and it FAILED for an instructive reason.** The plan was `-O1` support
objects plus a byte-loop `memcpy`, so neither known defect would be in play. A build knob
`BEEBS_MEMCPY_BYTES_ONLY` was added (default OFF) to skip memcpy's aligned capability-copy path.
The result on `caplifive_r20.bit` is WORSE: the domain ENTERS and WEDGES (mcause 25, commit pc
the `0x2` junk sentinel), where the `-O1` build merely returned an error.

The reason is structural, not a bug in the knob: `memcpy`'s middle loop copies via `void *`, and
`sizeof(void *)` is **16** on this target -- one capability. That path exists precisely so that
copying a struct containing pointers PRESERVES CAPABILITY TAGS. A byte loop cannot: it strips
every tag, so SQLite ends up dereferencing untagged pointers and the core dies. **A byte-wise
memcpy is not a valid workaround on a capability machine and never can be.** The knob is kept,
default OFF, documented with this result so nobody re-derives it.

So S-04 cannot be worked around in `memcpy`'s copy strategy. The remaining options are: fix the
silicon defect that makes the 7-byte copy vanish; or find why the compiled `-O1` byte tail-loop
is skipped, which is a codegen question and may be fixable in the compiler.

**Still true:** running with `-O0` support objects means the documented `-O0` `strlen` defect is
in play at the same time, so `stage=create rc=11` is measured under two known defects, not one.

Reached only with the S-04 workaround applied. `sqlite3_open` succeeds, statements prepare, and
`sqlite3_step` returns 21 with an empty message. First thing to establish: whether this is the
documented `-O0` `strlen` defect (see the S-04 cost note above) rather than a new one -- i.e.
re-run with `-O1` support objects plus a byte-loop memcpy ONLY, so neither known defect is in
play at once.



The blocker S-03 was hiding. The domain enters, initialises, reaches `sqlite3_open` and gets
`rc=7` (`SQLITE_NOMEM`). No wedge, no trap -- a clean error return, so the core is healthy and
this is debuggable normally.

**Not simply the heap size.** `SQLITE_HEAP_SIZE` was raised from the 256 KiB default to 448 KiB
(`domdata-budget.py`: storage 524032, stack still 124304, "fits") and the result is byte-identical:
`rc=7` at the same stage. So either the heap is not being applied
(`sqlite3_config(SQLITE_CONFIG_HEAP)`, see `build-sqlite-silicon.sh:533`), or something other than
raw heap size returns NOMEM.

**What is already ruled out (2026-08-10), each measured on silicon:**

| # | ruled out | evidence |
|---|---|---|
| 1 | a software config bug | the SAME build PASSES under QEMU (`__CAPSTONE_SQLITE_SILICON_PASSED__`, all five markers). Board-only divergence |
| 2 | heap size | 256 KiB -> 448 KiB gives a byte-identical `rc=7` |
| 3 | a missing allocator | memsys5 is compiled in and every symbol linked, so `CONFIG_HEAP` is honoured |
| 4 | unwritable heap | staged bounds probes returned rc=0 across the full range (but see the caveat below) |
| 5 | an early-stage failure | stage 0 -> 0, stage 2 -> 0: entry, `CONFIG_HEAP` and `sqlite3_initialize()` all succeed |
| 6 | the allocator being broken | stage 11: `sqlite3_malloc(64/4096/65536)` ALL succeed. stage 13: `sqlite3_malloc(sizeof(sqlite3))`, `sqlite3_malloc(700)`, `sqlite3_malloc(120000)` and **`sqlite3MallocZero(sizeof(sqlite3))` -- the exact call openDatabase makes -- all succeed**, and the memset sticks |
| 7 | the second `sqlite3_initialize()` | stage 14: both calls return SQLITE_OK and `sqlite3GlobalConfig.isInit` persists across them, so openDatabase's own autoinit is not the early return |
| 8 | the lookaside allocation | stage 15: `SQLITE_CONFIG_LOOKASIDE` disabled, `sqlite3_open` still returns `rc=7` |

**A RETRACTED INFERENCE.** Stage 12 was read as "db == NULL, therefore the handle allocation
failed". That is WRONG: `opendb_out` (amalgamation ~line 191097) does
`if( (rc&0xff)==SQLITE_NOMEM ){ sqlite3_close(db); db = 0; }`, so the handle is nulled for a
NOMEM raised ANYWHERE in openDatabase. `db == NULL` localises nothing.

**Where that leaves it.** Every allocation openDatabase needs succeeds when called directly, its
autoinit succeeds, and lookaside is not involved -- yet it returns NOMEM. So something inside
openDatabase calls `sqlite3OomFault(db)` (or returns NOMEM) WITHOUT a real allocation failure.
That is the shape to attack, and it is a different animal from "out of memory".

**LIVE SUSPECT: S-04 may be R-20 again, through `MOVC`.** The compiler workaround only keeps
x10 out of an `stc` BASE. But `issue_read_operands.sv:568` drops x10's claim for any
non-CAPENTER capability op with `rs1 == x10` and `rd != x10`, and the wrong value comes from the
rs1-cursor mux gated by `check_fwd_rs1` = {SPLIT, MOVC, CJALR, CCSRRW, STC}. Measured with
`fpga-repros/R20-.../sim/scan-fwd.py` on the workaround build, at the tightest adjacency window:

| window | before | with workaround |
|---|---|---|
| 1 | 4481 (MOVC 2914, STC 1567) | **1051 — all MOVC** |
| 4 | 6006 | 2396 — all MOVC |

So ~1051 `MOVC`-shaped sites survive, and `movc a1, a0` is the ABI return-value copy, which no
register allocation can remove -- a0 IS the return register. If S-04 is one of these, the real
fix is the RTL change (`capstone-ariane` branch `r20-fix`) in a bitstream, not more compiler
work. **This is a hypothesis, not a finding**: nothing yet ties the S-04 NOMEM to a specific one
of those sites.

**ROOT CAUSE 2026-08-10: a 7-byte `memcpy` into freshly allocated memory leaves the destination
ALL ZERO.** Not a hash bug, not an allocator bug, not a phantom -- real data corruption that
SQLite converts into `SQLITE_NOMEM`. Four staged board runs, each a strict narrowing:

| stage | result | what it establishes |
|---|---|---|
| 160 | `0x15` | first tripping step is **5**, `createCollation(BINARY, UTF16BE)`; `mallocFailed` set, `errCode` clean, step rc not NOMEM |
| 161 | `0x11` | `sqlite3HashFind("BINARY")` returns 0 although the table holds 1 element; the SOURCE string is intact (`strlen`=6, `[0]=='B'`) |
| 162 | `0x5b` | hash mismatch AND compare mismatch, `strHash` deterministic across two calls, and **the stored key COPY differs byte-wise** |
| 163 | `0xbf` | **the stored key is ENTIRELY ZERO** -- all six letters differ, the NUL "matches" only because zero equals zero |

The chain: `findCollSeqEntry` (132484-132494) does `sqlite3DbMallocZero(...)` -- which zeroes the
block -- then `memcpy(pColl[0].zName, zName, nName)`. On silicon the destination is still all
zeros afterwards, so `sqlite3HashFind` legitimately cannot match the key, `findCollSeqEntry` takes
the create path a second time, `sqlite3HashInsert` finds the key present and returns non-zero, and
`sqlite3OomFault` fires (132502). SQLite's `assert(pDel==0 || pDel==pColl)` that would have caught
this is compiled out under NDEBUG, which is why it surfaced as NOMEM.

**Source is exonerated, destination or the copy is not.** Stage 161 read `sqlite3StrBINARY`
correctly. So either `pColl[0].zName = (char*)&pColl[3]` computes a wrong destination, or the
small `memcpy` writes are lost.

**This is the R-18 SHAPE -- a store silently zeroed -- on a bitstream whose name claims R-18 is
fixed.** Do not assume it is R-18; assume nothing. But it is the first thing to check.

**Next step, one boot.** Replace the `memcpy` with an explicit byte loop in a stage and re-read
the key: if the loop's bytes stick, `memcpy` is implicated; if they also vanish, the destination
pointer is wrong. Then check `pColl[0].zName == (char*)&pColl[3]` directly.

**Prediction for the new bitstream (recorded before the run):** S-04 is NOT the `stc` shape, so
the R-20 fix alone should NOT clear it. If S-04 disappears on the new bitstream anyway, that says
the two are related after all and this entry needs revisiting.

**(superseded) Earlier localisation -- step 5 via a hash lookup that disagrees with itself.** Stage 160 re-walks openDatabase step by step on its own handle, checking
`db->mallocFailed` and `db->errCode` after each. Result `0x15`: first tripping step **5**,
`mallocFailed` SET, `errCode` clean, and the step's own rc NOT NOMEM.

`createCollation` -> `sqlite3FindCollSeq(create=1)` -> `findCollSeqEntry` (132474-132504):

```c
pColl = sqlite3HashFind(&db->aCollSeq, zName);          /* 132480 */
if( 0==pColl && create ){
   pColl = sqlite3DbMallocZero(...);                    /* succeeded */
   pDel = sqlite3HashInsert(&db->aCollSeq, ..., pColl); /* 132494 */
   assert( pDel==0 || pDel==pColl );                    /* COMPILED OUT under NDEBUG */
   if( pDel!=0 ){ sqlite3OomFault(db); ... }            /* 132502 -> mallocFailed = 1 */
}
```

Step 4 inserted `"BINARY"` (UTF8). Step 5 asks for the SAME key. Reaching `sqlite3OomFault` here
requires `sqlite3HashFind` to report the key ABSENT while `sqlite3HashInsert` finds it PRESENT --
**the two disagree about the same key in the same table**, with no allocator involved. That is a
read returning a stale value, and SQLite silently converts it into a phantom `SQLITE_NOMEM`
because the assert covering exactly that case is compiled out.

**RETRACTED 2026-08-10: the MOVC route is NOT the explanation.** This was recorded as "R-20 again
through the ~1051 MOVC sites the workaround leaves". RTL simulation refutes it with a matched
pair: the identical structure corrupts with `stc` and does NOT corrupt with `movc`, on the
UNPATCHED RTL where the `stc` arms fail, positive control firing in the same run. Of the
value-corrupting set only `stc` and `cjalr` are long-latency, and the SQLite domain contains no
`cjalr`, `split` or `ccsrrw` sites of this shape -- so the workaround is probably complete and
this NOMEM is probably NOT R-20.

The hash-lookup disagreement is still real and still unexplained; it just should not be assumed
to be R-20. It is NOT proof of anything: nothing ties the failing hash walk to a specific
instruction.

**PREDICTIVE TEST, recorded before the run.** On a bitstream carrying the RTL fix, with NO
compiler change: stage 160 returns `0x00` and `sqlite3_open` succeeds IF S-04 is R-20. Given the
MOVC retraction above, the honest expectation is now the opposite -- **S-04 probably survives the
new bitstream**. Either result is informative and the prediction is recorded before the run.

**Next step (only if the bitstream does NOT clear it).** Bisect openDatabase itself, the same way S-03 was bisected. The domain is
`#include`d into the amalgamation TU, so its internals are callable: add a stage that walks
openDatabase's sequence (handle alloc -> `setupLookaside` -> `sqlite3ParseUri` ->
`createCollation` x3 -> `sqlite3RegisterPerConnectionBuiltinFunctions` -> `sqlite3BtreeOpen`) and
returns the index of the first step after which `db->mallocFailed` is set. That converts "NOMEM
from somewhere" into a line number in one boot.

**Caveat on reused evidence.** Row 4's bounds probes (stages 4/5) were measured on 2026-07-31 with
the PRE-workaround compiler and a bitstream that still had R-20. Re-run them before leaning on
that row.

Diagnostic stages 11-15 are committed in `sqlite_capstone_domain.c` and selected at run time
(`SQLITE_STAGE_DOMS=".../s8.dom:15"`), so none of this has to be rebuilt from scratch.

