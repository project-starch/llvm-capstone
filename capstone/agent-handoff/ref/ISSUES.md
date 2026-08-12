# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure,
**S-n** = **unattributed** — reproducible, but origin NOT yet established (may be RTL, compiler or
software). An S-n is promoted to R-n/C-n only when the origin is demonstrated, never on suspicion.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-08-09.

---

## S-02 — SQLite wedges inside `sqlite3_initialize()` in a pure-capability domain · `OPEN`

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

## S-03 — SQLite wedges inside `sqlite3_initialize()` · `RESOLVED 2026-08-10 -- root cause R-20; cleared on silicon by the R-20 compiler workaround`

**S-03 is gone from the board.** Root cause is **R-20** (a capability store loses its x10 clobber
claim, so a later reader of x10 gets the store's base address instead of the loaded value). With
the R-20 compiler workaround applied (commit `30c275b5d781`, keeps a0/x10 out of a capability
store's base register), the full unclamped SQLite domain now runs the complete marker sequence
`A/dom -> B/mkregion1 -> C/mkregion2 -> D/mapped -> E/share1 -> F/share2 -> G/enter -> H/return`
and RETURNS, where every previous build wedged the core. Measured 2026-08-10, control k800 green
in the same boot.

It no longer wedges; it reports a clean SQLite error instead:

```
SQLITE ERROR stage=open rc=7            (rc=7 = SQLITE_NOMEM)
```

That is a DIFFERENT blocker, tracked as **S-04** below. Everything under S-03 from here down is
the investigation trail that led to R-20, kept because several of its models were refuted and
re-deriving them would waste board time.

## S-04 — SQLite returns SQLITE_NOMEM from `sqlite3_open` on silicon · `WORKED AROUND 2026-08-10 with NO second defect in play: memcpy alone at -O0 via BEEBS_MEMCPY_OPTNONE. Board-confirmed by a matched pair. Underlying silicon defect still OPEN.`

### RESOLUTION 2026-08-10 — per-FUNCTION opt scoping, confirmed by a matched pair on silicon

The two string primitives SQLite depends on are broken at OPPOSITE optimisation levels, and they
live in the same file, so one file-wide `-O` flag forced a choice between them:

| | memcpy | strlen |
|---|---|---|
| `-O0` | works | **WRONG on silicon** (re-loads the string cap with `ldc` every iteration; returned 15, then 26, then hung, vs 36 on QEMU) |
| `-O1` | **WRONG on silicon** (S-04) | works (zero `ldc` in the loop) |

`SQLITE_SUPPORT_OPT_LEVEL=-O0` was therefore a trade, not a fix, and every result taken under it
was taken under a second known defect. `BEEBS_MEMCPY_OPTNONE=1` (default on, set by
`build-sqlite-silicon.sh`) applies `__attribute__((optnone))` to **memcpy only**, so the build has
NEITHER defect for the first time.

**Verified on the artifact, not assumed.** In the linked domain, `strlen`/`strcmp`/`strcpy`/
`memset`/`memmove` are byte-identical to the plain `-O1` build; only `memcpy` changes, to the form
that spills the destination capability at entry and reloads it with `ldc` before each `sb`.

**Board evidence — a matched pair in ONE boot, control green (`k800` = 4):**

| arm | build | stage 164 | bit 2 (`memcpy does not stick`) |
|---|---|---|---|
| `sm0.dom:164` | `-O1` memcpy | `0x74` | **SET** |
| `sm.dom:164` | `optnone` memcpy | `0x70` | **CLEAR** |

The two images differ in **memcpy and nothing else** (per-symbol encoding comparison). One bit
apart, and it is exactly the memcpy bit. The failing arm is also the positive control: it proves
the probe can report failure, so the `0x70` is a real negative and not a dead test. `0x70`
reproduced across two separate boots. `sm.dom:163` went `0xbf` -> `0x00` (the stored key is no
longer zero) and `sm.dom:160` went `0x15` -> `0x00` (no step of `openDatabase` trips).

**This is a WORKAROUND, not a fix — the -O1 code is CORRECT.** Disassembly of the linked domain
(`memcpy` at `0x14ca1c`) shows that for the failing case (n=7, dst and src both 16-byte aligned)
the `-O1` form branches over the head loop (`beqz a5`), does not enter the capability loop
(`bgeu a2, a4` with a4=16 > n=7), and issues seven `sb` stores from the tail loop at `+0x9c`. The
stores are ISSUED and do not stick. **So "the `-O1` byte tail-loop is skipped" is REFUTED** — that
was recorded here as the remaining codegen avenue and it was based on a false premise. The only
difference between the working and failing forms is which capability register holds the
destination base: `-O1` uses the incoming argument `a0` directly, the working form reloads it from
a stack slot. The underlying silicon defect is untouched and still needs reporting.



**The cause is `memcpy`, not the allocator.** A 7-byte `memcpy` into freshly zeroed memory leaves
the destination ALL ZERO, so `findCollSeqEntry`'s key copy never lands, `sqlite3HashFind` cannot
match, the create path runs twice, `sqlite3HashInsert` reports the key present, and
`sqlite3OomFault` fires (132502). The `assert(pDel==0 || pDel==pColl)` that would have caught it
is compiled out under NDEBUG, which is why real data loss surfaced as `SQLITE_NOMEM`.

Narrowed by staged board runs, each one variable:

| stage | result | establishes |
|---|---|---|
| 160 | `0x15` | first failing step is 5, `createCollation(BINARY, UTF16BE)` |
| 161 | `0x11` | `HashFind` misses although the table holds 1 element; SOURCE string intact |
| 162 | `0x5b` | hash AND compare mismatch, `strHash` deterministic, stored COPY differs |
| 163 | `0xbf` | the stored key is ENTIRELY ZERO |
| 164 | `0x74` | destination pointer CORRECT; an explicit byte LOOP writes all 7 bytes; **`memcpy` does not** |
| 165 | `0x00` | both `dest & 15` and `src & 15` are 0 -- the aligned path |
| 166 | `0x05` | poison bytes 7..15 UNTOUCHED -- memcpy stays in bounds; its stores simply do not stick |

Stage 166 also **refuted** the intermediate theory that memcpy wrongly took its 16-byte
`ldc`/`stc` capability block path: it does not overrun.

**Workaround, one env var, no code change:** `SQLITE_SUPPORT_OPT_LEVEL=-O0`. The support objects
(the string primitives, `build-sqlite-silicon.sh:739`) default to `-O1`; at `-O0` memcpy becomes
142 instructions instead of 49 and the copy sticks. With it, the full unclamped SQLite on silicon
progresses from `stage=open rc=7` to **`stage=step rc=21`** -- it now configures, initialises,
OPENS the database and prepares statements.

**THIS WORKAROUND HAS A KNOWN COST -- do not adopt it silently.** `build-sqlite-silicon.sh:710-733`
records that the support objects were moved TO `-O1` because at `-O0` `strlen` re-loads its string
capability from a stack slot every iteration and on silicon sporadically returns the wrong length
(stage 13 returned 15, then 26, then hung, where QEMU returns 36 every time). So `-O0` trades one
silicon defect for another, and the new `rc=21` at `sqlite3_step` may BE that defect resurfacing.
Treat `-O1`-vs-`-O0` as two different broken configurations, not as a fix.

## S-05 — SQLite fails building the schema · `OPEN -- SQLITE_CORRUPT at CREATE, now measured with NO other known defect in play`

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

## S-06 — an untagged 128-bit `ldc`/`stc` round trip loses the HIGH 64 bits · `OPEN — SILICON DEFECT, root-caused in RTL, needs a hardware fix`

> **SILICON CONFIRMATION 2026-08-12 on `caplifive_s06.bit`: the LCC-query repair WORKS on
> hardware.** The RTL enabler (`capstone-ariane` `fpga-testing-dev-s06`, one line making LCC's
> type query total) is now flashed, and rung `s06lcc` returned **171** with control `k800` = 4 in
> the same boot. The verdict is digit-encoded so each half is independently meaningful:
>
> | digit | question | expected | got |
> |---|---|---|---|
> | hundreds | type query on a REAL NONLIN capability | 1 | **1** |
> | tens | type query on PLAIN untagged data | 7 | **7** |
> | units | the 16-byte copy came back intact | 1 | **1** |
>
> The capability arm is not decoration: a query answering 7 unconditionally would pass a
> plain-data-only test. And the run doubles as a bitstream identity check — on `caplifive_r20.bit`
> the plain-data query RAISES, and a capability fault inside a domain wedges rather than traps, so
> an old bitstream produces NO RETURN rather than a wrong number.
>
> **So S-06 is repairable in software, demonstrated end-to-end on silicon**: read both halves
> plainly first, ask the type, then write the destination once. QEMU verified the same rung 4/4
> before the boot (first invocation failed on a build race, then 4 consecutive passes).
>
> **WHAT THIS DOES NOT SHOW, and the distinction matters.** The rung copies 32 bytes on a hot
> line. It does NOT show the repair survives at SQLite scale, and it does not touch the separate
> wedge: in the same boot, `sqA` (baseline) still returned `rc=11` and `sqB` (arm E) still WEDGED
> with mcause 25, exactly as on the old bitstream — expected, since this bitstream changes only
> LCC. Whether the query-based repair avoids that wedge is UNTESTED and cannot be assumed: every
> construction that repairs the data has so far wedged, and the experiment that would separate
> "the store pattern is toxic" from "correct data reaches a second fault" (see the rung design
> below) has still not been run.

**This is the blocker behind S-05, and it affects every capability-grained copy of plain data.**

> **UPDATE 2026-08-11 — a SECOND signature, and the baseline is now measured rather than inferred.**
> Both cache write-width signals are gated on metadata **content** (`|user`), not on the opcode, so
> a chunk whose high 8 bytes are **exactly zero** makes `st_wr_cap = 0` for a reason unrelated to
> tags: the `stc` degrades to a single-bank store and **never writes the destination's high half at
> all**, leaving whatever was already there. Measured in RTL simulation, 585 cycles, 0 exceptions,
> plain `sd`/`ld` control passing in the same run
> (`capstone-ariane verif/tests/custom/capstone/s06-mechanism-probe.S` @ `38962294d`, arm C): a
> destination poisoned with `deadbeefcafef00d` in its high half kept **`deadbeefcafef00d`** after a
> copy whose source high half was zero. Poisoning first is what makes this readable — a
> zero-over-zero test cannot distinguish "copied" from "not written".
>
> This is arguably worse than the documented signature: losing data to zeros is detectable, whereas
> silently retaining a **previous occupant's** bytes is not, and it is reachable from an ordinary
> struct copy where one field is zero. **Consequence for any fix: the write width must become
> OPCODE-derived. A fix that only stops the load zeroing the high half leaves this case broken.**
>
> Same run also measured the **baseline on `fpga-testing-dev-s06`** (arm B: low half survives, high
> half `0`), which until now rested on the two D-cache gates being source-identical to
> `capstone-bootstrap` — the original repro had never been run on that lineage.
>
> And arm D settled what `cap_type` ordinary compiled data pointers carry:
> LINEAR → `delin` → **NONLIN** → `cincoffset` → **NONLIN**. Since `ldc` clears its source only for
> *non*-NONLIN types, **the clear never fires for ordinary code** — so a copy instruction's tagged
> path is a pure copy (correct: NONLIN is freely copyable), and the capability-loss/atomicity risk
> that dominated the fix design does not arise for the real use case.

A 16-byte `ldc`/`stc` pair — the aligned middle loop of `memcpy`/`memmove`, which exists so that
copying a struct containing pointers preserves capability TAGS — keeps only the LOW 8 bytes of a
plain-data chunk. Measured on `caplifive_r20.bit`, both pointers 16-byte aligned, inside a
capability domain (stage 169, control `k800` = 4 in the same boot):

    src32 = c0c1c2c3c4c5c6c7 c8c9cacbcccdcecf d0d1d2d3d4d5d6d7 d8d9dadbdcdddedf
    dst32 = c0c1c2c3c4c5c6c7 0000000000000000 d0d1d2d3d4d5d6d7 0000000000000000

**Reproduced in RTL simulation in 499 cycles**, which removes the board from the loop entirely:
`capstone-ariane verif/tests/custom/capstone/untagged-ldc-stc-128.S`. The test reads the two
halves straight out of the RVFI trace and carries a plain `sd`/`ld` control in the SAME run:

| register | measures | value |
|---|---|---|
| `t3` | `ldc`/`stc` round trip, LOW | `0123456789abcdef` |
| `t4` | `ldc`/`stc` round trip, **HIGH** | **`0000000000000000`** |
| `t5` | control, plain `sd`/`ld` LOW | `0123456789abcdef` |
| `t6` | control, plain `sd`/`ld` HIGH | `fedcba9876543210` |

The control is what makes it attributable: same buffer, same bounds, same capability, written and
read with `sd`/`ld` only, survives exactly. So the loss belongs to `ldc`/`stc`.

### Mechanism, from the RTL sources (not from the instruction semantics)

It is NOT in `capstone_dyn_unit.anvil` — LDC/STC there operate on an already-decoded `fat_cap_t`
and contain no bit-level logic. It is in the D-cache, and BOTH sides contribute in sequence:

* **The load discards the bytes.** `core/cache_subsystem/wt_dcache_mem.sv:310` —
  `ruser = cap_tag_hit ? ruser_cl[rd_hit_idx] : '0;`. Bank 1's SRAM still physically holds the
  real bytes; they are MUXed to a literal `'0` whenever the line's 1-bit shadow capability tag
  (`cap_tag_q`, `:134`) is clear. Any plain `sd` to either half clears that tag for the line
  (`:418-423`), so a buffer filled by ordinary stores always reads back with a zeroed high half.
* **The store then never writes the high half at all.** `:140` — `st_wr_cap = |wr_user_i`, i.e.
  gated on metadata CONTENT, not on the opcode. With the metadata now zero, `:227-240` requests
  only the bank matching the store's own offset, so `dst+8..15` is left at its prior content
  (zero for a fresh buffer, STALE otherwise) — never written by that `stc`.

**QEMU cannot see this.** It carries an explicit `scalar_hi` shadow field for exactly this case
(`capstone-qemu target/riscv/cap.h:79-94`, `op_helper.c:1148-1188`), added so untagged `ldc`/`stc`
is bit-exact over the full 128 bits. There is **no RTL counterpart**. Every SQLite result ever
taken under QEMU has therefore been blind to this divergence. The `memcpy` header comment in
`beebs_freestanding_string.c` already names it as "gap 4" and treats it as closed — that is true
of QEMU only.

### ENV — the QEMU smoke and authority suites are currently BROKEN, unrelated to any codegen change `OPEN`

Noted 2026-08-11 while validating C-18. `run-smoke.sh` and `run-authority-suite.sh` crash QEMU on
every domain load with
`helper_cssplit: Assertion 'mid > rs1_v->val.cap.bounds.base && mid < rs1_v->val.cap.bounds.end'`
(some hit `csshrink` instead). It is NOT a codegen regression:

* `write_42.c` -- a domain with no `memcpy`, no `memmove` and no struct copies -- crashes identically;
* a compiler rebuilt with the codegen changes stashed crashes identically;
* the last nightly (`/tmp/capstone/nightly-20260728_161101/console.log`) recorded `smoke -> PASS`,
  `authority -> PASS`, so the environment worked ~2 weeks ago.

Most likely a source/binary mismatch: `capstone/capstone-qemu` carries an uncommitted WIP diff to
`target/riscv/op_helper.c` dated ~Aug 5 while the built `qemu-system-riscv64` is dated Jul 31.
NOT root-caused. Consequence for now: the QEMU runtime suites cannot gate a change, so codegen
work is gated on lit plus the SQLite silicon gate (which uses a different domain path and DOES
pass). Rebuild QEMU before trusting a smoke/authority result.

### C-18 — compiler-generated `memcpy`/`memmove` libcalls STRIP their pointer arguments `FIXED 2026-08-11`

Found while trying to route S-06's copies through the library. Independent of S-06, and it
affects the DEFAULT build, not only the workaround flag.

`SelectionDAG::getMemcpy`/`getMemmove` built the libcall's pointer arguments as
`PointerType::getUnqual(ctx)` — **address space 0**. On this target pointers are capabilities in
AS 200, so an AS-0 pointer argument is lowered as a plain 64-bit integer and the call site
materialises it with `mv a0, a0` (`addi rd, rs, 0`), which **strips the capability**. The callee
then faults on its first `cincoffset` with `rs1_v->tag` false.

It is reachable without any flag: a 16-byte-aligned copy larger than 512 bytes exceeds the
inline capability path's 32-chunk limit and falls through to a libcall. e.g.
`struct big { void *p; unsigned long a[127]; }` — `*d = *s` emits a `memcpy` call.

**Fix:** type the arguments in the operands' own address space
(`DstPtrInfo.getAddrSpace()` / `SrcPtrInfo.getAddrSpace()`). The values already had the right
address space; only the `Type` describing them was wrong. For targets with a single flat address
space nothing changes.

**Verified by a matched pair on the DEFAULT flags** (no workaround flag involved), same source,
same command, only the compiler differing -- baseline built by stashing the fix and relinking:

| build | before the `memcpy` call |
|---|---|
| baseline | `mv a0, a0` / `mv a1, a1` -- **2 strips** |
| fixed | `a0`/`a1` reach the call untouched -- **0 strips** |

**Blast radius is bounded by construction, not by testing:** `PointerType::getUnqual(C)` is
*defined as* `PointerType::get(C, 0)` (`DerivedTypes.h:729-731`), so for any target whose memcpy
operands live in address space 0 the constructed type is IDENTICAL and nothing can change. Only
non-AS-0 operands differ, and for those the old typing was already wrong. This matters because
the change is in generic SelectionDAG code and this build has only X86;RISCV;Capstone, so
AMDGPU and friends could not be tested here.

**Validation:** Capstone lit 47/47; X86 CodeGen 5246/5251 with the 5 `emutls` failures proven
PRE-EXISTING by a stash-rebuild-rerun baseline (none of them reference memcpy); RISCV 2256/2257,
same `emutls` family; Generic CodeGen 0 failures. The QEMU smoke/authority suites could not
contribute a verdict -- they are currently broken in this environment for unrelated reasons (a
domain containing no memcpy at all hits the same `helper_cssplit` assertion, and the baseline
compiler crashes identically); see the note below.

### SECOND EXPOSURE, measured 2026-08-11: the COMPILER emits the vulnerable pattern too

The memcpy workaround covers only copies that go through our memcpy. It does not cover the
compiler's own aggregate-copy lowering, and that is the larger exposure.

For `struct { void *p; unsigned long x; unsigned long y; }` a pointer is 16 bytes here, so `p`
occupies bytes 0..15 and `x`,`y` occupy 16..31. A plain `*d = *s` lowers to TWO capability-grained
copies:

```
ldc a2, 0x10(a1)     <- bytes 16..31 = x AND y, sixteen bytes of ORDINARY DATA
stc a2, 0x10(a0)
ldc a1, 0x0(a1)      <- the pointer: a real capability, therefore safe
stc a1, 0x0(a0)
```

Under S-06 the second word (`y`) is silently zeroed on every such assignment. Confirmed at both
`-O0` and `-O1`, with no `memcpy` reference in the object at all.

**Board-measured with a standalone rung** (`s06agg`, oracle 64, control `k800` = 4 in the same
boot): **retval 66, twice**, i.e. `y` gone and `x` intact. 66 rather than merely "wrong" is the
signature -- the defect keeps the LOW half of each 16-byte chunk and `x` is the low half. QEMU
returns 64.

**Consequence.** No library-level workaround can reach this, so `BEEBS_LDC_HIGH_HALF_FIXUP` is
necessary but NOT sufficient. It is also a live suspect for the `INVALID_CAPABILITY` fault inside
`CREATE TABLE`: silently zeroing a pointer-adjacent word throughout SQLite is exactly how a
capability ends up invalid.

**ATTEMPT 1, and why it FAILED — recorded so it is not retried blind.** The obvious compiler-side
fix is to stop expanding these copies inline so they call the library memcpy, which already
carries the validated sequence: one implementation of the workaround rather than two. Flag
`-capstone-lower-memops-via-libcall` does that (`findOptimalMemOpLowering` returns false).

Two things were learned:

* It could NOT be done via `MaxStoresPerMemcpy`. The capability-aligned branch of
  `CapstoneTargetLowering::findOptimalMemOpLowering` returns early **without consulting `Limit`**,
  so zeroing those limits leaves the emitted code byte-identical. Measured, not assumed.
* It surfaced **C-18** (above), a real latent bug in the libcall argument types, now fixed.

**It still does not work**, and the flag is committed default-OFF and marked as such. With it on,
SQLite faults under QEMU at `helper_cscincoffset: Assertion rs1_v->tag failed` immediately after
domain entry. Ruled out by measurement: not self-recursion and not a missing symbol (the support
object built with the flag has zero `mem*` relocations inside memcpy/memmove/memset/strcpy and no
undefined symbols), and not C-18, which is fixed and verified gone at the instruction level.

**Hypothesis, NOT established:** a compiler-generated libcall resolves its target through `gp`
(`auipc; cincoffset a3, gp, a2; cjalr`), and on the gp-captable ABI `gp` is not a live capability
during early domain startup, so any `mem*` libcall emitted before the glue installs `gp` faults
exactly this way. If that is right, "route every copy through a libcall" is structurally wrong for
this ABI rather than merely buggy.

**ATTEMPT 2 — inline expansion. WORKS for the target construct, board-validated; NOT yet usable
on the full SQLite build.** `-capstone-memcpy-high-half-fixup` (default OFF) implements
`CapstoneSelectionDAGInfo::EmitTargetCodeForMemcpy`, emitting per 16-byte chunk: plain-store BOTH
64-bit halves, then the `ldc`/`stc` on top -- the sequence validated in RTL simulation.

**Board acceptance PASSED**, matched pair in one boot, same source, differing only by the flag,
control `k800` = 4:

| rung | build | retval |
|---|---|---|
| `s06agg` | unfixed | **66** (the defect; also the positive control that the probe still fires) |
| `s06aggf` | with the flag | **64**, twice |

**AUDITED, and the claim SURVIVED.** An adversarial audit attacked four gaps and could not break
it. Two corroborations worth keeping because they are independent of the retval:

* **`minstret` witnesses which path each domain took.** Counting retired instructions along the
  claimed paths from the disassembly gives 71 (unfixed, y wrong) vs 74 (fixed, both ok), delta 3;
  measured delta is exactly 3 (140 vs 143). Had the fixed arm reached 64 by any route where the
  y-check did not run, the count could not land there.
* **The replay hazard was real and was avoided.** The capture contains one boot but FOUR stale
  RESULT lines replayed from a previous one, including a `s06agg retval=66` that is not from this
  run. A whole-file grep would have read three 66s, two of them stale.

**Reproducibility and slot order, settled by an INVERTED-ORDER boot** (control, fixed, unfixed,
fixed): 64, 66, 64 -- the fixed build returns 64 whether it runs before or after the unfixed one,
and the unfixed returns 66 even when it follows a fixed run. 4 observations of 64 across 2 boots.

**What this evidence does NOT support** (do not let these ride downstream):

* **The mechanism.** This pair cannot distinguish "`stc` does not write the high half" from "`ldc`
  does not read it" -- both predict 66 -> 64. The store-side mechanism rests on the RTL simulation
  (`untagged-ldc-stc-fixup.S` arm E), so cite that, not this rung.
* **Any shape other than the one tested.** The hook declines `size % 16 != 0`, alignment < 16, and
  more than 32 chunks. Unaligned and odd-sized aggregates are neither fixed nor tested.
* **Anything above `-O0`.** The board build is `-O0`, where no MI scheduling runs. At `-O2` the
  scheduler DOES reorder the sequence; the required per-chunk order survived there only by the
  direction of the moves, not because a serial chain is unbreakable. The source comment that
  claimed otherwise has been corrected. The rung cannot currently be built above `-O0` at all --
  clang hits a pre-existing backend limit ("Cannot materialize arbitrary >64-bit constants as
  capabilities"), which reproduces with the flag removed.

Correction to an earlier note: `s06copy` is NOT part of this acceptance test. It writes its
capability copy explicitly in C rather than as an aggregate assignment, so the compiler is right
to leave it alone -- that shape is the library's job and stays at 16.

**Two implementation traps, both measured, both worth keeping:**

* **The hook alone is dead code.** `SelectionDAG::getMemcpy` tries the INLINE expansion FIRST and
  only calls `EmitTargetCodeForMemcpy` if that fails. So `findOptimalMemOpLowering` must DECLINE
  exactly the same shape the hook accepts; the two conditions have to match, or a copy declined in
  one place and rejected in the other falls through to a libcall, which is broken on this ABI.
* **The pre-write stores must be VOLATILE.** Without that they are dead by the compiler's own
  model -- it believes the `stc` writes all 16 bytes of the chunk -- so DSE deletes them and
  silently regenerates the unfixed sequence. Measured: for a copy into a stack slot the pre-writes
  vanished entirely and the output was byte-identical to the unfixed build. A build that looked
  fixed and was not.

**FIXED 2026-08-11. The cause was mine: integer pointer arithmetic stripping the capability.**

The hook computed each chunk address with the generic offset helper, which builds an `ISD::ADD`.
That selects to an INTEGER `addi` whenever it is materialised instead of folded into the
addressing mode -- and `addi` strips the capability, so the access on that address faults with
`UNEXPECTED_OPERAND`. Mapped to the exact instruction in `sqlite3Parser`:

```
ldc  a3, -0x110(s0)      <- source pointer, a capability
ld   a4, 0x0(a3)         <- plain half 0
sd   a4, -0xe0(s0)       <- offset folded into the immediate: base s0, FINE
ld   a4, 0x8(a3)         <- plain half 1
addi a5, s0, -0xd8       <- materialised instead: INTEGER add, capability STRIPPED
sd   a4, 0x0(a5)         <- fault: rs1 = x15 = a5, imm = 0
```

Note the neighbouring store was fine because its offset happened to fold. That is why it never
appeared on a small rung and only at SQLite scale: it needs an address that does not fold.

Fix: use `CapstoneISD::CIncOffset` -- the node that means "advance a capability's cursor" -- for
every address the hook computes. **The full SQLite QEMU gate now PASSES with the fixup on**, on
the amalgamation alone and with library+compiler fixups on all objects, and Capstone lit is 47/47.

**How the pc was mapped**, since this is reusable: add `CAPSTONE_PRINT_LOAD_BASE=1`, which makes
the domain print the RUNTIME address of `sqlite3_initialize`. `base = printed - VA_from_readelf`,
then `image_VA(fault) = fault_pc - base`. The base measured **0x1015f0000 and was identical across
two different builds**, so it is stable and can be taken from a build that RETURNS -- which is
necessary, because `output_text` buffers into the shared region and the host flushes it only
after the domain returns, so a FAULTING domain prints nothing.

### The software workaround is NOT VIABLE on this silicon. S-06 needs the RTL fix.

This is the conclusion of the whole codegen line of work, and it is a change of position: earlier
notes said the fixup was "not yet usable" as if it were a bug to be found. It is not.

**State of the fixup:** correct under QEMU (the full SQLite gate passes with library + compiler
fixups on all objects), correct on the isolated rung on silicon (66 -> 64, four observations
across two boots, both slot orders), and it WEDGES SQLite on silicon with
`mcause 25 = INVALID_CAPABILITY`.

**Matched pair, one boot, control `k800` = 4** -- two staged builds differing only by the ldc
fixups, both running stage 168 (open + a SHORT create):

| build | result |
|---|---|
| fixups OFF | RETURNS `rc=11`, twice |
| fixups ON | **WEDGES** |

~~So the fixup itself causes the wedge; this is not merely "SQLite runs deeper once the data is
correct".~~ **THAT INFERENCE IS WITHDRAWN 2026-08-11 — this pair is confounded like the others.**
At stage 168 the fixups-OFF build's `CREATE` **fails** (`rc=11`) while the fixups-ON build repairs
the data so the same `CREATE` **succeeds**, and therefore executes strictly more code. Depth and
data-correctness are coupled by construction: *any* SQLite arm that repairs the data also goes
further, so no SQLite pair can separate "the store pattern is toxic" from "correct data reaches a
second fault". A shorter stage does not break the coupling, it only shortens both sides of it.

**The experiment that WOULD separate them drops SQLite entirely.** Run the fixup's exact store
pattern (`ld, ld, ldc, sd, sd, stc`) in a ladder rung over a working set large enough to force
eviction (the D-cache is 32 KB, so >= 64 KB), with real capabilities interspersed among plain
data, returning a checksum — against a control rung doing the baseline `ldc, stc` over the same
data. There is no data-dependent control flow, so the store pattern is the only variable.

* rung wedges, control returns -> the store pattern is intrinsically toxic at scale, and every fix
  using it is dead, including the current workaround
* both return -> the store pattern is exonerated at scale, and SQLite's wedge is a second fault
  that only becomes reachable once the data is correct

**RUN 2026-08-12 on `caplifive_s06.bit`. RESULT: the store pattern is EXONERATED at scale.**

| rung | per-chunk pattern | result |
|---|---|---|
| `s06sbase` | `ldc, stc` | **2048** (all chunks correct), 804k cycles |
| `s06sfix` | `ld, ld, ldc, sd, sd, stc` | **2048**, ~1.01M cycles, **4 boots out of 4** |

64 KB working set (the D-cache is 32 KB, so a single pass evicts everything it touched),
capabilities interspersed one per 256 bytes, 2048 chunks, control `k800` = 4 in both boots. The two
arms are a genuine matched pair: read from the emitted assembly, the fix loop is
`ld; ld; ldc; sd; sd; stc` and the base loop is `ldc; stc`, differing in exactly the four plain
accesses. Both QEMU-verified at 2048 before the boots.

**So the fixup's store pattern does not wedge, and SQLite's wedge is NOT caused by it.** That is the
first unconfounded result in this line, and it settles a question three previous experiments could
not touch. Consequences:

* The **`ld, ld, ldc, sd, sd, stc` sequence is safe at scale** — the shipped workaround's store
  pattern is not the problem, and neither is the LCC-query design's.
* The surviving explanation is the one this file already carried: **repairing the data lets SQLite
  run deeper and meet a second, distinct fault.** Fixing S-06 in the compiler will therefore NOT
  make SQLite pass; it will move the failure.
* **The second fault is now the blocker, not S-06.** ~~Lead worth pursuing first: the wedge is
  `mcause 25 INVALID_CAPABILITY`, which comes from `get_node_query_validity` failing on the address
  capability (`capstone_dyn_unit.anvil:337`, `:404`).~~ **RETRACTED — see the entry immediately
  below. `mcause 25` is not `INVALID_CAPABILITY`, and that lead is excluded by the number itself.**

**RETRACTION 2026-08-12: `mcause 25` HAS BEEN MISNAMED THROUGHOUT THIS FILE. It is
`UNEXPECTED_OPERAND`, not `INVALID_CAPABILITY`.** The observed *value* is sound — the monitor's
`handle_exception` `default:` arm does `csrr a5, mcause; csrr a6, mepc; 1: j 1b`
(`sbi_capstone.c:748-752`), so the trap was delivered normally and 25 was read out of `a5`. Only the
*name* attached to it is wrong, and it sent three consecutive investigations after the wrong
subsystem.

The encoders, three independent sources that agree:

* `ex_stage.sv:469` (FLU) and `cva6.sv:1360` (DYN) both compute
  `cause = 64'd24 + exception_code`, with `7/8/9` special-cased to `LD_ADDR_MISALIGNED`,
  `ST_ADDR_MISALIGNED`, `ILLEGAL_INSTR`.
* The `ex_code` enum (`capstone_unit.anvilh:290-300`) is
  `NO_EXCEPTION, UNEXPECTED_OPERAND, INVALID_CAPABILITY, UNEXPECTED_CAP_TYPE,
  INSUFFICIENT_PERMISSION, OUT_OF_BOUNDS, ILLEGAL_OPERAND_VALUE, LOAD_ADDRESS_MISALIGNED,
  STORE_ADDRESS_MISALIGNED, ILLEGAL_INSTRUCTION` — ordinals 0..9. Positions 7, 8, 9 are exactly the
  three the encoders special-case, which pins the ordinals independently of any comment.
* `riscv_pkg.sv:349-353`: `DEBUG_REQUEST = 24`, `UNEXPECTED_OPERAND_TYPE = 25`,
  `INVALID_CAPABLITY = 26`, `UNEXPECTED_CAPABLITY_TYPE = 27`.

So `UNEXPECTED_OPERAND` (code 1) → **25**, and `INVALID_CAPABILITY` (code 2) → **26**.

**Where the error came from:** the inline comments in the enum itself read
`UNEXPECTED_OPERAND, // 24` and `INVALID_CAPABILITY, // 25`. They are off by one, and this file
cited them (`capstone_unit.anvilh:289-296`) as the authority. A comment was trusted over the
encoder.

**What this immediately excludes.** `capstone_dyn_unit.anvil:337` (LDC) and `:404` (STC) — the
revocation-validity check on the address capability — raise `INVALID_CAPABILITY`, which encodes to
**26**. They cannot produce the observed 25. The named lead is dead *arithmetically*, before any
experiment. This also explains why three rev-node hypotheses in a row failed: they were all
`INVALID_CAPABILITY` theories chasing an `UNEXPECTED_OPERAND` fault.

**What is now in scope.** `UNEXPECTED_OPERAND` is raised at 11 sites in the DYN unit and 11 in the
FLU unit, and the guard is almost always the same shape: an operand's `cap_type` is `NOT_CAP` where
a capability was required, or is a capability where a plain integer was required. The two that
matter for a memcpy-heavy workload:

| site | condition |
|---|---|
| `capstone_dyn_unit.anvil:306` `LDC` | `rs1.metadata.cap_type == NOT_CAP` — **the base register is not a capability** |
| `capstone_dyn_unit.anvil:370` `STC` | `rs1.metadata.cap_type == NOT_CAP` — same, for the store |

i.e. the failure is "the address operand stopped being a capability", not "its revocation node says
invalid". Those are different subsystems with different fixes.

**One ambiguity remains, and it must be settled before the name is trusted again.**
`commit_stage.sv:205-228` — the PC-capability check — uses a **different base, 23**, and its own
comments say so (`64'd25; // INVALID_CAPABILITY (23 + 2)`). So that block emits 25 for
`INVALID_CAPABILITY` on the *fetch* capability, colliding with the execute path's 25. That block is
inconsistent with `riscv_pkg.sv` and with both other encoders and looks like an off-by-one bug in
its own right. Until it is resolved, an observed 25 has two readings:

* **(A)** `UNEXPECTED_OPERAND` from FLU or DYN — an operand is the wrong shape. Base 24.
* **(B)** `INVALID_CAPABILITY` on the PC capability from `commit_stage.sv` — `pc_revnode_valid_d`
  is false. Base 23.

**The discriminator is `mepc`, which the monitor already captures in `a6` at the wedge.** Under (A)
`mepc` points at a capstone instruction with an operand that can be inspected; under (B) it points
at whatever was committing, and the fault is not tied to any operand. Reading `a6` at the next wedge
settles it with no new experiment. That is the next step, and it costs one boot that was going to
happen anyway.

**Not yet established, and NOT to be written up as a mechanism until measured:** a plausible chain
for (A) is that the fixup's plain-store-then-`stc` leaves a destination granule untagged while stale
metadata survives in bank 1 (the same single-bank behaviour S-06 is made of, and the same effect
just measured in `linear-clear-audit`), so a later `ldc` of that granule yields `NOT_CAP` and using
it as a base raises `UNEXPECTED_OPERAND`. This is a hypothesis with a mechanism, not a finding.
Against it: `s06sfix` copied 2048 capability-bearing chunks at 64 KB scale without wedging — though
it never DEREFERENCED a copied capability, so it does not test this.

**POOL EXHAUSTION IS REFUTED**, cheaply: `REVNODE_HEAD_BITS = 16` (`capstone_unit.anvilh:498-515`),
so the pool holds 65536 nodes with 65535 reserved as the "full" sentinel. The `rev_node_head`
observed at the wedges — 0x25e (606), 0x1a2 (418), 0xf9 (249) — is two orders of magnitude below
that, which is exactly why the overflow bit was clear in every dump. The pool is nowhere near full,
and it is not the mechanism.

**~~THE LIVE LEAD: rev-node tag loss zeroes `valid`~~ — REFUTED 2026-08-12 by rung `s06rev`, and
the reason is a field-layout fact worth keeping.** The hypothesis was that revocation nodes, being
cacheable but never shadow-tagged, come back untagged after an evict+refill, so `ruser` is
force-zeroed and the node reads `valid = 0`. **`valid` does not come from `ruser`.** `rev_node_t` is
`depth[32], prev[30], next[30], valid[1], linear[1]` = 94 bits
(`capstone_unit.anvilh:521-525`), assembled as `{data_ruser[29:0], data_rdata[63:0]}`
(`ex_stage.sv:1030`). So `ruser` carries the top 30 bits — part of **`depth`** — and `valid` sits at
bit 1, inside `data_rdata`. Zeroing `ruser` cannot clear `valid`.

Measured: rung `s06rev` round-trips a capability, streams 64 KB (twice the 32 KB D-cache) to evict
everything, then round-trips it again. It **returns 11** — both arms survived — with control `k800`
= 4 in the same boot. QEMU-verified at 11 beforehand.

Note what this does NOT clear: the region facts are real and unchanged — the pool at
`[0xBFF0_0000, 0xC000_0000)` IS cacheable (`config_pkg:142-144`) and IS excluded from the
shadow-tag write (`wt_axi_adapter.sv:139-145`, deliberately, per the assert at `:987-992`). An
evicted rev-node line therefore still loses its top 30 bits, i.e. part of `depth`. That is worth its
own investigation — a corrupted depth would affect revocation-tree walks — but it is not the
mcause-25 mechanism.

**ELIMINATED so far for the second fault:** pool exhaustion (pool is 65536, observed heads ~250-600);
the fixup's store pattern (`s06sfix` returns 2048 at 64 KB scale); rev-node tag loss zeroing `valid`
(this entry); and — added 2026-08-12 — **the entire revocation-validity family, excluded
arithmetically**: those sites raise `INVALID_CAPABILITY`, which encodes to `mcause 26`, while the
wedge shows 25. The fault remains UNEXPLAINED, but the search has moved from "the revocation node
says invalid" to "the operand is the wrong shape". See the retraction entry above.

Worth noting for anyone reading the eliminations as a run of bad luck: the first three all targeted
`INVALID_CAPABILITY`, so a single naming error accounts for all of them at once. The measurements
themselves were sound and their exclusions still hold.

~~Original lead text, kept for the region facts it establishes:~~ `capstone_rev_node.anvil:36-42` (`get_rev_node`) issues `mem_ch.read_req` and
returns `data.valid`; `ex_stage.sv:1030` reconstructs the node from
`{rev_mem_rd_res_i.data_ruser[29:0], data_rdata}`. So the `valid` bit arrives via `ruser` — the
same channel `wt_dcache_mem.sv` force-zeroes when a line's shadow tag is clear. If a rev-node line
ever reads back untagged, `valid` reads 0 and the next `ldc`/`stc` through that capability raises
exactly `mcause 25`, which is the observed wedge.

Whether that can happen turns on one question: is the rev-node region covered by the shadow-tag
write? `wt_axi_adapter.sv:139-145` gates `needs_tag` on `in_data_region`, i.e. `paddr` within
`[MEMORY_BASE, DATA_MEM_TOP)`, while the nodes live at `CAP_REVNODE_MEM_BASE = 0xBFF0_0000`
(`ariane_pkg.sv:590`). **UNRESOLVED, and the next thing to settle.** It also fits the observed
shape: it needs cache pressure to evict a rev-node line, which is why it appears only at scale and
only once a fix adds store traffic.

**Two candidate causes tried and REFUTED:**

* *Integer pointer arithmetic* -- real, found and fixed (`CIncOffset`), and it is what made the
  QEMU gate pass. It was not the silicon wedge.
* *Self-copy tag loss* -- the sequence wrote the destination before reading the source, so an
  exact self-copy (`*d = *s` with `d == s`, which clang lowers to a memcpy) would clear the tag of
  the line the `ldc` then reads. Reordered to read-everything-then-write-everything; **the wedge
  persists**, so this was not it either.

**~~LEADING HYPOTHESIS~~ — REFUTED 2026-08-11. Struck, and recorded so it is not re-proposed.**
The hypothesis was that the fixup's tripled store traffic causes a line whose tag was set by an
`stc` to lose it across an eviction and refill, because the refill path gates on
`|wr_cl_user_i[7:0]` while the single-word path gates on `cap_tag_hit` — "two different
conditions". **They are not two different conditions.** On the refill path those eight bits are
not capability metadata at all: `wt_axi_adapter.sv:441-442` zeroes the word and writes a single
byte of `tag_wr_value_q = is_cap_req = |dcache_data.user` (`:196`, `:402`), and `:731-734` reads
exactly that byte back, so the value is `0x00`/`0x01`. The AXI USER sideband carries nothing
(`:204`, `axi_wr_user[0] = '0`). The two gates are the **same predicate over different
encodings**, and the cache is write-through with no dirty writeback, so a tag written by an `stc`
reaches the shadow-tag region directly.

**The arm-E wedge is therefore UNEXPLAINED, not hypothesised.** It still fits the observation
shape — passes in RTL simulation, passes on a hot 10 KB rung, fails only at SQLite scale — but no
mechanism is currently proposed.

**~~LOCALISED 2026-08-11: the wedge is caused by the PLAIN STORES~~ — RETRACTED 2026-08-11, the
experiment was CONFOUNDED.** The claim was that `ld, ld, ldc` is exonerated and the
plain-store-then-`stc` destination pattern is the trigger. It does not follow from the data.

| arm | per-chunk sequence | destination written by | DATA | result |
|---|---|---|---|---|
| `sqA` baseline | `ldc, stc` | `stc` only | **WRONG** (S-06) | returns, `rc=11` |
| `sqD` | `ld, ld, ldc, stc` | **`stc` only** | **WRONG** (S-06) | returns, `rc=11` |
| `sqB` arm E | `ld, ld, ldc, sd, sd, stc` | plain stores + `stc` | **CORRECT** | **WEDGES**, mcause 25 |

**Why it is confounded.** `sqD` drops the plain stores, so its destination is written *only* by the
`stc` — which under S-06 loses the high half. Verified at disassembly level: `sqD` has **zero**
`ldc→sd,sd`. So `sqD` is the baseline plus dead volatile loads and produces **the same corrupt
data**. The variable separating the two returning arms from the wedging one is therefore not the
store pattern; it is **whether the copy is correct**. `sqB` is the only arm whose data is right, and
the only one that wedges.

**The alternative explanation was already in this file** — see the note below at "Both fixups repair
the data, so SQLite runs deeper into `CREATE TABLE` than it ever has and meets a silicon-side
capability-validity fault". Three independent constructions have now wedged (the library fixup, the
compare-and-repair copy `aa600e1f3`, and arm E) and **all three repair the data**. That explanation
survives the experiment; the store-pattern one does not.

**Two further defects in the experiment, both verified:**

* **`sqA` was not a clean control.** The `-capstone-memcpy-high-half-fixup` flag silently converts a
  16-byte-aligned `memmove` into a **libcall** (confirmed by running `llc`: 4 `ldc`/`stc` with the
  flag off, a `memmove` libcall with it on), because `MemOp` carries no memcpy/memmove discriminator
  and `EmitTargetCodeForMemmove` is not overridden. So the flag-off and flag-on arms differ by more
  than the copy sequence.
* **N = 1 per arm**, on a machine whose non-determinism this project's own rules require bisections
  to control for. Determinism was never demonstrated.

**The same mistake, twice, in different disguises.** An earlier arm (`-capstone-memcpy-fixup-no-stc`,
`ld, ld, ldc, sd, sd` with no `stc`) also wedged and also looked like "source-side confirmed"; it was
confounded because dropping the `stc` destroys copied capabilities. `sqD` was built to fix that and
introduced the mirror-image confound. Both times, *source traffic* was held constant while *data
correctness* was allowed to float. Any future arm must state which variables it holds fixed.

**What would actually discriminate:** hold data correctness CONSTANT and vary only the store pattern
— e.g. a build writing the destination once via `sd, sd` for every chunk with no `stc` at all, on a
workload whose copied buffers contain no capabilities. Correct data, single write, no `stc`. If that
wedges, the store pattern is exonerated and the deeper-fault explanation stands. N ≥ 3 per arm.

**Consequence for fix design, and it is the opposite of what was recorded:** a repair that produces
CORRECT data sits on the same side of the line as every construction that has wedged. That includes
the LCC-query design. It should not be built until this is resolved.

**Consequence.** There is no software workaround for S-06 that is safe on this silicon:

* dropping the capability path strips tags and wedges;
* the compiler cannot ask whether a chunk is a capability (`LCC` faults on `NOT_CAP`);
* and writing a line as plain data first and restoring the tag with `stc` -- the only construction
  that is correct on paper and in simulation -- destabilises the workload.

**S-06 therefore needs the RTL fix**, and the handover package
(`capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half/`) is the deliverable. It now
carries **`FIX-PROPOSAL.md`** with two costed options:

* **Option B, recommended first:** a tag-preserving 16-byte memory-to-memory copy instruction.
  The value never enters a register, so `cap_pack_t`, the register file, forwarding, the
  scoreboard and `capstone_dom_switcher` are all untouched. It is exactly what software needs,
  because every failure here is a COPY. Compiler side is a one-line change: the aggregate-copy
  lowering already exists and would emit the new instruction instead of the `ldc`/`stc` pair.
* **Option A, the general fix:** add a tag bit to `cap_pack_t`, deliver bank 1 ungated, make
  `st_wr_cap` opcode-gated, widen the user lanes. The work is not in those four mechanical
  steps but in auditing every site that infers "is a capability" from `cap_type != NOT_CAP`,
  and in the domain switcher's saved-context format.

Acceptance for either is already in the folder and needs no SQLite: `./run.sh sim` (499 cycles,
self-controlled) and `./run.sh rung` (`s06copy` 16 -> 32, `s06agg` 66 -> 64). Both fixups
stay DEFAULT OFF: they turn a diagnosable error return into a wedge.

### Earlier note (superseded by the above): still blocked on silicon, and it is NOT this fix

With the fixup on, SQLite passes the entire QEMU gate but **wedges on silicon** with
`mcause 25 = INVALID_CAPABILITY`, `rev_node_head = 0xf9`, `overflow = 0` (pool healthy) -- the
SAME fault the library fixup produced, in the same place. Both fixups repair the data, so SQLite
runs deeper into `CREATE TABLE` than it ever has and meets a silicon-side capability-validity
fault that QEMU does not reproduce. That is now the single remaining blocker.

### Superseded localisation notes (kept: they cost real time and the exclusions still hold)

Narrowed 2026-08-11, all under QEMU:

* **It is a MISCOMPILE, not "SQLite getting further".** QEMU has no S-06 -- it preserves all 128
  bits -- so this flag should be semantically a no-op there. The baseline passes the whole QEMU
  gate including CREATE; with the flag it faults. That distinguishes it from the library fixup,
  whose silicon wedge really was "the data is now correct so execution goes deeper".
* **It is in the AMALGAMATION, not the support objects.** With the flag on the support objects
  only (`SUPPORT_EXTRA_MLLVM`) the full gate PASSES; on the amalgamation only
  (`AMALGAM_EXTRA_MLLVM`) it faults. Per-object `-mllvm` scoping was added to
  `build-sqlite-silicon.sh` for exactly this bisection -- `EXTRA_MLLVM` reaches all three objects
  at once, which makes a misbehaving codegen flag impossible to localise.
* **It is inside `CREATE TABLE`.** RUNSTOP ladder with the flag: after `sqlite3_config` RETURNS,
  after `sqlite3_initialize` RETURNS, after `sqlite3_open` RETURNS, after `CREATE TABLE` FAULTS.
* **One chunk is enough.** Restricting the fixup to 16-byte copies
  (`-capstone-memcpy-high-half-fixup-max-bytes=16`) still fails, so the defect is in the basic
  single-chunk emission and not in multi-chunk handling.
* **Two manifestations, same family:** `cause = 24 UNEXPECTED_OPERAND` (a capability access whose
  base is not a capability) at 512 bytes, and `helper_cscincoffset: Assertion rs1_v->tag failed`
  at 16 bytes. Both are "a capability operation received an untagged operand".

**Excluded by measurement, so do not re-derive:** `sqlite3_config` (it is one of only 44 functions
genuinely re-codegen'd by the flag, and it grew 436 -> 604 instructions, which made it the obvious
suspect -- but RUNSTOP=1 returns cleanly). `__capstone_cap_init` (its diff is *only* `auipc`/`addi`
immediates, i.e. relocation drift from code moving; identical 1522-instruction count). A `-O0`
spill/reload theory, in which the plain accesses cause the base to be reloaded with an integer
`ld` that the capability access then reuses -- at `-O0` the base is in fact reloaded with `ldc`.
Eight capability accesses on an integer-defined base found by a static scan of the amalgamation --
they are PRE-EXISTING, present with and without the flag.

**No cause is recorded.** The next step is to map the runtime pc back to the image, which needs
the domain load base; `SQ: self=` is an encoded capability, not a usable address, so that needs a
deliberate probe. With the flag on, the domain enters and takes
a capability fault under QEMU (so this is not silicon):

    [CAPSTONE] Cap mem access requires capability: pc = 101681134, rs1 = x15, imm = 0
    [CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x101681160, tval = 0x0

Cause 24 is `UNEXPECTED_OPERAND` -- a capability memory access whose base register is not a
capability. Established: it is the compiler flag ALONE (identical fault with the library fixup
both on and off), the pc is deterministic across runs, and `badaddr` VARIES run to run while the
pc does not, so the faulting site is data-dependent rather than a fixed bad pointer. NOT
localised -- mapping the runtime pc back to the image needs the domain load base, which was not
captured. Do not guess a cause; that is the next thing to measure.

**Fixable in OUR compiler, without the RTL change.** The backend uses `ldc`/`stc` for chunks it
does not know to be capabilities. It can instead emit two 64-bit `ld`/`sd` for a chunk that cannot
contain one, or apply the same pre-write sequence the library workaround uses (plain-store both
halves, then `ldc`/`stc` on top), which is correct for both kinds of chunk. That covers what the
library fixup cannot, and it is a codegen change rather than a datapath change.

### Why software cannot simply work around it AT THE LIBRARY LEVEL

* The aligned path cannot be dropped: it is the only one that preserves tags, and a byte-wise
  copy makes SQLite dereference untagged pointers and wedges the core (recorded under S-04).
* Code cannot ask whether a chunk is a capability. `LCC` with a `NOT_CAP` operand raises
  `UNEXPECTED_OPERAND` **before** it examines the requested field
  (`capstone_dyn_unit.anvil`, `func LCC`, the `cap_type==NOT_CAP` branch), so
  `__builtin_capstone_cap_get_tag` faults on exactly the plain data it would be used to detect,
  and a capability fault inside a domain wedges rather than traps.

### 2026-08-11 — the workaround is VINDICATED; the wedge is a DIFFERENT defect, now localised

The "primitive-correct, workload-fatal" reading below is **superseded**. Four things were
measured, control `k800` green in every boot:

**1. It is not S-01 image sensitivity.** A REDRAW control — the baseline plus one dead,
never-called function, which is exactly the perturbation S-01 was characterised with — **RETURNS**
(`stage=create rc=11`, same as the unperturbed baseline), while the fixup build wedges in the same
boot. So the wedge is attributable to the change, not to rebuilding the image. (Side finding: an
inert perturbation does NOT hang on `caplifive_r20.bit`, where S-01 was characterised on the older
`caplifive_fixed_forward.bit`.)

**2. The fixup does NOT lose capability tags.** Stage 170/171 copy a chunk holding a REAL
capability alongside a plain chunk in one `memcpy`:

| arm | stage 170 (bytes) | stage 171 (dereference the copied pointer) |
|---|---|---|
| fixup OFF | `0x31` — plain half WRONG (S-06) | `0x51` — tag survived |
| fixup ON | `0x30` — **both halves correct** | `0x51` — **tag survived** |

So it repairs the data and preserves tags. The "it must be clearing a tag" theory is REFUTED.

**3. The fault is named.** The wedge is a capability exception, `mcause 25 = INVALID_CAPABILITY`
(`capstone_unit.anvilh:289-296`), raised by `LDC`/`STC` when the **base** capability's revocation
node reports invalid (`capstone_dyn_unit.anvil:332-338` and `:400-405`,
`get_node_query_validity(rs1_v.metadata.revnode_id)`). It is NOT R-12: the same wedge dump reads
`rev_node_head = 0x25d` (605) with `overflow = 0`, nowhere near exhaustion. `commit pc` is the
`0x2` junk sentinel, as usual, so the pc says nothing.

**4. It wedges INSIDE `CREATE TABLE`.** RUNSTOP ladder on the fixup build, ascending, every arm
designed to return:

| clamp | result |
|---|---|
| after `sqlite3_initialize()` | RETURNS |
| after `sqlite3_open(":memory:")` | RETURNS |
| after `CREATE TABLE` | **WEDGES**, and prints no `SQLITE ERROR` — it faults rather than failing |

**Conclusion.** The workaround is not breaking the workload. Without it the schema text is half
destroyed, so SQLite bails out early with `SQLITE_CORRUPT`; with it the text is correct, the
schema re-parses, and execution continues deeper into the CREATE path than it has ever reached on
silicon — where it takes `INVALID_CAPABILITY`. The blocker has moved from silent data corruption
to a specific, named, localised capability-validity fault.

**Knob still defaults OFF**, for a narrower reason than before: a build that returns an error is
more diagnosable than one that wedges. Turn it ON to work the CREATE-path fault, which is the
next thing to bisect and needs correct data to be reachable at all.

**CAUTION on single verdicts.** Stage 170 with the fixup **wedged at boot position 2 and returned
at position 4** on the same image. Position-dependent nondeterminism is live in these builds, so
no single-position verdict on a full-workload domain is safe; the ladder above was read only where
arms returned consistently.

### SUPERSEDED (2026-08-11) — the workaround that was tried: primitive-correct, workload-fatal

`BEEBS_LDC_HIGH_HALF_FIXUP` writes both 64-bit halves with plain stores and then lays the
`ldc`/`stc` on top. It is branchless and exploits the mechanism above: for a capability the
metadata is non-zero so the `stc` writes both banks and restores the tag; for plain data the
`stc` degrades to a single-bank store that never touches the high half, so the plain store
survives. Validated on both kinds of chunk in simulation (`untagged-ldc-stc-fixup.S` arm E) and
**on silicon at the primitive level**: stage 169 = `0x40` with `dst32` byte-identical to `src32`,
stage 167 = `0x70`.

**It nevertheless WEDGES the full SQLite workload**, isolated in one boot with the control green:

| build | knobs | full run |
|---|---|---|
| `qC` | memcpy optnone only | RETURNS, `stage=create rc=11` |
| `qB` | + writers optnone | RETURNS, `stage=create rc=11` (neutral) |
| `qA` | + ldc high-half fixup | **WEDGES** |

Why it wedges at workload scale is **NOT established** — do not record a cause. An earlier shape
(copy, then compare the high halves and repair on a difference) is separately REFUTED and must
not be retried: for a genuine capability the destination's stored metadata word need not be
bit-identical to the source's, so the comparison can say "differ", run the repair store, and
CLEAR A LIVE TAG.

Knob default **OFF** (`build-sqlite-silicon.sh`), deliberately: it trades a diagnosable error
return for a wedge.

### What the hardware side needs

The RTL needs the QEMU behaviour: preserve the raw upper 64 bits of a `tag == 0` line across an
`ldc`/`stc` round trip.

**Handover package: `capstone/tests/fpga-repros/S06-untagged-ldc-stc-high-half/`** — self-contained,
and the folder is the report.

* `./run.sh sim` — RTL simulation, 499 cycles, no board. Carries a plain `sd`/`ld` control over the
  same buffer and the same capability, and FAILS LOUDLY if that control is wrong, so a run that
  proves nothing says so. This is the acceptance test for a fix: it passes when the high half reads
  `0xfedcba9876543210` instead of zero.
* `./run.sh rung` — a 10 KB standalone domain on the board. Returns **32** when every byte survives
  and **16** under the defect; measured 16 three times in one boot with the control `k800` green.
* `./run.sh verify` — checksums, plus a check that the shipped `.dom` still CONTAINS the copy under
  test. That check is keyed to `ldc rX, A(rB)` -> `stc rX, C(rB)` with `A != C`; a looser "is there
  an ldc near an stc" version answered YES for `k800`, which contains no copy at all.

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

### I-5 — every monitor error is invisible on the FPGA `OPEN, cheap fix identified`

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

### C-17 — `i128 SELECT_CC` is not selectable; the SQLite domain cannot build at `-O1` `OPEN — RECURRENCE`

    fatal error: error in backend: Cannot select:
      t88: i128 = CapstoneISD::SELECT_CC t9, Constant:i64<10>, seteq:ch, t93, t92

Building the SQLite domain with `SQLITE_OPT_LEVEL=-O1` crashes the backend. A `SELECT_CC`
producing an **i128** (a capability) has no selection pattern. `-O0` never forms the select,
which is why `OPT=${SQLITE_OPT_LEVEL:--O0}` (`build-sqlite-silicon.sh:41`) has always been the
SQLite default — the ladder rungs, by contrast, build at `-O1`.

**This is a RECURRENCE, not a new bug.** `ISSUES.md` already records an i128 `SELECT_CC` crash
in the stage-30..34 work ("Its first attempt built nothing (i128 `SELECT_CC`) ... both are now
fixed"). Either that fix was shape-specific or it regressed; the earlier entry does not say
which shape it covered. **Check the previous fix before writing a new one.**

The shape is ordinary and will recur elsewhere: `sqlite3Strlen30` is
`if( z==0 ) return 0; return 0x3fffffff & (int)strlen(z);` — a null check on a pointer feeding a
masked result is exactly what forms `select_cc` on a capability at `-O1`.

**Why it matters beyond the crash:** `-O1` is the cheapest available shot at the R-17 blocker
(see below), and this is what blocks it.

### M-1 — domains run with `mtvec = 0`, so a domain fault is an unbreakable loop `OPEN — OURS, FIX FIRST`

**A trap and a hang are the same observation from outside.** This is not the SQLite blocker; it
is why the SQLite blocker resisted 30+ board sessions.

Verified end to end:
* Seal slot 1 is the domain's trap vector — `csr_regfile.sv:399` (save) and `:1880-1884`
  (restore: `reg_id 1` → `ctvec_d = data[127:64]; mtvec_d = data[63:0]`). Confirmed by its
  neighbours: slot 2 is `{cscratch, mscratch}` (how `gp` is delivered), slot 3 is `mstatus`.
* The monitor never writes it — `sbi_capstone.c:801` zeroes all slots, `:823-825` write only
  0, 2 and 3.
* A trap does not install a new PCC — `frontend.sv:425-426` sets `npc_d = trap_vector_base_i`
  while `npc_metadata` is carried forward.

So a fault jumps to pc = 0 with the domain's PCC still installed, faults again on the
out-of-bounds fetch, and loops forever in M-mode with interrupts off and no UART.

**Every "no trap was reported, so it is not a fault" inference in this project is void.**

**CORRECTED 2026-08-05: the fix ALREADY EXISTS and needs NO monitor change.** An earlier
version of this entry said it required a glue handler, `dom_seal[1]` in `create_domain`, and a
new readback path. That is wrong and would send someone to write a monitor patch that is not
needed. `start-gp-captable-interp.S:760,824` already contains, behind **`INTERP_DOMAIN_MTVEC=1`**:
`lla t0, .Ldomain_trap; csrw mtvec, t0` — set from INSIDE the domain (M-mode, so `csrw` is
permitted), with `.Ldomain_trap` jumping to `.Ldomain_returned`, which already captures `mcause`
and `mtval`. Verified in a built image: `csrw mtvec, t0` at `0x102e4`, absent without the flag.

So M-1 is fixed by **building with `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1`**. What is NOT
yet established is whether the handler is reachable after a real fault — it touches
`sp`/`cscratch`, so a fault that corrupted those would make it fault again and look identical to
no handler. **Verify with `tagf`** (`fpga-repros/RTL-store-user-metadata/`, a deliberate fault):
returning with a cause proves the handler works.

Worth fixing on its own merits: **any** domain that faults for any reason is currently
undebuggable and takes the core with it.

### R-17 — a ~1.6 MB domain hangs after ANY perturbation of its image `OPEN — NOT ROOT-CAUSED`

**Reproducer:** `capstone/tests/fpga-repros/S01-image-perturbation-hang/` (has `run.sh`).

Two SQLite domain images differing by **one dead, never-called, empty function**:

    QEMU   uc.dom  stage 11 -> obs=1517161237      board  uc.dom  -> obs=1517161237  (5 obs)
    QEMU   dp0.dom stage 11 -> obs=1517161237      board  dp0.dom -> NEVER RETURNS   (2 obs)

Stage 11 executes only `sqlite3Strlen30` on a string literal and never calls the added function.
The hang is silent: no trap reported, no marker, core still services the console. **Nine**
structurally different perturbations of `uc` were built and every one hangs; only unmodified
builds (`uc`, `f10`) return.

**Attribution is NOT established.** The QEMU differential rules out a platform-independent
compiler/glue defect — a miscompiled `dp0` would fail there too, and does not — but QEMU is our
own model and is permissive where the RTL is not. The board may be correct while our software
relies on something it does not guarantee, and the difference may be timing rather than function.

**Tested and EXCLUDED** (see the package README for the artifact behind each): `.gct` size and
contents; carve count (8→208 synthetic, and `dvar` at 182); image size (`sz2048/8192/16384` are
all byte-identical in size to `dp0`); address of the executed code (`sqlite3Strlen30` is at the
**same** address in both); the amalgamation rewrite (byte-identical); run position (controlled
both ways); rev-node pool exhaustion (`head`=221/1021, `overflow=0`); bounds representability
(every carve representable, still hangs); operand forwarding (fix present in this bitstream).

**The debug mux is not diagnostic here** without a subtracted baseline: on the PASSING run it
reads byte-identical to the hanging run (`sw=255` `0x8f`, `sw=224` `0xff`, `sw=225` `0xd5`).

**Seven mechanisms were proposed and all seven retracted** during 2026-08-04/05. The recurring
cause was that every intervention which could observe the system also changed its behaviour.
Full trail in `ref/SILICON-BLOCKER.md`.

**A BETTER PROBE THAN THE HANG — sporadic wrong `strlen` results (2026-08-05).** Stages that
*return* are already wrong, which is cheaper and more bisectable than a hang:

    stage 13   board 15   expected 36 (5+8+11+12)    QEMU 36  CORRECT
    stage 16   board 124  expected 128 (128*5 & 0xff) QEMU 128 CORRECT

Stage 16 calls `strlen` on the **same** literal `"alpha"` 128 times and totals 636 instead of
640 — **4 of 128 calls returned 1**. So it is **sporadic (~3%), not length-dependent**; stage
13's `15 = 5+8+1+1` is the same effect at 2 of 4.

At `-O0` `strlen` re-loads the string capability with `ldc` **from a stack slot on every
iteration** (`ldc a0,0x0(a0)` → `cincoffset` → `lbu`, `strlen` at `0x14fc1c` in `uc`). At `-O1`
it would stay in a register — but `-O1` cannot build (see **C-17**).

**INFERRED, NOT ESTABLISHED:** wrong `strlen` → wrong hash in `sqlite3InsertBuiltinFuncs` →
corrupt chain → the stage-10 hang. Every link is measured *except* that last one. Do not treat
stage 10 as explained.

**CONFLICT that must be resolved before building on any of this:** `SILICON-BLOCKER.md` §0a8
records `stage 13 rc=0x24` = **36, CORRECT**, after the unaligned-copy fix, and states that
stages 11-14 are a *resolved* bug. Today's `f10` returns **15**. Either `f10` predates that fix,
or it regressed. Re-run stage 13 on a current build before trusting either number.

**Next:** ask whether the divergence survives on `caplifive_65536_nodes.bit` (and whether that
bitstream carries the forwarding fix). A waveform of `dp0` stage 11 around the hang would settle
in minutes what no software-visible observable here can.

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

### C-5 — 4 KiB code window `OPEN`
`link-gpfree.ld` forces globals to image offset `0x1000`, capping `.text` at 4096 B. One
hardcoded number, QEMU-validated at 16 KiB and 32 KiB and silicon-validated at 32 KiB. Lifting it
is what full CoreMark and Dhrystone need. Task #62.

---

> ## READ FIRST — most OPEN `R-*` entries predate the 2026-08-04 bitstream reflash
>
> The board ran `working-caplifive-captype-fixed.bit` until **2026-08-04**, when it was
> reflashed to **`caplifive_fixed_forward.bit`** (the operand-forwarding fix,
> `capstone-ariane 7aac52f93`).
>
> **R-14 and R-16 were both that one bug** — two entries that had each accumulated sessions of
> independent investigation turned out to be the same defect, and both are now FIXED and
> archived. Every other `R-*` measured before that date is therefore **suspect**: it may already
> be fixed, and its recorded mechanism may be wrong.
>
> Treat a pre-2026-08-04 `R-*` as *unverified on current silicon* until it is re-measured. Do not
> hand one to the board owner, and do not build a theory on one, without re-running it first.
> Re-running is usually one boot.
>
> Unaffected: `C-*` (compiler) and `I-*` (infrastructure) entries, which do not depend on the
> bitstream.

## Archive — fixed, kept for provenance

> **Archived below on 2026-08-05.** These are FIXED, CLOSED, SUPERSEDED or RETRACTED.
> They are kept in full for provenance -- several were re-opened once already when a
> later session hit the same shape. Nothing here is an open issue; do not hand any of
> it to the board owner as one.

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

### R-14 — struct-array init wedges `FIXED IN SILICON 2026-08-04; title is now WRONG`

> **RESOLVED 2026-08-04 — capability operand-forwarding bug, fixed in the RTL.** The fix is
> `capstone-ariane 7aac52f93` ("Fixed an operand forwarding bug", `issue_read_operands.sv`:
> capability-metadata forwarding was selected by an over-broad `check_cap_op`, narrowed to
> `check_fwd_rs1`), shipped as **`caplifive_fixed_forward.bit`**, board reflashed 2026-08-04.
>
> Verified across two valid boots with controls green: `r14sl 4`, `k800 4`, **`k1200 4`**,
> **`r14lp 4`** — i.e. both previously-failing rungs now return the correct value. The
> `SQLITE_STATIC_BUILTINS=1` workaround can come off.
>
> The 2026-08-03 reading below — that the faulting access was *architecturally legal* and the
> defect was in the RTL rather than the compiler — proved **correct**: the capability really
> was well formed, and the wrong bounds were being forwarded to the LSU.
>
> **R-16 was the same defect** reached from the other direction; see that entry.
>
> Reproducer packages archived to `capstone/tests/fpga-repros/ARCHIVED/R14-frame-pad/` and
> `.../ARCHIVED/R14-strline-struct/`. They are kept as **bitstream regression tests** — a third
> bitstream `caplifive_65536_nodes.bit` exists whose forwarding-fix status is unconfirmed, and
> `ARCHIVED/R14-frame-pad/` checks it in one boot. **Do not hand either over as an open issue.**

<details><summary>2026-08-03 report, as it stood on <code>working-caplifive-captype-fixed.bit</code></summary>

> **REPRODUCER: `capstone/tests/fpga-repros/ARCHIVED/R14-frame-pad/`** — two ~10 KB domains
> whose source differs ONLY in the size of a dead `volatile char pad[]`: `k800` returns 4,
> `k1200` never returns. Frozen `.dom` images are committed and pinned by `SHA256SUMS`. It
> supersedes `R14-strline-struct/` (1.5 MB SQLite builds, four confounded variables).
>
> **The evidence points at the RTL, not the compiler.** At the address the failing store
> targets, the capability is measurably well formed — `bnd2` returns 107 = type NONLIN with
> cursor ≥ start, cursor+16 ≤ end, and start 16-aligned; `bnds` returns 1322, i.e. **1312 bytes
> of headroom** against a 16-byte store. The identical binary computes the correct answer under
> QEMU. And every compiler-side mechanism proposed was tested on the board and refuted
> (merged string constants, repeated `ldc` from one slot, `ldc` count, capability stores as
> such, `ldc`+store in one loop, frame size alone, loops, and the non-zero `stc` immediate —
> `zoff` forces `imm=0` and still fails). So the hardware appears to fault an architecturally
> legal capability store.
>
> **NOT established** (see the repro README): permissions were never read (`lcc` field 5); the
> probes measure a capability materialised for a `volatile` access rather than provably the
> faulting `stc`'s own base register; and no `mcause`/`mepc` has been read for `k1200` itself —
> the `mcause=28` below came from a different, SQLite-derived domain.

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

- **Repro:** `tests/fpga-repros/ARCHIVED/R14-strline-struct/` (source, run recipe, and the rebuild
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

</details>

---

### R-16 — domain never returns from its FIRST entry (`SHA5` stall) `FIXED IN SILICON 2026-08-04`

> **RESOLVED 2026-08-04 — the same capability operand-forwarding bug as R-14**
> (`capstone-ariane 7aac52f93`), shipped as **`caplifive_fixed_forward.bit`**.
>
> The reproducer — a SQLite domain built with `SQLITE_STATIC_BUILTINS=1` — entry-stalled
> **8/8** on `working-caplifive-captype-fixed.bit` and **enters** on the new bitstream, on a
> boot whose control both entered and returned.
>
> Note the coupling: `SQLITE_STATIC_BUILTINS=1` **is** the R-14 workaround
> (`build-sqlite-silicon.sh:75`), so **R-16 was created by the R-14 workaround**. Two symptoms,
> one defect.
>
> **We never root-caused it ourselves.** Eight axes were eliminated (image size, carve count,
> the size+carve conjunction, `dom_data` geometry, blob size on the ladder path, the loader,
> the interp-glue pad bug, `BUILTIN_LIMIT`) and none separated entering from stalling images;
> the mechanism came from the RTL side. Worth stating plainly, because that eliminated-axes
> list reads like progress and was not. In hindsight the strongest hint was the observation
> below that the *same binary in the same boot* entered for `:0` and stalled on a later
> selector — a per-invocation effect, which is what a pipeline forwarding hazard looks like,
> and which the "deterministic per image" framing actively obscured.
>
> **Reproducer package: `capstone/tests/fpga-repros/R16-entry-stall/`** — source, build recipe,
> pinned image hashes and a `run.sh` that builds, stages and runs it behind a control gate and
> prints a present/absent verdict. Kept as a **bitstream acceptance test**: a third bitstream
> `caplifive_65536_nodes.bit` (larger revocation-node pool) exists whose forwarding-fix status
> is **unconfirmed** — if it lacks the fix, R-16 *and* R-14 both return.
>
> The classification rule below is NOT superseded and still governs every board run: `SHA5`
> last does not by itself mean an entry stall — **distinguish on `SQ: G/enter`**.

<details><summary>2026-08-03 investigation record, as it stood on <code>working-caplifive-captype-fixed.bit</code></summary>

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

</details>

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

### R-19 — a `movc rd, zero`-sourced store leaves `compress_cap(NULL)` IN its own bank-1 slot `OPEN — trigger established on silicon; MECHANISM NOT confirmed (simulation is clean); NOT yet reported`

**Separate from R-18 on purpose.** R-18, already reported, is the **zeroing** form: the victim is
written with `0` and counts up, and raw full-width readbacks (`craw` = `0x00000237`, `graw`, `gztr`)
show **no metadata anywhere**. R-19 is a **different observable**: the victim comes back holding
`compress_cap(NULL) + n`. Same trigger class and the same workaround clears both, but the signatures
differ, so they are tracked apart — the R-18 report already sent describes the zeroing form, and
folding this into it would misinform the owner.

**Measured on silicon**, `k800` control green in every boot, damaged arm reproduced on **three**
boots at two entry VAs:

| rung | build | returned |
|---|---|---|
| `fdp0` | accumulator initialised by `movc a0, zero; sw`, `-O0` | **`0x08000A31`** = `0x08000000` + 2609 |
| `fdp0fix` | same, initialised by `addi a0, x0, 0` | **2609** clean |
| `fdpraw` | returns the accumulator alone (no second term) | `0x08000A31` — the victim IS that slot |
| `fdpO1` | `-O1`, accumulator kept in a **register** | **2609** clean |

`0x08000000` is `compress_cap` of a null capability (`ariane_pkg.sv:754-772`) — a HARDWARE encoding
the program cannot materialise; it only ever writes `0` there. QEMU computes 2609 for the same
binary. So the **trigger** is a store whose data register carries a null-capability metadata shadow,
and the **immunity condition** is the accumulator's storage class: register-resident is clean,
memory-resident at row offset 8 is damaged.

**MECHANISM NOT CONFIRMED.** A directed Verilator test at the same geometry —
`fpga-repros/R19-movc-zero-metadata-in-slot/sim/movc-zero-self-clobber.S`, bank-1 slot at row offset
8, `movc`-zero initialiser, 64 increments, RMW row-mate, witnesses either side — returns **SUCCESS
in 1715 cycles**. The simulated RTL does not write metadata into the slot. The chain readable in the
source (`issue_read_operands.sv:1140` → `wt_dcache_mem.sv:138` → `:158`) fits every board
observation but is **not** reproduced, so it is not claimed. Untested candidates for the divergence:
the resident bitstream may not match this RTL revision; the board runs inside a capability domain
after `capenter` while the test is bare metal; or the test lacks a co-factor (no capability traffic
in the loop, no indirect calls, no cap-init).

**TWO CORRECTIONS TO WHAT WAS ALREADY SENT FOR R-18**, both found after the report went out:

1. The **`R XOR 8` splash rule is withdrawn.** It is arithmetically "the victim is 8 bytes from the
   trigger"; the corpus splits into distance-8 builds where it holds and distance-4 builds where it
   fails (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`), and distance is invariant under alignment.
2. The **dual-bank chain is not confirmed** — see the Verilator result above.

The reproducer, the trigger and the workaround are all unaffected by both.

Repro: `capstone/tests/fpga-repros/R19-movc-zero-metadata-in-slot/`.
Workaround (shared with R-18): `design/R18-workaround-movc-zero.md`.

### R-20 — after `stc`, a load into **x10** is read by the NEXT instruction as the store's base address `FIXED IN SILICON, verified 2026-08-10 on caplifive_r20.bit`

**VERIFIED ON HARDWARE.** The one-character RTL fix (`issue_read_operands.sv:568`, `=` -> `|=`,
`capstone-ariane` branch `r20-fix`) is in `caplifive_r20.bit` and clears the defect:

| test | before | on caplifive_r20.bit |
|---|---|---|
| `sbx8` -- the 13 KB rung repro | `0xD0000001` | **`0xD0000000`** |
| `Z.dom` -- the SQLite-level site | WEDGE | **RETURNED** |

Both instrument-validation arms of the rung stayed clean and the k800 control was green in the
same boot, so the run carries a verdict.

**The compiler workaround (`30c275b5d781`) WAS REVERTED** by `cdbb92360e2b`; `llvm/` is
byte-identical to its pre-workaround state and validation passed (lit 47/47 with 0 XFAIL, silicon
ladder 6/6 oracle-matched). The standing TODO that used to track it is discharged and removed.
The R-20 package's `WORKAROUND.md` is kept as the record of what was done and what was rejected,
not as a description of the current build.

`stc rX,0(a0)` immediately followed by `ld a0,0(a0)` immediately followed by any consumer of `a0`
gives that consumer **the address the store used**, not the value the load fetched. Memory is
correct; a consumer one instruction later is correct; the same sequence on any other register is
correct. Silent — no trap, nothing in any log — and correct under QEMU.

Normally invisible, because the stale value and the correct value are both non-zero at
essentially every site. It changes behaviour only where the loaded value is genuinely **zero**,
i.e. the `if (pointer)` idiom right after a call returns NULL. There are **736** instances of the
exact shape on x10 in the SQLite silicon image; one of them is the S-03 blocker.

**This is the root cause of S-03.** See the S-03 entry above for the investigation trail and for
the several models that were refuted along the way.

Reproduced standalone in a 13 KB rung, three draws, `retval = 0xD0000001` — no SQLite required.
Necessary conditions, each from a one-variable pair: register is x10 (`R13` on `a3` is clean);
store is a **capability** store (`sd` is clean); both adjacencies hold (one `nop` either side
cures it); the branch target is irrelevant (`Z` differs from base by one byte). The poisoned value
is **measured**: `V1` returns iff the value read was exactly the store's base address.

Suggested RTL sites, INFERRED from board behaviour and not yet confirmed in simulation:
`issue_read_operands.sv:568` overwrites x10's clobber entry unconditionally (`=` where the intent
looks like `|=`), and `issue_read_operands.sv:674-677` with `check_fwd_rs1` (which includes `STC`)
serves a reader whose `rs1` matches an in-flight STC's `rs1` with that STC's `rs1_cursor`. Which
of the two to change is NOT decided by anything measured so far.

**Fix:** `capstone-ariane` branch `r20-fix` (`2efb3604f`, on `e1b3db6ba`) changes
`issue_read_operands.sv:568` from `=` to `|=` so the CAPENTER x10 special case ADDS to the
clobber set instead of replacing the generic rs1 claim. Every arm of the directed test goes
correct, in the same 558 cycles; capenter/stc/capldc/cap-overwrite/cincoffset all still pass.
Needs pushing (`git -C capstone/capstone-ariane push -u origin r20-fix`) and a bitstream.

**WORKAROUND STATUS: none landed, and nop padding is NOT one.** Measured both ways: the board
cured it with ONE nop, simulation needs FOUR, on both sides of the load. The window is
context-dependent, so a fixed nop count is a workaround that works in one setting and silently
fails in another -- worse than none. The only cure that holds on board AND simulation, and is not
a timing window, is keeping x10 out of the capability store's base register (`R13`, sim arm B).
Two routes, neither implemented, both needing an LLVM rebuild and revalidation:
(a) give `STC`'s address operand a register class excluding X10 -- `CapstoneInstrInfo.td:2402`
uses `GPRMem`, which every load and store shares, so it needs a new class plus a new MemOperand;
(b) fix the i128 `SELECT_CC` gap (`CapstoneInstrInfo.td:1741-1747`) so the amalgamation can build
at `-O1`, which removes the spill pattern wholesale -- the build script already documents that
`-O1` eliminates it and already uses `-O1` for the string primitives.

Exposure in the current SQLite silicon image: 3657 capability stores based on `a0`, of which
**2186** are immediately followed by a reader of `a0`. Almost all are invisible because the stale
and correct values are both non-zero there; it bites only where the loaded pointer is NULL.

### R-18 — a scalar in the UPPER half of a 16-byte cache row is silently ZEROED `OPEN — REPORTED to the board owner; our compiler workaround is landed and silicon-confirmed`

> **STATUS 2026-08-08.** This issue has been **REPORTED to the board owner**, and our side is
> **worked around**: `-capstone-int-zero-for-zero-copy` (see `design/R18-workaround-movc-zero.md`),
> silicon-confirmed — `c8` 567 -> `c8fix` 576, one instruction apart, control green.
> **R-18 is also EXCLUDED as the SQLite blocker** (2026-08-08, board): the SQLite hang path's five
> triggering sites are all removed by the workaround and the wedge persists unchanged.
>
> **Do not re-open this entry to record mechanism work.** A second, DIFFERENT signature found on
> 2026-08-08 — the victim holding `compress_cap(NULL) + n` rather than being zeroed — is tracked
> separately as **R-19**, because the report already sent describes the zeroing form and mixing the
> two would misinform the owner. Two corrections to what was sent are noted in R-19.


> **2026-08-08 — the "better lead" below is CONFIRMED, by a single-variable pair rather than a fit.**
>
> Three boots, each with a passing control first and every rung entering. `c8` returned 67699255 on
> all three (ten consecutive boots overall).
>
> | arm | frame | qc | p | k | `stc` | result |
> |---|---|---|---|---|---|---|
> | `c8` | 0x50 | s0−0x34 | s0−0x38 | s0−0x3c | s0−0x50 | **567** |
> | `rg16` | 0x60 | s0−0x34 | s0−0x48 | s0−0x4c | s0−0x60 | 576 |
> | `rmB` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x4c** | s0−0x60 | 576 |
> | `rmC` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x3c** | s0−0x60 | **567** |
>
> **`rmB` vs `rmC` is the result.** Same frame, same victim address, same `p`, and the capability
> store two rows away in BOTH. `k` moves 16 bytes; the answer flips 576 ↔ 567. Cycle counts confirm
> all arms ran the same ~576 iterations, so nothing was cured by doing less work.
>
> That also refutes the competing reading — that what mattered was the store's row being adjacent to
> the victim's row — since `rmC` has the store two rows away and is damaged anyway.
>
> **~~The geometry, sharply:~~ RETRACTED within hours, by the very next experiment.** A "bank 1 at
> lanes L is zeroed by an RMW scalar in bank 0 at the same lanes L" refinement was recorded here on
> the strength of a 5/5 + 7/7 corpus fit. The control built to test it **refuted it**, and this is
> the seventh confound of the same class: a fit was mistaken for a mechanism.
>
> **2026-08-08 (later) — REGION IS EXCLUDED, and lane geometry is NOT the discriminator.**
> Stage 37 puts the trigger in a GLOBAL: `gc[16]`, 16-byte aligned, victim `gc[3]` at row offset 12,
> plus a second global scalar RMW'd in the same row. Board, control first, all four rungs entered:
>
> | arm | row-mate | victim | |
> |---|---|---|---|
> | `k800` | — | 4 | control OK |
> | `c8` | k @+4 | **567** | anchor — 11th consecutive boot |
> | `gnt` | `gc[2]` @+8 — bank 1, lanes 0-3 (*intended control*) | **9** | **damaged, severely** |
> | `gtw` | `gc[1]` @+4 — bank 0, lanes 4-7 (the "twin") | **567** | damaged, `c8` signature |
>
> 1. **A GLOBAL victim loses increments exactly as a stack one does.** Region is excluded by direct
>    measurement. The 2026-08-07 "must be on the domain stack" reading is dead twice over.
> 2. **The same-lanes rule is refuted.** `gnt` carries the row-mate at offset 8 — same bank as the
>    victim, wrong lanes — and is damaged *more* severely than the "twin" arm, its victim reset on
>    every outer pass rather than once.
>
> **What actually discriminates, on present evidence: the row-mate's STORE PATTERN, not its
> position.** `rmB` and `gnt` place the row-mate at the SAME offset (+8, same bank as the victim);
> `rmB`'s `p` is only incremented once per outer pass and is CLEAN, `gnt`'s is re-zeroed at the top
> of every outer pass and is DAMAGED.
>
> **2026-08-08 (later still) — THE TRIGGER IS ESTABLISHED; THE MECHANISM IS NOT.**
>
> Four arms, identical victim/row-mate addresses, identical store counts, same region, in-boot
> `k800` control and `c8` anchor in every boot:
>
> | arm | row-mate's per-pass reset store | outer-loop length | victim |
> |---|---|---|---|
> | `gz0` | `movc a0, zero; sw` | short | **9 — damaged** |
> | `gzn` | `movc a0, zero; sw` + 2 nops | **padded to match** | **9 — damaged** |
> | `gzl` | `ldc; lw; sw` — stores the VALUE ZERO from a load | padded | 576 — clean |
> | `gzs` | `lui; addi; sw` — nonzero | padded | 576 — clean |
>
> **The trigger is the capability METADATA on the store's data register.** Not the stored value
> (`gzl` stores zero and is clean), and not the outer-loop instruction count (`gzn` is padded to a
> clean arm's length and is still damaged — this was a real confound, caught by audit, and it is now
> excluded). The producing instruction is `movc rd, zero`, whose `compress_cap` of a null capability
> is `0x08000000`.
>
> **The RTL path for the trigger is traced and holds:** `movc` is a capstone-FLU op →
> `commit_stage.sv:279` writes `result_metadata` into the cap-metadata regfile under the INTEGER GPR
> write-enable (`issue_read_operands.sv:1663-1665`, `.we_i(we_pack)`) → `:1140`
> `cap_data.cap_metadata_b` is taken UNGATED by opcode → `load_store_unit.sv:1013` → `store_unit.sv:345`
> → `store_buffer.sv:173` → `wt_dcache_mem.sv:138` `st_wr_cap = |wr_user_i`. An ordinary `sw` is
> therefore classified as a capability store by VALUE.
>
> **But the recorded dual-bank chain is REFUTED as the corruption path, by direct measurement.**
> If bank 1 received `wr_user_i` (`wt_dcache_mem.sv:158`) the constant `0x08000000` would appear in
> memory. Raw, unmasked readbacks say it does not, anywhere:
>
> | probe | reads | raw value |
> |---|---|---|
> | `craw` | stack victim, `c8` geometry | `0x00000237` — clean count |
> | `graw` | global victim | `0x00000009` — clean count |
> | `gztr` | the row-mate (twin) itself | `0x00000009` — clean count |
>
> The victim is written with **zero** and counts up; no metadata value lands in the victim, and none
> lands in the twin. That also refutes the write-buffer 8-byte-merge candidate, whose specific
> prediction was `twin = 0x08000009`.
>
> **Instrument correction:** every earlier R-18 number masked the victim to 16 bits, so "lost N
> increments" and "overwritten with metadata, then counted up" were indistinguishable in all of them.
> The raw reads above settle it in favour of the former. `0x08000237` from the older `gz12` build is
> a separate observation and is not reproduced by any arm here.
>
> **Also corrected:** retraction item 3 above reads the `bar1` datum as refuting a metadata
> mechanism. That reading is withdrawn — the `lw` between `bar1`'s `movc` and its `sw` scrubs the
> destination's shadow (a non-FLU writeback carries `cap_result = '0`, `scoreboard.sv:246`), so
> `bar1` never produced a tainted store and could not have tested this. The same false premise was
> in `fdreg_kernel.h` and is fixed there.
>
> **2026-08-08 (final) — MECHANISM CONFIRMED IN SIMULATION, WORKAROUND CONFIRMED ON SILICON.**
>
> An ordinary `sw` whose data register carries capability metadata also writes its data into the
> **same byte lanes of the other bank** of its 16-byte row. Reproduced in Verilator in ~13 s
> (`scalar-store-movc-zero.S`); the RVFI trace shows only two architectural accesses to the
> corrupted slot in the whole run — the seed, and the readback returning zero.
> **The `R XOR 8` rule is WITHDRAWN (audited same day).** It is just "the victim is 8 bytes from the
> trigger"; the corpus splits into distance-8 builds where it holds (10) and distance-4 builds where
> it fails (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`), and distance is invariant under alignment so
> no carve-alignment argument rescues it. What survives are NECESSARY conditions only: the damaged
> scalar is in the trigger store's own 16-byte row, in bank 1, at offset 8 or 12. Which of two
> bank-1 candidates is hit is not predicted.
>
> **And the simulation does not reproduce the board's symptom.** At `gz0`'s geometry the sim leaves
> the distance-4 slot exact (576) and zeroes the distance-8 slot; the board damages the distance-4
> slot (`graw` = 9 raw). The sim shows a real dual-bank splash but not, so far, the board's. One
> boot with a witness at `gc+0x0` settles whether there are two effects or two different faults.
>
> **Workaround, silicon-confirmed.** `c8` and `c8fix` are the same source at the same frame
> geometry (frame 80, rmw [20,24,28], accumulator still at the damaged row offset 12), differing by
> ONE instruction — `movc a0, zero` vs `addi a0, x0, 0`:
>
> | rung | qc | cycles | |
> |---|---|---|---|
> | `k800` | — | 4743 | control OK |
> | `c8` | **567** | 44116 | damaged, 15th consecutive boot |
> | `c8fix` | **576** | 44075 | **cured** |
>
> Behind `-capstone-int-zero-for-zero-copy`, **default OFF**: flag-off is byte-identical on 4/4
> rungs, QEMU ladder 6/6 both ways, lit 47/47. Two blockers before any default-on, both recorded at
> the flag declaration: it also converts genuine null-CAPABILITY materialisations, whose ISA
> semantics under `stc` are unverified; and two lit tests FileCheck the literal `movc`.
>
> **It removes the common case, not the class.** `scan-r18-trigger.py` measures this: `c8` has 7
> trigger sites, `c8fix` has 2 — the loop`s four `movc rd, zero` gone, two register-to-register
> `movc` in `domain_main` remaining.
>
> **R-18 does NOT explain the documented silicon miscompiles.** `matmult_int` (R-1) and
> `beebs_recursion` both have ZERO trigger sites. (`matmult_int` does contain nine `movc rd, zero`
> — an earlier claim that it had none came from a `grep -c` that silently returned 0 — but none of
> them feeds a store, which is what the trigger requires.)
>
> **Two corrections to the record made in the same session.** (1) An earlier reading of this
> experiment claimed the arms' absolute addresses moved with the frame; they do not — `s0` is the
> caller's `sp` and the victim is at `s0-0x34` in every arm, so the victim's address, D-cache set and
> bank-row are EXCLUDED, not confounded. (2) The 2026-08-07 "boot 68" conclusion that the damaged
> scalar *must be on the domain stack* is the sixth confound of the same class: its global
> accumulator `gcnt[3]` is the only RMW'd word in its row, so the row-occupancy rule predicts it
> clean without invoking region at all. **Region and provenance are still untested** — every arm in
> these boots is a stack build. The next build needed is a GLOBAL scalar carrying the bank-0 twin.
>
> Trail: `history/07-08-2026_23-55-00_r18-localized-to-row-mate-traffic.md`.

> **RETRACTION 2026-08-07 (same day, before handover).** A causal chain was recorded here and in a
> defect report, and an adversarial audit refuted it. **The report was NOT sent.** What was wrong:
>
> 1. **The claimed asymmetry between the two forwarding ports does not exist.** The scoreboard-port
>    "validity gate" (`issue_read_operands.sv:765`) has `cap_result.result_metadata` in BOTH arms of
>    its ternary — it does not sanitise to zero. The proposed fix ("gate the WB forward on validity,
>    matching the scoreboard version") **would have changed nothing.**
> 2. **There is no demonstrated source of stale metadata on an ordinary store.**
>    `ex_stage.sv:1081` is `capstone_flu_result_o = capstone_flu_valid_i ? '{...} : '0`, so an
>    ordinary `addi` forwards ZERO metadata on both ports. `wr_user_i != 0` on a scalar store has
>    never been measured anywhere — board, simulation or waveform.
> 3. **Our own barrier experiment refutes it.** `movc rd, zero` does NOT write zero into the shadow;
>    `compress_cap` of a null capability is **`0x08000000`** (`ariane_pkg.sv:753-834`). So under the
>    claimed mechanism the `bar1` arm should have pinned the accumulator near zero on every
>    iteration. It returned 567, bit-identical to its `nop` control. That was a refuting datum and
>    it was read as inconclusive.
> 4. **Matched builds kill the geometry as a cause.** `c8` (qc@0x1c) loses 9 while `gp16`/`gp32`/
>    `t16` (qc@0x2c, 0x3c) are exact — same bank, same byte lanes, same instruction. Roughly 10
>    undamaged upper-half slots against 9 damaged ones. The upper-half rule is a NECESSARY
>    CONDITION, not an explanation.
> 5. **The `clobber + (576 - reset)` table is an arithmetic identity, not a fit** — two free
>    parameters per observation, so any value decomposes. It also omits the builds that do not fit
>    (+11, +330 are not multiples of 9).
>
> **A better lead came out of the audit:** all three measured reset points — 9, 72, 558 — are
> multiples of 9, i.e. they land exactly on OUTER-PASS BOUNDARIES (p ≈ 1/729 under a uniform null).
> Something that happens once per outer pass fits far better than the victim's own store: the
> `k = 0` re-initialisation, whose store at the shift8 geometry sits at `0x14` (bank 0, lanes 4-7)
> and whose dual-bank splash target is exactly `0x1c`. That is a DIFFERENT mechanism and is untested.


A plain `sw` whose address lies in the upper 8 bytes of a 16-byte D-cache row can have **its own
slot written with capability metadata instead of its data**. Where those metadata bytes are zero at
the store's byte lanes the variable is **silently ZEROED** — no trap, no tag violation, nothing in
any log. Present at the resident bitstream's commit (`7aac52f93`) and at `capstone-ariane` HEAD;
`git diff` touches none of the files involved.

**Chain (each line quoted in the report):**
`issue_read_operands.sv:690` forwards rs2 capability metadata from the **writeback port with no
validity gate** (its scoreboard-port sibling checks `cap_result.valid`) → `wt_dcache_mem.sv:138`
classifies a store as a capability store **by VALUE** (`st_wr_cap = |wr_user_i`), not by opcode →
`:230-238` a classified store writes **both** banks of the row → `:156-158` **bank 1 is the only
bank** that can receive `wr_user_i` instead of the store data.

**Evidence.** Victim in the upper half in **9/9** directly-measured builds (undamaged builds also
carry upper-half scalars, so it is a real constraint). A sentinel-initialised accumulator
(1,000,000) returns **567**, proving overwrite rather than skipped stores. Every victim decomposes
as `clobber + (576 − reset_iteration)`; one build returns `0x08000237 = 0x08000000 + 567`, i.e.
clobbered with metadata **bit 27** set. Cycle counts independently confirm iteration counts. QEMU is
correct throughout.

**Software impact.** Any `-O0` code mixing capability traffic with ordinary scalar locals is
exposed; which variable is hit depends only on where the allocator puts it. A loop-control variable
in the affected slot produces **extra iterations** rather than a wrong value.

**NOT reproduced in Verilator** at either RTL revision — the directed tests never create the trigger
(stale WB-forwarded metadata on a scalar store's rs2). Stated as the report's main gap.

Report + reproduction: `history/07-08-2026_RETRACTED_scalar-store-metadata-mechanism.md`.
Trail: `history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
Fix: gate the WB forward on validity, and/or classify by opcode. Both need a bitstream reflash —
the project lead's call.
---

### R-21 — `cincoffset`/`scc`/`tighten`/`shrinkto` do not consume their LINEAR source, and `init` DUPLICATES it `OPEN — SPEC VIOLATION, confirmed in RTL simulation 2026-08-11; NOT yet reported`

**A linear capability can be copied.** `capstone-spec/parts/intro.adoc:58-61` states the invariant
normatively -- "instructions can only **move, but not copy**, linear capabilities between
general-purpose registers" -- and the spec defines each instruction below as `MOVC rd, rs1` plus an
edit, where `MOVC` writes `cnull` to a non-NONLIN source (`cap-man-insn.adoc:36-38`). `MOVC` does
this (`capstone_flu_unit.anvil:20-24`). These do not:

| insn | anvil site | `rs1` slot of `create_result_pack` | |
|---|---|---|---|
| `CINCOFFSET` | `capstone_flu_unit.anvil:41` | `rs1` unmodified | not cleared |
| `CINCOFFSETIMM` | `capstone_flu_unit.anvil:60` | `rs1` unmodified | not cleared |
| `SCC` | `capstone_flu_unit.anvil:81` | `rs1` unmodified | not cleared |
| `SHRINKTO` | `capstone_flu_unit.anvil:205` | `rs1` unmodified | not cleared |
| `TIGHTEN` | `capstone_dyn_unit.anvil:220` | `rs1` unmodified | not cleared |
| **`INIT`** | **`capstone_flu_unit.anvil:112`** | **`create_result_pack(...,rd,rd)`** | **duplicates** |
| `MOVC` / `SEAL` / `SHRINK` / `SPLIT` | | | conformant |

`INIT` is the severe one: with `rd != rs1` it writes the **newly created LINEAR capability** into
`x[rs1]`, where the spec mandates `cnull`. That is outright duplication rather than a missing
clear. `create_result_pack(id, ex, rs1, rd)` maps arg 3 to `cap_rs1` and arg 4 to `cap_result`
(`capstone_unit.anvilh:348-362`); nothing downstream rewrites it (`ex_stage.sv:1083` ->
`cva6.sv:1370` -> `scoreboard.sv:242` -> `commit_stage.sv:278` is pass-through apart from the
CCSRRW gate at `commit_stage.sv:385`).

**Repro, ~12 s, no board:** `verif/tests/custom/capstone/linear-clear-audit.S` in `capstone-ariane`,
registered in `testlist_capstone.yaml`. Run per the `rtl-sim` skill. Result on
`capstone-bootstrap` @ `aef2baa79`, **541 cycles, 0 exceptions** (not the 2000013 timeout):

| arm | what it does | result | |
|---|---|---|---|
| 0 | `MOVC`, LINEAR source -- **instrument control** | `Reg[15]: 0000000000000000` | source CLEARED |
| 1 | `CINCOFFSET`, NONLIN source -- **conformance control** | `Reg[17]: ... Type : 2` | source SURVIVED, correct |
| 2 | `CINCOFFSET`, LINEAR source -- **the probe** | `Reg[19]: ... Type : 1` | **source SURVIVED = DUPLICATION** |

Arms 0 and 1 are what make arm 2 readable: arm 0 proves `CAPPRINT` can see a cleared source at all,
and arm 1 proves the core does not clear unconditionally. Arms 1 and 2 differ in exactly one thing,
the source type, so that difference is the variable. **No pre-existing test covered this** --
`cincoffset.S:22` sets its source to `CAP_TYPE_NONLIN` first, so it is the negative case only.
QEMU is no reference either: `trans_csldc`/`trans_csstc` write no `cnull` at all.

**Impact.** One 1-cycle instruction, no exception, yields a second capability with the same
`revnode_id`. That defeats **exclusive access**, the first of the architecture's goals. Revocation
is NOT defeated -- there is no refcount and aliases share a `revnode_id`, so one `revoke`
invalidates every copy -- so do not offer revocation as a mitigation, and do not claim it breaks.

**Do NOT simply fix this in RTL.** Our own code depends on the non-conformant behaviour.
`tests/scan-linear-clear-exposure.py` scans built artifacts for sites a conformant clear would
change; over the 76-image `.dom` corpus it finds **292 hits in only four distinct shapes**, all in
the domain entry glue and repeated across 31 images:

```
BREAK   rule-A  x70  cincoffset s2, gp, zero    (gp minted LINEAR by split; read again at stc gp, 0x10(s1))
BREAK   rule-A  x70  cincoffset t6, t2, zero    (t2 minted LINEAR by split; read again at stc t2, 0x0(s2))
SURFACE rule-B  x70  stc gp, 0x10(s1)
SURFACE rule-B  x70  stc t2, 0x0(s2)
```

LLVM does not model the clear -- `CapstoneInstrInfo.td:2413` takes `CIncOffset`'s `rs1` as a pure
use -- so a conformant `CINCOFFSET` would null `gp` in the entry glue of essentially every domain
we build. A real fix needs paired compiler work and is its own project.

**Two implementation gotchas for whoever does fix it.** `check_fwd_rs1` (`ariane_pkg.sv:929-935`)
lists only `{SPLIT,MOVC,CJALR,CCSRRW,STC}`; a clear added without a matching entry is forwarded
around and is architecturally invisible (`SEAL` is already in that state). And `MOVC`'s
`data.rs1 == data.rd` guard must be kept, or `cincoffsetimm sp, sp, -96` would null `sp`.

**Counter-evidence, recorded rather than buried.** The spec never states explicitly that a named
instruction inside a numbered step executes with its side effects; the reading rests on the spec
using "Write `x[rs1]` to `x[rd]`" (SPLIT `:309-315`, MREV `:541-542`) exactly where the source must
survive and `MOVC rd, rs1` where it must be consumed, and on `CALL` (`ctrl-flow-insn.adoc:121`)
being absurd otherwise. Upstream `capstone-qemu` implements the clear for this family
(`op_helper.c`, commit `b23d516401`, 2023) -- **except `helper_csshrinkto` (`:833-847`), which does
not**, so the reference model is itself inconsistent on one instruction.

### R-22 — `stc` does not write `cnull` to its register source `OPEN — SPEC VIOLATION; NOT yet reported`

`capstone-spec/parts/mem-access-insn.adoc:105`: "If `x[rs2]` is a capability and `x[rs2].type` is
not `1` (non-linear), write `cnull` to `x[rs2]`." The RTL does not.

The decoder aliases `STC`'s `rd` field to `rs2` (`decoder.sv:1308-1314`), which looks like it could
be the clear -- **it is not.** `capstone_dyn_unit.anvil:419` (normal path) and `:410` (uninitialised
path) both build `create_result_pack(..., rs1_v, rs2_v)`, and `rs2_v` is the **unmodified** stored
capability. It cannot be `cnull`, because the same field is the store data handed to
`cap_store_ri.req` on the next line. So the `rd`-aliased writeback is a self-write of the same
value. The store-syncer return path preserves it (`capstone_unit.anvilh:587-591`).

Note for anyone re-deriving this: `commit_stage.sv:278` drives the second capability write port
from `commit_instr_i[0].rs1`, i.e. it targets **`rs1`, not `rs2`** -- and for `STC` that port is
load-bearing, delivering the uninitialised-capability cursor advance (`dyn_unit.anvil:409-411`).
The accurate statement is "no port targets `rs2`, and the `rd`-aliased port carries the unchanged
value", not "no port other than `rd`".

**Positive control that the intent is the opposite:** the *memory*-side counterpart of the same
rule (`LDC`, `mem-access-insn.adoc:54-55`) **is** implemented, over exactly the non-NONLIN type set
(`load_unit.sv:447-453`). One half of the rule is in, the mirror half is missing.

**Same caveat as R-21: do not fix this alone.** `CapstoneInstrInfo.td:2402` declares
`STC ... (outs)` -- empty -- so LLVM believes `rs2` survives an `stc`. The exposure scan finds 140
rule-B sites across the corpus. QEMU also omits this clear
(`trans_capstone.c.inc:172-193` -> `op_helper.c:1192-1199` touch only the memory map), so **no
in-tree model currently distinguishes conformant from non-conformant behaviour** and a fix should
land in both or note the divergence.

### R-23 — `ldc` never checks WRITE permission, so a READ-ONLY capability can move a linear capability out of memory `OPEN — SPEC VIOLATION; NOT yet reported`

`capstone-spec/parts/mem-access-insn.adoc:44-46` requires `Insufficient capability permissions (27)`
when the loaded value is not a scalar or a non-linear capability and `2 <=p x[rs1].perms` does not
hold. `capstone_dyn_unit.anvil:314` implements the **read** check (`perm & 4`) only; there is no
second arm anywhere in `func LDC` (`:293-353`).

**What was disabled is a DIFFERENT check, and the distinction matters.** The commented-out block at
`capstone_unit.anvilh:571-585` references only `rd`, the loaded value; `cap_msg.cap_rs1.metadata.perm`
appears nowhere in it and there is no `2 <=p perms` conjunct. It would have faulted **every** `ldc`
of a linear capability -- the central Capstone operation -- so its `FIXME: overly restrictive` is
accurate. The correct framing is therefore: *the spec-required write-permission check has never
been implemented; what was disabled was a broken unconditional one.* Enabling that block verbatim
would not fix this and would break linear loads.

**Why it is authority amplification, not data loss.** `TIGHTEN` (`cap-man-insn.adoc:320-351`,
`capstone_flu_unit.anvil:326`) makes `perms = 4` (read-only) reachable from ordinary domain code.
Given a read-only `c` over a region holding a linear `L`:

1. `ldc rd, 0(c)` succeeds -- the holder of a **read-only** capability obtains `L` with `L`'s full
   authority, and the true owner loses it;
2. the RTL then writes `cnull` over that slot (`load_unit.sv:452` -> `store_unit.sv:399`, which
   injects straight into the store buffer and bypasses `check_store_data` entirely) -- **a write
   through a capability with no write permission.**

So read-only sharing does not confer read-only semantics: a reader can destroy 16 bytes per `ldc`
anywhere it can read, and can move linear authority out of a region it was only permitted to read.

**Related observation from the R-21 run, recorded because it is not yet explained.** In
`linear-clear-audit.S` arm 3 the `ldc` linear-clear **does fire** -- the slot's low 8 bytes read
back `0x0`, while the NONLIN control arm 4 is unchanged at `0x0000000080001000`, so the clear is
conditional and not unconditional. But the slot's **high 8 bytes survive** (`0x0000000b98044000`),
i.e. the clear is *incomplete*. That is consistent with S-06's mechanism -- the clear store sets
`user = '0` (`store_unit.sv:414`), so `st_wr_cap = |wr_user_i` is 0 and only the bank matching the
offset is written. Whether the granule's tag also goes clear (which would make the residue
harmless) was **not measured** and must be before anyone concludes either way.

### R-24 — the FLU/DYN exception encoder is +1 off the spec, so every capability `mcause` from the execute path is wrong `OPEN — SPEC VIOLATION, direction now determinate; NOT yet reported`

**RESOLVED 2026-08-12 against `capstone-spec`, and the answer is the opposite of the first guess
recorded below.** `capstone-academic-spec/parts/int-except.adoc:19-27` gives the authoritative table:

| Exception | spec code |
|---|---|
| Unexpected operand type | **24** |
| Invalid capability | **25** |
| Unexpected capability type | 26 |
| Insufficient capability permissions | 27 |
| Capability out of bound | 28 |
| Illegal operand value | 29 |
| Insufficient system resources | 30 |

With `ex_code` ordinals `NO_EXCEPTION = 0, UNEXPECTED_OPERAND = 1, ...`, the spec's numbering is
**base 23**. Therefore:

* `commit_stage.sv:205-228` (base 23) is **SPEC-CONFORMANT**.
* `ex_stage.sv:469` (FLU) and `cva6.sv:1360` (DYN) use **base 24** and are **+1 on every capability
  exception the execute path raises** — which is nearly all of them.
* `riscv_pkg.sv:349-353` encodes the same +1 error (`UNEXPECTED_OPERAND_TYPE = 25` where the spec
  says 24), so it corroborates the RTL's behaviour, not the spec.

**THE WEDGE HAS NOW BEEN REPRODUCED AND IT REPORTS `mcause 25` ON THE WEDGE ITSELF (2026-08-12).**
One boot, control green, three domains in ascending order:

| domain | result |
|---|---|
| `k800` control | `retval=4` in 2 s — **boot VALID** |
| `sqfixoff` (library fixup OFF) | entered, **RETURNED** `SQLITE ERROR stage=create rc=11 message=malformed` |
| `sqwedge` (`SQLITE_LDC_HIGH_HALF_FIXUP=1`) | `SQ: G/enter`, **NO RETURN in 300 s — WEDGED** |

Debug-mux readout taken at the wedge, before releasing the board:

* `sw=255 TRAP LOG {seen, mcause[6:0]} = 0x99` → `seen=1`, **`mcause = 0b0011001 = 25`**.
* `sw=224` → `privM=1`, `flush=1`, `flu_ready=1`, `dyn_ready=1`, `lsu_ready=1`, `ex_commit.valid=1`.
* `sw=249/250` → `rev_node_head = 0x1a1 = 417`, `overflow = 0`. **Fourth independent confirmation
  that pool exhaustion is not the mechanism** (417 of 65536).
* `sw=230-237` → `commit pc = 0x2`, the usual junk sentinel. Says nothing.

Read with the measured decoding above, `mcause 25` from the execute path is `UNEXPECTED_OPERAND`.
**It is still not discriminated from `commit_stage`'s PC-capability check**, which is
spec-conformant base 23 and emits 25 for `INVALID_CAPABILITY`; `privM=1` means that check's gate is
satisfiable. Both readings remain live.

**A PLAN PREMISE OF MINE WAS WRONG, and it is the reason this run did not settle it.** I recorded
that the monitor would print `mcause`/`mepc`/`mtval` at the next wedge because the `EXCX` reporting
is committed and compiled into the firmware (verified by disassembling the ELF). **It did not fire:
zero `EXCX`, `MCAU`, `MEPC`, `MTVL` in the whole run.** A capability fault inside a capability domain
**wedges rather than trapping to `mtvec`**, so it never reaches `handle_exception` and the monitor
can never report it. The monitor's reporting covers traps that reach the monitor; this one does not.
Do not plan around it again.

**THE DECISIVE DATUM EXISTS IN HARDWARE AND IS NOT READABLE.** `cva6.sv:1083` latches
`recent_nontrivial_mepc_log_q <= pc_commit` in the same block that latches the mcause this run read
out — so the faulting PC *is* captured. It is never selected by the debug mux (declared at `:980`,
cleared at `:1005`/`:1026`, written at `:1083`, read nowhere). Exposing it is a few mux arms next to
the existing trap-log arm, and it needs a **bitstream reflash**, which is ask-first. Until then the
two readings of 25 cannot be separated and the faulting instruction cannot be named.

**Also settled this run, as a clean negative:** the COMPILER-side fixup
(`-capstone-memcpy-high-half-fixup`) is **neutral for SQLite**. A matched pair in the preceding boot
— 220 fixup sequences versus 1, verified by disassembly, control green — returned the *identical*
`stage=create rc=11 malformed` on both arms. The S-06 damage that reaches SQLite is in the LIBRARY
`memcpy`'s `BEEBS_CHUNK_COPY` (`SQLITE_LDC_HIGH_HALF_FIXUP`), which never consults the compiler's
memcpy hook. Only the library knob produces the wedge, exactly as `build-sqlite-silicon.sh:777-795`
already documented.

**BOTH CODES NOW MEASURED DIRECTLY, in one matched pair.** `excode-base-audit.S` traps each
exception on purpose and asserts the delivered `mcause`:

| arm | how it is provoked | delivered `mcause` |
|---|---|---|
| U | `CINCOFFSET` with a still-tagged capability as `rs2` (`capstone_flu_unit.anvil:30` raises `UNEXPECTED_OPERAND` and nothing else for that condition) | **25** |
| I | `LDC` through a capability whose revocation node has been revoked — `capstone_dyn_unit.anvil:337`, **the exact site the SQLite wedge was blamed on** | **26** |

It PASSES, with a control inside arm I confirming the same `LDC` does *not* fault before the
`REVOKE`, and with each arm asserting `trap_count == 1` so an arm that silently failed to fault
cannot pass on a stale count. The RTL's own `$display` corroborates independently: the log contains
exactly one `Exception: UNEXPECTED_OPERAND` and exactly one `Exception: INVALID_CAPABILITY`.
Negative-tested — changing arm I's expectation from 26 to 25 makes it FAIL at `fail_i_cause`.

**So the retraction is settled empirically, not by reading encoders:** the revocation-validity check
delivers 26, and the board wedge showed 25. It cannot be that site.

**MEASURED, not inferred.** `cincoffset-stale-metadata.S` builds a genuinely tagged `CAP_TYPE_LIN`
capability and feeds it to `CINCOFFSET` as `rs2`. `capstone_flu_unit.anvil:30` raises
`UNEXPECTED_OPERAND` and nothing else for that condition. The test's trap handler records `mcause`
and the self-check asserts **both** `trap_count == 1` (so the trap provably happened, and a vacuous
pass is impossible) **and** `observed_cause == 25`, failing to `selfcheck_fail` otherwise. It
**PASSES** on this RTL. So on silicon `UNEXPECTED_OPERAND` really does deliver `mcause 25`, where
the spec says it should deliver 24.

This also closes the one link in the retraction's evidence chain that was unverified — whether the
Anvil compiler assigns enum ordinals in declaration order from 0. It no longer matters: the
end-to-end behaviour is measured directly.

**Consequence for the SQLite blocker.** The retraction under S-06 stands and is now empirically
backed: an `INVALID_CAPABILITY` from `LDC`/`STC` (`capstone_dyn_unit.anvil:337`, `:404`) leaves the
execute path as **26**, and the wedge showed 25. The two live readings of an observed 25 are
unchanged — `UNEXPECTED_OPERAND` from the execute path, or `INVALID_CAPABILITY` from the
spec-conformant fetch-path check — and `mepc` still discriminates them.

**Do NOT fix by changing the base yet.** The RTL is wrong, but so are `riscv_pkg.sv`, the monitor's
expectations, and at least one directed test (`cincoffset-stale-metadata.S` asserts 25 and would
start failing). They have to move together, and the change alters every `mcause` software receives.
This needs the board owner, not a patch.

<details><summary>Superseded first analysis, kept because it was acted on</summary>

### two exception encoders disagree by one, so a single `mcause` value has two names

Capability exceptions reach `mcause` through two different encoders using two different bases:

| encoder | base | covers |
|---|---|---|
| `ex_stage.sv:469` (FLU), `cva6.sv:1360` (DYN) | **24** | every capability op's exception |
| `commit_stage.sv:205-228` (PC-capability check) | **23** | the fetch capability only |

`riscv_pkg.sv:349-353` agrees with base 24 (`UNEXPECTED_OPERAND_TYPE = 25`, `INVALID_CAPABLITY = 26`,
`UNEXPECTED_CAPABLITY_TYPE = 27`), so `commit_stage.sv` is the one out of step, and its own comments
state the base explicitly (`64'd25; // INVALID_CAPABILITY (23 + 2)`), so this is not a
misreading — it is written that way.

**Consequence:** every capability `mcause` in the overlap is ambiguous. 25 is `UNEXPECTED_OPERAND`
from the execute path and `INVALID_CAPABILITY` from the fetch path; 26 is `INVALID_CAPABILITY` from
one and `UNEXPECTED_CAP_TYPE` from the other; and so on through 28.

**This is not theoretical — it has already cost.** The SQLite blocker's `mcause 25` was named
`INVALID_CAPABILITY` and three investigations were spent on the revocation subsystem before the
arithmetic was checked. See the retraction entry under S-06.

**NOT fixed here, deliberately.** Changing either base changes the `mcause` values software
actually receives, so it needs a decision rather than a patch: the monitor, any handler, and the
directed tests that check `mcause` (e.g. `cincoffset-stale-metadata.S`, which expects 25 for
`UNEXPECTED_OPERAND` and would be correct under base 24) all have to move together. The
off-by-one *comments* in the `ex_code` enum that caused the misnaming ARE fixed
(`capstone_unit.anvilh`, comment-only).

Verified against `capstone-spec`: **NOT YET.** The spec's exception numbering has not been checked
against either encoder, so which base is *correct* — as opposed to which is in the majority — is
still open. Do that before proposing a fix.

*(That check has since been done — see the top of this entry. The answer inverted the expectation:
the majority encoder is the non-conformant one.)*

</details>

### UNRESOLVED — two open questions from the same audit, recorded as questions and NOT as findings

Neither is asserted. Both need a directed test before they are anything.

**U-1: atomics are excluded from the shadow-tag update, a candidate capability-forgery path.**
`wt_axi_adapter.sv:143-145` gates the tag write on `dcache_data.rtype inside {DCACHE_LOAD_REQ,
DCACHE_STORE_REQ}`; `DCACHE_ATOMIC_REQ` (issued at `wt_dcache_missunit.sv:553`) is **absent**, so an
AMO writes DRAM without writing the shadow-tag byte, and self-invalidates the L1 line. `RVA` is
enabled in the active config. That composes to: `stc` a real capability to `X`; `amoswap.d` an
arbitrary value to `X+8`; `ldc` from `X` refills with the tag still set and the attacker's word
promoted to `ruser` (`wt_dcache_mem.sv:304`) -- a capability with chosen `cap_type`, `perm`,
`bounds` and `revnode_id`, which would be materially worse than S-06. **Unestablished:** whether the
AMO write lands at `X+8` through the FPGA interconnect; whether the `ldc` genuinely misses and
refills rather than being satisfied from the write buffer; and whether the result survives
`get_node_query_validity` (`capstone_dyn_unit.anvil:333`). Any one of the three kills it. The test
needs four arms in one image, and the arm that matters most is the positive control -- `stc` then
`ldc` with no interference, which **must** show a tag, or a clean probe arm proves only that the
instrument is dead.

**U-2: `CJALR`'s `rs1`/`rd` handling.** `capstone_flu_unit.anvil:223-232`: the `rs1 == rd` branch
writes `cnull` to both slots; the `rs1 != rd` branch passes `rs1` through **uncleared** and still
puts `cnull` in `rd`. `ctrl-flow-insn.adoc:37-39` requires `x[rd] <- pc` always and `cnull ->
x[rs1]` when `rs1 != rd`. Where (or whether) the pc capability reaches `x[rd]` for `CJALR` was not
traced, so this is not a defect claim. Separate signature -- **must not** be folded into R-21.

**Not a defect, recorded so it is not re-raised.** The two shadow-tag write paths (`wt_dcache_mem.sv`
`:419` store, `:412` refill) were suspected of using divergent predicates, with `wr_cl_user_i[7:0]`
read as the low byte of `bounds`. **REFUTED.** On the refill path those eight bits are a shadow-tag
byte of `0x00`/`0x01`: `wt_axi_adapter.sv:441-442` zeroes the word and writes one byte of
`tag_wr_value_q = is_cap_req = |dcache_data.user` (`:196`, `:402`), and `:731-734` reads exactly
that byte back. The AXI USER sideband carries nothing (`:204`). The two gates are the same
predicate over different encodings. This also withdraws the eviction hypothesis that S-06's
`FIX-PROPOSAL.md` once offered for the SQLite wedge.

**Cosmetic, non-security:** `ariane.core:48`, `Bender.yml:114` and `src_files.yml:50` all name
`core/capstone_dyn_unit.sv`, which does not exist; the file that reaches the build arrives via
`core/Flist.cva6:138` -> `core/anvil.Flist`.
