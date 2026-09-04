# The SQLite silicon wedge is ONE bytecode instruction: OP_Column

Date: 2026-08-13. Bitstream `caplifive_12august.bit`. Branch `capstone-bootstrap`.
Configuration: `SQLITE_LDC_HIGH_HALF_FIXUP=1` (schema text repaired, which is what lets the run
reach the fault at all).

## The result

Matched pair, one boot, control returning, the two arms differing by exactly one executed opcode:

| arm | what it does | outcome |
|---|---|---|
| `L2` | control | RETURNED |
| `C6` | VDBE clamped before opcode 6; reports the opcode it declined to run | **RETURNED, opcode 96** |
| `C7` | VDBE clamped before opcode 7 | **WEDGED** |

Opcode 96 is `OP_Column`. The statement is `SELECT name,value FROM items`, so its program carries
two consecutive `OP_Column` instructions at positions 6 and 7: `C6` runs neither, `C7` runs the
first and stops before the second. **The difference between the two arms is the execution of a
single `OP_Column`, on column 0 — the TEXT column.**

`C8`, from the previous boot, also wedged, so 6-returns / 7-wedges / 8-wedges is monotone rather
than a single point.

## How it was reached

Every step is a ladder whose arms RETURN. That is not stylistic: a wedge takes the core, the host
never returns, and the shared payload is never written out — so on this path an instrument that
only observes reports nothing at all. Two boots were spent proving that the hard way (a lookaside
fallback arm and a connection-substitution arm both wedged in silence).

```
CREATE TABLE   prepare / step / finalize    all return
INSERT         exec                         returns rc=0
SELECT         prepare                      returns rc=0
SELECT         first step                   WEDGES
   ... step and return, touching nothing    WEDGES     (S61)
   ... 6 opcodes of its bytecode            returns    (C6)
   ... 7 opcodes                            WEDGES     (C7)
```

Four board sessions before this, the state of knowledge was "SQLite fails somewhere on silicon".

## What is excluded, by construction rather than by argument

* Everything in the row loop after the step — `sqlite3_column_text`, `sqlite3_column_int`, and the
  `sqlite3_stricmp` call site — because `S61` returns before reaching any of them and still wedges.
  This closes the long-standing "where does SQLite's `char*` lose its tag before `sqlite3_strnicmp`"
  thread **for this path**: the wedge happens before any string comparison in that loop. (The
  function may still be called from inside `sqlite3_step`; what is exonerated is the call site.)
* Everything in the SELECT's bytecode before `OP_Column`, because `C6` executes all of it and
  returns.
* The compiler's aggregate-copy S-06 guard, which made no difference to any arm.
* A wrong `db` reaching the allocator: DBWHO's comparison is now on the returning path and has
  never reported a mismatch on this path.

## Why `OP_Column` is a plausible culprit — hypothesis, not measurement

`OP_Column` decodes a serialised btree record and materialises a value out of it. It is the first
thing in this workload to walk a packed row and produce a pointer into it. The measurement is the
matched pair; this paragraph is a lead for the next step, not a finding.

## The image-perturbation caveat — RAISED, THEN CLOSED

`C6` and `C7` are different binaries differing in one compiled-in constant, and this project has a
documented image-perturbation family (S-01) in which an unrelated image change moves a failure. So
the pair was REDRAWN: both arms rebuilt with `CAPSTONE_REDRAW_PAD=7717`, which changes the image
and nothing the test executes. Four images, all `sha256sum`-distinct:

```
C6 7ca45416b50be6   C7 ee08b4865a6b1c      (first draw)
R6 6ecb519eec052d   R7 63d82cee4d5109      (second draw)
```

Second boot, control green again:

```
L2  control    RETURNED
R6  clamp 6    RETURNED, opcode 96
R7  clamp 7    WEDGED
```

Identical to the first draw. Two independent draws, two green controls, same verdict — the result
is a property of executing `OP_Column`, not of any particular image.

## Instruments added, with their positive controls

* `CAPSTONE_CREATE_LADDER=<n>` — splits the workload into arms that return a `0x5A6E_ssrr` marker.
  Stages 1-3 inside CREATE, 4-7 across INSERT/SELECT, 61-66 inside the row loop. The sub-stages
  use exact-match tests sitting numerically above the `<=` ladder so they reach the loop unmodified.
* `CAPSTONE_VDBE_CLAMP=<n>` — stops `sqlite3VdbeExec` after n opcodes, returns `SQLITE_DONE`
  cleanly, and records the opcode that was **about to** run. **Armed per statement**: the counter
  is cumulative across every `sqlite3VdbeExec`, and the first unarmed version clamped `CREATE
  TABLE` itself at opcode 4, which would have measured a different statement entirely.
  Positive controls: `CLAMP=1000` returns `0x5A6E4200` (never fired); `CLAMP=12` returns
  `0x5A6E4256`, firing and naming `OP_ResultRow`. A clamp that has never fired is an unproven
  clamp, and `lastop=0` is exactly what a clamp that *cannot* fire would print.
* The marker had to be plumbed: the domain entry stub did `(void)run_sqlite(); *res = DONE;`,
  discarding it one frame above the ladder that built it. Only `0x5A6E`-tagged values pass through
  now, so `fail()` rc values and the success path are unchanged.

## Next step

Determine what inside `OP_Column` faults. It is one opcode, so the same clamp technique applies at finer
grain — or ask the RTL what access `OP_Column` makes that nothing before it does. `OP_Column`
walks a serialised record header, and the natural next question is whether the fault is in the
header walk, the payload fetch, or the `Mem` it materialises. The REDRAW that was the last
procedural step before treating this as the localisation is done (see the section above), so
nothing is blocking that question.
