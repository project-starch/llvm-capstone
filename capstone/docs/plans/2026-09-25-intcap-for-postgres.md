# intcap for PostgreSQL: a capability-carrying integer type first, the `uintptr_t` ABI later

*Status: PLAN, 2026-09-25. Branch `plans/intcap-datum`, on top of `compiler/intcap-design`, whose
`intcap-uintptr-model.md` is the analysis this plan builds on. Decision owner for the scope and for
anything on silicon: the project lead.*

## Why this plan exists

The PostgreSQL backend boots in a domain up to the first `Datum`, and stops there
(`postgres/9-boot-attempt`, `f9b7c9ffa746`). The storage side works:
- the data directory, the lock file and shared memory;
- `BootStrapXLOG` writes `pg_control` and the 16 MiB WAL segment, and recovery reads them back.

The first BKI statement, `create pg_proc`, faults at `GetTableAmRoutine+0x38`. The handler returns
`PG_RETURN_POINTER(&heapam_methods)` as a `Datum`, and a `Datum` is `uintptr_t` (`postgres.h:64`),
eight bytes on capstone64. The capability is cut to its address, and `DatumGetPointer` hands back a
pointer that faults on first use.

`Datum` is the fmgr's calling convention for the whole backend. In `src/backend` (counted in
`postgres-single-user-port.md` on that branch) there are:
- 781 `PointerGetDatum` and 374 `DatumGetPointer`;
- 729 bare `(Datum)` casts;
- 4750 `DatumGet*`/`*GetDatum` calls in 482 of 869 files.

This is not a patch series. PostgreSQL needs an integer type that can carry a capability, and that
is CHERI's `__intcap`.

`intcap-uintptr-model.md` costs the full switch, in which `uintptr_t` itself becomes capability
sized for every program, at 11-20 weeks. It lists PostgreSQL among that switch's costs, not its
beneficiaries. **This plan asks what PostgreSQL alone needs, and the answer is much less.**

## Recommendation: stage 1 is the type, not the ABI

**Stage 1.** Add `__intcap` and `__uintcap_t` to clang as builtin types with CHERI C's semantics.
Leave `uintptr_t`, `intptr_t` and musl untouched. PostgreSQL opts in with one line:

```c
typedef __uintcap_t Datum;          /* postgres.h:64, was uintptr_t */
```

**Stage 2**, only if wanted later, is the rest of `intcap-uintptr-model.md`: flip `IntPtrType`
to the new type, split musl's `_Addr`, and port the other programs. Stage 1 is a strict prefix of
it, so nothing is built twice.

Why stage 1 first:

- **It is what PostgreSQL needs, and nothing else needs to change.** No program's `uintptr_t`
  changes meaning, so there is no ABI break. musl, the runtime, CPython, nginx and FFmpeg are
  untouched: WP5 of the model plan disappears, and WP6 shrinks to PostgreSQL.
- **PostgreSQL already assumes it.** In 17.5, `postgres.h:81` defines `SIZEOF_DATUM` as `SIZEOF_VOID_P`,
  which is 16 on capstone64, while `sizeof(Datum)` is 8 today. With a 16-byte capability `Datum`
  the macro and the type agree again. On CheriBSD, which is the same situation, PostgreSQL runs.
  CHERI's own port is `CTSRD-CHERI/postgres`, branch `96-cheri`.
- **The front end is almost the same work either way.** The types, conversions, provenance rules
  and code generation are the bulk of WP2 in the model plan. Only the `IntPtrType` flip and its
  macros are left for stage 2.
- **It proves the code generation on one real program** before every program pays for it.

## What the machine has to do, and what it does today

An intcap holds either a capability or a plain integer, and nothing tells the two apart except the
tag at run time. Three instructions meet that:

| operation | intcap needs | QEMU today | silicon today |
|---|---|---|---|
| `MOVC` copy of an untagged value | keep the source | keeps it | **zeroes the source** (C-32). No single instruction avoids it; Q-04 (b) is a one-line RTL change |
| `SCC` set-cursor on an untagged value | untagged value with the new address (CHERI) | traps; `assert`s without capstone-qemu#8 | traps (cause 24) |
| `CINCOFFSET` on an untagged value | the same | traps | traps |

**For QEMU, stage 1 needs no ISA decision.** The compiler lowers every intcap arithmetic
operation to a tag dispatch: if tagged, `SCC`/`CINCOFFSET`; if not, integer arithmetic,
re-bridged as an untagged value through the existing `PseudoBRIDGE_CAP` path. No untagged
`SCC`/`CINCOFFSET` is ever executed. QEMU already keeps an integer `MOVC` source.

**Silicon needs Q-04 (b), or a copy detour in software** (`plans/2026-09-24-q04-movc-integer-source.md`,
merged as #96, not yet decided or built). Today a capability register holds a capability or NULL,
and NULL zeroed is still NULL, which is why `MOVC`'s rule has cost only C-32 so far. Under intcap a
capability register routinely holds a non-zero integer (`Int32GetDatum(42)`, an OID, a hash), and
every register copy the compiler makes (`copyPhysReg`, calls, phi elimination) would silently zero
the source on the board. `mv` would strip a real pointer's tag, so neither instruction is right for
a value whose taggedness is known only at run time. Two software copies do work, at a price on every
capability-register copy in intcap code:
- test the tag with `LCC` selector 1, the one query that does not trap on an untagged value (it
  answers 7, RTL and QEMU; C-33), and branch between `MOVC` and `mv`. That costs a branch and a
  scratch register, inserted after register allocation;
- copy through memory. `STC` leaves an integer source alone (spec and RTL) and `LDC` reads it back.
  That costs a store and a load, plus a slot.

Q-04 (b) makes `MOVC` leave an untagged source alone, as `STC` already does, so every copy is one
instruction again. Linear capabilities are still consumed, so security is unchanged. M5 therefore
has two routes, the RTL line or the software copy, and the software route's cost is measurable on
QEMU in M4 by switching it on there. When the SCC/CINCOFFSET decision
(`plans/2026-09-24-scc-cincoffset-untagged.md`) lands with CHERI's rule, the tag dispatch becomes a
single `SCC`. That is one switch in the back end, and C-66's `canTrap` drops those opcodes in the
same change. R-33 and R-29 are OPEN silicon defects that intcap exercises harder. They gate the
board, not QEMU.

**C-66 is a prerequisite, and it is done.** The tagged arm of the dispatch is a trapping
instruction under a guard. That is exactly the shape C-66 stops MachineCSE's PRE and MachineLICM
from moving above its test. Without C-66, the compiler would reintroduce the trap the dispatch
exists to avoid.

## Milestones

Every milestone ends in a gate that can fail. The gates are:
- lit on the Capstone directory, and X86/Generic unchanged;
- the nightly core tier compared against dev under identical conditions;
- for PostgreSQL, a one-variable pair: the same tree with `Datum` as `uintptr_t` and as
  `__uintcap_t`.

**M0: measure PostgreSQL under intcap before writing compiler code (1-2 days).** Cross-compile
PostgreSQL 17 for CheriBSD riscv64-purecap with the local CHERI SDK: clang 17 and the sysroot in
`/home/biecho/cheri/output/sdk`, the SDK the CheriBSD arms of the temporal-safety comparison use.
Collect `-Wcheri-provenance` and every error. That gives three things:
- the exact list of sites where intcap semantics are ambiguous or wrong in PostgreSQL 17;
- which of the 729 bare `(Datum)` casts are fine as they stand;
- what `96-cheri` had to change, carried forward from 9.6 to 17.

Also settle there the 24 preprocessor branches on `SIZEOF_DATUM` in 17.5 (13 `== 8`, 8 `>= 8`,
2 `< 8`, 1 `#ifdef`). The `== 8` ones must take the 64-bit path at 16. Settle as well the
capstone64 configure's `MAXIMUM_ALIGNOF` and `USE_FLOAT8_BYVAL`: a 16-byte `Datum` array needs
16-byte alignment. The output is a site list with a count, which replaces the judgement in the
PostgreSQL estimate below.

**M1: types and semantics in clang (2-3 weeks).** Port from CHERI's clang (reference tree
`/home/biecho/cheri/llvm-project`, LLVM 17), purecap-only:
- `__intcap` and `__uintcap_t`: integer rank above every other integer, value range the address
  width;
- conversions: pointer to intcap keeps the capability, integer to intcap is null-derived, intcap
  to integer is the address;
- CHERI's provenance rule for binary operators (the left operand for compound assignment and
  non-commutative operators, otherwise the one that is not provenance-free), and
  `-Wcheri-provenance` on ambiguity;
- comparisons on the address;
- `__SIZEOF_INTCAP__` and `ptraddr_t`.

`IntPtrType` stays as it is. The intrusive part, measured in the model plan, is provenance marking
on casts: CHERI threads an `ASTContext` through every `CastExpr` constructor, and LLVM 22 has
changed those. Gate: CHERI's intcap Sema and CodeGen tests, retargeted to capstone64.

**M2: back end and a QEMU probe (1.5-3 weeks).**
- Lowering: get-address is the integer read of the cursor. Set-address becomes a pseudo that
  expands after code motion into the tag dispatch above. A null-derived value comes from
  `PseudoBRIDGE_CAP` (the C-40 path).
- Load and store: an intcap in memory is a 16-byte `LDC`/`STC`, and `STC` already leaves an
  integer source alone.
- No 16-byte atomics: PostgreSQL's `pg_atomic_*` work on `uint32`/`uint64`, never on `Datum`
  (none found in `src/include`).
- Gate: a QEMU domain probe with a control per property:
  - a pointer through intcap survives a copy, a spill and a call;
  - an integer through intcap survives the same;
  - arithmetic is right on both;
  - comparisons and `switch` work;
  - `CAPSTONE_CINC_UNTAGGED_SURVIVE=1` logs zero untagged `cincoffset`s.

  The control is the same probe with `uintptr_t`, which must fault where intcap does not.

**M2b: the board question, answered on QEMU (about 1 week).** Is an RTL change needed at all? The
analysis below says it is not strictly needed. This milestone measures what doing without it costs.

*Which instructions intcap meets*, per `CapstoneISASemantics.md` and its RTL line references:
- `STC` keeps an integer source (`capstone_dyn_unit.anvil:439-446`).
- `LDC` loads an untagged value, and it clears the granule when it loads a linear one.
- Integer instructions read the cursor, and an integer writeback clears the metadata.
- `LCC` selector 1 is total on an untagged value (`capstone_dyn_unit.anvil:195,208`), so it is the
  tag test for the `SCC`/`CINCOFFSET` dispatch.
- That leaves `MOVC`, which zeroes an untagged source, as the one behaviour with no free answer.

*The software rule.* The back end emits `MOVC` in two places:
- `copyPhysReg` (`CapstoneInstrInfo.cpp:549`);
- frame addressing (`CapstoneRegisterInfo.cpp:313`), whose source is the tagged stack capability.

In `copyPhysReg`, a copy whose source dies stays a `MOVC`. A copy whose source is read again goes
through a reserved 16-byte stack slot: `STC`, then `LDC`. The RTL rows show this matches `MOVC` for
every type. An integer or a non-linear source is kept. A linear source is nulled by `STC`, and its
memory copy is cleared by `LDC`, so a move is still a move. The rule never needs to know whether a
value is an integer. That is why it would also reach the musl `iconv_open` instance of C-32, which
`ISSUES.md` records as out of reach of a compiler fix. That holds only for analyses that must know
in advance whether a value is an integer. The rule is not proven until it is built.

*Its size, measured 2026-09-25* on CPython 3.13.7's core: 129 files, the port's flags, the MIR
just before `postrapseudos`. Liveness was computed from the MIR itself, because post-RA kill flags
are conservative, and the analyser was checked on a two-call control with a known answer. Of
62,757 capability copies, 30,625 read their source again, 25,836 do not, and 6,296 copy from
`$c0`. That makes 30,625 store/load pairs against 901,489 static instructions, 3.4 %. The dynamic
cost is not measured.

*Gate.* QEMU with `CAPSTONE_MOVC_NULL_SCALAR=1` zeroes an untagged `MOVC` source as the RTL does.
1. Under it, the libc-test and nightly exposure run (`movc-null-scalar/exposure.sh`) must show no
   changed verdict with the rule on. `iconv_open` changes today, which makes it the positive control.
2. CoreMark, SQLite and, from M4, PostgreSQL are timed with the rule on and off.

If that cost is acceptable, the board needs no RTL change. If it is not, Q-04 (b) is the cheaper
fix, and now it comes with a number.

**M3: PostgreSQL past the wall (1-2 weeks).**
- Apply `typedef __uintcap_t Datum`, the M0 site list and the `SIZEOF_DATUM` branches on
  `postgres/9-boot-attempt`.
- Gate 1: `postgres --boot` gets past `create pg_proc`. This is the one-variable pair: the same
  tree with `uintptr_t` must still fault at `GetTableAmRoutine`.
- Gate 2: all five of initdb's recorded backend invocations (`record-initdb.sh`) complete, and the
  data directory they leave is accepted by a native `postgres`.

**M4: a real session (1-2 weeks).** Run the single-user workload from the survey in a domain: DDL,
2000 rows, an index, a join, update, delete and vacuum. Its output must be byte-identical to the
native run of the same SQL. Measure what a 16-byte `Datum` costs in memory and time against the
native build. Then the memory-context port (Sublet) under the whole backend: the temporal-safety
result on a real PostgreSQL, not a replay.

**M5: silicon (gated, not scheduled).** After Q-04 (b) and the SCC rule are in the RTL and
synthesised, and R-33/R-29 are resolved or ruled out for this workload: the same session on the
board.

## Estimate

| | | weeks | basis |
|---|---|---|---|
| M0 | measure PostgreSQL under CHERI's compiler | 0.2-0.4 | judgement; the toolchain exists |
| M1 | clang types, semantics, codegen | 2-3 | the model plan's WP2 (3-5 wk, ~1,200 lines purecap), minus `IntPtrType` and its macros |
| M2 | back end, tag dispatch, QEMU probe, tests | 1.5-3 | WP3 without atomics; WP4's retargeted tests |
| M3 | PostgreSQL through bootstrap and initdb | 1-2 | the survey's "two to four weeks on top of intcap", split; replaced by M0's count |
| M4 | session, output identity, cost, Sublet | 1-2 | judgement |
| **total to PostgreSQL on QEMU** | | **6-10** | against 11-20 for the ABI-wide switch |

The ranges are judgement, anchored where the model plan measured. Confidence is highest on M1. M3
is the least certain until M0 has run.

## Where this could break

- **Ambiguous provenance in PostgreSQL's own arithmetic on `Datum`.** Sort abbreviation, hashing
  and bit tricks do arithmetic on `Datum`s. Under intcap each one takes one operand's capability.
  `-Wcheri-provenance` finds them, and M0 counts them before anything is committed.
- **LLVM 17 to 22 drift in the files CHERI touched** (Sema, `CGExprScalar`, the `CastExpr`
  constructors). This is mechanical in principle and conflict-heavy in practice. It is the model
  plan's first risk as well.
- **Tag-dispatch cost.** A branch on every intcap arithmetic operation. PostgreSQL does little
  arithmetic on `Datum`, so the cost should be small, but M4 measures it rather than assuming it.
  The dispatch disappears once the SCC rule is in the ISA.
- **Linear capabilities in an intcap.** An intcap is copied freely, so a linear capability in one
  would be duplicated or destroyed. PostgreSQL should put only palloc'd and static pointers in a `Datum`.
  Whether the Sublet memory-context arm hands out non-linear aliases is to be checked in M3,
  not assumed. The exposure is the one a C pointer
  copy already has (C-46).
- **On-disk format: none expected.** Tuples store attribute bytes, not `Datum`s. By-value types
  are at most 8 bytes, so a wider `Datum` changes no page. M3's native check of the data directory
  is the test.

## Decisions needed

1. **Scope:** stage 1 (the type, PostgreSQL opts in) rather than the ABI-wide switch. This is the
   recommendation.
2. **QEMU first with tag dispatch**, without waiting for the SCC rule. Silicon stays gated on it.
3. **For the board: the software copy rule (M2b) or Q-04 (b) in the RTL.** M2b's measurement
   decides it, and it runs on QEMU. The SCC rule is not needed: the tag dispatch covers it.

M0 needs none of these and can start at once.
