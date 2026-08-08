# R-18/R-19 fixed in silicon; SQLite still wedges; the clamp axis is dead

Three board results on `caplifive_65536_r18_fix.bit`, all with a VALID control, plus the
root cause of the four VOID boots that preceded them.

## 1. The R-18/R-19 fix works on the arms tested — stated at audited strength

| arm | link addr | on `65536_nodes` | on `r18_fix` |
|---|---|---|---|
| `k800` | `0x10000` | 4 | **4** (control) |
| `c8` | `0xf0000` | 67699255 | **67699264** |
| `c8d` | `0x70000` | 67699255 | **67699264** |
| `fdp0` | `0x30000` | `0x08000A31` | **2609** |
| `fdd` | `0xb0000` | `0x08000A31` | **2609** |

Workaround flag OFF; these are the DAMAGED builds, byte-identical (`sha 9ecd8c6f…`,
`99676ee3…`) to binaries that failed before — which also proves the flag was off without
trusting a build log. Prior failures were bit-exact across **9 sessions** (`c8`) and **3**
(`fdp0`); no damaged arm has ever returned the correct value on the old bitstream.

**NOT yet "R-18 and R-19 are fixed."** Only ONE shipped arm per package ran (`c8`, `fdp0`);
`c8d`/`fdd` are ladder arms in neither package. The corpus splits into **distance-8** and
**distance-4** geometries and `c8` is distance-8 — **no distance-4 arm** (`gz0`, `gzn`,
`graw`), no raw-readback arm (`craw`), no localization arm (`rmC`) has been run. One ~7-rung
boot closes this: `k800, c8, gz0, gzn, craw, rmC, fdp0`.

**NOT "one defect, two faces."** RETRACTED before publication. The fix starves the chain at its
common input (`store_unit.sv` `st_user_n`); everything downstream is untouched, so curing at a
common ancestor silences all consumers whether there is one fault or five. Three of our own
documents already say this is unresolved. A 13 s simulation arm at MIRRORED geometry on pre-fix
RTL would measure it; the board run structurally cannot.

**Fix provenance:** the only functional change in `e1b3db6ba` is TWO assignments in
`core/store_unit.sv`, gating metadata onto the sideband by opcode (`STC`/domain-switch only).
`wt_dcache_mem.sv`'s only edit is a `cover property` — coverage, not synthesizable logic;
`st_wr_cap = |wr_user_i` at `:138` is UNCHANGED. It lands on the chain our packages named, and
their test cites our package by path. **UNVERIFIED:** nothing in this repo links the `.bit` file
to that commit — ask for a one-line confirmation.

## 2. SQLite is NOT blocked by R-18/R-19

`bl64` (full 64-entry array) on the fixed bitstream: **created, ENTERED (`SQ: A/dom-ok` +
`SQ: G/enter`, no monitor tag), then WEDGED.** A genuine domain wedge, not an entry stall. This
closes on repaired hardware a question previously only inferred from the workaround.

## 3. The builtin-registration clamp axis is DEAD

`bl0` — `BUILTIN_LIMIT=0`, the loop body never executes — **also wedged**, entered and all.
Control green in the same boot.

So the wedge is NOT in the `capstoneBuiltinFunc` registration loop. The clamp gate was repaired
and negative-tested before this run, and each image was verified `clamped x1 / unclamped x0`, so
this is a real result and not a clamp that failed to apply.

**Where that leaves the target.** `sqlite3RegisterBuiltinFunctions` makes **15 calls**, eleven to
the same target at a regular `0x44` stride; the clamp only emptied ONE of them. The next
bisection must clamp **by call index** — return a distinct marker after call N — not by array
entry count. `bl1`, `bl8`, `bl32`, `bl64` are built and staged but are now known to be on the
wrong axis; do not spend boots on them.

## 4. Root cause of the four VOID boots: wrong host (operator error)

`run_sqlite_stages_fpga.py` takes `host|selector:dom`; without a `host|` prefix it defaults to
`sqlite_host.user`. `k800` is a LADDER rung needing `lpc`.

    lpc k800 /test-domains/k800.dom          RETURNED 321 / 321
    sqlite_host.user /test-domains/k800.dom  RETURNED   0 / 5

`sqlite_host` does TWO region shares, `ladder_perf_ctl` does ONE, and every share IS a domain
entry — the control entered and RETURNED on share 1, then died on share 2. Trap `0x9a` =
**mcause 26 = INVALID_CAPABILITY**, a capability fault, not an R-16 stall. **Confirmed on the
board**: with `lpc|k800` the same control returns `obs=4, rc=0`.

Exonerated: image size, the 2 MiB padding step, staging a 1.5 MB domain, R-16, and the board.

---

## 5. The call-index bisection: `sqlite3RegisterBuiltinFunctions`' INTERIOR is exonerated

Six variants `rs0..rs5`, each returning early after sub-step N, gate-verified marker+early-return
in the exact file compiled, all six hashes distinct. One boot, ascending, control green
(`k800 obs=4`).

**`rs0` — early return at the FIRST instruction of the function — WEDGED**, created and entered.

### What this does and does not establish

**Does:** the wedge survives making `sqlite3RegisterBuiltinFunctions()` a complete no-op. Nothing
inside that function — not `AlterFunctions`, not the fixup loop, not `WindowFunctions`,
`DateTimeFunctions`, `JsonFunctions`, nor `InsertBuiltinFuncs` — is required to reproduce it.

**Does NOT:** localise anything further. **Design flaw in this experiment, stated so it is not
repeated:** the clamp truncates only that one function; `sqlite3_initialize()` then continues and
the domain runs the whole workload. `rs1..rs5` were therefore never going to bisect anything —
only `rs0` carried information, and only of the exonerating kind. A bisection must clamp the
TOP-LEVEL flow, not a leaf function, or every variant runs the same tail.

### This CONTRADICTS the standing localization

`SILICON-BLOCKER.md:3-22` (2026-08-06) records the wedge as appearing when
`sqlite3RegisterBuiltinFunctions()` is added to a cumulative staged probe, with every earlier
step returning a distinct rc. Tonight the same function skipped entirely still wedges.

Both cannot describe one fault. Either that localization is wrong, or there are **two independent
wedge sites** and the staged probe stopped at the earlier one. Do not treat `RegisterBuiltinFunctions`
as "the" location until this is resolved.

### Correct next axis

Clamp the DOMAIN's top-level flow, not a leaf: `DOMAIN_EXTRA_DEFS=-DCAPSTONE_SQLITE_STAGE=N`,
which returns from `domain_main` after step N. **This mechanism previously produced a VOID
bisection because the staged block was not compiled in** — `build-sqlite-silicon.sh:571-576`
already checks for the `capstone_probe_string` literal in the artifact, and that check must be
confirmed to FIRE before any verdict is read. Stage the set, one boot, ascending.

`rs1..rs5` are built and staged but carry no information by construction — do not spend boots on
them.
