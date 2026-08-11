# S-06 — an untagged 128-bit `ldc`/`stc` round trip loses the HIGH 64 bits

**Wrong symptom? Read this paragraph first.** This package is the **plain-data-loses-its-high-half**
signature: a 16-byte `ldc`/`stc` pair over memory that does **not** hold a capability keeps only
the low 8 bytes of every chunk. Sibling packages describe different signatures and are not this
issue: `../R18-scalar-store-metadata-clobber/` is a scalar in the upper half of a 16-byte row
being zeroed by a **scalar** store; `../R19-movc-zero-metadata-in-slot/` is a slot coming back
holding `compress_cap(NULL) + n`; `../R20-stc-rs1-cursor-forward-x10/` is a stale **register**
operand after a store, corrupting no memory at all; `../R01-lsu-hazard/` is a load through one
capability register missing a store through another. S-06 corrupts **memory**, silently, on
every capability-grained copy of ordinary data.

## The defect in one paragraph

`ldc` followed by `stc` is how a capability-aware `memcpy` moves 16 bytes: it is the only copy
that preserves a capability **tag**, so it is what any pointer-bearing struct is copied with. On
this silicon that pair is **not bit-exact for plain data** — each 16-byte chunk arrives with its
low 8 bytes correct and its high 8 bytes gone. There is no trap, no tag violation and nothing in
any log, and the same byte-identical binary is correct under QEMU. Half of every buffer copied
this way is silently destroyed.

## Reproduce it in RTL SIMULATION — 499 cycles, no board

```
./run.sh sim
```

`sim/untagged-ldc-stc-128.S` writes a known 16-byte pattern with two plain stores, round-trips it
through `ldc`/`stc`, and reads both halves back. The RVFI trace prints the value every
instruction retires, so the halves are read **directly** rather than inferred through a whole
`memcpy`. Crucially it carries a **control in the same run**: the same two constants, the same
buffer, the same capability, written and read with `sd`/`ld` only.

**Genuine completion in 499 cycles, no exceptions:**

| register | measures | value | |
|---|---|---|---|
| `t3` | `ldc`/`stc` round trip, LOW | `0x0123456789abcdef` | preserved |
| `t4` | `ldc`/`stc` round trip, **HIGH** | **`0x0000000000000000`** | **LOST** |
| `t5` | control, plain `sd`/`ld` LOW | `0x0123456789abcdef` | preserved |
| `t6` | control, plain `sd`/`ld` HIGH | `0xfedcba9876543210` | preserved |

The control is what makes this attributable. It rules out the buffer, the bounds and the
capability in one step: those same 16 bytes survive exactly when they are not routed through
`ldc`/`stc`. `run.sh sim` fails loudly if the control is wrong, because a run whose control fails
carries no verdict.

The frozen trace from the run above is `sim/rvfi-trace-128.log`.

## Reproduce it on the BOARD — a 10 KB domain, one number

```
./run.sh rung
```

`src/s06copy.dom` copies 32 bytes of ordinary data with exactly `memcpy`'s aligned middle loop
and returns **how many of the 32 bytes came back correct**:

| retval | meaning |
|---|---|
| **32** | every byte survived — no defect |
| **16** | exactly half survived — S-06 |

Measured on `caplifive_r20.bit`, three runs in one boot, with the known-good control `k800`
returning its oracle 4 in the same boot:

```
ladder-perf: RESULT k800    retval=4    cycles=4773
ladder-perf: RESULT s06copy retval=16   cycles=5803
ladder-perf: RESULT s06copy retval=16   cycles=5804
ladder-perf: RESULT s06copy retval=16   cycles=5809
```

The same domain returns **32** under QEMU. The two possible answers are far apart and neither is
a value the rung can return by accident, so this cannot be misread the way a 0/1 flag can.

The bytes themselves, captured from a larger probe on the same silicon, show the shape:

```
src32 = c0c1c2c3c4c5c6c7 c8c9cacbcccdcecf d0d1d2d3d4d5d6d7 d8d9dadbdcdddedf
dst32 = c0c1c2c3c4c5c6c7 0000000000000000 d0d1d2d3d4d5d6d7 0000000000000000
```

## It is NOT only memcpy — the compiler emits the same pattern

`./run.sh rung` also builds `src/s06agg.dom`, a second 10 KB rung that contains **no memcpy at
all**. It performs one ordinary struct assignment:

```c
struct { void *p; unsigned long x; unsigned long y; };   /* p is 16 bytes, so x,y are at 16..31 */
*d = *s;
```

which the compiler lowers to two capability-grained copies — `ldc/stc 0x0` for the pointer (a real
capability, therefore safe) and `ldc/stc 0x10` for `x` **and** `y` together, sixteen bytes of
ordinary data. Return value is the verdict:

| retval | meaning |
|---|---|
| **64** | both fields survived — no defect |
| **66** | `y` gone, `x` intact — S-06 via the compiler's aggregate copy |

Measured **66 twice** on `caplifive_r20.bit`, control `k800` = 4 in the same boot; QEMU returns
64. `66` rather than merely "wrong" is the signature: the defect keeps the LOW half of each
16-byte chunk, and `x` is the low half.

This matters for scoping a fix: a library-level workaround in `memcpy` cannot reach ordinary
struct assignment, so the exposure is not confined to code that calls `memcpy`.

## A SECOND signature: when the source's high half is ZERO, the copy writes NOTHING

Everything above describes the destination's high half coming back **zero**. There is a second,
distinct outcome that is easy to miss and is arguably worse, because the wrong value looks valid.

Both cache width signals are gated on the metadata **content** (`|user`), not on the opcode. So a
16-byte chunk whose high 8 bytes happen to be **exactly zero** produces `st_wr_cap = 0` for a
reason that has nothing to do with tags — and the `stc` degrades to a single-bank store that
**never writes the destination's high half at all**. The destination keeps whatever it already
held.

Measured in RTL simulation, 585 cycles, no exceptions, with a plain `sd`/`ld` control passing in
the same run (`capstone-ariane verif/tests/custom/capstone/s06-mechanism-probe.S`, arm C). The
destination's high half is poisoned with a recognisable value first, so "stale" cannot be confused
with "correctly zero":

| | source | destination before | destination after |
|---|---|---|---|
| low 8 bytes | `0123456789abcdef` | `deadbeefcafef00d` | `0123456789abcdef` — written |
| **high 8 bytes** | **`0000000000000000`** | `deadbeefcafef00d` | **`deadbeefcafef00d` — NOT written** |

Why this matters more than the zero case: losing data to zeros is at least detectable. Silently
retaining the **previous occupant's** bytes is not, and it is reachable from an ordinary struct
copy where one field happens to be zero — one of the commonest shapes in real code. A buffer reused
across two records can carry the first record's data into the second.

Any fix must therefore make the write width depend on the **opcode**, not on the data. A fix that
only stops the high half being zeroed on load will still leave this case broken.

## Where it is in the RTL

The instruction semantics are not where it lives: `LDC`/`STC` in
`core/anvil_build/capstone_dyn_unit.anvil` operate on an already-decoded `fat_cap_t` and contain
no bit-level logic. It is the D-cache, and **both sides contribute, in sequence**:

* **The load discards the bytes.** `core/cache_subsystem/wt_dcache_mem.sv:308` —
  `ruser = cap_tag_hit ? ruser_cl[rd_hit_idx] : '0;`. Bank 1's SRAM still physically holds the
  real bytes; they are MUXed to a literal `'0` whenever the line's 1-bit shadow capability tag
  (`cap_tag_q`, `:135`) is clear. Any plain store to either half clears that tag for the line
  (`:419`, `cap_tag_q[wr_idx_i][j] <= st_wr_cap`), so a buffer filled by ordinary stores always
  reads back with a zeroed high half.
* **The store then never writes the high half at all.** `:138` — `st_wr_cap = |wr_user_i`, i.e.
  gated on metadata **content**, not on the opcode. With the metadata now zero, `:228`
  (`if(!(st_wr_cap))`) requests only the bank matching the store's own offset, so `dst+8..15` is
  left at its prior content — zero for a fresh buffer, **stale** otherwise — and is never written
  by that `stc`.

So the load's force-to-zero *creates* the all-zero metadata that then *causes* the store's
bank-skip. This is not a case of "the store is fine and the load reconstructs", or the reverse.

## Why QEMU never showed it

QEMU carries an explicit `scalar_hi` shadow field for exactly this case —
`capstone-qemu target/riscv/cap.h:79-94` and `op_helper.c:1148-1188` — added so that untagged
`ldc`/`stc` is bit-exact over the full 128-bit word. **There is no RTL counterpart.** Every
result this project has taken under QEMU has been blind to the divergence.

## What it costs at the software level

This is the root cause of SQLite failing to build a schema on silicon. SQLite's schema text is
copied through that loop, so half of every 16-byte chunk is destroyed and the schema will not
re-parse. The error message was the tell and it was in plain sight: SQLite reports
`malforme` — the first **8** bytes of "malformed database schema (items)", with byte 8 gone so
the string ends there. It stays 8 bytes even when emitted into an empty output buffer, so it was
never output truncation. It also explains why a short `CREATE TABLE t(a INTEGER, b TEXT)`
**succeeds** on this silicon while a longer one fails: the damage is length-dependent.

## Why software cannot work around it

Two independent walls, both measured rather than assumed:

* **The aligned path cannot simply be dropped.** It is the only copy that preserves tags; a
  byte-wise `memcpy` strips them, and the domain then dereferences untagged pointers and wedges.
* **Code cannot ask whether a chunk is a capability.** `LCC` with a `NOT_CAP` operand raises
  `UNEXPECTED_OPERAND` *before* it examines the requested field (`capstone_dyn_unit.anvil`,
  `func LCC`, the `cap_type==NOT_CAP` branch), so `__builtin_capstone_cap_get_tag` faults on
  exactly the plain data it would be used to detect — and a capability fault inside a domain
  wedges rather than traps.

A branchless workaround **is** possible in principle and is included here as
`sim/untagged-ldc-stc-fixup.S` arm E: plain-store both halves first, then lay the `ldc`/`stc` on
top. It exploits the mechanism above — a capability has non-zero metadata so the `stc` writes
both banks and restores the tag, while plain data degrades to a single-bank store that leaves the
plain-stored high half intact. It is validated in simulation on both kinds of chunk and on
silicon at the primitive level (a 32-byte block copy returns all 32 bytes correct). **It
nevertheless wedges the full SQLite workload**, isolated to that one change against two
neighbouring builds, and why it does is not established. It is therefore not enabled anywhere,
and it is included as evidence about the defect rather than as a remedy.

One shape is **refuted** and should not be retried: copy, then compare the two high halves and
repair on a difference. For a genuine capability the destination's stored metadata word need not
be bit-identical to the source's, so the comparison can say "differ", run the repair store, and
clear a live tag.

## What a fix needs to do

**A concrete proposal is in [`FIX-PROPOSAL.md`](FIX-PROPOSAL.md)**, with two options: adding a tag
bit to the register representation, which is the general fix and what QEMU already does; and a
narrow 16-byte copy instruction restricted to **untagged** lines, which needs no change to how a
capability is represented and is the smaller of the two. An earlier draft of that second option
copied lines *with* their tags; it was withdrawn during a security review because it would have
duplicated linear capabilities, and the reasoning is kept in the document so the same idea is not
re-proposed. `FIX-PROPOSAL.md` also records, with board evidence, why no software workaround is
safe -- including the one that is correct in simulation and on an isolated rung and still
destabilises a real workload.



Preserve the raw upper 64 bits of a `tag == 0` line across an `ldc`/`stc` round trip — the
behaviour QEMU already implements. `./run.sh sim` is the acceptance test: it passes when `t4`
reads `0xfedcba9876543210` instead of zero, and it independently checks its own control first.
`./run.sh rung` is the same question on hardware, and answers 32 instead of 16.

## Contents

| path | what it is |
|---|---|
| `FIX-PROPOSAL.md` | proposed RTL fixes, and why software cannot work around this |
| `run.sh` | `sim` (~14 s, no board) · `rung` (board) · `verify` (checksums) |
| `sim/untagged-ldc-stc-128.S` | the directed test: round trip + plain `sd`/`ld` control |
| `sim/untagged-ldc-stc-fixup.S` | the workaround experiments, incl. the refuted compare-and-repair shape |
| `sim/rvfi-trace-128.log` | frozen RVFI trace of the 499-cycle run above |
| `sim/rvfi-trace-fixup.log` | frozen RVFI trace of the workaround run |
| `src/s06copy.dom` | the 10 KB board reproducer, frozen and checksummed |
| `src/s06agg.dom` | second reproducer: an ordinary struct assignment, **no memcpy involved** |
| `src/s06agg_kernel.h` | its source, and why 66 is the predicted value rather than just "wrong" |
| `src/s06copy_kernel.h` | its source — the copy under test and why each line is as it is |
| `src/s06copy_{app,fpga_app,host}.c` | QEMU-side app, board-side app, native oracle (32) |
| `SHA256SUMS` | `./run.sh verify` |
