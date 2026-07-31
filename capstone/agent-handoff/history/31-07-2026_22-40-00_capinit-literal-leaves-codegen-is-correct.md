# cap-init literal leaves: the codegen is correct (static proof), and two mechanisms are refuted

Date: 2026-07-31 (late)
Context: localising the stage-51 LIVELOCK (`rc=0xB1`, the domain RETURNED, so the core was
never hung). Stage 52 = `0xC1`: `lit[1]` is the first literal whose `strlen` walk overruns.
Stage 53 = `0xDF`: `lit[0]`'s first 8 bytes are `l t r i m \0 r t` — correct for a MERGED
container, so `lit[0]` is fine and its data is intact.

The open question was whether `lit[1]`'s POINTER or its DATA is wrong. The board went
unreachable before the probes could run, so this was answered **statically instead**, from
the linked domain ELF and the RTL. That is a stronger answer than the board would have
given for the codegen half, because it is a proof rather than a measurement.

## What was examined

`capstone/caplifive-system/sw/buildroot/build/target/test-domains/wd55.dom`, function
`__capstone_cap_init` (2718 instructions), disassembled with `llvm-objdump`.

## Result 1 — the merged container and the expected deltas

`"ltrim\0rtrim\0trim\0..."` lives in `.rodata` at vaddr `0x16e52e`. Laying the 16 literals
out end-to-end gives `&"ltrim"=0x16e52e`, `&"rtrim"=0x16e534`, `&"trim"=0x16e53a`, so the
correct deltas are **`lit[1]-lit[0] = 6`** and **`lit[2]-lit[1] = 6`**.

## Result 2 — the emitted code computes exactly those deltas

The 16 literal capabilities are derived at `0x14d6d8`+:

    14d6d8  cincoffsetimm s1, a0, 0x6da     <- lit[0]
    14d6e0  cincoffsetimm s0, a0, 0x6e0     <- lit[1]
    14d6e8  cincoffsetimm t6, a0, 0x6e6     <- lit[2]

`0x6e0-0x6da = 6` and `0x6e6-0x6e0 = 6`, all three from the same base `a0`. **The offsets
are right.**

## Result 3 — the same 16 registers are stored into THREE arrays, and stay live

There are three `lit` arrays in the TU (stage 51's, stage 52/53's `run_sqlite_staged.lit`,
stage 54-56's `run_sqlite_staged.lit.863`; both 256 B = 16 x 16-byte capabilities). They are
written at `0x14d6dc` (interleaved with the derivations), `0x14eef8` and `0x14ef40` — all
three storing the SAME 16 registers.

Between the derivation and the last store there are **1544 instructions, zero calls, zero
branches**. Only `a0` is redefined (539 times), and the compiler handles it correctly:
`lit[15]` is spilled to `0x260(sp)` at `0x14d758` and reloaded at `0x14e800`. The other 15
capabilities are never touched.

**So all three arrays receive correct pointers. The codegen is not the bug.**

## Result 4 — two plausible mechanisms REFUTED against the RTL

Both were checked in `capstone-ariane/core/anvil_build/`, not assumed:

* **`cincoffset` does NOT consume its source.** `capstone_flu_unit.anvil:43` and `:62` both
  return `create_result_pack(..., rs1, rd)` with `rs1 = data.cap_rs1` unchanged. So deriving
  12 pointers from one live `a0` is safe, and the "C-14 shape but for cincoffset" theory —
  that `lit[0]` survives and everything after it comes from `cnull` — is **wrong**, even
  though it fits the observed `lit[0]`-good/`lit[1]`-bad signature perfectly.
* **`STC` does NOT clear its source register** for LINEAR/NONLIN. `capstone_dyn_unit.anvil:427`
  returns `rs2_v` unchanged. Only the UNINIT path nulls it (`rcnull`, ~`:410`). So storing
  the 16 capabilities three times does not destroy them. (The documented "linear clearing"
  is on **LDC**, clearing the *memory* source — it is not symmetric on STC.)

Recording the refutations explicitly because both were good-looking theories that matched
the symptom, and both would have been asserted as the root cause without the RTL check.

## What this means for the pending probes

`wd54/55/56` are built and staged. Their expected values are now **proved, not assumed**:
`55 -> 6`, `56 -> 6`, `54 -> 0xDF`. That makes them a clean discriminator:

* deltas come back **6** => pointers correct on silicon too; the fault is in the WALK
  (the `strlen` loop / its `lcc` epilogue), not in cap-init.
* a delta comes back **wrong** => silicon EXECUTION diverges from provably-correct codegen,
  which is a hardware finding, not a compiler one.

Either outcome is informative, and neither can any longer be blamed on the emitted offsets.

## Incidental: an asymmetry worth reporting (NOT this bug)

`CINCOFFSET`'s operand type-check is commented out behind a `FIXME`
(`capstone_flu_unit.anvil:31-33`), while `CINCOFFSETIMM` keeps it (`:49-51`). So a
register-form `cincoffset` with a `NOT_CAP` rs1 proceeds silently where the immediate form
raises `UNEXPECTED_OPERAND`. `lit[3]`/`lit[4]` do use the register form. Not implicated
here, since `lit[1]` uses the immediate form and its base is shared with the working
`lit[0]`.

## Board status at time of writing

Unreachable. DNS resolves and TCP :443 connects instantly, but the **TLS handshake times
out** (15 s) — the console tunnel is up with its backend not answering. Three consecutive
runner attempts failed at `connect()`. Nothing was flashed; the firmware built and passed
its freshness check (initramfs 10,495,488 bytes, verified by decompressed content).

## Addendum — a third hypothesis measured and killed

The staged probes each add their own 16-element `lit` array plus 16 string literals, and the
known "every array lands in every build" trap means all of them are present in all of them.
That made carve-budget exhaustion (the ~1000-entry rev-node pool, R-12) an obvious suspect
for the SHA5 wedge.

**Measured, not assumed:** reading `count` from the `.capstone_gp_initdesc` header of each
staged domain gives **183 carves** for `wd54/55/56`, **184** for `wd57/58/59`, and **179** for
`sqlite_silicon.dom`. All are an order of magnitude below the budget. The 1059-carve figure
that motivated the original trim belongs to the FULL SQLite build, not to these staged
probes, which return long before most of SQLite is referenced.

So carve exhaustion is not the SHA5 wedge, and the pool budget is not currently a constraint
on the staged bisection at all.

## Addendum 2 — two process errors in this session, both self-caught

1. **Read the accumulated console buffer as results.** `board-<tag>.log` carries the whole
   console scrollback, so grepping it for `SQ: obs=` returned markers for stages 30..53 from
   EARLIER runs and none from the run just performed. Only the run-scoped file
   (`PROBE_SCOPED_OUT`) is valid. This is already documented and was done anyway; the tell
   was markers for stages that were not in `SQLITE_STAGE_DOMS`.
2. **Pruned the controls out of the image.** Trimming the initramfs removed `wd51/52/53`,
   so the load that followed had no known-good domain in it. When its only domain wedged at
   SHA5 there was no way to tell "this probe wedges" from "everything wedges now". The batch
   rule already says to include controls; a prune step silently violated it. Prune and
   ordering must be decided together.

## Board results 2026-07-31 late (BOARD_RC=0, run-scoped) — TWO MORE REFUTATIONS

    wd51  stage 51  rc=0xB1   watchdog CONTROL, UNGUARDED build   want 0xB1  OK
    wd53  stage 53  rc=0xDF   lit[0] bitmap CONTROL               want 0xDF  OK
    wd57  stage 57  rc=7      lit[1] read twice via volatile      want 7     OK
    wd58  stage 58  rc=7      lit[0] read twice, control          want 7     OK
    wd59  stage 59  rc=5      walk lit[1] after one read          want 5     OK
    wd54  stage 54  WEDGED    lit[1] bitmap, plain pointer

**"The SHA5 wedge is self-inflicted" — REFUTED.** The UNGUARDED `wd51`, carrying all three
foreign `lit` arrays contributed by stages 52-59, returned `0xB1` — identical to its result
before those stages existed. `wd53` likewise returned `0xDF`. So the injected arrays do not
cause the wedge; `wd55`/`wd54` wedge for their own reasons, not universally. The `#if` guards
are hygiene (they remove the "every array in every build" trap), **not** the fix. The claim
had been stated as "very likely self-inflicted" on correlation alone while counter-evidence
was already available: `wd51` carried one foreign array when it first returned `0xB1`, so the
story required "one harmless, three fatal" with nothing supporting it.

**"LDC consumes its memory source" — REFUTED.** Stage 57 reads `lit[1]` twice through a
`volatile` array pointer and returns **7**: both reads non-NULL AND equal. Stage 58 does the
same for `lit[0]` and also returns 7. The documented linear-clearing does not fire here.

**And `lit[1]` is fine in isolation.** Stage 59 reads `lit[1]` once and walks it: **rc=5**,
the correct index of the NUL in `"rtrim"`. Pointer, data and walk are all correct for that
element — which flatly contradicts stage 52's `0xC1` ("`lit[1]` never terminates").

## THE CONFOUND that invalidates the 52-vs-59 comparison

Every staged block declared its OWN local `static const char *const lit[16]`. In an unguarded
build all of them exist, so:

* stage 52 reads the **second** cap-init'd array,
* stage 54 reads the **third**,
* stage 59 reads the **fourth**.

Different objects, different addresses, initialised by different blocks of
`__capstone_cap_init`. So "52 fails, 59 works" does **not** isolate the access pattern; it is
equally consistent with *the Nth cap-init'd array is broken and the Mth is fine*. Those two
explanations need completely different fixes, and nothing measured so far separates them.

Note this also re-frames the static proof above: it showed all three arrays receive correct
pointers **in the emitted code**. It did not show they receive correct pointers at RUNTIME.

## Next experiment — stages 60/61/62, one array, three shapes

A single **file-scope** `capstone_probe_lit[16]`, read three ways:

* **60** — stage-52 shape: loop `i=0..15`, walk each. Expect **16**; `0xC0|i` on overrun.
* **61** — stage-54 shape: plain `z = lit[1]`, bitmap bytes 0..7. Expect **0xDF**.
* **62** — stage-59 shape: volatile read of `lit[1]`, bounded walk. Expect **5**.

Discriminator: if 60 overruns while 62 returns 5, the ACCESS PATTERN is the mechanism and the
array is exonerated. If all three agree, the 52-vs-59 split was about WHICH array, and the
fault is in cap-init's later blocks at runtime, not in any walk.

## THE CONFOUND-FREE RESULT — the ACCESS PATTERN is the mechanism

One file-scope array (`capstone_probe_lit[16]`), three access shapes, same addresses, same
layout, same build flags. Board, `BOARD_RC=0`, run-scoped:

    wd51  stage 51  rc=0xB1   control                        want 0xB1  OK
    wd60  stage 60  rc=0xC1   LOOP  for i=0..15, walk each    want 16    FAILS
    wd61  stage 61  rc=0xDF   PLAIN z = lit[1], bitmap        want 0xDF  OK
    wd62  stage 62  rc=5      VOLATILE read of lit[1], walk   want 5     OK

**Array identity is exonerated and the access pattern is the mechanism.** All three read the
identical object, so "the Nth cap-init'd array is broken" is dead, and so is every
layout-based explanation for THIS failure.

### That includes the granule hypothesis — retracted as the root cause

Immediately before this ran, the granule base misalignment (idx 170, `sqlite_heap`, 256 KB,
granule 512, `base%g = 64`) was the leading candidate, with a mechanism that looked
compelling: outward bounds rounding makes the heap capability overlap neighbouring carves, so
memsys5 zeroing its arena clobbers adjacent globals — which would explain literal bytes being
present while a walk never terminates.

`wd60/61/62` refute it for this failure: the three probes share one array at one address under
one glue build, and only the loop fails. A layout defect cannot be selective by access shape.

The granule finding still stands on its own terms — the simulation showing OFF -> 1
unrepresentable carve and ON -> 0 is correct, and idx 170's base really is misaligned — but it
is a **latent defect, not this bug**. Keep the flag off by default until it is tested for its
own sake.

### What is left

`wd60`'s loop walks `lit[0]` BEFORE `lit[1]`; `wd61`/`wd62` touch `lit[1]` alone. So the
remaining candidates are (a) walking one element corrupts the next, or (b) the variable-index
`lit[i]` form differs from the constant-index form. Stages 64/65 separate exactly this:

    64: walk lit[0] then lit[1], same array   expect 0x45, 0xB3 if the first walk breaks it
    65: walk lit[1] alone, same array         expect 0x45  (control)

plus stage 63 (four identical arrays, bitmap of which overran) and `ga60` (the loop shape
against a granule-aligned glue, to check whether granule-align rescues it anyway).

## THE PATTERN: the FIRST data-dependent walk succeeds, every later one fails

Board, `BOARD_RC=0`, run-scoped:

    wd51   stage 51  rc=0xB1  control                              want 0xB1  OK
    ga60   stage 60  rc=0xC1  LOOP shape + GRANULE-ALIGNED glue     want 16    FAILS
    wd63   stage 63  rc=0x0E  bitmap over FOUR identical arrays      want 0     FAILS
    wd65   stage 65  WEDGED   plain pointer + while-walk, shared array

**`ga60` closes the granule question on silicon.** The loop fails identically (`0xC1`) with
the granule-aligned glue, so the retraction made earlier from `wd60/61/62` is now confirmed by
measurement rather than inference. idx 170's misaligned base stays on the books as a latent
defect; it is not this bug.

**`wd63 = 0x0E` is the key result.** Four file-scope arrays with IDENTICAL contents, walked in
one loop: bit0 clear, bits 1/2/3 set — array 0's walk terminates, arrays 1, 2 and 3 all
overrun. Identical objects, so nothing distinguishes them except ORDER.

Every observation now fits one statement, with array identity, index value and pointer
provenance all dropping out:

| probe | walks performed | result |
|---|---|---|
| 61 | one (plain, indexed reads) | correct `0xDF` |
| 62 | one (volatile load, then walk) | correct `5` |
| 60 | lit[0] then lit[1] ... | first OK, second overruns (`0xC1`) |
| 63 | array a, then b, c, d | first OK, rest overrun (`0x0E`) |
| 51 | 16 walks over a local struct array | overruns (`0xB1`) |

> **The first data-dependent string walk in a domain succeeds. Every subsequent one fails.**

This subsumes the earlier "access pattern is the mechanism" finding and sharpens it: it is not
the loop construct per se, it is that a *second* walk happens at all. It also explains why the
whole campaign kept implicating `lit[1]` — `lit[1]` is simply whatever gets walked second.

### Still unexplained, and NOT to be folded into the above

`wd65` (plain pointer + `while` walk, shared array) **wedges**, while `wd62` (volatile pointer
load, same array, same walk) returns 5. Both perform a SINGLE walk, so "first walk works" does
not cover `wd65`. The failure modes also differ — `wd65` kills the domain outright rather than
overrunning and returning a marker. Treat as a separate open thread; do not assume one cause.

### Next: stage 66 — the same element, walked twice

Removes the last confound (different element, different array, different index) by walking
`capstone_probe_lit[1]` twice in a row and returning a 2-bit map. `3` refutes the statement
above; `1` confirms it exactly.
