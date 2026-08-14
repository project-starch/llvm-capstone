# The SQLite wedge is OUT_OF_BOUNDS on `Mem` capabilities — read from trap state already captured

Date: 2026-08-14. Bitstream `caplifive_12august.bit`. Configuration `SQLITE_LDC_HIGH_HALF_FIXUP=1`.

Follow-on from `13-08-2026_23-00-00_sqlite-wedge-localised-to-op-column.md`, which localised the
wedge to executing `OP_Column`. This note says what the fault *is*.

## It was already in the logs

The board driver dumps latched trap state on every wedge, and the bitstream exposes it on switches
196–203 and 255. Every read I had done was `sed -n '/=== STAGED BISECTION/,$p'` — which starts
*below* those lines. Five wedges' worth of trap state had been captured and never displayed.

**`sw=255 TRAP LOG {seen, mcause[6:0]} = 0x9d` on every one of them**: `seen=1`, `mcause = 29`.

29 is OUT_OF_BOUNDS on the ldc/stc DYN path (the `24+code` encoder). It is unanimous across five
wedges from four separately compiled binaries:

| arm | latched mepc | mcause |
|---|---|---|
| `C7` | `0x8304256c` | 29 |
| `R7` | `0x834425a4` | 29 |
| `C8` | `0x8344937c` | 29 |
| `S61` | `0x830424ac` | 29 |
| `MG` | `0x82c43064` | 29 |

## Mapping mepc to an instruction

The domain is allocated with `__get_free_pages(..., dom_pages_log2)` — a buddy allocation, hence
**naturally aligned to its order**. `dom_tot_size = code_len + max(code_len, 64 KiB)` ≈ 3.1 MB for
these images, so the order is 10 and the base is **4 MiB-aligned**
(`caplifive-system/sw/buildroot/package/modcapstone/module/capstone.c:107-113`). The loadable
segment's VA is `0x10000`, so `VA = (mepc & 0x3FFFFF) + 0x10000` recovers the image address
exactly. (An earlier, weaker assumption of 1 MiB alignment gives the same answers and is implied.)

| arm | VA | function | instruction | source construct |
|---|---|---|---|---|
| `C7` | `0x5256c` | `sqlite3VdbeMemGrow+0x188` | `ldc a0, 0x30(a0)` | `sqlite3DbMallocRaw(pMem->db, n)` — the `szMalloc <= 0` branch |
| `R7` | `0x525a4` | `sqlite3VdbeMemGrow+0x188` | `ldc a0, 0x30(a0)` | same |
| `S61` | `0x524ac` | `sqlite3VdbeMemGrow+0x164` | `ldc a0, 0x30(a1)` | `sqlite3DbFreeNN(pMem->db, pMem->zMalloc)` — the **opposite** branch |
| `C8` | `0x5937c` | `sqlite3VdbeExec+0x527c` | `ldc a0, 0x50(a1)` | `pDest->z = pDest->zMalloc` — inlined in OP_Column, **not in VdbeMemGrow at all** |
| `MG` | `0x53064` | `vdbeMemClearExternAndSetNull+0x3c` | `ldc a1, 0x0(a0)` | second-level load off a reloaded `Mem*` |

`C7`/`R7` are the same site; that pair is a REDRAW control that passed, not a duplicate.

## RETRACTION: this is NOT "the `pMem->db` load"

An adversarial audit refuted the field-specific reading, and the refuting datum was **in my own
evidence list**: `C8`'s mepc, which I failed to decode (its disassembly step errored and I never
returned to it) and then silently dropped from the table. `C8` faults in `sqlite3VdbeExec`, on
`pDest->zMalloc` at offset 0x50 — a different field, a different function, not `db` at all. `S61`
likewise faults on the *opposite* branch of `VdbeMemGrow` from `C7`. Writing this up as "the wedge
is the `pMem->db` load" would have sent the next session chasing a field-specific cause that one of
the five samples already contradicts.

**What survives, and it is not field-specific:** every fault is an `ldc` — a CAPABILITY-typed load
— of a capability field out of a `Mem`/`pDest` object during `OP_Column`'s dynamic extent, at
whatever offset that build happens to reach first (0x30 `db`, 0x50 `zMalloc`, 0x0 off a reload).
Meanwhile the *scalar* `lw 0x40` (`szMalloc`) in the same objects never faults — and returns
inconsistent values across builds differing only by a clamp constant or a dead pad, which is why
different arms take different branches. Plain integer loads are documented as unchecked in our
domains, so a scalar read succeeding proves nothing about the object.

That points at **the `pDest` capability's bounds**, not at any field. It is a hypothesis about the
cause; the measurement is the five faults and their common shape.

## Caveats the audit put on the record

* **N=1 per wedged variant.** No wedging configuration was run twice. The replication here is
  CROSS-VARIANT — four boots, four distinct binaries, one cause. The clean arm (`C6`) *did*
  reproduce identically across two boots. Given this project's history of a PASS→FAIL flip voiding
  a bisection, one re-run of `C7` in a later batch is cheap insurance and has not been done.
* **mepc proves where the last capability exception was, not that the core stopped there.**
  Supporting the identification: the latch keeps the latest trap with `cause ∉ {0,2}` and is
  cleared only by reset, Linux runs between tests and every syscall/page-fault would overwrite it
  with a *virtual* pc — so a surviving `cause 29` at a bare-physical `0x83xxxxxx` postdates all
  inter-test kernel activity. And `sw=224 = 0x9f` shows `ex_commit.valid=1, privM=1` sampled
  minutes after the hang. The surviving alternative — trapped here, resumed, wedged later without
  trapping — is unlikely but not excluded.
* **The mepc→VA recovery is SIZE-DEPENDENT, not a general trick.** It works because SQLite forces
  an order-10 (4 MiB) allocation. A small domain gets a low order and less than 1 MiB of alignment,
  and masking would then be unfounded.
* **Artifact identity** for `C7`/`R7`/`S61` rests on the driver's in-session gate (overlay bytes
  are inside the flashed initramfs) plus size coherence; their overlay copies have since been
  overwritten. `C8` is verified end to end against its bake directory.

## It does not reproduce under QEMU

`CAPSTONE_OOB_PROBE=1` with the fixup on: probe injected (build log confirms), report never fires,
workload passes end to end. Board-only. The probe is known to be capable of firing — it is the
instrument that produced the "cursor past its end" observation historically — so this zero is
informative rather than uninstrumented.

## The instrument that produced this, and its limit

`CAPSTONE_MEMGROW_PROBE=1` reports `pMem`'s type/cursor/base/end at the top of
`sqlite3VdbeMemGrow` and returns `SQLITE_NOMEM`, chosen so the error propagates out through
`abort_due_to_error` and the ladder's stage 66 returns — a wedge takes the core and the host never
writes the payload, so a probe that does not stop the flow reports nothing at all.

**It did not survive**: the error path itself wedges, so the report never reached the host. The
measurement that *did* come out was the moved mepc, which is why the latched trap state matters
more than the payload here. To read the bounds directly, the probe needs to return through a path
that does not touch another `Mem` — most plausibly by reporting at the top of `sqlite3VdbeExec`
(before any opcode runs) on the register array `aMem` and the specific `&aMem[i]` involved, with
`CAPSTONE_VDBE_CLAMP=6`, which is already known to return.

## RESOLVED: the capability was never corrupt -- the BYTECODE OPERAND was

A probe inside `OP_Column`, reporting `pDest` at the last point before any `Mem` is dereferenced
and leaving through `goto vdbe_return` (the clamp's escape, measured to return on silicon), got the
measurement three earlier instruments could not:

```
silicon:  PDEST p3=6a7 p2=0 cap: t=1 c=8301d360 b=82fbe200 e=82ffe400 hoff=OUT
QEMU:     PDEST p3=1   p2=0 cap: t=1 c=101feeac0 b=101fbe3a0 e=101ffe3a0
```

`pOp->p3` is the VDBE instruction's destination-register operand. It is 1. On silicon it reads
**1703**. `&aMem[1703]` is 1703 x 112 bytes past `aMem`, at `0x8301d360` -- outside the heap
(`end 0x82ffe400`), which is what `hoff=OUT` says.

**So the capability machinery was right all along.** The hardware raised OUT_OF_BOUNDS on a pointer
that genuinely was out of bounds. Everything above about damaged bounds, cursor re-encoding and
spill/reload is superseded: the bounds were fine, the tag was fine, and the two benign RTL
explanations (the reserved 16-byte tail, a SEALEDRET window) are moot because the cursor is
0x1e000 past the end and the type is 1. `s06bnds` returning 65535 fits -- nothing was ever wrong
with spill/reload.

It also explains the older loose ends: boot27's `db` arriving as a valid capability pointing at the
wrong object is what a wrong register index produces, as is the auditor's note that the scalar
`szMalloc` read returned inconsistent values across builds.

**The cause is unguarded compiler-emitted 16-byte granule copies -- S-06 -- corrupting the VdbeOp
array.** Matched pair, one boot, control returning:

| arm | | `p3` | pointer |
|---|---|---|---|
| `PD` guard off | | `0x6a3` (1699) | `hoff=OUT` |
| `PDG` guard on | | **1** | `hoff=30720`, QEMU's value |

`p3` also differs between boots with the guard off (`0x6a7`, `0x6a3`) -- garbage, not a fixed
constant.

## WITH THE GUARD ON: the basic workload PASSES on silicon

Ladder arms, guard on, one boot, control returning:

| arm | | result |
|---|---|---|
| `G6` after the row loop | | **returned `rc=3`, printed alpha/beta/gamma** |
| `G7` after finalize | | returned `rc=0`, three rows |
| `L2` control | | returned |
| `G9` full workload | | WEDGED (entered) |

CREATE, INSERT, the SELECT, all three rows and finalize now work end to end on hardware. The
residual failure is in `run_sqlite_extended`, past everything the ladder covers.

## The residual failure is a DIFFERENT defect

`G9`'s latched trap state: `sw=255 = 0x99` -> **mcause 25 = UNEXPECTED_OPERAND**, not 29, at

```
sqlite3DbMallocRawNN+0xd8   ldc a0, 0x2a0(a0)
```

`0x2a0` is exactly the `db->lookaside.pSmallFree` offset measured on boot27 (`fld - c = 0x2a0`).
mcause 25 means the operand is NOT_CAP -- an untagged value where a capability belongs. That is the
original lookaside fault the whole investigation started from, and it is **not** the out-of-bounds
defect this note is about: different cause, different mechanism, different fix.

So the guard resolves the OOB/corrupt-operand defect completely, and what remains is the separate
question of how `pSmallFree` comes to hold plain data -- boot27 measured it holding the ASCII
`' WHERE '`. A wild write or a granule copy at a site the guard does not cover are both live.

## The residual failure, bisected: extended phase 2->3, inside our own memcpy

`CAPSTONE_EXT_STOP` ladder, guard on, one boot:

| arm | | result |
|---|---|---|
| `E1` | | returned |
| `E3` | | WEDGED (entered) |

So it is extended phase 2 or 3 -- `CREATE INDEX idx_amount` (or its matched `ext-index-control`
arm). E3's latched trap: mcause **25 UNEXPECTED_OPERAND** at

```
memcpy+0x2a8:
   ldc          a2, 0x0(a2)      ; reload the pointer from a stack slot at s0-0x60
   cincoffset   a1, a2, a1       <== raises: a2 is NOT_CAP
   sb           a0, 0x0(a1)
```

An untagged pointer reloaded from a stack slot inside our `memcpy`'s byte loop. That is the exact
shape `s06spill` was built to test -- and `s06spill` returns 0xFFFF, all sixteen tags surviving.
So a bare spill/reload preserves the tag while this one does not, and whatever distinguishes them
is the remaining question. `s06bnds` adds that bounds survive too, so neither simple round-trip
property is at fault.

Note this is the SECOND distinct defect on this path, not a variant of the first: mcause 25 (lost
tag) here versus mcause 29 (out of bounds, from a corrupt operand index) for the one the granule
guard fixes.

## STILL OPEN: the guard fixes the operand, it does not make SQLite run

`FIXON` -- schema fixup on, `-capstone-guard-cap-granule-copies` on, no probes, no ladder -- enters,
emits garbage to the UART and wedges. A build that repairs the one measured corruption is not yet a
build that runs. Whether what remains is the same defect at a site the guard does not cover, or
something that was simply unreachable before, is not established.

**Latch limitation, first observed here.** That wedge's latched trap state reads mcause 9 at
`0xffffffff800072cc` -- a KERNEL virtual address. Cause 9 passes the latch's `cause not in {0,2}`
filter, so kernel activity after the domain died overwrote it. The argument for trusting the
earlier cause-29 readings was exactly that kernel traps would have overwritten them; this run shows
the overwrite happening. A kernel VA in that field means NO information about the domain fault.

## Next step

Measure the bounds, do not infer them: report `aMem` and `&aMem[p3]` at `sqlite3VdbeExec` entry
under a clamp that is known to return. Then the question is where a `Mem` capability with a cursor
outside its bounds is manufactured — candidates are the register-array indexing, and the interaction
with `-capstone-shrink-*` being off while the heap capability is the one carried around.


## Step 1 (characterise the residual defect) -- where it got to

Three ladder rungs, each with a firing positive control, refuted every simple explanation on
silicon:

| rung | question | silicon |
|---|---|---|
| `s06spill` | does a spilled capability come back TAGGED? | 65535 |
| `s06bnds` | ...with its BOUNDS intact? | 65535 |
| `s06wr` | ...surviving byte stores written THROUGH it? | 65535 |

**The capability round-trip machinery is sound on this hardware.** That is the load-bearing
conclusion for the S-06 decision: S-06 is plain data losing its high half on an untagged line, and
this defect is a lost tag on a genuine capability, so an S-06 RTL fix should not be expected to
clear it.

A fourth rung calling the REAL memcpy was written and deleted: `build-ladder-domain.sh` compiles
exactly one C file, so a rung never links `beebs_freestanding_string.c` and its `memcpy` is a
different primitive from the one SQLite calls.

The in-domain instrument (`BEEBS_MEMCPY_TAGCHECK`) is built, positive-controlled end to end, and
**did not stop the wedge**; its arm's latch was overwritten by kernel activity, so no location.
Two readings remain open and unseparated: the faulting call is not on the instrumented path (the
chunk loop rather than the byte tail), or the destination is tagged at the check and untagged at
the use. The second is the stronger claim and has no evidence yet.
