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

## Next step

Measure the bounds, do not infer them: report `aMem` and `&aMem[p3]` at `sqlite3VdbeExec` entry
under a clamp that is known to return. Then the question is where a `Mem` capability with a cursor
outside its bounds is manufactured — candidates are the register-array indexing, and the interaction
with `-capstone-shrink-*` being off while the heap capability is the one carried around.
