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

| arm | VA | function | instruction |
|---|---|---|---|
| `C7` | `0x5256c` | `sqlite3VdbeMemGrow+0x188` | `ldc a0, 0x30(a0)` |
| `R7` | `0x525a4` | `sqlite3VdbeMemGrow+0x188` | `ldc a0, 0x30(a0)` |
| `S61` | `0x524ac` | `sqlite3VdbeMemGrow+0x164` | `ldc a0, 0x30(a1)` |
| `MG` | `0x53064` | `vdbeMemClearExternAndSetNull+0x3c` | `ldc a1, 0x0(a0)` |

`C7`/`R7` agreeing is weak — `R7` is `C7` shifted by a uniform `0x38` (the REDRAW pad), so the two
are not independent. `S61` and `MG` are separately compiled binaries with different layouts, and
they land in `Mem`-handling code too.

Offset `0x30` in `struct sqlite3_value` (`Mem`) is `sqlite3 *db`, with 16-byte capability pointers:
`u` 0x00(16), `z` 0x10(16), `n` 0x20, `flags` 0x24, `enc` 0x26, `eSubtype` 0x27, pad, **`db` 0x30**.
`MEMCELLSIZE` is `offsetof(Mem,db)` — the same 0x30.

## What it means

**Capabilities pointing at `Mem` structures are out of bounds on silicon.** Not untagged — the tag
was never the issue — and not a permission fault: bounds.

The `MG` arm is the informative one. It carried a probe at the top of `sqlite3VdbeMemGrow` that
reports `pMem`'s bounds and returns `SQLITE_NOMEM` to stop the flow. It fired — the wedge MOVED, to
`vdbeMemClearExternAndSetNull` on the error path — and wedged there with the same mcause on a
**different** `Mem` dereference, this one at **offset 0**. An OOB at offset 0 means the cursor is
outside `[base, end)` altogether, not merely that the object is short.

So this is not one broken instruction. It is a class: multiple, independent `Mem` dereferences
fault, and `OP_Column` is simply the first place on this workload where one is reached. That is
consistent with the project's pre-existing OOB probe note — "p is the HEAP capability with a cursor
past its end" — and with the `db` measured on boot27 as a valid capability pointing at the wrong
object 0x3e00 away.

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
