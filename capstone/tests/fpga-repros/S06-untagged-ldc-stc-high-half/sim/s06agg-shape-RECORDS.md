# s06agg-shape.S — run records

Directed test for the STRUCT-ASSIGNMENT shape of S-06: a 32-byte object whose first 16-byte chunk holds a real
capability and whose second chunk at offset `0x10` holds two plain `u64`s, moved by the two capability-grained
copies the compiler emits for `*d = *s`. Written 2026-09-09 after board rung `s06agg` returned 66 on boot sw46,
because no existing arm creates that shape: `untagged-ldc-stc-128.S` round-trips offsets 0 and 32 with no
capability adjacent to the untagged chunk, and the `s06sec-*` arms test security properties, not this.

Codes: PASS both halves survived · 11 high half (`y`) lost, `x` intact — S-06's signature, what the rung's 66
encodes · 12 low half lost · 13 both lost · 14 the plain `sd`/`ld` control is wrong, so the run carries NO
verdict · 20 unexpected trap.

## ADJACENCY IS THE TRIGGER — read this before the table

The first version of this test placed **four instructions** between the `sd` of `y` and the `ldc` of that line,
including another plain store and the whole pointer-chunk copy. It **passed on all three revisions**, and that
pass was reported as if it exonerated the RTL. It did not: it contained the shape without creating the
condition.

The committed rung `s06agg.dom` (`249118220f8c`) has them adjacent — at `0x10438`, `sd a0, 0x18(a2)` writing
`src.y` and then `ldc a3, 0x10(a2)` reading the whole line as the very next instruction, with `src.x` stored
several instructions earlier and its constant build in between. Rewritten in that order, **the same test fails
on every revision.** The half the rung loses is the half whose store had not left the write buffer, which makes
this a **store-to-load-forwarding** shape — an 8-byte buffered store into the upper half of a 128-bit untagged
load — as much as a tag-path one.

## Readings

All runs on the delay-40 model, the define read back from `work-ver/Variane_testharness__verFiles.dat` by the
runner, in capability mode after `CAPENTER`.

| revision | what it is | adjacent (the rung's order) | apart (the first, wrong version) |
|---|---|---|---|
| `5097eb166` | the bitstream that flew BEFORE this flash | **FAIL 11**, 1984 cyc | PASS, 1828 cyc |
| `ef5a8eaf2` | the clean line this cycle branched from | **FAIL 11**, 1984 cyc | PASS, 1828 cyc |
| `66c4e7517` | the flashed R-25/26/27 bitstream | **FAIL 11**, 1984 cyc | PASS, 1828 cyc |

In every failing run `x` reads back `1111222233334444` and `y` reads back `0000000000000000` — not merely
wrong, **zeroed**, which is the board's original signature of low eight bytes kept and high eight zeroed. The
plain `sd`/`ld` control reads `0f0f0f0f0f0f0f0f` in all six runs, so each carries a verdict.

`5097eb166` is run as its own arm rather than inferred, because it is **not an ancestor of `ef5a8eaf2`** and the
S-06 tag-path files differ between them (`load_store_unit.sv`, `ex_stage.sv`, `cache_subsystem/wt_dcache.sv`
among eleven `core/` files). Ancestry would not have answered it.

## What this establishes

The struct-assignment shape **loses the high half on every revision we can build, including the silicon that
flew before the flash.** Therefore the board rung's 66 on boot sw46 is **not** a regression introduced by
`66c4e7517`, and the struct-assignment half of S-06 was **never fixed** — its acceptance cited a different
program under the same name, and the shape itself fails.

## What it does not establish

Which mechanism. The adjacency result points at store-to-load forwarding rather than the S-06 tag path, and
that is a different family (S-07/S-10). This test does not separate them; it shows the condition and the
signature. It also remains a bare M-mode test against `.data` buffers, where the rung runs inside a
monitor-created domain on a carved stack reaching globals through a cap table.
