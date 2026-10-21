# The `delin` residuals — every delin the C-27 fix does NOT remove, named

C-27's fix (`299465232cc3`) drops the `delin` on gp-derived global bases under the image-gp ABI.
Every OTHER `delin` in an image survives it, so each survivor is either a legitimate LINEAR operand
or the next cause 26. The board lane asked for them named before an image goes near the board.
This is that list, measured — not reasoned from the source.

## Instrument

`delin-residuals.py` is the exact complement of the board lane's `delin-census.py`
(`vp2-tier12 2a01393a18e4`, `tests/rtl-smoke/vp2-kit-2026-10-17/delin-census.py`): same
`INSN` regex, same `DERIVE` set, same `WINDOW = 5`, same "nearest writer wins" rule. It lists the
delins that census does **not** count, with the nearest writer and ten preceding instructions.

It asserts the complement adds up (`residuals == delins - gp_derived`) per image and exits non-zero
if it does not, because a complement that does not add up means the two scripts disagree about the
population and then neither number means anything. It held on all three images.

Cross-check worth recording: running the board lane's census from **this** lane's independently
built `llvm-objdump` reproduced their counts exactly — 525/530, 5,983/5,988, 9,859/9,863. Two
lanes, two build directories, same numbers.

## The residuals

| image | delin | gp-derived | residual |
|---|---:|---:|---:|
| contract.dom `bf433e107032fa8a` | 530 | 525 | **5** |
| sqlite3.dom `a4b2c0f05809f433` | 5,988 | 5,983 | **5** |
| mruby `5fee8d8f604229ea` | 9,863 | 9,859 | **4** |

**They are all one shape** — `ldc rd, off(rs); delin rd`, a capability LOADED FROM MEMORY and then
delinearised. Not one is fed by `scc`/`cincoffset` off anything, gp or otherwise. And they are the
same sites in all three images, so they live in the shared runtime/allocator glue rather than in
application code:

| site (contract.dom) | function | nearest writer |
|---|---|---|
| `0x117c4` | `capstone_virtual_thread_create + 0xb0` | `ldc s6, 0x0(s7)` |
| `0x120cc` | `mmap + 0x138` | `ldc a0, 0x0(s2)` |
| `0x1fb30` | `__capstone_malloc_group_attach + 0x68` | `ldc a4, 0x0(a1)` |
| `0x1fe60` | `__capstone_malloc_group_map + 0x94` | `ldc a5, 0x0(a0)` |
| `0x20954` | `__capstone_malloc_group_remap + 0x2b8` | `ldc a3, 0x0(s5)` |

mruby has four: it lacks the `capstone_virtual_thread_create` site, which is consistent with it not
creating threads.

The surrounding code is the allocator's own linear-capability handling — `stc`/`ldc` round trips
with `lcc` selector queries between them, which is the `CapstoneLiveSourceCopy` rewrite shape (a
`movc` whose source is read again becomes `stc`+`ldc` through a slot). A capability that the
allocator stored LINEAR and reloads is a legitimate `delin` operand.

## What this does NOT settle, and the measurement that does

**Whether each residual is LINEAR cannot be read off the disassembly.** `ldc` returns whatever type
was stored, so the operand's type is a dynamic property. Any static verdict here would be a guess
with a function name attached.

The strict-DELIN QEMU pin settles it empirically and cheaply. On
`capstone-qemu virtual-capstone-v7 852c5bc7677440b95736b5fc22bc158cce3f6479`, `helper_csdelin`
raises 26 for any non-LINEAR operand and 24 for an untagged one — so once the 525/5,983/9,859
gp-derived delins are gone, **any residual that is not LINEAR faults and names itself by pc.**
The reading is therefore:

* rebuilt images pass `run.py` on that pin → all residuals are LINEAR, and the class is closed;
* a fault → its pc names which of the five, and that one is a second defect in its own right.

So the residuals do not need a separate investigation ahead of the rebuild; they need the rebuild's
own run, read for a cause-26 at one of these five addresses rather than at the gp sites.

## The gate's blind spot, which the board lane named

A window count can fall to zero without the delin being gone: if codegen moved the definition more
than five instructions from the delin, `delin-census.py` would read 0 while the instruction is still
emitted. So the rebuilt images are reported with **totals beside the zeros** — the total must drop
by close to the old gp-derived count (525 / 5,983 / 9,859). A zero with an unmoved total is a
vacuous pass, not a fix.
