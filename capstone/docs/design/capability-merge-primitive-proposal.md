# A merge primitive for coalescing allocators — PROPOSAL

*Status: PROPOSED (2026-09-21), not implemented, not measured. An architecture
argument, written for the paper's discussion of Sublet's limits; the paper is
not edited by this note. The motivating measurement is committed and cited
below. Companion to `bounded-heap-allocator-proposal.md` (which first noted
that a `SPLIT`-based `free` "would have to merge it back") and
`heap-temporal-safety-revoke-on-free-proposal.md` (whose slab step exists
because "no coalescing, so no merge needed").*

## TL;DR

Capstone's capability-management instructions are `split`, `shrink`,
`shrinkto`, `tighten`, `delin`, `mrev`, `revoke`, `drop`, `init`, `seal` and
the two cursor moves (`capstone-qemu/target/riscv/insn32.decode:958-969`).
There is a `split` and no inverse. An allocator that **coalesces** freed
neighbours — Wireshark's `wmem` `block` backend, and every dlmalloc
descendant — therefore cannot hold per-object Capstone authority: it needs
the freed chunk to re-join its neighbours as one object, and the only way to
re-join two linear capabilities today is to revoke a handle senior to both,
which kills everything else that handle covers. The Sublet port of `wmem`
keeps authority at block granularity for that reason, and the corpus shows
the cost exactly once: case 12 (`xml`, fix `90bb3a5c9e`), a stale pointer
whose lifetime ended by an individual `wmem_free` into the recycler,
**completes under Sublet** and **faults under PoisonCap**, whose retirement
is on the memory side and needs no capability to be re-formed. The proposal
is a `merge` instruction — two adjacent linear capabilities in, one linear
capability out, no revocation — with the revocation-tree precondition that
makes it sound. Until it exists, the honest statement is: *authority-side
temporal safety composes with non-coalescing allocators; memory-side
retirement composes with both.*

## The measurement that forces the question

`bug-corpora/wireshark/wmem-repros/`, thirteen upstream-fixed defects, four
systems, 2026-09-21 (`results/20260921-qemu/`, `results/20260921-cheribsd/`):

| system | lifetime event it acts on | caught |
|---|---|:--:|
| Capstone spatial | none | 0 / 13 |
| Sublet | the block's epoch at a pool reset | 12 / 13 |
| CheriBSD, libc revocation on | `free()` to libc — which `wmem` never issues | 0 / 13 |
| PoisonCap | poison at reset, at release, and at the recycler's individual free | 13 / 13 |

Twelve cases end a lifetime at a packet-pool reset, and there the two
mechanisms agree. Case 12 ends it inside a live 8 MiB block, by
`wmem_block_free` → `wmem_block_merge_free` (`wmem_allocator_block.c:893,910`
at v4.6.8): the chunk is merged with any free neighbour and put on the
recycler, and the stale registry name reads it on the next packet. PoisonCap's
counters at that arm's marker read `epochs=0 released_chunks=1` where the
other twelve read `epochs=1 released_chunks=0`: no epoch ended, one chunk was
retired. Sublet has no hook to attach there, and the reason is not the port.

## Why Sublet cannot be per-chunk on a coalescing allocator

Sublet's recipe (`runtime/include/sublet/sublet.h`, head comment) is: carve
an object with `split`, hand it out with `mrev` + `delin`, take it back with
`revoke` on its handle, and for *merge, reset, destroy* — the header's own
words — "`sublet_give_to` on the handle senior to the children: one revoke,
whatever hangs below dies." That is the only merge available, and it is a
**revoking** merge: the handle senior to two adjacent chunks is the handle of
whatever they were split from, in practice the block, and revoking it kills
every live object in the block. A coalescing allocator merges *arbitrary*
adjacent free neighbours, whose split histories share no ancestor of the
right extent, so no hierarchy of handles taken in advance can match the
boundaries `merge_free` will choose at run time.

The PostgreSQL port is the control. `aset.c` keeps size-class free lists and
never coalesces, so the Sublet port there is per-chunk — `sublet_give` at
every `pfree`, a sixteen-byte capability slot per chunk in a side table
(`ports/postgres/sublet/README.md`, "The recipe, operation by operation") —
and its individual frees are caught. Same discipline, same hardware; the
allocator's coalescing policy is the whole difference.

Three ways out exist, and two have been taken:

1. **Authority at the coalescing unit** — the block. This is the shipped
   `wmem` port: epochs at reset, `release` at gc, nothing at an individual
   free. Twelve of thirteen, and a documented miss.
2. **Stop coalescing.** A `block` variant that never merges would let the
   port be per-chunk, at the cost of the port's fidelity claim: the shipped
   allocator is blob-identical to upstream from 4.4.0 to 4.6.8 and the replay
   suite holds the hooked build byte-identical to it, so a non-coalescing
   variant is a different allocator with `wmem`'s interface, whose
   fragmentation under a long capture is unbounded by design (upstream added
   coalescing to bound it) and whose reoccupation patterns no longer
   reproduce the ones the corpus's spatial arms observe. Not taken; this is
   option (a), and the plan (`plans/wireshark-wmem.md`) points here for it.
3. **Retire on the memory side.** PoisonCap poisons the chunk's granules and
   sweeps; the storage is re-joined by ordinary pointer arithmetic afterwards
   because nothing about authority was ever split. Taken, as the fourth arm.

## The primitive

    merge rd, rs1, rs2

*Preconditions*, all checked in hardware, any failure a capability fault:

- `rs1` and `rs2` are both linear (`LIN`), valid, same permissions, same type
  and seal state;
- `rs1.end == rs2.base` — exactly adjacent, `rs1` below;
- their revocation-tree nodes are **mergeable**: the same node, or one is the
  parent of the other, or both have the same parent (the three shapes one or
  two `split`s produce). This is the soundness condition: after the merge,
  every handle that could reach either half must reach the union, which
  holds when the nodes share the ancestor chain above the merge point.

*Effect*: `rd` = `[rs1.base, rs2.end)`, cursor `rs1.base`, linear; its node is
the more senior of the two (the parent if one is the parent, else `rs1`'s);
the other node is released; `rs1` and `rs2` are consumed, as `split` consumes
its source.

*In the emulator* this is one helper beside `helper_cssplit`
(`op_helper.c:1099`, which allocates the second half's node with
`cap_rev_tree_split`, a duplicate placed before the parent in the linearised
tree, `cap_rev_tree.c:107`) and one tree operation, `cap_rev_tree_merge`,
which checks the three shapes against the parent links, then calls
`cap_rev_tree_release` on the junior node (`cap_rev_tree.c:139`, which
asserts the node is reusable — a node with live descendants is not, and that
assertion is the emulator's form of the precondition).

*What it does not do*: merge across a `delin` (an alias is not linear; the
allocator that lost linearity must `revoke` to get it back, as now); merge
two halves whose handles are outstanding (an outstanding `mrev` on a half is
a live descendant, so the node is not reusable and the merge faults —
`sublet_give` first, then `merge`, which is the order a `free` runs anyway);
change anything about aliases held by the program (a merged region has no
aliases, by the precondition).

*Cost*: one instruction per coalescing free, replacing nothing (the shipped
port does no capability work at an individual free). The side table the
PostgreSQL port pays for is the same here: a capability slot per chunk,
which the `wmem` `block` chunk header, with its free-list node written into
the freed data, would have to move out of the chunk exactly as `aset.c`'s
`GetFreeListLink` did.

## What the paper can say, and what it cannot

**Can**: that the port measured a limit and named its cause — Capstone's
authority model has `split` and no `merge`, so per-object temporal safety
composes with non-coalescing allocators (PostgreSQL, the slab proposal) and
degrades to the coalescing unit on coalescing ones (`wmem` `block`, one miss
in thirteen); that a memory-side scheme (PoisonCap) is indifferent to the
distinction, and the corpus shows that as 13 / 13 against 12 / 13; that the
missing primitive is one instruction with a stated soundness condition.

**Cannot**: that `merge` is cheap in silicon (the revocation tree is a
linearised list with depth tags in both implementations; parent links may
not be free), that it has been simulated, or that it would have caught case
12 — a claim that needs the primitive built and the case rerun. Nor that
CHERI's deployed configuration catches case 12: it does not (0 / 13); only
the PoisonCap extension, itself a proposal, does.

## Considered and rejected

- *Documenting only the miss, without the primitive.* A limitation with no
  named remedy reads as a property of the approach; this one is a property
  of one missing instruction, and the paper should say which.
- *Building the non-coalescing variant now.* It answers "can Sublet be
  per-chunk on `wmem`" with a different allocator, which is not the question
  the corpus asks; the PostgreSQL port already answers it for a
  non-coalescing one.
- *Implementing `merge` in the emulator first.* Cheap in QEMU, but a
  measured 13 / 13 under an instruction that exists nowhere else would be
  quoted as a result; the argument stands without it, and the emulator work
  belongs with an RTL plan, which the standing synthesis rules put after
  everything the current bitstream generation needs.
