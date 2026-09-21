# A merge primitive for coalescing allocators — PROPOSAL

*Status: PROPOSED (2026-09-21), not implemented, not measured. An architecture
argument, written for the paper's discussion of Sublet's limits; the paper is
not edited by this note. The motivating measurement is committed and cited
below. Revised the same day: the first version said per-chunk Sublet was
impossible on a coalescing allocator, which is wrong — the individual free
can be revoked — and a variant claimed to preserve the allocator's policy
was then built and refuted at its first `mrev`. Both corrections are in
"What the primitives allow, and what they refuse". Companion to `bounded-heap-allocator-proposal.md` (which first noted
that a `SPLIT`-based `free` "would have to merge it back") and
`heap-temporal-safety-revoke-on-free-proposal.md` (whose slab step exists
because "no coalescing, so no merge needed").*

## TL;DR

Capstone's capability-management instructions are `split`, `shrink`,
`shrinkto`, `tighten`, `delin`, `mrev`, `revoke`, `drop`, `init`, `seal` and
the two cursor moves (`capstone-qemu/target/riscv/insn32.decode:958-969`).
There is a `split` and no inverse, and `mrev` — the only way to make an
object revocable on its own — takes a **linear** source, in the RTL
(`capstone_dyn_unit.anvil:81-82`, `UNEXPECTED_CAP_TYPE` otherwise) as in the
emulator (`op_helper.c:1001`). A linear object capability comes only from
splitting a linear source, which consumes it. So an allocator that gives
each object its own handle holds its storage as linear pieces, and when it
**coalesces** freed neighbours — Wireshark's `wmem` `block` backend, every
dlmalloc descendant — it has no way to re-form two pieces into one object
except by revoking a handle senior to both, which kills everything else that
handle covers. Per-object temporal safety on Capstone therefore composes
with non-coalescing allocators exactly and with coalescing ones only at the
price of their policy: a request that only a coalesced run could serve must
go elsewhere. The Sublet port of `wmem` chose the allocator's policy over
the one defect that distinction costs: case 12 (`xml`, fix `90bb3a5c9e`), a
stale pointer whose lifetime ended by an individual `wmem_free`, **completes
under the port** and **faults under PoisonCap**, whose retirement is on the
memory side and re-forms nothing. The proposal is a `merge` instruction —
two adjacent linear capabilities in, one linear capability out, no
revocation — with the revocation-tree precondition that makes it sound. It
is what would let a per-object port keep a coalescing allocator's policy.

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
retired. The shipped port attaches nothing there; what a hook would cost
is the next section.

## What the primitives allow, and what they refuse

**The individual free can be revoked.** Sublet's recipe
(`runtime/include/sublet/sublet.h`, head comment) is: carve an object with
`split`, hand it out with `mrev` + `delin`, take it back with `revoke` on its
handle. `sublet_give` at a `wmem_free` would end case 12's object exactly as
the PostgreSQL port ends every `pfree` (`ports/postgres/sublet/README.md`,
"The recipe, operation by operation"). The shipped `wmem` port does not do
it because it holds authority per *region* — one 8 MiB block is one `carve`
and one `take` (`src/allocators/sublet/regions.c`), and every object is a
`shrink` of the block's alias, which keeps the alias's node and so can be
revoked only with the block. That is a port decision, made for the reason
below, not a limit of the hardware; the first version of this note said
"Sublet has no hook to attach there", and that was wrong.

**A per-object handle needs a linear object.** `mrev` refuses a non-linear
source in both implementations (RTL `capstone_dyn_unit.anvil:81-82`; QEMU
`op_helper.c:1001`), and a linear capability is produced only by `split` of
a linear source, which consumes it. An allocator carving objects with their
own handles therefore holds its block as linear pieces — the objects, and
the linear remainder — and never as a copy it could carve from again.

**Linear pieces do not re-join.** `wmem_block_free` merges the freed chunk
with any free neighbour (`wmem_block_merge_free`, `wmem_allocator_block.c:910`
at v4.6.8) and serves later requests from the union. Two linear pieces X and
C, adjacent and both freed, are two capabilities, and there is no
instruction that makes them one. The only merge available is the revoking
one, in the header's own words for *merge, reset, destroy*: "`sublet_give_to`
on the handle senior to the children: one revoke, whatever hangs below
dies." A handle covers what was derived from its source after the `mrev`
(`cap_rev_tree_revoke`: the junior run of nodes, `cap_rev_tree.c:112`), so it
covers X∪C only if X∪C was split as a unit — the undo of one split — and a
coalescing allocator merges arbitrary neighbours: A, C, D carved in order,
A freed, then C freed, and the only handle over A∪C is the block's, which
kills D. Even the undo case returns the union UNINIT when a linear child
hung below, and `sublet_give_to` zero-fills it (`sublet.h:65-67`).

**So a per-object port diverges from the allocator's policy**, in exactly
one decision: a request that fits no single piece but would have been served
from a coalesced run goes elsewhere — the next free chunk, or a new block —
and from that request on addresses and fragmentation differ from upstream's.
Every request that fits one piece is served identically. How often the
decision is exercised is a property of the workload; measuring it needs a
recorded trace of a real dissection, which this repository does not yet
have. A per-object port also moves the free-list node out of the freed chunk
(`WMEM_GET_FREE`, unreadable through a revoked alias, as `aset.c`'s
`GetFreeListLink` was) and routes every chunk-header access through slots,
since no alias to the block exists: a rewrite of the allocator's internals,
of the size the PostgreSQL port was.

The PostgreSQL port is the control for the other half: `aset.c` keeps
size-class free lists and never coalesces, so linear pieces are its policy
already, and per-chunk Sublet there is exact — a sixteen-byte capability slot
per chunk in a side table, and every individual free caught.

**The trade, stated once:** linearity is not optional for per-object
revocation on Capstone, and linearity is what coalescing cannot have. Three
ways out exist, and two have been taken:

1. **Authority at the coalescing unit** — the block. This is the shipped
   `wmem` port: epochs at reset, `release` at gc, nothing at an individual
   free. Twelve of thirteen, the allocator's policy byte-identical to
   upstream (the replay suite holds the hooked build to it), and one
   documented miss.
2. **Per-object, linear, policy-divergent.** Catches case 12; serves
   union-only requests elsewhere; a rewrite of `block`'s internals. Not
   built. It is option (a) in `plans/wireshark-wmem.md`, and the divergence
   count on a real trace is the measurement that would decide whether it is
   "`wmem` with one different coalescing rule" or a different allocator.
3. **Retire on the memory side.** PoisonCap poisons the chunk's granules and
   sweeps; the storage re-joins by ordinary pointer arithmetic afterwards
   because nothing about authority was ever split. Taken, as the fourth arm,
   13 of 13.

## The variant that was tried, and where it died

A fourth way was proposed on 2026-09-21 as policy-preserving: keep the
block as the non-linear alias the port already holds, `split` each object
from a *copy* of it (split accepts a non-linear source, in the RTL at
`capstone_dyn_unit.anvil:120` as in QEMU at `op_helper.c:1121`, and gives the
new half a node of its own), `mrev` that piece, `delin` it, and revoke the
handle at the free — so that the allocator always carves afresh from a
senior copy and never needs to re-form anything. It was built
(`src/allocators/sublet/chunks.c`, a mode 2 of the domain build, 584 diff
lines, retained outside the repository with the run) and booted once on
case 12: the emulator aborted at the first object's `mrev`,
`helper_csmrev: Assertion 'rs1_v->val.cap.type == CAP_TYPE_LIN' failed`, and
the RTL raises `UNEXPECTED_CAP_TYPE` at the same instruction. The split step
was sound; the handle step is what the ISA refuses, and it is the step that
matters. The variant is withdrawn, and the sentence "linearity is not
optional" above is its residue.

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

**Can**: that the port measured a limit and named its cause — per-object
revocation on Capstone needs a linear object (`mrev`), linear pieces do not
re-join (`split` and no `merge`), so per-object temporal safety composes
exactly with non-coalescing allocators (PostgreSQL, the slab proposal) and
on coalescing ones either stays at the coalescing unit (`wmem` `block`, one
miss in thirteen, policy intact) or changes the allocator's policy; that a
memory-side scheme (PoisonCap) is indifferent to the distinction, and the
corpus shows that as 13 / 13 against 12 / 13; that the missing primitive is
one instruction with a stated soundness condition.

**Cannot**: that per-object Sublet is impossible on `wmem` (it is possible,
at the price above) or that it is policy-preserving (it is not, and the
variant that claimed so died at `mrev`); that `merge` is cheap in silicon
(the revocation tree is a linearised list with depth tags in both
implementations; parent links may not be free), that it has been simulated,
or that it would have caught case 12 — a claim that needs the primitive
built and the case rerun. Nor that CHERI's deployed configuration catches
case 12: it does not (0 / 13); only the PoisonCap extension, itself a
proposal, does.

## Considered and rejected

- *Documenting only the miss, without the primitive.* A limitation with no
  named remedy reads as a property of the approach; this one is a property
  of one missing instruction, and the paper should say which.
- *Building the per-object linear variant now.* It answers "can Sublet
  catch case 12" with yes and "at what cost" with a number that needs a real
  dissection trace to mean anything; the PostgreSQL port already shows the
  discipline exact on a non-coalescing allocator, and the cost side is the
  paper's to want before a multi-day port is spent on it.
- *Implementing `merge` in the emulator first.* Cheap in QEMU, but a
  measured 13 / 13 under an instruction that exists nowhere else would be
  quoted as a result; the argument stands without it, and the emulator work
  belongs with an RTL plan, which the standing synthesis rules put after
  everything the current bitstream generation needs.
