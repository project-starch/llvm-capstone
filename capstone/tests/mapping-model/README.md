# Caplified mapping tables: executable Stage-1 contract

This is a Python standard-library reference model of
[the mapping candidate](../../docs/design/caplified-mapping-tables.md), based on
contract commit `00626a232bda`, with the table-page write-permission precondition
made explicit during modeling in design §4. It implements the experiment in §10.1, including
deliberately broken comparison variants. It does not implement translation in
QEMU, the compiler, the runtime or RTL. Passing this model does not qualify any
of those layers or establish an unbounded hardware proof.

## Run

From the repository root:

```bash
source capstone/tests/capstone-test-env.sh
python3 -B capstone/tests/mapping-model/check.py \
  --output /tmp/capstone/mapping-model.json
```

Python 3.10 or later is sufficient; no packages or toolchain build are needed.
The common environment script can warn about an absent compiler build, which
this host-only experiment does not use. Run without `-O` or `PYTHONOPTIMIZE`;
the runner rejects disabled assertions.

`--scenarios-only` runs the named contracts and all mutation controls.
`--replay walk_only_drain` prints the correct and broken traces for that
variant. `--depth`, `--seeds` and `--steps` set the lifecycle search and random
trace bounds; defaults are 3, 32 and 100. A state-limit failure is an incomplete
search and a failing exit, never a passing bounded result.

The checked [result record](results.json) includes source SHA-256 values, Python
version, exact search bounds, state/edge counts, operation coverage and the
mutation traces. It contains results rather than environment logs. Reproduce
it with the command above and compare the JSON; no timestamps or hostnames are
included.

## Machine and instruction boundary

[model.py](model.py) represents actual capability storage in wallets, memory
words and protected registry entries. Walk snapshots and cached translations
are not extra software-owned capabilities. Revocation nodes have parents,
liveness and linearity; logical nodes carry their immutable mapping binding.
Mappings have finite ids and generations. Exhaustion refuses creation instead
of wrapping. Physical page identities are distinct even when logical offsets
are equal.

CREATE consumes one exclusive, writable physical root page. POPULATE consumes one
exclusive frame and, if necessary, one additional writable table page. Requiring
retained write permission for table pages prevents conversion from increasing
a read-only supplier's authority, even when its page is currently UNINIT.
Both expose
`prepare` and `publish` microsteps. Preparation holds the operands outside
software access; publication rechecks authority and atomically clears tags,
links tables and publishes the result. A concurrent revoke or DETACH can make
publication fail. The caller can abort and recover the original operands;
operands revoked in the meantime are still dead. There is one outstanding
preparation, but revocation, DETACH and ordinary accesses can interleave with
it. This abstracts internal zeroing and publication circuitry.

DETACH, UNMAP and REVOKE separate break, per-hart invalidation, per-hart drain
acknowledgement and final return. All use a conservative global barrier:
new accesses wait; old walks may progress until their hart invalidates them;
checked accesses may complete or cancel. A hart cannot acknowledge drain with
a checked access outstanding. No return occurs before both acknowledgements.
This over-invalidates compared with a targeted implementation; it makes no
performance claim. DROP of a linear logical capability uses the same barrier.

DESTROY consumes a whole-mapping token, requires completed DETACH, and releases
only the registry entry. Frames and lower tables can remain present. Their
senior physical handles work after root loss, after DESTROY and after a new
generation takes the id. Root revoke leaves a reserved registry entry holding
a dead root. No reverse index is consulted for correct revocation.

Logical SPLIT, DELIN, MREV, REVOKE, DROP, bounds/rights attenuation and cursor
movement are modeled. Moving and storing a linear capability removes the source;
non-linear capabilities can be copied. A linear capability load additionally
needs write permission because it clears its memory source. UNINIT scrubbing
advances one word at a time; INIT requires the cursor at the end. Logical
UNINIT also has a checked synchronous word-store abstraction for allocator
reclamation.

UNMAP has no independent mapping-id operand: its token selects the mapping.
The foreign-token test therefore checks that a token for mapping 1 can return
only mapping 1's frame while mapping 0's PTE stays intact. It does not invent
an extra target operand to manufacture a rejection. Ordinary REVOKE returns
into its issuing context; a domain's own object handle is not monitor authority.

## Independent checks

The checker uses ghost bookkeeping which correct instructions never consult
to authorize an operation. It checks after every successful microstep:

| Contract | Check |
|---|---|
| I1 | No live linear node has two storage locations; no exclusive physical page has overlapping usable capabilities |
| I2 | Table capabilities occur only in registry/table storage; ordinary instructions cannot load or dereference them |
| I3 | The monitor receives no logical data capability through its management handles; UNINIT remains unreadable |
| I4 | Every table entry is an admitted, linear physical frame or table capability, published by POPULATE; unused slots are clear |
| I5 | Used slots never change target or return to none; roots stay fixed; only DESTROY releases a registry reservation |
| I6 | A completed operation leaves no relevant old walk, checked access or cached translation that could act later |
| I7 | Logical binding and physical/logical kind agree with protected node identity across derivation and storage |
| Goal clauses | Physical exclusivity, correct mapping selection for each access, and no monitor observation of domain-written words |

The environment supplies only existing capabilities to instructions. Private
domain programs can pass logical authority between the two domain contexts;
they do not deliberately grant private data authority to the monitor. That is
the design's “holding no matching data authority” premise. Monitor operations
can revoke, race, withhold backing and attempt invalid operands. Refused
operations must leave the complete state unchanged.

## Searches and controls

[check.py](check.py) runs three complementary explorations:

1. Eight named contract scenarios cover permission/type refusal, sparse backing,
   scrub-before-read, explicit and implicit locked slots, normal and orphaned
   reclamation, late old-generation handles, generation exhaustion, object reuse,
   cross-context pointers, capability transfers and publication races.
2. Forty schedule families exhaust **all enabled interleavings to terminal
   states for their fixed initial workloads**: five operations (load, store,
   atomic read-modify-write, capability load and capability store), four removal
   events (frame/table/root revoke and DETACH), and cold/warm TLBs. Both harts
   have an outstanding access. Live completions as well as cancellations are
   explored; a terminal unfinished barrier is an error. UNMAP is separately
   exercised after DETACH: a pre-DETACH access must already be drained there.
3. A breadth-first lifecycle search explores its finite action alphabet through
   depth 3 from two populated mappings. Seeded traces exercise longer sequences,
   checking each microstep. They favor access progress in 60% of steps to avoid
   the many revocation operands starving walks. A random campaign with zero
   executed data accesses fails rather than becoming a vacuous pass.

These are bounded experiments, not exhaustive exploration of all possible
programs or an induction proof. The record reports the bounds and coverage.

All eight broken variants must fail at their intended property while their
correct control either completes safely or refuses the unsafe instruction:

| Variant | Witness |
|---|---|
| `plain_detach` | DELIN, then ordinary REVOKE on the detach handle yields readable logical authority (I3) |
| `uncleared_create` | A stored non-linear capability survives root conversion (I4) |
| `uncleared_populate` | A stored non-linear capability survives lower-table conversion (I4) |
| `walk_only_drain` | A checked store remains outstanding when reclamation returns (I6) |
| `weak_binding` | A pointer to mapping 1 selects mapping 0 despite both being live (redirection) |
| `duplicate_return` | Saved-root reclamation returns a second capability after senior-handle reclamation (I1) |
| `root_release` | Root revoke frees a still-owned registry entry (I5) |
| `stale_record` | Old-root cleanup keyed only by id deletes the new generation's entry (I5) |

`weak_binding` removes id selection on the data path. There is no claim that
removing only the generation comparison has been exposed independently: old
logical nodes also die at DETACH. Generation exhaustion and stale-generation
physical cleanup have their own controls. The comparison variants are confined
to this model; none changes production enforcement.

## Boundaries and next work

- Two harts/contexts, two mapping ids, three generations per id, two table levels
  with fanout two, four words per page; the standard fixture has sixteen pages.
  Addresses and bounds count words, not bytes. Physical overlaps reduce to page
  identity; sub-page physical grants and compressed bounds are excluded.
- Revocation-node identities are monotonic and never reused. This experiment
  does not qualify the RTL/QEMU node reclaimer or finite node-generation encoding.
- A memory effect, including the move/clear of a linear capability, is one
  atomic abstract event. Walks, buffered effects and revocation are interleaved,
  but bus beats, partial byte stores, misaligned/cross-page instructions, cache
  coherence, DMA, speculation and instruction fetch are not implemented.
- SHARED, PROTECT, SURRENDER, exports, paging, fork and COW remain later stages.
  Only Stage-1 whole-mapping tokens can be produced.
- The model assumes the base capability/tag machinery and trusted publication
  boundary. It does not prove their RTL realization or close R-44.

The next gate is review of these abstraction boundaries and a concrete walker,
protected binding lookup and global completion protocol against §10.3. Any
QEMU or RTL implementation must establish refinement to the modeled contract
and add the omitted access shapes before claiming them.
