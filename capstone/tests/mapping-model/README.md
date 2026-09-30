# Caplified mapping tables: executable Stage-1 contract

This is a Python standard-library reference model of
[the mapping candidate](../../docs/design/caplified-mapping-tables.md), based on
contract commit `00626a232bda`, extended by the candidate's protected CREATE
delivery, globally disjoint logical ranges and anonymous frame initialization.
It implements the experiment in §10.1, including
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
trace bounds; defaults are 6, 32 and 100. A state-limit failure is an incomplete
search and a failing exit, never a passing bounded result.
Depths below 6 explicitly disable the lifecycle completion gate in the record.
Short random traces can fail the per-seed data-coverage requirements.

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
of wrapping. Physical page identities are distinct even when offsets within
their pages are equal. Logical ranges occupy a separate, globally reserved
numeric region. CREATE checks every registry entry for overlap, including dead
roots and DETACHED mappings; DESTROY releases the range with the id. This is
protected instruction state, not the ghost reservation oracle.

CREATE takes a bootstrapped domain handle and a typed `ResumeSlot`. Preparation
and publication both check that the slot belongs to that handle's domain. A raw
wallet name, mismatched recipient or occupied slot is refused before publication.
The published capability is stored directly in the domain wallet representing
that protected slot; it never appears in a monitor wallet. The model abstracts
resume consumption as ordinary linear movement from that location. It does not
model domain creation, trap-register sealing or libc's pending-request validation.

CREATE consumes one exclusive, writable physical root page. POPULATE consumes one
exclusive writable data frame and, if necessary, one additional writable table
page. Requiring retained write permission prevents initialization from increasing
a read-only supplier's authority, even when a table page is currently UNINIT.
POPULATE zeros every data word and clears tags before linking the frame. The PTE
retains the supplied rights; the logical capability limits effective access to
the mapping's max protection. An R-only mapping can therefore enclose an RW frame
for zeroing and later UNINIT scrubbing without allowing domain stores.
Both expose
`prepare` and `publish` microsteps. Preparation holds the operands outside
software access; publication rechecks authority and atomically clears tags,
links tables and publishes the result. A concurrent revoke or DETACH can make
publication fail. The caller can abort and recover the original operands;
operands revoked in the meantime are still dead. There is one outstanding
preparation, but revocation, DETACH and ordinary accesses can interleave with
it. This abstracts internal zeroing and publication circuitry, including ordering
against earlier physical supplier accesses. Supplier stores in this model are
atomic; it does not explore their store buffers or multi-beat page clearing.

DETACH, UNMAP and REVOKE separate break, per-hart invalidation, per-hart drain
acknowledgement and final return. The default uses a conservative global barrier:
new accesses wait; old walks may progress until their hart invalidates them;
checked accesses may complete or cancel. A hart cannot acknowledge drain with
a checked access outstanding. No return occurs before both acknowledgements.
This over-invalidates compared with a targeted implementation; it makes no
performance claim. DROP of a linear logical capability uses the same barrier.

A separate `table_record` experiment represents the optional §10.3 optimization.
CREATE and POPULATE write protected `node -> (id, gen)` records for table pages.
REVOKE examines the affected live nodes before breaking them; a matching record
selects the mapping scope. Logical operations and UNMAP already know their
binding. An unindexed physical revoke (including frame revoke) falls back to
the global barrier. Issue, invalidation and drain use the same scope, so foreign
walks and checked effects can proceed during a scoped barrier, including after
that hart acknowledged drain. Returns may coexist with a foreign pending access.
This is implementation state, separate from the checker’s ghost table ownership.
Records persist in this finite experiment; no record reclamation, node reuse or
hardware lookup cost is modeled. The default global mode never consults them.

DESTROY consumes a whole-mapping token, requires completed DETACH, and releases
only the registry entry. Frames and lower tables can remain present. Their
senior physical handles work after root loss, after DESTROY and after a new
generation takes the id. Root revoke leaves a reserved registry entry holding
a dead root. No reverse index is used to remove registry entries; even a scoped
old-generation revoke leaves a new generation's registration intact.

Logical SPLIT, DELIN, MREV, REVOKE, DROP, bounds/rights attenuation and cursor
movement are modeled. Moving and storing a linear capability removes the source;
non-linear capabilities can be copied. A linear capability load additionally
needs write permission because it clears its memory source. UNINIT scrubbing
requires retained W and advances one word at a time; INIT requires the cursor at
the end. Revoking a read-only object does not create write authority. Logical
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
| I4 | Every table entry is an admitted, linear physical frame or table capability, published by POPULATE; unused slots are clear; a ghost snapshot records all anonymous frame words and tags as clear at admission |
| I5 | Used slots never change target or return to none; roots stay fixed; only DESTROY releases a registry reservation |
| I6 | A completed operation leaves no relevant old walk, checked access or cached translation that could act later |
| I7 | Logical binding and physical/logical kind agree with protected node identity across derivation and storage |
| Goal clauses | Physical exclusivity, correct mapping selection, no monitor observation of domain-written words, and globally disjoint reserved logical ranges above physical addresses without wrap |

The observation oracle records **physical** monitor reads. Logical disclosure is
caught earlier by I3, at acquisition of a live logical capability by the monitor,
before it can issue a read. The observation oracle alone is not a check of both
read paths.

The environment supplies only existing capabilities to instructions. Private
domain programs can pass logical authority between the two domain contexts;
they do not deliberately grant private data authority to the monitor. That is
the design's “holding no matching data authority” premise. Monitor operations
can revoke, race, withhold backing and attempt invalid operands. Refused
operations must leave the complete state unchanged.

## Searches and controls

[check.py](check.py) runs these complementary checks:

1. Twelve named contract scenarios cover permission/type refusal, sparse backing,
   scrub-before-read, explicit and implicit locked slots, normal and orphaned
   reclamation, late old-generation handles, generation exhaustion, object reuse,
   cross-context pointers, capability transfers and publication races. The four
   additional scenarios check typed recipient delivery and loss of its authority,
   numeric address separation and reservation lifetime across domains, and dirty
   anonymous frames with both supplier data and stored capabilities. Address
   comparisons use the numeric cursor projection, including aliases with different
   bounds/rights and one-past equality at a SPLIT boundary. Geometry and recipient
   negative tests invoke the instruction with absolute operands directly. The
   fourth applies the range rule, kind classification, grain rule and binding
   word to the architectural constants of the encoding decision; it walks nothing.
2. Forty schedule families exhaust **all enabled interleavings to terminal
   states for their fixed initial workloads**: five operations (load, store,
   atomic read-modify-write, capability load and capability store), four removal
   events (frame/table/root revoke and DETACH), and cold/warm TLBs. Both harts
   have an outstanding access. Live completions as well as cancellations are
   explored; a terminal unfinished barrier is an error. Each workload has one
   removal event and one initially outstanding access per hart. UNMAP is separately
   exercised after DETACH: a pre-DETACH access must already be drained there.
3. The same forty workloads run again with `table_record`. Fifteen additional
   workloads combine each of the five data operations with table/root revoke or
   DETACH, followed by a second table/root revoke. One victim store is outstanding
   on hart 0; hart 1 issues one access to mapping 1 after the first break, at any
   enabled point during/between/after the two barriers. Every terminal must have
   completed both removals and resolved both accesses. Each family must reach
   foreign issue during a barrier, issue after that hart's drain, foreign memory
   completion during a barrier, and return with foreign work pending. These are
   edge-coverage counts, not counts of distinct executions. Direct controls check
   global refusal, affected-mapping refusal, preserved foreign TLB entries,
   global frame fallback, and old-generation record isolation. An intentionally
   misattributed table record must fail I6 with a victim store still outstanding.
   The late capability-load workload has a warm foreign TLB from its payload
   store during setup; the other late-access workloads start with it cold.
4. Three breadth-first lifecycle searches start in ACTIVE, DETACHED and DESTROYED
   states, each reached by explicit setup instructions from two populated mappings.
   Their depth-6 bounds exclude setup. The alphabet uses one representative
   frame/table/root/logical revocation operand, mapping 0 for lifecycle changes,
   one store at offset zero, two POPULATE shapes, CREATE, DESTROY and UNMAP.
   It permits at most one additional barrier and one additional access per path;
   preparation, walking, invalidation, drain and return remain separate microsteps.
   The gate requires completed revoke, detach and unmap, a memory effect, destroy
   and publication across the searches. Counts include the depth frontier, whose
   successors are not explored. This is neither a full lifecycle from one start
   nor exhaustive exploration of the broader random-action alphabet.
5. Seeded 100-step traces select weighted opcode groups, then shuffle operands
   within each group. REVOKE is capped at two per trace; REVOKE/DETACH/DROP together
   at three. Accesses use offset zero. Mapping 1 and its backing stay live as a
   data-control workload, while mapping 0 can be dismantled and recreated.
   Every seed must complete at least four memory effects including a load and a
   store. The record includes each seed's operation and effect counts, not just
   aggregate nonzero coverage. The policy is constrained random testing, not
   uniform adversarial sampling or a claim about arbitrary revocation rates.

These are bounded experiments, not exhaustive exploration of all possible
programs or an induction proof. The record reports the bounds and coverage.

The checked default record contains:

| Search | Families | Sum of states |
|---|---:|---:|
| Global removal interleavings | 40 | 11,022 |
| Same workloads with table records | 40 | 11,022 |
| Late foreign issue and two removals | 15 | 21,683 |
| Reduced lifecycle, depth 6 | 3 | 1,343 |

State totals sum separate families, not one deduplicated state space. The random
campaign completes 581 memory effects in 3,200 steps, at least eight per seed;
all 32 seeds include loads, stores and REVOKE, with at most two REVOKEs each.

The earlier record at `de39f471` remains valid for its narrower workloads. Its
depth-3 lifecycle search checked only prefixes (zero finish, memory step,
DESTROY and UNMAP); 20 of 32 random seeds had no memory effect, with 24 total.
The independent review also found depth 4 exceeded 100,000 states with the old
alphabet. Those results do not supply lifecycle-completion or foreign-issue
coverage; the new checks address those specific gaps with explicit restrictions.
Earlier records also did not establish protected CREATE delivery, globally
distinct pointer addresses or zero-filled anonymous frames. Their passing
workloads remain evidence for those workloads, not for the added contracts.

All ten broken variants must fail at their intended property while their
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
| `monitor_delivery` | CREATE accepts a monitor wallet as destination and exposes readable logical authority (I3); the control refuses inside CREATE |
| `unzeroed_frame` | POPULATE publishes supplier value 4242 and a stored capability in an anonymous frame (I4); the control clears both before publication |

`weak_binding` removes id selection on the data path. There is no claim that
removing only the generation comparison has been exposed independently: old
logical nodes also die at DETACH. Generation exhaustion and stale-generation
physical cleanup have their own controls. The comparison variants are confined
to this model; none changes production enforcement.

## Boundaries and next work

- Two harts/contexts, two mapping ids, three generations per id, two table levels
  with fanout two, four words per page; the standard fixture has sixteen pages.
  Addresses and bounds count words, not bytes. Physical overlaps reduce to page
  identity; sub-page physical grants and compressed bounds are excluded. Physical
  pages have numeric addresses `[4, 68)`; logical addresses start at 128 and remain
  below 256, including one-past cursors. Each logical range fits an aligned
  sixteen-word window. Fixtures choose bases 128 and 160, and low address bits
  index their respective tables. The harness's `offset_action` converts convenient
  relative test offsets to absolute operands before execution; traces record the
  latter. It may inspect ghost setup history, but no instruction consults that
  history or the adapter. The architectural partition is decided in the
  [encoding decision](../../docs/design/caplified-mapping-encoding-decision.md);
  `check_architectural_partition` applies the model's range rule, kind
  classification, compressed-bounds grain rule and 32-bit binding word to those
  constants (physical below `2^56`, logical in `[2^57, 2^63)`, 4 KiB pages, 12-bit
  id, 20-bit generation), and the record's `partition` field repeats them. The
  walked geometry stays finite, and the 128-bit encoding at those addresses
  remains unqualified on QEMU, whose side table keeps fat bounds, and on RTL.
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
protected binding lookup and completion protocol against §10.3. The record
experiment additionally needs a bounded record lifetime and a node-reuse protocol.
The global prototype remains the default. Any
QEMU or RTL implementation must establish refinement to the modeled contract
and add the omitted access shapes before claiming them.
