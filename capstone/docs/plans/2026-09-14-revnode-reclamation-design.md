# Revocation-node reclamation: a design, its safety invariant, and the two storage figures H1/M3 asked for

**RTL lane, 2026-09-14.** Requested by the board lane on the lead's decision, for the Sublet paper's
M1 study. Two asks: a reclamation design with its invariant stated **before** any RTL and audited
against the security model; and the node/tag storage figures the H1 and M3 studies need from the RTL
**as flashed** (`1bfff7776`).

**Nothing here is implemented.** This is the proposal, per the project rule that a substantial new
direction gets a committed doc for review first.

---

## Part 2 first, because it is measured rather than proposed

All figures read from the flashed revision. Where two independent constants had to agree, they are
checked against each other rather than quoted separately.

### Node format and size

    struct rev_node_t { depth : logic[32], prev : logic[30], next : logic[30],
                        valid : logic[1],  linear : logic[1] }          94 bits used

**Bytes per node: 16.** Not from the struct width — from the address arithmetic, which is what
actually determines the footprint (`ex_stage.sv`):

    node_query_full_addr = CAP_REVNODE_MEM_BASE + {22'd0, node_query_addr, 4'd0}

The `4'd0` shifts the index left by four, so each node occupies a **128-bit slot**. 94 bits are used
and **34 bits (27%) are padding**. That padding is where the generation field below lands for free.

### The pool, and the whole memory map it sits in

| region | range | size | |
|---|---|---:|---|
| data | `0x80000000`–`0xBC2D2D2C` | 962.82 MiB | |
| tag shadow | `0xBC2D2D2D`–`0xBFEFFFFF` | 60.18 MiB | exactly `ceil(data/16)` |
| node pool | `0xBFF00000`–`0xBFFFFFFF` | 1.00 MiB | 65,536 slots × 16 B |
| | | **1024.00 MiB** | the whole DDR3 |

**The map is exact, and that is a check rather than a coincidence.** The tag shadow is precisely
`ceil(962.82 MiB / 16)` and it abuts the node pool with no gap. Confirmed numerically; a floor
instead of a ceiling is off by one byte, which is the partial final granule.

**Mintable ids: 65,532.** The head is 16 bits, resets to 3, sentinel 65,535, `head++` at both mint
sites. So ids 3..65,534 are reachable and four slots are not.

**The id field is 30 bits wide and only its low 16 are reachable** — `node_id := #{14'd0, *head}`
puts fourteen hard zeros above the head. That matters for Part 1.

### Tag storage attributable to the configuration

**On-chip: 256 bytes.** There is no tag-cache module. The dedicated store is `cap_tag_q` in
`wt_dcache_mem.sv`, a **one-bit** array indexed `[NUM_WORDS][SET_ASSOC]`. For this core's 32 KiB
8-way D-cache with 128-bit lines that is 256 × 8 = **2,048 bits**, i.e. **0.78% of the data array**,
and it is exactly 1 bit per 16-byte line — the minimal encoding.

*Do not attribute the 64-bit `user` path to the capability configuration.* `DCACHE_USER_WIDTH` is
`AxiUserWidth` and is inherited CVA6 machinery that exists upstream; the tag rides it, but it is not
storage this configuration adds.

**External: 60.18 MiB, and it is 8× larger than it needs to be.** The shadow is addressed
`CAP_TAG_MEM_BASE + (data_paddr >> 4)` — **one whole byte per 16-byte granule**, where one bit
carries the meaning. So the external tag cost is **6.25% of DRAM**; a bit-packed shadow would be
**7.52 MiB, 0.78%** — the same ratio the on-chip array already achieves. **That 5.47-point gap is a
result in its own right for M3** and is independent of anything M1 does.

---

## Part 1: the design

### Why the obvious option is unsafe, stated first

**A free list is not safe here and should be ruled out explicitly rather than passed over.** Push a
revoked node's index onto a list, pop it on the next mint, and a capability still carrying that index
is **indistinguishable from a freshly minted one**. Capabilities are copied freely into memory, so
"prove no stale alias exists" is not a property this system can establish. A free list trades the
stall for silent authority resurrection, which is strictly worse than the stall.

### The proposal: generation-tagged reuse, and it costs nothing in the capability format

Split the existing 30-bit `revnode_id` into the index that is already there and a generation in the
fourteen bits that are **already hard-wired to zero**:

    revnode_id[15:0]   index       -- as today, 65,532 mintable
    revnode_id[29:16]  generation  -- currently the literal 14'd0 in `#{14'd0, *head}`

and spend 14 of the node slot's 34 padding bits on the node's current generation.

**So neither the capability format nor the node slot grows.** `fat_cap_t` is untouched, `bounds_t` is
untouched, the 16-byte slot is untouched, and the AXI address arithmetic is untouched. The change is
confined to the node unit and the comparison site.

### The safety invariant

> **A capability whose revocation reference is `(i, g)` confers authority only if
> `node[i].generation == g` **and** `node[i].valid`. Reclaiming index `i` increments
> `node[i].generation`, and that single increment simultaneously invalidates every alias of every
> generation before it.**

The validity query becomes a two-field comparison instead of a one-bit read. That is the whole
mechanism: a stale alias does not need to be found, because it fails the comparison wherever it is.

### When a node becomes reclaimable

> **Index `i` is reclaimable when `node[i].valid == 0` and no node in its subtree is valid** — the
> subtree being the following nodes with greater `depth`, which is the same walk `REVOKE_NODE`
> already performs against `depth_bound`.

Reclaiming an ancestor while a valid descendant remains would strand the descendant's authority under
a reused id, so the subtree condition is not an optimisation, it is part of the invariant.

### The wrap hazard, and the one design decision that closes it unconditionally

A 14-bit generation wraps after 16,384 reclaims of the same index. On wrap, an alias that old would
match again. **Do not paper over this with "wrap is unlikely".**

> **Saturating retirement: when `node[i].generation` reaches its maximum, index `i` is never
> reclaimed again.** The index is retired, the pool shrinks by one, and the invariant holds
> unconditionally rather than probabilistically.

**Capacity, for the paper's table.** Today: 65,532 node lifetimes, then a deliberate stall. Under
this proposal: 65,532 × 16,384 ≈ **1.07 × 10⁹ lifetimes** before the first index retires, and the
pool degrades gracefully rather than stalling. For scale, the size-1 Sublet speedtest1 run that mints
43,355 nodes today would need ~24,700 such runs to retire a single index.

### What the audit must attack, named in advance

Per the rule that an ISA change is audited against the security model before it is built, and naming
the weakest link rather than asking for a general review:

1. **The comparison site is the whole security boundary.** Every path that today reads
   `node.valid` must read both fields. **Find one that does not.** A single site left comparing
   validity alone reinstates exactly the free-list hazard, and it would pass every functional test.
2. **Reclaim versus an in-flight query.** R-27 already showed this unit can have a response in flight
   across a flush. If an increment lands between a query's read and its use, does the consumer act on
   a generation that no longer exists?
3. **The subtree condition under a partial revoke.** `REVOKE_NODE` walks and stops at `depth_bound`.
   Can an index be judged reclaimable while a valid descendant exists outside the walked range?
4. **Retirement as a denial-of-service surface.** Can an unprivileged domain drive one index's
   generation to saturation deliberately, and does repeated retirement degrade the pool faster than
   the stall it replaces?

### What this does not address

Node **allocation** is still `head++` within the index space; this proposal reuses indices, it does
not change how a fresh one is chosen. And it does not reduce the 1 MiB pool or the tag shadow — the
8× external tag finding above is a separate lever, and a better one for M3.

---

## Two findings from 2026-09-15 that change what this design claims, before any RTL exists

### It does NOT flatten the revoke cost curve, and the plan must not say it does

`apollo-board` measured per-allocation release cost growing **~12x within a domain**
(`give_cyc/n` 176 -> 2115) while minting grows only ~1.5x, and located the mechanism, audited
SUPPORTED: `core/anvil_build/capstone_rev_node.anvil:13-34`, where `REVOKE_NODE` re-enters its own
FSM once per visited node, terminates only on `node_in.depth <= *depth_bound`, and **revoked nodes are
never spliced out of the chain** — so round *r* performs *r+2* dependent 16-byte node reads. The
`.anvil` source is byte-identical between the flashed revision and dev's pin, so this is measurable
today with no synthesis.

**This design does not splice.** Generation-tagged reuse makes reuse SAFE — a stale alias fails the
two-field comparison wherever it lies — and it lifts the 65,532 ceiling to ~1.07e9 lifetimes. Neither
of those shortens the walk. So:

> **Splicing fixes the COST. Generation-tagging fixes the CAPACITY and the SAFETY of reuse. They are
> different changes, and this document is only the second one.**

Anything that reads "the reclaimer will flatten the release curve" does not follow from what is
written here. The decision — splice only, generation-tag only, or both in one change — belongs to the
lead and should be made before the RTL, not discovered after the measurement disagrees with the
prediction. The paper lane's independent caveat, that a reclaimer may *complicate* P1 rather than
improve it, is the same worry approached from the other side.

### The generation field lands in the HIGH half of the granule, which is this core's worst neighbourhood

Raised by the paper lane and **measured here rather than assumed**. `rev_node_t` is
`depth:32, prev:30, next:30, valid:1, linear:1` (`capstone_unit.anvilh:543-549`), and the generated
`capstone_rev_node.anvil.sv` packs it as a **`[93:0]`** slice. So the 94 used bits occupy bits 0..93
and the 34 "free" bits are **94..127 — every one of them above bit 64**. A generation placed in the
padding therefore lives wholly in the high 64-bit word of the 16-byte granule.

That is the half R-29, S-06, S-10 and R-10 are all about — R-29 being a plain 8-byte store into a
granule's high word followed by a 128-bit load returning the high half **zeroed**, still OPEN. **A
corrupted generation is not a benign wrong number**: it is either a false match, in which case a stale
alias confers authority, or a false mismatch. That is the entire safety property of this design.

**The exposure is NOT established and must not be assumed either way.** R-29's trigger is `sd` then
`ldc` through the LSU and the write-buffer/refill path; the node slot is reached through the rev_node
unit's own memory endpoint (`node_query_full_addr = CAP_REVNODE_MEM_BASE + {22'd0, node_query_addr,
4'd0}`) and the pool is hardware-managed, not written by software stores. The shape matches; the path
may not. **What would settle it** is a source read: do rev_node's accesses share the write-buffer
overlay and refill leg the R-29 mechanism sits in (`wt_dcache_mem.sv:354-358`, overlay gated at word
granularity at `:283/:335/:397`), or do they bypass the D-cache?

**And if it does reach, there is a cheap alternative that should be on the table before the placement
is baked in.** The free bits are all high only because the used fields happen to total 94.
`depth : logic[32]` is far wider than any revocation tree needs; narrowing it to ~18 bits frees 14
bits **inside the low half**, and the generation could live there instead, entirely below bit 64. That
changes `rev_node_t` and the packing, so it is a design decision rather than a tweak — but it would
make the safety property independent of the high-half defect family rather than contingent on it.

## Sequencing

The figures in Part 2 are available now and unblock H1/M3 without any of Part 1. Part 1 needs the
audit above before RTL, then the lint gate, then synthesis, and any bitstream is the lead's
ask-first decision. None of that is started.
