# Revocation-node reclamation v2: a specification with a mechanism

**RTL lane, 2026-09-16.** For the M1 study. **Nothing here is implemented, and nothing may be
scheduled from it** — M1's start gate requires the lead to name the runtime/RTL owner and approve the
algorithm and the stale-reference invariant first. This document exists to be the material that
approval acts on.

**This replaces `2026-09-14-revnode-reclamation-design.md` (v1), which was audited twice and rejected.**
The reconciled verdict was *"INCOMPLETE, not merely unsafe — it specifies the safety property of reuse
and the condition for reclaimability, and never specifies the mechanism that reuses"*. The direction
survives: the 65,532-lifetime ceiling is real and generation tagging lifts it. What follows supplies
the missing mechanism and answers each item the reconciliation named.

> # ⚠ AUDITED 2026-09-16 — THIS DOES NOT SHIP AS WRITTEN EITHER. Read this box first.
>
> The audit this document asked for has run, against this text and not a draft. **v1 was rejected as
> incomplete — having no mechanism. v2 has mechanisms, and three of them are individually wrong in
> ways that would have shipped.** All are fixable and none refutes the direction; the corrections are
> folded in below and flagged **[AUDIT]**. Nothing here is approved.
>
> **1. FATAL AS WRITTEN — §6's mechanism relies on an event that does not occur.** I claimed the write
> clearing `valid` broadcasts the index at reclaim. It does not: that write happens at **invalidation**
> (`capstone_rev_node.anvil:50`, DROP at `:93`), and under §1–§3 as written **nothing writes `node[i]`
> at reclaim at all**, so no broadcast carries index `i` there. The residual is also misdated — the
> window opens at invalidation, not recycle, and stays open until the entry is rewritten. Worse, the
> adopt guard is **per-CPMP-entry**, so clearing entry *k* does nothing for the same stale reference
> later installed into entry *m*. **Fix: make the generation increment itself a write to `node[i]` with
> `valid = 0` at reclaim.** That makes §1's push two node writes, not one.
>
> **2. FATAL AS WRITTEN — the allocator never composes the new id, so the scheme is inert.** §1's pop
> takes a slot off the list and stops. `capstone_rev_node.anvil:107` and `:151` still emit
> `#{14'd0, *head}`, so a recycled slot is handed out at **generation 0** and every §3 check passes
> vacuously. The pop must return `(node[i].generation, i)`.
>
> **3. FATAL IF IMPLEMENTED AS DESCRIBED — §2's `free` bit must not go in the shared helper.**
> `change_rev_node_validity` (`capstone_unit.anvilh:552-554`) is called by **both** the revoke walk
> (`:50`) and DROP (`:93`). Setting `free` inside it marks every dropped node — which is `valid == 0`
> and **still linked** — as reclaimable, producing exactly the free-list/tree alias §2 exists to
> prevent. It must be set at the walk's call site, or passed as a parameter. The most natural reading
> of §2's own sentence is the fatal one.
>
> **4. CONFIRMED — `depth` is not tree depth, and reclamation unbounds it.** INIT copies the parent's
> depth (`:112`); only MREV increments, and it increments *the parent* (`:166`). So depth counts
> **MREVs on one node**, bounded today only by the 65,532 mint ceiling. Reclamation removes that bound,
> so a 17-bit depth wraps at 131,072 MREVs on a single node, and `if(node_in.depth <= *depth_bound)`
> (`:18`) then terminates the walk immediately and **silently under-revokes**. §3's "tree depth cannot
> exceed the pool size" was the wrong mechanism for the right-looking answer.
>
> **5. CONFIRMED — `next` is NOT dead on an invalid node.** The walk's third branch (`:58-61`) follows
> `next` on a node with `valid == 0`. §1's "its `prev` and `next` carry nothing" is false in general;
> it holds only *because* the node is unlinked, which puts the entire design on §2's `free` bit.
>
> **6. WAS UNRESOLVED — NOW SETTLED BY MEASUREMENT.** The audit's sharpest item: the generation's home
> is `data_ruser[29:16]` = `depth[31:18]`, structurally always zero today, so the **wire** was verified
> in both directions but the **round trip** never had been by any existing test. **Run: a node written
> with every depth bit set reads back `4294967295` — all 32 bits survive.** (Probe installed in a
> scratch tree at `INIT_STAGE`, using the read-back that `check_dummy_node` already performs and its
> commented-out print; tree restored afterwards and proven restored by reproducing a known ladder point
> to the cycle.) **Narrow residual, stated rather than closed:** that is the L1-resident path. The
> refill path is proven only for the low bits, which today's small depths exercise every time a big-N
> ladder evicts and re-reads. **`ruser[29:16]` after an eviction remains unexercised.**
>
> *Incidental, and it costs an hour if rediscovered:* the Anvil front end will not parse a literal
> ≥ 2³¹ — `32'd4294967295` is a syntax error reported against the enclosing `let`, not the literal.
> Use the concatenation idiom the file already uses: `#{2'd3,30'd1073741823}`.
>
> **What survived the audit unchanged:** the MSB-first packing; both directions of `depth`'s plumbing;
> the id-as-address census; that masking is a no-op today so nothing depends on the full 30 bits; §2's
> `free`-bit *consistency* argument, attacked three ways (flush cannot abort a walk mid-run, a mint
> cannot interleave, the exit splice covers everything the walk marked); the R-29 full-granule
> argument; and the CAPNODE decode finding, verified independently.

**Part B (graceful exhaustion) is separable and can be approved alone.** It needs no generation
tagging, no allocator and no change to the id format, and the reclaimer needs it regardless.

---

## 0. Ground truth, verified at primary source for this document

Every row was read directly, not inherited from v1 or from an audit summary. v1's central measurement
error was reading a *width* and inferring field *order*; these were checked against generated RTL.

| Fact | Evidence |
|---|---|
| Node record 94 bits, **MSB-first**: `depth[93:62] \| prev[61:32] \| next[31:2] \| valid[1] \| linear[0]` | `capstone_unit.anvilh:532-538`, confirmed twice in generated SV |
| `revnode_id` is **30 bits at every site** — capability, all six channels, all trackers, the broadcast | `ariane_pkg.sv:634`; compressed metadata is 30+3+3+28 = 64, **zero slack** |
| `head` is **16 bits**; the allocator only ever emits `#{14'd0, *head}` | `capstone_rev_node.anvil:218`, `:107`, `:151` |
| ⇒ **`id[29:16]` is permanently zero. Fourteen generation bits are free with no width change anywhere.** | corollary |
| **The id IS the address**: `addr = 0xBFF00000 + (id << 4)`, all 30 bits | `ex_stage.sv:1159-1161` |
| ⇒ generation 1 addresses 1 MiB past base; a saturated generation ~16 GiB past a **1 GiB** DRAM | `calculate_memory.py:21-23`, `ariane_pkg.sv:591` |
| The slot's 34 spare bits are **not plumbed**: the read path drops them and bit 94 is a hard constant 1 | `ex_stage.sv:1149` (`data_ruser[29:0]`), `:1163` (`{34'd1, …}`) |
| **The S/U boundary caches a validity bit, defaults an unseen id to VALID, and never reads a node** | `pmp_data_if.sv:82-99` — the comment says *"assume the revnode is live until proven otherwise"*; used `:132`, gated `ld_st_priv_lvl_i != PRIV_LVL_M` at `:288` |
| The other two trackers are M-mode only — **[AUDIT] and not the same shape**: `pmp_data_if` and the LSU compare the broadcast against the **registered** id, `commit_stage` against the **post-adopt** `_d`, so a broadcast can clear an id adopted in the same cycle there | `load_store_unit.sv:944`, `commit_stage.sv:244` |
| The broadcast is an **address-bus tap** — any node write with `valid` clear, sentinel maintenance included | `ex_stage.sv:1207-1208` |
| The four response channels are **1 bit**: they cannot express "no" | `capstone_unit.anvilh:500-508` |
| `CAPNODE` copies `rs1`'s id onto `rs2` with no validity check, and **its decode arm is not privilege- or capmode-gated** | `capstone_flu_unit.anvil:398-404`, `capstone_unit.anvilh:444-447`, `decoder.sv:1111` (`OpcodeCustom3`, no gate in any enclosing scope) |

---

## 1. The allocator — an intrusive free list threaded through `next`

v1 ruled a free list out and supplied nothing, which is why it was vacuous. The free list is the
missing half of the proposal, not a rejected alternative. What makes it safe is the discriminator
(§3), not the absence of a list.

**Mechanism.** Thread the free list through `next`. **[AUDIT] `next` is NOT dead on an invalid node** —
the walk's third branch (`capstone_rev_node.anvil:58-61`) follows `next` on a node with `valid == 0`.
Reusing it is safe *only because* the slot is unlinked, never because it is invalid, so the whole design
rests on §2's `free` bit and on nothing else.

**[AUDIT] The pop must compose the new id, and the first draft did not say so.** `:107` and `:151` emit
`#{14'd0, *head}`; a pop that returns a bare index hands out a recycled slot at **generation 0** and
every §3 check passes vacuously — the scheme would be inert while looking complete. **Pop returns
`(node[i].generation, i)`.** Allocation prefers the list and falls back to
bumping `head`; the pool is exhausted only when both are empty. **Zero new storage** — no slot growth,
no capability change, no plumbing change.

**Two forms, and the choice is a real trade the lead should make rather than one this document should
bury.**

**(a) LIFO, and the push is FREE.** One register, `free_head`. The revoke walk already writes every
node it invalidates (§2), so that same write can also set `next := free_head` while the register takes
the node's index — **no extra memory traffic at all**. One care point: the walk currently advances via
the *written* node (`set revoke_index := *temp_revnode.next`), so it must instead advance via the value
already in hand (`node_in.next`) before `next` is overwritten. Pop is one read.

**(b) FIFO, and the push costs a write.** Two registers, `free_head` and `free_tail`; push writes the
old tail's `next := i`. One extra node write per reclaimed node, and — see below — one extra broadcast.

**What the choice buys.** Under LIFO a mint/revoke loop gets the same index back every time, so **the
first index retires after 16,384 reclaims of that one index**, which a single 43,355-mint Sublet run
reaches. Under FIFO reuse spreads and the first retirement is far away. **This is not a capacity
difference** — retiring the *whole* pool costs ~1.07 × 10⁹ mints under either policy, against 65,532
for today's stall — it is a difference in when the first index is lost, and therefore in what capacity
figure the paper can state. **Recommendation: FIFO**, on the grounds that the figure is the point of
the study; but LIFO is free and the lead may reasonably prefer it.

**A cost I claimed here and then withdrew — recorded because the reasoning is worth keeping.** The
rev-node unit is a single FSM (`capstone_rev_node.anvil:248-257`) and `mrev_req` is received only in
IDLE_STAGE at stage 1 (`:139`), so while a revoke walks the unit services no mints. I argued from that
structure that work added *inside* the walk pays twice — its own cycles, plus a longer window in which
concurrent mints stall — and cited a board-lane anomaly as evidence.

**Both halves were wrong.** The board lane refuted the prediction from data already in hand: minting
cost is flat at 129–131 cycles across a **12× range of walk length** (254 → 3,072 nodes) and departs
only in the smallest-arena geometry, so it tracks slot count rather than walk length. And the deeper
reason is structural: **`capstone_dyn_unit` has a single dispatch loop** (`:533`) that receives one
request and runs it to completion, and REVOKE holds that loop blocked on
`recv rev_node_ep.rev_res` (`:59-61`) for the entire walk. MREV is the same loop (`:115-117`). **A mint
therefore cannot be in flight while a revoke walks, so the starvation window does not exist.**

**What that leaves, and it is better news for the design than what I claimed:** the rev-node FSM is
never contended from software, so work added inside the walk costs **only its own cycles**. FIFO's
extra write is a write, not a write plus a stall.

**[AUDIT] An internal contradiction between §1 and §2 that must be resolved before costing anything.**
§1's FIFO push is a write **per reclaimed node**, while §2, two paragraphs later, refuses exactly that
cost for the adjacent branch of the same walk ("that would add a write per dead node to every revoke
and undo W1"). W1's entire value is that unlinking is O(1) in run length; a per-node push restores
O(n) and must be costed against the ~12× release-cost growth W1 was built to remove. **An O(1) bulk
alternative exists and is not yet specified:** the marked run is already threaded by its own `next`
pointers, so `node[free_tail].next := serving_next` with `free_tail := <last marked index>` splices the
whole run onto the list in **one** write. It drags in the branch-3 nodes, which carry `free == 0`, so
the push guard and the deliberate-leak paragraph both need rewriting to adopt it.

**A side effect of (b) that must be recorded.** The free-list push is a node write with `valid` clear,
and `ex_stage.sv:1207-1208` turns *any* such write into an invalidation broadcast. So every FIFO push
emits a spurious broadcast for the old tail's index. It is **semantically harmless** — that index is
genuinely free, so clearing any tracker holding it is correct — but it is unintended, it scales with
the reclaim rate, and under §6's index-compare it reaches all three trackers. LIFO does not have this
problem, because its push rides inside a write the walk was already making.

**One hazard to state, because it is the shape of R-12's original hang.** A slot pushed twice would
make the free list circular and a pop would return a live node. Every push must be guarded by the same
condition that authorises the reclaim, and the guard must be provable from the node itself, never from
the caller's assertion. §2 shows that `valid == 0` alone cannot establish this and supplies the bit that
can: the push guard is **`valid == 0 && free == 1`**.

## 2. Reuse eligibility is UNLINKED, not merely invalid

Invalidation preserves `prev`/`next` (`capstone_unit.anvilh:552-554` — **[AUDIT]** the first draft cited `:558-562`, which are the `next`/`prev` helpers, not the validity one), so an invalid node stays in its
old parent's chain. A reissued index that is still linked corrupts the doubly-linked list that
`REVOKE_NODE`'s depth walk and MREV's pointer surgery both depend on: a later unrelated REVOKE either
revokes a fresh capability that merely inherited the index, or terminates its walk early and silently
under-revokes.

**W1's deferred splice (`capstone_rev_node.anvil:26-42`, landed) discharges this for every node a later
revoke crosses** — it unlinks the whole revoked run in two writes. It does **not** cover a dropped node
that no revoke ever crosses. So:

> **Eligibility rule: a slot may enter the free list only when it is `valid == 0` AND unlinked.** DROP
> (`capstone_rev_node.anvil:90-101`) clears `valid` and does not unlink, so a dropped node is **not**
> eligible until a revoke sweeps it. Making DROP eligible requires giving DROP its own unlink — two
> reads and two writes on every DROP, which the W1 analysis deliberately declined. That trade is open.

**But that rule is not decidable from the node record, and saying so was the first thing this spec got
wrong.** W1's deferred splice unlinks a run by rewriting the two **boundary** nodes
(`capstone_rev_node.anvil:26-42`); it never touches the spliced-out nodes themselves, which is exactly
what makes it O(1) instead of O(n). So a spliced-out node still carries its **old** `prev`/`next`, and
is indistinguishable from a linked one. **There is no bit that means "unlinked".** Clearing each node's
links would reintroduce the per-node cost W1 was built to remove.

**Fix, and it is free.** The walk's invalidate-and-advance branch (`capstone_rev_node.anvil:50-57`)
**already writes every node it invalidates**, via `change_rev_node_validity` and `send_revnode_update`.
Set a dedicated `free` bit in that same write:

> **`free` is set by, and only by, the revoke walk, in the write that clears `valid`. Zero extra memory
> traffic. A slot is eligible for the free list iff `valid == 0 && free == 1`.**

**[AUDIT] Set it at the WALK'S CALL SITE, never inside the helper — the natural reading of the sentence
above is the fatal one.** `change_rev_node_validity` (`capstone_unit.anvilh:552-554`) is shared by the
walk (`capstone_rev_node.anvil:50`) and by DROP (`:93`). Putting `free` in the helper marks every
dropped node, which is `valid == 0` and **still linked**, as reclaimable — producing precisely the
free-list/tree alias this section exists to prevent. Pass it as a parameter or set it at the call site.

Consistency holds at revoke completion: every node the walk marks is unlinked by the exit splice at the
end of that same revoke, and a walk that invalidates nothing splices nothing.

**The deliberate leak, stated:** the walk's *third* branch (already-invalid, just advance) performs no
write, so nodes dropped earlier are unlinked but never marked `free`, and are never reclaimed. This is
the same safe leak as the DROP case above — the slot is lost, not corrupted. **Do not "fix" it by
writing those nodes**; that would add a write per dead node to every revoke and undo W1.

## 3. The discriminator, and where it is stored

**Split the 30-bit id: `id[15:0]` index, `id[29:16]` generation.** Those fourteen bits are already
hard zero everywhere, so no field anywhere widens.

**Store the node's current generation in the top bits of `depth`, narrowing `depth` from 32 bits to
17** (the freed 32 bits become `generation:14 | free:1 | depth:17`; the `free` bit is §2's). This is the
one place v2 differs materially from v1, and it matters:

* v1 proposed spending the slot's padding. The audits showed the padding is **not plumbed** — the read
  path drops slot bits 94+ entirely — so v1's storage did not exist without widening `ex_stage.sv:1149`
  and `:1163`.
* `depth` is **already plumbed end to end**, in both directions, inside the 94 bits the unit reads and
  writes. Putting the generation at `[93:80]` and leaving `depth` at `[79:62]` costs **no plumbing
  change at all**.
* **[AUDIT] `depth` is NOT tree depth, and the obvious bound does not hold.** INIT copies the parent's
  depth (`capstone_rev_node.anvil:112`); only MREV increments, and it increments *the parent* (`:166`).
  So depth counts **MREVs performed on one node**. Today that is bounded by the 65,532 mint ceiling —
  **which is exactly what reclamation removes.** A 17-bit depth wraps after 131,072 MREVs on a single
  surviving node, and `if(node_in.depth <= *depth_bound)` (`:18`) then terminates the walk at the first
  node and **silently under-revokes**. The spec must state a per-node MREV bound with a saturate-and-
  refuse policy, which pulls in Part B again.

**Final field budget for the 32 bits `depth` occupies today: `free:1 | generation:14 | depth:17`.**
**[AUDIT] The order matters and the first draft had it wrong.** With `free` immediately above depth's
MSB, a carry out of depth sets `free = 1` on a **live, linked, valid** node — the fatal alias, reached
by arithmetic. Nothing safety-critical may sit adjacent to depth's MSB.
Seventeen bits of depth is 131,072 against a 65,536 pool — sufficient with 2× margin.

**The sites that must follow the narrowing are enumerable, and this is the whole list** —
`capstone_unit.anvilh:532-538` (the struct) and `:553`, `:557`, `:561` — **[AUDIT] these three helpers
rebuild the record field-by-field, so they must gain the new fields or silently drop them.** Anvil has a
record-update form, used at `capstone_dyn_unit.anvil:298`: writing them as `rev_node_t::{node with
valid = 1'd0}` carries new fields automatically and makes the whole class of omission impossible. **Use
it.** (Whether Anvil errors or zero-fills on a missing field is UNRESOLVED — settle it by compiling one
literal with a field omitted. If it zero-fills, a single missed site resets a generation to 0 and the
build ships green, which is the v1 failure mode exactly.); and in `capstone_rev_node.anvil` lines 18, 78, 112,
152, 166, 185, 202, 204, 206 and the `reg depth_bound : logic[32]` at `:227`. The only arithmetic on
depth anywhere is the `+32'd1` at `:166`.

**One dependency that is load-bearing and invisible, so it is recorded here.** The generation lands in
the node's **high** 64-bit word (`rev_node_t[93:80]`, returned as `data_ruser[29:16]`). That is exactly
the half R-29's stale write-buffer forwarding corrupts. The rev unit is safe from R-29 only because
**every one of its accesses is a full 16-byte granule** — `data_is_cap : 1'b1`, `data_size : 2'b11`,
`data_be : '1` on both the read and the write (`ex_stage.sv:1166-1205`, verified) — so one write-buffer
entry always supplies both halves, where R-29 needs a word-1-only entry. **If any rev-unit access is
ever made partial, the generation is the field that goes stale, and a stale generation reads as a
match.** Any future change to those transactions must be checked against this.

**The invariant** — and note it is scoped to *use*, not to *conferring authority*, which is what closes
the site list:

> **Any use of a revocation reference `(g, i)` — reading the node, conferring authority, or mutating
> the node — is permitted only if `node[i].generation == g`. Reclaiming index `i` increments
> `node[i].generation`, and that single increment simultaneously invalidates every alias of every
> earlier generation.**

**Saturating retirement:** when a generation reaches its maximum the index is never reclaimed again.
The pool shrinks by one and the invariant holds unconditionally rather than probabilistically.

## 4. Every id-as-address site must mask to `[15:0]`

`ex_stage.sv:1159-1161` shifts the **full 30-bit id** left by four, so id bit 16 is worth 1 MiB.
Generation 1 alone addresses past the top of the region; a saturated generation reaches ~16 GiB past a
1 GiB DRAM. This works today only because the allocator hard-wires the top fourteen bits to zero — v1's
claim that *"the AXI address arithmetic is untouched"* was false.

**Required:** mask to `id[15:0]` at `node_query_addr`/`node_update_addr` (`ex_stage.sv:1159-1161`), and
at every call site that passes a capability-supplied id into the unit. **That list is closed and short,
because `capstone_dyn_unit` is the channel's only client** — seven sends, of which six take a raw
capability field and are therefore the sites where a generation would reach the address adder:

| site | op | id source |
|---|---|---|
| `capstone_dyn_unit.anvil:33` | DROP | `rs1_new.metadata.revnode_id` |
| `:59` | REVOKE | `rs1.metadata.revnode_id` |
| `:116` | MREV | `rs1.metadata.revnode_id` |
| `:161` | SPLIT / `init_req` | `rs1.metadata.revnode_id` |
| `:179` | LCC validity selector | `_debug_inp.cap_rs1.metadata.revnode_id` |
| `:509` | DELIN | `rs1_v.metadata.revnode_id` |
| `:132` | `get_node_query_validity` helper | argument — masked by its callers, not itself |

**[AUDIT] Masking at `ex_stage` is NECESSARY BUT NOT SUFFICIENT**, because by then the unit has already
stored a channel-supplied id into link fields and link-valued registers: `capstone_rev_node.anvil:106`
(`revoke_index := parent`), `:112` (`prev = parent`), `:152` (`next = parent_revnode`) and `:183`
(`serving_idx := msg`, later written as `revoke_index.prev`). **The mask belongs at the channel entry
inside `capstone_rev_node.anvil`**, after which the `ex_stage` adders are redundant rather than load-
bearing.

**Mask at the boundary, once, not at each of the six.** Six sites is six chances to miss one, and the
one missed is silent at generation 0. Masking where the id enters the unit — and keeping the generation
in a separate field on the request — makes the omission impossible rather than merely unlikely. `next`/`prev` hold **bare indices**; the generation
lives in the node, not in the links. This is the answer to the question v1 never asked ("what do
`next`/`prev` hold?"), and it is the only answer that keeps the address arithmetic correct.

**Consequence for the broadcast:** the broadcast carries `node_wr_req[29:0]`, which under masking is
the index alone. §6 depends on this.

## 5. The widened invariant closes DROP and DELIN

Both mutate a node with **no preceding query**: `capstone_dyn_unit.anvil:33` (DROP) and `:509` (DELIN)
send straight to the unit, and the DELIN handler (`capstone_rev_node.anvil:74-83`) rewrites the node
without testing `valid` at all. Under reuse a holder of a stale `(g_old, i)` destroys or de-linearises
whichever capability now owns the slot. v1's invariant covered "confers authority" and said nothing
about "mutates", which is why this was a hole in the *property*, not merely in its coverage.

**Under the §3 invariant both must check the generation before mutating**, and refuse otherwise —
which requires Part B, because neither channel can currently say no.

## 6. The three sites that never consult a node — the fatal item, and the honest answer

`pmp_data_if.sv:82-99` is the S/U-mode load/store and fetch authority. It caches a validity bit per
CPMP entry, **adopts any id it has not seen as valid**, and clears only on exact equality with a
broadcast. It never reads a node. `load_store_unit.sv:965-970` and `commit_stage.sv:235-240` have the
same shape and are M-mode only.

**The answer cannot be "widen the comparison", because these sites perform none.** v1's invariant said
a stale alias "fails the comparison wherever it is"; at the site that matters there is no comparison.
That is what would have shipped green, because every test runs at generation 0.

**Proposed mechanism, with its residual stated rather than hidden:**

1. **Compare the recycle event on the INDEX, not the full id** — and **[AUDIT] make sure a recycle
   event actually exists.** Index-masking the compare at `pmp_data_if.sv:96` is necessary regardless:
   under §4 the broadcast carries a bare index, so a 30-bit compare would never fire for a tracker
   holding `(g,i)` with `g ≥ 1`, and invalidation would stop working entirely at generation ≥ 1.
   **But the first draft claimed the reclaim itself broadcasts, and it does not.** The write clearing
   `valid` happens at *invalidation* (`capstone_rev_node.anvil:50`, DROP `:93`); under §1–§3 nothing
   writes `node[i]` at reclaim, so no broadcast carries index `i` there. **Fix: the generation
   increment IS a write to `node[i]` with `valid = 0`, performed at reclaim.** The tap then fires with
   index `i` and the mechanism is true as stated — at the cost of making §1's push **two** node writes.
2. **The residual, and [AUDIT] it is wider than the first draft said.** The window opens at
   **invalidation**, not at recycle, and stays open until the entry is rewritten — under reclamation
   those moments can be arbitrarily far apart. Worse, the adopt guard at `pmp_data_if.sv:90` is
   **per-entry**: clearing entry *k* does nothing for the same stale reference later installed into a
   different entry *m*, which re-adopts it as valid with no second broadcast to undo it. The general
   case is therefore **a capability sitting in memory, loaded and installed at any point after its
   node was invalidated** — not merely after the recycle. No
   broadcast can close this, because the tracker never asks. It is closed by a **software contract**:
   the monitor must validate a revocation reference at CPMP-install time. **This is a requirement on
   the monitor, not a hardware guarantee, and it must be written down as such.** If the lead wants a
   hardware guarantee instead, the only route is giving `pmp_data_if` a query path into the revocation
   unit, which puts a memory read on the S/U access path — a timing cost this design cannot estimate
   pre-synthesis on a part already at WNS −12.425 ns.

**Also settled here:** the false-*deny* direction. Under a bare-index reuse scheme, a tracker holding
index `X` with `valid = 0` would never re-adopt `X`, because the adopt guard is `incoming != tracked` —
a legitimate new capability on a recycled index would be permanently refused. Generation tagging fixes
this for free: `(g_new, i) != (g_old, i)` as 30-bit values, so the guard fires and the entry is
re-adopted. **This is an argument for the generation split that neither audit made**, and it is a
reason not to adopt a bare-index free list even if §6's grant problem were solved.

**[AUDIT] Three further residuals the first draft did not name.**

* **The reserved sentinels are not excluded from reclamation.** `commit_stage.sv:197` plants `30'd1`
  and `capstone_flu_unit.anvil:385` plants `30'd2` into capabilities as hard constants. If index 1 or 2
  ever reached generation ≥ 1 those constants become generation-0 aliases that fail every §3 check.
  Unreachable today because nodes 1 and 2 are never invalidated — **an unstated invariant the reclaimer
  must carry explicitly.**
* **DELIN becomes an availability attack, which makes §5 load-bearing for more than integrity.**
  `capstone_rev_node.anvil:74-83` rewrites a node with no `valid` test, preserving `valid` from the
  read. On a free-listed node that write emits a broadcast, and under an index-only compare it clears
  **every** tracker holding index `i` at any generation. With §5's observation that DELIN takes an
  unvalidated capability-supplied id, a stale reference can repeatedly false-deny the live owner.
* **`commit_stage` compares post-adopt.** Its broadcast compare is against `_d` (`:244`), so a
  same-cycle write to index `i` can clear a legitimate, freshly adopted `(g+1, i)`. Low probability,
  but it is the false-deny direction §6 claims generation tagging eliminates, and at that one site it
  does not.

## 7. A failure encoding — see Part B

Items 5 and 6 both require an operation to be *refused*. None of `init_res`, `rev_res`, `drop_res` or
`delin_res` can express refusal (`capstone_unit.anvilh:500-508`), and the only in-file precedent for
refusal is to never answer and hang the core — the shape R-27 was, before it was fixed and archived. **This is a prerequisite,
not a detail**, and it is specified as Part B so it can be approved and built independently.

## 8. Owner and approval — **owner NAMED 2026-09-16; the two approvals remain OPEN**

M1's start gate: *"The lead must name the runtime/RTL owner and approve the algorithm and stale
reference invariant before anyone schedules the reclaiming arm."* Three parts.

**Part 1 is now satisfied: the lead has named the RTL lane as owner** (2026-09-16). That lane is
accountable for the algorithm, the invariant, the implementation and the wrap bound — which is the
role this document was written in.

**Parts 2 and 3 are NOT satisfied and the owner cannot satisfy them.** They are approvals *of* the
owner's work; an owner approving their own algorithm makes the gate decorative. That is not a
formality here — the audit above found three mechanisms in this very document individually wrong in
ways that would have shipped, and it found them after the author had already reviewed it twice.

**So: nothing is scheduled, and no reclaiming arm may be planned, until the lead approves the algorithm
(§1–§3 as corrected) and the stale-reference invariant (§3).** The protocol's own escape applies if
they do not: *"If no valid algorithm is available, return an implementation-blocked handoff."*

## 9. The id-transplant primitive

`CAPNODE` copies `rs1`'s `revnode_id` onto `rs2` (`capstone_flu_unit.anvil:398-404`) through
`modify_cap_revnode`, which validates nothing. It cannot forge an arbitrary id — the id must come from
a capability already held — so it is an **amplifier of the reuse hazard, not an independent hole**:
retain a stale capability, wait for its index to be recycled, transplant the id onto a capability with
different bounds and permissions.

**Its decode arm is not privilege- or capmode-gated** (`decoder.sv:1111`, `OpcodeCustom3`; no gate in
any enclosing scope, verified by walking outward from the arm). Under the §3 invariant the transplanted
reference fails its generation check at use, so the invariant covers it — but the spec must say so
explicitly, because v1's site list was a judgement call about which uses mattered and this one would
not have been on it.

**Related, and worth recording even though it is not a reuse issue:** `decompress_cap_tagged`
manufactures `revnode_id = 0` for an untagged word (`ariane_pkg.sv:769`), and node 0 is the permanently
invalid sentinel whose maintenance writes broadcast id 0 routinely. Id-0 broadcasts are therefore an
existing unintended-invalidation channel, independent of anything proposed here.

---

## Part B — graceful exhaustion (separable, approvable alone)

**Today, exhausting the pool hangs the core.** `capstone_rev_node.anvil:128-138` and `:171-178` drop
the request rather than alias an id, and the comment states the intent: *"a forced, visible stall
instead of silent id reuse."* The stall is correct as a safety choice and wrong as an interface — the
dynamic unit blocks forever on `recv`, and the only external signal is a debug LED
(`capstone_rev_node.anvil:275-279`). It is invisible to software by construction.

**Proposal:** give the response channels a way to say no, and turn exhaustion into a reportable
architectural fault rather than an undiagnosable wedge. No generation tagging, no allocator, no change
to the id format, no change to the capability format.

**Why it stands on its own:** a wedge cannot be diagnosed from a board run, and today a pool-exhaustion
wedge is indistinguishable from R-27's orphan-response deadlock and from any other hang. A trap names
itself. It also makes the capacity ceiling *measurable* — which M1 needs whether or not a reclaimer
is ever built, and which the bounded-prototype claim needs if one is not.

**Design note:** the refusal must propagate to an architectural exception with a distinct cause, not a
silent zero — a mint that returns a null id which the caller then uses is strictly worse than the
stall. The cause encoding and the dyn-unit side are the substance of this part and are deliberately
left for the owner named under §8.

---

## The requirement tension, with a recommendation

The reclaimer target requires *"implemented reclamation algorithm, reference-accounting invariant,
completion event and finite progress or generation limits"*.

**Hardware cannot establish reference counts.** Capabilities are copied freely into tagged memory the
hardware never enumerates; there is no refcount anywhere in this RTL; and QEMU's software refcount is
documented as never reaching zero in practice, because a node is released only when every capability
copy in memory is overwritten.

**Recommendation: "reference-accounting invariant" names the required property, not refcounting as the
mechanism, and the §3 invariant satisfies it.** The paper's own `impl.tex:38-47` states the property as
*"recycling that identifier before all relevant old references are accounted for can make the stale
capability refer to new authority"*, and asks the experiment to *"test that none becomes usable
again"* — which a discriminator establishes directly and a count does not. `appendix-evidence.tex:74-80`
already says *"Account for node turnover, exhaustion, and generation limits."* The M1 protocol
explicitly sanctions a discriminator and requires its finite wrap bound be recorded.

**Limit, stated plainly:** `appendices/b-target-results.tex` lives in the other paper repo
(`capstone/paper`, `pointer-safe-paper`) and is **not reachable in this checkout**; that wording
reaches this lane only second-hand. The lead confirms or rewords. This lane edits neither paper.

---

## Verification — and the one requirement that decides whether any of this is testable

> **Every functional test, lit run and QEMU suite today executes at generation 0, where `{14'd0, i}`
> and `(0, i)` are identical. A correct implementation and a broken one are indistinguishable there.
> A check that has only ever run at generation 0 is not a passing check.**

1. **Force wrap early with a reduced-width generation** — and note what that costs here. Narrowing the
   generation to two or three bits makes wrap reachable in a directed test, and the M1 protocol
   sanctions exactly this, requiring the reduced-width result be reported **separately** from the
   production-width bound.

   **It cannot be a `+define+`.** The Anvil compiler "does not reliably resolve `const` definitions when
   referenced from a value-comparison position inside a proc (only from a type/bit-width position)", so
   the existing pool width is kept as **hand-synchronised literals — 21 sites across four files**
   (`capstone_unit.anvilh`, `capstone_rev_node.anvil`, `ex_stage.sv`, `cva6.sv`). A narrowed generation
   inherits exactly that problem. The reduced-width diagnostic is therefore a **hand-edited variant
   branch that must be built, hashed and reported as its own artifact**, never a flag on the production
   build — and the production build must be proven byte-identical to the unmodified one, because a
   variant that silently ships is the failure this whole document exists to avoid.
2. **Negative-test at generation ≥ 1 through every group-(b) site**, not only through a load: the CPMP
   S/U path, the LSU M-mode path, the PC path, DROP, DELIN, REVOKE, and a CAPNODE transplant.
3. **Create the condition, do not merely contain the shape.** The test must retain a capability across
   an actual reclaim of its index and then use it. A test that never reclaims passes vacuously.
4. **Positive control:** the same test must show the stale reference *succeeding* on the unmodified
   tree, or it is not testing what it claims.
5. **Free-list integrity:** a directed test that reclaims and re-mints the same index repeatedly, with
   an assertion that no index appears twice in the list — the circular-list hazard of §1 is the same
   shape as R-12's original unbounded walk.

**No RTL is in this deliverable**, so no lint or synthesis gate applies to it and nothing is requested
from the synth lane. Those gates bind whatever is built from it.

## What must not be repeated from v1

* **"The change is confined to the node unit and the comparison site."** Refuted: `ex_stage.sv`
  hard-codes the datapath by literal bit index. v2's *storage* choice is genuinely confined because
  `depth` is already plumbed — but the *consumer* changes (§4, §5, §6, §9) are the bulk of the work and
  are not confined to anything.
* **"It costs nothing."** Refuted on glue-change count alone, before timing is considered.
* **The ≈1.07 × 10⁹ capacity headline.** Not computable until the reissue policy exists, and false by
  up to 65,532× under LIFO or lowest-free-first. **Under §1's FIFO the figure is computable** — but it
  must be derived and stated with the policy beside it, never quoted bare.
* **Treating the slot padding as free.** It is 33 bits, not 34, and it is not plumbed.
* **Ruling out a free list without supplying an allocator.** The list is the mechanism; the
  discriminator is what makes it safe.

## Open for the lead

1. **Name the RTL owner, and approve the algorithm (§1–§3) and the invariant (§3).** Nothing is
   scheduled before this.
2. **§6's residual** — accept the monitor-side validation contract, or require a hardware query path
   on the S/U access path with the timing risk that carries.
3. **§1's LIFO-vs-FIFO trade.** LIFO is free — its push folds into a write the revoke walk already
   makes — but the first index retires within a single workload run. FIFO pushes first retirement out
   to ~10⁹ and costs one extra write plus one spurious invalidation broadcast per reclaimed node. (An
   earlier draft added a third cost, a longer mint-stall window; it is withdrawn — see §1. The dyn unit
   runs one op at a time, so no mint can be in flight during a walk.) Recommendation remains FIFO:
   the capacity figure is the study's point and the real cost is one write.
4. **§2's DROP trade** — leave dropped nodes ineligible until a revoke sweeps them, or pay two reads
   and two writes on every DROP to make them eligible immediately.
5. **The requirement tension** — confirm that a discriminator satisfies "reference-accounting
   invariant", or reword the target.
6. **Part B alone?** It is separable, it is a prerequisite for §5 and §7, and it has standalone value.
