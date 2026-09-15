# Two audits of the revocation-node reclamation design, reconciled (2026-09-15)

*The design is `docs/plans/2026-09-14-revnode-reclamation-design.md`, proposed for M1. Two adversarial
audits were commissioned independently within hours of each other — one by the design's own lane, one
by the paper lane. This note is the single reconciled record. Verdicts are on the design document
itself as banners; this file says what each audit attacked, where they agree, where they contradict
each other, and who was right.*

## Verdict

**The design is INCOMPLETE, not merely unsafe.** It specifies the safety property of reuse and the
condition for reclaimability, and never specifies the mechanism that reuses. Both audits reached
failure; this framing is the one that survives reconciliation.

## What each audit attacked

| | targets |
|---|---|
| **RTL lane's** | the four the document names itself — the comparison site as the whole security boundary, reclaim against an in-flight query, the subtree condition under partial revoke, retirement as a DoS surface — plus whether `revnode_id`'s top bits are truly hard-wired zero and whether any software consumer treats the id as flat |
| **Paper lane's** | field placement and whether the padding is free, the two-field comparison's timing path, saturating retirement against the subtree rule, and whether the design's own admissions hide a dependency |

Only field placement was unique to the paper lane's. The overlap was real and is resolved below.

## Where they agree

* **`DROP` and `DELIN` mutate a node without reading `valid`** (`capstone_rev_node.anvil:62-75`,
  `:44-56`), so a holder of a stale `(i, g_old)` can destroy or de-linearise the **new** owner's node.
  This is the one finding that is a hole in the **property**, not in its coverage: the invariant is
  scoped to *conferring authority* and says nothing about *mutating*. A second design needs it widened
  to **any use** of a revocation reference — which also makes the site list **closed**, since
  enumeration then terminates instead of being a judgement call about which uses matter.
* **Reclamation needs unlinking, and the design never says so.** Invalidation preserves `prev`/`next`
  (`capstone_unit.anvilh:563-565`), so a reclaimed index stays in its old parent's chain; a later
  unrelated `REVOKE` either revokes a fresh capability that merely inherited the index, or terminates
  early and under-revokes.
* **The QEMU/RTL compressed-metadata layouts diverge from bit 27 upward**, because the RTL's
  `bounds_t` carries a `cursorless` bit QEMU lacks — `revnode_id` is 31 bits at `[63:33]` in QEMU
  against 30 bits at `[63:34]` in the RTL.

## Where they contradict each other, and who was right

**The node's bit packing. The paper lane's audit is right; the RTL lane's measurement was wrong.**

The RTL lane read the generated `[93:0]` slice — a **width** — and inferred field **order** from the
declaration sequence, assuming first-declared occupies the low bits. Anvil packs **MSB-first**:

    depth node_o[93:62] | prev [61:32] | next [31:2] | valid [1] | linear [0]

The decisive evidence needs no archaeology. `ex_stage.sv:1207` is
`revnode_invalidation_valid_o = mem_rev_wr_req_valid && !node_wr_req[31]`, and the consumer's own
comment at `commit_stage.sv:242` reads *"Invalidate if rev_node is writing valid=0"*. The message is
`{node_o, node_id[29:0]}`, so `node_wr_req[31]` is `node_o[1]` — which is `valid` only under
MSB-first. Under the asserted order it would be a `depth` bit, never 1 for a real tree depth, and the
broadcast would fire on nearly every node write. Confirmed independently by the paper lane, then
re-derived and accepted by the RTL lane. Both audits are cited in the retraction at `28eefbe8d88a`.

**Two claims fall with it, and both were the paper lane's only because they rested on the RTL lane's
measurement. Stated plainly at the RTL lane's own request:**

1. *"`next` straddles bit 64, so the node already depends on high-half integrity today, and a shared
   refill leg would be a live defect in the shipped design."* **Withdrawn.** `next`, `prev`, `valid`
   and `linear` are all in the **low** word. The only node field in the high half is `depth[31:2]`.
2. *"Narrowing `depth` lands the generation below bit 64, so do it deliberately rather than as a
   fallback."* **Backwards.** Narrowing `depth` frees **high** bits. Landing a generation below bit 64
   means narrowing `prev`/`next` — 30 bits each, only 16 reachable, 28 free inside `node_o[63:0]`.

**The error's class is already in this project's rules**: a field's position in a declaration is not
its position in the generated slice — the standing example is a `trans_id` read at `[2:0]` where the
generated code reads `[255 +: 3]`. What makes this instance worth recording is that the RTL lane's own
auditor had derived the packing correctly from a bootstrap literal, and its answer was overwritten by
the earlier inference rather than reconciled against it. **The right answer was in hand and the two
were never compared.**

## Found by one audit only

**By the RTL lane's** — and it is fatal to the mechanism:

* **The unprivileged boundary performs no comparison at all.** `core/pmp/src/pmp_data_if.sv:82-97`
  caches a validity bit, marks **valid any id it has not seen** (`:90-92`), clears only on exact
  30-bit equality, and never reads a revocation node. Split the id and the broadcast carries
  `{14'd0, i}` against a tracker holding `(g, i)`: after the first reuse of any index the match never
  fires and **revocation silently stops invalidating unprivileged capabilities**. Nothing faults.
  Everything runs at generation 0, where the two forms are identical, so **every functional test, lit
  run and QEMU suite would pass**. It would have shipped green.

**By the paper lane's** — and it is fatal to the design's completeness:

* **`head` is only ever incremented.** `capstone_rev_node.anvil:79` and `:141` increment, `:179`
  resets to 3, and there is **no free list, reclaim queue or reuse scan anywhere in the unit**. With
  the design exactly as written **no index is ever reused, the generation is never consulted, and the
  invariant is vacuously true while doing nothing.** The capacity figure has no allocator behind it.
  The free list the design rules out in its opening section is the missing half of the proposal, not a
  rejected alternative.
* **`revnode_id[29:16]` are address bits.** `ex_stage.sv:1160-1161` takes the **full 30-bit id** and
  shifts it left by four, so id bit 16 is worth 2^20 bytes: **generation 1 alone addresses
  0xC000_0000**, past the top of the DRAM the design's own memory table ends at, and a saturated
  generation reaches 16 GiB past base. It works today precisely because `#{14'd0,*head}` makes the top
  fourteen bits hard zeros. The design's *"the AXI address arithmetic is untouched"* is false.
* **The padding is not plumbed.** `ex_stage.sv:1149` reads back only `data_ruser[29:0]`, so anything
  at slot bits 94..127 is invisible to the unit; `node_update_b = {34'd1, …}` makes **bit 94 a hard
  constant 1**, giving 33 free bits rather than 34, exactly where a naive append lands.
* **`NODE_ID_INVALID` collides with the re-typing.** `verif/tests/custom/capstone/asm_insn.h:64`
  defines it as 0x7FFF_FFFF, whose bits [29:16] are all ones — precisely "retired index at maximum
  generation" under the proposal. An existing sentinel silently re-typed.

## Unresolved, with what would settle each

* **The R-29 question is REFUTED but leaves a residual.** The rev-node endpoint *does* share the
  overlay and refill leg (`cva6.sv:2191-2193`, `wt_dcache.sv:219/314`,
  `wt_dcache_mem.sv:353-358/394/397`), but never reaches the mechanism: every rev-unit access is
  granule-aligned, full-granule, word-0 `is_cap`, so one write-buffer entry supplies both halves,
  while R-29 needs a word-1-only entry. **Residual:** a *software* plain `sd` into
  0xBFF0_0000-0xBFFF_FFFF would recreate the trigger exactly. `ex_stage.sv:1195` asserts the region is
  monitor-only by PMP and **that assertion is unverified**. Settle it by reading the PMP/`cpmp`
  configuration the monitor installs and confirming the monitor issues no plain store there.
* **Timing: no claim either way is supportable.** The design's premise that the comparison is on every
  capability access is false — `query_req` issues only from the dynamic unit
  (`capstone_dyn_unit.anvil:105` and its callers). The per-access comparators at
  `load_store_unit.sv:944` and `commit_stage.sv:244` are already 30-bit equalities and would not
  widen. But only synthesis proves synthesizability, and the flashed bitstream already misses timing
  at WNS −12.425 ns with 102,508 of 174,960 failing endpoints. The design's *"costs nothing"* is
  refuted on the glue-change count regardless of timing.
* **The QEMU/RTL layout divergence is latent, not live.** No artifact was found in which the two sides
  exchange a compressed metadata word. Settle it by checking whether any differential test compares a
  compressed word or a `CAPNODE` result across QEMU and silicon; if one does, it has been comparing
  misaligned fields. Do not upgrade it without such an artifact. See also the rule in
  `15-09-2026_15-44-40_sublet-draft-audit.md` §6 about R-11: QEMU has no `cursorless` encoding at all,
  so *"the RTL does X, QEMU never does"* is a coverage gap rather than a behavioural disagreement.

## What a second design must assert

Not "the same design plus plumbing". The direction survives — the 65,532-lifetime ceiling is real and
generation tagging lifts it, and nothing in either audit touches that. But a second design must state:

1. **the allocator** — a free list or equivalent; without it the proposal is vacuous;
2. **the reissue policy** — the capacity figure is uncomputable until it exists, and it decides
   whether the first index retires after ~10⁹ lifetimes or after a single workload run;
3. **the unlink** on reclaim;
4. **the index/generation split at every id-as-address site**, the address arithmetic included;
5. **the widened invariant** — any *use* of a revocation reference, not only conferring authority;
6. **what happens at the sites that grant authority today without consulting the node at all** —
   `pmp_data_if.sv:82-97`, `load_store_unit.sv:967-969`, `commit_stage.sv:236-240`.

Whether M1 gets that second design or a different direction is the lead's decision; neither audit
forces it.
