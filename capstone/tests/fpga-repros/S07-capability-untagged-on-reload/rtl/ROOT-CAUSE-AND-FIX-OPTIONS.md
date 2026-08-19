# S-07 root cause, and the two fix options — from the RTL lane

Self-contained. Siblings: **S-06** (untagged `ldc`/`stc` high half, fixed), **S-08** (dom-switch
context width, fixed), **S-09** (the same write-buffer mechanism seen from the store side).
If you arrived here about a *forged* tag rather than a *lost* one, read S-09 — and read its
retraction, because the forgery framing was withdrawn.

---

## 1. What the defect is

**The D-cache write buffer tracks entries at 64-bit word granularity, but a capability store
writes a 16-byte granule. The two disagree, and the write buffer cannot see the overlap.**

    wt_dcache_mem.sv:242-251     if (!(st_wr_cap)) bank_req |= one-hot(word)   // plain: ONE word
                                 else              bank_req  = '1              // stc:  BOTH words

    wt_dcache_wbuffer.sv:444     hit compare on wtag == {address_tag,
                                                        address_index[11:XLEN_ALIGN_BYTES]}
                                 -> 8-BYTE WORD, so G+0 and G+8 never merge

    wt_dcache_wbuffer.sv:410     wr_idx_o  = wr_paddr[11:4]      -> the 16-BYTE GRANULE index
    wt_dcache_wbuffer.sv:416     wr_ctag_o = wbuffer_q[rtrn_ptr].ctag
    wt_dcache_mem.sv:459         cap_tag_q[wr_idx_i][j] <= wr_ctag_i   // whole granule's tag
    wt_axi_adapter.sv:158        tag byte = (paddr - DATA_MEM_BASE) >> 4  // same byte, both halves

    wt_dcache_wbuffer.sv:479     i_dirty_rr = rr_arb_tree over `dirty`   -> rotation, NOT age

A capability entry at `G+0` covers **both** words of the granule. A plain store to `G+8` gets its
**own** entry, because the hit compare is on the word address. Both entries write the same
physical word and the same single tag bit, and the drain arbiter is round-robin, so **whichever
drains last wins** — regardless of program order.

**The tag disagreement is a symptom, not the disease.** The disease is that `is_cap` entries span
two words while being tracked and merged as one. An earlier version of this document said "the
loser's tag wins"; that was a symptom presented as a mechanism and is superseded here.

### The two outcomes

| program order | drains last | result |
|---|---|---|
| plain `G+8`, then `stc G` | plain | **tag cleared, capability destroyed** — the program faults on first use. This is S-07. |
| `stc G`, then plain `G+8` | `stc` | capability intact, **the plain store is silently dropped** |

Neither outcome produces a capability over program-chosen data: when the `stc` entry drains last
it rewrites the whole granule, cursor and metadata together, so the survivors carry the *original*
capability. Measured: the corrupted-but-tagged bucket is **empty in both directions**.

### Why the second row still matters

A dropped store is not merely an integrity bug when the dropped store was a **scrub**.
`memset(p, 0, sizeof *p)`, `explicit_bzero`, free-list poison, clearing a slot before reuse — all
of them are plain stores over a granule holding a capability, and all of them are how software
destroys authority it no longer wants to hold. When that store is dropped, **the capability
survives the operation intended to destroy it**. That is a failure to revoke by overwrite: weaker
than fabricating authority, stronger than losing a scalar.

---

## 2. Evidence

**Simulation** (`verif/tests/custom/capstone/s07-wbuf-tag-reorder{,-ctl}.S`, ~15 s each, no board):

    s07-wbuf-tag-reorder      4 UNEXPECTED_OPERAND   9150 cycles   (positive control + 3 losses)
    s07-wbuf-tag-reorder-ctl  1 UNEXPECTED_OPERAND   9020 cycles   (positive control only)

One variable between them: a plain `sd` to the granule's high word before the `stc`. Each carries
its own positive control — an `ldc` through a plain integer, run first — so a zero from the
detector is never believed on trust.

**Silicon**, five arms, one boot, on the already-flashed bitstream — no reflash was needed:

    wb0  stc G only                    lost     0        control clean
    wb4  plain G+16; stc G             lost     0        granule-scoped
    wb3  plain G+8; 64 stores; stc G   lost     0        buffer drained
    wb2  stc G; plain G+8              lost 15193/16384  detector fires
    wb1  plain G+8; stc G              lost  1107/16384  *** program order says 0 ***

**wb1 against wb3 is the decisive pair**: identical stores, identical granule, differing only by
64 unrelated stores that drain the buffer. 1107 versus 0 isolates buffer co-residency and nothing
else. QEMU returns 0 for every arm — it has no write buffer, no per-word entries, no drain
arbiter — so the emulator structurally cannot reproduce this and never could have found it.

**Everything previously unexplained now follows.** A deterministic site under an intermittent
trigger (drain order is fixed by the instruction stream; occupancy by whatever ran before).
`rd_ctag_src` never discriminating (both legs deliver the same corrupted value, so no `src`
reading could ever have separated them). And the trigger is ordinary C on the failing path —
`sqlite3JournalOpen` does `memset(p, 0, sizeof(MemJournal))` and stores `pMethods` twelve lines
later, into the granule whose load faults.

---

## 3. THE FINDING THAT GOVERNS EVERY FIX AND EVERY TEST

**`ctag` is sampled TWICE, at different times.**

    wt_dcache_wbuffer.sv:298-299   miss_wctag_o = wbuffer_dirty_mux.ctag   <- TX ISSUE  -> DRAM
    wt_dcache_wbuffer.sv:415-416   wr_ctag_o    = wbuffer_q[rtrn_ptr].ctag <- TX RETURN -> L1

So **any fix that mutates a resident entry after its transaction has issued writes one tag to L1
and a different one to DRAM.** L1 wins every immediate readback; DRAM wins once the line is
displaced. A fix can therefore look perfect and leave the capability resurrectable.

**Consequence, and it is not optional: every acceptance test must include a forced-eviction
reload leg** — touch `DCACHE_SET_ASSOC+1` other lines in the same set, then reload. The existing
S-09 detector reads back immediately and would report a broken fix as working.

The AXI bus is *not* the reorder source: data writes are pinned to ID 1 and tag writes to ID 0
(`wt_axi_adapter.sv:214, 452`), and same-ID writes complete in order. The reorder is drain
*selection* (`i_dirty_rr`), which is why the fix belongs in the write buffer.

## 4. The fix options, after four independent audits

### REJECTED — (A) granule-aware merge

The entry's `valid`/`dirty`/`txblock` are 8-bit masks over ONE word and `.user` has **no byte
tracking at all** (`wt_dcache.sv:70-83`). "Bytes 8-15 dirty" is inexpressible. Making it
expressible means widening masks across five files, ~+224 flops, 200-400 lines.

Worse, **as sketched it converts a visible failure into an invisible one.** A scrub merged into an
entry whose TX has already issued sets no dirty bits (its bytes live in `.user`), so
`bdirty = (|txblock) ? '0 : ...` never re-drains it; the return writes the merged value to L1 and
`evict` then frees the entry. **L1 says the scrub landed; DRAM still holds the tagged capability.**
The current detector would call that a success.

### REJECTED — (C) allocation-time fixup

Two independent kills. **Partial byte-enable:** `bank_wdata[1]` is written only by an `is_cap`
entry (`wt_dcache_mem.sv:170-172`), so demoting the resident capability entry means a `sb G+9`
leaves the other seven metadata bytes holding **pre-`stc`** content. **Wrong-address tag write:**
invalidating an in-flight entry leaves `tx_stat_q[id].ptr` pointing at it; the entry can be
reallocated to a different address, and the return then writes that new occupant's `ctag` —
a tag write at an address the program never stored a capability to. `tx_valid1`
(`wt_dcache_wbuffer.sv:693`) is the tell. C is also a strict superset of B's cost.

### REJECTED — drain-side ordering (make a conflicting entry ineligible for arbitration)

**There is no age state in the entry** (`wt_dcache.sv:70-83`) and the slot index is not a proxy for
age — allocation is lowest-free-index (`:461-467`). Masking therefore fixes an *arbitrary* order:
correct for one program and silently wrong for the other. It also removes the nondeterminism that
made the bug findable. Naive symmetric masking deadlocks.

### REJECTED — stalling in the store buffer

Incomplete. Port 3 is a mux (`cva6.sv:2353-2354`) and the **rev-node write port bypasses the store
buffer entirely** while asserting `data_is_cap` unconditionally (`ex_stage.sv:1142-1152`). A stall
there cannot see those entries. This producer was missing from every earlier analysis.

### RECOMMENDED — (B) forbid granule co-residency, accept-side

Refuse to allocate an entry that conflicts at granule level with a resident entry when either side
is `is_cap`. Place it beside `ni_conflict` inside `p_buffer` (`:530, 592-598`), not folded into
`rdy`, so the `write_full` assertion keeps its meaning.

    gran_eq[k]       = (wbuffer_q[k].wtag[52:1] == req_wtag[52:1]);   // 52 of 53 bits shared
    gran_conflict[k] = valid[k] & gran_eq[k] & (wtag[0] != req_wtag[0])
                                & (wbuffer_q[k].is_cap | req_port_i.data_is_cap);

**Why it wins, on measurements rather than taste:**

* **It is the only option with no post-issue mutation**, hence the only one immune to the L1/DRAM
  split in section 3 above. Smallest illegal window — arguably zero: memory passes through exactly
  the states a legal program produces.
* **The cone concern was mine and it was wrong.** `rdy` has exactly one consumer (`:592`); its
  entire transitive fan-in is registers; `wt_dcache_wbuffer` appears nowhere on the UNOPTFLAT list
  (`wt_dcache_ctrl` is instantiated only for ports 0-2); and the ~100 `config_pkg.sv:413` timing
  criticals cannot arrive through this path because its only `range_check` call site is `is_ni`,
  and `NonIdemPotenceEn` is constant 0 in this config, so it folds away. Cost: **+1 logic level**,
  sharing 52 of 53 bits with a comparator already on the path.
* **It covers in-flight entries.** An entry stays `valid` while `txblock` is set (`byteStates`,
  `:712`; `valid` clears only on evict when `dirty==0`, `:560-565`), so the conflict check sees
  transactions still on the bus.
* **It fixes a second, unmeasured defect for free** — see I-b below.
* **Deadlock-free by construction:** drain, tag-check and evict are functions of `wbuffer_q` /
  `tx_stat_q` / the return FIFO only. Nothing on those paths consults `rdy`, `data_gnt` or
  `req_port_i`, so a stalled accept always resolves.
* Failure direction is safe: the linear-source clear can only be *delayed*, never demoted or
  dropped, and backpressure parks the LDC in `LDC_CLEAR_WAIT` without committing.

**Real risk is liveness, not security**, and it needs a directed test rather than an argument —
specifically the interaction with the existing NI drain-and-block and with the `hot1` /
`write_full` assertions, which assume today's hit/allocate relationship.

## 5. Acceptance criteria — tests that FAIL if the invariant breaks

1. **Eviction leg, mandatory on every tag test.** `stc G`; force the TX to issue; `sd G+8` with a
   distinctive non-zero pattern; evict by touching `SET_ASSOC+1` other lines in the same set;
   reload; `lcc`. FAIL if it reports a capability type, or if `G+8` != pattern. Positive control:
   this must FAIL on the unfixed design at >=4%, or the eviction leg is not evicting.
2. **Three-bucket kernel with the eviction leg.** FAIL if CORRUPTED-BUT-TAGGED is non-zero. That
   bucket has a demonstrated negative (0/3840 unfixed), so a non-zero reading means something.
3. **Liveness stress, non-negotiable.** Alternating `stc G` / `sd G+8` at rate across >=256
   granules, mixed with an NI-region store, an `ldc` of a LINEAR capability with a pending store,
   and an AMO to the same line. FAIL on watchdog timeout or if `hot1`, `write_full`, `tx_valid1`
   or `byteStates` fire. **Confirm assertions are actually enabled in the run first**, or this
   criterion is vacuous.
4. **Linearity.** `ldc` a LINEAR capability from `G`; `sd G+8`; evict; reload; `ldc G`. FAIL if the
   second `ldc` yields a tagged capability (duplication) or if `G+8` reads back as pre-clear
   metadata (clear partially dropped).

## 6. Two adjacent defects that MUST NOT be folded into this one

**I-b — store-to-load forwarding is word-granular too, and is measured nowhere.**
`wt_dcache_mem.sv:280` compares at word granularity and `:344-345` selects *metadata* bytes using
the *data* half's valid mask. So `stc G; sd G+8; ldc G` with both entries resident returns the
pre-scrub capability with tag 1, and `stc G; ld G+8` reads stale memory. Exists today at
`a3dbae618`. **(B) makes it unreachable; (C) leaves it entirely.** It deserves its own repro
folder — one issue per folder is a hard rule here.

**I-c — AMO over a capability granule (invariant I4).** `needs_tag` excludes atomics
(`wt_axi_adapter.sv:141-152`), so an AMO leaves both the DRAM tag byte and `cap_tag_q` set. **None
of A/B/C touches this.** Any post-fix "tags are correct now" claim must state that AMO is excluded,
and the failure must be recorded *before* the fix so the fix is not blamed for it.

## 7. The two fix options (superseded — retained for the reasoning)

**Neither has been built.** This is a data-path change, not observation-only; it needs its
security invariants argued rather than assumed, and the choice trades correctness completeness
against synthesis risk.

### (A) Make the merge granule-aware for capability entries

A plain store to `G+8` merges **into** a resident `is_cap` entry at `G+0`, its bytes routed to
that entry's metadata half, `ctag` last-writer-wins. One entry, one writeback, order-independent.

*For:* correct and complete — it removes the overlap rather than sequencing around it, and it is
exactly the semantics the existing same-word merge already implements
(`wt_dcache_wbuffer.sv:616-630`).
*Against:* real surgery on the merge path, which is live for every store in the machine.

### (B) Forbid granule co-residency

Refuse to allocate an entry that conflicts at **granule** level with a resident entry when either
side is `is_cap`; stall until it drains. Same-granule entries then never coexist, so no reorder is
possible.

*For:* small, obviously correct, and confined to capability-adjacent stores, so throughput cost is
near zero in ordinary code. **wb3 is the empirical evidence it works** — draining between the two
stores gave 0 losses out of 16384.
*Against:* adds a term to `rdy` (`wt_dcache_wbuffer.sv:458`), which feeds the grant path. The
dcache request/grant cone is on the standing `UNOPTFLAT` list and is the neighbourhood where
synthesis has twice gone pathological.

### One fix that looks obvious and is WRONG

Propagating the youngest store's tag to co-resident same-granule entries, so drain order stops
mattering. It fails in the dangerous direction: the older plain entry still writes its stale
scalar over the metadata half, and giving it `ctag=1` as well produces **a valid tag over
corrupted metadata** — converting a loss into a forgery. A tag-only test reports this as a
complete success.

**Any candidate fix must therefore be validated against metadata integrity, not against the tag
alone.** The extended kernel does this, with three outcome buckets — tag lost / intact /
corrupted-but-tagged — and the third reads 0 on unfixed hardware, so it is a working detector with
a proven negative.

---

## 4. Recommendation

1. **Do not spend the next bitstream on instruments.** The pending batch was built to observe a
   defect that is now root-caused by other means; the correlation bit answers a question that no
   longer gates anything. The next bitstream should carry the fix.
2. **Design the fix against both arms and both severities**, then validate with the three-bucket
   kernel, then in simulation with the directed pair above, then on silicon with wb1/wb3.
3. A **software mitigation exists** if one is wanted before silicon: separate the plain store from
   the capability store so the buffer drains between them. wb3 is exactly that experiment and it
   is clean. This is a workaround, not a fix — it depends on compiler scheduling and cannot be
   relied on.

## 5. Limits of what is claimed

- The mechanism is confirmed; the **fix is not designed**, and neither option above has been
  built, simulated or timed.
- The dropped-store consequence is measured for the tag and the capability's fields; the arm that
  reads the scalar back from `G+8` directly is the cleanest confirmation and is the one to build
  next if the scrub case is to be leaned on.
- Provenance: the same per-line tag write and plain-store tag path exist at `25035c4c0^`, before
  the S-06 work. S-06 changed the tag *value* semantics (`|user|` → `ctag`); it did not introduce
  the granularity mismatch or the drain ordering. **Pre-existing.**
