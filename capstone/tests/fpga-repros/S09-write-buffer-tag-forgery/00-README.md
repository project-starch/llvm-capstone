# S-09 — a capability survives the plain store meant to destroy it

> **TIMING CAVEAT, NARROWED 2026-08-20 — this fix is EXONERATED as the cause; the measurement was
> still taken on a bitstream that misses setup.** Read directly from the archived reports of the
> build that produced `caplifive_s07fix.bit`, not taken on report.
>
> Post-route **WNS −10.629 ns**, and it is **one clock**: `clk_out1_xlnx_clk_gen`, 96727 of 174481
> endpoints. Its constraint is **correct, not misapplied** — the Clock Summary gives it as
> 40.000 ns / 25.000 MHz, exactly what `xlnx_clk_gen` is generated for. Every other clock closes
> (`clk_out2` +0.694, `eth_rxck` +4.140, every MIG-derived clock positive). Hold is fine
> (WHS +0.054).
>
> **RETRACTED: "it is not this subsystem". THE VIOLATED CONE DOES TRAVERSE THE WRITE BUFFER.**
> Every violated path in every archived report shares a single source,
> `i_ariane/i_cva6/dom_switcher/cur_idx_q_reg[1]/C`, fanning out to 10 distinct destinations in the
> scoreboard (8) and `issue_read_operands` (2) — but each of those 10 paths traverses **22 nets**
> under `i_wt_dcache_wbuffer` / `wt_dcache_mem`, named after real RTL signals: `rd_req[1]`,
> `rd_ack[0]`, `rd_req_masked[0]`, `vld_sel_d[0]`, `wbuffer_hit_oh[5]`, `wbuffer_hit_idx[0]`,
> `data_rdata_q[...]`. Positive-controlled on net names, which are reliable where hierarchy
> prefixes are not: 220 such nets appear across the report, so the matcher fires.
> **The earlier "0 of 100 touch the dcache" was worse than vacuous — the true answer is the
> opposite.** It came from matching `Source:`/`Destination:` endpoint fields only, which name the
> two ends of a path and never its interior.
> **What is NOT established either way: whether the S-07 fix's own logic is on that cone.** The
> traversed nets are all **read/tag-check side** (`rd_req`, `rd_ack`, `wbuffer_hit_oh`,
> `data_rdata_q`), while the fix adds logic to the **allocation** side (`gran_hazard` →
> `data_gnt`/`wbuffer_wren`); and `wbuffer_hit_oh` pre-dates the fix. But a search for
> `gran_hazard|gran_conflict|gran_eq|word_ne|req_wtag` returns **0 of 8946 nets across the whole
> report**, and `gran_` appears nowhere at all — so that zero cannot distinguish "the fix's logic
> is off the cone" from "those net names did not survive synthesis". **It is unproven, not
> exonerated**, and settling it needs the routed checkpoint on the Vivado machine. The worst
> path runs 50.536 ns against 40.000 ns over **123 logic levels with 82% of the delay in routing**,
> at 169415/203800 LUTs (83%) with place and route both on `-directive RuntimeOptimized`. It is
> structural, in domain-switch machinery. `core/anvil_build/` is byte-identical across the range
> and `capstone_dom_switcher.anvil` last changed at `25035c4c0`, an **ancestor** of the healthy
> reference build. A change confined to the inside of one cache module cannot induce it.
>
> **What carries the results is the DIFFERENTIAL structure**, now on firmer ground: every number
> here is a comparison between arms differing by exactly one thing, on one bitstream, in a
> subsystem the failing cone does not touch, agreeing with Verilator — which has no timing model
> at all. Repeatability is deliberately not offered: a setup-failing path at fixed voltage and
> temperature can fail deterministically.
>
> **What remains caveated:** these are absolute numbers taken on a bitstream missing setup, and
> constraints, flow, directives and the failing module are unchanged back to at least `618f4ce36`
> — so this build is not special, and very likely **no** silicon measurement this project has ever
> taken was made on a timing-clean bitstream. That is a separate finding, tracked in
> `agent-handoff/ref/RATE-RULE.md`, and it is not S-10.


> # SEVERITY SETTLED 2026-08-19, MEASURED TWO WAYS: A CAPABILITY SURVIVES THE STORE MEANT TO DESTROY IT
>
> The final framing is neither of the two above. **This is a failure to revoke by overwrite.**
>
> In the `stc G; plain store G+8` arm the plain store is the **younger** one, so architecturally
> it MUST clobber the granule and clear the tag. Usually it does. When it is dropped, **the
> capability survives an operation whose entire purpose was to destroy it.**
>
> That is the shape of a scrub: `memset(p, 0, sizeof(*p))` over a struct holding a capability,
> `explicit_bzero`, a free-list poison, clearing a slot before reuse. All are plain stores over a
> granule holding a capability, and all are how software destroys authority it no longer wants to
> hold.
>
> ### Measured directly, and cross-checked by two independent counts
>
> Arm `wf5`: `stc G`, then scrub `G+8` with a **distinctive non-zero pattern**
> (`0xD15CA5DBAD5C2BA1 ^ i` — a zero scrub could not be told from metadata that was already
> zero), then **read `G+8` back**. 3840 slots:
>
> | quantity | count | how it was measured |
> |---|---|---|
> | tags cleared (scrub landed) | 3664 (95.42%) | `lcc` type query == 7 |
> | **scrubs DROPPED** | **176 (4.58%)** | scalar readback of `G+8` != pattern |
> | surviving capabilities | 3840 − 3664 = **176** | independent of the readback |
>
> **The two counts agree EXACTLY: 176 = 176.** One counts capabilities that survived; the other
> counts scrubs that never landed. They are separate code paths measuring the same event, and
> they match to the slot.
>
> ### The correct severity
>
> * **NOT forgery.** No new authority is fabricated. The survivors carry the ORIGINAL
>   capability's `start`/`end`/`perm`/`cursor` — the CORRUPTED-BUT-TAGGED bucket is empty.
> * **NOT merely a dropped scalar.** A dropped **scrub** is retention of authority against the
>   program's explicit intent.
> * **IT IS a failure to revoke by overwrite**, measured at **4.58%** here and **9.95%** in the
>   arm without the readback loop (the readback adds traffic and changes buffer occupancy, which
>   the mechanism predicts).
>
> Weaker than fabricating authority. Stronger than losing a scalar. And the trigger is the
> standard C idiom for clearing a structure, which is why it is not exotic.
>
> ### What this does NOT establish
>
> Every capability observed surviving is one the program legitimately held moments earlier. This
> is **retention past intended destruction**, not creation of authority from nothing. Whether a
> retained capability is later reachable by code that should no longer have it is a
> software-lifetime question this test does not address.


> # SEVERITY CORRECTED 2026-08-19 — MEASURED. THIS IS NOT TAG FORGERY OVER SCALAR DATA.
>
> **Retracting this folder's original claim** that the defect "fabricates a usable capability over
> bytes the program wrote as scalars". It does not. That was inferred from a tag-only measurement
> and is **refuted by direct measurement** of the capability's fields.
>
> The `wbuf` kernel was extended to check `start`/`end`/`perm` (the at-risk metadata half) and
> `cursor` (a negative control) on every capability that survives, giving three buckets:
> LOST / INTACT / **CORRUPTED-BUT-TAGGED**. Measured on silicon, 3840 slots per arm:
>
> | arm | LOST | survivors | **CORRUPTED-BUT-TAGGED** |
> |---|---|---|---|
> | `wf1` plain `G+8`; `stc G` | 233 (6.07%) | 3607 | **0** |
> | `wf2` `stc G`; plain `G+8` | 3458 (90.05%) | 382 | **0** |
>
> **The corrupted-but-tagged bucket is EMPTY in both directions.** The 382 survivors in `wf2`
> carry the ORIGINAL capability's `start`, `end`, `perm` and `cursor` — not the plain store's
> scalar.
>
> ### What is actually happening, and it follows from the mechanism
>
> A capability entry writes **both** words of the granule (`wt_dcache_mem.sv:241-250`,
> `bank_req = '1`). So when the capability entry drains last it overwrites the whole granule,
> metadata included. The result is the **intact original capability** — and the plain store that
> was supposed to land on `G+8` is **silently dropped**.
>
> So the two consequences of the reorder are:
>
> 1. **plain entry drains last** -> the tag is cleared, the capability is destroyed, the program
>    faults on first use. This is **S-07** (availability).
> 2. **capability entry drains last** -> the capability survives intact and **the program's scalar
>    store never takes effect** (silent data loss).
>
> **Neither direction produces a capability over attacker-chosen data.** The soundness framing in
> the original text below is wrong and is retracted.
>
> ### This is still a real defect, and it is still worth its own folder
>
> A silently dropped store is a correctness bug in its own right: the program wrote a value, no
> fault was raised, and the value is not there. Any C code that writes a scalar field adjacent to
> a capability field in the same 16-byte granule can lose that write, at ~7% per opportunity. It
> is simply an integrity bug rather than a capability-model soundness hole.
>
> **A limitation of the current test, stated rather than left to be discovered:** it verifies the
> capability's fields but does **not** read back the scalar written to `G+8`. The dropped-store
> consequence above is inferred from the mechanism and from the empty corruption bucket, not
> measured directly. An arm that reads `G+8` back and compares it would measure it, and has not
> been built.
>
> **The fix-validation value of the extended test is unchanged**, and is the reason it exists: a
> candidate fix that propagates the youngest tag WOULD populate the corrupted-but-tagged bucket,
> and the bucket now reads 0 on unfixed hardware — so it is a working detector with a
> demonstrated negative, ready to catch exactly that mistake.


**A capability store and a plain store to the same 16-byte granule are reordered, so a plain
store's data can end up carrying a VALID CAPABILITY TAG.** Measured on silicon at **7.27% per
opportunity**.

This is a **soundness** defect, not an availability one: it fabricates a usable capability over
data the program wrote as scalars. It is not a denial of service — it is the capability model
failing open.

> **Sibling issues, so a reader with the wrong symptom is redirected immediately.**
> `S07-capability-untagged-on-reload/` is the **opposite direction of the same mechanism** — a
> capability's tag is LOST, the program faults with mcause 25, and the domain wedges. Same root
> cause, same fix, different consequence. `S06-untagged-ldc-stc-high-half/` is a distinct,
> already-fixed defect about untagged `ldc`/`stc` and is not this.

## The defect

The write buffer hits at **64-bit word** granularity, so the two halves of a 16-byte granule
occupy **separate entries** — but every entry writes the **whole granule's single tag bit** when
it drains, and drain order is `rr_arb_tree` rotation, **not program order**.

    wt_dcache_wbuffer.sv:444   hit compare on wtag == {address_tag, address_index[11:3]}   -> 8-byte
    wt_dcache_wbuffer.sv:410   wr_idx_o  = wr_paddr[11:4]                                 -> 16-byte
    wt_dcache_wbuffer.sv:416   wr_ctag_o = wbuffer_q[rtrn_ptr].ctag
    wt_dcache_mem.sv:459       cap_tag_q[wr_idx_i][j] <= wr_ctag_i
    wt_axi_adapter.sv:158      tag byte address = (paddr - DATA_MEM_BASE) >> 4

So an older `stc` to `G+0` can drain **after** a younger plain store to `G+8`, and the tag the
plain store should have cleared stays set — over the plain store's data.

### MECHANISM CORRECTED 2026-08-19 — it is not tag ordering, and that matters for the fix

The description above ("the loser's tag wins") is a **symptom**. The root is that
**`is_cap` entries span TWO words but are tracked and merged as ONE**:

    wt_dcache_mem.sv:241-250
      if (!(st_wr_cap)) begin
        bank_req |= dcache_cl_bin2oh(wr_off_i[...]);   // ONE word
        bank_we  =  dcache_cl_bin2oh(wr_off_i[...]);
      end else begin
        bank_req = '1;                                  // BOTH words
        bank_we  = '1;
      end

A capability entry writes **both** words of the granule; a plain entry writes one. They overlap
on the high word — the **metadata** half. But the write buffer's hit/merge compare is on the
**word** address (`wt_dcache_wbuffer.sv:444`), so it cannot see that a cap entry at `G+0`
already covers `G+8`. Two entries end up writing the same physical word in arbitrary order.
The tag disagreement is a consequence of that, not the disease.

**THE OBVIOUS FIX IS WRONG, AND WRONG IN THE DANGEROUS DIRECTION.** Propagating the youngest
store's tag to any co-resident same-granule entry — so drain order stops mattering — leaves the
older plain entry still writing its stale scalar over the **metadata** half. Giving it `ctag=1`
as well would produce a capability with a **valid tag over corrupted metadata**, converting the
loss case into a forge case. It would have turned an availability bug into a soundness bug, and
**the directed test would have gone green, because it only checks the tag.**

**That is a limitation of the test in this folder and it is stated rather than discovered
later:** `wbuf_kernel.h` verifies the TAG via `lcc` field 1 and does **not** verify the
capability's bounds, permissions or cursor. It can distinguish tagged from untagged. It cannot
distinguish a correct capability from a tagged one with corrupted metadata. Any candidate fix
must be validated against metadata integrity, not against this test alone.

**Two real options, neither built yet:**

* **(A) make the merge granule-aware for capability entries** — a plain store to `G+8` merges
  INTO a resident cap entry at `G+0`, its bytes routed to that entry's metadata half, `ctag`
  last-writer-wins. One entry, one writeback, order-independent. Correct and complete, but real
  surgery on the merge path.
* **(B) forbid co-residency** — refuse to allocate an entry that conflicts at GRANULE level with
  a resident entry when either side is `is_cap`, stalling until it drains. Small and obviously
  correct, but it adds a term to `rdy`, which feeds the grant path, and that cone is on the
  standing `UNOPTFLAT` list where synthesis has twice gone pathological.

**The `wb3` arm is the evidence that (B) works:** 64 unrelated stores between the pair gave 0
losses out of 16384 — co-residency broken by drain rather than by design.

Choosing between them trades correctness completeness against synthesis risk in a cone that has
already cost two blowups, and the forgery arm proves this subsystem can produce soundness
failures. That is a design decision for the project lead, not a late-session patch.

## Reproduction

`capstone/tests/runtime-qemu/silicon-ladder/wbuf_kernel.h`, arm 2, staged as `wb2.dom`:

    per slot, back to back:
        wbuf_slots[i] = base;        /* stc to G+0  */
        *(volatile unsigned long *)((char *)&wbuf_slots[i] + 8) = value;   /* plain store G+8 */
    then reload and query the type with lcc field 1

`lcc` field 1 is the TOTAL type query, which returns 7 for NOT_CAP **without raising**
(`capstone_dyn_unit.anvil:195`), so the run always returns a number and never wedges.

**Measured, 16384 slots, one boot:**

    expected (program order): 16384 tags cleared by the trailing plain store
    observed:                 15193 cleared
    ** 1191 slots (7.27%) kept a VALID TAG over scalar data **

## Controls, in the same boot

| arm | sequence | lost | meaning |
|---|---|---|---|
| wb0 | `stc G` only | 0 | the round trip alone is clean |
| wb4 | plain `G+16`; `stc G` | 0 | effect is granule-scoped, not "a nearby store" |
| wb3 | plain `G+8`; **64 stores**; `stc G` | 0 | **draining the buffer removes the effect** |
| wb1 | plain `G+8`; `stc G` | 1107 (6.76%) | the opposite direction — see S-07 |

**wb1 vs wb3 is the decisive pair:** identical stores, identical granule, differing only by
intervening traffic that drains the buffer. 1107 versus 0.

## QEMU cannot reproduce it, by construction

QEMU's capability store is one atomic 16-byte-plus-tag operation with no write buffer, no
per-word entries and no drain arbiter. Arms 0/1/3/4 return zero loss under emulation. **The
silicon/emulator difference is the mechanism**, and no amount of emulator testing could have
found this.

## Why it matters more than the availability direction

A lost tag faults loudly and stops the program. A **forged** tag does not: it produces a
capability the hardware will honour, over bytes the program wrote as ordinary data, with no trap
and no indication. Any code that writes scalars into a granule adjacent to a capability field can
manufacture one.

## Status

Mechanism identified by the RTL lane from the sources above; confirmed on silicon by the directed
test in this folder. The fix is open. Whether the `ctag`/`cap_tag_q` path predates the S-06 work
is under independent audit and does not change the defect.

## A note on how the number in this folder was obtained

**The 1191 was a POSITIVE CONTROL that turned into the finding.** `wb2` exists only to prove the
detector can report a loss at all: `stc G` then a plain store to `G+8`, which in program order
legitimately clears the tag, so all 16384 slots were expected to lose it. 15193 did. The 1191
shortfall was treated as **signal rather than noise**, and it is the entire evidence for the
forgery direction.

Recorded because it is the exact inverse of the failure mode this investigation spent a day on —
a check that cannot fire, or a clean result from an instrument that never created its triggering
condition. Here an instrument built to prove itself disagreed with its own oracle, and the
disagreement was the result.
