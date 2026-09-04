# The S-10 fix closes one route to the defect, not two

**Date:** 2026-08-20
**Fix audited:** `capstone-ariane` `4fee13b2d`, merged at `3d3ed1502`
**Status of the fix:** correct for what it covers, and **PARTIAL**. Do not record S-10 as fixed.

## What the fix does close

The write-buffer route. `wt_dcache_mem.sv` looked the capability tag up through
`wbuffer_hit_oh`, which compares at 64-bit **word** granularity, while the tag is a per-16-byte
**granule** property. A granule-aligned `LDC` always compares word 0, so a resident plain store
at word 1 was invisible and the load fell through to the stale `cap_tag_q`.

Measured, with the matched pair that already existed:

```
s07-wbuf-forward-residual        9 exc -> 17   (8 of 16 legs -> 16 of 16)
s07-wbuf-forward-residual-ctl   17 exc -> 17   (control, unchanged)
```

## What it does NOT close — confirmed by audit, verified independently

**A store that has not yet reached the write buffer is invisible to a write-buffer fix.**

The only RAW interlock between a load and a not-yet-drained store is `page_offset_matches_o`,
and it compares at **word** granularity in all three of its comparisons:

```
core/store_buffer.sv:279   page_offset_i[11:3] == commit_queue_q[i].address[11:3]
core/store_buffer.sv:287   page_offset_i[11:3] == speculative_queue_q[i].address[11:3]
core/store_buffer.sv:293   page_offset_i[11:3] == paddr_i[11:3]
```

The load side feeds it the full page offset (`core/load_unit.sv:309`) and stalls only on that
signal (`:398`, `:439`, `:519`). An `LDC` at granule base G presents `[11:3]` of word 0; a plain
`sd` at G+8 presents `[11:3]` of word 1. **They do not match.** The load proceeds, the store is
still in the store buffer, `wbuffer_data_i` does not contain it, and `wbuffer_gran_clr` cannot
see it. Same symptom, different route.

Why it is reachable rather than theoretical: it opens whenever write-buffer allocation is
stalled — `full` (`wt_dcache_wbuffer.sv:496`) or the S-07 `gran_hazard`/`ni_conflict` gate
(`:722`) — which parks a committed store in the commit queue while loads continue.

### The consequence for the evidence we have

**The 16/16 result cannot have exercised this route.** A leg whose store was still in the store
buffer at compare time would not have trapped. All 16 legs landing on the write-buffer route is
consistent with the fix being complete *for that route only*.

The header of `verif/tests/custom/capstone/s07-wbuf-forward-residual.S` claims it "measures the
composite architectural window" and cites `store_buffer.sv:279/287` while doing so. **That
overstates what the number supports** and should be corrected when the test is next touched.

## Why this is not being fixed in the same change

The correct fix is a **granule-granular** hazard compare for capability loads specifically —
`[11:4]` rather than `[11:3]` when the load reads a whole granule. Widening it unconditionally
would over-stall ordinary loads that have no hazard, so it needs an "is capability load" signal
routed into `store_buffer`, i.e. a cross-module port change into the LSU hazard path.

That is deliberately **not** bundled here:

* it is **pre-existing**, not introduced by this fix, and the fix is a strict improvement;
* it would invalidate the two audits already run against `3d3ed1502`;
* the LSU hazard path feeds the same `req_port_o` ring that already cost this fix
  `UNOPTFLAT 39 -> 40`, so a second change there is not a small one.

Track it as its own item. Suggested handle: **S-10b, store-buffer-resident route**.

## Acceptance work still missing for S-10 itself

**No test on either side has a forced-eviction reload leg.** The board lane confirmed `wr6`/`wr7`
drain by cycling a 512-byte `wbuf_sink`, which cannot evict a line from a 32 KB 8-way cache. Both
arms therefore read back while the line is still resident, and **nothing yet distinguishes a real
fix from one that only repairs L1 while DRAM keeps the stale tag** — which is precisely the
signature the twice-sampled `ctag` produces in this subsystem. The board lane is writing `wr8`
(stride-4096 eviction, chosen because the cache is physically indexed on `paddr[11:4]` and a
4 KiB stride preserves the set index under translation).

That arm should exist before any bitstream is treated as validating S-10.

## Two smaller items from the same audit

* `ctag_implies_cap` (`wt_dcache_wbuffer.sv:798-822`) is inside `` `ifndef VERILATOR ``, so the
  invariant the bare OR depends on holds **by source inspection only** — the 16/16 is not
  evidence for it. Closing it needs a simulator that evaluates SVA, plus a negative test forcing
  `data_wtag=1, data_is_cap=0`.
* The S-10 granule slice hardcodes the one-bit drop as a literal, where the sibling S-07 fix
  *derives* it (`WBUF_WORD_BITS = DCACHE_OFFSET_WIDTH - XLEN_ALIGN_BYTES`) and says why. Both are
  correct only while that expression equals 1, and nothing asserts it at elaboration. True for
  every config actually built today.
