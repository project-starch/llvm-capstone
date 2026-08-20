# S-10 — a scrubbed capability still reads back LIVE while the store is in the write buffer

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


**A plain store that clears a capability's tag is not visible to a load of that capability until
the store drains.** For the whole time the store sits in the write buffer, `ldc` returns the
capability as if it had never been scrubbed. Measured on silicon at **3837 of 3840 — 99.92%**.

> **Sibling issues, so a reader with the wrong symptom is redirected immediately.**
> `S07-capability-untagged-on-reload/` is the tag-reordering defect and is **FIXED** by forbidding
> granule co-residency. **This issue is NOT that defect and is NOT repaired by that fix**, which is
> an allocation-time check between two write-buffer *entries*; this residual needs only **one**
> entry, so the check never fires and a load never consults it. `S09-.../` is the dropped-scrub
> consequence of S-07 and is also fixed.

## The defect

    stc  G, cap      drains to L1, cap_tag_q[G>>4] = 1
    sd   x, G+8      ONE plain entry, word 1, STILL RESIDENT in the write buffer
    ldc  G           granule-aligned, so it compares WORD 0, misses the word-1 entry,
                     and falls through to the STALE cap_tag_q  ->  returns a LIVE CAPABILITY

Mechanism in `wt_dcache_mem.sv:280`, `:301`, `:337`.

## Measured on silicon

Bitstream `caplifive_s07fix.bit`. Two arms differing by **exactly one thing** — the delay between
the scrub and the check. Verified in the disassembly that the only instruction-level difference is
the drain loop. 3840 slots per arm.

| arm | sequence | live capability after its own scrub |
|---|---|---|
| `wr6` | scrub, then type-query **immediately** | **3837 / 3840 (99.92%)** |
| `wr7` | identical, **+ 300-iteration drain** | **0 / 3840** |

**Probe control, in the same run, on both arms:** the delayed type-query loop reported `NOT_CAP`
**3840/3840**. So the query is proven to fire on this bitstream in this binary, which is what makes
`wr7 = 0` a real negative rather than a dead instrument.

The probe is the **type query** (`lcc` field 1, which returns 7 for NOT_CAP *without raising*), not
a scalar readback: the defect is `ldc` returning a live capability, so the capability is what must
be interrogated.

## It is PRE-EXISTING — measured, not inferred

Same binaries, pre-fix RTL (`a3dbae618`) in a worktree against the fixed model:

| test | pre-fix `a3dbae618` | fixed |
|---|---|---|
| `s07-wbuf-forward-residual` | 9 exc / 9234 cyc | 9 exc / 9234 cyc |
| `s07-wbuf-forward-residual-ctl` | 17 exc / 26361 cyc | 17 exc / 26361 cyc |

Identical: **the fix neither repairs nor worsens it.**

**Model-identity control**, because identical numbers are also what running the same model twice
produces: `s07-wbuf-tag-reorder` in that worktree gives **4 exceptions at 9150 cycles** — the
pre-fix signature — against **1 at 9138** on the fixed model, plus zero occurrences of
`gran_conflict` in the worktree source. So the worktree really is the pre-fix RTL.

## Severity — transient, but not an edge case

The window closes completely once the entry drains (`wr7` = 0/3840). But **while the entry is
resident the failure is very nearly certain**, at 99.92%. Short-lived is not the same as unlikely.

**Operational consequence:** a program cannot trust an immediate re-read to confirm it has
destroyed a capability. It can trust the destruction once the buffer has drained.

Contrast with S-07's dropped scrub, which was **persistent** — there the capability survived
indefinitely. This one is bounded by drain time.

## What this folder does not establish

Whether any real workload reads back a capability quickly enough after scrubbing it to observe the
window. The measurement is a directed test built to hit it deliberately; no naturally-occurring
instance has been identified.

Do not quote the Verilator figure (8 of 16 legs) as a rate — the trap handler drains the buffer
between legs and resets the phase, which is what made that pattern alternate. The silicon number
is the magnitude.
