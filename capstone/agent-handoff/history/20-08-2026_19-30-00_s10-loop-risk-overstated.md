# The S-10 loop risk was overstated, and one artefact was misquoted

**Date:** 2026-08-20
**Applies to:** `capstone-ariane` `4fee13b2d`, merged at `3d3ed1502`
**Effect:** the synthesis run is still the right call, but for weaker reasons than were given.

## What I said, and what is actually true

The S-10 fix commit and the report to the project lead both said the loop is *"Verilator naming
`wt_dcache.rd_ctag`"*, and framed the change as the category CLAUDE.md calls the highest-risk
edit available — *a new signal fed into a cone that already carries a combinational loop*.

**Both statements need correcting.**

> ## CORRECTION TO THIS NOTE — the same day, one hour later
>
> **Section 1 below is WRONG and is withdrawn.** A lint run pinned byte-exactly to HEAD names
> `cva6.gen_cache_wt.i_cache_subsystem.i_wt_dcache.rd_ctag` — the parent-level net — which is what
> the original report said. The "brand-new internal signal, weaker signature" reading is dead.
>
> **How the mistake happened, because it is the same shape a fourth time.** Three lint captures
> existed and *none of them was HEAD*: each was taken of a different intermediate formulation
> during the search for one that avoided the loop. `.lint-withfix` named `wbuffer_gran_clr` at
> `wt_dcache_mem.sv:139:9`, and HEAD's source really does have `logic wbuffer_gran_clr;` at line
> 139 column 9 — so the location matched exactly, and the match was a coincidence: the later edit
> (removing the `2'd3` probe arm) sits *below* line 139 and did not move it. A line:col match was
> treated as a byte-pin. It is not one.
>
> **The rule this yields, and it is narrow enough to be usable:** an artefact is authoritative for
> a tree only if it was produced *from that tree*. Location matching is not provenance. The two
> intermediate captures have been deleted so they cannot be re-read as HEAD.
>
> **What does NOT change.** Sections 2 through 5 stand, and section 2 is the load-bearing one: it
> is an argument about `rd_ctag_o` specifically, so it applies directly to the signal that is
> actually named. The cone's source set is unchanged; 30 combinational loops already exist in a
> design that completed and routed; the shape differs from the precedents; the cost is ~88 LUT6.
> **The GO stands, on section 2 rather than on section 1.**
>
> Also confirmed by the pinned run: `GONE at HEAD: []` — no pre-existing UNOPTFLAT entry
> disappeared, so no existing strongly-connected component was restructured. Exactly one added.

### 1. The misquoted artefact — WITHDRAWN, SEE THE CORRECTION ABOVE

Three formulations were linted. The one that is at HEAD is the explicit granule slice, captured in
`verif/sim/.lint-withfix.txt`, and it names:

```
wt_dcache_mem.sv:139:9  Circular combinational logic: '...i_wt_dcache_mem.wbuffer_gran_clr'
```

HEAD's source has `logic wbuffer_gran_clr;` at **line 139, column 9** — an exact location match.
`.lint-withfix2.txt`, the capture that names `wt_dcache.rd_ctag`, contains **zero** occurrences of
`wbuffer_gran` and is a *different* formulation that is not at HEAD.

So the flagged signal is **a brand-new internal signal inside one leaf module**, not a
parent-level net being pulled into a cycle. That is a materially weaker signature than was
reported.

### 2. The cone's source set does not change

At `618f4ce36`, `rd_ctag_o` is already

```systemverilog
assign rd_ctag_o = (|wbuffer_be) ? wbuffer_data_i[wbuffer_hit_idx].ctag : rd_ctag;
```

`wbuffer_hit_idx` is a **dynamic index**, so all eight entries' `.valid`, `.wtag` and `.ctag` are
already in `rd_ctag_o`'s fan-in, as is `wbuffer_cmp_addr`. `wbuffer_gran_clr` depends on exactly
that set and nothing else. **The cone's source set is unchanged; only the function over it
changed, and no signal crosses a module boundary into it.**

The CLAUDE.md rule is about a signal *crossing a module boundary into* a looping cone. That is
not what this does, and the rule should not have been invoked as though it were.

### 3. Combinational loops are not what blows synthesis up here

From the routed design of the build that **completed and routed** (`ariane.check_timing.rpt:81`):

```
9. checking loops (30)
 There are 30 combinational loops in the design. (HIGH)
```

Vivado already sees **thirty** combinational loops in a design that synthesised in 1h48m. Loops
per se are demonstrably not the mechanism behind the 100 GB / 343 GB incidents.

### 4. The shape is not the shape of the precedents

`d65c67589` (>100 GB) was `core/cva6.sv +166` plus cross-module plumbing through `ex_stage`,
`load_store_unit`, `load_unit` and `store_unit` into a **top-level debug LED mux** — a large new
cone for retiming to search. S-10 is 62 lines confined to one leaf module with an unchanged cone
boundary. Not the same shape.

### 5. The cost is negligible

`WtDcacheWbufDepth = 8`, `WBUF_MEM_WTAG_W = 53`. The new comparator is a **52-bit equality × 8**,
one bit narrower than the existing 53-bit `wbuffer_hit_oh` array and **parallel to it, not stacked
on it**, plus eight ANDs, an OR-8 and one mux term. Roughly 88 LUT6 against `i_wt_dcache_mem`'s
1206 LUTs and a top-level 169415 / 203800. About 0.04% of the device.

## What the run is actually for

The delta against the last build that **completed** is one file:

```
$ git diff --name-status e1140aeea HEAD -- core/ corev_apu/
M   core/cache_subsystem/wt_dcache_mem.sv
```

`run.tcl` is unchanged from that build, so `RETIMING true` is the same setting that completed, and
`wt_dcache_wbuffer.sv` is byte-identical. **A genuine single-variable run.**

The residual risk that survives all of the above: retiming is a global register-movement search,
and `rd_ctag_o`'s cone gained a second comparator array and one more mux level inside a cone that
participates in a loop. Vivado excludes loops from retiming, but the loop-detection pass runs over
a slightly larger cone than before. That cannot be bounded statically.

## Kill criteria, and post-run checks that can falsify the reasoning above

**Kill the run if** `synth_design` RSS climbs past roughly **2× the reference envelope**
(~13 GB PSS / ~21 GB tree-summed at `e1140aeea`) while still rising, or wall time exceeds ~2× the
1h48m reference with RSS not plateauing. Both are far below the 100 GB / 343 GB incidents, so the
40 GB guard ceiling has wide margin.

**Afterwards, check these — each can prove the reasoning wrong:**

1. `check_timing` loop count should still be **30**. The analysis above predicts no new netlist
   cycle, only one new member of an existing SCC. **If it returns 31, item 2 is wrong** and the
   result must be re-audited before anything is concluded from the bitstream.
2. `i_wt_dcache_mem` should rise from **1206 LUTs** by roughly 100. A rise of thousands means the
   comparator did not synthesise as predicted.
3. Compare combinational-loop warnings (`Synth 8-295`) against the reference run's set.

## Two things that are not clearance

* **The lint gate is RED at HEAD by design.** `verif/sim/rtl-lint.REF.txt` holds `UNOPTFLAT 39`
  against HEAD's 40. That is a deliberate merge block. **HEAD must not be merged on the strength
  of a green synthesis alone** — the baseline is only updated once the run has actually shown the
  loop is harmless.
* The functional soundness of the bare-OR reduction rests on the S-07 `gran_hazard` stall, which
  the *correctness* audit covered separately and supported. Nothing here speaks to it.

## Hygiene

`/tmp/capstone/_bitstreams/d.out` and the archived `ariane.check_timing.rpt` both contain a shell
prompt and a `Host:` line carrying a personal account name and a build hostname. **Neither may be
committed or pasted into shared content.** They live under `/tmp/capstone/` and must stay there.
