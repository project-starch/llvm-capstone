# Sublet Section 5, phase 2: the record cache (bitstream B5a)

Status: RTL draft and pre-registration, 2026-11-16. Every reading below carries **key 5 NOT MET** (P1's bitstream
missed key 5; B5a is built on it).

The lead asked on 2026-11-16 for the remaining Section 5 RTL to start at once, through the runtime lane. The
phases come from the Section 5 plan (spec-silicon `capstone/docs/plans/2026-11-15-sublet-section5-rtl.md`).

## Order

- **B5a**: the record cache alone, on P1 (capstone-ariane d83d3a633). It is the one part that changes performance, and
  it is measured against P1 with nothing else moving.
- **B5b**: the 63-bit format, the 128-bit record, single-operand CREVOKE, CRENEW and the deletions, in one bitstream.
  The external collaborator's kernel needs Phases 3 and 4 together.
- Each step gets a synthesis-only gate under synth-guard (40 GB) before its full build.

## B5a design

- Paper Section 5 "Lookup": 2 ways, 8 KiB of records, 64-byte lines of four 16-byte records, LRU.
- Keyed by slot: line `slot[24:2]`, set `slot[7:2]`, tag `slot[24:8]`, record `slot[1:0]`.
- Placed between the rev unit's two D$ channels and P1's walker (`core/ex_stage.sv`).
  - A hit needs no walk, no record-TLB lookup and no D$ read.
  - A miss goes to the walker exactly as on P1.
- Active only INSIDE A CONTEXT, and emptied on every edge of it. Two flags are both required: `ctx_rev_q`
  (`vctx_i` held with the namespace, the one the unit's operation sees) and `rc_live_q` (`vctx_i` sampled every
  cycle). `ctx_rev_q` alone resamples only while the unit is idle, so an operation straddling an escape could keep
  it at 1 into the S code after the escape (the claim-auditor's finding on a38fddb70). `vctx_i` falls when the
  escape's switcher starts, before any post-escape instruction. Outside a context, in the table namespace (S under srevroot, M under csupctl.table) and in the physical
  pool, the path is P1's: no lookup hits and nothing installs. (The first cut cached in the whole table namespace;
  see "Correction" below.)
- **Inside a context the cache accepts every unit read and write from its own flops.** Only its registered request
  reaches the walker and the D$. The unit's address selector therefore leaves the D$ request path there, the same
  concern behind P1's last commit (no slot compare on the D$ request path).
- One operation at a time, held to completion; write first when both are presented, as the walker does.
  `rc_busy` holds the namespace while the cache owns an operation.
- **Coherence.** Software touches the table in four places, read at source in spec-silicon `capstone_vm.c` and
  confirmed by the runtime lane:
  - the monitor's emulated CSCHECKR/W, which reads records;
  - the module's header reads;
  - grow, which writes the header capacity and directory entries;
  - pool reuse, which zeroes a released table.

  The rules that keep all four exact:
  - slots 0 and 1, the header, are never cached;
  - records are cached only inside a context, and the cache empties on the context edge;
  - every namespace flush empties the cache in one cycle, and an operation in flight across a flush installs nothing.
    A flush is a vctx or srevroot edge, the csupctl bit 2 strobe, a dropped unreachable write, or the context edge;
  - a DEAD (unreachable) walker response is never installed, so a slot a grow makes reachable is read fresh;
  - the cache is write-through (below).

  **Why "inside a context" is exact: every writer of a record, enumerated.**
  - **The module (runtime lane's enumeration, `capstone_vm.c` of the bkit91 kernel):** its only record-slot stores
    are memsets of batches. Those are fresh batches before a directory entry links them (`:815`, unreachable, and
    DEAD is never installed), pooled tables in `context_create` before the new context's srevroot write (`:925-940`),
    released tables after the context ended (`:946-995`), and fresh flat tables (`:1025`). `:660` and `:3706-3716`
    write the header and root words. All of them run in the kernel, after an escape.
  - **No kernel or monitor code runs with a context supervised.** Under sup_active every commit-time exception and
    interrupt, except a debug request, becomes an escape (`commit_stage.sv:311-321`). While the escape or the switcher is pending, the
    exception is stripped (`:1127-1129`), so no trap reaches the CSR file's trap entry, and medeleg/mideleg are
    never consulted. The escape's handshake raises `sup_clear` (`:950`), which ends supervision and runv
    (`csr_regfile.sv:2153-2156, 2181-2186`). The switcher's own walks run with `vctx_eff` low (`cva6.sv:843`).
    Two causes do not escape: the R-43 replay marker (a re-execution, no handler) and debug-mode entry, which this
    stack does not use.
  - **A context's own code does not address the table.** This is the software's layout (the table is kernel memory,
    not mapped to the context), not an RTL check. QEMU would see a context's store into its own table; this cache
    would not.
  - **Not covered:** debug mode inside a context (it runs with `vctx_i` = 1, as on P1; this stack does not use it),
    AMOs, and other bus masters (R-73).
  - **The monitor's CSCHECK emulation** runs in M after an escape, so it is uncached and reads memory, which
    write-through keeps current.

### Correction (2026-11-16): the first cut cached outside contexts

The first cut (8260ef930..46fb0ab83) cached in the whole table namespace (`vctx_table_o`: a context, S with srevroot
set, or M with csupctl.table, `csr_regfile.sv:2941-2943`). This section then listed the software READERS and the
kernel's writers. It did not ask who can store a record in each mode the cache was active in.
- va-runv6-w runs in M under csupctl.table. A plain `sd` rebuilds a record (slot 7, 0x80006070, cycle 12289), and
  the next CSRETIRE hit the stale line and was refused: rd 1, where P1 reads 0. The 165-test table-namespace pass
  found it in the retired (RVFI) trace.
- The sim oracle of 46fb0ab83 could not see it. Its shadow was fed only by the cache's own installs and updates.
- The oracle now also takes MEMORY as the reference: the physical address each record was read from, and every
  store-unit write to the D$ (port 2). A read hit on a record a store touched since its install is `rc_stale_hit`.
- With the oracle applied to 46fb0ab83 (an uncommitted tree) it fires on va-runv6-w (slot 7, pa 0x80006070). The
  fix's control is `va-rc-ctx` (below).
- The fix came in two commits:
  - a38fddb70 gated caching on `ctx_rev_q`;
  - 7ba62edde added `rc_live_q`, after a claim-audit found the straddle window above. The same commit closed an
    oracle gap: a store racing a record read between its grant and its install was lost. It also fixed a38fddb70's
    lint FAIL (UNUSEDSIGNAL 1483 against 1480, the sim-only oracle's unused low bits).
- No hash had gone to synthesis.

### Recorded deviations from Section 5

1. **Write-through, not write-back.**
   - Every record write goes to the table exactly as on P1. A write hit updates the line; a write miss allocates
     nothing.
   - The software readers, above all the monitor's CSCHECK emulation (`capstone_cscheck.S`, about 1k-4k checks per
     SPEC run), therefore read what they read on P1. No drain or stall is needed at an edge.
   - A write-back cache needs a drain at every edge and a registered stall that orders later software reads after
     it. That stall falls in the commit or CSR path (Section 5 plan, synthesis risk 5).
   - A first proposal used a firmware srevroot 0-and-back toggle as the drain. It was withdrawn: `srevroot_rev_q`
     samples only while the unit is not busy (`ex_stage.sv:1693-1697` at d83d3a633), so the toggle can be missed,
     and nothing ordered the emulator's loads after the drain.
2. **Behind the LSU's 64x4 verdict cache, not in place of it.**
   - The record cache serves the unit's record reads: the verdict cache's probes and CDERIVE's and CREVOKE's walks.
   - The access path's lifetime compare (Section 5 plan, synthesis risk 1) is therefore unchanged.
   - Whether this is acceptable for the paper is the lead's call; the runtime lane has put the question.
3. **Whole-line fill, through the unit's 16-byte port, yielding to the unit.**
   - The paper's cache fetches the 64-byte line.
   - Here the demanded record is read first and returned at once. The line's other records follow, one D$ read at a
     time, in the background.
   - The fill stops when the unit presents its next request, so an allocation's read-then-write does not wait.
   - A first cut (e8fa8593a) filled by demand only, one record per miss. The model below showed it would have measured
     nothing.

### HPM events (new)

| Event | Meaning |
|---|---|
| 34 | record-cache read hit |
| 35 | record-cache read miss (slots 0 and 1 are not looked up) |
| 36 | record-cache write hit (the line is updated; the write still goes through) |
| 37 | background fill read issued (one per record the fill fetches beyond the demanded one) |

Events 29-33 keep P1's meanings. `rev_walk_perf` widens from 4 to 8 bits (`core/cva6.sv`, `core/perf_counters.sv`).

## The instrument: a QEMU model of the record cache

- capstone-qemu working tree `qemu-rvcm` (533163eb25 plus the RVCM hook of 2026-11-08, uncommitted), extended
  2026-11-16 with an RCM model.
- It is fed by every 64x4 verdict-cache MISS in a context, from RVCM's own model of the verdict cache.
- RVCM is the validated half: it predicted mcf test's 64x4 misses at 180,583,760, against the board's ev23 of
  180,631,581 (0.03 %).
- Geometries: the paper's 64x2 with 4-record lines, plus 64x4, 128x2, 1024x2 and a 256x2 with 1-record lines (the
  same 8 KiB with no spatial reuse). It also counts distinct lines per flush window.
- Run directory: `perf/rcm` (mcf test, gcc test, perl test).

Readings, with the instrument control: RVCM's 64x4 misses on mcf test reproduce exactly, at 180,583,760, the
board-matched figure. The RCM model's lookups equal those misses on both rows.

| row | record reads (64x4 misses) | 64x2, 4-record lines (the paper) | 64x4 | 128x2 | 1024x2 | 256x2, 1-record lines | distinct lines per window, summed |
|---|---|---|---|---|---|---|---|
| mcf test | 180,583,760 | 47,945,563 misses (73.4 % hits) | 46,447,084 | 46,151,465 | 1,843,053 | 177,756,068 (1.6 % hits) | 1,843,053 over 32,512 windows (max 1,594) |
| gcc test | 27,319 | 19,082 (30.2 % hits) | 19,077 | 19,078 | 19,077 | 25,710 (5.9 % hits) | 19,077 over 2,160 windows (max 60) |
| perl test (specdiff FAIL on this QEMU run: no grade, context only) | 75,811,206 | 45,444,800 (40.1 % hits) | 36,320,666 | 38,300,388 | 32,298,455 | 51,676,569 (31.8 % hits) | 32,202,158 over 831,945 windows (max 4,096) |

- The reuse is spatial: 1-record lines of the same capacity hit on 1.6 % of mcf's record reads, 4-record lines on
  73.4 %.
- gcc test's misses are almost all compulsory: the first touch of a line in a window.
- Capacity beyond 8 KiB buys little until it holds a whole window (1024x2 reaches the compulsory floor).
- The model fills every line completely. The RTL's fill yields to the unit, so its hit count is at most the model's.

## Pre-registration

The hash is capstone-ariane `rtl/record-cache` **6e82d7bd2**. Its synthesised logic is 7ba62edde's: 91699f266's
cache with the context gate (`ctx_rev_q` and `rc_live_q`) and the context-edge flush. 6e82d7bd2 differs from
7ba62edde only inside `ifndef SYNTHESIS`: the oracle's `rco_idx_q` holds index bits [11:4] instead of [11:0].
`verilator -E -P -DCAPSTONE_ITFC_EN -DSYNTHESIS core/ex_stage.sv` gives identical text on both commits (1,761
lines, sha256 2fbb1f109074e593..., `rc_live_q` present). Without `SYNTHESIS` the two differ: that is the control.
7ba62edde's lint was UNUSEDSIGNAL 1481 against 1480, and that register's low bits were the one new name. The wide
pass and the sweep below ran on 7ba62edde, and the oracle computes the same key on both commits.

### Pre-synthesis gates (sim)

- **Lint gate:** PASS at d83d3a633's baseline. UNOPTFLAT 39 with identical loop names, UNUSEDSIGNAL 1480, LATCH 44,
  MULTIDRIVEN 3. Read on 6e82d7bd2 (`.sup/lint-rc9`, dirty=no): all counters at baseline: LATCH 44, MULTIDRIVEN 3,
  UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2, SELRANGE 100, ALWCOMBORDER 0, COMBDLY 0,
  ANVIL_UNOPTFLAT 0.
- **17 testlist_sup arms:** CAPPRINT readings identical to d83d3a633's by value.
  - `va-alloc-rcflush-w` reads the control vector.
- **Positive controls**, each in a context, where the cache is active:
  - `va-rc-ctx` (the context-edge arm) reads 3, 0x1000000000000040, 0x1000000000000045 on P1 and on a38fddb70, with
    retired traces identical;
  - `RC_SIM_NOCTXFLUSH` trips the memory oracle there (`rc_stale_hit`, slot 2);
  - `RC_SIM_NOINV` trips `rc_hit_mismatch` there (its earlier trigger, va-alloc-rcflush-w, runs in M and no longer
    caches);
  - `RC_SIM_BADFILL` trips `rc_addr_mismatch` (slot 3) on `va-rc-ctx` and on `va-fetchsplit`;
  - the memory oracle, applied to 46fb0ab83, fires with va-runv6-w (slot 7, pa 0x80006070);
  - all three mutants were run on 7ba62edde's RTL, before the commit.
  - On the handed hash, 6e82d7bd2 (`rc9ctx`, `rc9m1`, dirty=no):
    - va-rc-ctx, va-fetchsplit and va-runv6-w retire traces identical to their 7ba62edde runs;
    - hits are 4 of 11, 31 of 52 and 0 of 80;
    - the oracle is silent;
    - `RC_SIM_NOCTXFLUSH` trips `rc_stale_hit` slot 2 (pa 0x80009020) on va-rc-ctx, so the narrowed key still
      fires.
- **A/B grading is on the retired trace (RVFI `.log`) or on values printed by retired instructions.** It is never on
  the FLU's CAPPRINT `$display`, which fires at execute: six of the seven cmp.py differences on 46fb0ab83 were that
  print on squashed paths after an excepting LDC, with retired traces identical.
- **95-test sweep** against P1's `pwsw6` (d83d3a633): on 7ba62edde (`rcsw5`, dirty=no, SWEEP-RAN 95, 92 logs),
  92 identical in values, 92 identical in retired traces (26 of them parked), and 0 of 92 differing in cycles. The
  same 3 tests (rand, perf_tcache_random, perf_riscv_random) are missing on both sides. The sweep has no test that
  caches.
- **165 table-namespace and context tests** (va, vb, r12, ids55) under the oracle, compared with d83d3a633:
  on 7ba62edde (`rcwide3`, dirty=no, 166 tests with va-rc-ctx), retired traces were compared file by file:
  - **163 identical.** va-rc-ctx was compared with its own P1 run, `rcctx3p1`.
  - **No P1 base:** va-alloc-rcflush-w.
  - **2 differ in order only:** va-runv-fp-pre and va-runv-fp-pre2. They retire the same multiset of trace lines
    (the sorted traces hash equal), and the preemption tick lands one loop iteration earlier.
    - The first shift **coincides with** a background fill, on one instance; the mechanism is not shown. In
      va-runv-fp-pre both runs have a gap before the ADD at 0x800021b0: 79 cycles on P1 and 126 on this tree. It
      looks like a fetch wait. Inside it the cache installs slot 2 on a demand miss at 67,601 and a fill installs
      slot 3 at 67,694 (`RC-INS slot 3 way 0 fill`). The ADD commits 47 cycles later than on P1, and the run ends 60
      cycles later (180,514 against 180,454).
    - The trace has no D$, AXI or I$ events, so it cannot show the fill's read contending with the fetch. The fill's
      reads are the only new memory traffic there, and no hit ever uses that line. If the fill does compete with
      other memory traffic, that is a cost the unit-yield rule does not cover. Sim memory delay is not silicon's.
  - **The oracle is silent in all 166.** On the same grep its three mutants fire: `rcq8m1` rc_stale_hit,
    `rcq8m2` rc_hit_mismatch, `rcq8m3` rc_addr_mismatch.
  - **Lookups and hits:** 1,812 lookups and 619 hits, all in contexts, in 37 of the 141 tests that look up.
    - va-fetch* 31 of 52 per test;
    - va-alloc* 2-12 per test;
    - va-flat 6 of 12;
    - va-rc-ctx 4 of 11;
    - va-runv4 5 and va-runv-grow 4;
    - va-q15 3 and va-fetchkill-k 2.
- **Not exercised by any arm:**
  - a parked read (a read presented with a write; the unit's single thread may never do it);
  - a dropped write that hit;
  - a lookup in a flush cycle;
  - an operation straddling an escape (the window `rc_live_q` closes);
  - the oracle's racing-store branch (a store granted between a record read's grant and its install).
  - Fill-installed hits are exercised: 230 of the 619 hits read a line that a background fill installed (15 per
    va-fetch* test, 2 in va-flat, 1 in va-rc-ctx).

- **Claim-audit for synthesis** of the whole cache at 6e82d7bd2, the full file rather than the diff alone. It
  SUPPORTED four points:
  - `rc_live_q`/`rc_live_qq`/`ctx_rev_qq` feed only `rc_flush` and `rc_cacheable`, and every endpoint below them is
    a flop D input or a RAM write enable;
  - the rev unit's D$ read and write requests, `rc_busy` and `rev_busy` take no new term, so nothing enters the rev
    read port's standing UNOPTFLAT cone;
  - `dom_switch_busy` reaches the cache only at the `rc_live_q` flop (cva6.sv:843, :2001);
  - the record array's code, reset and addressing are unchanged, and 6e82d7bd2 lies inside `ifndef SYNTHESIS`.
  
  It left two points open:
  - whether the array really maps to RAMB (the synth log's Block RAM mapping report for `g_rc_way[*].dat_mem`
    settles it; key 3 below);
  - `rc_live_q` is a new endpoint one AND gate from `dom_switch_busy`, the launch point of every failing path in the
    2026-09-07 synthesis (cva6.sv:845-850), so read it in the full build's timing report.
  
  Correction from the audit: `ctx_rev_qq` and the first `rc_flush` came in a38fddb70; 7ba62edde added only
  `p_rc_live`, the third flush term and `&& rc_live_q`.

### Synthesis-only gate (under synth-guard, 40 GB)

Keys, graded against P1's gate (d83d3a633, board lane, `synth/pw-d83d3a633-gate`):
1. Mapping in P1's class: 5,041.7 MB at the marker, ±500 MB. A blow-up is a fail, with no retry.
2. Loops: check_timing 0, and LUTLP-1 0 with its control. No timing-loop header names an `rc_*` cell.
3. Own footprint, by name (`rc_*`, `g_rc_way`):
   - **0.6-1.5 k LUT** (muxes: 128-bit read response, 56-bit walker address and 128-bit write data; tag compares; the
     fill adder);
   - **0.7-1.2 k FF** (512 valid bits, 64 LRU, and the held slot, address and data; `rc_live_q`/`rc_live_qq` add 2);
   - **RAMB36-equivalent 4** for the 2 x 256 x 128-bit records;
   - the tags in LUTRAM, about 40 cells.
   - Records mapped to flops or LUTRAM (+32 k FF or +8 k LUT) are a FAIL: the BRAM inference did not happen. Read it
     from the synth log's Block RAM mapping report for `g_rc_way[*].dat_mem`, as well as from the RAMB count.
4. core LUT within ±2,000 of P1's 162,403 beyond the own footprint. perf_counters gains 4 events (HPM 34-37), and
   `rev_walk_perf` widens from 4 to 8 bits (cva6.sv, perf_counters.sv): a few tens of LUT. (Corrected 2026-11-17
   before the gate ran: this line said "+1 event", left over from a draft with one counter.)

### Gate result, 2026-11-17 (board lane, rc-6e82d7bd2): KEY 1 FAIL by its letter; keys 2, 3, 4 PASS

- **Key 1: FAIL.** The peak at the Technology Mapping marker is 6,865.6 MB, against 5,041.7 ±500. The whole run
  peaked at 7,012.9 MB, with no Synth 8-326. The full build was not launched, and the FAIL is not re-keyed.
  - Read phase by phase, the step is not in mapping. Mapping adds 8.0 MB on both runs.
  - It lies between the two "Netlist sorting complete" lines around "Processing XDC Constraints / Initializing
    timing engine": 4,310 to 6,838 MB on B5a, against 4,205 to 4,998 on P1.
  - Across the 12 gate runs on this flow, Constraint Validation peaks at the elaboration peak plus 760-850 MB. B5a is
    the outlier at +2,593. The host's free memory does not correlate: runs with 18-36 GB free sit at 5.0 GB.
  - The guard's process-group RSS peaked at 26.17 GB, against P1's 27.35. Wall time was 51 min.
- **An anomaly of the run, not explained.** B5a's elaboration section is 505 lines; every other gate's is about
  22,000. It lacks the module messages (Synth 8-6157, 8-6155, 8-294) and carries Synth 8-7128 and 8-7156, which no
  other run has. The run script `ariane_xilinx.tcl` is identical apart from the cache path, and so is
  `~/.Xilinx/Vivado/Vivado_init.tcl`, which has not changed since 2026-11-04.
- **Key 2 PASS:** 0 loop headers, LUTLP-1 0 (toy 1), check_timing 0 (toy 1).
- **Key 3 PASS:** g_rc_way[0/1].dat_mem map to 2 RAMB36 each, and no dat_mem is in distributed RAM. Reported, not
  gating:
  - own LUT +2,471, OUTSIDE the 0.6-1.5 k band;
  - own FF +897, inside;
  - tags 34 LUTRAM.
- **Key 4 PASS:**
  - core LUT 165,786 (+3,383; +912 beyond own), FF +832;
  - ex_stage +5,647 LUT (+3,176 beyond own);
  - perf_counters +6 LUT.

### Diagnostic rerun (pre-registered 2026-11-17, before the run)

The same hash, 6e82d7bd2, and the same flow and guard. One question: is key 1's +1.8 GB a cost of the RTL or of
that run?
- **H-RTL** (the 8 KiB record array, 65,536 bits of RTL memory before RAM inference runs after Cross Boundary
  Optimization, seen by the timing-engine initialisation): Constraint Validation peaks within ±50 MB of 6,837.7, and
  the elaboration section's message set again lacks Synth 8-6157.
- **H-flow:** Constraint Validation peaks within the other runs' class (elaboration + 760-850 MB, about 5.0-5.1 GB),
  and the elaboration section carries the usual about 22,000 lines.
- **Anything else** (a peak between the two, or a message set matching one hypothesis and memory matching the other)
  is reported as read, with no decision drawn from it.
- **What follows:**
  - On H-flow, the gate reads PASS on key 1 at the rerun's reading, and the full build goes under the keys below.
  - On H-RTL, key 1's band was keyed to the wrong phase. Its purpose was a mapping blow-up (the 94 GB class), and
    mapping added 8 MB, but re-keying it is the lead's call, not mine. The full build waits for that word, and the
    rerun's keys 2-4 are recorded either way.

**RETRACTION and addendum (2026-11-17, after the rerun's launch; the pre-registration above is unchanged).**
- "An anomaly of the run" is withdrawn. I read `synth_1.runme.log`, the launcher's stdout capture, and it had
  dropped one block of lines.
  - Vivado's own `ariane_xilinx.vds` is complete: elaboration ends at line 22,410 (P1 22,365), with Synth 8-6157
    ×100 on both. Re-read here.
  - The same file gives the same Constraint Validation peak, 6,837.656 MB (P1 5,013.762).
- So the rerun's message prong is void for both hypotheses, and **the memory prong alone decides**, as written.
- Synth 8-7128 (×22) and 8-7156 (×18) are real and absent on P1. They are mux and shifter range optimisations at
  scoreboard.sv:213, capstone_rev_node.anvil.sv:6135 and ariane_pkg.sv:848/862: logic that ex_stage's change
  feeds. That is an observation, not an attribution.
- What would have caught the slip: reading the tool's own log (`.vds`) rather than a copy of its stdout. A
  difference that only the copy shows says nothing about the run.

**Rerun read (board lane, rc2-6e82d7bd2, from the .vds): H-RTL.**
- Constraint Validation reads 6,839.7 MB (+2.0 against the first run, inside ±50) and Technology Mapping 6,867.7.
  The whole-run peak was 6,935.8 and the guard RSS 26.10 GB.
- Keys 2-4 are identical to the first run.
- The +1.8 GB is therefore deterministic for this RTL, not noise of the run. **That the record array is the
  mechanism remains a hypothesis:** the rerun excludes the run, not the other parts of the change.
- As pre-registered, re-scoping key 1 is the lead's decision, and the full build waits for it.
- **Recommendation to the lead: re-scope and launch.**
  - Key 1's purpose was a mapping blow-up of the 94 GB class. Mapping adds 8 MB on both trees.
  - The whole run stays under P1's guard RSS (26.1 GB against 27.35) and far below the 40 GB ceiling.
  - Proposed key 1 for every later gate on this flow: growth across Technology Mapping itself (from the "Start
    Technology Mapping" peak to the "Finished" peak) at most +500 MB, plus a whole-run peak inside the guard.
  - Recorded as a key that was mis-scoped, not as a pass of the key as written.

**Launched by the runtime lane, 2026-11-17 06:16 +08**, as b5a-6e82d7bd2, the full build. It cites the lead's
standing instruction of 16 Nov to launch the Section 5 work "in any case". Key 1 is recorded as mis-scoped (the
recommendation above), not as a pass. The lead had not answered the key-1 question by then. This build serves the
B5a-against-P1 performance pair.

**THE LEAD'S RULING on key 1 (2026-11-17, in the board lane's session): "Re-scope, build".**
- Key 1 is recorded as mis-scoped, not as a pass.
- Its replacement, for this and later gates on the flow: growth across Technology Mapping itself of at most
  +500 MB, plus a whole-run peak inside the guard.
- The ruling came after the runtime lane had launched both full builds on the standing instruction. Both builds were
  already sealed when it arrived.
- Board lane's dev record: 21859e83e215.

### Combined gate: B5a + R-76 (pre-registered 2026-11-17, before the run)

- **Hash:** capstone-ariane `r76-ccsrrw-priv` 7180a83b6. It descends from 6e82d7bd2, and its RTL differs from it in
  `core/csr_regfile.sv` only (+7 lines: R-76's privilege term; see ISSUES R-76).
- **Lint on 7180a83b6** (`.sup/lint-rc10`, dirty=no): PASS, every counter at baseline: LATCH 44, MULTIDRIVEN 3,
  UNOPTFLAT 39, UNUSEDSIGNAL 1480, UNDRIVEN 25, BLKSEQ 2, CASEOVERLAP 2, SELRANGE 100, ALWCOMBORDER 0, COMBDLY 0,
  ANVIL_UNOPTFLAT 0.
- **The 95-sweep** is identical to B5a's.
- **Purpose:** the bitstream that reaches silicon with R-76. The 6e82d7bd2 build keeps the one-variable provenance for
  the performance numbers.

Keys, graded against B5a's gate (rc-6e82d7bd2 / rc2-6e82d7bd2):
1. **Memory, in the re-scoped form:**
   - growth across Technology Mapping itself (the "Start Technology Mapping" peak to the "Finished" peak) at most
     +500 MB;
   - the whole-run peak inside the guard.
   - Reported against prediction, not keyed: Constraint Validation 6,840 ±100 MB.
2. **Loops:**
   - check_timing 0, and LUTLP-1 0 with its control;
   - **no "Found timing loop" header at all**: privilege_violation is on the lint loop list, and the fix adds a term
     to it.
3. **BRAM:** as B5a. g_rc_way[0/1].dat_mem each map to 2 RAMB36, and no dat_mem is in distributed RAM.
4. **Area:**
   - core LUT within ±2,000 of B5a's 165,786;
   - csr_regfile_i within ±2,500 of B5a's own row (the redistribution floor).
   - Reported against prediction, not keyed: R-76's own cost is a 4-input AND into privilege_violation, under 20 LUT.

Its full build, on a gate pass, follows B5a's full-build keys below. The flash is the lead's decision.

**Combined gate read** (runtime lane's run b5ar76-7180a83b6; graded here from its `.vds` and post-synth report;
synth_design completed successfully):
- **Key 1 PASS.** Technology Mapping adds +8.0 MB (Timing Optimization 6,700.8 to Technology Mapping 6,708.8), and
  the guard peak RSS is 25.26 GB.
  - Reported, not keyed: Constraint Validation 6,680.8 MB, outside the predicted 6,840 ±100 and 157 MB below B5a's
    6,837.7. Not explained.
- **Key 2 PASS.** 0 "Found timing loop" headers and 0 Synth 8-326.
  - check_timing: CHECKLOOPS 0, toy 1 (`twc/rc/rck-b5ar76-7180a83b6.log`, `rck-toy-b5ar76.log`);
  - LUTLP-1: 0 rows, toy 1 (`lutlp/b5ar76-7180a83b6-synth.rpt`, `toy-chk-…`);
  - all four Vivado runs exit 0. Run by the runtime lane on the gate's dcp (sha256 80b20d5875702d1c…), and the logs
    were re-read here.
- **THE COMBINED GATE PASSES ALL FOUR KEYS.** Its full build, b5ar76-7180a83b6, was launched by the runtime lane at
  07:14 +08, under B5a's full-build keys.
- **Key 3 PASS.** g_rc_way[0/1].dat_mem each map to 2 RAMB36. dat_mem rows: 4 in the Block RAM sections, 0 in
  Distributed RAM.
- **Key 4 PASS.**
  - i_cva6 164,962 LUT, against B5a's gate 165,786 (−824; key ±2,000);
  - csr_regfile_i 28,077, against 28,776 (−699; key ±2,500);
  - ex_stage_i +940, reported only.

### Full build (only on a gate pass)

- Routes, 0 errors.
- Congestion <= 5.
- Loops 0.
- Timing report: the worst slack of any path ending at `rc_live_q` is recorded (reported only, not a key).
- **WNS inside ±3.74 ns of P1's -13.173.** P1 missed its own key 5 against G. This band says only that B5a does not
  move timing beyond the null band; it does not say B5a is flashable.
- P1 placed at 99.71 % of slices (187,368 LUT). B5a adds about 1 % more LUT and FF. **Routing-failure branch:** no WNS
  is recorded; the placed area, the congestion windows and a cell census of the hot windows are.
- The flash is the lead's decision: the board lane's session, with this grade in front of the lead.

### Full build read: B5a 6e82d7bd2 (sealed by the runtime lane, graded here; key 5 NOT MET carries over)

- **Seal:** `caplifive_ids55-…-pagedtbl-reccache_6e82d7bd2.bit`, sha256 806c5f55a40ca851a9a9b29f48272137f543dd3714d6d15a0acdf4457eee9356,
  11,443,722 B. Tarball sha256 2b5b69fbd216…, with the .bit cmp-identical; PROVENANCE HEAD 6e82d7bd2956. Run guard
  peak 26.02 GB (exit 0, 1 h 17 m), collector 15.90 GB.
- **Routes, 0 errors: PASS.** `work-fpga/ariane_xilinx_route_status.rpt` reads 246,477 fully routed nets and 0 with
  routing errors. At the end of route_design, failed, unrouted and partially routed nets are all 0, with 0 overlaps.
- **Congestion at most 5: PASS.** Effective levels 5, 5, 4, 5 by direction. Corrected source: route_design's
  initial-routing congestion estimate in the implementation log (`route_design` at log line 27045), not
  `report_design_analysis` as first written.
- **Loops 0: PASS.** check_timing: "There are 0 combinational loops in the design". The methodology report has no
  TIMING-23 and no LUTLP-1, while it does list other checks (TIMING-16 ×1000, SYNTH-6 ×109), so it ran.
- **WNS inside ±3.74 ns of P1's −13.173: PASS.** WNS −13.691 (Δ −0.518), TNS −259,470, 50,477 of 234,469 endpoints
  failing, WHS +0.056.
- **rc_live_q slack (reported only):**
  - It is not among the 2,000 worst failing endpoints, whose slack runs from −13.691 to −11.477, so its slack is
    better than −11.477. The exact value is not read.
  - No rc_* cell is on the worst 500 paths (positive control `fu_data_q`: 1,250 hits).
  - The worst path runs from csr_regfile `cpmp_q[3][comp_metadata][bounds][iE]_rep` to `fu_data_q[0]…
    cap_metadata_a[20]`, the base's documented family.
- **Not run:** the in-flight loops-only gate and the LUTLP stop. The runtime lane launched with launch-job.sh
  directly. The routed loop checks above cover what they would have stopped.
- **Grade: a flash candidate on every key.** Key 5 is NOT MET against G, as on P1. The flash is the lead's decision.

### Full build read: B5a + R-76 7180a83b6 (sealed by the runtime lane, graded here; key 5 NOT MET carries over)

- **Seal:** `caplifive_ids55-…-pagedtbl-reccache-r76_7180a83b6.bit`, sha256
  b339149665e4a141b5a9efb778fad625e4db4a7d9cf6640806f61bfc71b95ee7, 11,443,722 B. Tarball sha256 0388b34ae40f…, with the
  .bit cmp-identical; PROVENANCE HEAD 7180a83b61942ea5. Run guard peak 26.16 GB (exit 0, 1 h 10 m), collector 16.46 GB.
- **Routes, 0 errors: PASS.** The route_status report reads 245,529 fully routed nets and 0 with routing errors; the
  end of route_design reads 0 failed, unrouted and partially routed nets, with 0 overlaps.
- **Congestion at most 5: PASS.** The router's initial estimate is 5, 5, 5, 5 by direction. B5a's was 5, 5, 4, 5.
- **Loops 0: PASS.** check_timing reports 0 combinational loops. The methodology report has no TIMING-23 or LUTLP-1,
  and lists TIMING-16 ×1000, so it ran.
- **WNS inside ±3.74 ns of P1's −13.173: PASS.** WNS −13.615 (Δ −0.442; B5a −13.691), TNS −288,193, 53,219 of 234,474
  endpoints failing, WHS +0.049.
- **Reported only:**
  - rc_live_q is not among the 2,000 worst failing endpoints (−13.615 to −12.146), and no rc_* cell is on the worst
    500 paths (control fu_data_q: 4,434).
  - `privilege_violation` (R-76's term) is on neither the worst 500 nor the 2,000 (the same on B5a). Synthesis may
    rename the net.
  - The worst path runs from dom_switcher `_thread_0_event_counter_870_1_q` to `fu_data_q[0]…cap_metadata_b[18]`. That
    is bitstream B's recorded family, swapping order with B5a's cpmp_q family inside the band.
- **Grade: a flash candidate on every key.** Key 5 is NOT MET against G. It is not scheduled on a board; R-76's
  silicon flash is the lead's decision.

### Board readings (B5a against P1, same kit and board; key 5 NOT MET on both)

- **Identity,** per paged run: HPM 35 (read miss) + HPM 37 (fill read) + (record writes) = ev31 (walker record
  accesses). The writes are not counted separately; pb-grade's run-own ev31 band applies, and the identity is checked
  as ev31 - ev35 - ev37 >= 0.
- **Hits:**
  - mcf test (pair B's native row is `505.mcf_r test`): HPM 34 / (HPM 34 + HPM 35) is predicted at **at most 73.4 %**,
    the model's figure for a fill that always completes. The fill yields, so the RTL's rate is lower.
  - A rate below 30 % on mcf means the fill is yielding on the probe path. That is a finding, not noise.
- **Cycles,** mcf: Delta(B5a - P1) between **-6.4 G** (records L1D-missing on P1, about 50 cycles saved per hit) and
  **-0.4 G** (L1D-resident, about 3 saved), plus 2 cycles per miss and per write. Neither bound counts the fill's
  own memory traffic. In the wide pass, one +47-cycle shift coincides with a fill in flight (one instance, mechanism
  not shown).
  - A Delta above +0.2 % (+0.29 G) on mcf refutes "the cache does not cost on the probe path".
- **gcc** (train01, about 30 % hits in the model on test): predicted inside ±0.2 % of P1.
- **perl:** NO GRADE at N=1 on cycles (run-to-run variance, Addendum 153a); HPM identities only.

**Readings so far (runtime lane, train PREREG Addenda 157/158; key 5 NOT MET on both bitstreams):**
- **perl makerand** (ungraded on cycles):
  - board 1 (cr-b1, firmware = P1's): run_cycles 7,025,320,964, inside P1's four-run range 7,001.9..7,038.8 M;
  - board 2 (cs-b2, record-cache event set): 7,000,880,881, −0.21 % against cq-b2;
  - hashes equal P1's on both.
  - Record-cache counters (cs-b2): read hit 9,757, read miss 27,856, so **hit rate 25.9 %**.
  - Fill reads 77,408, which is **2.78 per miss out of at most 3**: the fill completes about 93 % of the time; it is
    not mostly yielding.
  - Write-lookup hit 2,459. Identity ev31 − ev35 − ev37 = 3,368 ≥ 0, MET. That difference counts record writes plus
    reads outside a context, not writes alone.
  - The model's 40.1 % was QEMU's perl test input (specdiff FAIL), not makerand.
- **gcc train01, board 1** (cr-b1): run_cycles 42,045,959,257, against P1's two runs on the board, 41,867,859,977 and
  41,880,603,782 (spread 0.03 %): **+0.41 %. THE PRE-REGISTERED gcc PREDICTION (inside ±0.2 % of P1) IS REFUTED.**
  - On gcc the record cache costs time.
  - Paged event set, P1 to B5a:
    - record-TLB misses −9 % (3.33 M to 3.02 M);
    - walk cycles −4 %;
    - walker requests +19 % (22.84 M to 27.20 M);
    - verdict-cache replays unchanged;
    - **L1D misses +3.5 % (+12.9 M)**.
  - Inference, not attribution: the fill's record reads go through the shared L1D. The extra 171.7 M cycles divided
    by 12.9 M extra L1D misses is about 13 cycles each. The fill's traffic and its pollution of the L1D are the
    candidate cost, and fewer walks do not repay it on gcc.
  - What would settle it is a matched pair that differs only in the fill: B5a with the background fill off. No switch exists:
    it is a small edit at `RC_FSEL`'s transition (`ex_stage.sv:2363`, always to RC_IDLE), and a new bitstream. cs-b2's gcc fill count comes first.
- **What would have caught this before the board:** the pre-registration's cycle bounds counted only hit savings and
  "2 cycles per miss and per write". The wide pass's one +47-cycle fill coincidence was recorded and then added to the
  bounds as "neither bound counts the fill's own memory traffic". It was not turned into a prediction about L1D
  misses. A fill that reads three records per miss into the shared L1D needed its own predicted L1D-miss delta.
