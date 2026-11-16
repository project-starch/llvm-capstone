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
