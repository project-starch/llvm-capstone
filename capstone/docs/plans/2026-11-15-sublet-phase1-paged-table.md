# Sublet Section 5, phase 1 in RTL: the dynamically grown paged lifetime table (plan and pre-registration, 2026-11-15)

The scope and the phases are in spec-silicon `capstone/docs/plans/2026-11-15-sublet-section5-rtl.md` (rows 3, 4 and 6).
The runtime lane relayed them from the project lead. Phase 1 widened the same day to dynamic growth, which the external
collaborator asked for ("as described in the paper, the table grows dynamically").

Phase 1 answers one question on silicon: **what do paging and growth cost?** It runs on today's 55-bit CSRUNV stack, with
everything else as in G (capstone-ariane 9fd9d2530, the adopted bitstream). It is a proxy for the collaborator's kernel,
which needs the 63-bit format and the trap route (phases 3 and 4).

## The change: capstone-ariane rtl/paged-table d83d3a633 (on 9fd9d2530; pushed)

- **A walker outside the rev unit.** It is an SV shim in `ex_stage.sv` on the unit's two D$ ports, with no Anvil change.
  - Under the table namespace, a table whose root+0 has PAGED (bit 62) set is QEMU's tree (`cap_rev_table_abi.h` on
    capstone-qemu address-space-root): 4 KiB pages, the root page with the header at +0..+31 and 32 root entries from
    +64, slot split 5:9:9:8, plain 4 KiB-aligned physical entries, 0 = none.
  - Header slots 0 and 1 stay addressed flat. Records keep F's two-word format; only their place changes.
  - A 25-bit slot means root entry 0, directory 1 at slot[24:17], directory 2 at slot[16:8], so every miss is three
    dependent reads.
- **A 16-entry record TLB** {slot[24:8] -> record-page PA}, round-robin.
  - It caches only successful walks and is flushed on the namespace edge (`revnode_ns_flush_o`).
  - Pages never move, so no entry goes stale. A page published after a fill is reached at once.
- **Bypass, from flops only.** Flat tables and the physical pool pass straight through once root+0 has been probed (one
  read after each namespace flush). The unit sees only a later grant, and PAGED = 0 in its slot-0 view.
- **One request held at a time,** write first. The unit's own send discipline orders the rest.
- **Growth (the kernel's protocol, capstone_vm on the compiler lane's table-tree lane, 722416c35e77).**
  - The unit's existing node-pressure check raises cause 30 before any record changes; inside a context it escapes as
    kind 5, paused at the allocating instruction.
  - The kernel zero-fills a batch of 16 record pages, writes their directory entries, fences, then raises the capacity at
    root+0.
  - CSRUNV action 0 re-executes the instruction.
  - The walker checks no capacity; the unit re-reads the header at every allocation (`vp_load_hdr`).
- **HPM** (the event field widens to 6 bits):
  - 29 walk started (record-TLB miss);
  - 30 walk cycle;
  - 31 paged record access (TLB lookup);
  - 32 unreachable record;
  - 33 cause 30 taken, counting both a trap at commit and a context's escape.

## Deviations from Section 5 and QEMU (confirmed by the runtime lane for phase 1; closed in phase 3)

1. **Cause 30 fires with 256 identities still free** (the unit's D5 reserve inside a context), not at zero. Each cause 30
   is still one grow answered by one batch. So the measured growth cost, (grows) x (escape + zero-fill + resume), is
   right to within one grow per context.
2. **NOGROW (+48) is not read, and there is no no-fault failure result.** A table that cannot grow raises 30 again on
   resume, so the module must END the context instead (compiler lane: a module mutant shows it ends and does not loop).
3. **The table is capped at 2^25 slots** (the 25-bit slot), not 2^31.
4. **An unreachable record fails closed:** a read returns DEAD, and a write is dropped with one tracker flush. Section 5
   raises cause 29.

Recorded limits (claim-auditor, 2026-11-15), both excluded by the protocol:
- A dropped unreachable write's tracker fill is visible to the LSU revocation cache for a few cycles before its flush.
  This matters only if a kernel unpublishes a page it published.
- The record TLB assumes a table is never rebuilt under an unchanged srevroot.

## Sim gates (all rtl_dirty=no)

| arm | reading | result |
|---|---|---|
| va-alloc-w, va-alloc-csret-w on the walker (flat tables) | identical to G (24/24, 18/18) | the probe costs +93 / 0 cycles (68e14c3a3 against G on the same test source; later runs use the edited va-alloc.S and are not cycle-comparable to G) |
| va-alloc-pg-w, va-alloc-csret-pg-w (the same table, paged) | identical to flat except root+0 0xc000000000000400 (PAGED kept) | 1 walk, then TLB hits |
| va-alloc-far-w (free records in other subtrees, capacity 0x40000) | as flat except root+0, +8 0x20307, +16 0x100020306 (the slot ids) | 3 misses x 3 levels |
| va-alloc-hole-w (an unpublished directory-2 entry) | the pop reads DEAD (`PW-DEAD rd slot 0x20305`), refused 29, nothing written | as registered |
| va-alloc-flush-w (srevroot to a tree whose page is unreachable; A's record 7 FREE) | B's CSNODE refused 29 | the TLB forgot A's page |
| va-alloc-flushctl-w (B's page published) | B's CSNODE 0x8, next 9, free 0 | as registered |
| va-runv-grow (a full paged table in a context) | kind 5 at the CDERIVE; M grows; action 0 re-executes; the child's load reads 0x77; root+8 0x101; HPM 33 = 1; record 256 VALID, parent 2 | as registered after two first-run corrections (below) |
| mutant PW_SIM_BAD_I2 (wrong directory-2 index) on far | refused 29 | the arm fires |
| mutant PW_SIM_NOFLUSH (TLB kept across srevroot) on flush | reads as the control, 0x8, 3 walks | the arm fires |

- **First-run corrections, all recorded in the tests:**
  - FAR's first cut put next-unused below the far slots, and the unit refuses a free-list head at or above next-unused
    (29). HOLE's 29 was therefore that refusal, not the walk's.
  - va-runv-grow's +32 label sat one instruction before the CDERIVE, so it reads 4, the CDERIVE's own pc.
  - HPM 33 read 0, because the escape strips cause 30 from commit's exception. It now counts the escape event too.
- **Lint (rtl-lint-gate, d83d3a633): PASS,** every counter at baseline: LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0,
  COMBDLY 0, UNOPTFLAT 39, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0.
  (b039c3a05 read UNUSEDSIGNAL 1481: the walk address's low bits, narrowed.)
- **claim-auditor, aimed at synthesis risk, on d148c2107:**
  - Bypass equivalence, response ownership and ordering: SUPPORTED.
  - Synthesis: one REFUTED claim. A 25-bit compare of the unit's address selector reached the D$ read port's data_req,
    and that port's grant is on the standing loop list (`rtl-lint-loops.REF.txt:31`). Fixed in d83d3a633: the match is
    the unit's valid alone; the compare is a sim-only `CAPSTONE-CHECK-FAIL` that never fired.
  - A one-cycle mode window at a namespace edge: fixed (pw_ns_edge).
  - Every arm reads identically on d83d3a633 and d148c2107, values and cycles.
- **95-sweep against G: PASS.** sim14 `pwsw6`, TAGMLP_PROBE, on d83d3a633, rtl_dirty=no, SWEEP-RAN 95, 184 logs. The
  reference is G's sim11 `v6csw4` on 9fd9d2530. Both PROVENANCE files carry the same five Anvil unit hashes, since there
  is no Anvil change.
  - sweep-compare: 92 identical (exceptions, readings, retired), 0 differ, 3 excused on both sides (rand,
    perf_tcache_random, perf_riscv_random).
  - sweep-compare-rvfi: 92 traces identical, 0 differ, 26 parked.
  - Cycles, from the final line of each `.log.iss`: 0 of 92 differ.
  - The registered prediction was values and RVFI identical, with cycles identical except where a test enters the table
    namespace. By the test list, no sweep test is a virtual-context test (no va-/vb- family), so the probe never ran and
    no cycle could move. The +93-cycle probe cost is read in va-alloc-w (above), not here.
  - sweep-compare-cycles.py (tools-wt) no longer runs: it calls a `main()` that sweep-compare.py has dropped. The cycle
    column above was read with the same `.log.iss` end-line pattern, from a stand-alone copy of its reader.

## Synthesis-only gate: keys, written before it runs (base: G 9fd9d2530's synthesis-only gate)

1. Mapping in G's class (5,036.8 MB at the marker, ±500 MB); a blow-up is a fail, no retry.
2. Loops: check_timing 0; LUTLP-1 0 with its control; no timing-loop header names a `pw_*` cell or the port-2 grant.
3. Own footprint: the walker's cells by name (`pw_*`, `rv_rd_res`) at about 1.5-3 k LUT and 1.0-1.6 k FF (the TLB is
   16 x 62 bits). ex_stage within the ±2,500 floor of G plus that; core within the floor of 163,539 plus that.
   perf_counters' 6-bit field is a few dozen cells.
4. Timing: report_timing to and from the rev read port's grant
   (`gen_rd_ports[2]...i_wt_dcache_ctrl__rd_ack_i`) and the R-35 endpoints (the LSU revocation cache's write decode),
   for G and the trial. It is reported, not graded, at synthesis; WNS is graded at the full build only.
5. A full build goes only on a pass of keys 1-3, graded on: routes (0 errors), congestion <= 5, loops 0, and WNS in the
   ±3.74 ns null band of G's -9.107. On a route failure, no WNS is recorded; the placed area and the census are.

**GATE RESULT (board lane, apollo, tag pw-d83d3a633, 2026-11-15): keys 1-3 PASS. The full build was launched on the
conditional go.**

1. **Key 1 PASS.**
   - Mapping 5,041.7 MB against G's 5,036.8, so +4.9.
   - 0 Synth 8-326.
   - Guard peak 27.35 GB.
2. **Key 2 PASS.**
   - 0 "Found timing loop" headers.
   - LUTLP-1 0, with the toy control reading 1.
   - check_timing loops 0, with G 0 and the toy 1.
3. **Key 3 PASS on the floor clauses.**
   - Own cells by name: 2,008 = 1,197 FF + 752 LUT, all `pw_*`. `rv_rd_res` has 0 cells, which fits a combinational
     mux. NEG 0, POS 1.
   - ex_stage +1,425 LUT / +1,092 FF; less own, +673.
   - core 162,403 LUT (-1,136); less own, -1,888.
   - Design 189,693 (-1,142).
   - Predictions, not gating:
     - own FF 1,197 is in band;
     - own LUT 752 is BELOW the 1.5-3 k band (missed low). Post-synthesis LUT names are lossy, and ex_stage's top level
       is +1,032;
     - perf_counters +98 LUT / +6 FF is above "a few dozen".
   - Redistribution past the floor in modules whose source did not change:
     - rev_node -7,458, dyn_unit `_spawn_0` +5,935, `_spawn_1` -788. Summed, -2,311, inside the floor.
     - Source check: forcing PAGED to 0 in the unit's slot-0 view removes one term of one refusal
       (`capstone_rev_node.anvil:301`). That cannot fold 7.5 k LUTs, so the shift reads as re-homing between the Anvil
       hierarchies.
     - Also csr_buffer +2,112, issue_read_operands -2,388, csr_regfile -1,254.
4. **Key 4 (reported).** Synthesis estimates on a created 40 ns clock.
   - Through the D$ arbiter's two grant copies (rd_ack_q_i_1, vld_sel_q[1]_i_1):
     - worst 7.591 on G, 6.624 on the trial;
     - to the fill taps 23.957 on G, 24.085 on the trial;
     - the walker adds grant -> `pw_slot_q` at 17.784, with 85 of its 1,197 cells in the grant cone.
   - Into port 2's data_req: 13.203 on G, 13.019 on the trial; from `pw_*`, 13.924.
   - Read data into `pw_*`: rvalid 23.357, D$ sequential 23.698.
   - The board lane retracted its first G reading ("the grant reaches neither the fill taps nor the rvc flops"), taken
     through port 2's rd_ack_q_reg/D net. That cone was local. The arbiter-output readings above replace it, and the
     rvc flops were not re-read by that method.

**FULL BUILD, KEY 5 (sealed 2026-11-15 15:24Z by the board lane; graded here from the build's own reports).**

- **Seal:** `caplifive_…-tagoverlap6c-pagedtbl_d83d3a633.bit`, sha256 dc411e36fc5e68ae…, 11,443,722 B.
  - The tarball `synth-d83d3a633-exit0.tar.gz` (4d3f3990…) holds the same .bit; PROVENANCE HEAD is d83d3a633.
  - Run peak 27.39 GB.

| key 5 | trial d83d3a633 | G 9fd9d2530 | grade |
|---|---|---|---|
| routes | 241,265 / 241,265 nets, 0 routing errors | routed | PASS |
| loops | check_timing loops 0, latch loops 0 | 0 | PASS |
| congestion <= 5: placer final | "No congestion windows above level 5" | same | PASS |
| congestion: router initial estimate | Long N/S level 6 (20.3 / 20.4 % of tiles); Global/Short N/S level 5 | East Short 5 only | reported: WORSE than G |
| WNS in ±3.74 of G's -9.107 | **-13.173** (TNS -201,219; 47,712 / 231,920 failing; WHS +0.049) | -9.107 | **NOT MET (0.33 ns outside, bad side)** |

- Placed: 187,368 LUT (91.94 %), 76,543 FF, 50,800 slices (99.71 %). G: 188,525 LUT, 50,685 slices (99.48 %).
- The board lane's own grade agrees: FULL BUILD FAIL 3/4, key 5 WNS.
  - Its congestion readings come from different reports: placed level 4 (G 4), router effective max 5 (G 4).
  - Route 35-447 appears 2 times (G 0): the router put completion ahead of timing.
  - TNS -201,219 against -77,207; failing endpoints 47,712 against 23,259.
- **Worst paths.** `worst_500_paths.rpt` on both builds, read by script.
  - Trial worst 20: dom_switcher `cur_idx_q_reg[3]` -> issue_read_operands `fu_data_q` (cap metadata), 109-111 levels.
  - By family on the trial: `cur_idx_q` -13.173 at 111 levels. `cpmp_q` -12.937 at 120 levels, against G's -9.107 at
    116, so -3.83 ns. On G, `cur_idx_q` lies above the worst-500 floor of -7.303, so it is at least 5.9 ns worse here.
  - Sources across the trial's 500: csr 463, cur_idx_q 37. Destinations: scoreboard `mem_q` 423, `led_stretch_cnt`
    47, `fu_data_q` 30.
  - ~~Walker cells on the worst 500: 0 on the trial. The control, paths touching `fu_data_q`, reads 389.~~
    **RETRACTED.** The count matched cell NAMES (`pw_`) only, and had no walker-logic positive control.
    - 39 of the trial's 500 run through the walker's net `ex_stage_i/rev_node/rv_rd_res[data_rvalid]` (fan-out 129).
      They sit at ranks 261-481, at -11.370 to -11.142. G has 0. Found by the board lane; re-counted here by script.
    - That net is the walker's response filter, `(data_rvalid && !pw_own_rd_q) || pw_dead_rv_q` (`ex_stage.sv:1977`).
      It hides the walker's own reads from the unit and injects DEAD. On G the unit read the D$ rvalid directly (9fd9d2530
      `ex_stage.sv:1733-1744`), so the walker adds logic on this net. How many LUT levels it adds is not counted: that would be
      a netlist claim, and it is inferred from source only.
    - The 39 run `cpmp_q[5]..bE[1]` -> D$ port-2 ctrl -> the filter LUT -> rev_node event-counter logic ->
      `led_stretch_cnt`, the debug-LED aperture counter.
    - None is in the worst 260, so the WNS path and the two compared families stand as net-delay readings.
    - "No walker logic on the failing paths" does not stand: the walker is on 39 of them.
- **Reading.** Both pre-existing families lost about 4-6 ns at about equal depth, beside a router estimate one
  congestion level worse than G's (Long level 6 over a fifth of the N/S tiles).
  - That is delay per level: placement and routing in a design that is 99.7 % slice-full.
  - The WNS-setting paths do not pass through the walker. The walker IS on 39 lower-ranked failing paths (retraction above).
  - One build cannot say whether the walker's 2 k cells caused the placement shift or merely sampled it. The null band
    itself is 3.74 ns, and this sits 0.33 ns beyond it.
- **Attribution (board lane, routed dcps; logs `synth/twc/attr5/`).** The figures below were checked here in
  `spW_dump.log` / `spW_onG.log`.
  - Same-cell retiming is NO-DATA: synthesis renumbered the LUTs, and 148 of the trial path's 224 pins do not exist on
    G. So the comparison uses the same start and end pins.
  - Trial worst, `cur_idx_q_reg[3]` -> `fu_data_q[..][cap_metadata_a][19]`: trial -13.173, 111 levels, logic 8.399, net
    44.857. G -8.032, 114 levels, logic 9.247, net 38.699.
  - G's worst, `cpmp_q[9]..iE_rep` -> `fu_data_q[..][rd_metadata][23]`: G -9.107, 116 levels, logic 9.833, net 38.617.
    Trial -12.333, 120 levels, logic 9.586, net 42.738.
  - **Both directions: equal or less logic, 4-6 ns more net delay.** That is placement and routing on pre-existing
    paths, not added logic.
  - The walker's own routed paths all meet timing:
    - worst into a walker flop +0.284 (`cpmp_q` -> `pw_slot_q` CE, 78 levels);
    - worst out of one +3.160 (`pw_tag_pw_q` -> `fu_data_q`);
    - grant copies -> `pw_slot_q` +1.803;
    - grant -> fill taps +10.812 on the trial, +14.364 on G.
  - What it cannot separate: whether the walker's +1.1 k FF crowded placement, or whether this is draw-to-draw noise. A
    second build of the same commit would answer it, and none is proposed.
- **Consequence.** WNS -13.173 is 0.94 ns past the worst bitstream known to run correctly (-12.229, per the runtime
  lane).
  - The board readings carry "key 5 NOT MET".
  - Pair A's outputs bit-exact against G (specdiff) plus the smoke are the functional check. No pair-B number is cited
    before pair A has passed it.

Order (amended 2026-11-15, after the sweep): the synthesis-only gate is requested on d83d3a633 now, because it builds no
bitstream and its keys are above. The full build (key 5) waits until the board pre-registration below has been pushed.

## Board pre-registration (written 2026-11-15 before the full build; event-23 rows from G, runtime Addendum 143)

- **Pairs, one bitstream:**
  - A = trial on the flat kit against G on the flat kit (the null control);
  - B = trial on the paged kit against the trial on the flat kit: the paging cost, with the growth module's own layout.
    - Table-tree 722416c35e77 grows in 64 KiB batches: 16 record pages = 4,096 slots, physically contiguous within a
      batch.
    - Batches are independent allocations, directory pages are single pages, and a pooled tree keeps its batches.
    - The brackets do not depend on that layout. Walks are per 4 KiB record page and cost three directory reads
      wherever the pages sit. Layout enters only through record-line conflicts in L1D, which is pair B's L1D-miss delta.
    - (Corrects this section's first wording, "record pages consecutive".)
  - optional C = a deliberately scattered layout, the collaborator's call.
- **L1D misses are a reading in pair B, not a control.** Walks read the directories through the D$, and record lines
  sit at new addresses. The paged arm's L1D-miss delta over the flat arm is the second-order term that check 2's 25 %
  allows for.
- **Inputs in hand** (the compiler lane's twin census, FRCDKMSa3; the per-op counts from the walker trace above):

| row | CDERIVE | CREVOKE | next | record-page span | grows |
|---|---|---|---|---|---|
| mcf | 60,282 | 60,238 | 8,256 | about 33 | 2 |
| gcc train01 | 1,088,904 | 1,036,504 | 73,574 | about 288 | about 17 |
| perl makerand | 921 | 141 | 1,974 | 8 | 0 |

- Per-op record accesses through the walker: CDERIVE 9-10 and CREVOKE 11, of which 2-3 are header slots (no TLB).
- **Each verdict-cache miss costs exactly one record read** (by source, d83d3a633).
  - The LSU's miss probe calls `get_rev_node` once (`capstone_rev_node.anvil:809-817`). That is one `read_req` at the
    node index, which is always >= 2 (`:232-238`).
  - Commit replays the marked access once its probe has resolved (`commit_stage.sv:907-912`; ev23 counts these replays).
  - So ev31 (paged record accesses) = ev23 + 6-8 x CDERIVE + 8-9 x CREVOKE. These are the per-op record accesses from
    the walker trace, minus the 2-3 header slots per op. Header slots pass through LOOK and pay k, but they are not
    counted in ev31.
  - Every paged access, header slots included, is A = ev23 + 9-10 x CDERIVE + 11 x CREVOKE.
- **Cost per access, by source and trace** (corrects the first draft's "x 1" and "4-8 cycles"):
  - **k = 2 cycles per paged access on a record-TLB hit.** In bypass mode a request is presented and granted in the IDLE
    cycle. In paged mode it goes IDLE -> LOOK -> ISSUE and is presented at ISSUE (`ex_stage.sv:2005-2013`, `:1970`,
    `:2198`). The probe pays this, and so does every header access.
  - **Walk, directories L1D-hot: about 7 cycles more.** That is three dependent levels at 2 cycles each, plus the
    LOOK -> WALK cycle. The trace is va-runv-grow, `pwall2`: levels at 84,853 / 84,855 / 84,857.
  - **Walk, directories missing L1D: + 3 x c_mig.** The first walk in that trace took 216 / 93 / 53 cycles per level.
    That is sim memory delay, not silicon. The bracket assumes c_mig of 25-40 cycles on silicon; this is an assumption and not a measurement. ev30 / ev29 reads the real figure.
  - **m, the walk fraction, is not predicted. Pair B reads it** as ev29 / ev31.
    - The TLB covers 16 x 256 = 4,096 slots.
    - Uniform use of mcf's ~33 record pages would give m ~ 0.5. That is reported, not graded.
  - Grows (mcf 2, gcc ~17): each is one escape, a 64 KiB zero-fill and a resume. That is under 0.1 M cycles per grow,
    negligible against the terms above.
- **Brackets, Delta run_cycles of pair B (paged - flat on the trial)**, from G's Addendum 143 rows (bv-b1, board 1,
  kit bkit86, exact end - begin):

| row | ev23 | run_cycles | m = 0 (k only) | m = 1, dirs hot | m = 1, dirs to MIG (25-40) |
|---|---|---|---|---|---|
| mcf | 180,614,513 | 144,015,279,285 | +0.36 G (+0.25 %) | +1.63-1.64 G (+1.13-1.14 %) | +15.2-23.4 G (+10.6-16.3 %) |
| gcc train01 | 6,409,096 | 41,536,277,321 | +0.06 G (+0.13-0.14 %) | +0.20-0.23 G (+0.49-0.55 %) | +1.80-3.16 G (+4.3-7.6 %) |
| perl makerand | 30,296 | 6,755,758,710 | +0.08 M (+0.001 %) | +0.35 M (+0.005 %) | +3.1-5.0 M (+0.05-0.07 %) |

  - The brackets come from k x A + walk x R.
    - mcf's op traffic is about 0.7 % of ev23, so its cost is the verdict-cache misses.
    - gcc's op traffic is about 15-18 M record accesses, 2.3-2.8 x its ev23, so its cost is the capability ops.
  - Predicted ev31:

    | row | ev31 | graded band (0.9 x low .. 1.1 x high) |
    |---|---|---|
    | mcf | 181.46-181.64 M | 163.3-199.8 M |
    | gcc | 21.23-24.45 M | 19.1-26.9 M |
    | perl | 36,950-38,933 | 33,255-42,826 |
  - Computed with k = 2, a hot walk of 7 cycles, and 3 x 25-40 cycles for a walk to the MIG.
  - perl makerand is the NULL row. Its 8 record pages fit in the 16-entry TLB, so after the first walks m is about 0,
    and every bracket sits inside the 0.2 % run noise. Its Delta run_cycles must read within 0.2 %.
    - Each namespace flush empties the record TLB, so ev29 is at most 8 x the context entries.
    - perl has 1,140 runs, so ev29 is a few thousand or more; the count is reported.
    - For mcf the same refill term is at most 33 x 48,930 runs, about 1.6 M walks, under 1 % of its ev31.
  - The m = 1 columns are bounds on the record traffic, not expectations. gcc's ops allocate near next-unused, so their
    walks should be rarer than its verdict misses.
- **Graded on the board (pair B):**
  1. ev31 lies in its row's graded band, recomputed with ev23 from pair B's own paged run. The op counts come from the
     twin census. A reading outside the band means the walker's view of the traffic differs from the source reading, and the bracket
     does not apply.
  2. Self-consistency: Delta run_cycles is within 0.2 % of run_cycles + 25 % of (2 x ev31 + ev30 + ev29). The paged
     run's events 29/30/31 are used.
     - The replay waits for its probe, so the walker's latency is on the critical path, fully exposed.
     - The 25 % covers second-order terms the counters cannot see: directory lines evicting program data, and header
       accesses. Header accesses are A - ev31, at 2 cycles each; on gcc that is at most about 13 M cycles.
     - A miss outside the band means the cost model is wrong, and it is reported as such.
  3. Delta run_cycles lies inside its row's [m = 0, m = 1 to MIG at 40] bracket, widened by 0.2 % of run_cycles of run
     noise on each side.
- **Reported, not graded:** m = ev29 / ev31, cycles per walk = ev30 / ev29, ev32 (expected 0), and ev33 (= grows).
- **Grader, committed before any pair-B reading:** `tests/rtl-smoke/sublet-p1-2026-11-15/pb-grade.py <row> run_flat
  run_paged ev23 ev29 ev30 ev31 [ev32 ev33]`. Exit 0 = checks 1-3 PASS, 1 = a check FAILS, 4 = NO DATA. Every line it prints
  carries "key 5 NOT MET".
  - Check 3 uses the brackets above as pre-registered, on G's ev23. The same bracket on the run's own ev23 is printed as
    information only.
  - Tested on synthetic inputs before use: PASS; check 1 alone FAILS; check 2 alone FAILS; checks 2 and 3 FAIL; check 3
    alone FAILS; the perl NULL row passes; a short argument list, an unknown row and a non-numeric count each give NO DATA.
- **Pair A (trial flat vs G flat):** the run_cycles delta is within 0.2 %. The only added work is one probe per
  namespace flush.

## Pair A on silicon (key 5 NOT MET)

P1 (d83d3a633, paging off) on bkit88 against G (9fd9d2530) on bkit86, board 1, runtime lane's native runner. Re-checked
here from the raw files in `spec/board/out-nat-{p1,g}-hpmnest-frcdkms-b1/<bench>/`.

- **Kits, one variable by content:** rootfs contents identical (128 files, `diff -rq --no-dereference` after extraction,
  a one-byte edit fires it). The kernel config and native-tree check are identical, but the Image was relinked (#164
  19:12 vs #171 23:30), so it is byte-different, and the firmware with it. Both relinks built the same 18 objects
  (initramfs, version stamp, kallsyms, vmlinux; the `native-kernel.log` build lines are identical). Bytes of the two
  17,468,928-byte Images, compared here:
  - 4,628 KiB is the initramfs gzip stream. It is 4,725,139 vs 4,725,612 bytes long, and decompresses to 12,551,680 bytes
    on both (cpio metadata differs; contents are identical, above).
  - Version strings at three places, and the GNU build-id note.
  - 129 bytes in 0x402788-0x42aee7, which are instruction immediates. Two decoded by hand (an `ld` offset and a `jalr`
    offset) both move by 472 bytes, which is the gzip length difference (473) aligned to 8. That is the expected shift
    for references to objects linked after the initramfs. Not every one of the 129 was decoded.
  - One data pointer at Image offset 0x10a8cc0: 0xffffffff80a90b28 vs 0xffffffff80a90d00, also +472. It points just past
    the gzip stream, so it points into the moved region, not from it.
  - Nothing else differs. Above the initramfs's page (0xa93000), only these strings, the build-id and that one pointer
    differ. The code there did not move.
  Resident bitstreams were read back as `…-tagoverlap6c_9fd9d2530.bit` and
  `…-pagedtbl_d83d3a633.bit`.
- **Both boots are UNGATED:** PREFLIGHT=0 on both. `spec-verdict.txt` reads "NO RESULT: no @@SPEC-DONE" on both: that is
  the SPEC-harness verdicter, which this native runner does not feed. The specdiff verdict is `read.log`'s.

### mcf (505.mcf_r test): PASS

- **Functional gate:**
  - `read.log`: specdiff 2/2 PASS, rc 0, on both.
  - inp.out, mcf.out and inp.err are byte-identical to G's (`diff -rq read/files`).
  - smoke 9/9 on both (task's two accounting checks WAIVED, as on G).
  - One TRAP LOG line on each, the final aperture (sw=255, 0x89), identical.
- **Cycles**, end minus begin of `@@NATIVE-CNT`, recomputed here; they equal the runtime lane's table:

  | counter | G | P1 | delta |
  |---|---|---|---|
  | run_cycles | 144,015,279,285 | 144,092,501,093 | +77,221,808 (+0.054 %) |
  | hpm8 (ev23) | 180,614,513 | 180,611,374 | -3,139 |
  | hpm4 (L1D) | 3,750,161,783 | 3,750,150,967 | -10,816 |
  | hpm7 (issue stall) | 80,267,261,637 | 80,332,542,400 | +65,280,763 (+0.081 %) |
  | between_cycles | 4,106,904,200 | 4,092,972,054 | -0.339 % |
  | runs | 48,930 | 48,940 | +10 |

- **Grade: +0.054 % is inside the 0.2 % band, so pair A's mcf row passes.** The +77 M is inside run noise (0.2 % is
  288 M), so it is not attributed. Pair B's mcf row may be graded once both its arms land.

### gcc (502.gcc_r train01, tuned stage, arena_pin=Y on both): PASS

- **Functional gate:**
  - `read.log`: specdiff 1/1 PASS, rc 0, on both.
  - The two output files (`train01.opts-O3_-finline-limit_50000.out` and `.err`) are byte-identical.
  - smoke 9/9 on both.
  - One identical TRAP LOG line each (sw=255, 0x89).
  - Bitstreams read back as for mcf.
- **Kit:** the same structure as mcf.
  - rootfs contents are identical (128 files).
  - The Images are relinked (#168 vs #172). The gzip stream is 447 bytes shorter, and both unpack to 22,782,464 bytes.
  - Below the stream, 134 bytes of immediates differ in 0x402788-0x42aee7. Above it, the shifted init tail, two version
    strings, the build-id and one pointer page.
- **Cycles**, recomputed here; they equal the runtime lane's table:

  | counter | G | P1 | delta |
  |---|---|---|---|
  | run_cycles | 41,536,277,321 | 41,559,126,329 | +22,849,008 (+0.055 %) |
  | hpm8 (ev23) | 6,409,096 | 6,408,717 | -379 |
  | hpm4 (L1D) | 367,122,291 | 367,145,747 | +23,456 |
  | hpm7 (issue stall) | 10,629,521,357 | 10,639,408,003 | +9,886,646 (+0.093 %) |
  | between_cycles | 1,816,335,961 | 1,821,258,166 | +0.271 % |
  | runs | 13,643 | 13,647 | +4 |

- **Grade: +0.055 % is inside 0.2 %, so pair A's gcc row passes.**
- mcf and gcc both read about +0.055 % with issue stall up by about 0.08-0.09 %. Each is inside the band, so it is not
  attributed. Two rows with the same sign are not a measurement of a walker cost in flat mode.

### perl (500.perlbench_r test, makerand, the NULL row): PASS

- **Functional gate:**
  - `read.log`: specdiff 1/1 PASS, rc 0, 558 hashes, on both.
  - The two output files are byte-identical.
  - smoke 9/9 on both.
  - One identical TRAP LOG line each.
  - Bitstreams read back as above.
- **Kit:**
  - rootfs contents are identical (683 files).
  - The Images are relinked (#169 vs #173). The gzip stream is 22 bytes shorter, and both unpack to 23,700,992 bytes.
  - The byte differences outside the stream are in the same places as gcc's.
- **Cycles:**

  | counter | G | P1 | delta |
  |---|---|---|---|
  | run_cycles | 6,755,758,710 | 6,753,096,627 | -2,662,083 (-0.039 %) |
  | hpm8 (ev23) | 30,296 | 30,422 | +126 |
  | hpm4 (L1D) | 91,413 | 98,053 | +6,640 (about 1 per million cycles) |
  | runs | 1,140 | 1,142 | +2 |

- **Grade: -0.039 % is inside 0.2 %, so pair A's perl row passes.**

### Pair A: PASS on all three rows (key 5 NOT MET)

With paging off, P1 is output-identical to G on mcf, gcc and perl makerand. Each row's run_cycles lies within 0.2 %:
+0.054 %, +0.055 % and -0.039 %. Pair B may now be graded (Addendum 149).

- **Checkers, both run here:**
  - `SPEC_BOARD_DIR=<spec/board> tests/rtl-smoke/sublet-p1-2026-11-15/pa-check.py <bench>` checks the functional gate, the bitstreams, the counters and
    the band. It was tested to fail on another bench's outputs and cycles and on the wrong bitstream, and to report NO
    DATA on a missing directory.
  - `pa-kit-check.sh <bench>` compares the rootfs contents and the Image bytes.
