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
- **Pair A (trial flat vs G flat):** the run_cycles delta is within 0.2 %. The only added work is one probe per
  namespace flush.
