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
- **95-sweep against G:** PENDING (sim14 `pwsw6`, TAGMLP_PROBE). Predicted: values and RVFI identical. Cycles are identical
  except tests that enter the table namespace, which pay one probe per flush. No sweep test builds a paged table.

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

## Board pre-registration (PENDING the event-23 counts, before any synthesis request)

- **Pairs, one bitstream:**
  - A = trial on the flat kit against G on the flat kit (the null control);
  - B = trial on the paged kit against the trial on the flat kit (the paging cost, at its lower end, with record pages
    consecutive);
  - optional C = scattered record pages, the collaborator's call.
- **Inputs in hand** (the compiler lane's twin census, FRCDKMSa3; the per-op counts from the walker trace above):

| row | CDERIVE | CREVOKE | next | record-page span | grows |
|---|---|---|---|---|---|
| mcf | 60,282 | 60,238 | 8,256 | about 33 | 2 |
| gcc train01 | 1,088,904 | 1,036,504 | 73,574 | about 288 | about 17 |
| perl makerand | 921 | 141 | 1,974 | 8 | 0 |

- Per-op record accesses through the walker: CDERIVE 9-10 and CREVOKE 11, of which 2-3 are header slots (no TLB).
- Formula: paging cost per row = N_acc x 1 (the registered lookup) + walks x 3 x c_dir + grows x (escape + 64 KiB zero-fill
  + resume).
  - N_acc = about 10 x (CDERIVE + CREVOKE) + the verdict-cache-miss reads (event 23, the board's G run tonight);
  - walks = m x N_acc for the non-header share;
  - c_dir is 4-8 cycles on an L1D hit and 25-40 to the MIG.
  - Events 30 and 31 read the two cost terms directly, so the pre-registration also states a self-consistency band:
    pair B's run_cycles delta against ev30 + ev31.
