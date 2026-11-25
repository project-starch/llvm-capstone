# Spatial-only core on Line B: what the parameter removes, and what software sees (2026-11-26)

**Why.** The campaign's +Spatial row must come from the same core family as Full. Otherwise the difference measures P1 against B5b-2, not lifetime checking (the semantics owner's request). The base is the final B5b-2 candidate, capstone-ariane b5b2-prefix b42c8c644 (core = 7861abcfa; seal sha256 6a09f34d…190a).

**Prior art.** This is P1's spatial-only core, steps A-C (capstone-ariane spatial-only-c 321af31a3; plans 2026-11-20-spatial-only-core.md and 2026-11-21-spatial-only-core-c.md; on silicon, dev 90e0489ff109):
- one define, `CAPSTONE_SPATIAL_ONLY`, set in the config package. `+define+CAPSTONE_LIFETIME_ON` restores the full core on the same tree for simulation A/B;
- the lifetime unit replaced by a stub with the same ports, each request answered "complete, no effect" from flops, one cycle later;
- every verdict reads live;
- the PC tracker never holds.

What carries over:
- **Step C's retraction:** a census by NAME missed `__mrev()` / `__revoke()` raw encodings, so any illegal set here is taken by ENCODING.
- **Step A's missed prediction:** parked tests turn saved cycles into counts.

**What differs on Line B.** The lifetime unit is `capstone_sublet_life` (instance `rev_node`), not P1's generated rev_node. Its ids are 63 bits, not 55. It has the Sublet channel meanings (capstone_sublet_life.sv :15-27) and extra ports (irq_pending_i, ns_key_i, st_drained_i). So P1's stub is a model, not a file to reuse. The decoder needs no change: every op P1 had to make illegal (REVOKE, MREV, DROP, DELIN, SPLIT, CSRETIRE, CSNODE, CALL, RETURN, SEAL, TIGHTEN) is already illegal under the Sublet format.

## What the parameter removes (the RTL)

| part | b42c8c644 | under CAPSTONE_SPATIAL_ONLY |
|---|---|---|
| the lifetime unit `capstone_sublet_life` (5,756 LUT / 1,910 FF at b42c8c644's gate) | the Sublet table: derive, kill, renew, create, query, probe, the 4(b) chunked walk and its saved position | a stub `capstone_sublet_life_stub`, the same port list; each channel accepts from a flop and replies from flops one cycle later; the memory channel idle; debug outputs 0 |
| the table walker and record path in ex_stage (pw_*, the unit's memory ports) | the unit's reads and writes, through the page walker to D$ port 2 / the write port | never requested; left for synthesis to prune (P1 step A: synthesis removed most of the dead logic itself) |
| the LSU's revocation cache (four 64x63 RAMs, 1,008 LUTRAM), ALLOW/DEAD records, the probe FSM and timeout | the per-access verdict: hit-dead, same-cycle invalidation, probe DEAD, timeout (all 25), and the R-43 replay marker | `rvc_lu_hit = 1, rvc_lu_nodeval = 1, rvc_inv_now = 0`: every access live, no miss, no probe, no 25, no replay. The cache, records and probe become dead logic |
| the DYN unit's lifetime ops | CDERIVE_S, CREVOKE_S, CRENEW, CSMINT's create, LCC's query, each waiting on the unit | unchanged; they wait one cycle on the stub |
| the store-queue drain before a lifetime op (4(b), st_drained_i) | lifetime ops wait for older stores to drain | unchanged (a few cycles; it guards the table, which no longer exists, and is not worth a second variable) |
| the PC tracker (commit_stage) | armed only in M with capmode, constant 0 under Sublet | unchanged: already inert on Line B |

## What software sees

The stub's answers, chosen so that every lifetime op "completes with no effect" in the sense of P1's ruling, applied to the Sublet meanings:

| op | Full (b42c8c644) | spatial-only |
|---|---|---|
| CDERIVE rd, rs1, rs2 | 24 / 29 checks, the unit's mrev check (25 dead, 29 bad table), then derive: a child with a NEW id; 30 when out of slots; the no-fault failure writes an untagged 0 | the same 24 and 29 operand and bounds checks; never 25, 30, the table's 29 or the untagged-0 failure; the child carries **the parent's id** |
| CREVOKE rs1 | 24 for untagged; the run is killed (every descendant's later access takes 25); 29 / 25 from the unit; may stop PARTIAL on an interrupt | 24 for untagged; otherwise **completes and does nothing**: every reference stays usable, and there is no PARTIAL |
| CRENEW rd, rs1 | a new id for the same object (the old one dies); 30 / untagged-0 failure | rd = rs1 with **the same id**; the old reference stays live; never fails |
| CSMINT | the value refusals (load_store_unit.sv :2189-2190: bounds order, perms, cursor ≥ 2^38), then a record written to the table and the minted root returned | the same value refusals; **no record is written**; the root carries a fixed id (see below) |
| LCC rd, rs1, 0 (validity) | the unit's query: 1 live, 0 dead | always **1** |
| LCC, other selectors | unchanged | unchanged |
| every load, store, LDC, STC, fetch | the verdict: 25 on a dead id; a miss replays through commit | **never 25 from lifetime**; no replay; bounds (the format), tags and the FLU rules unchanged |
| srevroot (CSR 0x5C1) | the table root; a change is a namespace flush | still a CSR that flushes the pipeline on write; it names nothing |
| the lifetime table in memory | read and written by the unit | **never touched by the core** |

**The fixed id for CSMINT's root:** {generation 0, slot 3}, which is the reference's first root slot after the header (the tests' {0,3}).
- Stub replies echo the requester's id everywhere else.
- A CSMINT has no requester id, so it needs a constant.
- It must be non-zero, because 0 is CDERIVE's no-fault failure.

**What software must provide** (the semantics owner's and the compiler lane's half):
1. **Any software that reads or writes the table itself.** CSMINT no longer writes its record, and no op maintains the list. A kernel module, the monitor or a firmware emulator that walks records (a CSCHECK emulation, for instance) would read whatever the software last wrote there, not the core's view. That needs a census by symbol AND by encoding of table readers in every image the +Spatial run boots (step C's lesson), and a decision per reader: stubbed in software, or harmless.
2. **The QEMU twin, under the same rule.** CDERIVE's child takes the parent's id; CREVOKE and CRENEW have no effect; LCC validity is always 1; no 25 from lifetime; CSMINT writes no record. Without it, a twin comparison is invalid for every program that revokes and then uses.
3. **The userland.** The SDK's __heap_mapping / __heap_unmapping (CDERIVE and CREVOKE) need no change: they complete. Programs whose correctness relies on a 25 after free (the Juliet temporal controls) change verdict by design, and are not +Spatial rows.

**What it does not change:** the Sublet format and its compact bounds (exact below 2^12 bytes, an 8-byte granule above), tags, the FLU's untagged-x[rs1] rule, CSMINT's value checks, the 2^38 cursor rule, the trap route, and R-80's pop. R-80's replay-marker trigger disappears with the replay marker. R-77 and R-81-R-83 cannot arise: there is no probe, and the stub never waits on memory.

## Predictions (written before any build)

- **Lint:** every counter at b42c8c644's baseline, except UNUSEDSIGNAL. That rises by the stub's unread inputs and falls by the now-constant verdict paths, named when read. UNOPTFLAT is 39; a member may be renamed, as in P1 step A where the D$ read-port ack loop was renamed.
- **The 95-sweep against b42c8c644's own sweep, by name:**
  - IDENTICAL in values and RVFI: every test that issues none of CDERIVE, CREVOKE, CRENEW, CSMINT, LCC selector 0, and has no verdict miss.
  - MAY DIFFER: the lifetime tests (the b5b2 lists' crev*, sbdrain*, srvcost, sref*, sublet-*, csmint*, ldcexc2 and its walk variant).
  - Cycles may differ in every test that probes or issues a lifetime op, and parked tests may differ in counts (step A's miss).
- **The B5b-2 and B5b-1 lists (57):** each read with its Full value beside it. Predicted differences: every arm whose value is a 25 from lifetime, a new id, a free-list position or a table checksum.
- **Synthesis-only gate**, against b42c8c644's gate:
  - mapping in class (within 10 % of 4,968 MB), 0 loops;
  - core LUT -6,500 +- 2,500: the unit 5,756 to a stub of a few hundred, plus the LSU's cache and probe, minus synthesis's own pruning (P1 step A -7,774);
  - FF -2,600 +- 1,000;
  - LUTRAM -1,008 (the four RVC RAMs);
  - per-module: rev_node (the stub) under 500 LUT, lsu_i -1,000..-3,000.
  - A miss outside these bands is a reported deviation with its per-module census, not a stop.
- **Full build:**
  - placed LUT 150,303 - 6,500 +- 3,000 and FF 68,506 - 2,600 +- 2,000;
  - routed clean, congestion <= 5, 0 loops;
  - clk_out1 inside [-1.47, +6.01]; the rev_node worst-500 count falls toward 0.
  - Slack is predicted inside the band, not better. P1's spatial-only seal moved WNS by +0.17 against the null band, and this core's worst family (issue read operands to fu_data) does not run through the removed logic.
- **Silicon key:** the board's Line B boot (kernel, monitor, the SPEC chain) with the twin under the same rule. The first silicon question is a boot that completes. A +Spatial row is graded only after the table-reader census in point 1 is closed.

## The 57 Line B tests: what MUST change, predicted before the run (2026-11-26; the tree is capstone-ariane b5b2-spatial 5ff493d0a)

The base is the same 57 tests on 7861abcfa (b42c8c644's core): s2pre-wt `.sup/pf` (B5b-2 list, 40) and `.sup/pf1` (B5b-1 list, 17). The run is the committed spatial tree, plus the same tree with `+define+CAPSTONE_LIFETIME_ON` as the control, which must reproduce the base in every value.

**The rules, from source.** Every cause 25 reachable on Line B is a lifetime verdict:
- the LSU's four verdict arms (load_store_unit.sv :1545-1627, :1705-1715) need a hit-dead, a same-cycle invalidation, a DEAD record or a miss, and none survives `rvc_lu_hit = rvc_lu_nodeval = 1, rvc_inv_now = 0`;
- the DYN unit's reachable 25s are the mrev, derive, kill and init statuses (capstone_dyn_unit.anvil :636, :651, :689, :718), which the stub answers 0;
- the FLU raises none;
- under DYN_BYPASS, LDC and STC do not reach the DYN unit's 25s;
- the PC check (commit_stage.sv :449) needs capmode, which is 0.

So:
- R1: every cause 25 becomes no fault (0x0 where a cause is printed), and every trap count falls by the number of 25s it counted.
- R2: a derived or renewed capability's id prints as its PARENT's id, and every CSMINT root's rd as 0x3. An LCC-8 id print is {generation[31:0] << 31 | slot}, so a child of the root {0,3} prints 0x3.
- R3: CREVOKE has no effect. A later CDERIVE from the "revoked" object succeeds, and a later access through it reads memory.
- R4: table memory is never written by the core. Every print of a table word (header, pool, record) or a checksum over the table pages reads what software wrote, and differs from Full.
- R5: no 30 (out of slots) and no untagged-0 failure.
- R6: a timed lifetime op or verdict miss gets cheaper. A CREVOKE costs tens of cycles, not 0x37f0.

**MUST-SAME, every value and SR:**
- b5b2-illegal, -illegal-runv, -illegal-csup, b5b2-ldchold, r79-cpmp: no lifetime op;
- b5b2-untagged: the FLU rule; its CDERIVE's 29 is the format's, which comes after the stub's live check;
- sublet-fmt, csmint-pmp, csmint-pmp-vm: one root at {0,3} in Full too, and no 25;
- b5b2-ctxport-f4ctl, -widehi, -widelo: no 25 and no id printed;
- b5b2-cptag, -cptag2, -raw, -rawlcc, -rule2, -rule2-pair, -scratch: CDERIVE without revocation, no 25, no id print;
- b5b2-mtrap-s, -u, -call, -plain-s, -plain-u, -plain-call, -plain-detect: their status words (position 1). The timer-trap counts (positions 2-3) are timing and are recorded.

**MUST-CHANGE** (position: Full → predicted; positions not listed are predicted unchanged):

| test | prediction |
|---|---|
| sublet-life | 4, 7, 11, 12: 0x19 → 0x0 (R1, R3); 15-34, the pool and header words: differ (R4); 35 recorded |
| b5b2-crevint | 2: cycles ≪ 0x37f0 (R6); 3-5: 0x80000007 0x800000cf 0x800000ce → 0x3 0x3 0x3 (R2); 6: 0x19 → 0x0 (R3); 11: checksum differs (R4); 12: 0x1 → 0x0 (R1) |
| b5b2-crevint-mt | as crevint at 2-6, 11, 12; 8-9 (timer traps, at the site) recorded. The arm's own VOID rule (no trap at the site) may now fire, because the CREVOKE lasts a few cycles |
| b5b2-crevint-sh | 3-5: 0x3 0x3 0x3; 6, 7: 0x19 0x19 → 0x0 0x0; 11 differs; 12: 0x2 → 0x0; 8-10 recorded (the arm may go VOID as -mt) |
| b5b2-crevsv-none, -kill | 3-5 → 0x3 0x3 0x3; 6: 0x19 → 0x0; 11 differs; 12: 0x1 → 0x0; 8-10 recorded |
| b5b2-ctxport-f4 | 1: 0x19 → 0x0, 2 → 0x0 (no fault), 6: 0x1 → 0x0; 3-5 read the data (recorded) |
| b5b2-mwrite | 3: 0x19 → 0x0; 8: 0x2 → 0x1 |
| b5b2-nsedge | 3, 7, 9, 11, 14, 15: 0x19 → 0x0; 17: 0x6 → 0x0 |
| b5b2-uprot | 3, 6, 13: 0x19 → 0x0; 19: 0x6 → 0x3 |
| b5b2-uprot-mis | 3, 6, 12, 17: 0x19 → 0x0; 23: 0xa → 0x6 |
| b5b2-sref | 12-13 (LCC 8 of x23 and of x26): 0x6 0x6 → 0x3 0x3 |
| b5b2-sref-gen | 12-14 (the generation-bearing ids): → 0x3-based values, differing from Full (R2) |
| b5b2-sbdrain-fill | 0x1e 0x0 0x100 0x1 → 0x0 0x0 0x3 0x0 (R5, R2, R1) |
| b5b2-sbdrain-sret | 1: 0x1 → 0x0 (no 30); 3: differs (R2) |
| b5b2-sbfwd-hold-cap | 0x0 0x100 0x0 → 0x0 0x3 0x0 |
| b5b2-srvcost | 1-4 (D1-D4) within a few cycles of Full; 5-10 (P1-P6): all within a few cycles of each other, so P3 no longer exceeds P2 (no refill after a namespace change, R6); 13 the trap count unchanged |
| csmint | 4: 0x6 → 0x3 (the second root, R2); 35: 0x19 → 0x0; 36: 0x80000007 → 0x3; 41-50 table words differ (R4); 51: 0xd → 0xc |
| csmint-vm, -vm-noncanon | 3: 0x6 → 0x3; 29-30: 0x3 0x6 → 0x3 0x3; 26-28 table words differ |

**MAY-CHANGE, recorded and not predicted:** sublet-ns and its six variants, va-runv-snap, b5b2-mtrap-rules, b5b2-ctxport-edgem. They use CREVOKE or LCC with prints whose meaning I did not trace before writing this.

**Grading:**
- a MUST-SAME test that changes, or a MUST-CHANGE position that stays at its Full value, is a FAIL against this list, each investigated;
- a MUST-CHANGE position that changes to a value other than the one predicted is a MISS, reported with the value.

## The 57 tests, read on 5ff493d0a (2026-11-26)

**A VOID first run, recorded.** The first spatial run (`.sup/sp`) is VOID.
- An earlier, unregistered run in the same worktree kept simulating: my stop had matched `docker ps`'s truncated Command column, so nothing was stopped.
- Its trace interleaved with the registered run's shared trace file. Eight B5b-2 logs came back spliced: another test's tail, or no boot line.
- The B5b-1 list and the control ran after it ended, and are as clean as the base by the same splice check.
- The rerun (`.sup/sp2`, `.sup/sp21`, the tree checked idle with `docker ps --no-trunc` first) has exactly one boot per log in all 57. Every reading below is from it.

**The control:** the same tree with `+define+CAPSTONE_LIFETIME_ON` (`.sup/spc`, `spc1`, the define confirmed in the model's file list) equals the base, 57/57 in every value. So every difference below is the spatial mode's.

**Against the predictions above.**
- **MUST-SAME held for:**
  - illegal (3), ldchold, r79-cpmp, untagged, sublet-fmt, ctxport-f4ctl;
  - cptag, cptag2, raw, rawlcc, rule2 (2), scratch;
  - mtrap (8, including the timer counts);
  - the MAY-CHANGE group: sublet-ns (8), va-runv-snap, mtrap-rules, ctxport-edgem.
- **MUST-SAME that changed (prediction errors; the stub is right):**
  - csmint-pmp and -pmp-vm position 15 (0x2 → 0x0) is `ld 24(0xbff00000)`, the pool header's allocations word: a table print (R4) I had not seen.
  - ctxport-widehi and -widelo positions 4-5 are the derived children's raw id words (slot << 1). They echo their parent {0,2} as 0x4 (R2). On this core those arms no longer exercise the widened slots, by design.
- **MUST-CHANGE, held:**
  - crevint, crevsv-none and -kill exactly: ids 0x3, the dead-o 0x0, the checksum differs, M traps 0;
  - CREVOKE's cost 0x68 / 0x69 / 0x32 cycles against 0x37f0-0x468e;
  - ctxport-f4, mwrite, nsedge, sbdrain-fill, sbfwd-hold-cap exactly;
  - srvcost's P3 = P7 = P2 = 0x83, no refill;
  - csmint position 35 0x19 → 0x0, 36 → 0x3, 41-44 the pool words, 51 0xd → 0xc;
  - sublet-life positions 4, 7, 12 → 0x0, and 15-34 the table.
- **MUST-CHANGE, other values than predicted (prediction errors; the stub is right):**
  - **A 25 is not always replaced by success.** The reference checks the tag, then liveness, then type, rights and bounds. An access that was both stale and out of bounds now reaches the bounds check: uprot and uprot-mis position 6, 25 → 28. A stale and misaligned access goes on to 4 (uprot-mis 12). A dead-parent CDERIVE goes on to its size check, 29 (sublet-life 11). The trap counts fall by the 25s that became no fault: uprot 6 → 4, uprot-mis 10 → 8.
  - **The root in the table tests is {0,2}**, the physical pool's root x18, not CSMINT's {0,3}. sref and sref-gen's ids read 0x2 (positions 9-10, 12-13), and their check flag (position 15) still holds. sref-gen's position 14, the table's generation, reads 0 (R4).
  - **Positions I had misread:**
    - sbdrain-sret's position 3 is a TYPE word (1 = allocated), which stays 1;
    - csmint's position 4 is the first root's raw hi word (3 << 1 = 0x6), which stays. The second mint (positions 6 and 8) moves 0x4 → 0x3 and 0x8 → 0x6;
    - csmint-vm position 3 stays for the same reason. Its 20-21 move as csmint's 6 and 8, and 26-29 are the pool header.
  - **crevint-mt and -sh** read 0 timer traps at the site (position 9). The arm's own VOID rule fires, as the list warned: a CREVOKE of a few cycles leaves no room for a timer.
- **srvcost's D1, TRACED (2026-11-26):** D1 is 64 sscratch writes in S; it reads 0x591 on Full and 0x379 here.
  - Full's D1 window (cycles 46,233-47,654 of the base's trace) contains 8 write-buffer evictions. Here it contains none.
  - The 8 entries, by their DC-WB-ALLOC addresses: four record words at 0x80009030-0x80009260 inside tbl, the table header's two quadwords at 0x80006000/0x80006010, and two of the test's own stores (arr[30], arr[31]).
  - So on Full the lifetime unit's table writes from the setup's CDERIVEs are still draining when D1 begins, and D1's CSR writes wait behind them. The stub writes no table, so nothing is in flight.
  - This is a real cost of lifetime checking, and the +Spatial row will show it.
  - P1 is the first U pass, where every verdict misses on Full and none here.

No stub defect was found: every deviation is explained by the stub's stated semantics, and D1's by the trace.

**Clarification of the gate's memory key, before any reading (2026-11-26, asked by the board lane):** "mapping in class (within 10 % of 4,968 MB)" is ONE-SIDED. The binding key is a technology-mapping peak of at most 5,465 MB, under the 40 GB guard: it guards against a blow-up. A peak below 4,471 MB is REPORTED as a deviation and does not fail the key. Removing the lifetime unit is expected to lower it, as P1's spatial step A did (4,914 against 5,042 MB).
