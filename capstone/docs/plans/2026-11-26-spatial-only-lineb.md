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
