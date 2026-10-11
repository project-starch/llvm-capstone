# R-12 — revocation nodes were never reused: the pool ran out at 65,532 allocations and the core DEADLOCKED. Fixed by the M1 reclaimer (`054cea69b`), on silicon since 2026-09-17

**What it is.** Every capability carries a revocation-node id; every mint (`SPLIT`, `MREV`, `INIT`)
takes a fresh node from a pool of 65,536 sixteen-byte records, and **nothing ever gave one back**. The
allocator was a bump pointer: at the sentinel it dropped the request, the dynamic unit blocked forever on
its reply, and the core wedged — deliberately, to avoid aliasing a live node, but indistinguishable from
any other wedge (see M-1). A second, separate half: the revoke walk kept dead nodes linked, so every later
revocation re-read them at 3 cycles a node. This is the study called **M1** — "M1 the reclaimer", not to be
confused with **M-1** the trap-vector defect (`../RTL-domain-trap-vector-unset/`).

**Siblings and nested issues, so a reader with a neighbouring symptom is redirected:**
- **R-35** (`../R35-revoked-reference-retains-authority/`) was FOUND on this fix's bitstream, by the M1
  harness: the reclaimer reuses storage correctly, but the LSU's revocation check did not fire, so a stale
  reference read the new occupant. Not this defect; the trackers' adopt is theirs.
- **R-27** (`../R27-revnode-orphan-response/`) is a flush orphaning a rev-node response — a hang in the same
  unit for a different reason; its fix is an ancestor of this build.
- **R-34 / R-24** (exception delivery, cause encoding) ride in the same bitstream through the merge
  `f714d2a72`; they are not part of the reclaimer.
- **RTL-give-cost-tracks-global-placement** is the release-cost artefact measured on this bitstream.

Registry: `docs/ref/ISSUES.md` R-12 (the dated blocks from 2026-09-16 on). Design as implemented:
`docs/plans/2026-09-16-revnode-reclamation-v2.md` with its audit banner. Lineage: `capstone-ariane`
`m1-reclaimer`, commits `f1331daed` (splice) → `b49673357` (Part B) → `35081fdb9 … 054cea69b` (A1–A6).

**Record, by date:**
- **2026-09-10:** characterised on the flashed RTL: the pool is 65,536 nodes (not 1,024), the failure is a
  visible hang (not silent id reuse); the "99.3 % consumed" board reading withdrawn (a dead aperture).
- **2026-09-15/16:** two audits reject the first design (no mechanism); the v2 spec is audited and three of
  its mechanisms corrected before implementation.
- **2026-09-16:** splice (cost half) measured; Part B makes exhaustion a fault (cause 30); the reclaimer
  A1–A6 proven in simulation by a matched pair; A6 corrects A5's "safe" claim (a generation double-add).
- **2026-09-17:** `054cea69b` synthesised — WNS −8.307, best on record, loops 1 (the loop drop is the
  merge's, not the reclaimer's: resolved 2026-09-18 by building `f714d2a72`); flashed as
  `caplifive_m1_054cea69b.bit`. Silicon: 200,031 mints from a 65,532-index pool, so reuse is demonstrated.
- **2026-09-17 … 21:** the M1 study runs on it (1,314,737 allocations in one boot, cost curves flat) and
  exposes R-35.

## The defect, in two pictures

**Capacity.** The id was `{14'd0, head[15:0]}` and `head` only ever went up:

```
   rev-node pool: 65,536 records x 16 B at 0xBFF00000        id = index, address = base + (id << 4)
   +---+---+---+---+---+---+---+-- ... --+-------+
   | 0 | 1 | 2 | 3 | 4 | 5 | 6 |         | 65535 |   0..2 sentinels, 65535 = REVNODE_SENTINEL
   +---+---+---+---+---+---+---+-- ... --+-------+
                 ^ head: bump by 1 per mint, NEVER decremented, nothing is ever freed
                 |
   mint #65,533:  head == 65535 -> the request is DROPPED, no init_res is ever sent
                  -> SPLIT/MREV blocks forever on recv  ->  the core WEDGES (no trap, no return)
```

A REVOKE invalidated its subtree's nodes (`valid := 0`) but left them linked and allocated. Under M-1 the
wedge was indistinguishable from every other wedge, and on this hardware the head register cannot be read
on a healthy boot (the aperture returns the sentinel), so nobody could measure how close a workload came.

**Cost.** Invalidated nodes stayed on the tree, so every later walk crossed the corpses:

```
   revoke r:   r -> [dead] -> [dead] -> [dead] -> ... -> [dead] -> next live subtree
                    each corpse = one dependent 16-byte read: 3.000 cycles/node at L1 hit (lower bound)
   S1 ladder (S12_MEM_DELAY=12):  N=8: 273   N=160: 729   N=1,024: 3,321   N=3,072: 96,822 cycles
   P1 on silicon: release cost rose ~12x within a domain while minting rose ~1.5x
```

## The fix — three parts, one bitstream

**1. The splice (`f1331daed`, cost half).** At the walk's exit the index in hand is the first node outside
the revoked subtree, so one splice unlinks the whole dead run in **two writes, independent of run length**:

```
   before:  parent -> r -> d1 -> d2 -> ... -> dn -> L       (walk crosses d1..dn every time)
   after:   parent -> r ------------------------> L         (two writes at walk exit; d1..dn unlinked)
   spliced marginal cost per dead node: 0.000  (328 cycles at every N; crossover with unspliced at N ~ 26)
```

**2. Part B (`b49673357`): exhaustion is a FAULT, not a hang.** The unit answers with a two-bit status and
the dynamic unit raises cause 30 (`INSUFFICIENT_SYSTEM_RESOURCES`). Matched pair: the pre-change tree
runs to the 3,000,013-cycle ceiling with the 65,533rd MREV never retiring; the fixed tree traps at exactly
65,532 mints. Prerequisite for the reclaimer: two of its audit corrections need an operation to be
*refused*, and no response channel could say no before.

**3. The reclaimer (A1–A6, capacity half).** Freed nodes are reused, and a stale reference to a reused slot
is told apart from the new owner by a **generation** stored in the node and carried in the id:

```
   node record (94 bits, same slot, same positions for prev/next/valid/linear):
     was:  depth[31:0]                        | prev | next | valid | linear
     now:  free[1] | generation[14] | depth[17] | prev | next | valid | linear     (A1: inside the old depth)

   id = (generation[13:0], index[15:0])       address and every LINK use index[15:0] only (A2, A6)

   REVOKE walk invalidates node i  --->  push: free := 1, next := free_head, free_head := i     (A3)
                                          (index > 2 and generation < 16383 only)
   mint  --->  free list non-empty?  pop i: generation+1, free := 0, WRITE BACK  -> id = (g+1, i)  (A4)
                                          (that write is a node write with valid = 0, so the existing
                                           invalidation tap BROADCASTS index i: every tracker drops (g, i))
               else                    bump head as before; head at sentinel -> cause 30 (Part B)

   every use of a reference (g, i) at the unit -- query, drop, delin, mrev, init -- is honoured only if
        node[i].valid  AND  node[i].generation == g                                                 (A5)
   a slot retires for good at generation 16383 (never wraps); the trackers' adopt guard stays 30-bit,
   so a fresh (g+1, i) is re-adopted over a tracked (g, i), while the broadcast compare is 16-bit.
```

**The approval test is a matched pair, not a green suite.** `1ac15c4ef` (A3/A4) is the deliberately
**gen-blind control**: it reclaims but checks no generation, so a retained stale reference succeeds
against the slot's new owner (stale LDC/MREV/SPLIT/DROP all cause 0, the fresh owner's `LCC` reads 0,
4 traps). `d9620b907` (A5) on the same tree with the same fixtures flips every one of those arms to
**refused, cause 25**, the fresh owner reads 1 (8 traps). **A6** (`054cea69b`) then corrected A5's own
subject line: an audit found a composed id written into a neighbour's LINK, so the allocator added the
generation twice and generations could SKIP — and a skip over 16383 wraps, which is the one condition
under which a stale reference becomes acceptable again. No fixture had reached it; `r12-recl-composed-link.S`
constructs it. Links are bare indices at both ends now.

## What the silicon showed

```
                                   pre-fix (flashed 1bfff7776 lineage)      caplifive_m1_054cea69b.bit
   mints in one invocation          wedge at #65,533 (deliberate stall)     200,031 from a 65,532 pool -> reuse
   allocations in one boot          impossible past the pool                1,314,737 (20.1x the pool)
   take cost per allocation         --                                      72.03 cycles, sd 0.01, flat to 651,264
   accounting                       --                                      minted - revoked = 31 on EVERY snapshot
   revoke over dead nodes           3 cycles per corpse                     flat (the splice)
   exhaustion                       hang                                    cause 30 (sim); not reached on silicon
```

Full readings: `docs/ref/fpga-silicon-measurements-for-paper.md`, "M1 on silicon". Which image ran was
settled by a **label-independent** argument (a run that mints 200,031 nodes from a 65,532-index pool and
completes cannot be a non-reclaiming build) after the console proved unable to name the resident image
by hash — the incident that produced the three-part form of the "cite by hash" rule.

## Residuals — stated, not claimed away

- Only the **walk** frees nodes: a DROP'd node stays allocated (unlinked-but-dead is not reclaimable), a
  handle never frees itself, and an index retires permanently at generation 16383. The leak fraction is
  set by the handle rule, not the hardware.
- `depth` counts MREVs on one node (not tree depth) and is 17 bits; MREV refuses at 131,071 rather than
  carrying into `generation[0]` — the field order was chosen so that a carry could only false-deny.
- The **trackers** (LSU, CPMP, PC) still adopted an unseen id as valid — that is what R-35 measured on this
  bitstream, and R-44 is its S/U-mode remainder.
- Cost: +1 cycle per allocation (the allocator's call boundary), a pop 6 cycles more than a bump.

## Evidence

| what | result | where |
|---|---|---|
| Part B pair (`r12-pool-exhaust`) | 65,532 mints then cause 30, 3 traps; control runs to the cycle ceiling, MREV #65,533 never retires | ISSUES.md R-12, block "R-12's HANG is gone" |
| approval pair (`r12-recl-stale`, `r12-recl-drop-not-free`, `r12-recl-composed-link`, retirement run) | control: stale succeeds, 4 traps; A5/A6: all refused, 8 traps; index 3 reclaimed 16,384 times then retired; 0 traps in 16,387 pops | ISSUES.md R-12, block "THE RECLAIMER … PROVEN IN SIMULATION"; commit messages A3–A6 |
| S1 walk ladder | unspliced 3.000 cycles/dead node exactly; spliced 0.000 | ISSUES.md R-12, block "S1 LADDER" |
| synthesis `054cea69b` | exit 0, WNS −8.307, loops 1, 168,757 routed LUTs (456 below the flashed base), LUTLP-1 = 0 | ISSUES.md R-12, block "S1: THE RECLAIMER SYNTHESISES" |
| silicon, boots 2026-09-17 … 21 | the table above | `docs/ref/fpga-silicon-measurements-for-paper.md`, "M1 on silicon" |

This is a source-level package: the fixtures are `verif/tests/custom/capstone/r12-*.S` in `capstone-ariane`
at `054cea69b`, and every number above is reproducible from that revision with the `cva6-build-rv` container
at `S12_MEM_DELAY=12`, `--sv_seed` pinned. No binary is carried, and none is needed.
