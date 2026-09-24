# R-43 — resolve a revocation-cache miss by asking the rev-node unit, instead of denying

*Plan, RTL lane, 2026-09-25. Branch `r43-query-on-miss` in capstone-ariane, off `6cbdaeeb4` (R-35 fix + R-42).
Audited before implementation and again after, as standing practice for this fix lineage.*

## Why

R-35's fix (`4ad0df694`) allows an M-mode LSU access only when its exact 30-bit `{generation, index}`
is resident in a 4-way x 64-set cache marked live, and **denies on a miss**. That is fail-closed and
safe, but a LIVE capability whose entry was evicted is refused. The cost is now real:

- **On silicon**, both R1 harness runs trapped cause 25 on `caplifive_r42_6cbdaeeb4.bit` (boots r42b3,
  r42b4). In r42b3 a globals capability was allowed at `+0x4f40` and denied at `+0x4ff4`, a few
  instructions and one `mrev` later. The emulator runs the same invocation to completion.
- **In RTL simulation**, `verif/tests/custom/capstone/r43-evict-live.S` (capstone-ariane `93f509f54`)
  reproduces it on `6cbdaeeb4`. An alias reads fine after 16 new live nodes, and traps 25 after 512
  more, while `LCC` says it is live.

So no workload that churns more than a few hundred revocation ids can run on the R-35-fixed silicon,
and that includes the paper's R1 measurements.

**The same test shows the remedy's mechanism.** After the `LCC`, whose node read passes the cache's
read tap, the same alias reads fine again. A node read resolves a miss.

## Design (REVISED 2026-09-25 after the before-audit; the first draft is in git history)

The before-audit returned **UNSUPPORTED as written**. The mechanism is sound: the read tap is keyed on
the memory channel (`ex_stage.sv:1273-1281`), so any `get_rev_node` read installs. But four problems
change the design, and all four were re-checked in source:

- **The stall cannot hold the load unit in every state.** `WAIT_GNT` keeps `data_req` up and advances on
  grant without re-sampling `valid_i` (`load_unit.sv:432-446`). The cap check is re-evaluated every
  cycle, and the grant cycle's verdict is the one delivered. So an access that hit at accept and was
  evicted before grant reaches grant as a miss, where no gate can stop it.
- **My loop-safety premise was false.** `misaligned_ex_i` ends in the MMU flop `misaligned_ex_q`
  (`cva6_mmu.sv:500-523`), so today the whole lookup sits behind a register. Gating `ld_valid_i` makes the
  lookup a FIRST-TIME combinational input to `accept_req`, `translation_req` and `data_req`, the last of
  which is in the standing `load_unit.sv:99` cone. No loop was found if the gate uses only the lookup and
  registered state, but the path is new and on the dominant slack term.
- **Anvil lifetime:** a no-response message must be copied into a node register before
  `get_rev_node(*reg)`, because the argument is used after `>>`.
- **ARM 4 as drafted is not separable from ARM 5:** any SPLIT pops the freed index first, so it reissues it.

**The revised design.**

**1. Probe endpoint** (unchanged in intent): `lsu_ep.probe_req(logic[29:0])`, no response, lowest
priority in `IDLE_STAGE`. The handler latches the id into a new register, then calls `get_rev_node(*reg)`.
The generated `try recv` chain splits only the final `else`, so the existing `ep` acks keep their structure.
A tied-off build is the bisect control.

**2. A one-entry ALLOW record, `rvc_ok_q` / `rvc_ok_id_q[29:0]`, that survives eviction.**
- **Set by ANY allow:** a cache hit marked live, or a probe resolution whose tap shows the same
  30-bit id live.
- **Cleared** by either `rvc_inv_now` term matching its 16-bit index (clear wins over set), by flush and
  by reset. It is NOT cleared on accept or consumption: it is a safe one-entry cache, because the
  broadcast is the only event that can make an id dead, and every broadcast clears it.
- **Verdict order in `cap_exception`:** hit-dead, then the `rvc_inv_now` same-cycle terms, then
  **allow** if `(hit && live) || (rvc_ok_q && rvc_ok_id_q == id)`, then miss.
- This closes the `WAIT_GNT` hole. An access allowed at accept stays allowed through grant even if its
  cache entry was evicted, and a revocation in between clears the record through the broadcast.

**3. Stall only where it can hold.** On a miss, suppress cause 25 and gate valid low ONLY while the owning
unit samples `valid_i`: load `state_q` in {IDLE, SEND_TAG}, store in {IDLE, VALID_STORE}. The state comes
from a register. In any other state a miss still denies, which is fail-closed. That residual is now narrow:
it needs a miss at grant for an access that was never allowed.

**4. The gate's inputs:** the lookup (`rvc_lu_hit`, the earlier-check terms) and registered flags only.
**No `revnode_invalidation_*` and no `probe_ack` in the gate.** The audit traced the loop the first would
close: `ld_valid_i` -> `data_req` -> read arbiter -> rev-port grant -> the Anvil write selector -> the
invalidation. `probe_valid` is driven from a flop, and `probe_ack` goes only into flops.

**5. Probe state:** `rvc_pend_q`, `rvc_pend_id_q`. It is set on a stalled miss and resolved by the read
tap whose index matches:
- generation equal and live: set the ALLOW record;
- otherwise: a one-shot DEAD verdict for that id (not sticky, because an unallocated `(0,i)` reads dead
  and later becomes live).

It is cleared on flush and on ANY pop, since an older store's exception can pop a stalled younger head
(`store_unit.sv`, the exception block).

**6. Wedged node:** a bounded wait (a counter), then deny. That fails closed, so a wedged rev-node cannot
turn today's deny into a core hang. The bound is a parameter.

## Traps, pre-registered

1. **Stale generation must deny, not stall.** The tap installs the node's CURRENT generation, so
   resolution compares all 30 bits and denies on a mismatch. This is the security-critical case.
2. **A new combinational path onto the load unit's request.** Synthesis first, compared by loop IDENTITY
   against `6cbdaeeb4` AND by WNS: the lookup was already the dominant slack term.
3. **Rev-node FSM change:** bisect with a tied-off build (it must reproduce `6cbdaeeb4` exactly).
4. **Same-cycle invalidation:** handled in `cap_exception`, never in the gate (see design point 4).
5. **The ALLOW record's clear is load-bearing.** REVOKE_NODE reads node i live (the tap would set the
   record), then writes it dead; only the broadcast clears the record. A mutant without the clear must
   produce an ESCAPE in simulation, which is the positive control that the clear is needed.
6. **Starvation** is bounded: the scoreboard stops issuing once the stalled load blocks commit.
7. **Stores, AMOs:** the same path. LDC/STC: DYN-gated, not in this block. **Scope:** not the CPMP
   (R-44), not the PC tracker.

## Acceptance — written to fail

| test | on `6cbdaeeb4` | required on the fix |
|---|---|---|
| `r43-evict-live` ARM 2 (live, evicted), load | cause 25 | **value = sentinel, no trap** |
| ARM 2s / 2a: the same with a store and an AMO | cause 25 | **no trap; the readback shows the write** |
| ARM 4: DROP the node (dead, generation equal, NOT reissued), evict by LCC reads of pre-minted ids, access | cause 25 (miss) | **cause 25 via the probe** (trace witness: the probe fired) |
| ARM 5: REVOKE, then one SPLIT (pops and reissues the index as g+1), access the OLD capability | cause 25 (miss) | **cause 25, no hang** (trace witness: the reissue happened, the probe fired) |
| ARM 6: flush or interrupt while a probe is pending | — | no hang, correct verdict on re-execution |
| all arms at `S12_MEM_DELAY=12` (opens the WAIT_GNT window) | — | as above |
| mutants: no ALLOW-record clear / no generation compare | — | each must produce an ESCAPE (the positive control) |
| `r35-rotate-stale` | exactly 7 traps | **exactly 7** |
| tied-off build (trap 3's bisect control) | — | identical to `6cbdaeeb4` |
| 92-test neutrality sweep | — | 0 trap-count differences |
| `rtl-lint-gate` | baseline | PASS at baseline |
| synthesis | 1 loop (TIMING-23 `lsu_i/state_q[3]_i_19`), WNS −10.615 | the SAME single loop; LUTLP-1 = 0; WNS reported against −10.615 (noise band several ns) |
| board (after the reflash) | R1 traps 25; live512 traps | R1 completes with its QEMU oracle; live512 returns 17408 |

Positive control for ARMs 4 and 5: on `6cbdaeeb4` they trap for the WRONG reason (a miss), so a fix
that reached "no trap" on them would be an escape. That is why they are pre-registered as 25 on both.

## Alternatives considered and rejected

- **Fail-open on a miss:** reopens R-35 for every evicted stale id.
- **A bigger cache:** lowers the rate, does not remove the false deny, and costs area on a congested
  design.
- **Pinning long-lived ids:** needs a notion of "long-lived" the hardware does not have.
- **A second requester on `ep.query_req`:** already failed silently (above).
