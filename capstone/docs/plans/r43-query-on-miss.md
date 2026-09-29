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

## Design — SECOND VERSION (2026-09-29): REPLAY a missed access, never hold the request

The first version (`0f5185a6d`) held a missed load/store by gating its request with a combinational term
built from the revocation lookup. It was correct in simulation and **refuted by synthesis**: WNS −24.495,
with all of the worst 500 paths through that gate (`tests/fpga-repros/R43-revocation-cache-false-deny/
results/synth-0f5185a6d.result-lines.txt`). Everything below the gate is kept; the gate is replaced.

```
   M-mode load/store, revocation-cache MISS (no hit, no ALLOW/DEAD record)
          |
          v
   LSU: mark the access with an INTERNAL replay cause on the EXISTING registered exception path
        (cap_exception -> MMU flop -> the load/store unit finishes WITHOUT a memory operation)
        and start the probe (probe_ep -> rev-node reads the node -> the read tap installs it)
          |
          v
   commit: the marked head is NOT an exception (stripped from exception_o, commit_ack stays 0);
           wait while the LSU's REGISTERED probe-pending flag is set;
           then flush and re-fetch THE SAME pc (frontend: pc_commit + 0, and restore the head's own
           PC-capability metadata, so R-46 cannot fire on the replay)
          |
          v
   the re-executed access: hits, or matches the ALLOW record (set by a live probe) -> allowed;
                            matches the DEAD record (dead / stale / timeout)         -> cause 25
```

Rules that make it hold (from the before-audit):
- **Nothing new enters the load unit's request path.** The only new wire from the LSU is the probe-pending
  flag, a flop, to commit. Commit's replay term reads the head and that flop, never `commit_ack` or
  `pc_cap_ex_valid`.
- **The probe outlives the access's pop.** It clears only on resolve, timeout or a flush.
- **ALLOW** is set only by a live probe resolution, keyed by the exact 30-bit id, cleared by any
  invalidation of its index (broadcast or a dead write), never by a flush.
- **DEAD** is set by a dead/stale resolution or a timeout, cleared only by a write to its index, never by
  flush or pop; a dead 30-bit id never becomes live again, so a stale DEAD cannot falsely deny.
- **Younger rev-node operations cannot run twice:** DROP/REVOKE/MREV/SPLIT/DELIN issue only when every
  older instruction has committed (`issue_read_operands.sv`), and a marked head is uncommitted.
- **Bound:** at most 3 replays per dynamic access; the trace counts them.
- **Timeouts** (fail closed, unchanged): ~1M cycles before the rev-node accepts the probe, 65,535 after.
- **R-45** (the REVOKE/DROP commit flush) is unchanged; **R-46** stays accepted for the ordinary refetch and
  is closed on the replay refetch.

**The refusal record** (observation only, batched in): the FIRST cause-25 verdict since reset, `{v,~v}`
+ one-hot arm (hit-dead / same-cycle invalidation / probe DEAD / timeout) + 30-bit id with two parity bits,
at switch values 204..208 (bank 110, regs 01100..10000).

## Design — FIRST VERSION (REVISED 2026-09-25 after the before-audit) — REFUTED BY SYNTHESIS, kept as history



*Correction: an earlier version of this heading, and the commit that added it, said the first draft is in
git history. It is not: the draft was revised before it was ever committed, and only the revised design
was committed.*

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

**1. Probe endpoint** (unchanged in intent): `probe_ep.probe_req(logic[16])`, a bare INDEX, since the node
read addresses by index and the read tap supplies the generation. No response, lowest priority in
`IDLE_STAGE`. The handler latches the index into a 16-bit register, then calls `get_rev_node(*reg)`.
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

**6. Wedged node, in two bounds (revised after the second after-audit):** before the rev-node accepts the
probe it may be busy for a long time, because the probe is served only in IDLE_STAGE and never during a
walk; silicon walks cost up to 11,811 cycles for 256 nodes. So the pre-acceptance bound is about 1M
cycles, and the post-acceptance bound is 1,023 cycles. Either one expiring denies (fail closed), so a
wedged rev-node cannot turn a deny into a core hang. The first single 1,023-cycle bound would have
falsely denied live accesses that missed during a walk of more than ~40 nodes.

**7. R-45, the revocation ORDERING window, closed in the same bitstream (the lead's decision).**
`commit_stage.sv` raises `flush_commit` when a REVOKE or DROP commits (after its walk), so younger
instructions re-execute against the revoked state. See ISSUES.md R-45.

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

*Arm numbers are those of `verif/tests/custom/capstone/r43-evict-live.S`. Every arm counts only if the
`R43_TRACE` lines show its triggering condition was actually created.*

| arm / build | condition | on `6cbdaeeb4` | required on the fix |
|---|---|---|---|
| ARM 2 | live alias, evicted, load | cause 25 | sentinel, no trap (the ALLOW record) |
| ARM 2s | live alias, evicted, store + readback | cause 25 | no trap; readback shows the write (probe -> ALLOW) |
| ARM 5 | revoked, then its index reissued at g+1; old capability accessed | cause 25 | cause 25 via the probe (generation mismatch -> DEAD) |
| ARM 6 | allowed, revoked, churned (dead entry evicted, index reissued) | cause 25 | cause 25 via the probe |
| ARM 7 | revoke with NO barrier, then read an EVICTED alias at once (R-45) | cause 25 (by miss) | cause 25 (the REVOKE-commit flush re-executes the read after the walk) |
| ARM 7r | the same with the alias RESIDENT (the case already open on `6cbdaeeb4`) | cause 0 (allowed) | cause 25 |
| ARM 8 | a live alias at generation >= 1, evicted | cause 25 | sentinel via probe -> ALLOW with a nonzero generation |
| ARM 9a | AMO through an evicted live alias | cause 25 | old value returned, memory updated |
| ARM 9b | DROP, evict, access | cause 25 | cause 25 via the probe |
| build **tieoff** | `R43_PROBE_TIEOFF` | — | every miss denies; no probe (it still carries R-45's flush, so arm 7r denies) |
| build **noclear** | ALLOW record ignores invalidations | — | superseded: R-45's flush ALSO clears the record, so this single removal is masked (arm 6 denied) |
| build **noclearflush** | both clears removed | — | ARM 6 ALLOWED: at least one of the two is required |
| build **mgen** | resolution drops the generation compare | — | ARMs 5/6 ALLOWED (positive control: the compare is load-bearing) |
| build **mto** | the rev-node never receives the probe | — | every stalled miss denies after the timeout; no hang |
| build **noflush** | R-45's REVOKE/DROP commit flush removed | — | ARM 7r ALLOWED (positive control: the flush is what closes the window) |
| ~~build tiny~~ | a 2 x 2 cache | — | dropped: it did not create a `samp=0` verdict in its run, so it tested nothing (see below) |
| build **flushproxy** | the probe state cleared once just after a probe was sent | — | a second probe for the same id, then the correct verdict; no hang |
| `r35-rotate-stale` | R-35's acceptance fixture | exactly 7 traps | exactly 7 |
| 92-test sweep | neutrality | — | 0 trap-count differences |
| `rtl-lint-gate` | hazards | baseline | every hazard counter at baseline; UNUSEDSIGNAL +1 (the probe read's unused return value, by design), re-baselined with that named |
| synthesis | loop / timing | 1 loop (TIMING-23 `lsu_i/state_q[3]_i_19`), WNS −10.615 | the SAME single loop; LUTLP-1 = 0; WNS reported against −10.615 (noise band several ns) |
| board | after the reflash | R1 traps 25; live512 traps | R1 completes with its QEMU oracle; live512 returns 17408 |

**Not constructible deterministically, and said so:** a miss in the load unit's WAIT_GNT window on the
shipping geometry (it needs an eviction inside a grant hole of a few cycles). The **tiny** build forces the
window; the shipping design fails closed there, as a false deny, and that residual is documented rather
than claimed away.

## Alternatives considered and rejected

- **Fail-open on a miss:** reopens R-35 for every evicted stale id.
- **A bigger cache:** lowers the rate, does not remove the false deny, and costs area on a congested
  design.
- **Pinning long-lived ids:** needs a notion of "long-lived" the hardware does not have.
- **A second requester on `ep.query_req`:** already failed silently (above).
