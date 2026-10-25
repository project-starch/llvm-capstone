# S-23 containment: a flushed lifetime op cannot orphan its walk, because the issue stage serialises it

*RTL lane, 2026-10-19 evening. Continues `19-10-2026_21-00-00_s23-cpmp-rearm-rtl-half.md` (the data face) and the
three directed arms that preceded it. Lane branch `virtual-profile-2-c` of `capstone-ariane`; the arms and the
sim-only displays are committed at `af2d6bad7` and `ae40eddd1`, and the semantics owner's retraction of the
speculative route is on dev at `c84903d44c1f`.*

## The finding

**Every operation that can WRITE a revocation-node record is refused issue until no older instruction is still in
flight.** `core/issue_read_operands.sv:1557-1565`, inside `` `ifdef CAPSTONE_ITFC_EN ``:

    if (issue_instr_i[i].op inside {DROP, REVOKE, MREV, SPLIT, DELIN, CDERIVE, CREVOKE, CSNODE, CSRETIRE} &&
        |fwd_i.still_issued) begin
      issue_ack[i] = 1'b0;
    end

with `still_issued[i] = mem_q[i].issued & ~mem_q[i].cancelled` (`core/scoreboard.sv:129`, exported `:349`) and the
macro defined for the active target at `core/include/capstone_cv64a6_imafdc_sv39_config_pkg.sv:17`. Issue is in
order and single-port here, so the bit set means an older instruction has not committed.

Every source of `flush_ex` is a commit-stage or debug event (`core/controller.sv:95-245`); a branch mispredict
raises only `flush_unissued_instr_o` and `flush_if_o` (`:111-117`), never `flush_ex_o`; and the EX stage's
`flush_i` is `flush_ctrl_ex` (`core/cva6.sv:1776`). A guarded op in flight is therefore the oldest instruction in
the machine, nothing younger can commit ahead of it, and no non-commit source reaches EX. Its request's owed
window lies entirely inside that period, because the rev-node unit answers every walk at its **exit**
(`core/anvil_build/capstone_rev_node.anvil:48-62` for the revoke/kill/retire walk, `:117`/`:121` for deepen) and
every handler writes before it answers.

**So the "orphaned walk" route to S-23 is closed by construction for application CDERIVE and CREVOKE**, not merely
unreached by the arms that looked for it. Both guard-list commits (`b330433fe`, CDERIVE/CREVOKE, 2026-10-13, and
`41fa120bf`, CSNODE/CSRETIRE, 2026-10-17) are ancestors of `bf3db3175`, the RTL the vp2 bitstream was built from,
so this reaches the field artifacts rather than only the current tip.

### Four preconditions, each load-bearing

Not caveats — remove any one and the argument fails:

- `SuperscalarEn: bit'(0)` (`config_pkg:89`), so `NrIssuePorts` is 1. `still_issued` is **registered**, so with two
  ports an older instruction on port 0 and a lifetime op on port 1 would both be acked in the cycle before the
  older one appears in the vector. **Setting `SuperscalarEn` would defeat this guard silently.**
- `SpeculativeSb` is dead (`build_config_pkg.sv:51`, gated on `SuperscalarEn`), so `cancelled` is never written and
  `still_issued == issued`.
- `NrCommitPorts: unsigned'(1)` (`config_pkg:90`), so there is no second commit port to take an exception.
- `DcacheFlushOnFence` is 0 (`config_pkg:52`), so `fence_active_q` is never set and a fence's flush is a
  one-cycle pulse rather than a held level.

### What remains unguarded, and it cannot broadcast

One writer of node records exists — the unit itself (`core/ex_stage.sv:1607` is the only driver of
`mem_rev_wr_req_valid`) — and all 24 of its write sites sit in handlers for the nine guarded ops. The **domain
switcher sends no rev-node request at all**: its proc has four endpoints and none is that channel
(`capstone_dom_switcher.anvil:602-607`). The only unguarded channel is `query_req`, which is read-only
(`capstone_rev_node.anvil:340-349`, whose own comment names its clients: LCC, SPLIT, REVOKE, TIGHTEN, CALL,
RETURN, LDC, STC), and the R-43 probe, also read-only. **No unguarded operation can write a node, so none can
broadcast an invalidation.** An owed query at a flush does happen — R-27 measured that shape — and is drained
(`ex_stage.sv:1088`), with no walk behind it.

### The invalidation tap, and the one place allocation does broadcast

`ex_stage.sv:1689` — `revnode_invalidation_valid_o = mem_rev_wr_req_valid && !node_wr_req[31] && !vp_hdr_wr`, so
the broadcast fires **only on a write that clears VALID** (bit 31), with header writes excluded under the table
flag (`:1469`). But the **free-list pop** writes the popped record while valid is still 0:
`capstone_rev_node.anvil:288` enters only `if (fn.valid == 0 && fn.free == 1)` and `:303` writes
`{fn with generation = g+1; free = 0}`, sent at `:307`. `alloc_slot` is reached from `init_req` (SPLIT),
`derive_req` (CDERIVE) and `create_req` (CSNODE), so **a CDERIVE that allocates off the free list broadcasts an
invalidation for the index it allocates.** An allocation from the bump pointer writes nothing (`:315`), and under a
virtual context the pop writes only the header, which the tap excludes.

### The re-arming namespace flush is permissive, so it cannot be the deny

`core/pmp/src/pmp_data_if.sv:111-114` is the **last** clause of the comb block and overrides both the adopt and the
invalidate in the same cycle: the tracked id is zeroed and `cpmp_revnode_valid_d[i]` is set to 1. Every entry
leaves the pulse live, and a deny needs `cpmp_revnode_valid_q[i] == 0` (`:149`). Further, a broadcast-cleared entry
**stays** denying: the invalidate leaves the tracked id alone and the adopt fires only on an id *change*, so the
entry refuses until something rewrites it with a different id or another namespace flush arrives. A transient deny
that clears within one trap round trip does not fit that unless the monitor rewrites CPMP on the way back.

A candidate that does fit the transience, from the same pulse: the namespace flush empties the LSU revocation cache
un-permissively (`core/load_store_unit.sv:1221-1224`, "Empty, never permissive"), the first access afterwards
misses, and a probe that times out denies with cause 25 — a one-shot a retry clears. **The discriminator is already
in silicon:** `rvc_den_arm_o` is one-hot, 0 hit-dead, 1 same-cycle invalidation, 2 probe DEAD, 3 timeout
(`load_store_unit.sv:221`), read by the board driver from aperture 204
(`capstone/tests/rtl-smoke/fpga_driver/refusal_record.py:8-9`). Arm 0 or 1 means a broadcast cleared something;
arm 2 or 3 means the RVC race; a latched deny with **no** arm set is not in that record at all and points at the
CPMP tracker.

## What re-arms a denied tracker: the trap itself, because `vctx_table_o` is privilege-dependent

Read on tree A (`8522d72fe`), after the semantics owner excluded the RVC story by the cause. **Exactly two
conditions write `cpmp_revnode_valid_d[i] = 1`** (`core/pmp/src/pmp_data_if.sv:83-116`, one comb block, three
clauses, the flush last so it overrides the other two in the same cycle):

- **the ADOPT** (`:88-94`): `cpmp_tag_i[i]` **and** `cap_type != NOT_CAP` **and** a 30-bit `revnode_id` **change**
  against `cpmp_tracked_revnode_id_q[i]`;
- **the NAMESPACE FLUSH** (`:109-114`): unconditional, all sixteen entries, tracked id zeroed and valid set.

There is no third, and `cpmp_tracked_revnode_id_d[i]` is written only in those same two branches. That settles two
paths the monitor's `swap_cpmp` takes: a **CCSRRW swap of an untagged value and back cannot re-arm** (on the
swap-in `cpmp_tag_i` is 0, so no adopt *and* the tracked id is not forgotten; on the write-back the id is
unchanged, so the change term fails), and an **LCC zimm 0 query's response never reaches the tracker** (its whole
input set is `cpmp_tag_i`, `cpmp_i[*].comp_metadata`, `revnode_invalidation_*` and `revnode_ns_flush_i`).

**The re-arm comes from the privilege term.** `core/csr_regfile.sv:2876-2878`:

    vctx_table_o = CAPSTONE_EXT && ( (sup_active_q && (csupctl_virt_q || sup_runv_q))
                                  || (!sup_active_q && priv_lvl_q == PRIV_LVL_S && srevroot_q != 0)
                                  || (!sup_active_q && priv_lvl_q == PRIV_LVL_M && csupctl_table_q) )

Term 2 is gated on S and term 3 on M, so with `sup_active = 0`, **srevroot nonzero at S and csupctl.table clear at
M, `vctx_table` is 1 at S and 0 at M: the trap flips it 1→0 and the mret flips it 0→1, two namespace flushes per
round trip**, each marking all sixteen entries live. Each flip gives exactly one pulse
(`core/ex_stage.sv:1399-1416`: `vctx_rev_q <= vctx_table_i` on the first `!rev_busy` cycle, `vctx_rev_qq` one cycle
behind, the flush on their difference). With srevroot 0 at S and csupctl.table clear — which is how this lane's
M-mode arms run — there is **no edge and the deny persists**, which is exactly what the Q4 arm read.

So a tracker deny's life is "from the broadcast to the first trap after it", and the trap it causes is its own
cure. Three consequences:

- **No read from M can ever observe the denying state**: entering M has already re-armed every entry before the
  first instruction of the handler. Any instrument for this face must be in RTL or at S.
- **The refutation is one `csrr` each in a path the monitor already prints from**: read `srevroot` (0x5C1) and
  `csupctl.table` on a refusing boot. srevroot 0 there, or csupctl.table set at M, excludes this mechanism.
- The deny's cause is confirmed for the field signature: a CPMP data deny raises the fault in the CPMP block
  itself (`pmp_data_if.sv:302-312`), `ST_ACCESS_FAULT` for a store and `LD_ACCESS_FAULT` (5) for a load, with
  `tval = cpmp_paddr_x`, the physical address.

### And the precondition holds, because nothing ever restores `srevroot`

The clock above needs srevroot nonzero at S when the caller runs. Traced in
`core/anvil_build/capstone_dom_switcher.anvil` on tree A, there are exactly **three** id-67 sites:

- **entry** (`runv_load_advance:170-173`): the live srevroot is loaded from the **frame's second word**
  (`dom_base + 8`) — the context's table;
- **resume** (`runv_resume_step:316`): restored from the hidden word at `dom_base + 936`;
- **escape** (`runv_event_step:481`): **saved** to that hidden word.

**The caller's srevroot is in neither walk.** The SAVE/RESTORE walk around the exchange covers registers 3..66 —
`walk_advance` calls `walk_end()` at `cur_idx == 7'd66` on every branch (`:527-541`). So id 67 is never saved at
entry and never restored at escape or return, and the only other writer of `srevroot_d` in the design is the CSR
instruction itself (`csr_regfile.sv:2014`). There is no caller-srevroot register: the hardware saves the caller's
privilege (`runv_caller_priv_q`) and nothing of this kind.

**So after any escape or return the caller runs with the context's table base in srevroot**, until its own next
write of 0x5C1. That satisfies the precondition without the module doing anything, and it lands where the field
refusal does: `_raw_spin_lock_irqsave` is how the module *enters* an irq-off window, so it executes before the
`select_root`/`restore_root` write inside that window.

**This is a defect in its own right, independent of S-23**: the caller's lifetime namespace after an escape is the
context's, so any S-mode lifetime op the caller performs between the escape and its next `restore_root` resolves
against the context's table — and by `vctx_table_o`'s S term that is a namespace change, not merely a stale base.
The fix would be for the caller's snapshot to carry id 67 as it carries 66; that is a switcher revision and a
semantics call, not this lane's to make.

**The gap this left, and how the semantics owner closed it.** This lane looked for a **broadcast falling after
the last namespace flush** (the mret's 0→1 edge) and could not source one: after that edge the caller issues no
lifetime op and the switcher issues no rev-node request. Two candidates are excluded from the RTL side — an
unadopted entry carrying node 0 (its `__capfield(cap, 0)` would read 0, node 0 being the invalid sentinel, while
the field read 1 every time) and the switcher.

The error was in the requirement. **A deny does not need a broadcast after the last flush; it needs a broadcast
that no flush FOLLOWS.** The semantics owner's sequence, from the module at `8807df747411` (their citations, read
in their lane, not re-read here): with srevroot already stuck at the previous context's table, `select_root`
(`:683-689`, `csrr old, 0x5c1` then `csrw 0x5c1, virt_to_phys(c->table)`) writes the **same** value, so there is
no edge and no flush; the CSRETIRE (`:716`, `:947`) or CSMINT (`:858`) inside that window walks in the context's
namespace and broadcasts its indices, clearing any tracker whose index collides; `restore_root(old)` writes the
same value again, so again no flush; and the window ends with the tracker still clear. The kernel then reaches
`unpin_user_page`'s `mod_node_page_state` on CPU 0's per-CPU node stats, or the per-CPU `__mmiowb_state` of the
next `spin_lock_irqsave` — both in slot 7, both the field's refusal sites. The refusal traps, and the trap and
the mret are the two flushes that re-arm all sixteen trackers, so the retry passes.

**The one unread fact is the collision itself**: whether one of the context's virtual indices equals slot 7's
tracked physical index. Slot 7's revnode id has never been printed.

## Measurements

`s23-kill-window` on `af2d6bad7`, two memory delays, both SUCCESS, all 21 readings exactly as pre-registered
including the chain ruler reading `0 0 0` for the 8-chain's head, middle and tail — no truncation anywhere.

| kill arm | nodes walked | window @ `S12_MEM_DELAY=12` | @ 40 |
|---|---|---|---|
| leaf | 1 | 16 | 16 |
| chain of 4 | 4 | 34 | 34 |
| chain of 8 | 8 | 105 | 77 |
| chain of 8 behind the divides | 8 | 70 | 57 |

**The window scales with the junior run, and the per-node cost is ~9-13 cycles and does NOT rise with main-memory
latency** — the slope is 12.7 cycles/node at delay 12 and 8.7 at delay 40 — so the walk's node reads are served
from the cache, not from memory. That slope is the transferable figure; the absolute windows are testbench
quantities. Four points only, so no extrapolation past N≈8. A chain is the only shape that lengthens a walk:
siblings sit at equal depth and the walk stops at the first one (`capstone_rev_node.anvil:508-566`, `depth_bound`),
which a chain is buildable against because CDERIVE gives the child `(parent perm & 7) | 8`, i.e. MANAGE is
inherited (`capstone_dyn_unit.anvil:291`).

Two pre-registered predictions in that table failed, both recorded rather than reinterpreted: the two 8-node
windows should have agreed and differ by 35% (the quiet-pipeline reading is a hypothesis), and the total run was
**shorter** at delay 40 than at 12 (6,516 against 8,548 cycles), which is unexplained and means `S12_MEM_DELAY` is
not a simple latency scale in this harness.

### The guard pair fired and under-determined

The arm's second half measured the gap from a marker query to the op's own owed window: **216 cycles for a CREVOKE
behind three dependent divides, 144 for a node query behind the same divides**, identical at both delays. 144 is
exactly the divides' latency, so the **unguarded** op waited for them too — the serial divider is not pipelined and
stalls in-order issue of whatever follows. The residual 72 cycles is consistent with the guard and not attributable
to it. **This is a check that fires correctly and still cannot separate the hypotheses on the table**, which is why
the guard is recorded above on the source and not on this measurement. `s23-issue-guard` (`ae40eddd1`,
pre-registered, not yet run) repeats the pair behind a cache-missing load, which does not block younger issue, and
keeps the divides pair as the negative control for the method.

## The instrument, and what its silence is now worth

The must-fire (`ex_stage.sv:1097`, a flush while a rev-node request is owed) has now been **silent in six arms** —
`s23-wrongpath-revoke`, `s23-phantom-kill`, `sup-escape`, `sup-mtip`, `sup-fullswitch` and `s23-kill-window` —
across 248 owed-vector events. That silence was worth nothing until both of its conjuncts were shown to be live in
the same scope. The owed-side control came first (42 events in one arm); the **flush-side control** was added here
(`af2d6bad7`) and printed 10 lines per run. Only with both does the zero read as a result.

**The phantom arm's generator is defeated by the very guard it was built to test**, which is the cleanest
explanation of its silence: the CSR instruction's scoreboard entry still reads `issued = 1` in the cycle it
commits, so the CREVOKE cannot be acked in that cycle, and by the next the flush has fallen. No generator the guard
defeats can ever provide the conjunction's positive control. The only remaining generator is a **level**-held flush
rising while the unit is busy, i.e. `dom_switch_busy_i`, which `controller.sv:215-225` holds high for the whole
domain switch — and that would also settle the contradiction at `ex_stage.sv:1434-1435`, a comment asserting that
"lifetime instructions issue speculatively, so a handler can straddle an escape", added *after* the guard list
gained CDERIVE/CREVOKE. Either that sentence is wrong and the `rev_busy` hold is merely conservative, or the
argument above is.

## Retractions and corrections from this session

- **"The owed windows are 3 to 11 cycles"** — withdrawn. That was the phantom arm's log read two events deep and
  reported as a range; pairing all 42 events by channel gives query 3 (×14), derive 11/165/197/197, kill 16/100/16.
  The negative it supported is *stronger* for it: P's own kill window is the 100-cycle one, so the flush missing it
  says the generator is not timed to the request. The counts reconcile exactly — 4 derives, 3 kills, 14 queries =
  4 + 6 + 4 — which is the check that found the error.
- **"Measured on bf3db3175"**, for the phantom arm — wrong as an artifact identification. That commit does not
  contain the displays, which arrived with `e654ec103` and `2d311fabf`; the run was that tree plus working-tree
  patches. The six load-bearing files are byte-identical between the two, so the measurement stands, but the hash
  does not name what ran. Cite the patched file's hash.
- The guard is at `issue_read_operands.sv:1557-1565`; an earlier message and commit message said `:1558-1566`.
- `capstone_rev_node.anvil:4` says the write is tapped at `ex_stage.sv:1207`; it is tapped at `:1689` and `:1207`
  is now the R-27 ready gate. The namespace-hold block is `ex_stage.sv:1441-1462`.
- `s23-kill-window` produced 48 query spans against 46 pre-registered. Derives (22) and kills (4) are exact; the
  two extra spans sit immediately before the two marker queries, each after a fence, and the candidates are a
  mis-mapping of a positional count or a query flushed and replayed by the fence. **Not explained.** The follow-up
  arm anchors every measured op with a CDERIVE so positions are marked rather than counted, and has no fences.
