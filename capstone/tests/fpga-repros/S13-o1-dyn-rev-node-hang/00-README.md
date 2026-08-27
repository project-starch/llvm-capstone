# S-13 — at `-O1` the domain HANGS in the DYN/rev-node path, with no exception

**If you arrived here with a capability fault that STICKS at commit — `mcause 25`, `tval = 0`,
aperture 225 reading `0x80` — that is [S-12](../S12-wherecode-notcap-operand-vs-memory/), not this.
This folder is the OTHER failure: no exception at all, and the machine stopped because two syncer
requests are outstanding.** Related: [S-07](../S07-capability-untagged-on-reload/) (untagged
reload), [S-09](../S09-write-buffer-tag-forgery/), [S-10](../S10-write-buffer-forward-residual/).

## The observation

Board, 2026-08-27, resident `caplifive_s07clear_84ed6eafb.bit`, no reflash. Two boots, control
passed in both. Two DISTINCT `-O1` SQLite images running a two-level join both entered (`ENT1`) and
neither returned.

    aperture 225   0xd5   trace_buf_empty + dyn_wait_store_syncer + dyn_wait_rev_res
                          + stall_issue + mem_wait_flag
    aperture 224   0x7f   ex_commit.valid = 0, load_syncer_req_set = 1,
                          store_syncer_req_set = 1, all three ready bits = 1

Both packings verified in source (`cva6.sv:1177-1186` and `:1189-1199`), MSB→LSB, not taken from a
label string.

## Why this is not S-12 — four independent lines

| | S-12 | S-13 |
|---|---|---|
| `ex_commit.valid` (224 bit 7) | **1** — an exception is stuck at commit | **0** — no exception |
| aperture 225 | `0x80` — **nothing** is waiting | `0xd5` — **three** wait conditions |
| `mcause` | 25, a real capability fault, `tval = 0` | none; the latch holds stale kernel traffic |
| fault site | present in the image | **verified absent** from every draw artifact |

Measured in the SAME boot series, same bitstream, same query, same compiler — an `-O0` arm gave the
S-12 column and the `-O1` arms gave the S-13 column, so optimisation level is the only variable.

**The one-sentence statement: `-O1` converts a stuck capability fault into a non-exception hang.**
That is consistent with the S-12 fault site being gone — `pWInfo` stays in callee-saved `s2`, zero
`ldc`+`cincoffsetimm 0xb0` pairs, verified in each artifact — and something previously masked or
previously survivable now being what stops the machine.

## What is already excluded

**Not an instruction-stream density or coupling effect.** Two independent measures point the wrong
way for that explanation:

    capability memory ops     -O0  ldc 54,809  stc 21,915      -O1  ldc 31,700  stc 16,036
    stc -> next ldc distance  -O0  median 2, 59.9% within 2    -O1  median 5, 30.2% within 2
                              (normalised: 4.43% of the stream vs 2.30%)

So `-O1` does **fewer** DYN-path operations, **more** widely spaced, and is still the arm that
hangs there.

**Not a stale store-syncer flag.** `store_syncer_req_set = 1` on 224, so the store side agrees with
`dyn_wait_store_syncer`; the bit is not lying on these wedges.

**Not a transitive nesting path in the source.** The three store-syncer wait regions in
`capstone_dyn_unit.anvil` (set at 14/435/451, cleared at 19/438/455) contain no rev-node reference,
directly or through the only functions they call (`create_cnull`, `create_result_pack`,
`create_capability` — 0 rev-node references each, against `get_node_query_validity` showing 2 as a
positive control in the same check).

**Not two independent FSMs.** In the generated `capstone_dyn_unit.anvil.sv` both flags are driven
from the SAME `always_ff` (opening at 7572; `waiting_for_rev_res` at 7578 and 7702-7750,
`waiting_for_store_syncer` at 7579 and 7759-7777; next `always_ff` not until 9421).

## The open question

Three statements are each supported and do not fit together:

1. the board says both wait bits are asserted with a store genuinely outstanding;
2. the `.anvil` source says no sequence sets both;
3. the generated hardware says they live in one process.

So the question is not *can* both be asserted — measurement says yes — but **what SEQUENCE gets
there**: an interleaving, a compiler-generated state the `.anvil` does not read like, or entry into
the rev-node wait from a state whose store-syncer `recv` never completed and so never cleared.

Next step is reading the generated FSM around `7572-7790` against the `.anvil`. Hours, not minutes,
and it belongs to the RTL lane.

## Caveats, so nobody over-reads this

- **`ex_commit.valid` is a LIVE bit**, sampled when the switches are read. It says no exception is
  valid *at that moment*, not that none ever occurred. It carries weight here only because the
  `-O0` arm reads 1 on the same aperture, same driver, same read point.
- **N = 2.** Two distinct `-O1` images, both wedged. Against the `-O0` base rate of 54% that is
  p = 0.29 by chance, so this says nothing about a RATE — only that `-O1` is not immune and that
  when it fails, it fails differently.
- **The commit-pc aperture is stale by construction at a wedge** and must not be used to localise
  this. Both boots reported an identical `commit pc = 0x82c1c3fc` from two images holding different
  instructions at that address. Read 225/224 for *which class*, never for *where*.

## The two wait flags are never observed APART — 60 boots, no intermediate state

**The structural question is settled, and it went the other way from the first reading.** The RTL
lane's initial argument — both flags are driven from the SAME `always_ff`, therefore not two
independent FSMs — is **retracted by its own author** after reading the generated code.
`capstone_dyn_unit.anvil.sv`'s `_thread_1_st_transition` is not a linear state machine but a flat
series of INDEPENDENT event guards over two SEPARATE registers, driven by different events
(`EVENTS1[26,28,…]` for the store flag, `EVENTS1[20,23,…]` for the rev flag), with no mutual
exclusion. **A state with both flags set is directly representable and needs no nesting path.**

So `0xd5` is not anomalous and never needed explaining away. The question became: *which clearing
event failed to fire?*

**That produced a falsifiable prediction, and it fails.** If the store path is the stuck one while
a rev query proceeds and clears independently, a wedge with **store SET and rev CLEAR** — `0xc5`
shaped — should occur. Mined from every wedge on record, scoped per boot:

    0x00    3 boots   void / dead reads
    0x80   52 boots   trace_buf_empty ONLY -- nothing waiting          (the S-12 class)
    0xd5    8 boots   trace_buf_empty + store_syncer + rev_res + stall_issue + mem_wait
    0xc5    0 boots   <-- PREDICTED, NEVER SEEN

P(0 store-only in 8 both-set boots | equally reachable) = 3.9e-3. On 8 samples that weakens the
account rather than killing it.

**The sharper observation is the absence of ANY intermediate state.** Across 60 boots the flags are
always both-clear or both-set, never one without the other — even though the hardware provides no
interlock forcing that. Independence predicts separable states; we see none.

**CAVEAT, and it is load-bearing:** these are all WEDGE-TIME samples. The flags may take
intermediate values in normal operation and we only look when the machine has stopped. So this is
a statement about the WEDGED state, not about the flags in general.

**Where that leaves the account:** "two memory clients, no interlock, one stuck" predicted a
separable state that 8 samples do not show. An account in which both waits are entered as part of
ONE sequence, or in which a single stuck resource blocks both, fits better. Note this is closer to
the *retracted* structural instinct than to the correction — the FSM finding still stands (no
interlock EXISTS), the data merely says the machine does not appear to use that freedom.

**GENERAL HAZARD WORTH CARRYING BEYOND S-13:** the `.anvil` reads as a sequential program
(`set := 1 >> send >> recv >> set := 0`) and invites being reasoned about as one; the generated
hardware is a concurrent event machine over independent registers. Any conclusion in this
investigation that rests on an `.anvil` sequence CONSTRAINING ordering — the store-buffer FIFO
argument that excluded the S-12 granule row among them — may want re-examining on those grounds.
