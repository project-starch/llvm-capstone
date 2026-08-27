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
