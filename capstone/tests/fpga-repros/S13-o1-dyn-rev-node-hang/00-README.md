# S-13 — at `-O1` the domain HANGS in the DYN/rev-node path, with no exception

> ## THIS MEASUREMENT CANNOT BE REPRODUCED FROM THIS FOLDER (added 2026-09-04)
>
> **The two `-O1` images behind all 8 boots are not here and are not reconstructable.** This folder
> contains one file: this README. No `.dom`, no `SHA256SUMS`, no build command, no image identity,
> no compiler revision. Everything below is a faithful record of what was *observed*; none of it can
> be re-run against the artifacts that produced it.
>
> That matters more than it looks. The project's convention is that a repro folder **is** the
> report, and the evidence here is unusually strong — 63 boots, two apertures agreeing perfectly —
> which makes it exactly the kind of result someone will want to re-test after an RTL or compiler
> change. They cannot. A fresh `-O1` two-level-join build is a **different image**: it can show
> whether a given defect appears in that *class* of build, and it can say nothing about the two
> images that actually wedged.
>
> **So: quote the observation, do not claim a re-run.** If you rebuild, say in the record that the
> image is new.
>
> ### A related gap in the reasoning below
>
> **This report never considers an unhandled fault.** There is no mention of a trap vector, a fault
> storm, or address 0 — although `../RTL-domain-trap-vector-unset/` records, root-caused and
> confirmed on silicon, that a domain enters with **no trap vector**, so a fault inside one storms
> at address 0 and does not present as an exception. That possibility should have been considered
> and excluded explicitly rather than left unmentioned.
>
> **It was raised on 2026-09-04 as a specific hypothesis — that S-13 is C-40 (an `-O1` LSR-generated
> `cincoffset` off a null base, cause 24) made invisible by the missing vector — and that mechanism
> is REFUTED by this document's own data.** A fetch storm gives no account of
> `dyn_wait_store_syncer` and `store_syncer_req_set` being asserted, and those two signals agree
> across all 63 boots. The core is WAITING, not faulting. Recorded here so the same hypothesis is
> not re-derived: the general question (was an unhandled fault excluded?) is open; this particular
> answer to it is closed.


## THE STORE SYNCER IS CLOSED — three mechanisms, all on structure rather than absent counters

The single-entry `capstone_store_syncer` holds ONE pending trans id and sets it on a new `init`
with **no guard** on `req_set`. That missing guard is visible in the source and anyone reading it
later will reach for the same hypotheses; this section is why they should not.

**1. Overlapping inits — unreachable.** `func STC` sends `cap_store_ri.init` at `:391`, then at
`:436`/`:452` sends the req and does `let _ = recv cap_store_ri.res >>`. The `recv` BLOCKS the
issuing thread, so the round trip completes before another init can be reached. Measured on top of
that reading: `s13-stc-pressure.S`, eight independent capability stores one per cache line after an
eviction sweep, repeated — **192 inits, `init-while-pending` = 0**, against 7-8 inits in ordinary
tests, so the positive control climbed 24-fold. The identical experiment on the load side
(`s12-ldc-pressure.S`) gives the same 192 / 0.

**2. Trans-id aliasing — precondition unreachable, and reported as NO VERDICT.** `cap_trans_id` is
3 bits against an 8-entry scoreboard, so ids wrap, and a stale response matching a new
`cap_trans_id` would clear `req_set` for a store that never completed its check — a MISSED CHECK
rather than a hang, invisible to counters that fire on init. A checker on the response match reports
`wraps=0 ambiguous-matches=0`. **The rule was "refuse the zero unless wraps are non-zero", so this
did not test aliasing** — it showed the precondition is unreachable, for the same reason as (1): no
trans id can have two outstanding uses, so no wrap is constructible.

**3. Flush desync — impossible.** If a flush cleared `req_set` but left `lsu_msg_set`/`lsu_reg` set,
the next store's check would run against a PRE-FLUSH LSU result. The flush block is byte-for-byte
symmetric with reset — `_init_0`, `cap_trans_id_q`, `lsu_msg_set_q`, `lsu_reg_q`, `req_set_q` and
the event counter all cleared in both. It is also **last-assignment-wins**: the flush `if` sits
after the `EVENTS0[4]` assignment in the same `always_ff`, so a flush beats an event setting
`lsu_msg_set_q` in the same cycle.

**So the single-entry design is SAFE, and that is a positive result rather than a null one.** It is
not a latent defect awaiting the right pressure: the protocol guarantees at most one outstanding
transaction.

> **THE CONSTRAINT THIS LEAVES IS TIGHTER, NOT LOOSER.** Thread 1 owns BOTH wait flags, blocks on
> its own recvs, and should not be able to reach a rev-set event while holding the store flag — yet
> 8 wedge boots show exactly both-set. Something in that chain is wrong and it is **no longer the
> syncer**.
>
> **A METHOD NOTE THAT ALMOST COST THIS RESULT.** The flush asymmetry was first "found" via a
> filtered view — a grep matching only `flush`, `req_set_q` and `_init_0` — which showed the flush
> block containing just two clears and hid the other four. That is precisely the hypothesised
> asymmetry, produced by a filter narrowed by the query rather than by the data. It was caught by
> printing the block unfiltered before writing it up. Paired with a bit-slice bug the same day
> (`trans_id` read at `[2:0]` when the generated code reads `[255 +: 3]`), the lesson is one rule
> from two ends: **confirm against the UNFILTERED artifact, whether the filter is a bit-range you
> assumed or a grep you wrote.**


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
| **`store_syncer_req` + `dyn_wait_store_syncer`** — THE DISCRIMINATOR | **both 0**, all 52 boots | **both 1**, all 8 boots |
| aperture 225 | `0x80` — **nothing** is waiting | `0xd5` — **three** wait conditions |
| `ex_commit.valid` (224 bit 7) | 1 in 45 of 52 — **weak, do not quote as separating them** | 0 in 7 of 8 |
| `mcause` | 25, a real capability fault, `tval = 0` | none; the latch holds stale kernel traffic |
| fault site | present in the image | **verified absent** from every draw artifact |

Measured in the SAME boot series, same bitstream, same query, same compiler — an `-O0` arm gave the
S-12 column and the `-O1` arms gave the S-13 column, so optimisation level is the only variable.

**THE DISCRIMINATOR IS THE SYNCER PAIR, NOT `ex_commit.valid`.** Two signals from different
apertures and different parts of the design agree **perfectly across 63 boots** — `store_syncer_req`
(224 bit 5) is 0 in every one of the 52 `0x80` boots and 1 in every one of the 8 `0xd5` boots, never
separating from `dyn_wait_store_syncer` (225 bit 6). `ex_commit.valid` is 45 of 52 on one aperture.
Quote the first; do not quote the second as separating the classes.

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


## Aperture 224 paired with 225 across 63 boots — one result clean, one weakening

| 225 | 224 | n | `store_syncer_req` | `ex_commit.valid` |
|---|---|---|---|---|
| `0x80` | `0x9f` | 45 | 0 | 1 |
| `0x80` | `0x1d` | **7** | 0 | **0** |
| `0xd5` | `0x7f` | 7 | 1 | 0 |
| `0xd5` | `0xff` | 1 | 1 | 1 — but see caveat |
| `0x00` | `0x9f` | 3 | — | — (225 all-zeros, void) |

**CLEAN RESULT: the DYN wait flag and the store-side indicator never separate.** `store_syncer_req`
is 0 in **all 52** `0x80` boots and 1 in **all 8** `0xd5` boots. Across 63 boots there is no case of
one without the other. On the reading the RTL lane pre-registered, that points at the **store path
as the shared resource** and makes *"one stuck thing blocks both"* the better account — and it is
consistent with the `0xc5` absence rather than in tension with it.

**WEAKENING: `ex_commit.valid` is a poorer discriminator than this folder claimed.** **Seven of the
52** `0x80` boots have `ex_commit.valid = 0` (`224 = 0x1d`). So "S-12 means an exception stuck at
the head of commit" holds for 45 of 52, not for all. **`0x1d` NOW HAS AN ACCOUNT and is not an anomaly.** It differs from `0x9f` in exactly two bits,
`ex_commit.valid` and `flush_ctrl_if`, and those are CAUSALLY LINKED: `controller.sv:232` asserts
`flush_if_o = 1'b1` on `ex_valid_i || eret_i || set_debug_pc_i`. So `0x9f` is "exception valid,
hence flushing" and `0x1d` is "no exception valid, hence not flushing", with every other bit
identical. Both are internally consistent, and `0x1d` is a coherent SUB-POPULATION of the `0x80`
class rather than a corrupt read. Since `ex_commit.valid` is LIVE, the honest reading is that in
those 7 the exception was not valid AT THE MOMENT THE SWITCHES WERE READ — whether that is a third
wedge class or the same wedge sampled differently is NOT determined by these bits.

**The one `0xd5` boot with `ex_commit.valid = 1` is SUSPECT, and the reason is sharper than
"all-ones looks dead":** `0xff` differs from the population mode `0x7f` in EXACTLY ONE BIT, with
the other seven identical to all seven other `0xd5` boots. A single-bit difference from the mode is
far likelier to be a stuck bit than a distinct state. **Recorded as suspect rather than discarded —
if a second `0xff` ever appears it stops looking like a stuck bit.**

**So the surviving clean statement is the syncer one, not the exception one.** The `225`-based
two-class split still looks solid; the `ex_commit.valid` bit should not be quoted as separating
them cleanly.
