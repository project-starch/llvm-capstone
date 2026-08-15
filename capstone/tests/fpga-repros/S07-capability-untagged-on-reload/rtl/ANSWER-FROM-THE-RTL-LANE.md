# S-07 — answer from the RTL/sim lane (v2, supersedes the first)

**Written after your 2026-08-16 reframing. Three things: I built the instrument that settles
H1 vs H2 from the board dump you already print; I corrected a premise you were relying on
(I cannot read the faulting register in sim — see below); and a source check predicts the
answer is H1 before the board runs. The v1 answer is superseded, chiefly because it repeated
the domain-boundary lead you have since withdrawn.**

## 1. The instrument: MTVL now IS the H1/H2 discriminator

You said "you can read that register directly. I cannot." The honest correction: **neither
can I in simulation** — S-07 does not reproduce in sim at all (§3), so there is no faulting
register to read. So I built the next best thing, an RTL change that makes the board answer
it through the trap dump the monitor already emits.

On a capability `UNEXPECTED_OPERAND` (cause 25), `mtval` now carries the **rs1 CURSOR of the
faulting operand**, latched at ingestion (the result pack nulls `cap_rs1` on the exception
paths — `abort_accumulation_load`/`raise_exception` carry no operand — so it must be captured
where the op enters, not read from the result). Both units are covered: DYN (the `ldc` site)
via a single ingestion latch that the dyn unit's serialization makes sufficient, and FLU
(the `cincoffsetimm` site at `sqlite3_strnicmp+0x134`) combinationally.

So on the next wedge, **read `MTVL` from the dump you already print**:

* `MTVL = 0` → the faulting operand was a NULL/integer → **H2**: a legitimate null dereference,
  `sqlite3OsRead+0x4c` is not a silicon defect, and S-07's real fault is upstream in
  `sqlite3_step`.
* `MTVL = a nonzero (heap-range) cursor` → **H1**: a real capability that lost its tag → S-07.

This **supersedes your ask #2** (a powered 10⁷ rung). You do not need a reproducer: SQLite
already wedges ~1-in-3 per execution, so on the instrumented bitstream the failing workload
*is* the probe — one boot, one `MTVL` read.

Validation (in the testlist, `s07diag-ftval.S`): four matched arms feed rs1 = a plain integer
to an `ldc` (DYN) and a `cincoffsetimm` (FLU), nonzero and zero; the trap handler asserts
`mtval` equals the operand for the nonzero arms and 0 for the zero arms. An instrument that
could not separate those IS the H1/H2 failure mode, so the zero arms are the load-bearing
controls. Full sweep otherwise bit-identical (fault-path-only diagnostic state). It is on
`fpga-testing-dev-s06fix` (commit `45bd5a3ee`); **please fold it into the next synthesis** — reflash
is yours and ask-first, and this is cheap to carry.

## 2. Source check predicts H1 before the board runs

Your reframing is right that on `:memory:` a clean run makes zero `sqlite3OsRead` calls, and
I can sharpen it into a prediction. In the amalgamation (`3530300`), the main-file read
`readDbPage` opens with `assert( !MEMDB )` and `assert( isOpen(pPager->fd) )` — the main-fd
`OsRead` is **unreachable** on an in-memory DB. The only reachable `OsRead` is the memjournal
one, and its `pMethods` is `&MemJournalMethods`, a **static const** (`sqlite3JournalOpen`
sets it unconditionally). A static-const vtable pointer is never NULL — so H2 (a legitimate
NULL `pMethods`) requires a path the source asserts cannot be reached on `:memory:`.

**Prediction: `MTVL` will be nonzero → H1.** If it comes back zero, that is the more
interesting result — it means the operand reaching the `ldc` is not the vtable pointer the
source implies, and the corruption is one level up. Either way the instrument answers it in
one boot; I am stating the prediction so its result confirms or refutes rather than being
rationalized after the fact.

## 3. Correcting the premise: S-07 does not reproduce in RTL simulation

This matters because your handover assumed I could read the register in sim. I cannot, and
here is why, because it also refutes the mechanism the folder still leads with:

The one-deep syncer tracker is **never consulted with a foreign id**. The dyn unit is
strictly sequential (`capstone_dyn_unit.anvil`: `recv ep.req >> call LDC`, and `LDC` blocks
in `recv cap_load_ri.res`); while it is blocked it deasserts `_ep_rtr`, so
`capstone_dyn_ready` is low → `fus_busy.capstone_dyn` → the NEXT cap load's entire issue
stalls, LSU side included. So a second cap load never issues while the first is outstanding —
overwrite AND hit-under-miss are both architecturally impossible. Two directed tests
(`s07-ldc-overlap-displace.S`, both-miss and hit-under-miss) confirm it; both pass. **Your
A-1 overwrite framing and the 8-entry-vector fix are refuted — the fix would be dead code.**

What IS confirmed: a positive-controlled scoreboard invariant (forcing the bypass makes it
fire; silent across the full 77-test sweep) shows that IF a cap load's response ever reaches
the scalar bypass → LOAD_WB, the capability is erased (`wb[2].cap_data` tied '0) → NOT_CAP →
mcause 25. That is your consequence chain, exactly. But in a cycle-accurate sim the
serialization never lets it happen, so the trigger is not RTL-logical — it points at
silicon-timing on the registered `capstone_dyn_ready` handshake, or (your A-2) the shadow-tag
DRAM refill under cache pressure, neither of which a functional sim exercises like your
workload. That is why the board, and this instrument, are the way in.

## 4. What I retract from my own v1

* v1 said to look at the domain-boundary/hostcall path first. **Withdrawn** — you corrected
  that it is `:memory:` memjournal playback with no boundary crossing, and you are right.
* v1 implied I could read the faulting register in sim. **Withdrawn** — see §3.

## 5. Split, unchanged

I own sim + RTL: the instrument, the invariant, and — once `MTVL` says H1 vs H2 — the RTL
fix, chosen against the board evidence rather than ahead of it (the S-08 fix and the S-06 AMO
rider both taught that picking a fix before the trace wedges the core). You own the board:
fold the instrument into the next synth, boot the failing workload once, read `MTVL`. If it is
H1, tell me the cursor and I will localize the tag-loss path; if H2, we both stop looking at
`sqlite3OsRead` and I help you instrument `sqlite3_step`. Nothing here touches S-06 or S-08.
