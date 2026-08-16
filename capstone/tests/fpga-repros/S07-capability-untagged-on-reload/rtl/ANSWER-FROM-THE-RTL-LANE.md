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


---

# ADDENDUM 2026-08-17 — the tag-history probe (batch 2), and what the first probe answered

## What the sticky displacement bit answered

Boot 5 caught a genuine wedge — mcause 25 at `sqlite3OsRead+0x4c`, control passed, domain
confirmed entered, trap latch confirmed as the domain's own — and **switch 204 read 0x00**.
So on that wedge nothing was displaced onto a scalar writeback port: **case (a) is not
supported**. Per the rule agreed before the data, that is "not case (a) on this run", not
"case (b) established"; a second wedge is wanted. The same byte also read 0x00 across three
complete extended workloads on a clean boot, which rules out displacement as a routine
background event.

## The gap that made a retry probe unsafe: case (c)

The faulting site spills a capability to a stack slot and reloads it. Cases (a) and (b) both
assume the granule in memory was tagged and the tag went missing. It need not have been:
post-S-06 an `stc` writes its tag from the rs2 register tag, so an untagged register produces
an honestly untagged granule and the reload returning NOT_CAP is **correct**, with the fault
upstream of both memory and the syncer. A retry probe cannot see this — it re-reads a granule
that is honestly untagged and stays untagged, which is indistinguishable from case (b) — so a
retry would have produced a confident wrong verdict.

## The probe that separates all three

Verdict byte at **switch 208** (UART-safe, readable mid-run):

| bit | meaning |
|---|---|
| 7 | `ldc0_valid` — an LDC came back untagged and was recorded (one-shot) |
| 6:5 | `ldc0_src` — 0 = L1 hit, 1 = miss refill (tag memory), 2 = write-buffer forward |
| 4 | `stc_valid` — a capability-granule store was recorded |
| 3 | `stc_ctag` — the tag that store WROTE |
| 2 | `gran_match` — both records are the same 16-byte granule (computed in hardware) |
| 1 | `stc_clobbered` — a plain store later overwrote that granule |
| 0 | 0 |

Decode, in order:

1. `clobbered = 1` → **no verdict**: a plain store legitimately cleared the tag.
2. `match = 1, stc_ctag = 1, clobbered = 0` → the tag was written and read back 0: **genuine
   loss**, and `ldc0_src` says where (hit = cache-side, refill = memory/tag path, wbuffer =
   forwarding).
3. `match = 1, stc_ctag = 0` → **case (c)**: stored untagged; the reload is correct and the
   fault is upstream.
4. `match = 0` → the untagged load's granule was not the last recorded capability store.
5. `ldc0_valid = 0` → no untagged capability load was seen at all.

Also in the batch: **switch 212** mirrors the trap summary at a UART-safe aperture (bank 7
reg 31 is only reachable at 255, which hijacks the console); the faulting operand (`tval`) is
latched with mcause/mepc so it survives the wedge; and the two granule addresses are exposed.
Everything except 208 and 212 is a wedge-only read — post-run odd apertures are safe because
`debug_led_o` is the LED pin, not the console TX.

## Two bugs this probe had, caught in simulation

Recorded because both would have produced a confident wrong answer on silicon:

* the store record latched **paddr = 0** on every store (captured at request time; the
  physical address arrives from the MMU a cycle later). Now latched when the entry is pushed.
* a granule stored tagged, then legitimately cleared by a plain store, then reloaded
  untagged would have read as **"hardware tag loss"**. `cap-tag-cache` performs exactly that
  sequence, which is how it surfaced. Hence the `clobbered` bit.

## Why the mtval instrument is not the readout

A capability fault inside a domain **wedges rather than trapping to mtvec** (`core/cva6.sv`,
the debug-mux comment), so the monitor's EXCX/MCAU/MTVL block never runs; the board lane
measured this over four boots (3 mcause-8 wedges print it, 6 mcause-25 wedges print none) and
GDB does not rescue it either, since a nested trap clobbers the CSRs first. The mtval work
stays — it is the readout in simulation and for non-domain faults — but on this path the
debug-mux bits are what survive.
