# S-12 — `mcause 25` at `sqlite3WhereCodeOneLoopStart+0x8c`: the operand is zero, and it is not software

> ## WHAT S-12 ACTUALLY BLOCKS — read this before calling it "the SQLite blocker"
>
> **It has never blocked SQLite's built-in correctness workload.** That workload passed on silicon
> 3/3 on 2026-08-20, days BEFORE this folder existed, and 14/14 on 2026-08-27 on the current
> compiler. S-12 was found in the **SLT** path, running `q_two` — a test written to push past what
> the built-in workload covers.
>
> **The distinction is the query PLAN, not the SQL text.** The built-in workload does contain a
> two-table join, and that fact was used on 2026-08-27 to argue S-12 could reach it. **That was
> wrong**, and it is the error this project already has a rule about: *same shape of SQL is not
> same execution plan, and the plan must be READ rather than inferred*. Read, the plans are:
>
>     built-in workload   SCAN nums + SEARCH link USING AUTOMATIC COVERING INDEX   14/14 clean
>     qj4                 SCAN t1   + SEARCH t3 USING COVERING INDEX               RETURNS
>     qj2 / q_two         SCAN ...  + SCAN ...                                     WEDGE
>
> **What it DOES block: SLT COVERAGE.** The runner itself works on silicon — `qj4` passed its query
> outright (`query_pass=1 query_fail=0 completed=1`). Any test whose inner where-loop level is an
> unindexed nested SCAN wedges. The real sqllogictest corpus is full of those, so **S-12 is what
> stops the corpus running on silicon**, which is the standing "SLT on silicon" goal.
>
> Three claims that must not be merged: *the built-in workload runs* (true, never blocked by S-12);
> *SLT runs* (true, one domain per boot, indexed joins pass); *the SLT CORPUS runs* (false).
>
> ### WEAKENED SAME DAY — the plan is a CORRELATE, not a mechanism
>
> The table above is measured and stands. The **explanation** attached to it does not, and the
> check that broke it is one instruction lookup:
>
> **The S-12 fault pair sits at `sqlite3WhereCodeOneLoopStart+0x88`** — near the top of a
> 4,606-instruction function, on the path taken on EVERY call, whatever the plan. Verified in the
> SLT image that ran both queries (`fn` at `0x104788`, the same address as the historically
> faulting binary), with the documented sequence intact:
>
>     +0x80  movc a4, zero
>            stc  a4, 0x0(a5)
>     +0x88  ldc  a4, 0x0(a0)
>            cincoffsetimm a4, a4, 0xb0
>
> `qj4` is a two-level plan, so it called that function **twice** and executed `+0x88` twice —
> and returned. **So the plan does not gate whether the vulnerable code runs.** "Indexed joins are
> safe because they avoid the failing path" is therefore NOT available as a mechanism.
>
> What survives is a correlation: unindexed nested SCAN plans have wedged repeatedly, indexed ones
> have not. A mechanism consistent with both facts would have to work through something the plan
> changes INDIRECTLY — how much codegen runs before the second call, heap and cache state, timing —
> rather than through which instructions execute.
>
> ### RETRACTED THE SAME DAY — `qj4` WEDGES. The plan is not the variable at all.
>
> The redraws were run rather than assumed, and they kill it. Five further DISTINCT `qj4` images
> (`TEXT_PAD` 0/48/96/144/192, sha256 5-of-5 unique), one SLT domain per boot:
>
>     pad 0    returned      pad 48   NO RETURN      pad 96   returned
>     pad 144  NO RETURN     pad 192  returned
>
> With the original pass that is **4 returned / 2 wedged in 6 draws**. Against the `qj2` rate of
> 54%, P(<= 2 wedges in 6) = **0.27** — statistically indistinguishable. **An indexed two-level
> join wedges too.**
>
> So "the trigger is generating code for an unindexed nested SCAN" is **RETRACTED**. It was stated
> twice today — first as a mechanism, then weakened to a correlation — and the correlation is now
> gone as well. Both rested on `qj4` = 1 clean draw at p = 0.46.
>
> **What the numbers now say, and it is a different shape of question:**
>
> | workload | wedges / draws | vs a 54% rate |
> |---|---|---|
> | built-in extended (no `--slt`) | **0 / 14** | p = 1.9e-5 — genuinely different |
> | SLT `qj4`, two levels, INDEXED | 2 / 6 | p = 0.27 — indistinguishable |
> | SLT `qj2` / `q_two`, two levels, unindexed | wedges | the baseline |
> | SLT `q_one`, one level | returns (3/3 as controls) | — |
>
> The only arm statistically distinguishable is the **built-in workload**, which is also the only
> one that does not go through the SLT harness. So the live question is no longer the query plan —
> it is what differs between the SLT path and the built-in path, with one level versus two still
> unexplained inside SLT.
>
> **Method note, because this is the second self-refutation of the same account in one day:** both
> versions rested on a single clean draw, and both were written before the redraws were run. The
> redraws cost five boots and would have cost the same five boots before the claim as after it.

> **ARRIVED WITH A HANG RATHER THAN A FAULT?** If the domain stopped with **no exception**
> (`ex_commit.valid = 0` on aperture 224) and aperture 225 reads **`0xd5`** — three wait conditions
> asserted, a store genuinely outstanding — that is **[S-13](../S13-o1-dyn-rev-node-hang/)**, a
> different defect, not this one. S-12 is the case where a capability fault STICKS at commit:
> `mcause 25`, `tval = 0`, aperture 225 reading `0x80` with nothing waiting. At `-O1` the S-12
> fault site is verifiably absent and the machine hangs as S-13 instead.

**Sibling issues, so a reader who arrived with the wrong symptom can leave now.** If your symptom is
an **untagged capability surviving a store/reload pair**, that is **S-07**, not this. If it is a
**write-buffer tag forgery**, that is **S-09**; a **write-buffer forwarding residual**, **S-10**; a
**scalar store clobbering capability metadata**, **R-18**; **`movc zero` metadata left in a slot**,
**R-19**. The issue this one is closest to, and which it may yet turn out to BE, is
**R-20 — `stc`/rs1 cursor forwarding** (`R20-stc-rs1-cursor-forward-x10/`). Read that one before
this one.

**This folder is about a fault whose LOCATION AND MECHANISM ARE BOTH UNESTABLISHED.** It is open,
and written to be handed over in that state rather than held until it is tidy. What is settled is
what has been *excluded*, and the exclusions are real even though the localisation is not:

> **A committed store put cursor `0x82be4cf0` into WORD 0 of the slot. That word still held it in
> DRAM at the wedge. The FLU received cursor 0.**
>
> This excludes a **software NULL**, and excludes anything having **persistently overwritten**
> word 0 in DRAM. It does **NOT** locate the fault.

**The shadow-tag half of that evidence is WITHDRAWN as evidence about the load.** The byte read
`0x01`, and that is a true statement about DRAM and a useless one about the `ldc`. The tag the load
consumed comes from a **different storage**: `wt_dcache_mem.sv:143` declares a per-{set,way} L1
array `cap_tag_q`, and `rd_ctag_o` (`:319-338`) is sourced from the write buffer, from refill
`user` bits, or from `cap_tag_hit` — **never from a DRAM tag read**. A GDB read of `0xBC58CC63`
therefore has no path to the bit the load actually saw.

Worse, the documented desync runs in **exactly** the direction that would fool this measurement.
`wt_dcache_wbuffer.sv:604-620`: *"ctag is sampled TWICE — at TX ISSUE for DRAM and at TX RETURN for
L1 … `stc G` then `sd x0, G+0` gives DRAM ctag=1 and L1 ctag=0."* **Stale-high DRAM with correct-low
L1 is a known failure mode of this hardware, and the measurement sits inside it.** The usual
mitigation — that the two converge — does not obviously apply on a **wedged** core, since
convergence needs the entry to re-drain and whether that window is still open at read time is
unmeasured.

**Read that limitation, because a stronger claim was published from these same numbers and had to
be retracted.** The reasoning was: memory intact at the wedge ⇒ the memory path is innocent ⇒ the
fault is in operand delivery. That inference is invalid. **Every documented memory/load-path defect
on this core is transient and self-heals long before a debugger read seconds later** — write-buffer
residency, the issue/return desync (`wt_dcache_wbuffer.sv:612-619`), S-10b. So "the granule is
intact at T+seconds" is predicted by **both** arms of the fork and separates nothing.

The existence proof is on this very silicon: **S-10b (`c867dfcbb`) is ABSENT from the resident
bitstream** — verified, `git merge-base --is-ancestor c867dfcbb 84ed6eafb` returns false — and its
commit records a load returning `0x0000000000000000` while the store's data was in memory. A
load-path fault producing a clean zero with memory intact is exactly the combination the retracted
claim asserted was impossible.

**WHAT THE LOAD RETURNED HAS STILL NEVER BEEN MEASURED.** That is the open question, and both a
memory-path and a delivery-path explanation remain live.

---


> ## ⚠ STATUS, 2026-08-25 — READ BEFORE ANY SECTION BELOW
>
> **Every section of this file headed "THE MINIMAL REPRO REPRODUCES", "CONFIRMED … produces the
> S-12 SIGNATURE", "…still reproduces", "…and still faults", and "IT FAULTS ON ONE ITERATION" is
> RETRACTED.** All of them rest on arm 4, which read a subject slot it never wrote. See
> **"RETRACTION — every arm-4 measurement in this folder is VOID"** at the end of this file.
>
> **Net status: S-12 has NO minimal reproducer.** With the detector proven to fire (arm 2 reports
> `bad == REPS`), the real shape returns clean:
>
> | arm | shape | consumer | reps | result |
> |---|---|---|---|---|
> | 1 | subject store + 9 intervening stores | `lcc` (counts) | 512 | `0xC12A1000`, bad=0 |
> | 4 | **identical** | `cincoffsetimm` (**raises**) | 1 | `0xC12A4000`, no fault |
> | 4 | **identical** | `cincoffsetimm` (**raises**) | 512 | `0xC12A4000`, no fault |
>
> Arms 1 and 4 are a matched pair differing in exactly one thing — whether the consumer counts or
> raises — and both are clean, so the consumer is not the missing ingredient. The 40-line window
> does not reproduce S-12. That is a result, not an absence of one.
>
> **Also corrected:** the recorded finding "the stored value is REVOKE-typed" is **inverted** — it
> is **NONLIN**. Read the scope of that carefully, because it is narrower than it looks: it is
> about **the repro's `v`**, not about SQLite's value. NONLIN means the move-clear does not fire
> *in the repro*. It kills the move-clear account **for S-12** only if SQLite's stored value at
> the fault site is also NONLIN — **and that has never been measured.** Same caveat on "the five
> NONLIN simulations were in the right configuration": right only if production's value is NONLIN.
>
> **That makes SQLite's value type the discriminating unknown**, and the kernel's own line — `v`
> is "any tagged capability; its identity is irrelevant" — is now the weakest assumption in this
> folder, because type is exactly what gates the clear set at `load_unit.sv:225-226`. If SQLite's
> value is in that set, the repro never exercised the mechanism it was built to test, and an arm
> with a matching-type `v` is the first variant that *can* reproduce.

## REFUTED: the load-syncer mispair. The precondition is unreachable under maximum pressure

This was the strongest surviving mechanism, and the only one so far proposed that produces BOTH
`tval == 0` and `cap_type == NOT_CAP` from a single event. `capstone_load_syncer` holds ONE pending
trans id and sets it on a new `init` with **no guard** on `req_set`. If a second LDC's init landed
while the first was outstanding, the first LDC's response would fail the trans-id match, be diverted
onto the scalar return path, and `check_load_data` would pair whatever DID match against whatever
`cap_msg` it dequeued — substituting a whole `fat_cap_t`, cursor and metadata together. One coupled
substitution, both halves of the signature.

**It does not happen, and the reason is structural rather than statistical.**

The sim-only checker in the generated `capstone_dyn_unit.anvil.sv` counts the PRECONDITION
(`init-while-pending`) separately from the OUTCOME (`clobbers`), precisely so a clean run can
distinguish "unreachable" from "not caught". Every earlier run reported zero — but those tests issue
7–8 LDCs across ~1700 cycles, widely spaced and hitting in cache. **A zero there measures the test,
not the syncer**, and reading it as a refutation would have been this project's most-repeated
mistake.

`s12-ldc-pressure.S` was built to create the condition: capabilities planted one per 64-byte cache
line, a sweep to evict them so the loads genuinely miss and stay outstanding, then bursts of EIGHT
independent LDCs into eight different destination registers with no data dependency — eight because
`trans_id` is 3 bits and the scoreboard has 8 entries, so that is the most that can be outstanding
at once.

    S12-SYNC: ALIVE (load-syncer checker compiled in)
    S12-SYNC: tick 4600  inits=192  init-while-pending=0  clobbers=0

**192 inits against 7–8 in the ordinary tests**, so the pressure materialised — that counter is the
positive control and it climbed 24-fold. And `init-while-pending` is still **0**: across 192 inits
under the heaviest load the machine can carry, a second init never arrived while one was pending.

The source says why. `func LDC` (`capstone_dyn_unit.anvil:317-378`) does
`send cap_load_ri.init(...)` and then `send cap_load_ri.req(result) >> let msg1 = recv
cap_load_ri.res >>` — the `recv` **blocks the thread**, so the round trip completes before another
LDC can reach its own init. The usual caveat that the generated Anvil hardware is a flat concurrent
machine, and that sequential `.anvil` reading does not constrain ordering, is why this is recorded
as **measurement first and source-reading second**: the count is the evidence, the structure is the
explanation for it.

**Neither leg alone would be enough.** A source argument on a concurrent machine is not binding, and
a zero without occupancy is not a result. Together they are, and the mispair is dead.

## The LDC clear does not race the NEXT store either — matched pair, clear proven to fire

`s12-linear-clear.S` was drafted to test the one write-ordering question left: an LDC of a
clear-class capability zeroes the source granule, and that clear write and the FOLLOWING `stc` to
the same granule share an 8-entry write buffer. If a merge ever landed the clear after the store,
the granule would read back as the literal clear payload — `store_unit.sv:462-469` drives data 0,
user 0, ctag 0, which is bit-for-bit `create_cnull()` and bit-for-bit the observed operand, `tval`
included.

    arm   value type   load returned            granule word 0 after the load
    A     LINEAR       correct cap, Type 1  16/16   0            <- clear FIRED
    B     NONLIN       correct cap, Type 2  16/16   0x80004000   <- clear correctly did NOT fire

The arms differ in exactly one thing and the gate is shown to discriminate rather than to be stuck
on: the clear fires throughout A and never in B. **No misordering occurs.** Each iteration's store
lands after the previous iteration's clear — otherwise iteration N+1's `ldc` would have returned the
null payload, and all 16 returned correct capabilities.

**Two honest limits.** The granule read-back sits between the load and the next store, so it proves
the clear fired but does not itself observe the ordering; the ordering is answered by the following
iteration's load result, which is a weaker instrument than intended. And 16 iterations of bare-metal
write pressure is not SQLite's after millions of instructions.

> ### THE PREVIOUS VERSION OF THIS TEST FAILED IN A WAY THAT LOOKED EXACTLY LIKE S-12
>
> It raised `UNEXPECTED_OPERAND` — mcause 25, the S-12 exception — and by exception name alone that
> reads as "S-12 reproduced in simulation". It is nothing of the kind. The test made its BASE
> capability LINEAR and then did `MOVC(a0, s1)`; a linear capability is move-only, so that consumed
> `s1`, and the next `cincoffsetimm` on the result raised correctly. Correct hardware, broken test.
>
> **What settles it is `ldc-inits=0` and `dyn-dispatches=0` in the same log — NO LDC EVER ISSUED**,
> at cycle 533, during setup. A mechanism that begins with a load cannot be under test in a run that
> never reached one. The lesson is not "check the exception name" but that the counter proving the
> construct actually executed is what makes any verdict readable, and it is the reason the rewritten
> version treats `ldc-inits` as its positive control.

## S-10b excluded for this window — the STALL hazard only, and not on the store list

S-10b's mechanism can only stale the METADATA HALF OR TAG of a granule-aligned LDC: a pending store
to word 0 shares the load's `[11:3]` and does stall (`store_buffer.sv:279,287,293`), so only a
word-1 store can be missed. S-10b therefore predicts `tval` = the stored cursor
(`ariane_pkg.sv:766-784` passes the cursor through untagged; `ex_stage.sv:490` reports it as
`fu_data_i[0].operand_a`). The latched `tval` is **0**. That is the argument, and it holds without
enumerating a single store — which matters, because a store list cannot cover stores OLDER than the
window that are still resident in the queues, and a resident older store to `s0-0x68` is exactly the
S-10b case.

**Scope, because the wider wording would be false.** This excludes the store-buffer STALL hazard
only. It does NOT clear S-10 (`wt_dcache_mem.sv:280`, still word-granular in the resident bitstream),
S-07, or write-buffer forwarding. Neither S-10 nor S-10b is in the flashed tree —
`git merge-base --is-ancestor` returns false for both against `84ed6eafb`.

> ### RESOLVED, SAME DAY — the value is NONLIN, and the subject slot is EXONERATED on two legs
>
> The contradiction below is settled, and not by preferring one instrument. A translate-time probe
> on the `stc` at `+0x40`, reading `cap_type` out of QEMU's register file at that exact pc, measures
> **NONLIN 16/16** — qj2 7/7, q_one 4/4, q_two 5/5, with `LIN=REV=UNINIT=SEALED=SEALEDRET=untagged=0`
> in every run. The pc match is unique in the image (`pc & 0xfff == 0x7c8` plus operands
> `rs1=x10, rs2=x12, imm=0`, exactly one match). **The instrument has a positive control**: the same
> reporting path emits `LIN`, `SEALED` and `SEALEDRET` for OpenSBI's own boot stores, so it can name
> clear-set types and simply does not see one here. The value is argument 3, `pWInfo` — a
> `WhereInfo*` whose measured bounds span exactly the 256 KiB memsys5 arena.
>
> **And a structural leg that needs no type measurement at all.** The slot is stored ONCE and loaded
> **three times per call** in a straight-line entry block with no intervening store — `+0x88`,
> `+0xb8`, `+0xf4`, with `a0` not redefined between them and no branch in `0x104788..0x1048b0`
> (12 loads, 1 store across the whole function). If the type here were clear-set, load #1 would zero
> the granule and load #2 at `+0xb8` would raise off a NOT_CAP base **on every call, one-level plans
> included**. One-level has never wedged in 11 draws. So the type at this site cannot be clear-set,
> whatever any single probe reported.
>
> The board reading of REVOKE (`retval 0xC12A5200`) therefore does not survive as a statement about
> THIS value. It is N=1 against a positive-controlled 16/16, and the structural leg rules it out
> independently. What it might still be a true statement about — some other object, or a genuine
> type corruption on silicon that QEMU cannot see — is **not established either way** and should not
> be quoted in either direction.
>
> **The mechanism CLASS is untouched by this and remains live.** QEMU's `csldc` performs no source
> clear, so a clear-set capability in a slot that is loaded twice is invisible under QEMU and fatal
> on silicon — cursor 0, NOT_CAP, `tval = 0`. The `-O0` codegen makes store-then-reload-the-same-slot
> pervasive (`0x10485c`: `cincoffsetimm a1,s0,-0x100; stc a3,0x0(a1); ldc a1,0x0(a1)`). The open
> question is now narrow and answerable: **which `stc` sites store clear-set types, and is any of
> those slots loaded more than once?**

> **THE CONTRADICTION AS IT STOOD, kept because the reasoning matters.** Two measurements of "the value's type"
> are recorded here and they disagree, because they are measurements of DIFFERENT OBJECTS:
>
> * above, on the board, of **the stored value**: `retval 0xC12A5200` → `lcc` selector 1 = 2, and
>   that selector reports `cap_type - 1`, so raw type **3 = REVOKE** — which IS in the clear set;
> * later, under QEMU, via `ARGP`: `ty1=1 ty2=1` → **NONLIN** — but `ARGP` reports the function's
>   INCOMING ARGUMENTS, not the value `a2` that `stc` writes at `+0x40`.
>
> The section headed "The discriminating unknown is answered: SQLite's value is NONLIN too" treats
> the second as having settled the first. **It does not.** Until this is resolved, no claim in this
> folder that rests on "the clear cannot fire at the S-12 site" should be quoted. The clear-set
> mechanisms are refuted here on their own evidence — the six-type sweep and the matched pair above —
> rather than on the type argument, so those refutations stand either way.

## REFUTED: the double-loaded clear-set slot. Zero such slots exist, on any arm

The mechanism was: a clear-set capability stored into a slot that is then loaded TWICE. On silicon
the first load is a MOVE and zeroes the granule, so the second reads cursor 0 / NOT_CAP and the
consumer raises with `tval = 0`. Under QEMU it is harmless forever, because `csldc` performs no
clear. Silicon-only by construction, which is S-12's most distinctive property — and the `-O0`
codegen makes store-then-reload-the-same-slot pervasive, so the shape is abundant.

A whole-program, address-keyed sweep over 16-byte granules, modelling the RTL condition
(`result_tag_o && type in clear-set && rs1_perm_write`, `load_unit.sv:224-230`):

    arm                   domain stc   tagged    CLEAR-SET   DOMAIN HITS   monitor hits
    qj2      (two-level)      72823    58223         3           0             18
    q_two    (two-level)      49483    39143         3           0             18
    q_one    (one-level)      48361    38208         3           0             18
    builtin  (no --slt)      187231   155414         3           0             24

**Zero, on every arm.** The domain's only three clear-set stores are all `SEALEDRET`, all from one
pc — the domain entry glue — one per `call_dom` entry, each to a different stack granule, and none
is loaded again before the next store to it. That is the intended move.

**Every hit is monitor-side, and the counts run the WRONG WAY.** The built-in arm, which passes
14/14 on silicon, has MORE of them (24) than the two-level SLT arms that wedge ~54% (18). So they
cannot be the trigger, and there is no discriminator between one-level and two-level anywhere in
this data.

**Three positive controls, all fired**, which is what makes the zeros admissible: forcing NONLIN
into the clear set yields 144,425 hits on qj2 (so the detector works); the unforced type filter
reports clear-set stores as a proper SUBSET of tagged stores, naming `LIN` and `SEALEDRET`; and the
plain-store disarm path fired 38/85 times in controls and 0 in real arms. Coverage is structural
rather than hopeful — all three `cap_mem_map_add` call sites plus the context save/restore path were
hooked, closing the QEMU analogue of the dom-switch LDC blind spot, and the "tagged capability
loaded from an untracked granule" meter reads 0 everywhere.

> **An anomaly worth carrying, not chased here.** The model says the monitor-side hits should be
> fatal on silicon, yet the board boots. The likely explanation is the known two-monitor-copies
> gotcha — QEMU boots `caplifive-buildroot`'s `fw_jump` while the board's firmware is built from the
> `caplifive-system` fpga/ariane tree. The domain verdict does not depend on it: it rests on there
> being zero double-loads in the domain at all.

## THE REFLASH HAPPENED, AND S-12 SURVIVED IT. S-10 is EXCLUDED as the cause

The board was reflashed to `caplifive_s10fix_80843404c.bit` on 2026-08-27 (non-volatile;
`flash_state: done`, `nv_bitstream_name` confirmed). The same two images were then re-run through
the same harness, alternating, so the ONLY variable is the silicon:

    image     BEFORE (caplifive_s07clear_84ed6eafb)   AFTER (caplifive_s10fix_80843404c)
    pad48     wedged 3/3                              wedged 2/2
    pad144    wedged 2/2                              RETURNED once, wedged once

**`pad48` wedges 5 of 5 across both bitstreams.** `pad144`'s single return is unremarkable — it was
2/2 on N=2 before, and one return in two draws is expected at any plausible per-image rate, let
alone the ~54% population rate.

**So the S-10 write-buffer fix does not address S-12, and S-10 is excluded as its cause.** This was
the pre-registered reading, written before the runs: *both still wedge → S-10 is excluded, and
either one of the ten exclusions is wrong or the trigger is outside every model we have.*

**This was worth the reflash.** S-10 was the last live "the silicon has a known unfixed defect"
explanation, and it is now a measurement rather than an open question. What it does NOT clear:
S-07, write-buffer forwarding generally, and anything else divergent between the two trees — the
bitstreams are DIVERGENT, not sequential, and `s10fix` also lacks the S-07 LDC recorder (an
instrument, not a fix), which is why switch-208 readings in the after-runs carry the "UNKNOWN
SEMANTICS for this bitstream" caveat and must not be read as tag verdicts.

> **Note on how the flash was finally performed**, since our own docs are wrong about it. The
> documented tool `board_reflash_only.py` **has never existed in git**. The working sequence is:
> power ON, **let the board settle** (~30 s — flashing within a second of power-on returns `error`),
> then `POST /api/flash-bitstream {filename}` and **wait for a FRESH `flash_state` event**. Two
> apparent failures were our own instrument errors: reading the *cached* `flash_state` from a prior
> attempt, and a presence check that iterated `{"files": [...]}` as a dict and so enumerated the
> single key `'files'` instead of the filenames. The board was never at fault for either.

## BASELINE PINNED for a future reflash: two images that wedge REPEATEDLY, not once

The S-10 fix is synthesis-proven (`80843404c`) and **not flashed**; the exclusions in this folder
cover the store-buffer STALL hazard only, so S-10 itself, S-07 and write-buffer forwarding remain
live on the resident silicon. The obvious experiment is to reflash and re-run — but at the ~54%
population wedge rate a single clean draw afterwards would mean almost nothing (P ≈ 0.46 by chance).

So the per-image rates were pinned FIRST, on the resident `caplifive_s07clear_84ed6eafb.bit`, using
two images that had each wedged exactly once (N=1 apiece). Runs alternated between the two images
rather than repeating one, so board or firmware drift cannot masquerade as an image property.

    run        enter  return   verdict
    pad48        2      0      real wedge
    pad144       0      0      VOID -- entry stall (R-16), the domain never ran
    pad48        2      0      real wedge
    pad144       2      0      real wedge

**One of four is VOID and must not be counted.** `SQ: G/enter` present with no `H/return` is the
documented signature of a genuine wedge; `enter = 0` means the domain never started, which says
nothing about the code. Counting it would have inflated the baseline with a boot that ran nothing.

**Admissible: `pad48` wedges 3/3 and `pad144` wedges 2/2**, each including its original draw — five
real draws, five wedges. Under the 54% population rate P(5 of 5) = 0.046, so these are plausibly
higher-rate images, consistent with this folder's per-image clustering finding.

**Why this matters for the reflash.** These two images are now matched before/after subjects where
the ONLY variable is the silicon. If either returns repeatedly after a reflash, that is a per-image
reversal rather than a lucky draw from a population. Budget roughly four clean returns per image to
be decisive; two would not be.

> **METHOD CAVEAT, stated because it weakens the runs slightly.** These boots ran the SLT domain
> ALONE, with no known-good control first — the standing rule is that a boot whose control fails is
> VOID, and without one a "no return" could in principle be a boot failure rather than a wedge. What
> carries the verdict instead is internal: `enter = 2` and `dom-ok = 2` show the monitor created the
> domain and the domain entered, so the boot was healthy up to the point of the test. That is
> weaker than a real control and is why the VOID row above was caught at all.

> **The reflash could not be performed from this side.** `POST /api/flash-bitstream` returns
> `200 {"state":"loading"}` and then transitions to `error` for BOTH server-side registrations
> (`caplifive_s10fix_80843404c.bit` and `caplifive_s10fix.bit`), leaving `nv_bitstream_name`
> unchanged — the board still holds the resident bitstream, and no damage was done. The API exposes
> no failure reason. Note also that `board_reflash_only.py`, cited in `HOW-TO-LAUNCH-ON-FPGA.md` as
> the reflash tool, **has never existed in git**, so this path was never proven from our side. The
> flash needs GUI access or the server-side log.
>
> Checked before attempting, and worth recording: the two bitstreams are **DIVERGENT, not
> sequential**. `s10fix` KEEPS apertures 224/225 and the syncer bits used for classification, and
> loses only the S-07 LDC recorder and its switch-160 clear, which this experiment does not use.

## WEAK, AND RECORDED AS WEAK: clamping the 5th call reduces the wedge but does not remove it

`WhereCodeOneLoopStart` runs 3 + plan depth times, so `dd2_join` makes five calls and the fifth is
the one the extra plan level adds. A build that returns early from call 5 onward
(`CAPSTONE_WCLAMP=sqlite3WhereCodeOneLoopStart:5`), run against the unclamped build in the SAME
boots, clamped first:

    draw   clamped (5th call suppressed)   unclamped control
    k1     *** WEDGED ***                  collateral
    k2     returned                        *** WEDGED ***
    k3     returned                        *** WEDGED ***

Clamped 1 wedge / 3; unclamped 8 / 8 across the session. Fisher ~0.02.

**This is suggestive and NOT decisive, for two reasons that are part of the result rather than
caveats bolted on:**

* **The clamp changes the PROGRAM, not just the call count.** It returns a wrong `notReady`, which
  corrupts the generated code, so the clamped build differs from the unclamped one in more than the
  thing under test. Part of any rate difference measures that corruption.
* **k1's wedge may not be S-12 at all.** With codegen deliberately corrupted a fault can arise
  anywhere, and a wedge reports no `mcause`/`mepc`, so an S-12 wedge is indistinguishable from one
  the clamp itself created.

**More draws would not fix this** — they would only sharpen a confounded comparison. What would fix
it is an intervention that removes the fifth call WITHOUT corrupting codegen, and no such
intervention is currently known.

**So the load-bearing evidence for the nesting result stays the UNCONFOUNDED pair** — `dd6_twostmt`
0/5 against `dd2_join` 6/6 at equal call counts, same binary, same boots, no perturbation. The clamp
neither strengthens nor weakens that; it is recorded here so the next person does not re-run it
expecting a verdict.

## LAYOUT REFUTED: a NOP at the identical point wedges 4/4 where a FENCE returns 0/7

The objection that killed the previous mechanism claim — that the fence's +4 displacement, not its
semantics, might be the cure, since S-12 is layout-sensitive — is now tested and **refuted**.

`CAPSTONE_WNOP_BEFORE` puts a 4-byte `nop` (`addi x0, x0, 0`) at the *identical* injection point.
**The two builds have BYTE-IDENTICAL SYMBOL TABLES** — verified by diffing `llvm-objdump -t`, whose
only difference was the filename in objdump's own header line; stripped, both hash to
`a6174d6e9d1b90a7e24a`. Same four bytes, same displacement of the reload (`0x104814`) and its
consumer (`0x104818`), same addresses throughout. The sole difference is whether the inserted
instruction carries memory semantics.

    build    layout                  inserted instruction        result
    nop      identical symbol table  addi x0, x0, 0  (inert)     WEDGED 4 / 4
    fence    identical symbol table  fence rw,rw     (ordering)  0 wedges / 7

**So the cure is SEMANTIC, not positional.** Layout is excluded as the explanation, and a
memory-ordering mechanism is reinstated on evidence rather than on assumption. The `nop` arm also
supplies something the fence arm could not: a same-layout POSITIVE control, wedging 4/4, which
proves the comparison can produce a wedge at that exact geometry.

**WHAT IS STILL NOT SEPARATED.** Two memory-ordering accounts remain, and a fence cures both:

* **(a) store-to-load drain** — the subject `stc` at `+0x40` has not landed when the `ldc` at
  `+0x88` reads. Predicts a STALE value, which here is a non-zero cursor, so it does not by itself
  explain `tval = 0`.
* **(b) wrong-address forward** — the null capability written one instruction earlier by
  `movc a4, zero; stc a4, 0x0(a5)` at `0x10480c` is forwarded to the reload. Predicts
  `{cursor 0, NOT_CAP}` EXACTLY, which is what is observed.

**(b) fits `tval = 0` and (a) does not**, so (b) is currently the better-supported of the two — and
it is the S-10 / R-19 / R-20 write-buffer forwarding family this folder already lists as live, which
the S-10 reflash did NOT clear.

**THE NEXT DISCRIMINATOR:** move the null-capability store out of the window. `Index *pIdx = 0;`
compiles to that `movc`/`stc` pair; relocating the initialiser to after `pWC = &pWInfo->sWC;` keeps
semantics identical (it is not read in between) and removes the null store from the window.
Wedge disappears → **(b)**, the null store is the forwarded value. Wedge persists → **(a)**.

## RETRACTED: the "store-to-load drain hazard" mechanism, and the dose-response that supported it

**The cure is real. The mechanism is not established, and one of the three data points was
fabricated by an instrument error.**

**1. The middle rung never ran.** `f3`, the only wedge in the entry-fence arm, was an **R-16 entry
stall**, and the driver said so in the log: *"INFRASTRUCTURE WEDGE ... NO VERDICT ... no `SQ:
G/enter` -- the domain was CREATED but never ENTERED ... Do NOT attribute this to the code under
test."* The classifier counted `SLT-SUMMARY` lines behind a `booted` guard testing `Linux version`
or `SQ: A/dom-ok` — **an entry stall emits both**. A gate whose condition the failure mode always
satisfies, which is the exact class this project keeps paying for. It needed `SQ: G/enter` per arm.

    CORRECTED:  not drained          wedged 4/4 in the paired boots
                fence at entry       0 / 3   + 1 VOID
                fence before reload  0 / 4

**Flat. There is no dose-response**, and "Fisher = 0.004" and "the residual is the three trailing
stores" both go with it.

**2. The fence is NOT layout-neutral, and layout now fits BETTER than drain.** It shifts the reload
and its consumer by +4 (`0x104810/0x104814` → `0x104814/0x104818`) and moves **1165 of 3633
symbols**. Codegen is otherwise neutral — 2866 instructions, same order, same registers — so
*semantically* neutral was true and *layout*-neutral was never checked. Decisively: the two fence
placements have **byte-identical symbol tables**. They are the SAME layout perturbation but
DIFFERENT drain doses (3 stores remaining vs 0). Drain predicts they differ; layout predicts they
match. **They match.**

**3. N=4 cannot bear the weight.** This folder's own finding is that behaviour is a deterministic
function of the image and layout selects it, so the fenced boots are **one image draw**, not four.
Layout-null probability is ~0.21–0.46, not 0.045. Pairing controls boot state; it does not control
image identity, which is the confounded variable.

**4. Drain does not predict `tval = 0`.** An unforwarded in-flight store yields the STALE prior
content, and here that is non-zero (`0x82be4cd0` in the halted read). A surviving alternative
predicts `{cursor 0, NOT_CAP}` exactly: a **wrong-address forward** of the null capability stored
one instruction earlier by `movc a4, zero; stc a4, 0x0(a5)` at `0x10480c` — the S-10/R-19/R-20
family this folder already lists as live. **A fence cures both; this experiment cannot separate
them.**

**Also corrected:** the fenced reload was labelled `+0x88`; from `0x104788` it is `+0x8c`. The
address shift went unnoticed at the moment the mechanism was written — which is itself the tell.

**WHAT SURVIVES, and it is worth having:** a `fence rw,rw` immediately before the reload eliminates
the wedge, 0/4 against 4/4 unmodified in the same boots (3 of the 4 signature-confirmed `mcause 25`
/ `mepc 0x828f4814` / `tval 0`; `g4` was a genuine wedge with a non-capability `mcause 3` and is not
an S-12 confirmation). A fence at entry gives the same 0/3. **Usable as a mitigation. Not a
mechanism.**

**THE DISCRIMINATOR, one boot:** rebuild with a **4-byte `nop`** at the identical injection point,
verified to produce a symbol table identical to the fenced build modulo one opcode. `nop` returns →
the +4 displacement is the operative variable and drain is refuted as the cure's mechanism. `nop`
wedges while `fence` returns → the fence's SEMANTICS do the work, and drain or the null-store
forward survives as a class.

## SUPERSEDED — the drain mechanism as originally written
 Closing the window eliminates the wedge, 0/4 vs 4/4

Fencing immediately before the reload removes the failure entirely, and the three conditions form a
monotone dose-response with a **semantically neutral** intervention throughout — no value, control
flow or generated program changes, only the timing of the store path:

    write path between the spill (+0x40) and the reload (+0x88)      wedges
    NOT drained          (unmodified)                                15 / 15
    PARTIALLY drained    (fence at function entry; 3 stores remain)   1 / 4
    FULLY drained        (fence immediately before the reload)        0 / 4

The final arm ran the fenced and unmodified builds in the SAME four boots, fenced first: fenced
returned 4/4, unmodified wedged 4/4. Placement verified in the disassembly —

    10480c  stc a4, 0x0(a5)               last store in the window
    104810  fence rw, rw                  the drain
    104814  ldc a4, 0x0(a0)        +0x88  the reload
    104818  cincoffsetimm a4, a4, 0xb0    the fault site

**STATEMENT OF THE MECHANISM.** The capability spilled by the `stc` at `+0x40` is reloaded by the
`ldc` at `+0x88` **while that store is still in flight in the write path**, and the load returns
`{cursor 0, NOT_CAP}` rather than the stored capability. The consumer then raises
`UNEXPECTED_OPERAND` with `tval = 0`. Draining the write path before the reload prevents it.

**IT ACCOUNTS FOR EVERY PROPERTY THIS FOLDER HAS RECORDED**, which is why it is worth more than the
statistics alone:

* the 36 instructions to the fault are **identical on every call** — the difference is timing, not
  instructions, so no instruction-level repro was ever possible;
* **nine bare-metal reconstructions came back clean** — they reproduced the instructions faithfully
  and had none of the write pressure;
* **nesting, not repetition** — the extra where-codegen level fills the write path; two separate
  one-level prepares (`dd6_twostmt`, same 5 invocations) do not and never wedge;
* **silicon-only** — QEMU models no write buffer, so the window cannot exist there;
* **layout- and timing-sensitive with per-image clustering** — the window is a drain race;
* **one level never wedges** (0/11) — less write pressure ahead of the reload;
* **`tval = 0` with DRAM intact** — the store had not landed when the load read, and had landed by
  the time a debugger looked.

**WEIGHT AND LIMITS, stated rather than implied.** N=4 for the full fence. Because the control
wedged 4/4 in the same boots and the unfenced rate is 15/15, the paired comparison is strong, but
0/4 is not 0/20 — the residual rate is bounded, not measured. And a fence is a *sufficient*
mitigation, which does not by itself identify WHICH hardware structure mishandles the in-flight
store. It points hard at the write/store-buffer forwarding path, and notably the S-10 reflash did
**not** clear that family: S-07 and the word-granular `wbuffer_hit_oh` (`wt_dcache_mem.sv:280`,
still word-granular against a 16-byte capability) both remain live on this silicon.

**MITIGATION AVAILABLE NOW.** One `fence rw,rw` before this reload removes the failure at zero
semantic cost. That is a workaround, not a fix, and must not be described as a fix while the
hardware structure is unidentified.

## A SEMANTICALLY NEUTRAL FENCE CUTS THE RATE FROM 11/11 TO 1/4 — the strongest mechanism evidence yet

Given the path to the fault is 36 branch-free instructions identical on every call, the trigger is
STATE. The state with prior evidence is a store-to-load drain window: the `stc` spilling `pWInfo` at
`+0x40` is reloaded by the `ldc` at `+0x88` eighteen instructions later.

`CAPSTONE_WFENCE` puts `fence rw,rw` at the top of the function. **It is semantically neutral** — no
value, control-flow or generated-program change, only the timing of the store path — which is
exactly what the `WCLAMP` experiment lacked and why that one had to be recorded as weak.

**Verified by disassembly, the fence lands INSIDE the window:**

    1047c8  stc a2, 0x0(a0)      +0x40  the subject store
    1047ec  fence rw, rw                the injected drain
    1047f8  stc a4, -0x5a0(s0)          three stores still follow ...
    104810  stc a4, 0x0(a5)             ... one immediately before the reload
    104814  ldc a4, 0x0(a0)      +0x88  the reload
    104818  cincoffsetimm a4, a4, 0xb0  faults

    draw   fenced        unmodified control
    f1     returned      *** WEDGED ***
    f2     returned      *** WEDGED ***
    f3     *** WEDGED ***  (collateral)
    f4     returned      *** WEDGED ***

**Fenced 1 wedge / 4; unmodified 11 / 11 this session. Fisher = 0.004.**

**Reading.** Draining the write path between the spill and the reload removes most of the failure
but not all of it. That is direct support for the store-to-load drain window as a MAJOR contributor,
obtained without perturbing the program — and it is consistent with the delay-dependence recorded
earlier (bracketed 10 < T <= 600) whose mechanism had been retracted.

**The residual has an obvious candidate rather than being mysterious:** three stores execute AFTER
the fence and before the reload, one of them immediately prior. The fence closes most of the window,
not all of it. A fence placed immediately before the reload would test that, but the reload is
compiler-generated `-O0` spill code and there is no source point that maps there.

**WEIGHT: N=4, and 1/4 is one draw from 0/4 or 2/4.** The direction is significant against 11/11,
but the residual rate is not usefully estimated at this N. More draws would sharpen it, and unlike
the clamp they would sharpen an UNCONFOUNDED comparison.

**Practical consequence:** a fence in this one function cuts the SQLite failure rate roughly
fourfold at zero semantic cost. That is not a fix, and it must not be presented as one while the
mechanism is unidentified — but it is a usable mitigation and a strong pointer for the hardware side
toward the write/store buffer path, which the S-10 reflash did NOT clear (S-07 and the
word-granular `wbuffer_hit_oh` at `wt_dcache_mem.sv:280` both remain live).

## WHY THIS CANNOT BE REDUCED TO AN INSTRUCTION SEQUENCE — measured, and it explains nine clean repros

The natural question for a hardware report is "which instructions?". For S-12 that question has a
definite answer and it is **not the one anyone wants**: the instructions are IDENTICAL between the
calls that are safe and the call that faults.

Disassembled from the running image, function entry `0x104788` to the faulting
`cincoffsetimm a4, a4, 0xb0` at `0x104814`:

* **36 instructions.**
* **ZERO branches and ZERO calls** — the path is entirely straight-line.
* **`iLevel` is never TESTED on that path.** It arrives in `a3` and is spilled once
  (`1047d4: sw a3, 0x0(a2)`); nothing reads it again before the fault.

So every invocation — the four safe ones and the wedging one — executes the same 36 branch-free
instructions, in the same order, with the same frame layout. **There is no instruction-level
difference to find.**

**That is the explanation for this folder's most puzzling row.** Nine directed reconstructions
reproduced the window — the four-instruction shape, the offset-for-offset 40-line window, the
intervening-store sweep, the adjacent-granule scalar, all six capability types, store-buffer
pressure, cache pressure — and every one came back clean. They were not missing an instruction. The
instructions were never the variable.

**What the trigger therefore is: MACHINE STATE**, produced by a prior nested codegen level, arriving
at an instruction sequence that is correct and unchanging. That is a statement about
microarchitecture — cache occupancy, write/store buffer contents, scoreboard state — and NOT about
decode or execution of any sequence.

**For the hardware side, the report should read:** *these 36 branch-free instructions are correct
and execute identically on every call; they fault only when a prior, nested where-codegen level has
run within the same prepare. Look for state that level leaves behind, not for an instruction
pattern.*

## SETTLED: it is NESTING, not repetition — two levels inside ONE prepare

The sharpest result of the delta-debug, and it retires the "the codegen path runs twice" framing
that every mechanism proposal has rested on.

**The invocation count was measured** (after repairing a probe that had latched its own counter and
reported `calls=1` forever): `WhereCodeOneLoopStart` runs **3 + plan depth** times, exactly. So a
2-level plan enters it 5 times and a 1-level plan 4 times.

**`dd6_twostmt` reaches 5 too** — two ONE-level queries in one file — and it does not wedge. Run as
a matched pair in the SAME boots, `dd6` first so it always gets a verdict and `dd2_join` last as the
positive control:

    arm            invocations   levels per prepare   result
    dd6_twostmt         5          1  (x2 statements)   0 wedges / 5 draws
    dd2_join            5          2                    6 wedges / 6 draws

The control fired in all four boots. Same call count, same domain binary, same boot, opposite
outcome. Fisher exact on 6/6 vs 0/5 = **0.002**.

**So cumulative invocation count is NOT the variable, and "it runs twice" is dead.** Running the
function twice across two separate prepares is SAFE. What distinguishes the wedging case is that the
second level's code is generated **while the first level's codegen state is still live** — the
levels NEST. That is a different thing from repetition and it is a much smaller target.

**Current statement of S-12, every clause measured:** it requires **two scan levels within a single
query's code generation**; it fires at **PREPARE time** (all tables empty, no rows ever processed);
it is not join-specific (`IN (SELECT ...)` qualifies, a flattened subquery does not); and it is not
a function of how many times the codegen path runs.

**The next delta is the `iLevel` parameter.** `sqlite3WhereCodeOneLoopStart` takes the level index,
so the natural next bisection is what it does differently when `iLevel > 0` — inside the function,
not in the SQL.

## CORRECTED, SAME NIGHT: it is PLAN DEPTH >= 2, not a join — and it happens at PREPARE time

**"It is a JOIN" is RETRACTED.** `dd5_inselect` — `WHERE a IN (SELECT a FROM t1)`, which is not a
join — WEDGED on its third draw. And the rung that motivated the join framing, `dd3_subq`, turns out
not to be a two-level query at all: **SQLite FLATTENS it**. Plans read rather than inferred:

    dd1_one       `--SCAN t1                                 1 level
    dd3_subq      `--SCAN t1                                 1 level   <- FLATTENED, identical to the control
    dd2_join      |--SCAN t1   `--SCAN y                      2 levels
    dd5_inselect  |--SCAN t1   `--LIST SUBQUERY `--SCAN t1    2 levels
    dd4_three     |--SCAN t1  |--SCAN y  `--SCAN z            3 levels

That is this folder's own standing rule biting the person who wrote it down: *same shape of SQL is
not same execution plan, and the plan must be READ rather than inferred.* `dd3_subq` was classified
as "two levels, no join" from the SQL text; it is one level, so it was never a counterexample.

**With plans read, the split is exact — on PLAN DEPTH:**

    plan depth   rungs                                    wedged / draws
    1 level      dd1_one, dd3_subq                        0 / 11
    >= 2 levels  dd2_join, dd4_three, dd5_inselect        4 / 8   (~50%)

P(0 wedges in 11 one-level draws | the ~54% rate) = 0.46^11 = **1.6e-4**. The positive control
(`dd2_join`) wedged in both of its draws, so the set is not a dead harness.

**So S-12 requires a query plan with at least two scan levels — by ANY route, join or list-subquery
— and fires while COMPILING it.** Both tables are empty throughout, so no rows are ever processed
and the whole effect lives in `sqlite3_prepare_v2`, which matches the fault site
(`sqlite3WhereCodeOneLoopStart` is a codegen function). The ~50% rate on qualifying plans matches
the long-recorded per-draw rate.

**What this does NOT say.** It does not identify the mechanism, and it does not distinguish "the
codegen path runs twice" from "the second level's codegen differs from the first". The next delta is
inside that function rather than in the SQL.

## SUPERSEDED — the join framing, kept for the reasoning


Everything else in this folder is an exclusion. This is the first statement of what S-12 *is*, and
it came from the SQLite-side delta-debug this folder recommended three times (`:1482`, `:1589`,
`:2188`) and that was never run until 2026-08-28.

**Two facts reframe the target before any board result.** `q_one` and `q_two` differ by exactly
`, t1 AS y`, and **both tables are EMPTY** — so no rows are ever processed and the entire difference
lives in `sqlite3_prepare_v2`. **S-12 is a fault while COMPILING the query, not while running it.**
That fits the fault site exactly: `sqlite3WhereCodeOneLoopStart` is a code-generation function.
Every mechanism hunted on 2026-08-27 assumed a data-path event.

**The ladder.** One build, six `.test` files staged together and driven as a runtime argument, two
domains per boot, so every rung executes from the BYTE-IDENTICAL domain binary — image-to-image
variance is removed from the comparison entirely. The control (`dd1_one`, one level) completed in
all five boots.

    rung          shape                                    result
    dd2_join      FROM t1, t1 AS y        2-table JOIN     *** WEDGED ***
    dd4_three     FROM t1, y, z           3-table JOIN     *** WEDGED ***
    dd3_subq      FROM (SELECT a FROM t1) 2 levels, NO join   completed
    dd5_inselect  WHERE a IN (SELECT ..)  2nd loop via IN     completed
    dd6_twostmt   two separate 1-level queries                completed

**Both JOINs wedged; neither non-join route to a second level did.** So the trigger is a
**multi-table FROM clause**, NOT "two levels of where-codegen" in general — a subquery reaches a
second level and survives, and so does `IN (SELECT ...)`.

**WEIGHT, stated rather than implied.** The two wedges are strong: a wedge is a wedge at any N. The
three clean rungs are WEAK at N=1 — at the ~54% per-draw rate, P(all three clean | all three can
wedge) = 0.46^3 = **0.10**. So this is roughly 90% confidence, not settled. **What would settle it:
three to four repeat draws on `dd3_subq` and `dd5_inselect`.** If they stay clean, join-specificity
is solid; if either wedges, the characterisation narrows to "a second where-codegen level" after all.

> **A PARSING TRAP THAT PRODUCED TWO WRONG TABLES BEFORE THIS ONE.** The UART truncates markers
> mid-line (`### TEST 1/2 END'` with no filename), and the driver ECHOES the whole command including
> its own `END` markers before running it. A parser that requires the filename after `END` reports
> completed arms as WEDGED; one that counts `END` markers anywhere counts the echo. **Count
> `SLT-SUMMARY records=` occurrences in the run-scoped section instead** — one per completed arm.
> A first pass also mixed logs from an aborted earlier attempt with live ones, and nearly produced a
> "the board degrades after a wedge" conclusion from stale files. Check log mtimes against the run.

## The fault

A pure-capability SQLite domain running a two-table join wedges with
**`mcause 25` (UNEXPECTED_OPERAND)** at a fixed instruction.

    sqlite3WhereCodeOneLoopStart+0x8c :   cincoffsetimm a4, a4, 0xb0

`mcause 25` from the FLU means the rs1 operand arrived with `cap_type == NOT_CAP`
(`capstone_flu_unit.anvil:57-59` — that is `CINCOFFSETIMM`'s only `UNEXPECTED_OPERAND` guard).

**The producer is the FLU, not the commit stage.** The two producers are distinguished
arithmetically, not by inference: the commit-stage path sets `tval == mepc`
(`commit_stage.sv:225-226`, `:604`), and the observation is `tval = 0` with
`mepc = 0x828f4ba0`. Excluded.

## The instruction window (`sqrt.dom`, fn at VA `0x104b14`)

    +0x38  cincoffsetimm a0, s0, -0x70      a0 = the slot
    +0x40  stc  a2, 0x0(a0)                 <== THE SUBJECT STORE
           ... 9 stores, 5 of them stc, none to word 0 of the subject granule ...
    +0x7c  cincoffsetimm a5, s0, -0x120
    +0x80  movc a4, zero                    <== a4 := {cursor 0, NOT_CAP}   ** READ THIS TWICE **
    +0x84  stc  a4, 0x0(a5)
    +0x88  ldc  a4, 0x0(a0)                 <== the reload, same a0
    +0x8c  cincoffsetimm a4, a4, 0xb0       <== THE FAULT, rs1 = a4

`a0` is never rewritten between +0x38 and +0x8c and there are **zero branches or calls** in that
window, so no callee can have touched the slot.

## What was measured

**1. Software stored a real pointer, and nothing overwrote it.**

Store watchpoint (`CSR 0x811`) armed at the slot's physical address `0x82b9f360`, group 9 enabled
(`CSR 0x810 = 0x200`), both written by a `csrw` in the **host process** — outside the domain, zero
instructions added to any domain image. Full dump in `evidence/group9-watchpoint-dump.txt`:

    [ 19]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    [ 20]  WATCHPOINT   PC=0x0000000828f4b54   DATA = 0x0000000082be4cf0
    Total: 21 entries

`0x828f4b54` is `+0x40`, the subject store. The payload is the stored **cursor** — confirmed two
independent ways: by measurement (`verif/tests/custom/capstone/watchpoint-cursor.S` stores a
capability with cursor `0x80003030` and the payload reads `0x80003030`), and by source
(`st_commit_data_o` → `.data` → `lsu_ctrl.data`, while metadata rides the disjoint `user` lane,
`store_unit.sv:377`).

**21 entries against a 256-entry ring means it never wrapped** — so that is *every* committed store
to that word for the entire run, not a tail window. There is no later entry with `DATA = 0`.

**2. The consumer received zero.** `tval = 0` at the trap, and `tval` is the delivered rs1 cursor
(`ex_stage.sv:489`, same `fu_data_i[0]` the raising guard reads at `:797`).

**3. `tval = 0` also excludes plain S-07 tag loss.** A post-S-06 untagged `ldc` returns a verbatim
copy, so a de-tagged capability would have arrived as `{0x82be4cf0, NOT_CAP}` and `tval` would read
`0x82be4cf0`. It reads `0`. **The cursor bits themselves are zero, not merely untagged.**

## What this excludes, and what it does NOT

**EXCLUDED: a software NULL pointer.** The last committed store to that word wrote a live in-domain
pointer, and no committed store wrote it afterwards. Both surviving explanations are hardware.

**NOT ESTABLISHED: which hardware.** Two hypotheses remain and the evidence does not separate them:

| | prediction |
|---|---|
| **A. Memory path** — the slot returns zero on reload | matches `tval = 0` |
| **B. Operand delivery** — memory and load both correct, the consumer is handed the previous `a4` | **also** matches `tval = 0` |

**B is not speculative.** `movc a4, zero` at +0x80 makes the prior architectural value of `a4`
*exactly* `{cursor 0, NOT_CAP}` — byte-for-byte what the FLU received. And **R-20 is a
board-reproduced instance of precisely this class on this core**: *"memory is right, the load is
right, and only the consumer's operand is wrong."* R-20's fix is in the resident bitstream
(`issue_read_operands.sv:568`) but was **empirical and register-specific** (x10), not a proven
invariant; the general capability-forward path at `issue_read_operands.sv:674-677` is
register-agnostic.

**A retraction, recorded because it shaped the trail:** this was written up as "the reload returned
zero". That measurement was never taken. The instrument brackets
`[store commit → operand delivery]`, which contains writeback and forwarding, not only memory.

## The signature that points at B, and which sat unread for days

**Every software probe added inside the domain makes the fault disappear** — probed builds complete
~4/4, the un-probed build wedged 5/5 across several bitstreams. That is a **scheduling** signature,
and it is exactly what R-20 showed: cured by one nop on the board, four in simulation. It had been
recorded as a nuisance ("we cannot instrument this") rather than as evidence.

The fault is also **sporadic** — the same un-probed binary returned normally on a later draw — so
absence in any single boot is not evidence.

## The discriminator, for whoever picks this up

**Cheapest, no board:** an RVFI simulation of the exact four-instruction shape
(`movc a4,zero; stc a4,0(a5); ldc a4,0(a0); cincoffsetimm a4,a4,0xb0`) using the R-20 template at
`R20-stc-rs1-cursor-forward-x10/sim/r20-stc-ld-x10.S`. The RVFI trace prints the load's **retired
value** and the consumer's **operand** side by side — which no board instrument can.

**On the board:** read the 16 bytes at the slot and its shadow tag byte over GDB at the wedge.
Memory intact and tagged ⇒ **B**, and the memory-path reading must be dropped. Zeros ⇒ **A**.

## Blind spots that travel with the group-9 result

State these with the measurement, because each makes an *absence* mean less than it looks:

* The watchpoint compare is **word-granular** (`cva6.sv:904-906`, `st_commit_paddr[PLEN-1:3]`), so
  stores to `G+8..G+15` are invisible. The claim is about **word 0** only — which is where the
  cursor lives, so it is the word that matters, but the distinction is real.
* **AMOs are excluded** at `commit_stage.sv:339`.
* **Domain-switch stores bypass the speculative queue** the tap reads.
* The tap is **upstream of the write buffer and D-cache**, so S-07/S-09/S-10-class corruption is
  invisible to it *by construction* — this instrument cannot see the very defects it sits next to.
* Group 9 carries **no tag bit** (`tracer.sv:237-239`), so the stored value's *validity* is
  unmeasured. It is a cursor, not "a valid capability".

## Reproduction

    binary   sqrt.dom, sha ee9a9a86ed12f06b, built by benchmarks/sqlite/build-sqlite-silicon.sh
             (UN-probed: any added in-domain probe removes the fault)
    input    benchmarks/sqlite/slt/q_two.test -- SELECT t1.a FROM t1, t1 AS y over an EMPTY table
             (q_one.test is the matched pair: identical but for the second table reference)
    driver   tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py
             SQLITE_STAGE_DOMS="/test-domains/sqbase.dom,/test-domains/sqrt.dom:--slt /test-domains/q_two.test"
             WEDGE_TRACER=1  (arming is compiled into the host: CAPSTONE_TRACE_ARM / CAPSTONE_TRACE_WP)

Both `.test` files declare `----\n0` against an empty table, so both report one query failure on
**every** platform including native — that is the test file's own authoring bug and not a result.
Confirmed byte-identical against the native oracle.

Run a known-good control first in the same boot: a boot whose control fails carries no verdict, and
the control is also what proved the ring flood is ordinary trap traffic rather than a wedge spin.

---

## The minimal repro does NOT reproduce — with the detector proven to fire

Board run, `caplifive_s07clear_84ed6eafb.bit`:

| rung | retval | oracle | verdict |
|---|---|---|---|
| `k800` (known-good control) | 4 | 4 | **OK** — the boot carries verdicts |
| `s12a2` (must-fail control) | `0xC12A2200` | `0xC12A2200` | **FIRED** |
| `s12a1` (the window) | `0xC12A1000` | — | **bad = 0** |
| `s12a3` (window + one nop) | `0xC12A3000` | — | **bad = 0** |

**Arm 2 matched its oracle exactly**, so the detector demonstrably reports a bad reload, and arm
1's zero is a measured zero rather than a silent one. `0xC12A2200` decodes as arm 2 with
`bad = 512 = S12_REPS` — every iteration's reload came back NOT_CAP, which is *correct*
architecture (a plain store clears the granule tag) and exists only to prove the instrument.

**So the instruction window, executed inside a real capability domain — after `capenter`, through
a cap table, on a monitor-carved stack — does not fault.** That closes the largest fidelity gap
the five bare-metal simulations left open, and closes it negatively.

**This is NOT an exoneration, and the kernel said so before the run.** Two deviations from the
production shape remain, both recorded in `src/s12_kernel.h` in advance:

1. the consumer is `lcc` selector 1 (**total**, does not raise) rather than the production
   `cincoffsetimm` (**raises**) — so the exact consuming instruction differs;
2. this is a tight loop, where the board's window sits at a deterministic depth in a long call
   chain, executed twice.

The pre-written next step therefore stands: **move the consumer back to `cincoffsetimm` and accept
wedging arms.** That trades the rate for one bit per boot, which is why it was not the first
version — but a clean arm 1 with a non-raising consumer cannot rule out a fault that only the
raising consumer exposes.

---

## THE MINIMAL REPRO REPRODUCES — 130 lines instead of all of SQLite

| rung | result | meaning |
|---|---|---|
| `k800` (known-good control) | RETURNED, `retval=4` | the boot carries verdicts |
| `s12a4sent` (SENTINEL) | RETURNED, `retval=0xC12A4E17` | **glue, cap-init, entry and return all work** |
| `s12a4d2` (the window) | **NO RETURN** | the silence is attributable to the **BODY** |
| `s12a1` (same body, non-raising consumer) | RETURNED, `bad=0` | — |
| `s12a2` (must-fail control) | RETURNED at its exact oracle | the detector is proven |

**Three draws of arm 4** — `s12a4`, `s12a4d1`, `s12a4d2`, distinct images with a byte-identical
window — fell silent **3/3**. The sentinel, built from the same source with the body skipped,
returned in 599 cycles.

### Why the sentinel is the load-bearing arm

`SHA5`-last on the `lpc` path is **ambiguous by construction**: that controller emits no
post-entry marker, so it cannot distinguish *"the body wedged"* from *"the domain never ran"* —
`locagg_kernel.h:34-36` says exactly this, and the tree records every lpc-hosted domain dying in
share #1 on 2026-08-06 regardless of content. An earlier version of this note read `SHA5`-last as
an **R-16 entry stall**; that was unsupported and is **retracted**. Two draws falling silent at the
same point was itself evidence against R-16, which is per-image and should have shaken loose on a
redraw.

The sentinel removes the ambiguity: it returns, so entry works.

### The matched pair, and the part that does not yet add up

Arm 1 and arm 4 differ by **exactly one thing** — the consuming instruction:

* arm 1 consumes with `lcc` selector 1 (total, non-raising) → **returns, `bad = 0`**
* arm 4 consumes with `cincoffsetimm a4,a4,0xb0` (raises on NOT_CAP), verified byte-identical to
  the board's faulting instruction (`5b 27 07 0b`) → **no return, 3/3**

**But `bad = 0` in arm 1 means the reload was never NOT_CAP there.** And `CINCOFFSETIMM` raises
*only* on `cap_type == NOT_CAP` — it has no bounds arm (`capstone_flu_unit.anvil:57-70`). So the
two results cannot both be about the reload's value unless **the consumer changes the timing**.

That is not a contradiction so much as the signature this whole investigation has been chasing:
the fault is scheduling-sensitive, and *every added instruction makes it disappear*. `lcc` and
`cincoffsetimm` are different instructions with different timing, so arm 1 is not a control for
arm 4's value — it is a different experiment. **Stated here rather than resolved**, because the
alternative readings are still open:

* arm 4 wedges at the `cincoffsetimm` on a NOT_CAP operand that arm 1's timing never produces;
* arm 4 wedges somewhere else in the body for a reason the non-raising build tolerates;
* arm 4 fails for a reason unrelated to the window.

**The trap cause and `mepc` have NOT been read for arm 4.** Until they are, "it reproduces the
S-12 signature" is unproven — what is proven is that a 130-line domain, with the production
consumer, does not return, while the same domain minus its body does.

## CONFIRMED: the 130-line repro produces the S-12 SIGNATURE, not merely a hang

The prediction was written down **before** the measurement (`git log` proves the ordering): if this
is S-12, `mepc` must read `DBAS + 0x41c` and `mcause` must be 25.

    sw=255 TRAP LOG {seen, mcause[6:0]} = 0x99   ->  seen=1, mcause = 0x19 = 25
    mepc (latched)                      = 0x819e041c
    the wedged arm's own DBAS           = 0x819E0000
    mepc - DBAS                         = 0x41c   ->  VA 0x1041c

`VA 0x1041c` is `cincoffsetimm a4, a4, 0xb0`, encoded `5b 27 07 0b` — **byte-identical to the
instruction that faults in SQLite**, same registers, same displacement.

**Same cause. Same instruction. Same encoding. In 130 lines instead of 2.2 MB of SQLite, and in
599 cycles instead of minutes.**

### What this changes

* **The reproducer is no longer SQLite.** Any future experiment — including the pending bitstream
  — can use `s12a4`, which is small, fast, and whose entire source is in `src/s12_kernel.h`.
* **The sentinel makes it attributable.** `s12a4sent` (same build, body skipped) returns cleanly,
  so this is the body faulting and not an entry stall.
* **It is not sporadic in the same way.** Three distinct draws with a byte-identical window
  wedged 3/3, where the SQLite repro fires roughly 2 in 3. A deterministic repro is a much better
  vehicle than a sporadic one.

### What it still does NOT establish

The **mechanism**. This is the same fault, reproduced small — it is not an explanation of it.
Every named mechanism remains excluded (software NULL, S-07 tag loss, R-20 forwarding class,
wrong-producer scoreboard selection, adjacent-granule scalar stores, write-buffer depth,
domain-switch `cnull`, load-syncer mispair), and **what the load returned is still the unmeasured
fact**. The pending bitstream is still what answers it — this just makes the run cheap.

Also still open, and recorded rather than resolved: arm 1 (same body, non-raising `lcc` consumer)
returns `bad = 0`, so *its* reload was never NOT_CAP, while `CINCOFFSETIMM` raises only on
NOT_CAP. Both cannot be about the reload's value unless the consumer changes the timing — which is
consistent with the scheduling sensitivity seen throughout, but is not itself established.

## The matched-pair vehicle: page-aligned, one slot address, still reproduces

The pending experiment arms **one** watchpoint address and runs both consumers against it — the
raising `cincoffsetimm` arm and the non-raising `lcc` arm — so the loads can be compared with the
consumer as the only variable. That requires both arms to place the slot at the **same** address,
and at 16-byte alignment they did not:

    lcc arm            slot VA = 0x11750
    cincoffsetimm arm  slot VA = 0x11790     <- 64 bytes apart

The bodies differ in size and shift everything after them. One armed address could not have served
both, and arming per-arm would reintroduce the **wrong-allocation failure class** the static array
exists to remove — the one that fired for real on an earlier arm and produced a clean-looking
empty record.

**Fixed by page-aligning `s12_frame`**; 4096 swamps the inter-arm drift. Verified, not assumed:

    s12p1 (lcc)            slot VA = 0x12690
    s12p4 (cincoffsetimm)  slot VA = 0x12690     <- ONE address serves both

### And the aligned build still faults, which had to be checked

This fault is layout-sensitive — every added instruction has made it vanish — and page-aligning
moved everything after the frame. So the aligned build could easily have stopped reproducing,
which would have made it useless as a vehicle:

    k800   RETURNED  retval=4            control valid
    s12p1  RETURNED  retval=0xC12A1000   matches its oracle, bad = 0
    s12p4  NO RETURN                     still wedges

### Constants for the run

    watchpoint paddr  = DBAS + 0x2690        (slot VA 0x12690, 16-byte aligned -> granule base)
    expected mepc     = DBAS + 0x41c         (consumer VA 0x1041c, bytes 5b 27 07 0b)
    expected mcause   = 25 (UNEXPECTED_OPERAND)
    arm 1 returns     = 0xC12A1000           arm 4 returns 0xC12A4000 or wedges

Both derived from the ELF plus DBAS, with **no dependence on call depth** and no scraping from a
previous wedge.

## The vehicle is complete: the domain arms its own watchpoint, and still faults

Page-aligning the frame made the **virtual** slot address identical across arms. The watchpoint
compares a **physical** address, and `DBAS` is per-**ARM**, not per-boot — measured: one boot's
arms took `0x82400000` and `0x82800000`. So a single `csrw 0x811` before the boot covers at most
one arm of the matched pair, and the uncovered arm returns an empty record, which the decision
table reads as *"the load was fine"*.

**The host cannot fix it** — `create_dom` returns only a domain id, and `DBAS` is printed by the
monitor, so userspace never sees it. **The domain does not need it.** Capabilities on this design
carry PHYSICAL addresses (the trap `mepc` is physical; the monitor's `BASE:` trace is physical),
so `lcc` **selector 2** — the cursor query (`capstone_dyn_unit.anvil`, `64'd2 => cap.cursor`) — on
a capability pointing at the slot returns the slot's physical address directly:

    lcc  rd, &s12_frame[0x700-0x70], 2     -> the slot's PHYSICAL address
    csrw 0x811, rd                          -> arm the granule filter
    csrr rd, 0x811                          -> readback into a sink: a failed arm is VISIBLE
    csrw 0x810, 0x200                       -> group 9, so the arming is PROVEN this boot

No `DBAS`, no host, no race with the driver, and **each arm arms its own address by
construction** — which is the safe form of per-arm arming, as distinct from scraping an address
from a previous wedge and reusing it.

### And it still faults, which had to be measured rather than assumed

The arming code shifts layout, and this fault vanishes whenever anything shifts. Page-alignment
survived, but that is evidence about a *different* change and does not transfer:

    k800   RETURNED  retval=4            control valid
    s12w1  RETURNED  retval=0xC12A1000   matches oracle, bad = 0
    s12w4  NO RETURN                     still wedges

Verified in the artifact too: both arms share slot VA `0x12690` and contain exactly one
`csrw 0x811`.

### Build recipe for the pending bitstream run

    arm 1 (non-raising, lcc):    -DS12_ARM=1 -DS12_SELF_ARM_WP=1     expect 0xC12A1000
    arm 4 (production consumer): -DS12_ARM=4 -DS12_SELF_ARM_WP=1     expect a WEDGE
    control:                      k800, which must return 4 first

### A WEDGE IS NOT ENOUGH: it must be at the RIGHT INSTRUCTION

`lcc` selector 2 is **not total** — only selector 1 is (`capstone_dyn_unit.anvil:195` excludes
`zimm != 1` from the NOT_CAP guard). Every other selector RAISES on a NOT_CAP operand, so the
**arming `lcc` can itself wedge the domain during setup**, before the body ever runs. From
outside, that is indistinguishable from the subject fault: both are "no return".

The two candidate sites in the self-arming build:

    arming lcc         VA 0x103bc   DBAS + 0x3bc   db 96 26 08   lcc a3, a3, 0x2
    subject consumer   VA 0x10484   DBAS + 0x484   5b 27 07 0b   cincoffsetimm a4, a4, 0xb0

**PRECONDITION: a wedge whose latched `mepc` is not `DBAS + 0x484` is NOT the subject fault.**
It is an arming failure or something else, and that arm is VOID.

**MEASURED, on the build that will actually run:**

    sw=255 TRAP LOG = 0x99          ->  seen=1, mcause = 25
    mepc            = 0x819e0484
    the arm's DBAS  = 0x819E0000
    mepc - DBAS     = 0x484          ->  THE SUBJECT CONSUMER. The arming lcc did not raise.

### The constant this replaces, and why that matters

An earlier note committed `expected mepc = DBAS + 0x41c`. **That was for the pre-self-arming
build**: adding the arming code shifted the consumer from `+0x41c` to `+0x484`. Using the stale
value as the precondition would have declared a perfectly good wedge VOID and thrown the real
result away — a constant that was true when written, silently false after a change, and
confidently wrong at the point of use. **Re-derive both site addresses from the artifact whenever
the build changes**; they are not properties of the bug.

Both arms carry group 9, including the one that returns cleanly — an empty LDC record means
*"the load was fine"* only where group 9 has fired at the subject store on that same boot.

## Group 9 FIRES in this vehicle — the precondition of the row table, measured

The pending decision table reads an empty LDC record as *"the load was fine"* **only where group 9
has fired at the subject store on the same boot**. Nobody had ever confirmed group 9 fires in this
vehicle at all — and discovering it does not, after a reflash, would have cost the whole cycle.

Measured on the CURRENT bitstream, arm 1 (the arm that returns), self-arming:

    256 entries, ONE distinct PC:   0x819e043c  ->  VA 0x1043c  =  stc a2, 0x0(a3)   the SUBJECT store
    256 entries, ONE distinct DATA: 0x819ff760                                        the stored cursor

Group 9 fires **only** on stores to the armed address, so its firing is itself the proof that the
domain's self-arm computed the correct physical address — no separate check needed, and none
possible to forget. A single PC and a single DATA value across all 256 entries is exactly what the
loop should produce, and rules out the record having been filled by unrelated traffic.

**So the row table is usable**: an empty LDC record will be distinguishable from a mis-armed
filter, which is the property the whole matched-pair reading rests on.

### A driver message that would have inverted this result

The driver announced *"NO `SQ: tracearm=` ... Arming UNMEASURED; treat any dump as
uninterpretable"* — because that marker is printed by the SQLite **host**, and a ladder domain
arms **itself** by design, so the marker is absent for a correct run. Taken at face value it would
have discarded a fully-armed 256-entry dump. Scoped to say the opposite for this path: group 9
firing *is* the proof, and an **empty** group 9 is what carries no verdict.

Same class as the stale `mepc` constant: text that was true where it was written, wrong where it
was read, and pointing toward "nothing to see" in both cases.

## Why the CURRENT bitstream cannot answer this for ANY workload — measured, not argued

It was worth one boot to ask whether the first-wins LDC recorder becomes usable on the small
vehicle. The reasoning was that SQLite issues thousands of legitimately-untagged loads before the
subject and consumes the slot, whereas `s12` has **14 static LDCs**. With the clear applied
immediately before the arm, the first untagged LDC had a real chance of *being* the subject.

**It is not.** Clear confirmed running before both arms; selftest firing, so a zero would be a
controlled negative:

    subject slot paddr        0x819e2690   -> recorder should read granule 0xe2690
    recorder actually read                     0x82280

**And `0x82280` is the same value the SQLite runs recorded — on a domain 6 MB away.** A record
that does not move when the workload moves is not the workload's record.

**The reason generalises, and that is the useful part.** The monitor's trap entry issues
`LDC(gp, sp, -16)` on **every timer tick**, so between the clear and any domain instruction at
least one tick almost certainly lands and takes the first-wins slot. That is independent of how
small the domain is, so **no workload can win this race** — shrinking the vehicle was the wrong
lever.

**This independently justifies the granule filter in the pending bitstream.** The filter is
precisely what makes the recorder immune to the monitor's traps: it is not a convenience, it is
the thing that makes the record attributable at all. Rolling-without-filter would still be
consumed by trap traffic; filtered-without-rolling would still be first-wins. Both halves are
load-bearing, which is worth knowing before anyone considers dropping one under synthesis
pressure.

---

# ~~THE STORED VALUE IS REVOKE-TYPED~~ — **RETRACTED 2026-08-25, and the retraction never reached this file until 2026-08-27**

> **DO NOT READ THE SECTION BELOW AS CURRENT.** It is kept because the reasoning from the encoding
> onwards is still correct and useful; only its premise is wrong.
>
> **The value is NONLIN, not REVOKE.** The board arm that produced `retval 0xC12A5200` packed the
> type into bits 8-11 of the same word as its NOT_CAP counter, so `0x200` had two readings — and
> the arm read a subject slot **it never wrote**, so the value it reported was never stored. The
> never-written slot selects NONLIN. Rebuilt *with* the store, the same arm returns `0xC12A5100`.
> That retraction was recorded in `state/current-next-step.md` on 2026-08-25 and **was never
> propagated into this folder**, which is why this file spent two days asserting REVOKE while the
> state doc asserted NONLIN.
>
> **Independently confirmed 2026-08-27**, so this does not rest on the retraction alone: a probe
> reading `cap_type` out of the register file at the exact `stc` pc measures **NONLIN 16/16**
> (qj2 7/7, q_one 4/4, q_two 5/5), with a positive control showing the same path does emit
> clear-set names. And structurally, the slot is loaded **three times per call**, so a clear-set
> type here would wedge one-level plans too — and one-level has never wedged in 11 draws.
>
> **Consequence: the LDC move-clear does NOT fire at the S-12 fault site.** Everything below that
> depends on it firing is void as an account of S-12. The clear-set mechanisms are separately
> refuted on their own evidence (the six-type sweep and the linear-clear matched pair), so those
> refutations stand regardless.
>
> **The lesson, which is why this header is this long:** a retraction recorded in the state doc but
> not in the artifact folder is not a retraction. The folder is what gets read, and it kept a
> withdrawn claim alive in a document written to be handed to the hardware side.

## Superseded original section

# THE STORED VALUE IS **REVOKE**-TYPED — so the LDC MOVE-CLEAR FIRES, and every earlier sim was blind to it

Measured on the board with a type-probe arm: `retval 0xC12A5200` → `lcc` selector 1 returned **2**,
and that selector reports `cap_type - 1` (`capstone_dyn_unit.anvil`, `64'd1 => cap.metadata.cap_type - 3'd1`),
so the raw type is **3 = `CAP_TYPE_REVOKE`**.

**`REVOKE` is in the LDC move-clear set.** `load_unit.sv:225-226` fires the clear for
`{LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET}` with write permission. So on silicon, **every
iteration's reload also WRITES the source granule** — and the payload it writes is
`store_unit.sv:462-469`:

    .data_i  (load_unit_clear_i ? '0   : st_data_q)      cursor   = 0
    .user_i  (load_unit_clear_i ? '0   : st_user_q)      metadata = 0
    .ctag_i  (load_unit_clear_i ? 1'b0 : st_ctag_q)      tag      = 0

which is **bit-for-bit `create_cnull()`** — and therefore bit-for-bit the operand observed at the
fault, `tval = 0` included. This is the first candidate that produces the exact value as a
*designed payload* rather than as corruption.

## Why this invalidates five earlier "clean" simulation results

**Every one of the five directed sims set `CAP_TYPE_NONLIN`**, which `load_unit.sv:225-226`
explicitly excludes. The clear **never fired in any of them**. Their clean results were not
evidence about this mechanism — they were testing a configuration in which it is structurally
absent, and I read them as exclusions.

That is the same failure as the rest of this investigation, in its most expensive form yet: not a
broken instrument, but a **correct instrument pointed at the wrong configuration**, five times,
with three of them carrying proven-firing positive controls that made them look rigorous.

## What the clear implies about the traffic

With the clear firing, each iteration issues **two** writes to the subject granule — the `stc` and
the reload's clear — into a write buffer that is 8 entries deep and which this kernel already
overflows (≥16 dirty words per iteration). If a merge ever lands the clear *after* the following
iteration's `stc`, the granule reads back as the clear payload: a null.

## Status: NOT established, and one attempt already failed

A directed sim using `CAP_TYPE_LIN` **failed for an unrelated reason** — `MOVC` of a linear-class
capability nulls its source, so copying it around violates linearity and `CINCOFFSET` raised
`UNEXPECTED_OPERAND` on a legitimately-nulled operand. Correct architecture, invalid test.

**And the simple version of this account is already excluded by measurement:** if the domain's own
`v` were being nulled by move semantics, the store watchpoint would have recorded it. It recorded
**256 stores, one PC, one non-zero DATA value** — `v` is stable across every iteration. So whatever
happens, it is not the program nulling its own capability.

The next test must respect linearity: a fresh clear-class capability per iteration, or `REVOKE`
specifically (which is what the board actually has), rather than copying one around.

## MEASURED: the LDC move-clear fires on EVERY iteration (512/512)

Arm 6 reloads the same slot **twice** and asks whether the second reload comes back NOT_CAP. If
the clear fires, the first reload zeroed the granule and the second must find nothing.

    control k800   retval 4, oracle 4      OK -- the boot carries verdicts
    s12c           retval 0xC12A6200       arm 6, count = 512 of 512

**Every reload zeroes its own source granule.** That is correct architecture — a move — and it is
now measured rather than inferred from the type. Combined with the type probe (REVOKE, in the
clear set at `load_unit.sv:225-226`) and the clear's payload (`store_unit.sv:462-469`: cursor 0,
metadata 0, tag 0 = `create_cnull()`), the picture on silicon is:

    per iteration:  stc  writes the capability to the slot
                    ...9 intervening stores to other granules...
                    ldc  reads it back AND writes create_cnull into the SAME granule

So **two writes per iteration target the subject granule, from two different agents**, into a
write buffer that is 8 entries deep and that this kernel drives past capacity every iteration
(>=16 dirty words). The buffer's own comments (`wt_dcache_wbuffer.sv:611-619`) document a
merge-ordering residual of exactly this class, for a different instruction pair.

### The hypothesis this makes concrete — and it is a HYPOTHESIS, not a measurement

If iteration N's **clear** ever merges into the buffer *after* iteration N+1's **store**, then
iteration N+1's reload reads the clear's payload instead of its own store's value — and the
payload is `{cursor 0, NOT_CAP}`, which is the operand observed at the fault, `tval = 0` included.

**What is measured:** the clear fires every iteration; the value is REVOKE-typed; the clear's
payload is bit-for-bit `create_cnull`; the store writes a stable non-zero value 256/256.
**What is NOT measured:** that the two writes ever land out of order. Nobody has observed a
reordering; this is a mechanism that *would* produce the observed value, on hardware where both
writes demonstrably exist.

### Why this reopens the memory-path arm

The delivery account (a stale FLU operand read before the load lands) and this one both predict
`{cursor 0, NOT_CAP}`. They differ in **where** the null comes from: the register file versus the
granule. RTL's pending recorder distinguishes them directly — it reports whether the LOAD returned
untagged. That is now a fork between two *specific* mechanisms rather than between two vague
halves of the pipeline.

Note also what arm 1 already says: with the non-raising consumer, the FIRST reload was never
NOT_CAP across 512 iterations. So on that build no reordering occurred — consistent with the
DYN-serialisation difference, and consistent with the reordering being timing-gated.

## The GRANULE row is excluded — structurally, and without a boot

The reordering hypothesis above needs iteration N's clear to merge into the write buffer *after*
iteration N+1's store. **There is no path for that**, and the reason is not the commit gate alone:

    store_unit.sv:449    .valid_i (store_buffer_valid || load_unit_clear_i)
    store_buffer.sv      monotonic +1 pointers on BOTH queues -- a strict FIFO
    load_unit.sv:707-712 valid_o = 0 while the clear has not been accepted

**The clear uses the SAME store-buffer port as ordinary stores**, and that buffer is a strict
FIFO. The gate guarantees the clear is *enqueued* before the LDC commits; in-order commit puts the
next iteration's `stc` later; one FIFO then delivers them to the write buffer **in program order**.
Same granule and same word means they merge into one entry, where `ctag` is last-writer-wins and
the merge order *is* arrival order. The different-word case is covered by the `gran_hazard` stall
for the same reason, and if the clear has already drained there is no merge at all — but it still
landed first.

All three premises verified at the resident revision rather than taken from the report.

**So the fork narrows to one named mechanism**: the null comes from the **register file**, not the
granule — a stale FLU operand read before the load's writeback lands.

### What that does to the pending bitstream's reading

RTL's recorder reports whether the **load** returned untagged. Under the surviving account it
should come back **EMPTY on the faulting arm** — and that is now a **PREDICTION with a named
mechanism behind it**, not a default.

That distinction matters. An expected empty is the result nobody scrutinises; an empty that one
mechanism predicts while the competing mechanism — which predicted the opposite — has just been
excluded on structure, is a result. **A NON-empty record would mean the structural argument above
is wrong**, and that is the more interesting outcome of the two.

And it raises the stakes on the group-9 arming proof: with empty as the predicted answer, group 9
firing is the only thing separating *"empty because the load was fine"* from *"empty because the
filter never armed"*.

## And the REGISTER-FILE row takes a hit too: the RAW check HELD, with the window proven created

`s12-flu-raw.S` builds the window the surviving account needs — a strided, cache-missing `ldc`
followed **immediately** by an FLU consumer of its destination, every granule pre-seeded with a
real capability so a NOT_CAP operand can only come from the pipeline:

    RAW-DBG: ALIVE
    flu-issues = 131    ldc-pending-cycles = 82    HAZARDS = 0

**The condition was created** — 82 cycles with an LDC outstanding, 131 FLU issues alongside — and
**not once** did an FLU op get acked for issue while an LDC writing its `rs1` was still pending.
The generic RAW machinery stalled it every time.

This is a real negative rather than the earlier void: the previous run reported
`ldc-pending-cycles = 0` because its loads all hit, so nothing was tested. The totals are what make
this zero admissible.

### So both named mechanisms now have negative evidence

    granule (clear reordered after the next store)   EXCLUDED structurally -- one FIFO, program order
    register file (stale FLU operand)                RAW check HELD in sim, window proven created

**The fault is not in doubt** — it is deterministic on silicon, at a known instruction, with a
known cause. What is in doubt is every mechanism either lane has named for it.

**Caveat that keeps the register-file row alive, and it is not a small one:** this is simulation.
The RAW check holding here does not prove it holds on silicon, and the fault has never reproduced
in simulation at all. A hazard that is correct in RTL and marginal after synthesis would look
exactly like this. So the row is **weakened, not excluded** — and I am not recording it as an
exclusion for that reason.

### What this does to the pending bitstream

It makes it **more** valuable, not less. Both candidate mechanisms now predict outcomes that
disagree with a measurement, so the recorder is no longer confirming a favoured story — it is the
only thing that can say where the null comes from **without** a hypothesis in hand. An empty record
is no longer "as predicted": it is one of two informative answers, and the non-empty one would
resurrect the granule row against a structural argument, which would be the more interesting result.

## IT FAULTS ON **ONE** ITERATION — accumulation is dead, and so is the last granule story

Sweeping the loop count with the production consumer, ascending, control first:

    k800    RETURNED  retval=4
    s12r1   NO RETURN                <== S12_REPS = 1

Confirmed the same fault, not a different one — the discipline that the `SHA5` retraction bought:

    TRAP LOG 0x99            ->  seen = 1, mcause = 25
    mepc                     =   0x819e0460
    the arm's own DBAS       =   0x819E0000
    mepc - DBAS              =   0x460      -> the SUBJECT consumer
    (the arming lcc sits at   0x39c, so the two are distinguishable and it was not that)

### What one iteration excludes

* **Accumulation across iterations** — there are no other iterations. Any account needing state
  to build up over repetitions is dead: rev-node consumption, cache-set rotation, write-buffer
  phase advancing over a loop.
* **The move-clear as a source of the null.** With `REPS = 1` there is no *previous* iteration
  whose clear could be read back, and the reload's own clear happens at the reload — it cannot
  precede it. The granule story was already excluded structurally by the FIFO argument; this
  closes it independently, from the other end.
* **Any "low per-iteration probability" reading of the earlier threshold sweep.** That caveat was
  raised before this ran and it was the right one to raise — but a single iteration faulting
  deterministically leaves no room for it.

### What survives

One pass: store a capability to the slot, nine intervening stores, `movc a4, zero`, reload,
consume. The consumer receives `{cursor 0, NOT_CAP}` — and `movc a4, zero`, **in this same pass**,
is the only thing that put exactly that value anywhere.

So the **stale-operand account is now the only named mechanism that works at `REPS = 1`.** It also
remains the one the RAW-hazard simulation says should not happen — `HAZARDS = 0` with the window
proven created. Those two facts are in direct tension, and the tension is the finding: either the
RTL check has a gap the directed test did not construct, or it is marginal after synthesis in a
way simulation cannot show.

### The repro is now smaller again

`-DS12_ARM=4 -DS12_SELF_ARM_WP=1 -DS12_REPS=1` — no loop, roughly twenty instructions of body,
faulting deterministically at a known VA with a known cause. That is the artifact to hand anyone,
and it is what the pending bitstream should be pointed at.

---

# CORRECTION: the subject slot is a COMPILER stack spill, not the kernel's static array

The watchpoint address recorded above — `DBAS + 0x2690`, the slot inside `s12_frame` — is **the
wrong granule**. The fault is not on the array the kernel stores to; it is on a spill the compiler
inserts afterwards.

    0x10450  ldc a4, 0x0(a0)              load from the kernel's frame slot
    0x10454  cincoffsetimm a3, s0, -0xc0  a3 = a COMPILER stack slot
    0x10458  stc a4, 0x0(a3)              <- the REAL spill
    0x1045c  ldc a4, 0x0(a3)              <- the REAL reload
    0x10460  cincoffsetimm a4, a4, 0xb0   <- THE FAULT

**Corrected address, and it is corroborated independently rather than merely recomputed:**

    s0 at the wedge   = 0x819ff670
    subject slot      = s0 - 0xc0 = 0x819FF5B0     (== sp; 16-byte aligned, so a granule base)
    low 20 bits       = 0xFF5B0
    s07 STC recorder  = 0xFF5B0                    <- EXACT MATCH

**The STC recorder had been reporting the subject spill all along and I read past it**, because I
was checking it against the address I expected rather than asking what it was pointing at.

## How this happened, and why it is worse than a slip

For SQLite I had this right — the pair was `stc a2, 0x0(a0)` / `ldc a4, 0x0(a0)` with
`a0 = s0 - 0x70`, plainly a stack slot. When I built the minimal repro I **assumed my own static
array would be the subject**, because I had written the store myself. The compiler still spills
`a4` to its own frame slot, and the fault is on reloading *that*. The kernel's `*slot = v` is a
different, earlier store.

Everything downstream inherited the assumption: the page-alignment work, the `lcc`-selector-2
self-arming, the "one address serves both arms" property, and the constants handed over for the
pending bitstream run. **The repro itself is unaffected** — it faults deterministically at a known
VA with a known cause — but every *measurement aimed at a granule* was aimed at the wrong one.

**And it failed in the expensive direction:** an armed address the fault never touches yields an
empty record, and this folder's own decision table reads empty as *"the load was fine, the fault is
in delivery"*. It would have produced a confident wrong answer rather than a null one.

## Consequences for the earlier results

* **The group-9 measurement (256 stores, one PC, one DATA)** was real but was watching the
  kernel's array store, **not** the subject spill. It still proves the self-arm mechanism works and
  that group 9 fires; it says nothing about the subject granule.
* **The `a4 = 0` reading needs re-examination**, since the reload feeding the fault comes from `a3`
  (the stack slot), not `a0`. The precondition held and the value is real; what it means is not yet
  settled and is deliberately not being claimed here.
* **The corrected address is dynamic** — `s0 - 0xc0` — which reintroduces the hazard the static
  array was chosen to remove. `0xc0` is a fixed offset per build and can be re-derived from the
  artifact; only `s0` needs measuring.

---

# THE NULL PREDATES THE SPILL: the subject is one instruction EARLIER

With the watchpoint finally armed at the real subject slot — computed at runtime from `s0`, and
cross-checked against hardware — group 9 fired **exactly once**, at the spill, writing **zero**.

    0x1045c  ldc a4, 0x0(a0)              <- the null enters HERE
    0x10460  cincoffsetimm a3, s0, -0xb0
    0x10464  stc a4, 0x0(a3)              <- THE SPILL.  group 9: 1 entry, DATA = 0x0
    0x10468  ldc a4, 0x0(a3)              <- reload reads the zero back
    0x1046c  cincoffsetimm a4, a4, 0xb0   <- FAULT (mepc - DBAS = 0x46c, mcause 25)

`a4` was **already zero when the spill executed**. So the spill/reload pair is faithful — it stores
a null and returns the same null — and the whole investigation has been instrumenting **one
instruction too late**. The subject is `ldc a4, 0x0(a0)` at `0x1045c`, or whatever put a null in
the memory it reads.

**The cross-check is what makes this readable:** the s07 STC recorder independently reported
granule `0xFF5C0`, matching the armed slot `s0 - 0xb0`. Hardware agreed the right address was
armed, which is exactly the check that caught the previous wrong granule.

## What this retires

* **The spill/reload pair as the site.** It is doing its job correctly. Every measurement aimed at
  it — including the group-9 arming work and the corrected address — was aimed one step past the
  defect.
* **Both remaining accounts, as stated.** The stale-operand story was about the consumer reading
  around the *reload*; the memory story was about the *reload's* granule. Neither survives the null
  already being in the register before either happens.

## How the address problem was finally solved, since it will recur

The subject slot is a compiler spill at `s0 - OFF`, and **the compiler chooses OFF**: measured
`0xc0`, then `0xa0`, then `0xb0` across three builds, because adding the arming code moves the
frame. A hardcoded physical address is therefore stale the moment anything is rebuilt — and arming
it *is* a rebuild. That is circular.

The fix is to compute it at runtime: `s0` is a capability register, so the domain reads it, offsets
it, and queries its cursor for the physical address. Only `OFF` remains a build constant, and being
an immediate it does not change code size — verified by convergence, `0xb0` on two successive
builds. No DBAS, no host, no measured address.

## Recurrence worth noting

The `mepc` predicted before this run (`DBAS + 0x458`) was taken from the **previous** build's
artifact, not the one that ran (`0x46c`). The same stale-constant mistake as `0x41c`, one day and
one explicit fix later. The precondition still held — `0x46c` **is** the consumer in the build that
ran, confirmed by re-deriving from the correct artifact — but it was confirmed after the fact
rather than predicted, and that is the weaker form.

## Pre-registered reading of the one-level-up measurement (written BEFORE the dump)

The group-9 watchpoint is armed one level up, on the store at `s0 - 0xa0`:

```
0x10440  ldc  a1, 0(a0)     the REAL slot read
0x10448  stc  a1, 0(a0)  <- ARMED HERE
0x1045c  ldc  a4, 0(a0)     reads it back -> ZERO
0x1046c  cincoffsetimm a4   <- FAULTS (mcause 25, UNEXPECTED_OPERAND)
```

**The entry COUNT discriminates, not just the payload.** The armed slot holds a live REVOKE
capability, and the reload at `0x1045c` is exactly the LDC shape that fires the move-clear
(`load_unit.sv:225-226`), whose payload — cursor 0, metadata 0, ctag 0 (`store_unit.sv:462-469`)
— is written back to the **same granule** through the shared store port (`store_unit.sv:449`).
So the move-clear can post a *second* group-9 entry, at the same address, reading zero. Three
outcomes, all registered in advance:

| group 9 | reading |
|---|---|
| 1 entry, payload ≠ 0 | the store wrote a real capability and the **load** returned NOT_CAP — the move-clear never fired because the load never saw a cap type. A genuine load-path defect, and the repro vehicle is sound. |
| 1 entry, payload = 0 | `a1` was **already null** upstream; the subject moves up again to the `ldc` at `0x10440`. |
| **2 entries (≠ 0, then 0)** | the store wrote the cap **and** the load *saw* it (move-clear fired) — yet the consumer still faulted. The null then enters **after** the load, in writeback/forwarding, not in memory and not in the load path. This is the most informative signature available and must not be read as "the store wrote zero", nor discarded as noise. |
| empty | uninterpretable — says nothing about the subject. See the instrument caveat below. |

**If it comes back empty, suspect the instrument before the subject.** This is the first boot
combining `WEDGE_S07_CLEAR_PERARM=1` with `WEDGE_TRACER=1`, and the per-arm clear used to restore
the switches to `0` rather than to the tracer's resting value — parking `sw[2]`, the ring-mode
input, low for every arm after the first clear. Read the driver's own `switch_state` events for
the value in force while arm 2 ran before drawing any conclusion from an empty dump. (Fixed in
the driver as of this run's revision, but the running process loaded the older copy.)

## Two boots lost to a fail-open diagnostic — recorded because the class recurs

Two consecutive boots reported the **control** (`k800`, which returns in ~2 s) as NO-RETURN, which
looks like a board or firmware failure. It was neither. Both logs carry the same line:

```
[s07] early halt control failed (ActionTimeout) -- no verdict, and the run continues
```

An optional pre-run diagnostic halted the core over GDB, timed out, and left it **halted**; the
driver logged "the run continues" and carried on, so every stage then timed out at
`SQLITE_STAGE_TIMEOUT` and read exactly like a wedge. The discriminator that settled it: the
control failed **identically** twice. Its own flake rate is ~1 in 5 but that flake is an entry
stall, which varies — two identical control failures are a variable we introduced, and the diff
of the failing driver log against the last passing one lands on the line above.

Two changes followed: the early-halt control is now **fail-closed** (it forces a resume, then
proves the shell answers, and aborts the boot loudly if it does not), and the discriminator is in
the `board-run` skill. The fail-closed path is **unexercised** — this run disables the diagnostic
outright, since it already answered its question in an earlier boot — so by this project's own
standard it is an unproven gate, not a fixed one, until it has been negative-tested.

## RETRACTION — every arm-4 measurement in this folder is VOID

**Arm 4 read a slot it never wrote.** In the kernel the subject store lives inside

```c
#if   S12_ARM == 0                    *slot = v;
#elif S12_ARM == 1 || S12_ARM == 3    *slot = v;   /* + the 9 intervening stores */
#elif S12_ARM == 2                    *slot = v;   /* + the scalar scribble */
#endif
```

and **arm 4 matches no branch of it**, so no store to the slot is emitted. The reload
`void *back = *slot;` then reads `s12_frame + 0x690`, which the image never writes. **The
conclusion rests on "never written", not on "zero"**: a granule no capability store ever targets
has its tag clear, so the reload is NOT_CAP whatever the bytes hold. An earlier draft said
"zero-initialised BSS", which is not established — `llvm-readelf -l` shows `.bss` as `NOBITS`
inside a `NULL` program header rather than a `LOAD`, and nothing in the startup assembly or the
build script zeroes it. The observed zeros (`tval` 0, group-9 payload 0) are an empirical fact
about these runs, not a loader guarantee. `back` is therefore a genuine NOT_CAP, and the `cincoffsetimm` consumer raising `mcause 25` is
**correct ISA behaviour** — the hardware doing exactly what the spec says.

Verified with the preprocessor, not by reading:

```
clang -E -P -DS12_ARM=N -DS12_REPS=1 -DS12_DRAW=0 -x c s12_kernel.h
  ARM=1  ->  `*slot = v;`  then  `void *back = *slot;`
  ARM=4  ->  no `*slot = v` anywhere;  `void *back = *slot;` reads zeroed BSS
```

and corroborated in the artifact. **State that check over the whole loop body, not "between the
two `ldc`s"** — those are adjacent instructions four bytes apart, so a check for a store between
them spans zero instructions and cannot fire. That is this project's own "a clean result is not
evidence until the check is known to fire" class, committed *inside* the retraction about that
class. The check that does fire: the loop body of `s12g.dom` is `0x1042c`–`0x104a8` and holds
exactly four `stc`, all to the stack frame — `0x10434`→`s0-0x90`, `0x10448`→`s0-0xa0`, `0x10464`
and `0x10470`→`s0-0xb0` — and **none through `a0`**, the slot pointer loaded at `0x1043c`.

**What this voids.** Everything measured on arm 4 — the group-9 zero at the spill `s0-0xb0`, the
group-9 zero one level up at `s0-0xa0`, `a4 = 0x0` at the wedge, and the whole "the null predates
the store, move the subject up one level" chain of reasoning. Each of those was measuring
uninitialised memory. The backwards walk kept finding zeros because there was never a capability
in that slot to lose. **Arms 5 and 6 never write the slot either**, but they are not equally
affected, and the difference matters:

- **Arm 5's type report does not merely survive — it INVERTS, from REVOKE to NONLIN.** Arm 5
  encodes the type into bits 8–11 *of `bad`* and then also runs the counting consumer, because
  the `#if S12_ARM != 4 && S12_ARM != 6` at the reload site admits arm 5. So the type field and
  the NOT_CAP counter share one word, and the observed `retval=3240776192` = `0xC12A5200` has
  **two readings**, confirmed by simulating the loop exactly:

  | reading | yields |
  |---|---|
  | type = 2 (REVOKE), reload clean every iteration | `0x200` |
  | type = 1 (NONLIN), reload NOT_CAP every iteration, REPS=512 | `0x200` |

  Arm 5 never wrote the slot, so its reload *was* NOT_CAP every iteration — which selects the
  second row. Corroborated twice: arm 6 counted `bad = 512` on the same address (512/512
  NOT_CAP, direct measurement), and rebuilding arm 5 **with** the slot written makes the
  encoding unambiguous (`bad = t<<8`, zero low byte) — QEMU then returns `3240775936` =
  `0xC12A5100` = **type 1, NONLIN**.

  **This reverses a downstream conclusion rather than voiding it.** The kernel's own text
  (`s12_kernel.h:243-244`) says that if the type is NONLIN "the clear NEVER fires and that whole
  mechanism is dead". So the LDC move-clear account is dead, and the five earlier "clean"
  simulation results that used NONLIN were in the **right** configuration all along — their
  exclusions are **reinstated**, not invalidated. Any section of this file arguing that those
  five sims used the wrong type is itself retracted.

  Still open: QEMU's type model need not equal silicon's, so a board run of the rebuilt arm 5
  is queued to confirm NONLIN on the FPGA. Two independent lines already agree.
- **Arm 6's conclusion is VOID.** Its design — "reload the same slot twice; if the move-clear
  fires the second must come back null" — is vacuous when both reloads read zeroed BSS. Both
  come back null regardless of whether the clear fires, so the arm cannot distinguish its two
  hypotheses. Note arm 6's own comment cites arm 5's REVOKE result as its premise; that premise
  is sound, the experiment built on it is not.

**What survives.** Arms 0/1/2/3 do write the slot, so their results stand. The original SQLite
S-12 observation stands — it is a real workload fault and owes nothing to this repro.

**Why no existing check caught it.** The artifact was correct. The fault was the right `mcause`
at the right `mepc`, predicted from the disassembly before the run. The repro was deterministic
and reproduced at `REPS=1`. Every one of those reads as a strong result. The only thing that
distinguished it was the **absence** of a store, and nothing was looking for an absence. This is
the "directed tests that come back clean without ever creating the triggering condition" class
from CLAUDE.md, in its inverted form — a test that *fails* without ever creating the condition,
which is harder to spot because failure is what you were hoping for.

**The fix, and the guard.** Arm 4 now takes arm 1's branch. Each writing branch defines
`S12_SLOT_WRITTEN`, and the reload site carries `#if !defined(S12_SLOT_WRITTEN) #error ...` — so
an arm that reads the slot without writing it **fails the build** rather than producing a
beautiful void result. Negative-tested in both directions:

| arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|---|---|
| slot store emitted | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| guard fires | · | · | · | · | · | **✓** | **✓** | **✓** |

## Silicon confirmation: NONLIN, and the LDC move-clear does not fire

One boot, control returned, both arms run with the subject store present for the first time.

| arm | what it measures | retval | reading |
|---|---|---|---|
| 5 | cap type of `v`, bits 8–11 | `0xC12A5100` | **type 1 = NONLIN** |
| 6 | reload the same slot twice, count NOT_CAP on the second | `0xC12A6000` bad=**0** | **the move-clear does not fire** |

Arm 5's **low byte is 0**, which is the internal consistency check: it means the counting consumer
never saw a NOT_CAP reload, i.e. the hoisted store actually landed and the encoding is unambiguous
again. That is what makes this reading trustworthy where the earlier `0x200` was not.

Silicon agrees with QEMU (`0xC12A5100` both). So the REVOKE→NONLIN inversion is confirmed on the
FPGA, not merely in emulation, and arm 6 — which could not distinguish its two hypotheses while the
slot went unwritten — now measures directly that loading a NONLIN-stored slot leaves it intact.
Consistent with `load_unit.sv:225-226`, where NONLIN is absent from the clear set.

**What this does NOT establish.** Both results are about **the repro's `v`**. They say the
move-clear does not fire *here*. They bear on S-12 only if SQLite's stored value at the fault site
is also NONLIN, and **that is still unmeasured** — it is the discriminating unknown, and the reason
the kernel's "any tagged capability; its identity is irrelevant" is the weakest assumption in this
folder. If SQLite's value is in the clear set {LINEAR, REVOKE, UNINIT, SEALED, SEALEDRET}, this
repro has never exercised the mechanism it was built to test, and an arm with a matching-type `v`
is the first variant that could reproduce.

**That escape hatch is now closed by measurement, not by the type argument below.** The sections
that follow establish, on two independent legs, that SQLite's value here is NONLIN and therefore
cannot clear at all. That is a claim about *production*. It leaves open a claim about *the repro*:
that the window is clean only because the repro's type was wrong. `s12-value-type-sweep.S`
(`capstone-ariane 7fb91b5c7`) settles it by running the identical window six times, varying only
the subject store's value type, with a separate always-NONLIN filler for the intervening stores so
a linear-family subject cannot perturb them as a second difference:

    arm   type          load returned          source granule word 0
    A0    NONLIN        capability, type 2     INTACT (0x80003000)   <- clear correctly did not fire
    A1    LINEAR        capability, type 1     ZEROED                <- clear fired
    A3    REVOKE        capability, type 3     ZEROED
    A4    UNINIT        capability, type 4     ZEROED
    A5    SEALED        capability, type 5     ZEROED
    A6    SEALEDRET     capability, type 6     ZEROED
    AF    positive ctl  NOT_CAP, value 0       (slot scribbled scalar-wise before the reload)

**Each clean arm carries its own proof the condition held**, which is what makes it admissible: the
clear demonstrably fired in all five clear-set arms and demonstrably did not in the NONLIN control,
so no arm is void. **And the detector demonstrably produces the failing reading** — the positive
control prints the board's exact signature, the NOT_CAP form with value 0. 1744 cycles against a
2000000 timeout, so the pass is a real pass rather than a hang reported as one.

In all five arms where the clear fired, **the load still returned a correct, tagged capability of
the right type.** So the load does not observe its own side effect, and "the reload races the clear
it triggers" is refuted at the real window for every type that can trigger it. The window is clean
for every type it could have been, not merely for the one it had.

**A related mechanism this rules out for SQLite but which is worth knowing about generally.** If a
clear-set capability is loaded out of a slot and then loaded from that slot AGAIN without an
intervening store, the second load reads the zeroed granule and yields cursor 0 / NOT_CAP — S-12's
signature exactly. That cannot be what happens here, because NONLIN does not clear. It is recorded
because **QEMU cannot see that class of bug at all**: `csldc`
(`capstone-qemu target/riscv/insn_trans/trans_capstone.c.inc:146-169`) loads two words and sets the
register, with no source clear, while the RTL clears at `load_unit.sv:225-230`. Any software that
double-loads a linear-family capability runs clean under QEMU and dies on silicon.

## The discriminating unknown is answered: SQLite's value is NONLIN too

The open question above — is SQLite's stored value at the fault site in the LDC clear set? — is
**no**. The derivation, with each link re-verified against the primary source rather than taken on
report:

1. The monitor hands the domain `dom_data` as **LINEAR** (`sbi_capstone.c:302-305`; `split_out_cap`
   with `linear=1` *asserts* linearity and only the `!linear` arm calls `__delin`,
   `sbi_capstone.c:275-281`). It becomes the domain's `sp`.
2. The entry glue **delinearizes `sp` exactly once**: `delin(sp)` at
   `start-gp-captable-interp.S:268`. **Verified this is the shipping path, not a diagnostic one** —
   the line sits in the `#else` arm, and `INTERP_FAKE_COUNT` (which selects the other arm) is never
   defined for the SQLite build; it appears only for one BEEBS rung in `ladder-rungs.spec`.
3. `SPLIT` preserves `cap_type` (`capstone_dyn_unit.anvil:113-149` touches only start/end/cursor/
   revnode), and so does `cincoffset`/`cincoffsetimm` (`capstone_flu_unit.anvil:38-40`, `:68-70`
   copy the whole `metadata`). So every cap-table storage capability and every carved pointer
   below `sp` inherits NONLIN.
4. Both branches converge. **Stack** pointers and `s0` are `sp` itself or splits of it. **Heap**
   pointers are not what the premise assumed: this build does **not** use `umm_malloc` (that is the
   rv8 benchmarks, zero references in the SQLite path) — SQLite runs memsys5 over a static `.bss`
   arena, `sqlite_heap` (`sqlite_capstone_domain.c:28`, configured at `:1559`), which is itself a
   carved global reached through cap-table slot 176. So a "malloc'd" pointer is pointer arithmetic
   inside one NONLIN global. No `mrev`/REVOKE path is live: `revoke_on_free_alloc.h` is not
   referenced by the SQLite build.
5. **Empirically corroborated by S-02**: a `delin` on `hostcall_payload` *wedged the board*
   precisely because the capability was already NONLIN and the RTL's DELIN accepts LINEAR only
   (`sqlite_capstone_domain.c:208-238`).

**Consequence — a real exclusion, on two independent legs.** The linear move-clear mechanism is
**not live at the S-12 fault site**, for SQLite or for the repro. Measured on silicon here (arm 6,
`bad=0`) and derived-and-verified for SQLite above. It is likewise exempt from the STC rs2-clear
(`capstone_dyn_unit.anvil:439`, `:458`). Whatever loses the value at the fault site, **it is not
source-granule clearing on `ldc`/`stc`.**

**And it clears the repro of the fidelity charge that was open against it.** The worry was that
`v`'s type made the repro test the wrong thing. It did not — the repro's NONLIN matches
production's NONLIN. So the window *with the right type* is genuinely clean, and the missing
ingredient is elsewhere: cache pressure, interrupt/timer traffic, rev-node churn, or simply the
surrounding code volume. That is where an S-12 reproducer has to come from, and it points at
SQLite-side bisection rather than at another arm of this kernel.

**One caveat on provenance.** The derivation above was produced by an agent that read a *stale*
copy of `s12_kernel.h` and quoted its "the granule is intact AND TAGGED … never lost, it was never
delivered" paragraph as current. That paragraph is **withdrawn** (see the header rewrite). The
derivation does not depend on it, but do not carry that sentence forward from any report quoting it.

**The one measurement that would close this by observation rather than derivation** already exists
as an instrument: `CAPSTONE_ARG_PROBE=sqlite3WhereCodeOneLoopStart`
(`build-sqlite-silicon.sh:1030-1060`), expecting **1** on the incoming `pWInfo`. Note the encoding
trap: `lcc` selector 1 reports `cap_type - 1` with NOT_CAP special-cased to 7, so **1 means NONLIN**
and reading it as LINEAR against the raw `asm_insn.h` numbering is the natural mistake
(`sqlite_capstone_domain.c:217-222`).

## MEASURED: SQLite's pointers at the fault site are NONLIN (type 1)

The derivation above is now closed by observation. With the arg probe repaired (its caller query
used a non-total selector and aborted every run — see below), a QEMU run of the same domain and
the same `q_two.test` case reports, at `sqlite3WhereCodeOneLoopStart` itself:

```
ARGP calls=1 ty1=0000000000000001 ty2=0000000000000001 ra=0000000101ce154c
SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=0 query_fail=1 completed=1
```

`ty1`/`ty2` are the two incoming pointers. **`lcc` selector 1 reports `cap_type - 1`, with NOT_CAP
special-cased to 7 — so `1` is NONLIN, not LINEAR.** Reading it against the raw `asm_insn.h`
numbering is the natural mistake and is documented at `sqlite_capstone_domain.c:217-222`.

Two independent routes now agree — the five-link derivation (monitor LINEAR → `delin(sp)` →
type-preserving `SPLIT`/`cincoffset`) and this direct measurement. That matters because any one
broken link would have taken the derivation's conclusion with it.

**Caveat:** measured under QEMU, not silicon — a cross-check of the derivation, not a silicon
measurement. The comparable repro measurement did agree across both (arm 5 returned `0xC12A5100`
on QEMU *and* on the FPGA), which is evidence the type models match at least here, but the silicon
reading at this site remains unmeasured.

### The probe defect that had to be fixed first

`CAPSTONE_ARG_PROBE`'s two argument queries are selector 1 (total, answers 7 for an untagged
operand). Its **caller** query was not:

```c
__asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, x1, x2" : "=r"(ap_ra_));   /* lcc ra, selector 2 */
```

Selector 2 is not total: on an untagged operand QEMU trips `assert(rs1_v->tag)`
(`op_helper.c:762`) and **aborts the emulator**; silicon raises. And in this build that is
guaranteed, not incidental — the SQLite silicon domain contains **zero `cjalr`** (verified by
`objdump`), so every call is a plain `jal` and `ra` is always a plain integer. The probe therefore
died immediately after `SQ: G/enter` on every run, never reaching the site it was aimed at, and
would have faulted the domain on the board too. Its header's "non-perturbing — records and falls
through" was false for that half.

Fixed to ask the total selector first and take the cursor only when there is a capability to take
it from. Positive-tested the same day: zero `helper_cslcc` aborts, run completes, summary identical
to the un-probed run and to native.

## CONFIRMED LIVE on silicon, 2026-08-25 — and silicon-only

Two arms, one boot, host and domain staged from the same build:

| arm | markers | verdict |
|---|---|---|
| `sqslt.dom` (no `--slt`) — control | `G/enter` → `H/return` | returned in 7 s, **so the boot carries verdicts** |
| `sqslt.dom --slt q_two.test` | `G/enter`, **no `H/return`** | **entered and WEDGED** |

`G/enter` present with no `H/return` is a real wedge, not an entry stall — that distinction is the
whole reason the arm is readable. The fault site was re-derived from the binary that actually ran:

```
latched trap  mcause=25 (UNEXPECTED_OPERAND)   mepc=0x828f4814   DBAS=0x82800000
  -> VA 0x104814;  sqlite3WhereCodeOneLoopStart at 0x104788  ->  +0x8c

104808:  movc  a4, zero               the zeroed value
10480c:  stc   a4, 0x0(a5)
104810:  ldc   a4, 0x0(a0)            the reload
104814:  cincoffsetimm a4, a4, 0xb0   <- FAULTS
```

Exactly the documented signature. **Read the CSRs with care:** gdb reported `mcause=2 mepc=2`,
which the driver DISCARDED because it disagreed with the latched trap — a later trap had clobbered
them. Taking the gdb values at face value would have produced a completely different, wrong story.

**And it does not reproduce under QEMU.** The *same* domain image running the *same* `q_two.test`
completes in emulation (`G/enter → H/return`, `SLT-SUMMARY … completed=1`) and matches the native
baseline record for record. S-12 is **silicon-specific**, and there is now a reference model to say
so — the Q-01 fix.

### Two arms that were NOT verdicts, recorded so they are not re-run as if they were

* A three-arm boot put `q_two` **third** (domain `id=2`, region ids already `0x0A`). It returned
  nothing, but stopped at `SQ: C/mkregion2` — before `D/mapped`, `F/share2` or `G/enter`. That is
  a hang in **region creation, outside the domain**, while S-12 is inside it and necessarily
  follows `G/enter`. "NO RETURN" alone does not separate the two. Put the subject in **slot 2**,
  matching the historical configuration that wedged.
* An earlier boot used a ladder-style `label:path` entry with `SQLITE_HOST=sqlite_host.user`. The
  label became `argv[1]`, the loader printed `Failed to open the file.`, and the driver hard-stopped
  the arm as a PHANTOM — correctly refusing to read the `SQ: G/enter`/`H/return` that followed,
  which belonged to a domain that was never loaded.

### Where this leaves the mechanism

Open, and now bounded from both sides. The move-clear account is dead (NONLIN, measured and
derived). The 40-line window is clean with a proven-firing detector under both consumers. So the
trigger needs something the window does not have — cache pressure, timer traffic, rev-node churn,
or simply the surrounding code volume — and the next step is SQLite-side bisection between the
single-table case that passes and the self-join that does not. `q_one` and `q_two` differ by
exactly `, t1 AS y`, and both tables are EMPTY, so no data is involved either way.

## Silicon: NONLIN confirmed, and S-12 is PERTURBATION-SENSITIVE

Same domain, same `q_two.test`, same slot (arm 2 of 2), with the repaired arg probe enabled:

```
arm 1  sqprb.dom (no --slt), control   G/enter -> H/return   returned 7s
arm 2  sqprb.dom --slt q_two.test      G/enter -> H/return   returned 6s
       ARGP calls=1 ty1=1 ty2=1
       SLT-SUMMARY records=2 stmt_pass=1 query_pass=0 query_fail=1 completed=1
```

**1. The type is now an FPGA measurement.** `ty1=ty2=1` = NONLIN (selector 1 reports
`cap_type - 1`). Previously derived from source and measured only under QEMU. The linear
move-clear is not live at this site, confirmed on the hardware that actually has the bug.

**2. The probed build does not wedge** — where the un-probed build, in the identical slot, wedged
at `+0x8c` in the boot immediately before. And it completes *correctly*: its summary matches QEMU
and native record for record.

**What that does and does not support.** The two builds differ in **two** respects:

| build | size | `WhereCodeOneLoopStart` |
|---|---|---|
| un-probed (wedges) | 1624152 | `0x104788` |
| probed (completes) | 1624904 | `0x104b48` |

The probe adds instructions to the function **and** moves it 960 bytes. So *"S-12 is sensitive to
code perturbation"* is supported; *"the probe's instructions cure it"* is **not**. This is exactly
the arms-differ-in-more-than-one-respect case, and link address is the named confound.

Perturbation sensitivity is itself a strong hint: it is **R-20's signature** — cured by exactly one
nop — which points at a pipeline/scheduling hazard rather than data or memory corruption. It also
explains why the 40-line repro comes back clean: its instruction context is not production's.

**UNRESOLVED:** only one `ARGP` line appears for a two-table self-join, which should drive two
where-loop levels. Whether the report is first-call-sticky or the function is genuinely entered
once is not established.

**The one-boot discriminator:** inject the probe into `sqlite3WhereMalloc` (`0x102f54`, *before*
the fault site). That moves `WhereCodeOneLoopStart`'s address while leaving its body
**byte-identical**. Wedges → the instructions inside the function are what matter. Completes →
the address/layout shift is, and the instructions are incidental.

## The cure does NOT require touching the fault function's instructions

The discriminator ran: probe injected into `sqlite3WhereMalloc`, which sits *before*
`WhereCodeOneLoopStart` in the layout, so the fault function's address moves while its
instruction stream stays the same. Verified rather than assumed — same 2866 instructions in the
same order, the only body difference being one immediate (`addi a2,a2,-0x3d0` → `-0x3d4`), a
globals-offset adjustment that changes no timing or scheduling.

| build | size | `WhereCodeOneLoopStart` | body | result |
|---|---|---|---|---|
| un-probed | 1624152 | `0x104788` | baseline | **WEDGES** at `+0x8c` |
| probe **in** WhereCode | 1624904 | `0x104b48` | + probe instructions | completes |
| probe in **WhereMalloc** | 1624904 | `0x104c0c` | same stream, 1 immediate | **completes** |

Both probed builds complete, and their `SLT-SUMMARY` matches QEMU and native record for record —
they are running correctly, not failing differently.

**What is established:** the fault survives or vanishes according to the *surrounding layout*, not
according to the content of `WhereCodeOneLoopStart`. Adding instructions inside that function is
not necessary to cure it.

**What is NOT established, and must not be written as though it were:** *which* layout property
matters. Address, image size, and globals offset all moved together in both probed builds, so
"it is the address" is one candidate among three. Isolating it needs a build that shifts the
address while holding size and globals fixed, which the two-pass link does not currently offer as
a knob.

**The N=1 problem, being fixed before anything is built on this.** The wedge itself has been
observed **once**. On a system with known per-image nondeterminism — R-16's entry stall is
explicitly per-image, and the standing rule is to REDRAW rather than retry — a single observation
is not enough to carry the "layout decides it" reading. A repeat of the identical un-probed image
is running. If it wedges again the foundation is sound; if it does not, S-12 may be an
alignment/layout **lottery** of the same class as R-16, and every arm above needs re-reading in
that light.

### N=2: the wedge is DETERMINISTIC, not a lottery

The identical un-probed image was re-run on a fresh boot and wedged again, bit-identically:

| run | control | `--slt q_two.test` | latched trap |
|---|---|---|---|
| 1 | returned 7 s | `G/enter`, no `H/return` | `mcause=25  mepc=0x828f4814` |
| 2 | returned 7 s | `G/enter`, no `H/return` | `mcause=25  mepc=0x828f4814` |

Same PC, same cause. **The R-16-class "alignment lottery" hypothesis is refuted**: this image does
not sometimes wedge, it always does. Together with the probed images completing (two different
probe placements, plus their controls), the tally is:

```
un-probed image          q_two   WEDGED     2/2
probe-in-WhereCode       q_two   completed  1/1
probe-in-WhereMalloc     q_two   completed  1/1
```

So the behaviour is a deterministic function of the image, and **layout selects it**. That is what
licenses the layout reading; before this repeat it rested on a single observation, and "this image
lost a lottery" predicted exactly the same data.

**Two variables are now both known to matter, independently:**

* **The SQL.** In the *same* un-probed image and the same slot position, `q_one` completes and
  `q_two` wedges. The pair differs by exactly `, t1 AS y`, with both tables EMPTY — so it is the
  join's generated code path, not data.
* **The layout.** With the *same* SQL, moving `WhereCodeOneLoopStart` cures it without altering a
  single instruction of that function.

A mechanism has to explain both: a code path that only the join generates, which faults only at
certain placements. Deterministic, PC-sensitive, and not source-granule clearing (NONLIN, measured
on silicon) — that profile points at the front end or at something indexed by address, rather than
at data or memory corruption.

## Isolated: the deciding variable is the INSTRUCTION side, not the data side

`link-gpfree.ld` puts `.text` at `0x10000` and the globals at `0x10000 + GOFF`, so raising `GOFF`
moves the globals region while leaving every `.text` address untouched. A new diagnostic knob
(`SQLITE_GOFF_OVERRIDE`, refused if below the computed value so globals can never overlap `.text`)
builds exactly that image. Verified rather than assumed: **331778 instructions, 4 differing** — all
`auipc` globals-base constants (`0x161`→`0x171`) — and `sqlite3WhereCodeOneLoopStart` still at
`0x104788`.

| `WhereCode` @ | globals @ | `.text` | result |
|---|---|---|---|
| `0x104788` | `0x150000` | baseline | **WEDGE**, 2/2 |
| `0x104788` | `0x160000` | 4 `auipc` differ | **WEDGE**, 1/1 |
| `0x104b48` | `0x150000` | +probe in the function | completes |
| `0x104c0c` | `0x150000` | probe in `WhereMalloc` | completes |

**Globals placement is eliminated by direct experiment.** It was already suspect — `GOFF` is
`0x150000` in *all* earlier builds, wedging and curing alike, because it rounds up to 64 KiB and a
~800-byte probe never crosses the boundary — but now it has been moved deliberately and the fault
survived. Every image with the function at `0x104788` wedges (3/3); every image with it moved
completes (2/2).

## What the fault operand actually implies — stronger than "detagged"

`decompress_cap_tagged` (`core/include/ariane_pkg.sv:766-782`) does **not** zero the metadata word
on an untagged read; it stashes the raw 64 bits into `bound_start`, and the cursor passes through
unchanged:

```systemverilog
end else begin
  return '{ metadata: '{ ..., cap_type: NOT_CAP, ...,
                         bound_start: tagged_metadata[63:0],   // raw stash
                         bound_end: '0 },
            cursor: cursor };
```

So an operand that is bit-for-bit `create_cnull` — which `tval = 0` together with `cap_type ==
NOT_CAP` implies — requires the memory to have been **genuinely all-zero on both halves**, not
merely detagged. **A detagged capability would have delivered a non-zero cursor and `tval` would
not be 0.** That excludes the whole "a good pointer lost its tag" family at this site and demands
either a location that really held zero, or a load that returned a default without reading.

**One tempting instantiation, refuted by arithmetic.** The window stores an all-zero capability two
instructions before the reload (`movc a4, zero; stc a4, 0x0(a5)` → `s0-0x120`), so "the reload was
served that zero from the wrong address" is a bit-exact match for the signature. It does not
survive the addresses: at the wedge `s0 = 0x82b9f480`, so the zero-store lands at `0x82b9f360` and
the reload reads `a0 = 0x82b9f410` (= `s0-0x70`). D$ is 32 KiB / 8-way = 4 KiB per way, so the
index is `paddr[11:4]`: **0x36 vs 0x41**, and the granules (`0x82b9f36` vs `0x82b9f41`) differ too.
No set collision and no write-buffer granule match. (`s0` is corroborated independently: the s07
STC recorder reported the last capability store at granule `0x9f360`, exactly `s0-0x120`.)

## Two exclusions in the older notes that are narrower than they read

* **The writeback-port displacement detector** (`core/scoreboard.sv:335-347`) fires on
  `op == LDC` at the writeback trans_id and never inspects the data. Its own comment describes the
  signature it catches as *"retires with a correct cursor and NOT_CAP metadata"*. **Our fault has
  cursor 0 as well**, so a mechanism that zeroes *both* halves would not trip it. Reading its
  `0x00` as "displacement excluded" is only valid for "wrong port, data preserved".
* **Instruction realignment** (`core/instr_realign.sv:66-115`) branches on `address_i[1]`, which is
  layout-dependent in general — but **excluded for this pair by arithmetic**: `0x104788 & 3 == 0`
  and `0x104c0c & 3 == 0`, and the intervening stream is unchanged, so every instruction's
  alignment parity relative to the function start is preserved. (Not a general exclusion — a shift
  that changed the mod-4 residue would reopen it.)

**Still PC-indexed and not excluded:** the I-cache line phase (`CVA6ConfigIcacheLineWidth = 128`,
16-byte lines; `0x104788 mod 16 = 8` vs `0x104c0c mod 16 = 12`) and the BTB, which indexes on
`vpc_i[PREDICTION_BITS-1:ROW_ADDR_BITS+OFFSET]` with only 8 entries (`core/frontend/btb.sv:66-72`),
so aliasing shifts when a function moves. Neither has a demonstrated path to a corrupted capability
*value* — both are timing/contention knobs, which is how this folder should keep describing them
until a path is shown.

## RETRACTION x2 — the "byte-identical body" claim and the "NONLIN measured at the fault site" claim

Both were caught by an adversarial audit and both are verified against primary source here.

### 1. The fault function was NOT byte-identical between the two builds

Stated in this file and in commits `384a919666bf` / `96288aa211fa` as *"2866 instructions in the
same order, the only body difference one immediate"*. **False.** Measured over the exact symbol
bounds from `llvm-nm --print-size` (`0x104788`+`0x47e0` vs `0x104c0c`+`0x47dc`):

```
slt2 (wedges)    4600 instructions
slt5 (completes) 4599 instructions        <- not even the same COUNT
differing:       1469,  of which 958 differ in mnemonic/register, not just an immediate
```

Register allocation and the gp-table index materialisation changed. **How the error was made:**
the comparison helper's end-of-function regex terminated early (2866 instructions instead of
4600), and it printed only the FIRST difference — which I then reported as though it were the
ONLY one. Counting was one line away and would have caught it.

**Consequence:** *"the cure does not require touching the fault function's instructions"* is
**unearned**. The instructions were touched. The discriminator did not do what it was built to do.

The confound list was also wrong. Between the wedging and curing images `.text` size, every data
section's VA, `.capstone_gp_table` size (+0x90, nine more entries) and the cap-init entry count all
moved, along with register allocation inside the fault function. And **the globals offset was never
a confound at all** — `GOFF` is `0x150000` in every build (it rounds to 64 KiB); the
`-0x3d0`→`-0x3d4` difference is a per-reference offset, not the region base. The GOFF-override
experiment is still valid and still useful (it independently confirms the wedge at N=3 and rules
the globals region out), but it was answering a question that was not open.

### 2. NONLIN was NOT measured at the fault site

The arg probe reads arguments **1 and 2**. The faulting value is argument **3**:

```
1047a4:  movc a6, a0             a6 = arg1 (pParse)     <- ty1
1047a8:  ldc  a0, -0x600(s0)     a0 = arg2 (v)          <- ty2
1047c0:  cincoffsetimm a0, s0, -0x70
1047c8:  stc  a2, 0x0(a0)        arg3 -> s0-0x70        <- THE SUBJECT, never probed
104810:  ldc  a4, 0x0(a0)        reload arg3
104814:  cincoffsetimm a4, a4, 0xb0   FAULTS
```

So `ty1=ty2=1` says two *sibling* arguments are NONLIN. The value that is stored, reloaded and
faults was never measured — and it could not have been, because probing the image cures it.

**What survives:** the RTL half. NONLIN is absent from the clear set (`load_unit.sv:226-229`), so
*if* the value is NONLIN the move-clear cannot fire. The claim must read **"inferred NONLIN by
analogy with two sibling arguments, not measured"**. Re-pointing the probe at argument 3 would not
fix this — it can only ever measure a completing build. Only a non-perturbing hardware instrument
can read the operand on an image that actually wedges.

### 3. Corrections to two other things this folder asserted

* **"The two-pass link offers no such knob" is false.** `build-sqlite-silicon.sh:462` already has
  `CAPSTONE_TEXT_PAD=N` — dead, never-called code at the top of `.text`, no globals, written for
  exactly this purpose ("the null-shift control the audit asked for"). With `.text` at `0x144004`
  and globals pinned at `0x160000` there is ~49 KB of headroom before anything downstream moves.
* **"One subject arm per boot" is false.** Three domains have reached `H/return` in a single boot
  (`slt-board8/10/11.log`, ids 0,1,2), with `SPLB:0000E010` (`CAPSTONE_ERR_SPLIT_EXACT`) hitting on
  the *fourth*. `up6b`'s death at the third is pool-state dependence, already documented in
  `RATE-RULE.md`. **Budget three arms.**
* **"Deterministic" is overclaimed at 2/2.** `RATE-RULE.md` records a background wedge rate of
  p̂ ≈ 0.22, and 2/2 gives only a 95% lower bound of p ≥ 0.224 — indistinguishable from background.
  N=3 puts P(background) at 0.011, which is the cheap meaningful step; "deterministic" by counting
  alone would need n≈14–29. (The 0.22 was measured on a *different* bitstream, `s07debug_18august`,
  not the resident `s07clear_84ed6eafb` — quote it with that caveat.) The **signature** carries more
  weight than the count: two bit-identical `25 / 0x828f4814` readings are far less likely under
  background than two bare wedges, but nothing currently records per-wedge signatures to establish
  that. Worth starting to record.
* **Entry attribution should quote `ENT1`, not `SQ: G/enter`.** The monitor's own rule
  (`sbi_capstone.c:924-926`): `ENT0` then silence = died before the switch; `ENT1` then silence =
  control genuinely left M-mode and the domain owns the wedge; `ENT2` = returned. Both wedges show
  `ENT0`+`ENT1` then silence.

## RETRACTED (3rd time): "layout decides it" is DEAD. Address is not the variable.

The address-only experiment, built properly this time. `CAPSTONE_TEXT_PAD=1120` inserts dead,
never-called nops at the top of `.text` — no globals, no control flow — tuned offline in two
builds so `sqlite3WhereCodeOneLoopStart` lands at **exactly `0x104c0c`**, the address at which the
probe builds completed. Gates run BEFORE the boot, which is what the retracted attempt skipped:

```
instruction count          4600 == 4600
STRUCTURAL diffs in fn     0            (605 immediate/label-only, from pcrel shifts)
.capstone_gp_initdesc      VA+size IDENTICAL
.rodata / .data / .bss     VA+size IDENTICAL
residual                   .gct +0x50, gp_table VA +0x50 (SIZE unchanged -> same indices)
```

**It wedges.** `ENT0=1, ENT1=1`, no `ENT2` — control left M-mode and the domain owns it. Latched
trap `mcause 25`, `mepc 0x828f4c98` = `0x104c0c + 0x8c`, and the instruction there is the same
`cincoffsetimm a4, a4, 0xb0` preceded by the same `ldc a4, 0x0(a0)`.

| image | `WhereCode` @ | globals @ | fault | cause |
|---|---|---|---|---|
| baseline | `0x104788` | `0x150000` | `+0x8c` ×2 | 25 |
| GOFF moved | `0x104788` | `0x160000` | `+0x8c` | 25 |
| **TEXT_PAD** | **`0x104c0c`** | `0x150000` | **`+0x8c`** | **25** |
| probe in WhereCode | `0x104b48` | `0x150000` | — completes | |
| probe in WhereMalloc | `0x104c0c` | `0x150000` | — completes | |

**Same address, opposite outcomes.** `TEXT_PAD` and the WhereMalloc-probe build both put the
function at `0x104c0c`; one wedges and one completes. Address cannot be the variable, and neither
can the globals region. **What is left is the code change the probe introduced** — the register
allocation inside the fault function (958 structurally different instructions), the nine extra
gp-table entries, and the changed cap-init length. The `TEXT_PAD` build has none of those and
faults exactly like the baseline.

So the correct statement is the narrow one: **something about the code generated for this function
(or the cap-table/cap-init geometry around it) decides the fault, and pure placement does not.**

### And a self-inflicted error, recorded because it is the same class

This boot's driver line first read `latched trap (24/...)`, and I began interpreting mcause 24 —
checking whether it was `NO_EXCEPTION` or the `commit_stage` off-by-one at `commit_stage.sv:205-228`.
It was neither. Earlier today I "fixed" the message's hardcoded `25` by interpolating
`_latched & 0x7f` — but `_latched` is the latched **mepc**, not the cause, and `0x828f4c98 & 0x7f`
is 24 by coincidence. The trap log itself read `0x99` → cause `0x19` = **25** all along.

The original hardcoded `25` was wrong for a different reason (it printed 25 even when the latch
held cause 9), so the defect was real — but my fix invented a number out of an address. It now
reads the cause from the trap-log value (`traplog_v & 0x7F`), which is where it was available the
whole time. **A "fix" that produces a plausible number from the wrong variable is worse than the
bug it replaced**, and it very nearly bought a mechanism investigation into an exception code that
was never raised.

## FOUND: `.bss` SIZE decides S-12 — 32 bytes cures it, and the prediction was made first

Tabulating every image built against its outcome, exactly one property separates the two groups:

| build | fn @ | caps | `.bss` VA | **`.bss` size** | outcome |
|---|---|---|---|---|---|
| slt2 | `104788` | 328 | `170bf0` | **`409c0`** | WEDGE ×2 |
| pad2 (TEXT_PAD) | `104c0c` | 328 | `170bf0` | **`409c0`** | WEDGE |
| goff (globals +64K) | `104788` | 328 | `180bf0` | **`409c0`** | WEDGE |
| gp6 (+6 `.data` globals) | `1048e4` | 338 | `170d20` | **`409c0`** | WEDGE |
| slt4 (probe in fn) | `104b48` | 337 | `170cb0` | **`409e0`** | completes |
| slt5 (probe in WhereMalloc) | `104c0c` | 337 | `170cb0` | **`409e0`** | completes |
| **bss32 (+32 B `.bss`)** | `10479c` | 329 | `170c00` | **`409e0`** | **COMPLETES** |

Function address, cap count, `.bss` VA and heap VA each vary *within both groups*, so each is
refuted by its own data. **`.bss` size does not**: `0x409c0` in every wedging build, `0x409e0` in
every completing one. The `0x20` delta is exactly the four `unsigned long` globals the arg probe
adds.

**`gp6` is the control that makes this a mechanism rather than a coincidence.** It appends six
dummy globals *with initialisers*, so they land in `.data` and `.bss` never moves — same extra
globals, wrong section — **and it wedges.**

**And `bss32` was predicted before it was run.** Take the wedging build, append 32 bytes of
uninitialised global, change nothing else. Predicted: completes. It completed —
`ENT2=5117600D`, `SLT-SUMMARY records=2 … completed=1`, matching QEMU and native.

### The mechanism chain, visible in the build's own output

`.bss` size decides how much room `domdata-budget` leaves for the stack:

```
slt2   (WEDGES)     = STACK  2347072
bss32  (COMPLETES)  = STACK  2346992        <- 80 bytes smaller
```

So: **`.bss` size → stack size/base → `s0` → the address of the faulting slot at `s0-0x70`.** That
is a data-address mechanism, and it survives every code-side refutation — address, cap count,
globals region, and the code of the fault function itself (whose instructions are byte-identical
from entry through the fault in both a wedging and a completing build).

### What this retires

Every "layout"/"perturbation-sensitivity" reading in this folder above should be read as **stack
address sensitivity**. The probe builds never cured anything by perturbing *code*; they cured it by
adding four globals to `.bss` and moving the stack. That is why probing the function and probing an
uncalled function worked equally well, and why `TEXT_PAD` at the identical address did not.

### Next: the periodicity ladder

`CAPSTONE_BSS_PAD=N` is now the knob, and it is cheap and code-neutral. A ladder of N
(16/32/48/64/…/4096) maps wedge vs complete against stack displacement. If the pattern is periodic,
the period names the aliasing structure directly — D$ set (32 KiB/8-way → 4 KiB per way, index
`paddr[11:4]`), write-buffer granule, or page. That is a far sharper question than anything
available before today.

### Boot-budget caveat

The third arm of this boot died at `SPLB:0000E010` (`CAPSTONE_ERR_SPLIT_EXACT`) on domain `id=2`,
before `SQ: G/enter` — no verdict, exactly as in the earlier three-arm boot. Three arms is a
gamble, not a guarantee: it is pool-state dependent. **Put the arm that matters second.**

### But the slot's ADDRESS is not the variable — `gp6` moved it and still wedged

| build | `s0` | slot (`a0`) | outcome |
|---|---|---|---|
| slt2 | `0x82b9f480` | `0x82b9f410` | WEDGE |
| pad2 | `0x82b9f480` | `0x82b9f410` | WEDGE |
| goff | `0x82b9f480` | `0x82b9f410` | WEDGE |
| **gp6** | **`0x82b9f350`** | **`0x82b9f2e0`** | **WEDGE** |

`gp6`'s stack sits `0x130` lower and its faulting slot is at a different address — and it wedges
anyway. So the fault is **not** "the slot must land at one particular address", and not a fixed
D-cache index (`paddr[11:4]` = `0x41` for the first three, `0x2e` for gp6) or granule.

This matters because it kills the tempting chain *.bss → stack base → slot address → address-indexed
structure* at its **last** link, using data already in hand. `.bss` size still separates the two
groups across seven images; what it does downstream is not simply "moves the slot somewhere bad".

The first link is independently confirmed — the stack budget tracks the pad — but **not linearly**:

```
pad=0   STACK 2347072
pad=16  STACK 2347008     (16 bytes of .bss costs 64 of stack)
pad=32  STACK 2346992     (32 costs 80)
```

so the carve arithmetic has its own rounding and a stack address must be **measured**, not predicted
from a pad size.

## MEASURED AT LAST: memory holds a live capability; the FLU was handed `create_cnull`

The driver now reads the faulting granule directly at the wedge, via GDB, using `s0` from the
same halt. Baseline image, real wedge (`ENT0`,`ENT1`, no `ENT2`):

```
slot   s0-0x70  @0x8279f410:  0x00000000827e4cd0   0x000003c7a7462d16
zerost s0-0x120 @0x8279f360:  0x0000000000000000   0x0000000000000000
LATCHED tval (sw 210..217):   0x00 x8   ->  tval = 0
```

* **The slot holds a real capability** — cursor `0x827e4cd0`, a live pointer into the domain's
  data region, with non-zero packed metadata. Memory is INTACT.
* **The zero-store slot is correctly zero**, so the two granules are not being confused.
* **`tval` is 0**, and this is the LATCHED value from the trap latch — not the gdb CSR the driver
  correctly discards as clobbered by a later trap.

**Why that is decisive.** `decompress_cap_tagged` (`ariane_pkg.sv:766-782`) passes the **cursor
through unchanged** on an untagged read, stashing the raw metadata into `bound_start` rather than
zeroing it. So:

| if the load had… | the FLU would have seen | `tval` |
|---|---|---|
| returned memory, tagged | a valid capability | no fault |
| returned memory, **untagged** (tag loss) | cursor `0x827e4cd0`, NOT_CAP | **`0x827e4cd0`** |
| **observed** | cursor **0**, NOT_CAP | **0** |

A cursor of 0 cannot be produced from this memory content under **any** tag state. **The operand
the FLU consumed did not come from the slot.** S-12 is an operand-DELIVERY failure, not a
memory-path loss and not a tag loss — the S-07 family is excluded at this site by direct
measurement rather than by inference.

### This restores a claim I retracted — but the retraction was still correct

"The value was never lost; it was never delivered" was withdrawn earlier because its evidence was
the shadow-tag read (DRAM, not the L1 tag the load consumed) and arm 4, which never wrote its slot.
That evidence was worthless and the retraction was right. **The claim now rests on different
evidence**: a direct read of the granule showing a non-zero cursor, plus the latched `tval` of 0,
plus the RTL's own untagged-decompress semantics. Same conclusion, sound basis. Worth stating
plainly, because "it was right all along" is the wrong lesson — it was *unsupported* all along,
and is supported now.

### Standing caveat

A GDB read at the halt shows memory *after* the fault. Nothing between the fault and the halt has
any reason to write a domain stack slot, and the value is exactly the shape the code stores there
(`pWInfo`), but this does not strictly prove the content at the instant of the load. The
non-perturbing rolling-LDC recorder is what would close that, by recording what the load itself
returned.

### Where the mechanism must now live

Between the load's result and the FLU's operand: the DYN unit's load syncer and its `trans_id`
pairing, the forwarding/bypass network, or write-back arbitration. The RTL review already flagged
that the syncer's "one op in flight" gating is weaker than the 96-LDC overlap test proved — that
test likely never forced a cache miss, so the long-latency window it was built to open may never
have opened. That is the next thing to attack, and it is a simulation question, not a board one.

## The load-syncer overlap lead, re-tested under forced misses — and it holds up

The board result (memory intact, latched `tval = 0`, NOT_CAP) matches one specific hypothesis
already on file. `s12-ldc-overlap.S`'s own header says the load-syncer mispair produces "a coupled
substitution … the only shape consistent with the board's `tval == 0` AND `cap_type == NOT_CAP`".
That test measured `init-while-pending = 0` and the lead was recorded as dead by unreachability.

**That test never opened the window it was built to test.** Its four granules sit inside a 64-byte
span, they are `STC`-written immediately before the loop, and its whole buffer is 1 KiB against a
32 KiB L1 — every LDC HITS, resolving well inside the one-cycle throttle at `ex_stage.sv:900-904`.
So its zero was uninformative for the MISS case, which is the case SQLite hits.

`s12-ldc-miss-overlap.S` forces misses by construction: D$ is 32 KiB/8-way with index
`paddr[11:4]`, so a 4096-byte stride collides in one set — 16 capabilities against 8 ways, an
eviction walk, then back-to-back pairs. (A 4 KiB stride cannot be an LDC immediate: I/S-type
immediates are 12-bit signed, so each strided pointer is built with `CINCOFFSET`, which takes a
register offset. The first version failed to compile for exactly that reason.)

| configuration | cycles | inits | ldc-pending | per init | init-while-pending |
|---|---|---|---|---|---|
| hit-only (existing test) | 1480 | 96 | 687 | 7.2 | **0** |
| miss-forcing | 1880 | 112 | 848 | 7.6 | **0** |
| miss + `CUT_ALL_PORTS` | 2113 | 108 | 926 | 8.6 | **0** |
| miss + `CUT_ALL_PORTS` + `MaxMstTrans=8` | 2078 | 112 | 923 | 8.2 | **0** |

**Two testbench fidelity limits found and tested, not assumed.** `ariane_testharness.sv:517` sets
`LatencyMode: axi_pkg::NO_LATENCY`, and `:514-515` set `MaxMstTrans`/`MaxSlvTrans` to **1** —
one outstanding transaction — both carrying the upstream comment "Probably requires update". Either
could have made the overlap unreachable *by construction*, which would have made every
`init-while-pending = 0` an artifact. **Raising both changes nothing**: latency moves 7.2 → 8.2
cycles per load and no second init ever lands.

**Reading.** LDCs are issued back-to-back and each takes ~8 cycles, so a second `init` had ample
room to land if the hardware permitted it. It never does, across four configurations. The DYN unit
serialises LDCs, and the load-syncer mispair is **not** reachable at these latencies — the original
verdict was better founded than the fidelity worry suggested.

**What remains untested, stated because it is the whole residual.** The sim's miss costs ~8 cycles;
a real board miss is far longer. Nothing here tests what happens at 50+ cycles of latency, and the
crossbar knobs cannot produce that — the D-cache/memory model resolves too fast regardless. If the
serialisation has any timeout or abort path that only a long miss reaches, this suite cannot see
it. The testbench was restored to its baseline afterwards so other lanes' timing is unchanged.

## Arm 7: the exact production shape is NOT sufficient — and the hypothesis was already on file

**The measurement that motivated it.** The driver has carried a pre-registered discriminator since
before any of this session's runs (`run_sqlite_stages_fpga.py:2637-2644`): `a4 == 0` at the wedge
means the load never wrote it and the stale-operand account is WRONG; `a4 != 0` means the load did
write it and the consumer read something else. **Six wedges, six CONFIRMED**, with `a4` holding the
slot's cursor exactly (`0x827e4cd0`).

Production's window makes the candidate obvious:

```
104808  movc a4, zero              a4 := create_cnull, ALL ZERO
10480c  stc  a4, 0x0(a5)
104810  ldc  a4, 0x0(a0)           same register, two instructions later
104814  cincoffsetimm a4, a4       faults with exactly a4's PRE-LOAD value
```

**Arm 7 reproduces that shape exactly** — one asm block, four instructions, nothing schedulable
between them, verified in the artifact (`movc a2,zero / stc a2 / ldc a2 / cincoffsetimm a2` at
`0x1041c`). Arms 0–4 have the same register reuse but the compiler places `movc` **nine**
instructions before the reload where production places it **two**, so none of them could have hit
the window even in principle.

**Result: `retval = 0xC12A7000`, 1924 cycles, clean.** Control (`k800`) returned. The shape is not
sufficient, and that was the pre-registered reading for a return.

### The hypothesis was not new, and searching prior art first would have found it

`capstone-ariane` commit **`4c0def314`**, *"Refute the wrong-producer-forwarding lead by measuring
its precondition"*, proposes this exact mechanism — it names `movc a4,zero` followed by `ldc a4`
and states that `{cursor 0, NOT_CAP}` "is the board signature exactly" — and then refutes it by
measuring the precondition: **duplicate live-`rd` cycles zero throughout**, with a proven heartbeat
and peak occupancy 2 of 8. That is prior art in this tree, and CLAUDE.md's rule to search the
registry, the repro folders and the commit history before investigating exists for precisely this.

**What survives of it.** The refuting test is **scalar-only by its own commit message** (div, rem,
add), so it establishes that the generic scoreboard mechanism is unreachable, not that a real
`movc`/`ldc` pair with a capability writeback is. Re-running that checker with Capstone ops instead
of scalars is board-free and closes the one gap the refutation left open — the cheapest remaining
step, and it needs no bitstream.

### What this leaves

`a4` holding the slot's cursor 6/6 still says the load wrote it and the consumer received something
else. So an operand-delivery failure stands — but it is **not** reproducible from the instruction
shape alone, which means the trigger needs context the four instructions do not carry: cache state,
the specific capability loaded, surrounding code volume, or execution history.

**And the instrument now going to synthesis cannot see this mechanism.** A stale-operand read
leaves the LOAD perfectly normal — right data, correctly tagged, on whatever leg it used. The
rolling recorder captures the load's memory-side response, so under the tag filter it records
nothing and unfiltered it records a normal tagged load. The fault is in the operand-select mux
downstream of writeback, where the recorder has no observation point. That is flagged to the RTL
lane, with a narrow 7-bit alternative (latch `rs1`, whether `rd_clobber_gpr[rs1]` was non-NONE, and
the selected entry's valid bit, at FLU issue) that follows the precedent already set by
`cap_wb_displaced_o` (`scoreboard.sv:345-353`).

## Arm 8: a COLD load does not do it either — the shape family is exhausted

Arm 7 established that production's four-instruction shape at production spacing runs clean. The
one controllable difference left was cache residency: arm 7's `ldc` HITS (the slot was written
moments earlier), while production's almost certainly MISSES — SQLite's working set dwarfs a
32 KiB L1 and the fault sits deep inside `sqlite3_prepare_v2`. That also fit every simulation
result, none of which exceeded ~8 cycles of LDC latency.

Arm 8 is arm 7 with the subject line evicted first. The eviction runs **before** the `movc`, so
the four-instruction tail is byte-identical and residency is the only variable — the matched-pair
discipline, one difference.

**Verified in the artifact before the boot, not in intent:**

```
s12_evict  @0x13000, 40960 bytes          > 32 KiB L1, so the walk cannot fit
loop 0x1044c..0x1049c, byte load, 16-byte stride   -> 2560 distinct lines
tail 0x104dc: movc a3,zero / stc a3,0(a1) / ldc a3,0(a4) / cincoffsetimm a3,a3,0xb0
```

**Result: `0xC12A8000`, clean, 245623 cycles** (against arm 7's 1924 — the walk demonstrably ran).
Control returned.

### The plain statement, since it was pre-registered

**Shape, spacing and miss latency TOGETHER are insufficient.** The instruction-window approach to
reproducing S-12 is exhausted: every controllable property of the four-instruction window has now
been reproduced exactly and none of it faults. The trigger needs something the window does not
carry — execution history, the specific capability's provenance, i-cache/code-volume effects, or
a code path in SQLite that differs from the one modelled here.

That is a narrowing, not a failure. What it retires is a whole family of hypotheses: anything of
the form "these instructions in this order at this spacing on a cold line". The next move is
SQLite-side bisection between `q_one` (passes on silicon) and `q_two` (wedges), which differ by
exactly `, t1 AS y` with both tables EMPTY — not a ninth arm of this kernel.

### A near-miss worth recording, because every gate was green

The first arm-8 build contained **neither** the eviction loop nor the four-instruction tail, and
was byte-size identical to arm 7. Cause: the block was inserted at an anchor that sits inside
`#if S12_ARM == 7`, so arm 8's code was nested in arm 7's guard and never compiled. The build
succeeded, the artifact looked plausible, the domain would have entered and returned clean — and
it would have read as "the cold load does not matter". Nothing reported an absence. Only
disassembling the artifact before spending the boot distinguished it. That is the third time today
that check has caught a test which would have measured nothing.

## The slot's capability, decoded — NONLIN confirmed a third way, and the core is not stalled

Two readings that were already in the wedge dumps and had not been extracted.

**1. The capability in the faulting slot, decoded from its raw metadata word.**

```
cursor      0x827e4cd0
metadata    0x000003c7a7462d16
  revnode_id  241
  perm        7  (rwx)
  cap_type    2  ->  NONLIN
  bounds      0x7462d16, cursorless=0
```

Layout is `revnode_id[29:0], perm[2:0], cap_type[2:0], bounds[27:0]`, MSB-first
(`ariane_pkg.sv:636-642`), so `cap_type` is bits [30:28] — independently confirmed by the S-06
comment about "raw data whose bits [30:28] decoded as LINEAR/NONLIN".

**This is a third independent confirmation of NONLIN**, after the silicon arg probe and QEMU, and
the first taken from the faulting granule itself rather than from a sibling argument or a
completing build. NONLIN is absent from the LDC clear set (`load_unit.sv:227-229`), so the
move-clear account stays dead.

**Watch the numbering — it has caused one retraction already.** My first decode used a name table
with `LINEAR=0` and printed `cap_type 2 -> REVOKE`, which would have *revived* the move-clear
hypothesis, since REVOKE **is** in the clear set. The RTL enum (`ariane_pkg.sv:654-663`) is
`NOT_CAP=0, LINEAR=1, NONLIN=2, REVOKE=3, …`, identical to `asm_insn.h:76-83`. Caught before it
was recorded. **Three numberings are in play in this investigation** — the RTL enum, `asm_insn.h`,
and `lcc` selector 1's post-shift form (`cap_type - 1`, NOT_CAP special-cased to 7) — and two of
them differ by one.

**2. The core is not stalled anywhere at the wedge.** Aperture 225 is
`{trace_buf_empty, dyn_wait_store_syncer, dyn_wait_load_syncer, dyn_wait_rev_res, dom_switch_busy,
stall_issue, mem_write_flag, mem_wait_flag}` (`cva6.sv:1189-1199`). All six wedges read **`0x80`**
— only `trace_buf_empty`. No syncer wait, no rev-node wait, no memory wait, no issue stall.

That kills a hypothesis before it was built: the driver's own comment at that aperture says "every
wedge so far reads sw=225 = 0x95, i.e. wrev=1 AND memwait=1: the dyn unit is blocked in
`get_node_query_validity` while the rev-node unit waits on the node-table memory read". **That
comment is stale relative to these wedges** — it describes an older wedge class. The rev-node
blockage story does not apply here, and the quiescent core is instead consistent with the
trap-loop-at-`mtvec=0` picture the `mcause=2 / mepc=2` readings already showed.

## THE TRIGGER IS THE SECOND WHERE-LOOP LEVEL, not the self-join

The instruction-window family was exhausted, so the remaining variable was the one that flips the
outcome **inside a single image**: the SQL. The domain reads its `.test` from the shared region at
RUN TIME, so these arms share one binary — no rebuild, and layout, `.bss` size, cap-table geometry
and code are all held fixed *by construction*.

| case | SQL | where-loop levels | silicon |
|---|---|---|---|
| `q_one` | `SELECT t1.a FROM t1` | 1 | **passes** (3×) |
| `q_two` | `SELECT t1.a FROM t1, t1 AS y` | 2 | **wedges**, `mcause 25`, `mepc 0x828f4814` = `+0x8c` |
| **`qj2`** | **`SELECT t1.a FROM t1, t2`** | **2** | **wedges**, same cause, **same `mepc`** |

`qj2` joins two **distinct** tables — no alias, no self-reference, no shared cursor. It wedges
identically, at the same instruction. **So the self-join is irrelevant; the second where-loop level
is the trigger.** All four cases agree with the native baseline, so the divergence is silicon.

**Why this matters more than another exclusion.** `sqlite3WhereCodeOneLoopStart` is called once per
loop level. One level passes; two levels fault, at the same PC. The natural reading is that **call
#1 succeeds and call #2 faults** — which makes the target the state built *between* the two calls,
not the function's code, and not anything about the image. That is a far narrower object than "all
of SQLite", and it is the first hypothesis in this investigation that predicts the SQL-sensitivity
directly instead of accommodating it.

## And the arg probe's "cure" is probably the `.bss` effect in disguise

The probe was assumed to perturb execution and thereby hide the fault. Look at what it actually
does to the image:

```
un-probed   WEDGES     .bss = 0x409c0
probed      completes  .bss = 0x409e0     <- +0x20, exactly 4 unsigned long globals
bss32       completes  .bss = 0x409e0     <- same value, reached with dead padding
bss16       WEDGES     .bss = 0x409d0
```

The probe adds **four `unsigned long` globals = 32 bytes**, taking `.bss` from `0x409c0` to
`0x409e0` — precisely the value independently shown to cure the fault with *no* instrumentation at
all. So "probing cures it" and "`.bss` 0x409e0 cures it" are not two facts; they are very likely
one fact seen twice.

**That is testable and it matters, because it would give back an instrumented image that still
faults.** A probe carrying only **two** globals lands `.bss` on `0x409d0`, which is a measured
WEDGING value. If such a build still wedges *and* still reports, the "any in-domain instrument
cures the bug" constraint — which has shaped this whole investigation — is lifted, and the arg
probe becomes usable on a faulting image to report which invocation faults.

### CORRECTION to the level-2 claim: it is 3/4, not deterministic — and my log parsing was unsafe

I recorded "two where-loop levels wedge" as though it always does. The driver's own verdict lines,
which are the authoritative record, say otherwise:

| boot | 1 level (`q_one`) | 2 levels (`qj2`) |
|---|---|---|
| up21 (un-instrumented) | returned 6 s | **NO RETURN** |
| up22 (counter build) | returned 6 s | **NO RETURN** |
| up23 (counter build) | returned 6 s | **NO RETURN** |
| up24 (counter build) | returned 22 s | **returned 7 s** |

So `q_one` passes 4/4 and `qj2` wedges **3/4**. The level-2 dependence is real — 3/4 sits well
above the documented background wedge rate of p̂ ≈ 0.22 — but it is a RATE, not a certainty, and
"1 level passes" is consistent with both "never wedges" and "wedges rarely". Any single boot on
either case proves nothing on its own.

**And two of my own readings of these logs disagreed with each other**, which is why this needed
settling rather than asserting. Reassembling the UART out of the driver log and scoping to the
last OpenSBI banner picked up the previous boot's replayed content — for up23 it returned
`ENT=[0,1,0,1,2]`, which is two arms' markers concatenated, while the raw transcript showed the
arm's own region as empty. **Use the driver's `[stages] <-- TEST …` verdict lines.** They are what
the driver actually observed, they are one line per arm, and they cannot splice two boots
together. The UART reassembly is for reading detail INSIDE an arm already identified that way.

## Sharper constraint: `tval = 0` means the DATA was zero, not merely detagged

Hypothesis generation produced a tag-desync account (a same-word merge into an already-issued
capability-store transaction, leaving DRAM `ctag=1` and L1 `ctag=0` — the write-buffer residual
that `wt_dcache_wbuffer.sv:604-619` documents as pre-existing and deliberately un-fixed). It does
not survive the measurement, and the reason narrows the search:

**`decompress_cap_tagged` (`ariane_pkg.sv:766-782`) passes the CURSOR THROUGH unchanged on an
untagged read.** A pure tag loss therefore delivers cursor `0x827e4cd0` and `tval` reads
`0x827e4cd0`. The latched `tval` is **0**. So the load was not served correct data with a lost
tag — **it was served all-zero 128 bits.**

That is a much tighter constraint than "the operand was NOT_CAP", and it excludes every
tag-only mechanism at this site, not just the move-clear.

**And the window contains exactly such a value, two instructions earlier:**

```
104804  cincoffsetimm a5, s0, -0x120
104808  movc a4, zero          <- a4 := create_cnull, ALL ZERO
10480c  stc  a4, 0x0(a5)       <- stores it to s0-0x120     ... this is `Index *pIdx = 0;`
104810  ldc  a4, 0x0(a0)       <- reloads pWInfo from s0-0x70
104814  cincoffsetimm a4, a4, 0xb0    FAULTS with all-zero
```

The cnull store is the compiler initialising the local `Index *pIdx = 0;` — a capability-width
store of bit-for-bit `create_cnull`, issued two instructions before the reload that fails, into a
*different* granule of the same frame.

**So the shape to explain is: a zero capability stored to one stack granule, and the very next
capability load from a different granule of the same frame returning that zero.** That is
squarely the write-buffer/forwarding family, and it is the first account consistent with every
measurement at once — memory intact (the entry converges and drains correctly, which is why the
post-wedge read shows the right capability), `tval = 0` and NOT_CAP (the forwarded value is
literally `create_cnull`), and no stall anywhere.

**What is NOT yet explained**, and must not be glossed: the two granules differ. At the wedge
`s0 = 0x82b9f480`, so the zero store is at `0x82b9f360` and the reload at `0x82b9f410` — different
16-byte granules, and different D-cache indices under `paddr[11:4]` (`0x36` vs `0x41`). A
legitimate forward requires a granule match, so this needs an address-comparison defect, not merely
a timing window.

**Instrument note.** The proposed test — arm CSR `0x811` to filter the LDC recorder onto the
subject granule — is **not runnable on resident silicon**: `s07_ldc0_filter_addr_i` does not exist
at `84ed6eafb` (0 occurrences), it belongs to the design that failed to route. The *store*
watchpoint at `cva6.sv:905-906` **is** resident and CSR-`0x811`-armed, which is a different
instrument and can see stores landing on a chosen granule.

## RETRACTED: "two levels wedge" is a RATE (54%), and `q_two` never ran in those boots

An adversarial audit refuted the level-2 finding as I stated it. Verified independently before
retracting.

**1. `q_two.test` was never executed in up21–up24.** Every one of those boots' preflight lists it
as an unused file. My statement that the result was "measured with `q_one`, `q_two` and `qj2`, all
from the same binary" is **false** — only `q_one` and `qj2` ran. The only boot that ever ran
`q_one` and `q_two` from one binary is up14, whose runner voided that arm itself (`SPLB`, no
`SQ: G/enter`). **N for "same binary, q_one vs q_two" = 0.**

**2. I filed the strongest disconfirming observation under "void".** up24 is not void: it is a
clean completion of the two-level join — `records=3 stmt_pass=2 query_fail=1 parse_err=0
completed=1`, same image, same case, same bitstream as up22 which wedged. (I corrected the rate to
3/4 in `04afe159f074`, but the brief I handed the auditor still said "both void".)

**3. Across the whole corpus the level-2 wedge is a coin flip, and the IMAGE predicts it better
than the level count.** Tallying every 2-level arm ever run:

```
sqrt 8/0   sqslt 6/0   sqem 1/7   sqrtw 0/3   sqcc 2/1   sqpad10 2/1   (+14 images, 6/6)
TOTAL: 25 wedged, 21 returned  ->  54%
```

**Twenty-one boots have a two-level join completing on silicon.** "Two levels wedge" is refuted as
a property.

### What survives, and it is still a real effect

`q_one` (one level): **11 returned, 0 wedged**, and three of those ran in slot 2 of 3, so it is not
just "the first arm always survives". Position is controlled the other way too — up17 ran a 2-level
case as TEST 1/2 and it wedged.

If one level wedged at the two-level rate of 54%, P(0 wedges in 11) ≈ **0.0002**. So the level
dependence is real and significant; what is wrong is the word "wedge" as a certainty. The correct
statement is: **one level has never wedged in 11 draws; two levels wedge about half the time, with
strong per-image clustering.** Any single-draw arm on a 2-level case is therefore uninterpretable —
a ladder of N=1 hypotheses would measure the draw, not the hypothesis.

### Three corrections to the fault-site description

* **`+0x8c` is a build-specific label, not an instruction identity.** up22 latched `0x828f4838` =
  fn+**0x9c** on the counter build — the *same instruction* (`cincoffsetimm a4, a4, 0xb0`), shifted
  by injected code. Name the instruction; an offset silently breaks across builds.
* **`+0x8c` is NOT the function's first executable statement.** Three initialised declarations
  precede it (`iRowidReg = 0`, `iReleaseReg = 0`, `Index *pIdx = 0`), emitting `1047ec`–`10480c`.
  My "the fault is on the first statement" was wrong — and the `movc a4, zero` at `104808` that the
  zero-data hypothesis rests on is precisely one of those preceding instructions.
* **"Memory holds the correct capability" overstates.** A GDB read at T+seconds is predicted by
  *both* arms of the fork — a write-buffer entry that converges and drains correctly shows the same
  thing. This folder already withdrew that reasoning once; it must not be re-asserted.

### Instrument hazard fixed

`/tmp/capstone/up21.sh` had been edited in place across every experiment, so by the end it carried
up24's configuration under up21's name — anyone "re-running up21" would have run a different case
on a different binary. Renamed to `run-sqlite-arm.sh`, which is what it actually is. **There is no
surviving script for up21**; its configuration is recoverable only from the log.

## S-12 MATCHES A DEFECT THIS PROJECT ALREADY BOARD-MEASURED — and the fix is a compiler gap

The single most useful thing found this session was already written down, in our own build script,
on 2026-08-05. `build-sqlite-silicon.sh:2278-2288`:

> "At `-O0` `strlen` re-loads its string capability from a stack slot with `ldc` on EVERY
> iteration, and on silicon that **sporadically yields** 1 instead of the true length — stage 13
> returned 15, then 26, then hung across three boots of the same source, where **QEMU returns 36
> every time**. At `-O1` the pointer stays in a register (zero `ldc` in the loop) and stage 13
> returns 36 on silicon, twice. **This is a real wrong-answer defect, not a workaround.**"

and, at `:2266-2271`:

> "The board froze at exactly the **`cincoffsetimm`** of that sequence (image VA 0x14d884,
> ra → sqlite3Strlen30), pc not advancing under stepi with mcause=0."

**Every feature of S-12 is in that description:**

| S-12 | the 2026-08-05 defect |
|---|---|
| `ldc` reloading a capability from a stack slot | same |
| consumed by `cincoffsetimm`, which is where it dies | same, named explicitly |
| sporadic — 54% across 46 arms | sporadic — 15, then 26, then a hang |
| QEMU never reproduces | QEMU returns the right answer every time |
| `-O0` build, every pointer spilled | `-O0`, stated as the cause of the round-trip |

So S-12 is very likely **not a new defect to root-cause from scratch** but an instance of a known,
already-measured class: **at `-O0`, a capability round-tripped through a stack slot sporadically
returns wrong data on silicon.**

**It also explains the level dependence without needing a mechanism specific to loop level 2.**
The SQLite-side analysis (below) shows the two calls are otherwise identical — so two levels simply
means twice the calls and more stack round-trips, i.e. more draws against a per-round-trip failure
probability. One level is 0/11; two levels are 54%. (A strictly independent per-call model does not
fit exactly — p≈0.32 per call would predict ~32% for one level, and 0/11 has probability 0.012
under that — so the count of round-trips per call matters too, not just the call count.)

### Why the fix has not been applied: a defect in OUR backend

The mitigation is `-O1`, which removes the round-trip entirely. The build script records that the
amalgamation cannot go above `-O0` because of **C-17**: `Cannot select: i128 =
CapstoneISD::SELECT_CC`, since `Select_GPRCAP_Using_CC_GPR` is emitted only under `!is64Bit()`.

**C-17 no longer reproduces.** Its recorded reproducer — `char *pick(int n, char *a, char *b)
{ return n == 10 ? a : b; }` — now compiles clean at `-O1` **and** `-O2` on capstone64, as do
five harder select shapes (long compare, pointer-as-condition, struct field condition, nested
select). That blocker has been fixed at some point since the comment was written.

**A different backend limit now blocks it:**

```
fatal error: error in backend: Capstone PureCap: Cannot materialize arbitrary >64-bit
constants as capabilities; capabilities are unforgeable
```

That is the live obstacle to building SQLite at `-O1`, and therefore to removing the stack
round-trips that S-12 appears to be an instance of. **It is compiler/codegen work, which is in the
main session's lane rather than the RTL lane's** — and it is a far more tractable target than
continuing to characterise a sporadic silicon fault through a 6-minute board loop at 54% per draw.

### What the SQLite-side analysis contributed

Independently, the caller window turns out to be nearly empty, which removes a whole class:

* **Exactly one function runs between call #1 and call #2** — `sqlite3VdbeCurrentAddr`, which is
  `return p->nOp;` — plus three integer stores. `sqlite3WhereExplainOneScan` compiles to `0`
  (`SQLITE_OMIT_EXPLAIN`), `sqlite3WhereAddScanStatus` to `((void)d)`, and the auto-index and
  Bloom-filter branches are untaken (no WHERE terms; no ANALYZE). Verified in the disassembly.
* **No malloc, free, or realloc in the window**, so "a realloc moved a buffer" is out.
* **`pWInfo` is bit-identical on both calls** — same caller slot, same instruction; only `iLevel`
  (0→1), `pLevel` (+160) and `notReady` differ.
* **Call #2's frame lands on exactly call #1's bytes**, and the only intervening callee has a
  0x30-byte frame — no overlap with the spill slot at `SP−0x70`.
* Pass-vs-fail heap delta is nil in the way that matters: `WhereInfo` is 1488 vs 1648 bytes, and
  **both round to the same 2048-byte MEMSYS5 bucket**.

---

## One reading REMOVED: a plain ALU write DOES clear a register's capability shadow

**This closes a route that the stale-operand account could have rested on, so it belongs here
rather than in a lane message.**

The question arose from a compiler change, not from S-12: the Capstone backend now reads a
pointer's address with a plain `mv` instead of `lcc rd, rs, 2`, because the cursor query is not
total and traps on NULL. A reading of the RTL suggested that could be silently wrong on silicon —
QEMU's `gen_set_gpr` clears the tag on **every** integer write, whereas here the metadata shadow's
write-enable is gated on `cap_result.valid`, which is 0 for a plain ALU op. If the shadow were
left **stale**, an integer would keep looking like a capability to anything that checks
`cap_type`, and a register's metadata could outlive the value it described — which is exactly the
shape a "the operand carried the wrong metadata" account of S-12 needs.

**It is not stale. Measured in simulation, not argued.**
`capstone-ariane verif/tests/custom/capstone/alu-write-clears-shadow.S`, commit `eb43f5d09`.

`CINCOFFSET` is the detector because its rs2 check *is* the question —
`capstone_flu_unit.anvil:30` raises `UNEXPECTED_OPERAND` when `cap_rs2.metadata.cap_type` is not
`NOT_CAP`. Three arms, ordered with the expected-to-trap one last so a trap could not cost an
answer:

| arm | rs2 | result |
|---|---|---|
| A | never held a capability (`li a5, 8`) | retired, `x12 = 0x80003008` |
| C | held one, then overwritten by a plain `addi` | **retired**, `x13 = 0x80003008` — the answer |
| B | holds one right now (positive control) | **exception: UNEXPECTED_OPERAND** |

**Arm B is why arm C means anything.** Without it, "C did not trap" is equally consistent with a
check that never fires in this configuration; with it, the check is proven able to fire on the
very next instruction pair. Arm A rules out the other direction — that `CINCOFFSET` simply cannot
take an integer rs2.

**What this does and does not remove.** It removes *register-shadow staleness* as a mechanism: a
register that held a capability and was then overwritten by ordinary arithmetic does not carry its
old metadata forward. It says nothing about the **memory** path, where the slot's contents are
what they are, and nothing about the load itself — which is still the unmeasured step.

**Scope note, deliberately narrow:** this was run in bare M-mode simulation, not inside a
capability domain on a monitor-carved stack. Per the standing caveat on directed tests, a clean
simulation of a synthetic sequence is not exoneration of the production path; it is a specific
mechanism ruled out, not the bug.

---

## BOARD, 2026-08-27: `-O1` does NOT route around it — two distinct images, two wedges

**Two boots on the resident `caplifive_s07clear_84ed6eafb.bit`. No reflash.** The control passed
in both, so both boots carry a verdict.

| boot | arm | image | query | outcome |
|---|---|---|---|---|
| 1 | control | `sqctl.dom` `-O0` | `q_one` (1 level) | **returned**, `H/return`, `completed=1`, rc=0 |
| 1 | subject | `sqo1a.dom` `-O1` | `qj2` (2 levels) | `G/enter` + `ENT1`, no return — **WEDGED** |
| 2 | control | `sqctl.dom` `-O0` | `q_one` (1 level) | **returned**, rc=0 |
| 2 | subject | `sqo1b.dom` `-O1` | `qj2` (2 levels) | `G/enter` + `ENT1`, no return — **WEDGED** |

Arms after each wedge are collateral and carry no verdict. The third draw (`sqo1c`) has not run.

**The images are genuinely distinct draws**, not repeated boots of one image: redrawn via
`CAPSTONE_TEXT_PAD` 0/64/128, sha256 verified 4-of-4 unique before staging, and
`sqlite3WhereCodeOneLoopStart` sits at a different address in each.

**Every draw carries the property under test, verified in the artifact:** zero
`ldc a?,0x0(a0)` + `cincoffsetimm ?,?,0xb0` pairs, and the surviving `0xb0` consumer is
`cincoffsetimm a0, s2, 0xb0` — `pWInfo` register-resident in callee-saved `s2`, no stack
round-trip. **So the S-12 fault site is absent from these images and they wedge anyway.**

### What this refutes

**The strong form of the `-O0`-spill account is dead.** "Removing the stack round-trip removes the
wedge" is false: the round-trip is gone at the fault site and two independent draws still wedged.

### What it does NOT establish, and this is the half that matters

**2 of 2 is p = 0.29 at the `-O0` base rate of 54%.** That is not evidence that `-O1` is WORSE, or
even that its rate differs at all. It establishes only that `-O1` is NOT IMMUNE. Six distinct
images would be needed for p ~ 0.01, and four of those draws have not been spent.

### The wedge signature is NOT S-12's

Both wedges read aperture 225 = **`0xd5`**, identically. Against the two signatures already on
file:

    0x80   the six S-12 wedges -- core NOT stalled, only trace_buf_empty
    0x95   dyn unit blocked in get_node_query_validity while the rev-node unit waits on the
           node-table memory read (capstone_dyn_unit.anvil:106-112, capstone_rev_node.anvil:36-41)
    0xd5   MEASURED HERE = 0x95 plus wstore

So these are the **rev-node/dyn-blocked class, not the S-12 class**. Whether that means `-O1`
removed S-12 and exposed a different pre-existing wedge, or that the classes are related, is NOT
settled by two draws.

### RETRACTED BEFORE PUBLICATION: the commit-pc localization

The wedge read `commit pc = 0x82c1c3fc` in both boots, which maps to image VA `0x1c3fc` and
tempted a localization to `sqlite3_result_double`. **That is not supported and the check that
killed it is the point:** the two images hold DIFFERENT INSTRUCTIONS at that address —
`lui a3, 0x9` in `sqlite3_result_double` in one, `sw a1, 0x44(a0)` in `sqlite3_result_blob64` in
the other. An identical PC from two images whose code at that address differs means **the
commit-pc aperture is not tracking the domain**, exactly like the trap latch the driver already
refuses (`mcause 9`, kernel `mepc 0xffffffff800072cc` — ordinary traffic from earlier).

Apertures 224/225/255 were also byte-identical across the two boots, which is consistent with
them describing the WEDGED-SYSTEM state rather than the faulting domain. Read 225 as "which wedge
class", never as "where".

**Next:** the four unspent draws. Until then the honest summary is *`-O1` is not immune, at an
unmeasured rate, with a wedge signature that is not S-12's.*

### Boot 3 closes the confound: `-O0` still gives S-12, `-O1` gives something else

The `0x80` signature on file was measured on `-O0` images built with the **old** compiler; the
`0xd5` above was measured on `-O1` images built with **today's**. Compiler version was therefore a
second variable, and the comparison was not yet single-variable. Boot 3 removes it.

Same boot, same bitstream, same query, same compiler — only the optimisation level differs:

| arm | image | 225 | mcause | tval | commit pc |
|---|---|---|---|---|---|
| control | `-O0`, 1 level | — | — | — | **returned**, rc=0 |
| subject | `-O0`, 2 levels | **`0x80`** | **25** (real capability fault) | **0** | `0x2` |
| — | `-O1`, 2 levels (boots 1-2) | **`0xd5`** | 9 (stale kernel) | stale | frozen |

**So the signature difference is attributable to `-O1`, not to the compiler changes.** And the
S-12 reproducer is INTACT on the current compiler: mcause 25 with `tval = 0` and `commit pc = 2` is
the documented shape — a real capability fault, then the M-1 loop at pc 0 with `mtvec = 0`.

**Aperture 225 decoded** (`cva6.sv:1189-1199`, verified in source, MSB→LSB): `trace_buf_empty`,
`dyn_wait_store_syncer`, `dyn_wait_load_syncer`, `dyn_wait_rev_res`, `dom_switch_busy`,
`stall_issue`, `mem_write_flag`, `mem_wait_flag`.

    0x80  trace_buf_empty ONLY -- NOTHING is waiting. The core has simply stopped committing,
          the shape of an exception stuck at the head of in-order commit.
    0xd5  trace_buf_empty + dyn_wait_store_syncer + dyn_wait_rev_res + stall_issue + mem_wait
          -- THREE wait conditions asserted at once: a unit blocked on responses that never
          arrive. A DYN/rev-node deadlock, not a stall at commit.

**A wedge where nothing is waiting and a wedge where three things are waiting are different
failure modes.** That is decode, not interpretation.

**The `-O0` fault SITE has moved**, which is expected and worth recording so nobody reads the old
address as gospel: `mepc = 0x828f4814` → image VA `0xf4814`, in **`sqlite3WhereEnd`**, at
`lw a1, 0x0(a0)` with `a0 = cincoffsetimm s0, -0x114`. Not the historical
`sqlite3WhereCodeOneLoopStart+0x8c`. Same class, different site — today's codegen differs.
`tval = 0` again, so the operand's cursor was zero at ingestion.

**Running tally: `-O1` 2 wedged of 2 draws; the third (`sqo1c`) has still not run** — boot 3's
`-O0` subject wedged first and took the core. Controls returned in all three boots.

### The new site's mechanism, tested: FLU -> LSU adjacency is NOT the trigger

Boot 3's `-O0` fault is a **producer/consumer pair one instruction apart**, the same shape as the
original site but a different pair:

    original   ldc           a4, 0x0(a0)      DYN producer
               cincoffsetimm a4, a4, 0xb0     FLU consumer, rd == rs1

    boot 3     cincoffsetimm a0, s0, -0x114   FLU producer, rd != rs1
               lw            a1, 0x0(a0)      LSU consumer, IMMEDIATELY next

The `lw` took mcause 25 with `tval = 0`, i.e. it read `a0` as `{cursor 0, NOT_CAP}` — bit-for-bit
`create_cnull` — while the `cincoffsetimm` that wrote `a0` did not itself fault. Both readings are
sound at this site: `mepc` is `pc_commit` latched at the trap (`cva6.sv:1138`), and `tval` carries
the rs1 cursor for capability causes (`ex_stage.sv:490`, `:917,925`).

**`s12-flu-raw.S` covers DYN -> FLU and found HAZARDS = 0. Nothing covered FLU -> LSU.**

`s12-flu-lsu-raw.S` (new) does. Result: **the adjacency is not the trigger.**

    walk proven to have run   2560 lbu, exactly 40960/16
    cycles                    33,677 (warm variant: 373) -- and NOT the 2000013 timeout,
                              so a genuine pass rather than a hang reported as SUCCESS
    exceptions                NONE -- the `lw` did not fault
    a5                        0x80003020, the correct cursor
    lcc selector 1 on a5      1 = NONLIN, a valid capability, NOT 7 = NOT_CAP

The first version of this test ran the load WARM and passed in 373 cycles. That would have been
the void shape its sibling fell into — `s12-flu-raw.S` reported `ldc-pending-cycles = 0` on its
first run because every load hit, so its zero tested nothing. The eviction is what makes this
negative admissible.

**Scope, kept narrow:** bare M-mode, not inside a capability domain on a monitor-carved stack, and
the producer's own operand was not itself pending. So this removes FLU -> LSU adjacency as the
mechanism; it is not exoneration of the pipeline.

**The other candidate was excluded by prior art rather than re-tested.** A LINEAR rs1 with
rd != rs1 nulls rs1 and gives rd the capability (`capstone_flu_unit.anvil:29-53`); a swapped pack
would yield exactly the observed `{0, NOT_CAP}`. But `cincoffset-linear-clear.S` is a passing
regression test asserting precisely that rd is the valid cap and rs1 is the cleared one — and it
cannot explain the original site anyway, where `rd == rs1 == a4` and the LINEAR path does not
apply.

**So both named mechanisms for the new site are now excluded, and the adjacency shape shared by
the two board sites is not sufficient on its own.**

## Rev-node allocation EXCLUDED as the SLT-vs-built-in discriminator — measured, with a control

The RTL lane proposed `rev_node_head` as the one RTL-visible discriminator between the two
workloads: monotonic, no reclamation, one number, already on the debug mux. Measured, it does not
discriminate.

| workload | head | entry carves | runtime allocations |
|---|---|---|---|
| built-in extended (no `--slt`) | **250** | 211 | **39** |
| SLT + `q_one` | **254** | 215 | **39** |

**Runtime allocation is identical.** The SLT harness does not consume more revocation nodes than
the built-in path, so allocation volume is not the difference between an arm that is 0/14 and one
that wedges.

**THE APERTURE HAS A POSITIVE CONTROL FOR THE FIRST TIME, and it is arithmetic rather than a
second instrument:** head minus the domain's static carve count (from `gp-carve-count.py`) must be
the runtime component, and it lands at 39 for both. A reading that did not track carve count would
mean the aperture is not measuring what it claims.

That control retroactively classifies the earlier readings:

    421     ~2 domains x 215 carves = 430          PLAUSIBLE, a real head
    62496   would need ~290 domains of carving     JUNK -- and it appeared BYTE-IDENTICAL
                                                   from two different images, the same
                                                   frozen-aperture signature as commit-pc

**Two instrument facts worth carrying:**

* **Only SLOT 1 yields a halted read.** With `HALT_MUX_READS=1`, the halt succeeds for arm 1 and
  fails with `ActionTimeout` for arm 2, in both boots and in both orderings. The driver correctly
  voids the running reads rather than printing `0xFFFF` as a head — which is exactly the failure
  the sentinel fix was written for. Any head comparison must put its subject in slot 1.
* `0xFFFF` is `REVNODE_SENTINEL`, not a count, and is indistinguishable from an all-ones dead
  aperture. Never read it as a full pool.

### Closed in both directions: a WEDGING two-level run consumed exactly the same

The gap above is filled, and the answer is stronger than expected — the reading came from a run
that **wedged**, so it covers the case that mattered:

| workload | outcome | head | entry carves | runtime allocations |
|---|---|---|---|---|
| built-in extended | returned | 250 | 211 | **39** |
| SLT + `q_one` (1 level) | returned | 254 | 215 | **39** |
| SLT + `qj2` (2 levels) | **WEDGED** | **254** | 215 | **39** |

**Runtime allocation is 39 in all three — wedging and returning alike.** Rev-node consumption is
excluded as a discriminator in BOTH directions: not between harnesses, not between one and two
levels, and not between a wedge and a return.

**Cross-validated from two independent paths in the same boot**, which is what makes a
byte-identical repeat of 254 a measurement rather than the frozen-aperture pattern that has
already caught us twice: the `[s07] after` read reports 254, and the wedge path's raw bytes read
`sw=249 = 0xfe`, `sw=250 = 0x00` = 254. The halt succeeded (slot 1) with no `ActionTimeout`.

**And it says something about WHERE the wedge sits.** The wedging run had done exactly the same
allocation work as the returning ones — 39 runtime nodes, not fewer. So it did not die partway
through setup; it got as far in allocation terms as a run that completed. That is consistent with
the recorded PREPARE-time locus and inconsistent with any account in which the wedge follows from
having done more, or less, capability-lifetime work.
