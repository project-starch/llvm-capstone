# S-12 — what is DEAD, and what each refutation does NOT cover

**Why this file exists.** `00-README.md` is 2,200 lines of narrative in the order things were
learned. That is the right shape for evidence and the wrong shape for the question people
actually arrive with — *"has this already been tried?"* On 2026-08-27 the compiler lane and the
RTL lane independently re-derived the same already-recorded chain **on the same afternoon**, and
three separate prior-art hits landed in one session. This is the index.

**Read the fourth column.** A refutation is a fact about ONE experiment, not a permanent
exclusion. The project rule is *read PAST the root cause* — a fixed issue's folder records what
its fix did not cover, and reading only the headline gave the wrong answer twice. So every row
says what it leaves open.

**Rule for adding a row:** name the evidence (a commit, a test, a `file:line`), not a conclusion.
A row with no evidence pointer is an opinion and will be treated as one.

---

## Dead: image layout and geometry

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| The **function's address** decides it | `TEXT_PAD` build placed at the "curing" address still wedges. Retracted **three times** under "layout decides it" | Address is not the variable. It does NOT say layout is irrelevant to the *rate* — per-image clustering is real and unexplained |
| The **globals region** decides it | `SQLITE_GOFF_OVERRIDE` moved it +64 KiB; still wedges | Only the region's *base*. Cap-table geometry is a separate knob |
| The **cap-table entry count** decides it | 338 entries wedges, 337 completes — no monotone relation | Counts. A specific *index* colliding is untested |
| The **stack budget** decides it | 16 → 24 moved only the stack; no flip in outcome | Stack size. Slot *contents* and frame overlap were addressed separately |
| The **slot address** decides it | `gp6` moved the slot; still wedges | The slot's address. Not its alignment or granule sharing |
| The **`.bss` size** decides it ("32 bytes cures it") | **Superseded**, not refuted: the apparent cure was draw variance. The real figure is a **54% per-draw rate** with per-image clustering | Nothing was measured wrong; the *inference* over-read N. Any future "X cures it" needs n large enough for 54% |

## Dead: the instruction window

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| The **four-instruction shape** is sufficient | Arm 7 reproduces production's exact shape at production spacing and **returns clean** | The shape in isolation. It does not exclude the shape *plus* an untested precondition |
| **Cache residency** (a cold load) is the missing ingredient | Arm 8 = arm 7 with the line evicted first. Returns `0xC12A8000` clean, 245,623 cycles vs arm 7's 1,924 — the walk provably ran, so the load genuinely missed | Miss latency as *modelled here*. Real misses under memory pressure, interrupts and rev-node churn are not the same thing |
| **Instruction spacing** is the variable | Arm 3 added one nop before the consumer; no flip | Spacing at that granularity |

**Standing conclusion for this family:** shape, spacing and miss latency **together** are
insufficient. The trigger needs something the window does not carry — execution history, the
capability's provenance, i-cache/code-volume effects, or a SQLite path unlike the modelled one.

## Dead: mechanisms in the pipeline

| Hypothesis | How it was refuted | What that does NOT cover |
|---|---|---|
| **Tag loss** — a detagged but otherwise correct capability | `decompress_cap_tagged` (`ariane_pkg.sv:766-782`) passes the **cursor through unchanged** on an untagged read, so tval would be NON-zero. Measured tval is **0** | Tag-only mechanisms **at this site**. Says nothing about sites where tval was not captured |
| **LDC move-clear** fired on the subject | Value is NONLIN, confirmed three independent ways; the clear set is keyed to REVOKE (`load_unit.sv:225-226`) | The subject's type as measured. A type *change* mid-flight is not excluded by a static read |
| **Wrong-producer forwarding** | Prevented at issue: `issue_read_operands.sv:1478` gates `issue_ack` on `!stall_waw`; `:1418` defaults all-ones; `:1427-31` clears only when `rd_clobber_gpr[rd]==NONE` | The generic forwarding path. Capstone-specific result packs are separate |
| **Granule row** — the clear reordered after the next store | **Structural, no boot needed:** the clear shares the store-buffer port (`store_unit.sv:449`), both queues are strict FIFO with monotonic pointers, and `load_unit.sv:707-712` holds `valid_o` low until the clear is accepted. One FIFO, program order. All three premises verified at the resident revision | Reordering *within* the write buffer. Does not cover merge-time `ctag` behaviour, which is S-09/S-10 territory |
| **Register-file row** — a stale FLU operand read before the load lands | `s12-flu-raw.S`: window **proven created** (`flu-issues=131`, `ldc-pending-cycles=82`) and `HAZARDS=0`. The generic RAW machinery stalled it every time | DYN→FLU only, in bare M-mode. An earlier run of the same test was VOID at `ldc-pending-cycles=0` because every load hit — the totals are what make this zero admissible |
| **Register shadow staleness** — an ALU write leaves old capability metadata behind | `alu-write-clears-shadow.S` (`capstone-ariane eb43f5d09`): the capability-then-overwritten register **retires** through `CINCOFFSET`, while the still-a-capability arm **traps** as positive control | Register shadow only. Says nothing about the memory path, and it is bare M-mode |
| **FLU → LSU adjacency** — an LSU consumer reading an FLU producer's dest before it lands | `s12-flu-lsu-raw.S` (`capstone-ariane 21842a864`): no exception, correct cursor `0x80003020`, type query reads NONLIN. Made admissible by evicting first — 2560 `lbu`, exactly 40960/16, 33,677 cycles vs 373 warm | Bare M-mode, single iteration, producer's own operand not pending |

## Retracted claims — asserted, then withdrawn

These were **stated as findings** and are wrong. They are listed separately because a retracted
claim propagates further than a refuted hypothesis.

| Claim | Why it fell |
|---|---|
| "**Two levels wedge**" (deterministic) | It is a **54% rate**. `q_two` never ran in up21–24 — preflight listed it unused in all four — and up24 was a clean completion filed as void. Corpus: 25 wedged / 21 returned |
| "`+0x8c` **is the first executable statement**" | Three initialised declarations precede it |
| "**Layout decides it**" | Retracted three times |
| "The **byte-identical body**" and "**NONLIN measured at the fault site**" | Both retracted 2026-08-19, `00-README.md:1366` |
| "**Memory holds the correct capability**" (as an argument about the load) | The shadow-tag half of that evidence was withdrawn as evidence about the load |
| "The last committed instruction is in `sqlite3_result_double`" | Retracted before publication 2026-08-27. Identical `commit pc` from two images holding **different instructions** at that address ⇒ the commit-pc aperture is stale by construction at a wedge |

## Not S-12 at all

| | |
|---|---|
| The `-O1` hang | **[S-13](../S13-o1-dyn-rev-node-hang/)**. `ex_commit.valid = 0` (no exception), aperture 225 `0xd5` (three wait conditions). S-12 is `ex_commit.valid = 1`, `225 = 0x80`, nothing waiting |
| `caplifive_s10fix_80843404c.bit` as an S-10 control | **The filename is wrong.** `80843404c` is a synth-guard tooling commit that PREDATES every S-10/S-10b commit; `store_buffer.sv` hashes differently from the fix. It carries the S-07 fix and its census is clean, but it cannot test S-10 in either direction. The S-10 RTL fix has never been synthesised |

## Still alive

- **The sequence** that produces the S-13 wait state (RTL lane, generated FSM `7572-7790`).
- **Per-image clustering** — real, unexplained, and the reason redraws must be distinct images.
- **The rate** for the extended workload, which contains a two-table join (`sqlite_capstone_domain.c:1439`)
  and passed 3/3 — where a 54% per-draw rate would predict ≈0.10 for that outcome.

---

# Part 2 — mined from git history, 2026-08-27

Everything above was written from the current investigation. This part was recovered by reading
the commit history, because the reasoning on this project lives in commit MESSAGES and had not been
indexed anywhere. **Every hash below was verified to exist with a matching subject, and a sample of
the quotes was checked verbatim against the commit.** Rows are ordered by how likely they are to
stop someone re-deriving a dead end.

## Contradictions between live sources — read these first

| # | The contradiction | Why it matters |
|---|---|---|
| **C1** | `c49ed6041ae8` (2026-08-25) declares the load-syncer lead dead. `64662c9583cb` (2026-08-26) says of that same test: *"THAT TEST NEVER OPENED ITS WINDOW… every LDC HITS… Its zero was uninformative for the MISS case, which is the case SQLite hits."* The re-test then re-establishes the verdict | **Both commits are live and only the second one's residual is correct.** Reading the first alone gives a sound conclusion resting on a void experiment |
| **C2** | An auditor's report says the arm-position confound is unbroken; boot 26 shows the un-probed build wedging at arm 2. Recorded in `8a4003c7ab5f`: *"The report read the retraction without the later refutation"* | Reading `b25c056e84c0` without `0e9c74faf482` re-derives the auditor's error |
| **C3** | `4ecdc350b6a1` files the site under S-07's tag-loss family; `1b52fc0c674d`, later the same day, says *"the family is tag-loss and this measurement says the value was zero… That displaces the tag-loss reading which has framed this investigation from the start"* | The filing and the measurement point different ways, and the folder says so |

## Dead — and not listed in Part 1

| Hypothesis | How it was refuted | What it does NOT cover | Evidence |
|---|---|---|---|
| **Load-syncer / DYN-unit LDC mispair** — a second LDC init landing while the first is outstanding | 96 back-to-back LDCs: init-while-pending = 0 with the counter proven to move. Then forced misses by construction (4096-byte stride, 16 caps against 8 ways) across four configs: 96/112/108/112 inits, **overlap 0 in all four**. Testbench limits tested rather than assumed (`ariane_testharness.sv:517`, `:514-515`) | Verbatim: *"the sim's miss costs ~8 cycles and a real board miss is far longer. Nothing here tests 50+ cycle latency… If the serialisation has a timeout or abort path only a long miss reaches, this suite cannot see it"* | `c49ed6041ae8`, `64662c9583cb` |
| **The whole ACCUMULATION family** — rev-node consumption, cache-set rotation, write-buffer phase | The **single-iteration** build does not return, and is confirmed the same fault rather than assumed (mcause 25, mepc−DBAS = `0x460` = the subject consumer, distinguishable from the arming `lcc` at `0x39c`). Also kills move-clear a second way: *"with one iteration there is no previous clear to read back"* | The commit states a surviving tension, not a residual: the stale-operand account is the only named mechanism that works at one iteration **and** the one the RAW sim says should not happen | `15f12feed573cb` |
| **Six more sufficient causes** — granule count / write-buffer pressure; untagged-capability entry class; load→store→load chain; monitor-carved stack region; the subject's own derivation; revocation-tree depth | 5 arms × 16384 trials, each on a control-passing boot, **each with an in-arm positive control** so these are *"meaningful zeros rather than unproven ones"*. Rev-node depth retired by reading, not a boot | Verbatim: *"What remains untested is scale: the real workload runs about a hundred thousand instructions with a working set that thrashes the caches and takes timer interrupts, where the microbenchmark loop stays resident in four kilobytes"* | `734dbbfa0519`, `86d960b53df4` |
| **S-06, AMO I-4, S-10b, and S-10's `gran_clr`** as the cause | All four killed on static reading, no board time. S-06 **is** in `80843404c`. AMO I-4 is *"the opposite polarity"*. S-10b needs the pair to disagree at `[11:3]`; they share an address. S-10's `gran_clr` needs a second entry with `ctag=0`; a capability store is ONE entry. **Both rested on a granule-sharing plain store that does not exist** — verified per basic block | The narrow survivor is stated: the tag comes from the write buffer or L1, never the store buffer, *"That is the observed polarity."* Also: a first pass used a control-flow-blind linear scan and *"returned a confident WRONG answer"* | `311d5293471c`, `192099a140bf` |
| **Write-buffer DEPTH/contention** | Not refuted — **the clean sweep is VOID.** Counting the buffer's own occupancy gives *"a peak of ONE simultaneous entry across sixty stores and six arms, out of a depth of eight"*. Nine granule STORES ≠ nine CO-RESIDENT entries | *"the depth hypothesis is NOT TESTABLE in this simulation environment as configured"* — needs the memory slowed (`axi_delayer` is already vendored) or the board | `capstone-ariane 0a45ec2018e2` voided by `4c4224afbd6d` |
| **Four bare-metal directed sims** reproducing the window | All clean. Weighting stated in the commit: two carry **proven-firing** detectors, two do not *"and should be weighted lower"* | Fidelity gaps enumerated in the test headers: M-mode not a domain, register-resident not cap-table, `.data` frame not monitor-carved stack, cold cache | `capstone-ariane b1afedb37696`, `2d0c26b2723a`, `1594ca6a0b34`, `68232f84a940` |

## Retracted — and not listed in Part 1

| Claim | Why it fell | Evidence |
|---|---|---|
| "**The SQLite wedge is a NULL dereference in software**" (filed RESOLVED) | Retracted for want of a fired instrument — *"A zero from an unfired instrument reads exactly like a finding"* — then **partially re-supported** once a `0xBEEF` control fired. Carry the hedge: two mechanisms remain and tval alone does not separate them. A second error in the same chain: *"'25 not 29, therefore tag not bounds' excluded nothing, because CINCOFFSETIMM has no bounds arm and 29 was never reachable"* | `a223ac4fb264`, `b25c056e84c0`, `1b52fc0c674d` |
| "**Arm position is a perfect confound, 5 for 5**" | Broken in both directions by control-passing boots 26/28. *"the previous commit's retraction is itself partly withdrawn"* | `b25c056e84c0` → `0e9c74faf482` |
| "**Writeback-port displacement**" as the localization | Switch 204 reads `0x00` at the wedge on all six (later eight) boots with the selftest firing each time — a controlled negative | `e7816935b34b` → `1d046a5d013c` → `a1036c56709b` |
| "**The reload of that same slot produced zero**" | An auditor found the skipped instruction: `movc a4, zero` at `+0x80`, so a4's prior value is exactly `{cursor 0, NOT_CAP}`. *"What is established is that the FLU received {cursor 0, NOT_CAP}, not that the load returned zero"* | `98b17ce22de7` |
| "**Memory is intact, so the fault is in delivery**" (the inference, distinct from Part 1's row about the evidence) | *"That is only true of a PERSISTENT loss."* Every documented memory-path defect here is transient and self-heals long before a debugger read. So an intact granule at T+seconds *"is predicted by both arms of the fork and separates nothing"* | `d4c78aa6d4b5` → `037a9eef96fe` |
| "**`pad600` proves a store-to-load drain-latency window**" | Delay-dependence stands; the mechanism does not — *"DRAM refresh phase, a periodic interrupt, or another AXI master's traffic"* scale with a 619-instruction gap equally well. Threshold bracketed only as 10 < T ≤ 600 | `c56679fb175e` → `e7d92b488d08` |
| "**The S-07 fix renders the S-10 forwarding defect unreachable**" | *"the stall is an ALLOCATION-time check between two write-buffer ENTRIES. The residual needs only ONE entry"*. Matched pair: 8 traps/16 legs vs 16/16 | `capstone-ariane 6175ea654235` |
| "**S-07 is silicon-validated**" | Downgraded: `P(3 clean | defect live) = 0.875³ = 0.670`, *"nearly uninformative"*; and the pre-registered WNS criterion *"came back negative and was never applied"* | `ISSUES.md:321-330` |

## Dead: the LDC move-clear family (2026-08-27) — all four, each with the clear PROVEN to fire

This family was the strongest surviving one, because it is the only mechanism that produces the
observed operand *exactly*: the clear payload is `store_unit.sv:462-469` driving data 0, user 0,
ctag 0, which is bit-for-bit `create_cnull()` and therefore bit-for-bit `{cursor 0, NOT_CAP}`,
`tval = 0` included. All four routes are now closed.

| Hypothesis | How it was refuted | What that does NOT cover | Evidence |
|---|---|---|---|
| The **value's TYPE** gates the fault, and every earlier sim was blind because it used NONLIN | All six types run through the real window. Clear demonstrably FIRED in all five clear-set arms (granule zeroed) and demonstrably did NOT in the NONLIN control; positive control prints the board's exact NOT_CAP/0 signature. **In all five the load still returned a correct tagged capability** | The window at bare-metal fidelity. Says nothing about a domain after `capenter` under real pressure | `capstone-ariane 7fb91b5c7`, `s12-value-type-sweep.S` |
| The reload **races the clear IT triggers** | Structural: `load_unit.sv:706-712` holds `valid_o` off until the clear issues and delivers a **SNAPSHOT** (`:846-851`), not a re-read of the post-clear granule. Confirmed independently by the sweep above | Only self-observation. A clear from a *different, earlier* LDC is a separate question | `load_unit.sv:706-712`, `:846-851` |
| The clear **races the NEXT store** to the same granule under write-buffer pressure | Matched pair, 16 iterations each: LINEAR arm's granule zeroes every time (clear fires), NONLIN control's survives (clear correctly absent), and every following load returns a correct capability — so each store landed after the previous clear | 16 iterations of bare-metal pressure. Not SQLite's after millions of instructions. Read-back sits between load and next store, so ordering is inferred from the following iteration rather than observed | `capstone-ariane d835e0c00`, `s12-linear-clear.S` |
| A **clear-set capability in a DOUBLE-LOADED slot** — silicon-fatal, QEMU-invisible | Whole-program address-keyed QEMU sweep over 16-byte granules, modelling the RTL condition: **0 hits on all four arms** (qj2, q_two, q_one, built-in). The domain makes only 3 clear-set stores at all, all `SEALEDRET` from the entry glue, none re-loaded. Every hit is monitor-side and the counts run BACKWARDS — the silicon-passing built-in arm has 24 against the wedging arms' 18 | QEMU-visible paths. Three positive controls fired (forced-NONLIN 144,425 hits; type filter a proper subset; disarm guard 38/85) and the blind-spot meter reads 0, but silicon-side divergence is by definition outside it | `/tmp/capstone/s12-slot/RESULTS.md`, parent `aec276114ffd` |

**And the premise the whole family rested on is itself dead:** the stored value is **NONLIN**,
measured 16/16 at the exact `stc` pc with a positive control, and independently the slot is loaded
**three times per call**, so a clear-set type there would wedge one-level plans — which never wedge
in 11 draws. The old "the stored value is REVOKE-typed" section was retracted on **2026-08-25** and
**the retraction never reached this folder until 2026-08-27**, which is why the folder spent two
days contradicting the state doc. *A retraction recorded in the state doc but not in the artifact
folder is not a retraction.*

## Dead: register, forwarding and syncer paths (2026-08-27) — refuted at 10x the previous occupancy

Three of these were already "refuted" on occupancy of 2, 8, and *never ran at all*. Those were
unproven zeros; these are refutations.

| Hypothesis | How it was refuted | What that does NOT cover | Evidence |
|---|---|---|---|
| **Load-syncer mispair** — a second LDC's `init` overwrites a pending `cap_trans_id`, diverting the first response and substituting a whole `fat_cap_t`. *The only shape proposed that yields BOTH `tval = 0` and NOT_CAP from one event* | **192 inits** under designed maximum pressure (8 independent cache-missing LDCs after an eviction sweep), `init-while-pending` = **0**, against 7-8 inits in ordinary tests. Structural reason: `func LDC` blocks on `recv cap_load_ri.res` | Bare metal. If the DYN thread's blocking behaviour differs inside a domain, this does not transfer | `capstone-ariane 7fb91b5c7`, `s12-ldc-pressure.S` |
| **Wrong-producer forwarding** — the fixed-priority arbiter hands a consumer an older live entry's result | Occupancy raised from 2 to **5 of 8** with capability producers in the exact `movc → ldc → cincoffsetimm` shape, behind a cold-miss shadow that pins commit; **duplicate-live-rd = 0**. Structural reason: `gen_check_waw_dependencies` stalls issue while a clobberer is live | Occupancy 5, not 8. The prior refutation used a **scalar-only** test at occupancy 2 and was much weaker than it read | `capstone-ariane 36eed41f3`, `s12-cap-waw-pressure.S` |
| **Stale-regfile read** — the consumer bypasses forwarding and reads the regfile before commit | Same run: window (b) occupancy **8 → 80** with **64 real consumers**, `NO-FORWARD` = 0; window (a) 365 FLU issues / 560 LDC-pending cycles, window (c) 1605 — all zero | Its own author flagged occupancy 8 as bounding the rate "only loosely". At 80 it no longer does, but this is still bare metal | same run, `RAW-DBG2` counters |
| **S-10b granule hazard** — a missed `[11:3]` stall gives a stale read | Structural and stronger than the store list: S-10b can only stale the granule half the load does **not** present, so it predicts `tval` = the stored cursor. Measured `tval` is **0**. (`load_unit.sv:316` — a load presents ONE offset, its own base vaddr) | **The STALL hazard only.** S-10 itself (`wt_dcache_mem.sv:280`, still word-granular), S-07, and write-buffer forwarding are NOT cleared by this | auditor-verified; `store_buffer.sv:279,287,293` |

**Scope warning for this whole family:** `git diff 84ed6eafb HEAD -- core/` shows the only
differences in `issue_read_operands.sv` are inside `` `ifndef SYNTHESIS `` — sim-only — so these
results DO transfer to the resident bitstream. That was checked rather than assumed; the opposite
would have voided every one of them.

## Dead: the STC-as-forwarding-producer escape (2026-08-27) — the survivor of a 30-hypothesis brainstorm

Generated by a six-lens adversarial brainstorm in which 29 of 30 hypotheses were refuted; this was
the only survivor, and it was the strongest candidate this investigation has had, because it is the
only mechanism besides the clear payload that produces `tval = 0` **without any memory step at all**.

**All four legs are real and verified at source** — this is not refuted because the reasoning was
wrong:

| leg | evidence |
|---|---|
| `stc a4, 0(a5)` makes a4 the STC's **architectural `rd`** | `decoder.sv:1313` — `instruction_o.rd[4:0] = instr.rtype.rs2` |
| A store-buffer-full stall on an STC holds `we_gpr_o=1, waddr=a4, ack=0` for the WHOLE stall | `commit_stage.sv:331-347` clears **only** `commit_ack_o[0]`; `we_gpr_o[0]` set at `:323` is never retracted |
| That clears the consumer's WAW stall | `issue_read_operands.sv:1585` — a **second** clearing clause keyed on `rs1`, distinct from the `rd` one |
| A written-back STC entry is a forwarding candidate; an unproduced LDC is not | `:719-726` — candidacy needs `still_issued & sbe.valid` |

So the consumer could in principle take a4 from the STC — `{cursor 0, NOT_CAP}` — and raise with
`tval = 0`.

**IT DOES NOT HAPPEN, and the reason is structural.** `s12-stc-producer.S` replays the window
INCLUDING the `stc`, under store-buffer back-pressure:

    S12-ESC: tick 7250  escape=130  hazard=0
    RAW-DBG: tick 7250  ldc-pending-cycles=217
    SB-DBG:  tick 7250  duplicate-live-rd cycles = 0   max live entries = 5

**The precondition fired 130 times**, so this is a refutation and not an unproven zero, and
`ldc-pending-cycles = 217` shows the LDCs genuinely did stay unproduced. The decisive number is
`duplicate-live-rd = 0`: **the STC and the LDC are never simultaneously live.** CVA6 issues in
order, so the consumer can only issue after the LDC issues; the LDC also writes a4, so it is held by
the `rd`-keyed `stall_waw` while the STC claims a4. By the time the LDC can issue, the STC has
ceased to be a candidate. The escape clause the mechanism relies on clears `stall_waw_rs1` — the
CONSUMER's path — while the PRODUCER ordering closes the window first.

**What this does NOT cover.** Bare metal, not a domain after `capenter`. `ldc-pending-cycles` is 217
of ~7300, so the LDC is unproduced only ~3% of the time here, against SQLite's far colder cache —
if a longer LDC latency could hold the window open past the STC's retirement, this would need
re-testing. And it exposed a real gap in a neighbouring row: **`s12-cap-waw-pressure.S` used
`movc → ldc → consumer`, omitting the `stc`**, so its `duplicate-live-rd = 0` was never evidence
about any STC-producer mechanism. That row's scope is narrower than it read.

## Instruments known to be broken — do not build on their output

| Instrument | Status |
|---|---|
| **Switch 208, S-07 tag history** | Structurally unusable: one-shot, no clear, and boot software consumes the slot before any arm runs (*"the pre-run baseline already reads 0xb8 with ldc0_valid set"*). Recorded granules were neither the subject's. `2e37646fb54a` |
| **The wrong-producer-forwarding checker** | The conclusion survives structurally, but *"the checker counts a condition the hardware cannot enter. A positive control for it is unsatisfiable, a future zero from it is not evidence"* — so the capability-pair re-run two lanes wanted is unnecessary. `450be8638f88` |
| **The S-12 repro's arm 4** (and arm 6) | Every measurement void — arm 4 matched no `#if` branch, so no store was emitted and the reload read zero BSS. A `S12_SLOT_WRITTEN` build guard now `#error`s this, negative-tested both ways. `d9ccb82438fd` |
| **The VDBE clamp ladder** | PROVISIONAL — *"a wedged arm prints no ops=/lastop=, so 'clamp 8 wedges' is indistinguishable from 'the clamp value never arrived'"*. `4b348e348d2a` |
| **`s07evict`** | VOID, not negative: assumed 64-byte lines against a real 16-byte geometry, so *"Its eviction never happened"* |
| **Slot 3+ for large SQLite domains** | The monitor's `split_out_cap` spins (`SPLB`). Measured 2026-08-27: the same image returned at slot 1 and, at slot 3, was created (`A/dom-ok`) but never entered (`G/enter` = 0) with an `SPLB` tag. **Only 2 big domains per boot carry a verdict** |
