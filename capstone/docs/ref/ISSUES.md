# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure,
**S-n** = **unattributed** — reproducible, but origin NOT yet established (may be RTL, compiler or
software). An S-n is promoted to R-n/C-n only when the origin is demonstrated, never on suspicion.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-09-09 — the registry was split on this date: this file lists only what is still OPEN; resolved and retracted entries are in [`ISSUES-ARCHIVE.md`](ISSUES-ARCHIVE.md), verbatim and by ID.

---

## S-04 — SQLite returns SQLITE_NOMEM from `sqlite3_open` on silicon · `WORKED AROUND 2026-08-10 with NO second defect in play: memcpy alone at -O0 via BEEBS_MEMCPY_OPTNONE. Board-confirmed by a matched pair. Underlying silicon defect still OPEN.`

> **Sweep 2026-09-05, boot sw22 (19:21) — the blamed -O1 memcpy form does NOT reproduce S-04 on `5097eb166`.**
> `sqm0o1` (-O1 SQLite silicon domain, `SQLITE_MEMCPY_OPTNONE=0`, image `b4b453f90cd7edb4`, host
> `a1895d35f768b5d0`, control `k800 = 4` first) on `q_two.test`: `A/dom-ok` … `G/enter`, `H/return`,
> `SLT-SUMMARY records=2 stmt_pass=1 stmt_fail=0 query_pass=1 query_fail=0 oom=0 completed=1`. `sqlite3_open`
> succeeded and the query passed with exactly the memcpy form this entry blames (seven `sb` from the -O1
> tail loop, `a0` used directly). N = 1, one boot; the header stays WORKED AROUND until the lead moves the
> silicon default, but the defect this entry describes is not present on the s12fix bitstream in this
> draw — consistent with S-04 having been another face of the store-path defects fixed in RTL since
> 2026-08-10 (S-07/S-12 family), which is the hypothesis to test with a second draw and the stage-164
> bit read as the entry describes. Row: `board-results/2026-09-05.tsv` sw22.
>
> **Sweep 2026-09-05 (afternoon) — -O0 SQLite domains pass on 5097eb166 with and without the attribute; the blamed -O1 form was built and QEMU-verified, its board boot is pending (now done: sw22 above); header unchanged.** Boots sw14/sw15 (control k800 = 4): `SQLITE_MEMCPY_OPTNONE=1` (`sqm1` 1ff3686fe7763f48) and `=0` (`sqm0` 0a0489454fd63371) -O0 domains both `SQ: G/enter`, `SQ: H/return`, `SLT-SUMMARY records=2 stmt_pass=1 query_pass=1 oom=0 completed=1` on q_two.test — expected, since at -O0 the attribute changes nothing and both carry the working memcpy form; a first sweep line claiming NOT REPRODUCED on that pair was retracted the same afternoon (board lane's check, dev b5aa78aff7bb). The arm that carries the form this entry blames — the -O1 SQLite silicon domain with `SQLITE_MEMCPY_OPTNONE=0` (`sqm0o1`, sha b4b453f90cd7edb4, differs from its optnone-ON twin `sqbase` 680ebe68987badce which passed as sw18) — is built, passes q_two under QEMU (as S-04 always did), and is staged in the board image; its boot (sw20) lost the console mid-run and the rerun (sw21) could not connect: the console host stopped resolving from this machine at ~16:45 (DNS/tunnel, not the board). Boot it when the console is back: `SLT_SET="sqm0o1.dom sqlite_host.user q_two.test"` bake, then `board-slt2.sh <id> sqm0o1 q_two.test`; SQLITE_NOMEM at open = S-04 present, records=2 pass = not reproduced on this bitstream.

### RESOLUTION 2026-08-10 — per-FUNCTION opt scoping, confirmed by a matched pair on silicon

The two string primitives SQLite depends on are broken at OPPOSITE optimisation levels, and they
live in the same file, so one file-wide `-O` flag forced a choice between them:

| | memcpy | strlen |
|---|---|---|
| `-O0` | works | **WRONG on silicon** (re-loads the string cap with `ldc` every iteration; returned 15, then 26, then hung, vs 36 on QEMU) |
| `-O1` | **WRONG on silicon** (S-04) | works (zero `ldc` in the loop) |

`SQLITE_SUPPORT_OPT_LEVEL=-O0` was therefore a trade, not a fix, and every result taken under it
was taken under a second known defect. `BEEBS_MEMCPY_OPTNONE=1` (default on, set by
`build-sqlite-silicon.sh`) applies `__attribute__((optnone))` to **memcpy only**, so the build has
NEITHER defect for the first time.

**Verified on the artifact, not assumed.** In the linked domain, `strlen`/`strcmp`/`strcpy`/
`memset`/`memmove` are byte-identical to the plain `-O1` build; only `memcpy` changes, to the form
that spills the destination capability at entry and reloads it with `ldc` before each `sb`.

**Board evidence — a matched pair in ONE boot, control green (`k800` = 4):**

| arm | build | stage 164 | bit 2 (`memcpy does not stick`) |
|---|---|---|---|
| `sm0.dom:164` | `-O1` memcpy | `0x74` | **SET** |
| `sm.dom:164` | `optnone` memcpy | `0x70` | **CLEAR** |

The two images differ in **memcpy and nothing else** (per-symbol encoding comparison). One bit
apart, and it is exactly the memcpy bit. The failing arm is also the positive control: it proves
the probe can report failure, so the `0x70` is a real negative and not a dead test. `0x70`
reproduced across two separate boots. `sm.dom:163` went `0xbf` -> `0x00` (the stored key is no
longer zero) and `sm.dom:160` went `0x15` -> `0x00` (no step of `openDatabase` trips).

**This is a WORKAROUND, not a fix — the -O1 code is CORRECT.** Disassembly of the linked domain
(`memcpy` at `0x14ca1c`) shows that for the failing case (n=7, dst and src both 16-byte aligned)
the `-O1` form branches over the head loop (`beqz a5`), does not enter the capability loop
(`bgeu a2, a4` with a4=16 > n=7), and issues seven `sb` stores from the tail loop at `+0x9c`. The
stores are ISSUED and do not stick. **So "the `-O1` byte tail-loop is skipped" is REFUTED** — that
was recorded here as the remaining codegen avenue and it was based on a false premise. The only
difference between the working and failing forms is which capability register holds the
destination base: `-O1` uses the incoming argument `a0` directly, the working form reloads it from
a stack slot. The underlying silicon defect is untouched and still needs reporting.



**The cause is `memcpy`, not the allocator.** A 7-byte `memcpy` into freshly zeroed memory leaves
the destination ALL ZERO, so `findCollSeqEntry`'s key copy never lands, `sqlite3HashFind` cannot
match, the create path runs twice, `sqlite3HashInsert` reports the key present, and
`sqlite3OomFault` fires (132502). The `assert(pDel==0 || pDel==pColl)` that would have caught it
is compiled out under NDEBUG, which is why real data loss surfaced as `SQLITE_NOMEM`.

Narrowed by staged board runs, each one variable:

| stage | result | establishes |
|---|---|---|
| 160 | `0x15` | first failing step is 5, `createCollation(BINARY, UTF16BE)` |
| 161 | `0x11` | `HashFind` misses although the table holds 1 element; SOURCE string intact |
| 162 | `0x5b` | hash AND compare mismatch, `strHash` deterministic, stored COPY differs |
| 163 | `0xbf` | the stored key is ENTIRELY ZERO |
| 164 | `0x74` | destination pointer CORRECT; an explicit byte LOOP writes all 7 bytes; **`memcpy` does not** |
| 165 | `0x00` | both `dest & 15` and `src & 15` are 0 -- the aligned path |
| 166 | `0x05` | poison bytes 7..15 UNTOUCHED -- memcpy stays in bounds; its stores simply do not stick |

Stage 166 also **refuted** the intermediate theory that memcpy wrongly took its 16-byte
`ldc`/`stc` capability block path: it does not overrun.

**Workaround, one env var, no code change:** `SQLITE_SUPPORT_OPT_LEVEL=-O0`. The support objects
(the string primitives, `build-sqlite-silicon.sh:739`) default to `-O1`; at `-O0` memcpy becomes
142 instructions instead of 49 and the copy sticks. With it, the full unclamped SQLite on silicon
progresses from `stage=open rc=7` to **`stage=step rc=21`** -- it now configures, initialises,
OPENS the database and prepares statements.

**THIS WORKAROUND HAS A KNOWN COST -- do not adopt it silently.** `build-sqlite-silicon.sh:710-733`
records that the support objects were moved TO `-O1` because at `-O0` `strlen` re-loads its string
capability from a stack slot every iteration and on silicon sporadically returns the wrong length
(stage 13 returned 15, then 26, then hung, where QEMU returns 36 every time). So `-O0` trades one
silicon defect for another, and the new `rc=21` at `sqlite3_step` may BE that defect resurfacing.
Treat `-O1`-vs-`-O0` as two different broken configurations, not as a fix.

## S-10 / S-10b — a capability survives the store that destroys it, and a store's high word reads back stale · `S-07 and S-10 (route 1) ARE in the resident bitstream (2026-09-05); S-10b MEASURED STILL PRESENT there, and its fix is unsynthesizable`

> **STATUS LINE CORRECTED 2026-09-05.** The header used to say *"none synthesised into a flashed
> bitstream yet"*. That is **stale**: checked by content —
> `git log e1b3db6ba..5097eb166 -- core/cache_subsystem/` — the resident bitstream's lineage
> contains **`5c5f4e3a7` "S-07 FIX: forbid granule co-residency in the write buffer"** and
> **`4fee13b2d` "S-10 FIX: works in simulation, and costs a combinational loop"**, plus the S-07
> probe and audit commits. The sweep's RTL-sim pass at `5097eb166` reports the S-07/S-10 residual
> **not observed** (16 legs trapped + control), consistent with the fixes being present.
>
> Same error class as the R-20 alert retracted earlier today — a status written from what was
> *believed* about a lineage rather than read from it — caught this time before it produced a
> wrong board verdict, by the check that should always have been first.
>
> **Two things this does NOT establish:** `4fee13b2d`'s own message says *"NOT ready to merge"*
> (a combinational loop), so it is in the bitstream by lineage but its synthesis cost is not
> recorded here; and **S-10b** — the granule-granular load/store hazard — HAS a verdict as of
> 2026-09-05 (see "S-10b measured on the flashed bitstream" below): its tests were located on
> `origin/s10-merge-candidate` and it is **still present**. The paragraph below is retained as
> the state of knowledge before that measurement. Until they were found, S-10b was *unknown*, not
> *fixed*.

**Behaviour.** Software destroys authority by overwriting it (`memset`, `bzero`). A capability
load can miss that store and return the capability **intact and dereferenceable**. Separately, a
plain load of an `STC`'s HIGH word can read memory the `STC` has not landed in yet.

**One root cause, three structures.** The capability tag and the capability itself are per-16-byte
**GRANULE** properties; three different places checked them at 64-bit **WORD** granularity. An
`LDC` is granule-aligned, so it always presents word 0 and never matches a store at word 1.

| | structure | the check | status |
|---|---|---|---|
| S-07 | write-buffer **allocation** | `gran_conflict`, `wt_dcache_wbuffer.sv` | fixed in sim; **"silicon-validated" DOWNGRADED — see below** |
| S-10 | write-buffer **tag lookup** | `wbuffer_hit_oh`, `wt_dcache_mem.sv:287` | fixed; **IS in the flashed bitstream** (`5097eb166`) — see note below |
| S-10b | **store-buffer** hazard | `page_offset_matches_o`, `store_buffer.sv:309/317/323` | data route fixed; **tag route OPEN and measured LIVE on the flashed bitstream** |

**S-10 IS in the flashed RTL — the "not yet synthesised" status was stale.** Established
2026-09-05 by ancestry, not by content match: the S-10 merge commits `3d3ed1502` and `4fee13b2d`
are both `git merge-base --is-ancestor` of the flashed `5097eb166`, as are the S-07 fix
`5c5f4e3a7` and the S-12 fix `b9dd83249`. Confirmed behaviourally in the same pass by the
compiler lane — `s07-wbuf-forward-residual` and its `-ctl` report the residual NOT observed at
`5097eb166`, 16 legs trapped plus the positive control.

**S-10b measured on the flashed bitstream 2026-09-05 — STILL PRESENT.** Run by the compiler
lane in a detached worktree at `5097eb166` (both tests read from `origin/s10-merge-candidate`,
testlist entries added in the worktree only; nothing in the RTL lane's checkout touched), and
the numbers re-read here from its logs rather than taken on report:

| test | exceptions | verdict | cycles | reading |
|---|---|---|---|---|
| `s10b-storebuf-primed` | **1** (the control) | SUCCESS | 756 | **0 traps / 8 legs — the defect** |
| `s10b-storebuf-residual` | 9 (control + 8 legs) | SUCCESS | 980 | 8 / 8 — condition never created, uninformative |

**Polarity is inverted, as with S-10:** a trap is the CORRECT outcome — it means the tag was
cleared — and its ABSENCE is the defect. Both runs are genuine `RVTEST_PASS`es at their own
cycle counts, not timeout `SUCCESS`es.

**The control that makes the primed row a reading rather than a blind spot:** both runs retire
the SAME number of capability-opcode instructions (40). So the primed run's zero leg-traps is
not "the legs did not execute" — it is the same work, not trapping. The positive control fired
in both (one `UNEXPECTED_OPERAND` at cycle 403), and in the primed run it is the *only* handler
entry in the trace.

**WARNING — `s10b-storebuf-primed.S` HAS A STALE HEADER THAT CONTRADICTS ITS OWN RESULT, and it
is the boilerplate it inherited from the residual variant.** Lines 31-38 say in block capitals
"THIS TEST DOES NOT CURRENTLY CREATE ITS CONDITION" and "A SUCCESS row for this test in a sweep
therefore means 'the condition was never created'". That describes the UNPRIMED test. The primed
variant's own added section (lines 121-133) supersedes it and explains why: the extra
`ld x0, OFF(s1)` primes the line into L1, so the `LDC` hits and never enters the miss unit, whose
`wt_dcache_missunit.sv:242` granule-granularity collision check is what stalls the unprimed load
long enough for the scrub to reach the write buffer and trap. Read top-down, the header produces
the exact opposite of the correct conclusion. Anyone citing either test must read past line 120.

**This does NOT extend to S-10b, and the two findings compound.** `c867dfcbb` is
unsynthesizable — DRC LUTLP-1, a 69-LUT combinatorial loop, bitgen never ran. So S-10b is
present on the resident bitstream (measured above) *and* its only fix cannot currently be built
into one. Those are the two halves of the same problem, not independent statuses. Its directed
tests (`s10b-storebuf-primed.S`, `s10b-storebuf-residual.S`) live on `origin/s10-merge-candidate`
and `origin/s10b-fix`, not on any flashed branch, and characterise RTL that cannot currently be
built into a bitstream.

**Repro — S-10 (write-buffer route).** `verif/tests/custom/capstone/s07-wbuf-forward-residual.S`
with its matched control. **Polarity is inverted**: a trap is the CORRECT outcome and its absence
is the defect. Simulation 8 of 16 legs handed a live capability before the fix, 16 of 16 correct
after. Silicon, pre-fix, on `caplifive_s07fix.bit`: **3837 of 3840**.

> **CORRECTION (2026-08-21), from the ship audit.** Two things above need qualifying.
>
> **1. The 8-of-16 figure is the SHIPPED tree, and the closer is not in it.** Read straight off
> the committed sweeps, one row per tree:
>
> | tree | test | control |
> |---|---|---|
> | `s07-strip.txt` = `f231b5af0`, S-07 only — **what is on the board** | **9** | 17 |
> | `s10-sweep.txt` (+ S-10) | **17** | 17 |
> | `s10b-sweep.txt` (+ S-10b) | **17** | 17 |
>
> So the before/after across S-10 is real and the control is pinned — that part stands. But it
> means **8 of 16 legs are live on the bitstream currently flashed**, handing back a
> dereferenceable capability over memory the program has already scrubbed. The fix that closes
> them (S-10) is synthesis-proven — `80843404c` reached `write_bitstream` with exit 0 — and is
> **not shipped**. S-10b, the other candidate, is not shippable at all (`DRC LUTLP-1`, see
> `history/21-08-2026_01-15-00_s10b-is-not-synthesizable.md`).
>
> **S-10 alone closes it — MEASURED 2026-08-21, no longer an inference.** The earlier caveat was
> that no committed sweep isolates S-10 from S-10b. That simulation has now been run, in a
> worktree at `4fee13b2d` (S-10 only):
>
> ```
> s07-wbuf-forward-residual        SUCCESS  9371 cyc   17 exceptions   (shipped tree: 9)
> s07-wbuf-forward-residual-ctl    SUCCESS 26361 cyc   17 exceptions   (control, pinned)
> s07-wbuf-tag-reorder             SUCCESS  9138 cyc    1 exception
> ```
>
> 17 is the maximum — 16 legs plus the baseline — so every leg now traps and the window is shut.
> The control is pinned at 17 exactly as on the shipped tree, so the single variable is S-10.
> The third row is the **model-identity control**, and it is what makes the other two readable:
> `s07-wbuf-tag-reorder` returns the post-fix signature (1 exception at 9138 cycles) rather than
> the pre-fix one (4 at 9150), so the worktree really was built at this revision and did not
> silently reuse another model. **S-10b is not needed to close this**, which matters because
> S-10b cannot be built at all.
>
> **2. "silicon-validated" for S-07 is not supported as worded**, on two independent grounds.
> The pre-fix wedge rate was k = 2/16 = 0.125, so P(3 clean reps | defect still live) = 0.875^3 =
> **0.670** — a likelihood ratio of about **1.5:1**, nearly uninformative. And
> `timing_summary_routed.rpt:162` for that exact tree reports `WNS -10.629`, `TNS -438671.250`,
> `96727 of 246476 failing endpoints`, `Timing constraints are not met.` — while `RATE-RULE.md`
> had **pre-registered** the criterion *"WNS non-negative makes the S-07 validation
> unconditional; negative means everything measured on this bitstream needs re-reading."* The
> criterion came back negative and was never applied. The correctness weight sits on the
> simulation matched pairs; say that instead.

**Repro — S-10b (data route).** Not a new test —
`verif/tests/custom/capstone/untagged-ldc-stc-fixup.S`, which has been silently working around it
for as long as it has existed:

```
pc 0x800000d0   ld x29, 40(a1)      the high word of the granule an STC wrote at offset 32
  before:  x29 = 0x0000000000000000     STALE
  after :  x29 = 0xfedcba9876543210     what the STC actually wrote
```

The test's own source reads `ld t4, 40(a1)  # high half of the DESTINATION as it actually landed`
followed by a repair branch. **That repair branch is now dead code** — the trace is one
instruction shorter. Anyone touching that test should know it no longer exercises what it was
written for.

**Repro — S-10b (TAG route). REPRODUCED 2026-08-20 after three failed attempts.**
`verif/tests/custom/capstone/s10b-storebuf-primed.S`, a matched pair with
`s10b-storebuf-residual.S` differing by **one instruction**:

| | pre-fix (`c3ca1b270`) | with S-10b |
|---|---|---|
| `s10b-storebuf-primed` | **0 of 8 legs trapped** — eight live capabilities over scrubbed memory | 8 of 8 |
| `s10b-storebuf-residual` | 8 of 8 | 8 of 8 |

**Why the first two attempts failed, which is the reusable part.** An unprimed `LDC` is a first
touch and MISSES, so it routes through the miss unit — and `wt_dcache_missunit.sv:242` compares an
incoming miss against in-flight write TXs at `[PLEN-1:DCACHE_OFFSET_WIDTH]` = `[PLEN-1:4]`, i.e.
**granule** granularity, stalling the read outright (`:495`). By the time it is released the scrub
has allocated into the write buffer and S-10's granule tag lookup catches it. The single added
`ld` primes the line so the `LDC` **hits L1 and never enters the miss unit**, and no collision
check can fire.

The same mechanism explains why the DATA route reproduced without priming: there, the `STC` is
still in the *store* buffer when the load issues, so **no write TX exists** and the collision
check cannot fire either way.

`s10b-storebuf-residual.S` is retained as the *negative* arm and is labelled in its own header as
a test that does not create its condition — a `SUCCESS` row for it means the condition was never
created, **not** that the route is closed.

**Cost of the S-10b fix, measured.** Full 85-test sweep: 0 verdict changes, 0 exception-count
changes, 1 trace-hash change (the fix working, above), 4 rows with cycle changes totalling +1.61%
on those rows and +0.06% across all 85.

**What is NOT covered, so it cannot be overclaimed:**
- **AMO over a capability granule** — `wt_axi_adapter.sv:155` omits `ATOMIC_REQ` from `needs_tag`,
  so an AMO writes neither the DRAM shadow tag nor `cap_tag_q`. Recorded as invariant **I4**;
- **liveness of the three fixes composed** — S-07 refuses a store's allocation while S-10b stalls
  loads on the same granule. Read as non-cyclic (the stalled load parks in `WAIT_PAGE_OFFSET`
  with `data_req` low and stops competing for the port the write buffer needs), but **never
  observed**, because no test has yet opened that window.

**Blocking, and neither is settleable by reading:** the lint gate is RED by design
(`UNOPTFLAT 39` baseline against 40 — the S-10 change adds one loop on `wt_dcache.rd_ctag`), and
the S-10 synthesis regressed WNS from −10.629 to −16.400 ns with the cause **unattributed**. Only
a synthesis run settles the first; a determinism control of `e1140aeea` settles the second.


## Q-04 — QEMU's MOVC does not null a NOT_CAP source; the spec and the RTL say it must `OPEN — QEMU divergence, filed 2026-09-05`

`capstone-spec/parts/cap-man-insn.adoc` (MOVC): "If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to `x[rs1]`" — a NOT_CAP source qualifies, and the RTL does it (`capstone_flu_unit.anvil:13-26`, rtl-oracle 2026-09-04). QEMU's `helper_movc` nulls rs1 only under `rs1_v->tag && !captype_is_copyable(...)` (`op_helper.c:580-585`), so an untagged source survives a `movc` under QEMU and dies on silicon. Consequence: every copy of an integer-bridged pointer that stays live passes under QEMU and loses its value on the board (C-32, XFAIL `c32-movc-untagged-live.ll`); QEMU is a permissive oracle for that whole class until this is aligned with the spec. Fix belongs in `capstone-qemu`; the compiler side is C-32.


## Q-07 — QEMU's `INIT` requires `cursor == end` and aborts the host process otherwise; the spec and the RTL require `cursor > end` `OPEN — QEMU divergence, filed 2026-09-09 from the R-25 probe work`

**The accepted operand sets are DISJOINT, so an `INIT` path can be QEMU-validated and silicon-dead
at the same time.** The spec raises *Illegal operand value* when `x[rs1].cursor <= x[rs1].end`
(`capstone-spec/parts/cap-man-insn.adoc:421`) and the RTL does the same
(`core/anvil_build/capstone_flu_unit.anvil:139`). QEMU instead asserts:
`helper_csinit` is `assert(cursor == end)` (`target/riscv/op_helper.c:1200`) — a **host** assert,
not a guest trap. So `cursor == end` passes under QEMU and traps on silicon, while `cursor > end`
passes on silicon and kills QEMU with `SIGABRT`. Neither side's passing set is a subset of the
other's.

**Fix, QEMU side:** accept `cursor > end` (or `>=`, if the spec is amended first) and raise the
`ILLEGAL_OPERAND` trap instead of asserting, so a wrong operand is a guest fault the tests can see
rather than a dead emulator. Reported by the RTL lane; line numbers verified against the sources
named above (`docs/history/09-09-2026_16-00-00_r25-r26-fix-cycle.md`).


## RTL / FPGA

### R-27 — a pipeline flush inside the DYN unit's revocation-node query window orphans the node's response; the next capability operation waits forever (core dead, no trap) `OPEN — DEMONSTRATED IN SIMULATION 2026-09-09 on the unmodified ef5a8eaf2, three directed triggers; fix candidate sim-verified on branch r27-revnode-orphan-drain (458218562), lint at baseline, adversarially audited; NOT synthesised; silicon UNCONFIRMED`

**Found by the RTL lane while gating the R-26 fix** (`docs/history/09-09-2026_16-00-00_r25-r26-fix-cycle.md`,
"Named: the CALL-shaped hang", the directed arms, and the claim-auditor section; waveforms in
`~/dev/llvm-capstone-rebuild/records/r26/`). Mechanism, read off the waveform of `r26-v2-cscratch-fence` (two ticks
per cycle): the DYN unit sends a validity query to the revocation node at tick 1139 and waits; a `fence.i`
commits at 1141 and `ex_stage.sv` wires that flush to the DYN unit, whose thread resets (its response `ack`
drops at 1143); the node answers at 1145 — `rev_query_res_valid` rises and never falls, because nobody acks it
and the node is inside `send ep.query_res` (`capstone_rev_node.anvil:59`) with no fallthrough, timeout or drain
(`capstone_rev_node.anvil.sv:768-769, 872`); the re-issued CALL's request at 1683 is never acked; `ready` stays 0;
issue, decode and the frontend back up behind it. Not the frontend, not the switcher, not the R-26 flush as such:
the one ingredient is **an EX flush landing in the ≈3-cycle window between a DYN op's query request and the
node's response**. Flushes that reach EX: side-effecting CSR writes, `fence`, `fence.i`, `sfence.vma`, exceptions
and interrupts, `eret`, AMO/switch commit flushes (a mispredict does not, `controller.sv:111-115`). Ops that
query the node before completing: REVOKE, SPLIT, TIGHTEN, CALL, RETURN, LDC, STC, LCC. `fence`/`fence.i` commit
only when the store buffer is empty (`commit_stage.sv:411-456`), which is how a pending store moves their flush by
tens of cycles into a younger LDC's window.

Three independent triggers on the unmodified tree, all HANG (no trap, `tohost` never written): `fence.i` then
CALL (`r26-diag-csfence-mstatus`); `fence` then an LDC chain after a draining store (`r27-fence-nost`); a load that
faults immediately followed by an LDC (`r27-ldf-n0` — the core hangs instead of trapping; one `nop` of separation
and it traps cleanly). Pre-existing: none of them needs the R-26 flush. **Fix candidate** (`ex_stage.sv`, ~30
lines, one `always_ff`): per response channel remember a request the node accepted whose response is still owed;
if `flush_i` lands while one is owed, drain that response (ack it to the node, hide it from the DYN unit) and hold
`capstone_dyn_ready_o` low until it is gone. Read on the fix tree: all three triggers HANG → PASS with every
control unchanged; lint gate at the baseline counts exactly (UNOPTFLAT 40, ANVIL 0); the combined R-25+R-26+R-27
tree: 58-arm set clean, 88-row sweep identical except the predicted CCSRRW cycle deltas. **Label
correction (RTL lane audit, 2026-09-09):** every simulation labelled "40-cycle memory" before 05:55 UTC that
day ran at the DEFAULT latency (the define never reached Verilator through the model-build script);
nothing measured is invalid, the labels were. A verified 40-cycle record now exists: on the clean tip the
load-fault trigger `r27-ldf-n0` still HANGs; the fence-timed triggers miss the window at 40 cycles and hit
it at 0 (timing tickets, as the mechanism says); with the drain, on the combined tree, the 58-arm set has 0
hangs at 40 cycles (side-branch commits r27 f1d42e39a, r26 67d870cc8, r25 ec50837b5). Audit: mechanism
SUPPORTED on six attack lines; fix PLAUSIBLE, no misattribution sequence found; named residuals — one owed bit per
channel rests on the DYN unit's own serialisation; the node's designed non-answer when the pool is exhausted
(`head == 16'hFFFF`) remains a dead core after a later flush, not a new defect. The alternative (the anvil node
abandoning a send on flush) was tried upstream and reverted (`f5f9291c8`, `d15d45b33`, `7bcbdb39c`).

**Board relevance.** Every monitor `fence.i` after a capability operation, every domain trap and every interrupt
is a candidate placement of this window on silicon; whether any historic silent wedge was this is UNRESOLVED
(the Q-03 position-in-boot wedge is fixed by other means; the S-12 draws are explained). Whether the drain joins
the R-25/R-26 bitstream is the lead's decision; a bitstream carrying the R-26 flush without it is worse than
today's, by the RTL lane's own reading. Owner: the RTL lane.

### R-28 — the revocation-node WRITE ops (DROP/REVOKE/MREV/SPLIT/DELIN) can mutate node state for an instruction that never retires `OPEN — NAMED BY AUDIT 2026-09-09, not demonstrated, no directed test, no fix`

Named by the claim-auditor while attacking the R-27 fix (same history note). The write ops are held until they
are the oldest instruction (`issue_read_operands.sv:1525-1533`), but an interrupt or a debug flush can still land
after their request has reached the node, which may by then have mutated state — a minted node, a cleared
validity bit — for an instruction that is then killed and never retires. The R-27 drain discards the orphaned
response; it cannot undo the mutation. Consequence if real: a revocation-tree entry out of step with the
architectural state after an interrupt-timed flush (a node consumed, or a capability's lineage invalidated,
with the instruction re-executed afterwards and doing it again, or not at all). What would settle it: a directed
test placing an interrupt (timer, `S12_MEM_DELAY` to widen the window) on each write op's request→response window
and reading the node pool / validity bits before and after against the retired-instruction trace. Owner: the RTL
lane. Not to be conflated with R-27: that one is a deadlock, this one is a state divergence.

### R-26 — a Capstone CSR write (`CCSRRW` to `cpmp[i]` / `cscratch`) is not serialised against younger LDC/STC/CALL/RETURN, which read those registers combinationally `FIXED IN RTL 2026-09-09, SIM-VERIFIED, BITSTREAM PENDING (branch r26-ccsrrw-stale-read b7a794cfd: flush_o on every CCSRRW; ldmiss FAIL 11 → PASS, controls unchanged); silicon UNCONFIRMED; the flush must ship together with the R-27 drain (a flush inside the DYN unit's revocation-query window deadlocks the node); the monitor's four fence.i stay until the fixed bitstream is on the board`

**Found while asking whether the board monitor's eleven FPGA-only `fence.i` sites (Phase B item 8,
`docs/plans/monitor-unification.md`) are still needed.** Two rtl-oracle reads, quoted `file:line`:

* The `CCSRRW` write block (`core/csr_regfile.sv:2374-2550`; `CCSR_CPMP0` at `:2407-2414`,
  `CCSR_CSCRATCH` at `:2399-2406`) updates `cpmp_d`/`cscratch_d` only when `ccsr_we_i` is asserted,
  and that is driven from the **commit** stage (`core/commit_stage.sv:381-383`). It never sets
  `flush_o`; ordinary side-effecting CSR writes do (`:1160-1535`), and the controller flushes the
  pipeline on `flush_csr_i` (`core/controller.sv:209-212`).
* `LDC`/`STC` and `CALL`/`RETURN` decode to `fu = CAPSTONE_DYN` (`core/decoder.sv:1272-1309`),
  whose issue is gated only by `capstone_dyn_ready_i` (`core/issue_read_operands.sv:410-431,
  535-536`) — the `csr_ready`-based stall that holds `alu/ctrl_flow/csr/mult/capstone_flu` ops
  behind an uncommitted CSR write (`core/csr_buffer.sv:57-62`, `core/ex_stage.sv:521`) does not
  apply to them. They read `cpmp_q` combinationally in the bounds check
  (`core/pmp/src/pmp_data_if.sv:117-133`) and `cpmp_q`/`cscratch_q`/`mscratch_q` in the
  domain-switch save/restore (`core/csr_regfile.sv:401-432`).
* So a younger LDC/STC/CALL/RETURN can execute against the **old** `cpmp`/`cscratch` value while
  the older `CCSRRW` is still in flight. `fence.i` (`core/controller.sv:139-145`) is the flush the
  write never triggers; its icache flush is irrelevant here. `cepc`: no combinational consumer found
  (`mepc_q` is read only at `mret`'s commit, `:2811`) — no hazard, weaker evidence.
* Not every monitor `fence.i` is explained by this: of the nine in `sbi_capstone.S` only three follow a
  `CCSRRW` (lines 96, 188, 201 after 94, 181, 198); the five in `cap_env_init` follow split/store
  operations, not CSR writes. Fetch-side capability caching and ordinary icache non-coherence were
  both ruled out for those (`cpmp` never reaches the frontend; `split_out_cap` writes no code bytes).
* `sbi_capstone_init.S:44-51` has `CCSRRW … CSCRATCH` followed by domain entry with NO `fence.i`
  in between — whether that path reaches a `dom_switch` read soon enough to expose the race is
  UNRESOLVED.

**What would settle it on silicon** (proposed, not run): one boot, two staged firmware variants
— (a) as today, (b) `fence.i` inserted after every `CCSRRW … CSCRATCH/CPMP` and removed everywhere
else; a domain that returns a marker only if its restored `sp`/`gp` capability matches a baked
expectation. (a) intermittently wrong with (b) always right confirms the race; both always right
means the window is not hit by this code path (not proof of absence). **Owner: the RTL lane** (an
issue-time interlock for `CAPSTONE_DYN` behind a pending `CCSRRW`, or `flush_o` on `CCSRRW`, is the
RTL-side fix; the monitor keeps its `fence.i` until then).

> **DEMONSTRATED IN SIMULATION 2026-09-08 (RTL lane).** Write-up with the waveform readings:
> `docs/history/08-09-2026_22-00-00_r26-ccsrrw-stale-read-demonstrated.md`; tests on capstone-ariane
> branch `r26-ccsrrw-stale-read` (kept out of the gate testlist because the hazard arm is an expected
> FAIL). M-mode with `mstatus.MPRV=1, MPP=S` so loads are CPMP-checked, no `mret` between; CPMP[0]
> wide over three lines, `CCSRRW` narrows it, a load inside wide/outside narrow follows; positive
> control (load outside wide traps) and post-`fence.i` control (same load traps) fire in every arm.
> With no older instruction, or an older `div`, the load TRAPS — the VCD shows the divider holding the
> shared fixed-latency unit so the CSR op cannot issue until it finishes, commits two cycles after
> issue, and the load's check lands one cycle after commit on the edge the write becomes visible: a
> zero-cycle window by timing, not by design. With an older cache-missing `ld` ahead of the `CCSRRW`
> (`S12_MEM_DELAY=40`; label correction 2026-09-09: the pre-05:55 runs actually ran at the default latency, the define had not reached Verilator — a verified 40-cycle rerun on the clean tip still reads FAIL 11 and passes with the fix) the CSR write waits at commit behind it; the younger load's CPMP check runs
> at tick 1183 with `cpmp_allow = 1` against the OLD entry, the write lands at 1193, the load retires
> with data from outside the new bounds — **FAIL 11 (hazard)**; with `fence.i` between: PASS. So any
> older instruction that delays the CCSRRW's commit and is not on the fixed-latency unit opens a
> window of its latency for every younger load/store. A plain `ld` was used; `LDC`/`STC` hit the same
> `pmp_data_if` check but were not run. NOT exercised: the `cscratch`/`cepc` readers in
> `dom_switch_read_process` — the switcher reads after the CALL/RETURN commits, so that path needs
> the switcher's read to fall within a cycle of the write; a separate test with CALL semantics, and
> the shape with the board consequence (`sbi_capstone_init.S:44-51`). Proposed fix shape: `flush_o`
> in the CCSR write block for CPMPn/CSCRATCH/CEPC, the idiom the ordinary side-effecting CSR writes
> use (`csr_regfile.sv:1160-1394`); the `ldmiss` arm turning PASS with the controls still firing is
> the acceptance test. RTL, lint, audit, synthesis before any board use. For the monitor (Phase B
> item 8): the three `fence.i` after `CCSRRW` are now KNOWN load-bearing for CPMP writes, not merely
> kept on suspicion.

### R-25 — `INIT` writes the new LINEAR capability to BOTH `rs1` and `rd`, so linearity is broken `OPEN — SILICON, SECURITY-MODEL VIOLATION, CONFIRMED BY DIRECTED TEST`

> **CONFIRMED ON SILICON 2026-09-09 (boot sw41, board lane).** The domain probe `r25dup`
> (`tests/runtime-qemu/silicon-ladder/r25dup_fpga_app.c`; construction from the RTL lane's self-checking
> test: transferred LIN region, cursor past end, CAPTYPE in place to UNINIT = 4 in the RTL's numbering,
> `init a3, a1, a0` with rd ≠ rs1, then a capability store THROUGH rs1 and a load back through the region)
> returned **0x25000001** on caplifive_s12fix_5097eb166: the store through rs1 landed and a tagged
> capability came back, so rs1 still held a live LINEAR capability after INIT — the duplicate. The control
> `r25same` (rd == rs1, same chain) returned 0x25000001 as required; k800 = 4; zero fault tags. This is the
> pre-flash reading the RTL lane predicted; on the R-25-fixed bitstream (fpga-testing-dev 66c4e7517, C2
> 42a141c93) the same arm must trap at that store (UNEXPECTED_OPERAND). Boot sw39's first attempt of the
> same pair wrote the spec's type number 3 (= REVOKE on the RTL) and wedged at INIT with cause 27; VOID,
> recorded in the tsv. Two side findings: the RTL does not implement the spec's cincoffset-past-end rule
> (cap-man-insn.adoc:262), which is what makes the construction possible; and a domain fault is not
> delivered to the monitor on this RTL (M-1's open half) — the core wedged into a repeating 0xdead.. UART
> record.

**Reported by the compiler lane's rtl-oracle pass 2026-09-04; verified here against the RTL rather
than taken on report, including the control that makes it a defect rather than an idiom.**

`core/anvil_build/capstone_flu_unit.anvil:147`, the `INIT` path:

```
let rd = call create_capability(rd_temp.metadata, new_cursor);
let result = call create_result_pack(data.trans_id, ex_code::NO_EXCEPTION, rd, rd);
```

`create_result_pack(id, ex, rs1, rd)` assigns `cap_rs1 = rs1` and `cap_result = rd`
(`capstone_unit.anvilh:360-364`). Passing `rd` twice therefore writes the **newly created LINEAR
capability into `rs1` as well as `rd`** — two live LINEAR capabilities over one region, which is
precisely what linearity exists to prevent.

**THE CONTROL, which is what makes this a defect and not a house idiom.** The `rd,rd` form appears
four times in this file — `:42`, `:72`, `:106`, `:147`. The first three are each guarded by

```
if(data.rs1 == data.rd){
```

i.e. they are the *same-register* case, where writing both is writing one register and is correct.
**`:147` has no such guard.** Its enclosing conditions are only `rs1.cursor <= rs1.metadata.end`
(`:139`, raising `ILLEGAL_OPERAND_VALUE`) and the `else` around it. So the codebase demonstrably
knows the correct idiom and `INIT` omits it, for **any** `rs1 != rd`.

Every sibling in the file passes `rs1` unchanged, `rs1_out` modified, or `rcnull` when the source
must be consumed (`:173`). None of those apply here.

**QEMU nulls `rs1`**, so this is silicon-only and cannot be reproduced under emulation — the class
of divergence that has repeatedly cost this project board time.

**Not yet established, and needed before this is handed to the hardware side:**

- ~~whether it is reachable from our codegen~~ **ANSWERED 2026-09-04, and it IS reachable.**
  `INIT` carries **no tied-operand constraint** in the instruction definition, unlike
  `SHRINK`/`DELIN`/`DROP`/`REVOKE`. Measured on the cycle-1 compiler: a source pointer that stays
  live across the builtin produces `init a1, a0, a1` at `-O1` and `-O2`. Where `rd == rs1` does
  occur it is the register allocator reusing a register, not a constraint — nothing forces it.

  So: **real in the ISA and reachable by construction from any C that calls
  `__builtin_capstone_cap_init` with a live source.** The mitigation is that no in-tree C outside
  tests uses `cap_init` today, so no shipping domain hits it — that is a fact about our current
  programs, not about the hardware, and it expires the moment one does.
- **whether the duplicated capability is usable**, or whether a later consumer traps on it.
  **STILL OPEN** — the directed test below reads both registers but never dereferences `a5`,
  so "two LINEAR capabilities exist" is established and "the second one works" is not.
- ~~a **directed `.S`** in the simulator with `rs1 != rd`~~ **WRITTEN AND RUN 2026-09-05 —
  CONFIRMED.** `verif/tests/custom/capstone/init-rs1-ne-rd.S`, run on `s12-ldc-rolling-filter`
  (`05f2be6bd`) and independently by the compiler lane on the flashed `5097eb166`; `flu_unit`
  is byte-identical between the two, and both runs agree line for line. 443 cycles, 0
  exceptions, `tohost = 0` — a genuine pass, not a timeout `SUCCESS`.

  ```
  [Cycle 337] Reg[17]: 0000000000000000                       <- arm 0 CONTROL: MOVC consumed it
  [Cycle 371] Reg[15]: Cursor 0x80003200 ... Revnode_id 2 | Type : 1    <- a5, the SOURCE
  [Cycle 373] Reg[16]: Cursor 0x80003200 ... Revnode_id 2 | Type : 1    <- a6, the DEST
  [Cycle 407] Reg[19]: Cursor 0x80003400 ... Revnode_id 2 | Type : 1    <- arm 2, rd == rs1: LIN
  ```

  Source and destination are indistinguishable — same cursor, same bounds, same revnode, both
  `Type : 1` (LINEAR). Arm 0 is the **instrument control**: a `MOVC` of a linear source leaves
  `a7` printing a bare `0`, which proves `CAPPRINT` can render a cleared register, so arm 1's
  non-zero `a5` is a reading and not a blind spot. Arm 2 is the **conformance control**: with
  `rd == rs1` the result is a single correct LINEAR capability, which is why every compiled
  `INIT` in the tree today is unaffected.

  `INIT` traps `ILLEGAL_OPERAND_VALUE` unless `cursor > end`, so the UNINIT operands are
  fabricated with `end = base - 16` to satisfy that precondition without touching the
  capability under test.

**The fix, and the trap in it.** `MOVC` in the same file is the correct shape and shows the
required structure: `if (data.rs1 == data.rd)` pass `(rs1, rd)`, else pass
`(create_cnull(), rd)`. Its `else` branch catches **every** non-`NONLIN` source — `UNINIT`
included — which also **refutes the one alternative account** of R-25, that `MOVC` simply does
not consume `UNINIT` sources and `INIT` is faithfully reproducing it. It does consume them.
`INIT` alone is the defect site.

So the fix is *not* "replace `rd,rd` with `cnull,rd`". Applied unguarded that clobbers the
result in the `rd == rs1` case, which is the shape all shipping code uses — turning a defect
nothing currently hits into one everything hits.

**Mirror MOVC's GUARD, not its VALUES** — an earlier revision of this entry said "pass
`(rs1, rd)`" for the same-register arm, copying MOVC's variable names, and that is wrong for
`INIT`. MOVC's `rd == rs1` arm passes the *original* capability because MOVC does not
transform it. `INIT` does: its `rd` is the retyped LINEAR capability at the new cursor, and
its source `rs1` is the untouched UNINIT one. Passing the source in the same-register arm
would make every shipping `INIT` a no-op. The same-register arm must keep passing
`(rd, rd)` — i.e. exactly what `:147` does today, which is why that case is correct now:

```
if (data.rs1 == data.rd) { create_result_pack(id, NO_EXCEPTION, rd,             rd) }
else                     { create_result_pack(id, NO_EXCEPTION, create_cnull(), rd) }
```

Arm 2 of `init-rs1-ne-rd.S` is the backstop: it fails in ~14 s if the same-register arm is
got wrong. Which write wins when both target one register is the Anvil sequential-reading
hazard that has produced the opposite of hardware behaviour twice on this project — so run
both arms rather than reasoning it out.

**Related, from the same pass and NOT yet verified here:** a `REVOKE` landing on `UNINIT` leaves
`cursor = START` on RTL (`capstone_dyn_unit.anvil:67-68`) against `END` on QEMU, while RTL's own
`INIT` requires `cursor > end` (`:139` above) — which would make a post-revoke `UNINIT` capability
impossible to re-initialise on silicon. If that holds it is a second independent defect in the
same instruction pair. Verify before recording.


### R-8 — pure-scalar miscompute; the "accumulator" characterisation is TOO BROAD `OPEN`

> **Sweep 2026-09-05 — GONE on silicon.** `beebs_expint -O1` = 2021290181 = host oracle (boot sw01); `expint_diag` = 3883 on silicon (boot swd2), the correct value where the board once read 2.
Measured 2026-07-28 on `beebs_expint`, and it is the cleanest instance of this class yet.

| | capability | baseline (bare-metal) |
|---|---:|---:|
| retval | **2,223,116,741** ✗ | 2,021,290,181 ✓ |
| cycles | 110,988 | 110,844 |
| instret | **71,243** | **71,248** |

**The instruction counts differ by 5 out of 71,000.** The domain ran the whole
computation — this is not a hang, and not the "compute never ran" signature
(`beebs_insertsort`'s 560 instructions) — and produced a different number.

**Why R-1 cannot explain it:** `beebs_expint` has **no arrays at all**. Every value is a
scalar local; the only global is a `volatile long` accumulator. There is no same-object
load/store pair for a memory hazard to act on. The rung was in fact *selected* against
R-1's shape for exactly this reason.

**Why it is not a compile-time difference:** the identical binary is **QEMU-correct**
(`__CAPSTONE_LADDER_BEEBS_EXPINT_PASSED__`, oracle 2,021,290,181). So constant folding,
the `2e6`/`3e7` double-to-long literals, and shift-amount UB are all ruled out — those
would fail under emulation too.

**So: same instructions, same count, different arithmetic result, on silicon only.**

- **Companion case:** `beebs_fibcall` is also pure scalar and also miscomputes on silicon
  (at −O1 it retired 166,539 against a baseline 177,855 — ~94 % of the work, wrong answer).
  Two independent pure-scalar miscomputes make this a class, not a one-off.
- **Relation to R-6:** `beebs_janne`'s failing nest is likewise pure register arithmetic.
  R-6, R-8 and the `fibcall` miscompute plausibly share one mechanism that is **not** R-1.
- **Value:** this is the strongest evidence yet that **R-1 is not the whole story**, which
  the registry has flagged since R-6 but could not previously support with a clean case.
- **Repro:** `tests/runtime-qemu/silicon-ladder/beebs_expint_*`, `-O1`, oracle
  2,021,290,181, QEMU-green, baseline half clean (15/15 tied, spread 0).
#### BISECTED 2026-07-28 (`expint_diag`) — one slot diverges, and it names the fault

| slot | board | expected | |
|---|---:|---:|---|
| dbg0 branch / dbg1 init | 0 / 2 | 0 / 2 | ✓ |
| dbg2,3 `fact` (signed div) | 0 / 0 | 0 / 0 | ✓ |
| dbg4 `psi` (nested loop) | **3881** | 3881 | ✓ |
| dbg5 `ei_foo` (the shift) | 0 | 0 | ✓ |
| **dbg6 `del` at i==nm1** | **3881** | 3881 | ✓ **the addend is correct** |
| dbg8 trip count | **100** | 100 | ✓ **the loop ran fully** |
| dbg9 `sum(ii)` | 1225 | 1225 | ✓ |
| **dbg7 final `ans`** | **2** | **3883** | ✗ |

**`ans` is frozen at its INITIAL value.** The loop ran all 100 iterations, `del` was
computed correctly as 3881, and `ans += del` did not accumulate. Nothing else diverges —
division, shifts, the nested loop and control flow are all correct.

#### This is R-6's mechanism, and the two issues unify

`beebs_janne` (R-6) showed *exactly* this: `a` frozen at **2** after 200 iterations of
`a = a + 2`, with the loop counters self-consistent. Both cases are:
- **pure register arithmetic**, no arrays, no memory in the loop
- the loop **runs its full trip count**
- the per-iteration value is **computed correctly**
- the **accumulator retains its initial value**

> **Proposed statement (NOW KNOWN TOO BROAD): a scalar accumulated across loop
> iterations retains its initial value.**
>
> **⚠ REFUTED as stated, 2026-07-28.** A minimal probe --
> `long a = 0; for (i = 0; i < 100; i++) a += 1;` in a domain, returned as the retval --
> **comes back as 100, correct**, on the same board and toolchain. So plain accumulation
> is NOT broken, and whatever breaks `expint` and `janne` needs more than a loop and a
> `+=`. Candidate extra ingredients, none yet tested: a branch inside the loop body,
> register pressure, a nested loop, or the specific accumulate-inside-an-if shape both
> failing kernels share.

R-1 cannot explain either (no memory involved), and the identical binaries are
QEMU-correct. `beebs_fibcall`'s pure-scalar miscompute is very likely the same thing.

**Why this matters more than a benchmark row:** R-1 plus this account for essentially every
silicon failure seen — R-1 for the array kernels, this for the scalar ones. Two mechanisms,
not a fog. It is also a far better bug report: a five-line loop whose accumulator does not
accumulate, with a QEMU-correct binary and every neighbouring operation proven good.

**Probe status: TWO versions run, both INCONCLUSIVE — the blocker is our harness, not the
board.** v1 pinned accumulators to named registers (suspected of corrupting `res` in `a0`);
v2 removed all pinning, used a `volatile` store pointer and wrote each slot immediately
after its loop. **Both behaved identically**: `res[0]` and `res[2]` land (retval 100 and the
`0xD09E` marker both arrive) while `res[3..11]` all read zero, so the controller suppresses
the DEBUG line.

**The discriminating fact: `expint_diag` writes the SAME slots successfully** (it returned
`dbg0=0 dbg1=2 ... dbg7=2`). So slot delivery works in one probe and not another, and the
difference is in our two `domain_main` implementations, not in silicon. **Diff them before
running anything else** — `expint_diag_fpga_app.c` vs `accum_probe_fpga_app.c`. Do not
spend another boot on this probe until a QEMU-visible reproduction exists; note the QEMU
ladder harness gives only an 8-byte `res` region, so the debug-slot path is currently
board-only, which is itself worth fixing.

**Original probe status note (superseded, kept for the reasoning):**
It was designed to discriminate the important question (see below) across 9 debug slots. On
the board `res[0]` returned **100** — the plain accumulate, correct — but **all nine
`res[3..11]` slots read zero**, so the controller suppressed the DEBUG line and eight of
nine probes produced no data. The `res[3..]` writes did not land even though `res[0]` did;
the QEMU harness separately rejects this probe because its shared region is only 8 bytes.
**Fix the probe's use of the debug slots, then re-run** — the discrimination is still the
right experiment.

**The question the probe must answer, and why it matters more than the benchmark:**
"an accumulator does not accumulate" is an extraordinary claim about an ALU. An ordinary
explanation fits every observation equally well — the value lives in a **register that
something clobbers on silicon**: our entry glue, the `cscall` path, or a trap handler that
saves less than our QEMU fork models. That would present identically (right addend, right
trip count, value reverting to its initial state) and would be **our bug, not the board's**.
Reading: memory-form correct + register-forms wrong ⇒ ours; one register class failing ⇒
names the culprit; all forms failing ⇒ the hardware claim survives; short loop passing and
long failing ⇒ something periodic, i.e. a trap.

**Confidence, stated plainly:** R-1 is well supported (five-line repro, controls both
sides, 7 failed mitigations, a correct advance prediction). **R-6/R-8 are NOT** — calling
them hardware is currently an assumption, and the minimal probe passing makes a
software-side explanation *more* likely, not less.
- **Repro:** `tests/runtime-qemu/silicon-ladder/expint_diag_fpga_app.c`, `-O1`,
  expected `dbg7=3883`, board returns 2.

### R-6 — `beebs_janne` hangs although R-1 predicts it should pass `OPEN`

> **Sweep 2026-09-05 — GONE on silicon.** `beebs_janne -O1` = 484656629 = host oracle on caplifive_s12fix_5097eb166 (boot sw01, control k800 = 4; `tests/board-results/2026-09-05.tsv`).
BEEBS `janne_complex`: nested data-dependent loops whose conditions are computed **entirely from
locals**, with one `.bss` counter (`jc_iters++`) touched through a single capability register.
R-1 requires a load through one capability register with an intervening store through *another*;
that never occurs here, so R-1 predicts PASS. **The board hangs it.**

- **Repro:** `tests/runtime-qemu/silicon-ladder/beebs_janne_{kernel.h,fpga_app.c,host.c}`,
  `-O1`, oracle 484656629, QEMU-correct through the identical controller.
- **BISECTED 2026-07-27 (`janne_diag`), and the result does NOT fit R-1.** Safety bounds turned
  the hang into a returned diagnostic:

  | slot | board | correct |
  |---|---|---|
  | outer trips | **200** (its safety bound) | 9 |
  | inner trips | **500** (its safety bound) | 12 |
  | final `a` | **2** | 31 |
  | final `b` | **-339** | 27 |
  | `jc_iters` | 700 (= 200+500, self-consistent) | 21 |

  Neither loop terminates, and `a` is frozen at 2 — after 200 outer iterations of `a = a + 2` it
  should be ≥ 400. The board state is internally consistent (`a`=2 and `b`=−339 keep both
  conditions true forever), so the loops behaved *exactly* as if `a` stopped accumulating.

  **The damning part: the loop nest is pure register arithmetic.** Verified in both the emitted
  assembly and the shipped `.dom` — `a`=`a3`, `b`=`a2`, the counter accumulates in `a6`, and
  `jd_iters` is stored **once after** the loops. There are **no memory operations inside the
  nest**. R-1 is a memory hazard and therefore cannot explain this.

- **Status: mechanism UNKNOWN. Do not fold this into R-1.** Candidate explanations, none tested:
  a control-flow/branch-resolution issue on this RTL (the nest is unusually branch-dense); an
  interrupt landing inside the measured bracket (the measurements doc notes ~16k cycles when one
  does; this rung ran 11,167); or the emitted code differing from what actually executes.
- **Next probe RUN (`regloop_diag`, 2026-07-27) — and it PASSES, which deepens the puzzle.**
  A staircase of register-pure loops, no memory in any body:

  | probe | board | correct |
  |---|---|---|
  | simple counted loop, 100 trips | 100 | 100 ✅ |
  | nested 10x10 | 100 | 100 ✅ |
  | data-dependent branch in body | 100 | 100 ✅ |
  | multiply in body | 100 | 100 ✅ |
  | **janne's EXACT nest, bounded** | **21** | 21 ✅ |

  So pure-register looping is fine, and **janne's algorithm itself runs correctly on this board**.

- **The open puzzle: two register-pure implementations of the same nest, one fails, one works.**
  Verified by counting memory ops in the loop *region* (not the whole function): `janne_diag`'s
  nest has **0**, and it fails; `regloop_diag`'s dbg4 nest also has 0, and it passes. The
  differences are incidental — three counters vs two, bounds 200/500 vs 400, and dbg4 executes
  after four other loops. Nothing algorithmic.
- **Most likely reading: this is the known code-layout / perturbation sensitivity**, the same
  phenomenon as the 2026-07-26 controlled A/B where **four added instructions flipped a passing
  rung from correct to wrong**. That makes R-6 a *symptom class* rather than a distinct fault, and
  means **a passing rung is not stable ground** — already the standing caveat in
  `ref/fpga-silicon-measurements-for-paper.md` §5.
- **Do not merge R-6 into R-1** (R-1 is a memory hazard; these nests touch no memory), and do not
  claim it is understood. The honest status is: janne's algorithm works, one particular build of
  it does not, and the discriminator is not algorithmic.

### R-3 — Second domain at the same entry VA hangs within one boot `WORKED AROUND`
A domain reused at entry VA `0x10000` within a single boot silently hangs its `cscall` —
a missing icache invalidate on the domain switch. This forced **one full power-cycle +
JTAG firmware reload per rung** (~2.5 min), the dominant cost of every board sweep.

- **RESOLVED IN PRACTICE 2026-07-28: the fault is ADDRESS-KEYED.** Domains linked at
  *different* entry VAs run back to back in one boot. `beebs_bs` @`0x10000` then
  `beebs_prime` @`0x20000`, no power-cycle between them, both returned their oracles.
  Nobody had tested this; the per-rung power-cycle was an assumption, not a measurement.
- **Validated as measurement-safe, not merely correct.** The obvious risk was that a
  second domain runs with an icache warmed by the first, so cycle counts would not be
  comparable to the published first-domain numbers. A reversed-order control says no:

  | rung | as 1st domain | as 2nd domain | spread |
  |---|---:|---:|---|
  | `beebs_bs` | 2,258 / 2,246 | 2,263 | 0.75 % |
  | `beebs_prime` (−O1) | 9,746 | 9,749 | **0.03 %** |

  `instret` was byte-identical in both positions (875, 2,708).
- **A wedged rung poisons the rest of the sweep unless recovery is enabled.** On
  2026-07-28 `rv8_primes` hung and the runner kept "reusing" the dead boot, losing the
  **four** rungs after it — all of which had worked minutes earlier. Fixed: a rung that
  times out clears the boot flag so the next one power-cycles. One failure now stays one
  failure. Anyone re-implementing one-boot mode must include this.
- **How to use it:** `LADDER_DISTINCT_VA=1` on the build (assigns `0x10000`, `0x20000`, …
  64 KiB apart) **and** `LADDER_ONE_BOOT=1` on the runner. Both are opt-in: if the
  address-keying assumption ever fails the symptom is a silent hang that looks like a
  rung result, so this must not become a default without a control rung in the sweep.
- **Impact:** a 13-rung sweep goes from ~13 boots (~35 min) to **1** (~5 min).
- **Not a root fix.** The monitor still lacks the icache invalidate on domain switch, so
  same-VA reuse still hangs. Sidestepped, not repaired — the fix sketch remains in
  `plans/curried-crunching-gizmo.md`.
- **Mechanism note:** the domain-boundary `fence.i` was long suspected to fix R-1 as well;
  board test #63 disproved that. It remains the right fix for **this** issue only.

### R-4 — A shared-region word is silently corrupted `OPEN`
`rv8_primes` returned the *correct* result while a word of its shared region held a stray DRAM
address. Passing rungs were only ever clean where someone looked.
- **Evidence:** `ref/fpga-silicon-measurements-for-paper.md` §5

### R-5 — Illegal/meaningless capability ops wedge rather than trap `OPEN`
M-mode appears to spin (`capstone_error` = `while(1)`); only a power-cycle recovers. Seen for
`C_GEN_CAP` (QEMU-only op), for the R-2 `delin`, and for an `scc`-derived load.
- **Evidence:** `history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md`

---

### R-10 — a 16-byte capability copy MANGLES plain scalar data in its high half `ROOT CAUSE of C-13, board-confirmed 2026-07-29`

**THE MECHANISM, complete.** A capability's two halves are stored differently:

    low  8 bytes = cursor   -> written RAW      (wt_axi_adapter.sv:201, axi_wr_data[0] = dcache_data.data)
    high 8 bytes = metadata -> written ENCODED  (stored as compress_cap(...), ariane_pkg.sv:809)

`compress_bounds` (`ariane_pkg.sv`) is a genuine LOSSY encoder: leading-zero count, an
exponent E, and truncation to 21/14/12-bit fields. It is closed over real capabilities
and **not** over arbitrary bit patterns.

So the monitor's copy `dom_data[ci] = dom_code[gpoff_c + ci]` (`sbi_capstone.c:400-404`,
both `__linear void *`, i.e. one `ldc` + one `stc` per 16 bytes) does
decompress-then-recompress on the high half of every granule. Plain scalar data does not
survive it. The low half, being the raw cursor, does.

**BOARD-CONFIRMED, 4 rungs in one boot:**

    stage 7  reads blob +0 and USES it                    PASS  (582955588)
    stage 10 reads blob +8 and DISCARDS the value         PASS  x2
    stage 8  reads blob +8 and USES it as `count`         FAIL

The load does NOT fault -- stage 10 performs the identical access and passes twice. The
VALUE is wrong. A mangled `count` then makes `slli`/`sub`/`split` carve at a wild bound,
which is the wedge. The image descriptor is verified correct (built_flag=0, count=1), so
the corruption happens in the copy, not the compiler.

The monitor's own comment -- "the image bytes here are const initializer data with no
capability tags, so the 128 bits round-trip unchanged" -- is FALSE on real silicon.

**Secondary defect, same root.** `is_cap_req = |dcache_data.user`
(`wt_axi_adapter.sv:196`) and `st_wr_cap = |wr_user_i` (`wt_dcache_mem.sv:138`) decide
"holds a capability" by OR-reducing the metadata word; neither file references `cap_type`
(0 occurrences in each), so `cap_type == NOT_CAP` is never consulted. A consequence worth
noting separately: when the high half is ZERO, `is_cap_req = 0` sets `axi_wr_blen = 0`
(`:209`), so only ONE beat is written and the high 8 bytes are left at whatever was in
DRAM. That also means `dom_seal[i] = 0` zeroes only half of each granule.

**QEMU cannot reproduce any of this.** It stores exact fat structs with a discrete tag
(`cap.h:93`, `cap_mem_map`); there is no lossy codec and no content-derived tag. Third
RTL/QEMU divergence to cause a multi-session blocker, after DELIN and this.

**FIX IMPLEMENTED 2026-07-29, AND IT DID NOT UNBLOCK THE REAL PATH.** Root cause of the
16-byte copy turned out to be a capstone-c DECLARATOR BUG, not a design choice:
`__linear void *mem_l, *dom_code, *dom_data, *mem_r;` accumulates the `*` across
declarators (dag_builder mutates the shared decl type and never resets it), so only
`mem_l` got `void *` -- `dom_code` became `void **` and `dom_data` `void ***`.
Dereferencing them therefore yielded a POINTER (16 B), which is why the copy emitted
`ldc`/`stc` at all. Fix = one declarator per declaration, plus `>> 4` -> `>> 3`.
Verified by regenerating the monitor: exactly 6 instructions change in 4,653, and the
two that matter become scalar `ld`/`sd`. Confirmed present in the shipped firmware
(size 17,466,376, create_domain labels 30, `ld a4, 0(a3)` / `sd a4, 0(a7)` in the loop).

**Board result with that firmware: real interp STILL FAILS.** `beebs_primer1` and
`gpstress`, both real interp, both no END marker. The primer domain was byte-identical
to the one that failed before (sha 3e3980cd), so the monitor was the only variable.

So the copy corruption was REAL and board-confirmed (stage 8 fails / stage 10 passes on
the identical load), and fixing it is NECESSARY -- but it is NOT SUFFICIENT. Something
else also breaks the real path. **Do not record C-13 as fixed.**

Next experiment, one variable: re-run the stage ladder against the FIXED firmware.
Stage 8 (reads `count` from blob+8, the previously-mangled half) is the discriminator --
  stage 8 now PASSES -> the copy fix repaired the read; the remaining fault is downstream
                        in the record reads / gp-park / cap-init, all of which have knobs
  stage 8 still FAILS -> the copy fix did not repair the read and the mechanism story
                        above is incomplete despite being individually verified

**FIX DIRECTIONS (not yet implemented):**
1. *Monitor copies scalars with scalar accesses.* The correct general fix -- it also fixes
   the bulk initializer data, which matters at SQLite scale (1,059 globals). Open question
   is whether capstone-c can express a non-`__linear` view of the same span so the
   compiler emits `ld`/`sd` instead of `ldc`/`stc`. `sbi_capstone.c` has no `memcpy` and
   no scalar-pointer cast today. UNVERIFIED.
2. *Descriptor avoids metadata halves.* Lay the descriptor out so every 8-byte scalar sits
   in the LOW half of its own 16-byte granule. Purely a compiler+glue change, no monitor
   change. Fixes the descriptor but NOT the bulk initializer data, so it unblocks the
   glue and not SQLite's globals.

Both may be needed: (2) to unblock quickly, (1) for correctness at scale.

**Confirmed by direct quote, verified independently:**

    core/cache_subsystem/wt_axi_adapter.sv:196   assign is_cap_req = |dcache_data.user;
    core/cache_subsystem/wt_dcache_mem.sv:138    assign st_wr_cap  = |wr_user_i;

Both decide whether a 16-byte granule "contains a capability" by **OR-reducing the whole
64-bit metadata word**. Neither file references `cap_type` even once (0 occurrences in
each, checked). So the architectural notion of a capability — `cap_type != NOT_CAP`,
bits [30:28] of the metadata word (`ariane_pkg.sv:646`) — is **never consulted by the
memory subsystem**. The shadow tag is set from raw bit content.

**Consequence.** Any 16-byte capability-width store (`stc`) whose HIGH 8 bytes are
nonzero marks its destination granule as holding a capability, even when the value is
plainly not one. Copying ordinary scalar data with `ldc`/`stc` therefore poisons the
shadow tag across the whole copied region.

**Where this bites us.** The monitor copies the entire globals blob into `dom_data` with
16-byte capability accesses (`sbi_capstone.c:400-404`, `dom_data[ci] = dom_code[...]`,
both `__linear void *`). Its own comment asserts the bytes "round-trip unchanged" — the
BYTES do; the shadow tag does not. For the descriptor, `count = 1` sits in the high half
of granule 0, so `|1 = 1` and that granule is mis-tagged. For SQLite (1,059 globals, most
initialized) it would be most of the blob.

**QEMU cannot reproduce this class at all.** Its capability tag is a discrete per-register
boolean plus a side table (`cap.h:93`, `cap_mem_map`), content-independent, and
`helper_compress_cap` returns 0 for an untagged source (`op_helper.c:1155-1164`), so the
destination is never marked. Same shape of blind spot as the DELIN divergence (C-13).

**NOT yet established:** that this mis-tagging is what *wedges* the board. The data plane
reads symmetrically (`wt_dcache_mem.sv:261`, banks muxed by address bit 3) with no fault
found tied to the tag, and no explicit fault condition was located in `wt_dcache_ctrl.sv`
or `wt_dcache_missunit.sv`. The board experiment that separates "the load faults" from
"the value is wrong" is INTERP_DIAG_STAGE=10 (see C-13). Do not write this up as C-13's
cause until that lands.

**Unread, and needed to close the mechanism:** `capstone_dyn_unit.anvil` /
`capstone_unit.anvilh` for the `_load_ep_res` vs `_load_ep_normal_res` handshake —
`ex_stage.sv:791` decompresses EVERY load's result, not just `ldc`, and forwards it to
the DYN unit on a channel whose ack is left dangling (`ex_stage.sv:910`).

**Fix direction (unresolved):** M-mode must copy plain scalar data with scalar stores
rather than `ldc`/`stc`. Whether capstone-c can express a non-`__linear` view of the same
span is a compiler/ABI question, not an RTL one, and is unverified — `sbi_capstone.c` has
no `memcpy` and no scalar-pointer cast anywhere today.

### C-4 — split into a FIXED half and a remaining domain-creation bug
Renamed from "large read-only data cannot be delivered": size was never the variable.

#### C-4a — constant pools are unreachable in a domain `FIXED 2026-07-28`

> **Sweep 2026-09-05 — re-verified FIXED.** `rv8_sha512` rung under QEMU: 1390718314 = oracle.
**Root cause, with the emitted sequence:**
```
.LCPI0_0: .quad 81985529216486895        ; .rodata.cst8 -- a CONSTANT POOL entry
  auipc a2, %pcrel_hi(.LCPI0_0)
  addi  a1, a2, %pcrel_lo(...)
  scc   a1, gp, a1     ; set gp's cursor to a .rodata address
  ld    s6, 0(a1)      ; FAULTS
```
A pool entry is **not** a `GlobalVariable`, so it gets no cap-table slot (correctly);
`lowerConstantPool` then falls back to `LGA` → `scc gp`. Under gp-captable `gp` is bounded
to the **cap table itself**, so the cursor lands out of bounds. The tell in the fault line
is that the reported bounds are exactly the table:
`cursor = 0x101561000, bounds = (0x10157ffd0, 0x101580000)`.

**Fix:** `CapstoneSubtarget::useConstantPoolForLargeInts()` returns **false** whenever the
gp-free/gp-captable ABI is active, so the constant is materialised inline instead. Forming
a pool in a domain is always a miscompile, never an optimisation — the same reason
`-fno-jump-tables` is already mandatory (a jump table is `.rodata` too).

**Validated:** the previously-faulting `rv8_sha512` configuration now returns its oracle
(`__CAPSTONE_LADDER_RV8_SHA512_PASSED__`); 0 `.LCPI` entries remain in the emitted asm;
Capstone lit **43/43**; `beebs_bs`, `beebs_prime`, `beebs_cnt` still pass QEMU parity.

> **Two wrong turns on the way, both worth remembering.** First this was called a
> *large-data delivery* problem, because bigger constants are the ones that get pooled.
> Then, on seeing that all named globals DID have cap-table slots, the constant-pool
> explanation was **retracted as refuted** — but the faulting object was never a global,
> so the descriptors could not have refuted it. The lesson is to identify the faulting
> OBJECT before reasoning about the mechanism: a symbolised `-S` listing settled in one
> step what two rounds of inference got wrong.

#### C-4b — the large-RO COPY PATH in the generated glue is broken `FIXED 2026-07-28`

> **Sweep 2026-09-05 — re-verified FIXED.** `beebs_crc32big` under QEMU: 1703161001 = oracle (a first run exited 75 on a boot-login infra flake; the rerun is the result).

**FIXED 2026-07-28. Root cause: `cincoffset` CONSUMES a linear `rs1`.**

`op_helper.c:635-640` — `helper_cscincoffset` with `rd != rs1` does
`*rd_v = *rs1_v; if(!captype_is_copyable(rs1_v->val.cap.type)) *rs1_v = CAPREGVAL_NULL;`
and `cap.h:122` defines `captype_is_copyable(ty) { return ty == CAP_TYPE_NONLIN; }`.
`sp` arrives from cscratch as `CAP_TYPE_LIN`, and the builder's only `delin(sp)` was its
LAST line — so the copy path's `cincoffset(t4, sp, t5)` **nulled `sp` outright**, and the
next `split(t2, sp, t1)` tripped `helper_cssplit`'s `assert(rs1_v->tag && !rs2_v->tag)`
with `tag == 0`.

That accounts for every observed symptom: it fired only AFTER `Created domain ID = 0`,
only when `COPY_THRESHOLD` selected the copy path, and never in the zero-init path (which
`cincoffset`s `t2`, already delinearized) or the unrolled path (which never `cincoffset`s
`sp`). It is also why five careful static readings of the generated assembly missed it —
**the assembly is correct as written; the defect is in the ISA semantics of one operand.**

**Fix:** emit `delin(sp)` at the top of `BUILD_GP_CAPTABLE`. Minimal and correct rather
than a workaround — `helper_cssplit` asserts `type == LIN || NONLIN` so every split still
works, and `split` (unlike `cincoffset`) never consumes `rs1`. `sp` was delinearized by the
builder's last line anyway, so this only moves that transition earlier; the capability
handed to compiled code is unchanged.

Emitted **only when a global actually took the copy path**, so every currently-measured
rung stays byte-identical — verified by diffing generated glue against the previous
generator (`beebs_aha_mont64`: 0 differing lines; `beebs_crc32big`: gains exactly the
`delin` and a comment). The condition is derived from the emitted body, not by re-testing
the eligibility predicate, so the two cannot drift.

**Validated:** `beebs_crc32big` (2,048 B `const crc_32_tab`) returns oracle **1703161001**
through the copy path — the first time that path has worked end to end. Standing ladder
regression 6/6 green (`matmult_int` 774662735, `beebs_prime` 582955588, `beebs_bs`
887447230, `beebs_cover` 1993178309, `ctrsanity` 43260934, `beebs_aha_mont64` 2185097489).

*Previous status, kept for provenance:* the MONITOR half working, and the failure moving.
 C-11 (the
monitor could not be rebuilt) is fixed, so the monitor-side copy specified in
`plans/sqlite-on-silicon-scoping.md` is now implemented, built and running:
`create_domain` copies the image's initialized-globals bytes
`[base+GPFREE_GLOBALS_OFFSET, base+code_size)` into the front of `dom_data`, guarded so it
is skipped rather than overrunning when the image is large relative to the data region.
Source is uncommitted submodule state, mirrored at
`tests/vendor-patches/opensbi-capstone-sbi.patch`.

Evidence it works: `beebs_crc32big` (2,048 B `const crc_32_tab`, external linkage, the
rung built specifically for this path) previously **failed at domain CREATION**; it now
prints `Created domain ID = 0` and proceeds. The regression rung `beebs_aha_mont64` still
passes with the copy live (`retval = 2185097489`).

**What remains: the same `helper_cssplit` assertion (`rs1_v->tag && !rs2_v->tag`), but
later in the sequence** — no longer at creation, now after the domain exists. Static
reading of the generated glue does NOT explain it: every `split` there takes `sp` (tagged)
as rs1 and an `lcc`-derived integer as rs2, and the registers that do hold capabilities
(`t3`, `t4` in the copy loop) are re-loaded with `li` before any later split. So the next
step is to LOCATE the faulting `cssplit` rather than reason about it — QEMU aborts on the
assertion, so add a print of `rs1`/`rs2` provenance in `helper_cssplit`, or break there
under gdb, and find out whether it is in the glue at all or in the monitor's
`create_region`/`share_region` path that runs immediately after.

**One implementation trap already paid for, recorded so it is not repeated:** the copy
must index in **16-byte** units. `__linear void *` subscripting steps one CAPABILITY and
generates a 16-byte `ldc`/`stc` — `dom_seal`'s own zeroing loop uses the same convention
(`DOMAIN_DATA_SIZE = 16 * DOMAIN_DATA_N`). An earlier draft used `>> 3`, walked twice the
intended distance and stored past `dom_data`:
`Cap mem access OOB: cursor = 101562000, size = 16, bounds = (101560000, 101561020)`.

*Original entry, still accurate for the glue half:*

**Not a domain-creation bug, and not about size.** Earlier notes here (now corrected) chased
image geometry through the loader and kernel module. That was the wrong component:

> `Created domain ID = 0` appears **before** the assertion in the serial log. Domain
> creation **succeeds**; `helper_cssplit: rs1_v->tag && !rs2_v->tag` fires afterwards, in
> the **entry glue**.

**The actual trigger is a threshold in the glue generator, not a size limit.**
`gen-gp-captable-glue.py` has `COPY_THRESHOLD = 256` and picks between two paths:

| initializer size | glue path | result |
|---|---|---|
| 640 B (`sha512_k[80]`) | **large-RO copy loop** (`stor > 256`) | **FAILS** |
| 128 B (`sha512_k[16]`) | unrolled `li`/`sd` immediates (`stor <= 256`) | **passes** |

So every "size-dependent" symptom was just this threshold selecting a different code path.
The large-RO copy path is the thing that is broken; it is emitted for exactly one global in
the ladder today, which is why nothing else has hit it.

**The suspect sequence** (from the generated `.inc`):
```
lla t4, sha512_k
lla t5, __gpfree_globals_base
sub t5, t4, t5               /* blob offset = sym - base */
cincoffset(t4, sp, t5)       /* src */
cincoffset(t3, t2, x0)       /* dst */
```
`lla` on a Capstone target may not yield a plain integer, so `sub` of two such values --
and hence the operand feeding a later `split` -- is where a stray tag most plausibly comes
from. **Verify by dumping tags, not by reading:** that inference is exactly the kind that
has been wrong three times on this issue.

**Refuted along the way, recorded so nobody repeats them:** (a) `tot_size` invariant --
both images give `tot_size` 8192 and satisfy `tot_size > code_size + 1536`; (b) `code_len`
carrying the exec segment -- it is `image_size`, the whole loadable image
(`libcapstone.c:197`); (c) `dom_pages_log2` rounding -- it rounds **up** correctly
(`dom_pages == 1 ? 0 : ilog2(dom_pages - 1) + 1`).

**Experiment RUN (2026-07-28): the unrolled path is not a viable stopgap, and C-4b is
entangled with C-5.** Raising `COPY_THRESHOLD` above 640 so the big table takes the
unrolled `li`/`sd` path fails at link time:

```
ld.lld: error: unable to place section .text at file offset [0x1000, 0x2E77]
```

`.text` reaches **11,895 B** against the 4 KiB window — 640 B of data costs ~8 KB of
immediate-materialisation code, exactly the reason the copy path exists. So:
- The copy path is **necessary**, not an optimisation — it cannot simply be disabled.
- **C-4b cannot be worked around without first lifting C-5** (the 4 KiB window), or by
  fixing the copy path itself.
- Threshold reverted to 256; no code change kept from this experiment.

**The `lla`-produces-a-tag hypothesis is REFUTED (disassembly, 2026-07-28).** The emitted
glue uses plain integer addressing exactly as intended:
```
auipc t4, 0x1 ; addi t4, t4, -0x108     ; integer address of sha512_k
auipc t5, 0x1 ; addi t5, t5, -0x150     ; integer address of __gpfree_globals_base
sub   t5, t4, t5                        ; plain integer offset
<cincoffset t4, sp, t5> ; <cincoffset t3, t2, x0> ; li t6, 0x280 ; ld/sd loop
```
No capability reaches an operand that must be untagged in this sequence. That is the
**fourth** hypothesis refuted on C-4b (after the `tot_size` invariant, `code_len`, and
`dom_pages_log2`).

**New observation, unexplained:** `li t6, 0x280` (640) appears **TWICE** in the domain, at
`0x10164` and `0x10324` — two identical 640-byte copy loops, where only one global is
640 bytes. Either the glue is emitted twice, or the generator emits a duplicate descriptor.
A second copy loop would carve/copy storage a second time and could plausibly leave the
register state that the next `split` chokes on.

**Counted, and the GENERATOR IS CORRECT.** The emitted `.inc` contains exactly
**1** copy loop, **3** global headers, **4** `split`s (cap table + 3 globals) and **3**
`stc`s to the table — all as intended.

**The duplicate is BY DESIGN — this lead is refuted too.**
`start-gp-captable-generic.S` has two entry points and each expands the macro:
```
__test_reentry:  ccsrrw(sp, cscratch, x0) ; BUILD_GP_CAPTABLE  /* reentry */
_start:          ccsrrw(sp, cscratch, x0) ; BUILD_GP_CAPTABLE  /* normal entry */
```
Two copies in the image, exactly one executed per entry. Nothing wrong with it.

**Status: FIVE hypotheses proposed, FIVE refuted by measurement.** In order: the
`tot_size` invariant; `code_len` carrying the exec segment; `dom_pages_log2` rounding;
`lla` yielding a tagged value; a duplicated copy loop. Each looked sound on paper and each
died on contact with a dump, a count or a disassembly.

**What is solidly established, and is the whole of what a successor should trust:**
- Domain creation **succeeds** (`Created domain ID = 0` precedes the assertion) — the fault
  is in the **entry glue**, not `create_domain`, not the loader, not the kernel module.
- The discriminator is `COPY_THRESHOLD = 256` selecting the **large-RO copy path**, not
  image size: 640 B takes it and fails, 128 B takes the unrolled path and passes.
- The copy path is **not optional** — forcing the unrolled path for 640 B blows `.text` to
  11,895 B against the 4 KiB window, so **C-4b is entangled with C-5**.
- The generated glue is **correct by count** (1 copy loop, 3 globals, 4 splits, 3 `stc`),
  and the two copies in the image are the two entry points, by design.

**BYPASSED 2026-07-28 — C-5 dissolves C-4b.** The copy path exists only because the
unrolled `li`/`sd` alternative does not fit a 4 KiB window. Give it a **32 KiB** window and
it does, so the broken path can simply not be taken:

```
DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1 DOMAIN_OPT_LEVEL=-O1 run-ladder-qemu.sh rv8_sha512
  -> __CAPSTONE_LADDER_RV8_SHA512_PASSED__ (retval = 1390718314)
```

`rv8_sha512` now runs with its **full 640 B table** — the crypto/bitwise rung the ladder
lacked. Both knobs are **opt-in per rung, not defaults**: changing the window changes image
layout and this project has documented layout sensitivity (2026-07-26: four added
instructions flipped a passing rung), so every measured rung stays at 4 KiB and its
published number stands. `beebs_bs` and `beebs_prime` re-verified unchanged.

**C-4b remains open and still matters**: the copy path is still broken, and any initializer
needing more than ~32 KiB of unrolled materialisation will still hit it (SQLite is the
likely first). But it no longer blocks a benchmark. When someone does fix it: **instrument,
do not reason** — dump the capability tag at each `split` in the copy path. Five paper
hypotheses have failed here; the sixth should not be one.

**Related hazard — CHECKED 2026-07-28, NOT a bug.** `getGpCaptableIndex` derives its index
from a global's *position* in `M.globals()`, and GlobalMerge mutates that list (it merged
`sha_chain` + `sha_w` into one 192 B entry here), which raised the possibility of an access
lowered against the pre-merge order loading the **wrong capability slot** — silent wrong
data rather than a fault. It cannot happen: **GlobalMerge runs in `addPreISel`**
(`CapstoneTargetMachine.cpp`), i.e. before instruction selection, so `lowerGlobalAddress`
during ISel and `emitGpCaptableTable` in the AsmPrinter both see the same post-merge list.
Confirmed empirically as well — the merged-global `rv8_sha512` build and the 6-global
`beebs_cnt` both return their exact oracles, which mismatched indices would break.
**Recorded because the reasoning is the useful part:** any future pass that adds or removes
globals *after* ISel would silently break this positional scheme.

### R-12 — rev-node exhaustion is SILENT CORRUPTION, not a fault `OPEN, will bite at call_dom`

The revocation-node allocator's `head` is 10 bits (`capstone-ariane/core/anvil_build/capstone_rev_node.anvil:168`), so allocation
**#1025 wraps to node id 0 and reuses live ids**. `overflow_flag` reaches only a debug LED
(`cva6.sv:1185`) -- nothing traps, nothing prints. Only `SPLIT` and `MREV` allocate
(`capstone_dyn_unit.anvil:136, :91`); `ldc`/`stc`/`cincoffset` allocate nothing
(`:330-332, :399`, `capstone_flu_unit.anvil:29-44`).

`create_domain` does **5** splits, so this is NOT the current SQLite blocker. But SQLite's
entry glue does **1,060** splits (1 table + 1,059 globals) and will be the first domain to
cross 1,024 -- at `call_dom`, i.e. the moment after the present wedge is cleared. No
ladder rung approaches it (bigmany: 65).

### R-13 — `CINCOFFSET` duplicates a linear capability, untracked `OPEN`

It writes the unmodified `rs1` back alongside `rd` with the same `revnode_id` and
`CAP_TYPE_LINEAR` (`capstone_flu_unit.anvil:29-44`, `commit_stage.sv:278`), so one linear
capability becomes two with no bookkeeping. Sits directly next to C-14 in kind: an
instruction whose source-register behaviour diverges from what the compiler assumes.

### C-14 (superseded framing) — "a domain with MORE THAN ONE global fails" `RETRACTED`

**The split is exact.** Sorting every silicon result by the domain's global count:

| count | rungs | silicon |
|------:|-------|---------|
| 1 | beebs_primer1, bigwin, gpsz, gpcp, gptl, gpbg, gppv | all PASS |
| 2 | gpn2 | HANG |
| 4, 8, 16, 32, 64 | gpn4, gpn8, gpn16, gpn32, gpn64, bigmany | all HANG |
| 6 | gpstress | wrong value (444323487) |
| 1059 | SQLite | HANG |

**Two globals is the minimal reproducer**, established with a control in the SAME boot
(`LADDER_ONE_BOOT=1`, both transfers sha-verified, no reboot between them):
`beebs_primer1` returned 582955588 at 9775 cycles, then `gpn2` produced no END marker in
75 s. This is what the SQLite "hang" actually is; SQLite is not special.

**This supersedes the reading that the five initializer paths were validated.** `gpsz`,
`gpcp`, `gptl`, `gpbg` and `gppv` each have exactly ONE global, so none of them ever ran
the carve loop's second iteration. The paths are fine; the loop is not.

**Symptom.** `domain ID = 0` prints, then nothing — no `mcause`, `mepc`, `badaddr` or
`panic` anywhere in the capture. On silicon a monitor fault is `C_PRINT` + `while(1)` and
C_PRINT goes to the RTL trace, so a wedge and a hang are indistinguishable on the console.

**QEMU cannot see it, structurally.** gpn2, gpn4, gpn8 and SQLite are all green under
QEMU with `DOMAIN_GLUE=interp`. `helper_cssplit` keeps full 64-bit `{cursor, base, end}`
and never calls `cap_compress` (`op_helper.c:848-870`), and a tagged load overwrites the
decompressed bounds with exact ones from an out-of-band shadow map
(`op_helper.c:1128-1140`); the RTL round-trips EVERY capability write-back through
`compress_bounds` (`ex_stage.sv:1080-1098`) because the compressed form IS the
architectural register state. **A QEMU-green interp result is not evidence about
silicon.** Same shape as the DELIN divergence.

**Refuted, both without board time:**
- *Descriptor record order != cap-table index order.* `emitGpCaptableTable` and
  `emitGpCaptableInitDesc` both walk `M.globals()` with the same filter
  (`CapstoneAsmPrinter.cpp:857, 938`) and `getGpCaptableIndex` assigns indices in that
  order (`CapstoneISelDAGToDAG.cpp:134-138`). Record i IS slot i. Would have been a
  perfect no-op at count 1, hence worth checking.
- *`ldc rd, 16(gp)` is mis-decoded.* RTL uses the standard sign-extended 12-bit
  immediate added raw to the cursor, with the same address for the bounds check and the
  access and a trap on 16-byte misalignment — identical to QEMU
  (`decoder.sv:1300-1315, 1767-1770`; `capstone_dyn_unit.anvil:296-297, 318-328`).
- *Unrepresentable capability bases.* `split` sets cursor == base, selecting the
  cursorless branch where the base is exact at any alignment (see R-11).
- *Capability stack spills.* `beebs_primer1` already spills a capability
  (`stc 16(sp)` / `ldc 16(sp)` in `domain_main`) and passes.

**In flight.** `gpn2use0` / `gpn2use1` — both build a 2-entry table and run the carve
loop twice, but each reads only ONE slot (verified by disassembly): use0 reads slot 0,
use1 reads slot 1. Both pass => the fault needs two live slots. use0 fails alone => slot
0 was corrupted after being written, which points at the second store. Both fail =>
building a 2-entry table is itself fatal, and `INTERP_BUILD_LIMIT=1` then separates the
second split/store from the table split.

### C-14 — the COMPILER uses `movc` (a MOVE) for scalar register copies `ROOT-CAUSED 2026-07-30`

> **Sweep 2026-09-05 — GONE on silicon.** `gpn2` = 3976364985 = oracle and RETURNED (last in boot sw04, where it used to wedge); `gpw2` = 3983810698 = oracle. The movc-scalar-copy fix is in the flashed compiler/bitstream pair.

> **ATTRIBUTION WAS REVISED TWICE ON 2026-07-30. Read this box before the rest.**
>
> v1 "the RTL is buggy" -> v2 "the spec mandates it, the RTL is conforming, QEMU deviates"
> -> **v3 (current): the spec is UNDER-SPECIFIED here; the weight of evidence favours
> scalars being EXEMPT, so the RTL's MOVC is probably an oversight -- but this must be put
> to the board owner as a QUESTION, not an accusation.**
>
> What killed v2 (all verified in-tree):
> * `parts/mem-access-insn.adoc:45` glosses the very parenthetical v2 relied on --
>   "not **a scalar or** a non-linear capability (i.e., `type != 1`)". So in the spec's own
>   usage `type != 1` is shorthand for "scalar or non-linear", which EXEMPTS scalars.
> * `parts/mem-access-insn.adoc:105`, the one other place the consumption rule meets a
>   possibly-scalar operand (STC), writes the guard explicitly: "If `x[rs2]` **is a
>   capability and** `x[rs2].type` is not `1`". That is literally QEMU's `tag &&`.
> * `parts/prog-model.adoc:219-222`: a register holds "either a capability **or** a raw
>   `XLEN`-bit integer", so `type` is undefined for an integer and the MOVC clause's test
>   does not cleanly apply to one.
> * Spec commit `a1db3c2` ("MOVC now works with non-capabilities without generating
>   faults") removed the not-a-capability exception but never revised the consumption
>   clause -- so that clause was written when `rs1` was guaranteed to be a capability.
> * QEMU's guard is deliberate, not an accident: commit `b9c53f0d09`, subject
>   "[Capstone] movc allows scalars", is the change that added `rs1_v->tag &&`.
> * The RTL contradicts ITSELF: its STC exempts scalars
>   (`capstone_dyn_unit.anvil:408`, `if(rs2_v.metadata.cap_type != NOT_CAP)`) while its
>   MOVC does not (`capstone_flu_unit.anvil:14-25`). Internal inconsistency is the usual
>   signature of an oversight rather than a design choice.
>
> **What is NOT in doubt, through all three versions:** the mechanism (MOVC zeroes a scalar
> source on this silicon), the numeric proof, and that LLVM is emitting the wrong
> instruction. Only blame moved.

**What the spec says.** `capstone-spec/parts/cap-man-insn.adoc:33-37`, MOVC:

    * If `rs1 = rd`, the instruction is a no-op.
    * Otherwise
    . Write `x[rs1]` to `x[rd]`.
    . If `x[rs1]` is not a non-linear capability (i.e., `type != 1`),
      write `cnull` to `x[rs1]`.

Type encoding: `0` linear, `1` non-linear, `3` uninitialised, `5` sealed-return
(`parts/existing-insn.adoc:60-65`). A plain scalar is not a non-linear capability, so
`type != 1` holds and **the spec mandates zeroing the source.** `parts/intro.adoc:59-61`
states the design intent plainly: instructions "can only **move**, but not copy, linear
capabilities between general-purpose registers."

**So MOVC is a MOVE, by design.** It is the wrong instruction for an ordinary
register-to-register copy of a scalar, on any conforming implementation.

**Who is wrong, precisely:**

| component | behaviour | verdict |
|---|---|---|
| RTL (`capstone_flu_unit.anvil:13-21`) | zeroes source unless `type == NONLIN` | **spec-compliant** |
| QEMU (`op_helper.c:580-584`) | adds `rs1_v->tag &&`, exempting scalars | **deviates from spec** -- and this is what hid the bug from every QEMU test |
| LLVM (`CapstoneInstrInfo.cpp:520-523`) | emits MOVC for *every* GPR-to-GPR copy | **the actual bug** |

**Correct rule for the compiler:**
* scalar copy -> `addi rd, rs, 0` (`mv`). MOVC is simply wrong here.
* non-linear capability copy -> MOVC is correct and preserves the source (`type == 1`).
* linear capability -> cannot be copied at all, by design. MOVC moves it, which is the
  only legal semantics; the IR should never ask for a duplicate.

**STILL DO NOT PATCH THE RTL, but for a different reason than v2 gave.** Not because the
RTL is conforming -- it probably is not -- but because a reflash invalidates every silicon
measurement taken so far, is a hard stop needing approval, and the fix we control (the
compiler) is free and lossless. Ask the board owner which behaviour is normative; do not
assert that theirs is wrong.

**The LLVM bug is bigger than the scalar case.** `CapstoneInstrInfo.td:2455-2460` declares
MOVC with `hasSideEffects = 0` and `$rs1` as a pure USE with no def. LLVM therefore
believes MOVC never clobbers its source -- which is wrong for LINEAR capabilities on ANY
implementation, since every reading of the spec agrees those are consumed. Fixing only the
scalar path leaves that hole open.

**The fix is cheaper than first estimated:** `PseudoSCALAR_COPY_I128`
(`CapstoneInstrInfo.td:2446-2447`) already exists and expands to `ADDI`. The scalar-copy
machinery is in the backend; what is missing is routing scalar GPR copies through it
instead of through MOVC.

---

**Original mechanism analysis (unchanged and still correct as to WHAT happens):**

`capstone_flu_unit.anvil:13-21`, MOVC with `rs1 != rd`:

    if(data.cap_rs1.metadata.cap_type==cap_type_t::CAP_TYPE_NONLIN){
        let rs1 = data.cap_rs1;          // source preserved
        let rd  = rs1;
    } else {
        let rs1 = call create_cnull();   // SOURCE ZEROED
        let rd  = data.cap_rs1;
    }

A plain scalar is `NOT_CAP`, so it takes the else branch and the source register is
nulled (`create_cnull` zeroes cursor and metadata, `capstone_unit.anvilh:383-384`).

QEMU (`op_helper.c:580-584`) guards the same zeroing with `rs1_v->tag &&
!captype_is_copyable(...)`. A scalar has `tag == false`, so **QEMU preserves what silicon
destroys.** DIVERGENT, and invisible to every QEMU test.

**Delivery mechanism.** `copyPhysReg` emits MOVC for every GPR-to-GPR copy
(`CapstoneInstrInfo.cpp:520-523`), so ordinary register moves inherit it. The write
reaches the register file through an rs1 write-back port gated only by
`cap_result.valid` (`commit_stage.sv:278-281`), i.e. for EVERY op in `check_cap_op`.
A narrower set was evidently intended: `check_fwd_rs1` lists
`{SPLIT, MOVC, CJALR, CCSRRW, STC}` (`ariane_pkg.sv:925-931`) and is **dead code** --
defined and referenced nowhere in the tree, verified by grep. The broad gate is harmless
for ops that echo rs1 faithfully (CINCOFFSET does, `capstone_flu_unit.anvil:37-44`) and
fatal for MOVC, which writes a null.

**Both failure modes follow mechanically.** In gpn2:

    203c0: movc a4, a6       ; a4 := a6, and on silicon a6 := 0
    203c4: bne  a6, a5, back ; a6 is 0, a5 is 4 -> always taken -> INFINITE LOOP

That is the wedge: the domain never faults, it spins, which is why no capture ever showed
an `mcause`, `mepc` or `badaddr`.

**NUMERIC PROOF** of the other mode. `gpw2` ends its loop with `beq a6, a4` rather than
`bne`. With `a6` zeroed, `0 != 1`, the loop exits one iteration early and `g[1]` is never
written. Predicted checksum for `g = {1, 0}`: **3950255460**. The board returned exactly
**3950255460**. Derived before inspection, bit-for-bit.

**Scope.** Every measured rung sorts correctly: the four that pass have no `movc` whose
source is read afterwards; the nine that fail do. SQLite has 444 occurrences of the
pattern. `gpstress` has none and does NOT wedge -- it returns wrong data, so it stays a
separate defect.

**Fix is a design decision, not a one-liner.** No single instruction copies both scalars
and capabilities while preserving the source -- and per the spec, none should: copying a
linear capability is deliberately impossible. What the compiler needs is to pick the right
instruction per type:

| candidate | scalars | capabilities |
|---|---|---|
| `addi rd, rs, 0` | correct, preserves source | drops capability metadata |
| `movc rd, rs` | DESTROYS source | correct for NONLIN only |
| `cincoffset rd, rs, x0` | RTL preserves rs1; QEMU nulls it (C-4b) | same divergence |
| `cincoffsetimm rd, rs, 0` | traps UNEXPECTED_OPERAND on NOT_CAP (`:49-52`) | -- |

`copyPhysReg` cannot tell them apart -- scalars and capabilities share the GPR class. A
correct fix needs the type distinction (separate register classes, or a copy pseudo
selected by type at ISel). See `plans/c14-movc-source-destruction-fix.md`.

This is a CORRECTNESS fix, not a workaround for a hardware defect: emitting a move where a
copy was meant is wrong against the spec regardless of which core runs it.

**Retracted on the way here** (four hypotheses, all mine): more-than-one-global,
exactly-16-byte globals, unrepresentable capability bases, and stale shadow-RF metadata
poisoning cincoffset's offset. The last was refuted by the same RTL read that found the
real cause: ordinary ALU writes DO invalidate the metadata shadow entry, because the
metadata regfile shares its write-enable with the integer regfile
(`issue_read_operands.sv:1695-1709`, `commit_stage.sv:271-279`).

### C-17 — `i128 SELECT_CC` is not selectable; the SQLite domain cannot build at `-O1` ~~`OPEN — LATENT; and it was NOT the -O1 blocker`~~ `LATENT BY DESIGN 2026-09-05 — the crash is gone, a diagnostic stands in its place`

> **Sweep 2026-09-05 — crash GONE, diagnostic by design.** The pre-c128 compiler crashes on the wide arm with `Cannot select … SELECT_CC`; ae821a017089 emits the intended diagnostic instead: `Cannot materialize arbitrary >64-bit constants as capabilities`. Decision on lowering such an arm via two i64 halves is queued (B6); until then the limitation is diagnosed, not silent.

    fatal error: error in backend: Cannot select:
      t88: i128 = CapstoneISD::SELECT_CC t9, Constant:i64<10>, seteq:ch, t93, t92

Building the SQLite domain with `SQLITE_OPT_LEVEL=-O1` crashes the backend. A `SELECT_CC`
producing an **i128** (a capability) has no selection pattern. `-O0` never forms the select,
which is why `OPT=${SQLITE_OPT_LEVEL:--O0}` (`build-sqlite-silicon.sh:41`) has always been the
SQLite default — the ladder rungs, by contrast, build at `-O1`.

**This is a RECURRENCE, not a new bug.** `ISSUES.md` already records an i128 `SELECT_CC` crash
in the stage-30..34 work ("Its first attempt built nothing (i128 `SELECT_CC`) ... both are now
fixed"). Either that fix was shape-specific or it regressed; the earlier entry does not say
which shape it covered. **Check the previous fix before writing a new one.**

The shape is ordinary and will recur elsewhere: `sqlite3Strlen30` is
`if( z==0 ) return 0; return 0x3fffffff & (int)strlen(z);` — a null check on a pointer feeding a
masked result is exactly what forms `select_cc` on a capability at `-O1`.

**Why it matters beyond the crash:** `-O1` is the cheapest available shot at the R-17 blocker
(see below), and this is what blocks it.

---

**UPDATED 2026-08-26. SQLite builds at `-O1`. But C-17 is NOT closed, and an earlier version
of this very update said it was -- see the retraction below.**

**1. C-17's RECORDED REPRODUCER no longer fires. C-17 itself is LATENT, not fixed.**

`char *pick(int n, char *a, char *b){ return n == 10 ? a : b; }` compiles clean at `-O1` AND
`-O2`, as do five harder select shapes (long compare, pointer-as-condition, struct-field
condition, nested select). **RETRACTED: I first recorded that as "C-17 is closed."** It is not.
The matcher gap the entry names is still live and still reproducible on the current `llc`:

```llvm
define ptr addrspace(200) @wide_arm(i64 %n, ptr addrspace(200) %b) {
  %c = icmp eq i64 %n, 10
  %w = inttoptr i128 18446744073709551625 to ptr addrspace(200)
  %r = select i1 %c, ptr addrspace(200) %w, ptr addrspace(200) %b
  ret ptr addrspace(200) %r
}
```
    LLVM ERROR: Cannot select: t18: i128 = CapstoneISD::SELECT_CC t2, Constant:i64<10>,
                setne:ch, t4, Constant:i128<18446744073709551625>

`Select_GPRCAP_Using_CC_GPR` is still emitted under `!is64Bit()` -- the guard is visible in the
built matcher (`CapstoneGenDAGISel.inc`, `OPC_CheckPatternPredicate ... !((Subtarget->is64Bit()))`
immediately before both entries), and the prose is already in-tree at
`CapstoneInstrInfo.td:1748-1755`. What keeps the node from being FORMED in ordinary code is the
custom `lowerCapabilitySelect()` (`CapstoneISelLowering.cpp:10725-10834`), which bails at
`:10746-10747` when a constant arm has more than XLen active bits and falls through to
`lowerBranchSelect()`, which forms the unselectable i128 `SELECT_CC`.

So the surviving trigger is specifically **a select whose constant arm needs more than 64 bits**.
UNRESOLVED: whether that shape is reachable from C. The author's own comment at
`CapstoneISelLowering.cpp:10729-10731` asserts it is ("an offset that GlobalMerge + DAGCombine
sank into an i128 select"), which is a reason to treat this as latent rather than theoretical.

**The lesson is the same either way, and it is why the retraction is recorded rather than
quietly fixed:** "the reproducer stopped firing" and "the bug is fixed" are different claims, and
I collapsed them. Re-running one reproducer tells you about that reproducer.

**2. C-17 was NOT what blocked the `-O1` build.** This is the part worth reading. Once C-17
stopped reproducing, the amalgamation still failed, with a DIFFERENT backend error:

    fatal error: Capstone PureCap: Cannot materialize arbitrary >64-bit constants as
    capabilities; capabilities are unforgeable

Root cause: DAGCombiner merges runs of adjacent constant stores. On this target i128 is the
CAPABILITY carrier, so a merged 128-bit store is `stc` -- it writes a TAGGED CAPABILITY.
Reduced from `sqlite3FinishCoding`, the shape is SQLite's `VdbeOp` initialiser: six adjacent
constant stores (i8, i16, i32, i32, i32, i8) over 16 aligned bytes, merged into one i128 whose
bits are capability metadata the program never had authority to name. `0x10000000000000009`
decomposes to the four small integers 0, 1, 0, 9.

So the fatal error was the backend correctly catching a would-be FORGED CAPABILITY, not a gap
to fill in. Fixed at the merge decision (`CapstoneTargetLowering::canMergeStoresTo` refuses
i128), commit `d1fd1d33b905`. Merging up to 64 bits still happens and should.

**Evidence, A/B on the real module rather than inferred:** with the hook disabled the
amalgamation reports 24 of these errors; with it enabled, zero. lit 58/58. Pinned by
`llvm/test/CodeGen/Capstone/cap-store-merge-i128.ll`.

**RESIDUAL, found by audit and confirmed directly: `memset` still reaches the same forge.**
`findOptimalMemOpLowering` (`CapstoneISelLowering.cpp:26100`) guards every i128-avoidance
branch on `Op.isMemcpy()` (`:26131`, `:26157`, `:26179`), so a **memset** falls through to the
generic picker at `:26204`, which chooses i128 because i128 is a legal type. `canMergeStoresTo`
is never consulted. A 16-byte `llvm.memset` with a non-zero fill still produces
`Cannot materialize ... (value 0x7070707070707070707070707070707)`; reproduced at sizes 16, 17,
24, 32 and 48. **So "the i128 forge path is fixed" would be too strong -- what is fixed is the
store-merging route.** It does not affect the `-O1` result: the amalgamation has 30 `llvm.memset`
sites and NONE with size >= 16 and a non-zero fill, which is consistent with the zero-error run.
Other routes were attacked and are clean: `store <2 x i64>`, `store <4 x i32>`, a `bitcast
<2 x i64> to i128`, and a 16-byte `llvm.memcpy` all avoid it.

**THE PROCESS LESSON, which is the reusable part.** This entry asserted a blocking reason for
months and was never re-tested. Acting on it directly would have meant writing an i128
`SELECT_CC` pattern -- correct-looking work on a bug that no longer existed, which would not
have unblocked the build, because the actual blocker was somewhere else entirely. A recorded
blocker is a claim with a date on it; re-run the reproducer before building on it.

**What this does and does not do for S-12.**

The FAULT SITE is genuinely gone, and this part is solid. In `sqlite3WhereCodeOneLoopStart`,
`pWInfo` stays in callee-saved `s2` for the whole body, so the fault pair
`ldc a4, 0x0(a0)` + `cincoffsetimm a4, a4, 0xb0` becomes `cincoffsetimm a0, s2, 0xb0` with no
reload. Verified by full enumeration rather than by reading one site: `s2` is written exactly
three times in the function -- `stc s2, 0x3b0(sp)` (prologue save), `movc s2, a2` (the only body
definition), `ldc s2, 0x3b0(sp)` at `c5410`, which sits inside the contiguous restore block
`c5404`..`c543c` and is therefore the epilogue, not a cold-path reload.

**The IMAGE-WIDE count is metric-dependent, and the headline number oversold it.** "1043 -> 305"
counts an `ldc rX` with the consuming `cincoffsetimm rX` STRICTLY ADJACENT. Widening the window
(stopping at a redefinition of rX) shows -O0 barely moves while -O1 grows sharply -- the -O1
scheduler separates the pair rather than removing it:

| window | ratio |
|---|---|
| 1 (adjacent) | ~2.7-3.4x |
| 2 | ~2.0x |
| 4 | ~1.4-2.0x |
| 8 | ~1.3-1.9x |

Per-instruction density gives ~2.6x, and part of even that is the program simply being smaller.
Two further corrections to what was first written here: `.text` goes **1,305,384 -> 989,496
bytes (1.32x)**, NOT "2.2 MB -> 989 KB" -- the 2.2 MB was a figure quoted from a build-script
comment, never a measurement of this artifact. And the phrase "an `ldc` from a FRAME SLOT" is
the exemplar, not the metric: the counter is base-register-agnostic, and only ~9 of the 305 have
an `sp`/`s0` base.

**So: the specific fault site is removed; the population is reduced by somewhere between ~1.3x
and ~3.4x depending on how you count.** At a 54% per-draw wedge rate that lowers the draw rate
and nothing more. **A completing `-O1` board run MUST NOT be reported as S-12 resolved**, and
S-12's mechanism is still OPEN, so "strict adjacency is the operative property" is an assumption
and not a finding.

> **UPDATE 2026-09-04 — the paragraph above is superseded, and is kept because its reasoning was
> right at the time.** S-12 has since been root-caused (a scoreboard WAW clear that ignores
> `commit_ack`, letting a written-back-but-unretired STC entry forward `cnull` to a younger
> `ldc`), fixed in RTL, synthesised and flashed. "Strict adjacency" was indeed not the operative
> property -- store-buffer back-pressure is. The `-O1` caution still stands on its own terms:
> the reduced fault population lowered the draw rate without addressing the cause, so a
> completing `-O1` run was never evidence of a fix. See the S-12 entry above for the current
> status and for the honest strength of the post-flash verification. `OPT` stays defaulted to `-O0` until `-O1` is validated end to end; flipping
the silicon build default is a lead decision, not a side effect of this fix.

**Baseline provenance, UNRESOLVED:** the `-O0` disassembly counted here has
`sqlite3WhereCodeOneLoopStart` at `0x104248`, while the binary that produced the 2026-08-25
fault has it at `0x104788`. No artifact on disk matches `0x104788`. The SHAPE claim survives --
the four-instruction fault window reproduces at fn+0x8c, matching the record -- but these
numbers come from `/tmp/capstone/sqlite-silicon/` and not from the faulting binary.

### M-1 — domains run with `mtvec = 0`, so a domain fault is an unbreakable loop `OPEN — OURS, FIX FIRST`

> **2026-09-09, board lane — the remaining half is RTL-side, ownership moves to the RTL lane.** The
> firmware half (the trap-vector context slot written by `create_domain`) has been on silicon since
> 2026-09-02 (`tests/fpga-repros/RTL-domain-trap-vector-unset/`). Boot sw39 showed what the other half
> costs today: a domain that raised UNEXPECTED_CAP_TYPE at its INIT (`r25same`, first attempt) was not
> delivered to the monitor; the core wedged into a repeating 32-byte UART record carrying 0xdead..
> and the driver had to power-cycle. The RTL lane reads it as M-1's own path (the faulting instruction
> was the capability op itself, so R-27's window does not apply). Consequence for probes: on this RTL
> a fault inside a domain is a wedge, never a returned value, so every silicon probe must be written to
> RETURN on both outcomes or be the last arm of its boot. Nothing further for the monitor here.

> **Sweep 2026-09-05 — OPEN, NOT EXERCISED this sweep.** Boot sw13 (control k800 = 4): `tagr` 1017, `tagf` 1017 — the package's deliberate fault (`lcc` selector 1 on an untagged operand) is total now and does not fault: its second type query reads 7, which QEMU's `helper_cslcc` documents as the TOTAL type query's "not a capability" answer, encoded the same way in the RTL. So nothing faulted and the loop question was not asked. Fix known — build with `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1` (see below); verifying it needs a rung that actually faults, e.g. an out-of-bounds load, not built.

**A trap and a hang are the same observation from outside.** This is not the SQLite blocker; it
is why the SQLite blocker resisted 30+ board sessions.

Verified end to end:
* Seal slot 1 is the domain's trap vector — `csr_regfile.sv:399` (save) and `:1880-1884`
  (restore: `reg_id 1` → `ctvec_d = data[127:64]; mtvec_d = data[63:0]`). Confirmed by its
  neighbours: slot 2 is `{cscratch, mscratch}` (how `gp` is delivered), slot 3 is `mstatus`.
* The monitor never writes it — `sbi_capstone.c:801` zeroes all slots, `:823-825` write only
  0, 2 and 3.
* A trap does not install a new PCC — `frontend.sv:425-426` sets `npc_d = trap_vector_base_i`
  while `npc_metadata` is carried forward.

So a fault jumps to pc = 0 with the domain's PCC still installed, faults again on the
out-of-bounds fetch, and loops forever in M-mode with interrupts off and no UART.

**Every "no trap was reported, so it is not a fault" inference in this project is void.**

**CORRECTED 2026-08-05: the fix ALREADY EXISTS and needs NO monitor change.** An earlier
version of this entry said it required a glue handler, `dom_seal[1]` in `create_domain`, and a
new readback path. That is wrong and would send someone to write a monitor patch that is not
needed. `start-gp-captable-interp.S:760,824` already contains, behind **`INTERP_DOMAIN_MTVEC=1`**:
`lla t0, .Ldomain_trap; csrw mtvec, t0` — set from INSIDE the domain (M-mode, so `csrw` is
permitted), with `.Ldomain_trap` jumping to `.Ldomain_returned`, which already captures `mcause`
and `mtval`. Verified in a built image: `csrw mtvec, t0` at `0x102e4`, absent without the flag.

So M-1 is fixed by **building with `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1`**. What is NOT
yet established is whether the handler is reachable after a real fault — it touches
`sp`/`cscratch`, so a fault that corrupted those would make it fault again and look identical to
no handler. **Verify with `tagf`** (`fpga-repros/RTL-store-user-metadata/`, a deliberate fault):
returning with a cause proves the handler works.

Worth fixing on its own merits: **any** domain that faults for any reason is currently
undebuggable and takes the core with it.

### M-5 — the `REV_BORROWED` re-share path `C_INIT`s a revoke-derived `UNINIT` that cannot satisfy `INIT` on silicon `OPEN — LATENT on silicon, monitor; QEMU-validated only`

`shared_region_annotated`'s `REV_BORROWED` branch does `if (cap_type(r) == 3) C_INIT(r, r, 0)` on
the retained handle after a revoke. On silicon that operand cannot satisfy `INIT`: stores through an
`UNINIT` capability are bound to `[base, end-16]` and advance its cursor by 16 per store
(`capstone-spec/parts/mem-access-insn.adoc:93,:104`), so **no** store sequence reaches
`cursor > end`, and revoke leaves `cursor = start` on the RTL. The only way to reach the required
state is the `CAPTYPE` debug instruction, which production code does not use.

The path is QEMU-validated — Q-06's null-blk flow exercises it, under an emulator that wants
`cursor == end` (see Q-07, whose accepted set is disjoint from the RTL's) — and **has never run on
the board**. On silicon the first re-share-after-revoke would trap `ILLEGAL_OPERAND_VALUE` in
M-mode.

**What would settle it:** one board boot with a host that borrows a region, revokes it, and
re-shares it. No such host exists yet, which is why this is latent rather than measured. Filed
2026-09-09 by the RTL lane; line numbers verified against the sources named above.


### R-17 — a ~1.6 MB domain hangs after ANY perturbation of its image `OPEN — NOT ROOT-CAUSED`

**Reproducer:** `capstone/tests/fpga-repros/S01-image-perturbation-hang/` (has `run.sh`).

Two SQLite domain images differing by **one dead, never-called, empty function**:

    QEMU   uc.dom  stage 11 -> obs=1517161237      board  uc.dom  -> obs=1517161237  (5 obs)
    QEMU   dp0.dom stage 11 -> obs=1517161237      board  dp0.dom -> NEVER RETURNS   (2 obs)

Stage 11 executes only `sqlite3Strlen30` on a string literal and never calls the added function.
The hang is silent: no trap reported, no marker, core still services the console. **Nine**
structurally different perturbations of `uc` were built and every one hangs; only unmodified
builds (`uc`, `f10`) return.

**Attribution is NOT established.** The QEMU differential rules out a platform-independent
compiler/glue defect — a miscompiled `dp0` would fail there too, and does not — but QEMU is our
own model and is permissive where the RTL is not. The board may be correct while our software
relies on something it does not guarantee, and the difference may be timing rather than function.

**Tested and EXCLUDED** (see the package README for the artifact behind each): `.gct` size and
contents; carve count (8→208 synthetic, and `dvar` at 182); image size (`sz2048/8192/16384` are
all byte-identical in size to `dp0`); address of the executed code (`sqlite3Strlen30` is at the
**same** address in both); the amalgamation rewrite (byte-identical); run position (controlled
both ways); rev-node pool exhaustion (`head`=221/1021, `overflow=0`); bounds representability
(every carve representable, still hangs); operand forwarding (fix present in this bitstream).

**The debug mux is not diagnostic here** without a subtracted baseline: on the PASSING run it
reads byte-identical to the hanging run (`sw=255` `0x8f`, `sw=224` `0xff`, `sw=225` `0xd5`).

**Seven mechanisms were proposed and all seven retracted** during 2026-08-04/05. The recurring
cause was that every intervention which could observe the system also changed its behaviour.
Full trail in `ref/SILICON-BLOCKER.md`.

**A BETTER PROBE THAN THE HANG — sporadic wrong `strlen` results (2026-08-05).** Stages that
*return* are already wrong, which is cheaper and more bisectable than a hang:

    stage 13   board 15   expected 36 (5+8+11+12)    QEMU 36  CORRECT
    stage 16   board 124  expected 128 (128*5 & 0xff) QEMU 128 CORRECT

Stage 16 calls `strlen` on the **same** literal `"alpha"` 128 times and totals 636 instead of
640 — **4 of 128 calls returned 1**. So it is **sporadic (~3%), not length-dependent**; stage
13's `15 = 5+8+1+1` is the same effect at 2 of 4.

At `-O0` `strlen` re-loads the string capability with `ldc` **from a stack slot on every
iteration** (`ldc a0,0x0(a0)` → `cincoffset` → `lbu`, `strlen` at `0x14fc1c` in `uc`). At `-O1`
it would stay in a register — but `-O1` cannot build (see **C-17**).

**INFERRED, NOT ESTABLISHED:** wrong `strlen` → wrong hash in `sqlite3InsertBuiltinFuncs` →
corrupt chain → the stage-10 hang. Every link is measured *except* that last one. Do not treat
stage 10 as explained.

**CONFLICT that must be resolved before building on any of this:** `SILICON-BLOCKER.md` §0a8
records `stage 13 rc=0x24` = **36, CORRECT**, after the unaligned-copy fix, and states that
stages 11-14 are a *resolved* bug. Today's `f10` returns **15**. Either `f10` predates that fix,
or it regressed. Re-run stage 13 on a current build before trusting either number.

**Next:** ask whether the divergence survives on `caplifive_65536_nodes.bit` (and whether that
bitstream carries the forwarding fix). A waveform of `dp0` stage 11 around the hang would settle
in minutes what no software-visible observable here can.

### R-11 — RTL truncates a capability TOP past a 2 MiB window; QEMU never does `OPEN, not yet hit`

`compress_bounds` has two branches selected by `bounds.start == cursor`
(`ariane_pkg.sv:749`). `split` sets cursor == base on both outputs
(`capstone_dyn_unit.anvil:139-144`), so carved capabilities take the **cursorless**
branch: the base is returned as `start: cursor` verbatim (`ariane_pkg.sv:662-665`),
exact at any alignment, while the TOP is truncated DOWN to a multiple of 2**E with E set
by the highest bit at which base and top differ, floored at bit 20. E is 0 — and the
capability exact — only while base and top share one 2 MiB window.

Domains are exact **by construction** today: the module rounds the allocation to a
power-of-two page count (`capstone.c:83-84`) and the allocator returns it aligned, so
everything sits in one window. Past 2 MiB, interior splits straddle a boundary and
globals silently get SHORT capabilities. `check-repr.py` fails a build at that cliff.

The other branch (`ariane_pkg.sv:769-806`, reached once cursor != base) is the
`granule(L) = 1 << (max(0, floor(log2 L) - 12) + 3)` rule with the base truncated down —
that one is C-13, caused by the monitor's `C_SET_CURSOR`. Applying it to the glue's carve
instead was a wrong fix (765da7f8, reverted in 91685f14); do not re-derive it.

### C-5 — 4 KiB code window `OPEN`
`link-gpfree.ld` forces globals to image offset `0x1000`, capping `.text` at 4096 B. One
hardcoded number, QEMU-validated at 16 KiB and 32 KiB and silicon-validated at 32 KiB. Lifting it
is what full CoreMark and Dhrystone need. Task #62.

---

> ## READ FIRST — most OPEN `R-*` entries predate the 2026-08-04 bitstream reflash
>
> The board ran `working-caplifive-captype-fixed.bit` until **2026-08-04**, when it was
> reflashed to **`caplifive_fixed_forward.bit`** (the operand-forwarding fix,
> `capstone-ariane 7aac52f93`).
>
> **R-14 and R-16 were both that one bug** — two entries that had each accumulated sessions of
> independent investigation turned out to be the same defect, and both are now FIXED and
> archived. Every other `R-*` measured before that date is therefore **suspect**: it may already
> be fixed, and its recorded mechanism may be wrong.
>
> Treat a pre-2026-08-04 `R-*` as *unverified on current silicon* until it is re-measured. Do not
> hand one to the board owner, and do not build a theory on one, without re-running it first.
> Re-running is usually one boot.
>
> Unaffected: `C-*` (compiler) and `I-*` (infrastructure) entries, which do not depend on the
> bitstream.

## Infrastructure / procedure

### I-03 — a capability-bearing array at alignment 1 faults only when the linker lands it wrong, so `-O0` passing proves nothing `OPEN — latent, affects BOARD runs`

**Found by the compiler lane 2026-09-05 in BEEBS `ctl-stack`/`ctl-vector`; generalised and
verified here because it is a hazard for board runs, not a benchmark bug.**

Those probes store capabilities into `static char heap[HEAP_SIZE]` — declared `char`, so
**alignment 1**. At `-O0` the linker happened to place it 16-aligned and everything passed. At
`-O2` the layout put it **8-mod-16**, and the domain faulted with **"Unaligned cap access",
cause 4**. Same source, same compiler, different placement.

**So a passing `-O0` run carries no information about alignment.** The variable is where the
linker put the array, and that changes with optimisation level, with unrelated edits that move
symbols, and with any layout perturbation. It also means the *same image* can pass on one build and
fail on the next for reasons invisible in the diff — which is precisely the shape of
`S01-image-perturbation-hang`.

**The wider hazard, checked here.** `umm_malloc` — the project's real allocator — **documents** the
requirement in `umm_malloc_cfg.h:9-12`: block bodies are capability-aligned *"whenever the heap
array itself is 16-aligned"*, and a capability *"loses its tag if stored to an under-aligned slot"*.
`:42` then says it will *"rely on natural layout + a 16-aligned heap array"*.

**Nothing enforces it.** There is no `_Static_assert`, no runtime check, no alignment assertion
anywhere in `umm_malloc.[ch]` — grepped for `static_assert`, `assert`, `& 0xf` and `% 16`, all
absent. `umm_multi_init_heap(heap, ptr, size)` takes a bare `void *` and trusts it. So every caller
is one under-aligned array away from silent tag loss, and the allocator that knows the rule is the
one component that never checks it.

**For anyone running these on the board:** the fault is silicon-real, not a QEMU artefact —
it is an unaligned *capability* access, which the hardware enforces. A benchmark that passed on
silicon before can fault after an unrelated change, and the disassembly will look identical
because the difference is in the link map.

**Cheap fixes, in order of value:**

1. `_Static_assert(_Alignof(x) >= 16, ...)` on every capability-bearing backing array — turns a
   layout lottery into a compile error.
2. A runtime check in `umm_multi_init_heap` rejecting an under-aligned `ptr`. It is one `&& (ptr &
   15)` and it converts silent tag loss into a diagnosable failure.
3. `__attribute__((aligned(16)))` on the arrays themselves, which is what the compiler lane applied
   to the two `ctl` probes.

None of this is hypothetical: it already cost two benchmarks their `-O2` verdict, and the reason
they had no `-O2` verdict at all was read for a while as a compiler defect.


### I-02 — an allocated issue ID can be missing from this file, so grepping it is not a safe way to pick one `OPEN — needs an allocation convention`

**Found 2026-09-04** when a lane proposed reusing **C-25** for a newly found defect. C-25 is
already allocated — on the c128 line, by the external collaborator:

```
72c7733e2702  Add the mruby probe that found C-25, and record the issue   (2026-08-15)
738c8c94521f  Fix C-25: a pointer difference must not require its operands to be tagged
```

**And C-25 does not appear anywhere in this file** — despite a commit whose subject says "record
the issue". So the registry is missing an ID that the commit history treats as assigned, and the
obvious allocation method ("grep `ISSUES.md`, take the next free number") silently hands out a
number that is already in use. The collision was caught by a peer review, not by any check.

That makes it a **process defect, not a bookkeeping slip**: two different defects sharing one ID
is the kind of thing that survives into a paper, a handover, or a commit message, and it is
expensive precisely because both entries look correct on their own.

**Until a convention exists, allocate by checking BOTH:**

```
grep -n 'C-42' capstone/docs/ref/ISSUES.md
git log --all --oneline --grep='C-42'
```

The second is what would have caught this one.

**Worth deciding, and deliberately not decided here:** whether IDs should be allocated from a
single committed list (cheap, one file to lock) or whether a lane owns a numeric range (no
coordination, but ranges strand). The current state — allocate ad hoc and record where convenient
— is the only option that demonstrably does not work.

**Also open:** what C-25 actually was is now only reconstructable from two commit messages. It
should get a proper entry, written by whoever has the mruby context, so the registry stops
disagreeing with the history.


## Compiler / toolchain (ours)

### C-43 — under `-capstone-gp-captable`, ANY anonymous compiler-generated data faults OOB; the corpus is clean by luck `MITIGATED IN-BRANCH 2026-09-09 (compiler lane, 5d2932a941ea): every producer of anonymous unslotted data is avoided (pools via useConstantPoolForLargeInts, jump tables via areJTsAllowed, cttz via lowerCTTZNoTable) or slotted (private globals get cap-table slots); backstop guard diagnoseAnonymousConstantUnderGpCaptable with a -capstone-gpfree-constant-pools knob as its lit positive control; class record kept — slot-allocated pools remain a LEAD design item`

> **MITIGATED IN-BRANCH 2026-09-09 (compiler lane, 5d2932a941ea; detail in
> `docs/plans/bug-sweep-2026-09.md`, "C-43 — class fully characterised").** Two corrections to the text below:
> (1) the claim that a float literal faults on silicon with no warning is FALSE today — under the gp-free /
> gp-captable ABI floats, doubles and large i64 constants materialise inline (`useConstantPoolForLargeInts()`
> returns false under the ABI, the C-4 fix), no pool is formed; (2) the private/anonymous-global residual is
> CLOSED — a private `unnamed_addr` addrspace(200) constant array gets a cap-table slot exactly like a named
> global (`isGpCaptableGlobal` has no linkage filter), verified to lower to `ldc gp[i]` with a
> `.capstone_gp_table` entry, never the faulting pcrel+scc-into-.rodata form; SimplifyCFG switch-lookup tables
> are private globals and slot the same way. Backstop: `diagnoseAnonymousConstantUnderGpCaptable`
> (DiagnosticInfoUnsupported, gated on `capstoneGpFreeAbiActive`) with the hidden
> `-capstone-gpfree-constant-pools` knob whose only job is the lit positive control (`c43-anon-constant-pool.ll`
> fires, rc 1; `c43-addressable-data-ok.ll` silent). Gates: Capstone lit 102/102; the SQLite silicon corpus as
> the negative control — build rc 0, zero C-43 diagnostics, 490 real `ldc gp[i]` sites emitted, so the
> gp-captable pass demonstrably ran. Stays in this file because the slot-allocated-pools design decision is the
> lead's; nothing else remains for the compiler.

**Reported by the compiler lane 2026-09-05 from cycle-3 work under QEMU; the artifact claim
verified here.** Two instances, one mechanism.

**The mechanism, and it is by design.** Under the gp-captable ABI `gp` is bounded to the
capability table (`split(gp, sp, t1)` in the glue; QEMU prints it as one 32-byte table), and
globals are reached through `ldc gp[i]`. That works for **named** globals, which get slots and whose
initialised image is copied into `dom_data` by the monitor. It does **not** work for data the
compiler emits anonymously into `.rodata`: **jump tables** and **constant pools** have no slot, and
no capability the compiler can derive reaches them.

| instance | symptom under gp-captable | status |
|---|---|---|
| a jump table (forced through a hidden knob) | OOB at the table's own address, cause 5, `rs1 = x10`, cursor `0x10156105c` = link `0x1105c` | backend now **refuses** jump tables under this ABI (`areJTsAllowed`); the twelve `-fno-jump-tables` pins were redundant and are retired; silicon images byte-identical either way |
| the de Bruijn table behind the `cttz` fix (compiler lane's C-20) | OOB at `-O0` and `-O2`, cursor inside the pool | fixed by **avoidance**: under gp-captable `cttz` lowers to `popcount(~x & (x-1))` and no pool exists; the default ABI keeps the table. QEMU: 41 = native at both levels |

**Why nothing showed until now — checked, not assumed.** No board rung has a `cttz`, and the
`-O1` silicon SQLite domain (`sqslt1.dom`, 1031 records on the board) has **0 `LCPI` constant-pool
labels** — verified with `llvm-nm` on the artifact — while carrying a 48 KB `.rodata` of *named*
data that the cap table does reach. So the corpus passes because nothing in it happens to
generate anonymous data under this ABI. **That is luck, not a guarantee**, and it is the CLAUDE.md
class "directed tests that come back clean without ever creating the triggering condition": any
future source that makes the backend emit a constant pool — a float literal, a large switch that
survives the JT refusal as a lookup, a vector splat — faults on silicon with no warning.

**The real fix is a scope item with the lead:** place compiler-generated tables in
cap-table-managed data so they get a slot. Until then the two avoidances above hold, and the
hazard should be assumed live for any new construct.

**Related fact, same work, worth its own line:** jump-table entries on this target **must be
label differences, not absolute addresses.** A domain is linked at `0x10000` and executes at its
PCC base, so an absolute-entry switch domain halts with cause 1 at `pc = 0x10338` — one of its own
table entries. Log kept at `/tmp/capstone/board-cycle2/absentry/` (scratch).


### C-38 — the register-form `CAP_CALL` mnemonic collides with the `call` pseudo (`call a0, a1` is unassemblable) `OPEN — a backend naming decision`

`parseCallSymbol` claims `call` first. Fix is to rename the mnemonic or lower the parser's precedence; XFAIL pin: `cap-call-mnemonic.s`. Two commit messages on the validation branch that say "C-25" for this bug mean C-38.

### R-15 — a domain with a 9216-byte capability-bearing global wedges `OPEN — ATTRIBUTION RETRACTED 2026-07-31`

**Read the retraction before using this entry.** The observable is real; the mechanism
originally recorded here was wrong and has been withdrawn after an adversarial audit.

**What is actually observed.** With `SQLITE_STATIC_BUILTINS=1` (the R-14 workaround, which
moves `sqlite3RegisterBuiltinFunctions.aBuiltinFunc` from a stack array to a 9216-byte
compile-time-initialised global), a domain that does nothing but return WEDGES. Without it,
the same domain returns `rc=0`.

**What was withdrawn, and why:**

* **"Six domains in one boot."** Three ran. `run_sqlite_stages_fpga.py:120-123` breaks on the
  first wedge, so `ci-450`, `ci-550` and `ci-full` were never executed. The bracket rests on
  ctl / 200 / 350 only.
* **"The wedge is in cap-init."** NOT SHOWN, and the evidence points the other way. `ci_350`'s
  last output is `SHA5:00000002`, mid-way through the FIRST share entry — it never printed
  `SHA6`, `ECSZ`, `SQ: F/share2` or `SQ: G/enter`. `__capstone_cap_init` runs *after*
  `call_dom`, so on this run **it never executed at all**. The two earlier wedges of the same
  workaround build both printed `SQ: G/enter` before dying, so the console does deliver that
  marker at an entry wedge — `ci_350` has a materially different signature.
* **"aBuiltinFunc is implicated."** The 200→350 window admits **ten** holders, not one:
  `pragmaFunclistLine.azEnc`, `sqlite3Attach.attach_func`, `sqlite3ParseUri.aCacheMode`,
  `sqlite3ParseUri.aOpenMode`, `sqlite3Detach.detach_func`, `openStatTable.aTable`,
  `statInitFuncdef`, `statPushFuncdef`, `statGetFuncdef`, and `aBuiltinFunc`. Nothing in the
  data separates them.
* **"Control passes at 406 stores."** 403. The 406 count included three callee-save `stc`
  spills to `sp`.
* **"It is not the store count."** The comparison is confounded. `ctl` and the workaround
  build differ in far more than store count: `.data` +9216, `.bss` −10240,
  `aBuiltinFunc` moves `.bss`→`.data`, and **descriptor record 150 flips `blob_off` from the
  `-1` zero-init sentinel to `52240`** — so the entry glue goes from *zero-filling* a
  9216-byte carve to *copying* 9216 bytes into it, before cap-init is reached.
* **n=1**, one fixed order (ctl→200→350), no repeat, no order swap. The rev-node pool is a
  bump allocator with no reclamation and `ci_350` ran third.

**REFUTED 2026-07-31 (synthetic probe, board):** neither leaves-per-holder nor total
cap-init store count explains it. Five synthetic single-holder domains, one boot, no SQLite:

| leaves in ONE holder | total cap-init stores | returned |
|---|---|---|
| 40 | 446 | 40 — correct |
| 100 | 506 | 100 — correct |
| 160 | 577 | **0 — mismatch, non-monotonic, see below** |
| 300 | 733 | 255 (capped) — correct |
| 580 | 1017 | 255 (capped) — correct |

A holder with **580 leaves and 1017 total stores returns correctly**, against `aBuiltinFunc`'s
159 leaves and 596 total that wedge. So the size/count hypothesis is dead in both forms.

The 160-leaf mismatch is **non-monotonic** (160 fails, 300 and 580 pass), which points at the
probe rather than the platform — a genuine threshold cannot be crossed and then uncrossed.
n=1, not re-run, do not build on it.

**What is left of R-15:** only the bare observable — `SQLITE_STATIC_BUILTINS=1` makes a
do-nothing domain wedge, and without it the same domain returns rc=0. Every proposed
mechanism has now been refuted. The most likely remaining difference is the one the audit
surfaced and nobody has tested: descriptor record 150 flips `blob_off` from the `-1`
zero-init sentinel to `52240`, so the entry glue goes from *zero-filling* a 9216-byte carve
to *copying* 9216 bytes into it — before cap-init runs at all.

**What survives:** `-capstone-cap-init-limit` truncates in the same order
`-capstone-cap-init-print` prints (`CapstoneCapGlobalInit.cpp:213-236`; confirmed empirically
— each build's store sequence is an exact prefix of the next). All six `.dom` hashes differ,
so the flag took effect. `limit=200` returns and `limit=350` wedges — as an observation.

**Next experiments, in order:** (1) re-run `ci_350` alone, first in a fresh boot, ×3 — one
pass voids the bracket; (2) build `limit=223` vs `limit=224` and run them adjacently, the
only pair that separates `aBuiltinFunc` from the nine co-entering holders; (3) only then
bisect inside 223–381.

**Repro:** `CAPSTONE_SQLITE_STAGE=30..34` — **NOT YET RUN SUCCESSFULLY.** Its first attempt
built nothing (i128 `SELECT_CC`) and the harness reported a false pass; both are now fixed.

### R-19 — a `movc rd, zero`-sourced store leaves `compress_cap(NULL)` IN its own bank-1 slot `OPEN — trigger established on silicon; MECHANISM NOT confirmed (simulation is clean); NOT yet reported`

> **Sweep 2026-09-05 — NOT REPRODUCED on caplifive_s12fix_5097eb166 with the package's own frozen images.** Boot sw11 (control k800 = 4; `fdp0fix` first as the entry-contract control = 2609): `fdpO1` 2609, `fdpraw` 2609 = 0x0A31 exactly, no 0x08000000 residue (the README's defect word is 0x08000A31). Boot sw12: `fdp0` 2609, alone (its VA collides with fdp0fix). Both arms clean; raw lines in `tests/board-results/2026-09-05.tsv`.

**Separate from R-18 on purpose.** R-18, already reported, is the **zeroing** form: the victim is
written with `0` and counts up, and raw full-width readbacks (`craw` = `0x00000237`, `graw`, `gztr`)
show **no metadata anywhere**. R-19 is a **different observable**: the victim comes back holding
`compress_cap(NULL) + n`. Same trigger class and the same workaround clears both, but the signatures
differ, so they are tracked apart — the R-18 report already sent describes the zeroing form, and
folding this into it would misinform the owner.

**Measured on silicon**, `k800` control green in every boot, damaged arm reproduced on **three**
boots at two entry VAs:

| rung | build | returned |
|---|---|---|
| `fdp0` | accumulator initialised by `movc a0, zero; sw`, `-O0` | **`0x08000A31`** = `0x08000000` + 2609 |
| `fdp0fix` | same, initialised by `addi a0, x0, 0` | **2609** clean |
| `fdpraw` | returns the accumulator alone (no second term) | `0x08000A31` — the victim IS that slot |
| `fdpO1` | `-O1`, accumulator kept in a **register** | **2609** clean |

`0x08000000` is `compress_cap` of a null capability (`ariane_pkg.sv:754-772`) — a HARDWARE encoding
the program cannot materialise; it only ever writes `0` there. QEMU computes 2609 for the same
binary. So the **trigger** is a store whose data register carries a null-capability metadata shadow,
and the **immunity condition** is the accumulator's storage class: register-resident is clean,
memory-resident at row offset 8 is damaged.

**MECHANISM NOT CONFIRMED.** A directed Verilator test at the same geometry —
`fpga-repros/R19-movc-zero-metadata-in-slot/sim/movc-zero-self-clobber.S`, bank-1 slot at row offset
8, `movc`-zero initialiser, 64 increments, RMW row-mate, witnesses either side — returns **SUCCESS
in 1715 cycles**. The simulated RTL does not write metadata into the slot. The chain readable in the
source (`issue_read_operands.sv:1140` → `wt_dcache_mem.sv:138` → `:158`) fits every board
observation but is **not** reproduced, so it is not claimed. Untested candidates for the divergence:
the resident bitstream may not match this RTL revision; the board runs inside a capability domain
after `capenter` while the test is bare metal; or the test lacks a co-factor (no capability traffic
in the loop, no indirect calls, no cap-init).

**TWO CORRECTIONS TO WHAT WAS ALREADY SENT FOR R-18**, both found after the report went out:

1. The **`R XOR 8` splash rule is withdrawn.** It is arithmetically "the victim is 8 bytes from the
   trigger"; the corpus splits into distance-8 builds where it holds and distance-4 builds where it
   fails (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`), and distance is invariant under alignment.
2. The **dual-bank chain is not confirmed** — see the Verilator result above.

The reproducer, the trigger and the workaround are all unaffected by both.

Repro: `capstone/tests/fpga-repros/R19-movc-zero-metadata-in-slot/`.
Workaround (shared with R-18): `design/R18-workaround-movc-zero.md`.

### R-18 — a scalar in the UPPER half of a 16-byte cache row is silently ZEROED `OPEN — REPORTED to the board owner; our compiler workaround is landed and silicon-confirmed`

> **Sweep 2026-09-05 — NOT REPRODUCED on caplifive_s12fix_5097eb166 with the package's own frozen images.** Boot sw09 (control k800 = 4; c8fix first as the entry-contract control): `c8fix` 67699264, `rmB` 67699264, `c8` 67699264 = 0x04090240 (p=64, k=9, qc=576), the README's CORRECT word where the defect reads 0x04090237 (qc=567; fifteen consecutive defective boots recorded here). Boot sw10: `gzl` 590400 and `gz0` 590400 = 0x00090240 (k=9 intact; the "victim 9 damaged" signature absent), `sn0` 1000576 = correct. Frozen August interp-glue images on today's firmware, one bake per boot (C15); raw retvals with the packed oracles in `tests/board-results/2026-09-05.tsv`. "Not reproduced", not a mechanism verdict.

> **STATUS 2026-08-08.** This issue has been **REPORTED to the board owner**, and our side is
> **worked around**: `-capstone-int-zero-for-zero-copy` (see `design/R18-workaround-movc-zero.md`),
> silicon-confirmed — `c8` 567 -> `c8fix` 576, one instruction apart, control green.
> **R-18 is also EXCLUDED as the SQLite blocker** (2026-08-08, board): the SQLite hang path's five
> triggering sites are all removed by the workaround and the wedge persists unchanged.
>
> **Do not re-open this entry to record mechanism work.** A second, DIFFERENT signature found on
> 2026-08-08 — the victim holding `compress_cap(NULL) + n` rather than being zeroed — is tracked
> separately as **R-19**, because the report already sent describes the zeroing form and mixing the
> two would misinform the owner. Two corrections to what was sent are noted in R-19.


> **2026-08-08 — the "better lead" below is CONFIRMED, by a single-variable pair rather than a fit.**
>
> Three boots, each with a passing control first and every rung entering. `c8` returned 67699255 on
> all three (ten consecutive boots overall).
>
> | arm | frame | qc | p | k | `stc` | result |
> |---|---|---|---|---|---|---|
> | `c8` | 0x50 | s0−0x34 | s0−0x38 | s0−0x3c | s0−0x50 | **567** |
> | `rg16` | 0x60 | s0−0x34 | s0−0x48 | s0−0x4c | s0−0x60 | 576 |
> | `rmB` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x4c** | s0−0x60 | 576 |
> | `rmC` | 0x60 | s0−0x34 | s0−0x38 | **s0−0x3c** | s0−0x60 | **567** |
>
> **`rmB` vs `rmC` is the result.** Same frame, same victim address, same `p`, and the capability
> store two rows away in BOTH. `k` moves 16 bytes; the answer flips 576 ↔ 567. Cycle counts confirm
> all arms ran the same ~576 iterations, so nothing was cured by doing less work.
>
> That also refutes the competing reading — that what mattered was the store's row being adjacent to
> the victim's row — since `rmC` has the store two rows away and is damaged anyway.
>
> **~~The geometry, sharply:~~ RETRACTED within hours, by the very next experiment.** A "bank 1 at
> lanes L is zeroed by an RMW scalar in bank 0 at the same lanes L" refinement was recorded here on
> the strength of a 5/5 + 7/7 corpus fit. The control built to test it **refuted it**, and this is
> the seventh confound of the same class: a fit was mistaken for a mechanism.
>
> **2026-08-08 (later) — REGION IS EXCLUDED, and lane geometry is NOT the discriminator.**
> Stage 37 puts the trigger in a GLOBAL: `gc[16]`, 16-byte aligned, victim `gc[3]` at row offset 12,
> plus a second global scalar RMW'd in the same row. Board, control first, all four rungs entered:
>
> | arm | row-mate | victim | |
> |---|---|---|---|
> | `k800` | — | 4 | control OK |
> | `c8` | k @+4 | **567** | anchor — 11th consecutive boot |
> | `gnt` | `gc[2]` @+8 — bank 1, lanes 0-3 (*intended control*) | **9** | **damaged, severely** |
> | `gtw` | `gc[1]` @+4 — bank 0, lanes 4-7 (the "twin") | **567** | damaged, `c8` signature |
>
> 1. **A GLOBAL victim loses increments exactly as a stack one does.** Region is excluded by direct
>    measurement. The 2026-08-07 "must be on the domain stack" reading is dead twice over.
> 2. **The same-lanes rule is refuted.** `gnt` carries the row-mate at offset 8 — same bank as the
>    victim, wrong lanes — and is damaged *more* severely than the "twin" arm, its victim reset on
>    every outer pass rather than once.
>
> **What actually discriminates, on present evidence: the row-mate's STORE PATTERN, not its
> position.** `rmB` and `gnt` place the row-mate at the SAME offset (+8, same bank as the victim);
> `rmB`'s `p` is only incremented once per outer pass and is CLEAN, `gnt`'s is re-zeroed at the top
> of every outer pass and is DAMAGED.
>
> **2026-08-08 (later still) — THE TRIGGER IS ESTABLISHED; THE MECHANISM IS NOT.**
>
> Four arms, identical victim/row-mate addresses, identical store counts, same region, in-boot
> `k800` control and `c8` anchor in every boot:
>
> | arm | row-mate's per-pass reset store | outer-loop length | victim |
> |---|---|---|---|
> | `gz0` | `movc a0, zero; sw` | short | **9 — damaged** |
> | `gzn` | `movc a0, zero; sw` + 2 nops | **padded to match** | **9 — damaged** |
> | `gzl` | `ldc; lw; sw` — stores the VALUE ZERO from a load | padded | 576 — clean |
> | `gzs` | `lui; addi; sw` — nonzero | padded | 576 — clean |
>
> **The trigger is the capability METADATA on the store's data register.** Not the stored value
> (`gzl` stores zero and is clean), and not the outer-loop instruction count (`gzn` is padded to a
> clean arm's length and is still damaged — this was a real confound, caught by audit, and it is now
> excluded). The producing instruction is `movc rd, zero`, whose `compress_cap` of a null capability
> is `0x08000000`.
>
> **The RTL path for the trigger is traced and holds:** `movc` is a capstone-FLU op →
> `commit_stage.sv:279` writes `result_metadata` into the cap-metadata regfile under the INTEGER GPR
> write-enable (`issue_read_operands.sv:1663-1665`, `.we_i(we_pack)`) → `:1140`
> `cap_data.cap_metadata_b` is taken UNGATED by opcode → `load_store_unit.sv:1013` → `store_unit.sv:345`
> → `store_buffer.sv:173` → `wt_dcache_mem.sv:138` `st_wr_cap = |wr_user_i`. An ordinary `sw` is
> therefore classified as a capability store by VALUE.
>
> **But the recorded dual-bank chain is REFUTED as the corruption path, by direct measurement.**
> If bank 1 received `wr_user_i` (`wt_dcache_mem.sv:158`) the constant `0x08000000` would appear in
> memory. Raw, unmasked readbacks say it does not, anywhere:
>
> | probe | reads | raw value |
> |---|---|---|
> | `craw` | stack victim, `c8` geometry | `0x00000237` — clean count |
> | `graw` | global victim | `0x00000009` — clean count |
> | `gztr` | the row-mate (twin) itself | `0x00000009` — clean count |
>
> The victim is written with **zero** and counts up; no metadata value lands in the victim, and none
> lands in the twin. That also refutes the write-buffer 8-byte-merge candidate, whose specific
> prediction was `twin = 0x08000009`.
>
> **Instrument correction:** every earlier R-18 number masked the victim to 16 bits, so "lost N
> increments" and "overwritten with metadata, then counted up" were indistinguishable in all of them.
> The raw reads above settle it in favour of the former. `0x08000237` from the older `gz12` build is
> a separate observation and is not reproduced by any arm here.
>
> **Also corrected:** retraction item 3 above reads the `bar1` datum as refuting a metadata
> mechanism. That reading is withdrawn — the `lw` between `bar1`'s `movc` and its `sw` scrubs the
> destination's shadow (a non-FLU writeback carries `cap_result = '0`, `scoreboard.sv:246`), so
> `bar1` never produced a tainted store and could not have tested this. The same false premise was
> in `fdreg_kernel.h` and is fixed there.
>
> **2026-08-08 (final) — MECHANISM CONFIRMED IN SIMULATION, WORKAROUND CONFIRMED ON SILICON.**
>
> An ordinary `sw` whose data register carries capability metadata also writes its data into the
> **same byte lanes of the other bank** of its 16-byte row. Reproduced in Verilator in ~13 s
> (`scalar-store-movc-zero.S`); the RVFI trace shows only two architectural accesses to the
> corrupted slot in the whole run — the seed, and the readback returning zero.
> **The `R XOR 8` rule is WITHDRAWN (audited same day).** It is just "the victim is 8 bytes from the
> trigger"; the corpus splits into distance-8 builds where it holds (10) and distance-4 builds where
> it fails (`rs4`, `ka0`, `gnt`, `gz0`, `gzn`, `graw`), and distance is invariant under alignment so
> no carve-alignment argument rescues it. What survives are NECESSARY conditions only: the damaged
> scalar is in the trigger store's own 16-byte row, in bank 1, at offset 8 or 12. Which of two
> bank-1 candidates is hit is not predicted.
>
> **And the simulation does not reproduce the board's symptom.** At `gz0`'s geometry the sim leaves
> the distance-4 slot exact (576) and zeroes the distance-8 slot; the board damages the distance-4
> slot (`graw` = 9 raw). The sim shows a real dual-bank splash but not, so far, the board's. One
> boot with a witness at `gc+0x0` settles whether there are two effects or two different faults.
>
> **Workaround, silicon-confirmed.** `c8` and `c8fix` are the same source at the same frame
> geometry (frame 80, rmw [20,24,28], accumulator still at the damaged row offset 12), differing by
> ONE instruction — `movc a0, zero` vs `addi a0, x0, 0`:
>
> | rung | qc | cycles | |
> |---|---|---|---|
> | `k800` | — | 4743 | control OK |
> | `c8` | **567** | 44116 | damaged, 15th consecutive boot |
> | `c8fix` | **576** | 44075 | **cured** |
>
> Behind `-capstone-int-zero-for-zero-copy`, **default OFF**: flag-off is byte-identical on 4/4
> rungs, QEMU ladder 6/6 both ways, lit 47/47. Two blockers before any default-on, both recorded at
> the flag declaration: it also converts genuine null-CAPABILITY materialisations, whose ISA
> semantics under `stc` are unverified; and two lit tests FileCheck the literal `movc`.
>
> **It removes the common case, not the class.** `scan-r18-trigger.py` measures this: `c8` has 7
> trigger sites, `c8fix` has 2 — the loop`s four `movc rd, zero` gone, two register-to-register
> `movc` in `domain_main` remaining.
>
> **R-18 does NOT explain the documented silicon miscompiles.** `matmult_int` (R-1) and
> `beebs_recursion` both have ZERO trigger sites. (`matmult_int` does contain nine `movc rd, zero`
> — an earlier claim that it had none came from a `grep -c` that silently returned 0 — but none of
> them feeds a store, which is what the trigger requires.)
>
> **Two corrections to the record made in the same session.** (1) An earlier reading of this
> experiment claimed the arms' absolute addresses moved with the frame; they do not — `s0` is the
> caller's `sp` and the victim is at `s0-0x34` in every arm, so the victim's address, D-cache set and
> bank-row are EXCLUDED, not confounded. (2) The 2026-08-07 "boot 68" conclusion that the damaged
> scalar *must be on the domain stack* is the sixth confound of the same class: its global
> accumulator `gcnt[3]` is the only RMW'd word in its row, so the row-occupancy rule predicts it
> clean without invoking region at all. **Region and provenance are still untested** — every arm in
> these boots is a stack build. The next build needed is a GLOBAL scalar carrying the bank-0 twin.
>
> Trail: `history/07-08-2026_23-55-00_r18-localized-to-row-mate-traffic.md`.

> **RETRACTION 2026-08-07 (same day, before handover).** A causal chain was recorded here and in a
> defect report, and an adversarial audit refuted it. **The report was NOT sent.** What was wrong:
>
> 1. **The claimed asymmetry between the two forwarding ports does not exist.** The scoreboard-port
>    "validity gate" (`issue_read_operands.sv:765`) has `cap_result.result_metadata` in BOTH arms of
>    its ternary — it does not sanitise to zero. The proposed fix ("gate the WB forward on validity,
>    matching the scoreboard version") **would have changed nothing.**
> 2. **There is no demonstrated source of stale metadata on an ordinary store.**
>    `ex_stage.sv:1081` is `capstone_flu_result_o = capstone_flu_valid_i ? '{...} : '0`, so an
>    ordinary `addi` forwards ZERO metadata on both ports. `wr_user_i != 0` on a scalar store has
>    never been measured anywhere — board, simulation or waveform.
> 3. **Our own barrier experiment refutes it.** `movc rd, zero` does NOT write zero into the shadow;
>    `compress_cap` of a null capability is **`0x08000000`** (`ariane_pkg.sv:753-834`). So under the
>    claimed mechanism the `bar1` arm should have pinned the accumulator near zero on every
>    iteration. It returned 567, bit-identical to its `nop` control. That was a refuting datum and
>    it was read as inconclusive.
> 4. **Matched builds kill the geometry as a cause.** `c8` (qc@0x1c) loses 9 while `gp16`/`gp32`/
>    `t16` (qc@0x2c, 0x3c) are exact — same bank, same byte lanes, same instruction. Roughly 10
>    undamaged upper-half slots against 9 damaged ones. The upper-half rule is a NECESSARY
>    CONDITION, not an explanation.
> 5. **The `clobber + (576 - reset)` table is an arithmetic identity, not a fit** — two free
>    parameters per observation, so any value decomposes. It also omits the builds that do not fit
>    (+11, +330 are not multiples of 9).
>
> **A better lead came out of the audit:** all three measured reset points — 9, 72, 558 — are
> multiples of 9, i.e. they land exactly on OUTER-PASS BOUNDARIES (p ≈ 1/729 under a uniform null).
> Something that happens once per outer pass fits far better than the victim's own store: the
> `k = 0` re-initialisation, whose store at the shift8 geometry sits at `0x14` (bank 0, lanes 4-7)
> and whose dual-bank splash target is exactly `0x1c`. That is a DIFFERENT mechanism and is untested.


A plain `sw` whose address lies in the upper 8 bytes of a 16-byte D-cache row can have **its own
slot written with capability metadata instead of its data**. Where those metadata bytes are zero at
the store's byte lanes the variable is **silently ZEROED** — no trap, no tag violation, nothing in
any log. Present at the resident bitstream's commit (`7aac52f93`) and at `capstone-ariane` HEAD;
`git diff` touches none of the files involved.

**Chain (each line quoted in the report):**
`issue_read_operands.sv:690` forwards rs2 capability metadata from the **writeback port with no
validity gate** (its scoreboard-port sibling checks `cap_result.valid`) → `wt_dcache_mem.sv:138`
classifies a store as a capability store **by VALUE** (`st_wr_cap = |wr_user_i`), not by opcode →
`:230-238` a classified store writes **both** banks of the row → `:156-158` **bank 1 is the only
bank** that can receive `wr_user_i` instead of the store data.

**Evidence.** Victim in the upper half in **9/9** directly-measured builds (undamaged builds also
carry upper-half scalars, so it is a real constraint). A sentinel-initialised accumulator
(1,000,000) returns **567**, proving overwrite rather than skipped stores. Every victim decomposes
as `clobber + (576 − reset_iteration)`; one build returns `0x08000237 = 0x08000000 + 567`, i.e.
clobbered with metadata **bit 27** set. Cycle counts independently confirm iteration counts. QEMU is
correct throughout.

**Software impact.** Any `-O0` code mixing capability traffic with ordinary scalar locals is
exposed; which variable is hit depends only on where the allocator puts it. A loop-control variable
in the affected slot produces **extra iterations** rather than a wrong value.

**NOT reproduced in Verilator** at either RTL revision — the directed tests never create the trigger
(stale WB-forwarded metadata on a scalar store's rs2). Stated as the report's main gap.

Report + reproduction: `history/07-08-2026_RETRACTED_scalar-store-metadata-mechanism.md`.
Trail: `history/07-08-2026_02-30-00_nested-loop-capability-index-iteration-loss.md`.
Fix: gate the WB forward on validity, and/or classify by opcode. Both need a bitstream reflash —
the project lead's call.
---

### R-21 — `cincoffset`/`scc`/`tighten`/`shrinkto` do not consume their LINEAR source, and `init` DUPLICATES it `OPEN — SPEC VIOLATION, confirmed in RTL simulation 2026-08-11; NOT yet reported`

> **Sweep 2026-09-05 — cincoffset half GONE at 5097eb166; the INIT half is R-25.** `linear-clear-audit.S` arm 2 prints NOT_CAP (the linear source is consumed); R-25 (INIT with rd ≠ rs1 duplicates the source) is confirmed by directed test, see its entry.

**A linear capability can be copied.** `capstone-spec/parts/intro.adoc:58-61` states the invariant
normatively -- "instructions can only **move, but not copy**, linear capabilities between
general-purpose registers" -- and the spec defines each instruction below as `MOVC rd, rs1` plus an
edit, where `MOVC` writes `cnull` to a non-NONLIN source (`cap-man-insn.adoc:36-38`). `MOVC` does
this (`capstone_flu_unit.anvil:20-24`). These do not:

| insn | anvil site | `rs1` slot of `create_result_pack` | |
|---|---|---|---|
| `CINCOFFSET` | `capstone_flu_unit.anvil:41` | `rs1` unmodified | not cleared |
| `CINCOFFSETIMM` | `capstone_flu_unit.anvil:60` | `rs1` unmodified | not cleared |
| `SCC` | `capstone_flu_unit.anvil:81` | `rs1` unmodified | not cleared |
| `SHRINKTO` | `capstone_flu_unit.anvil:205` | `rs1` unmodified | not cleared |
| `TIGHTEN` | `capstone_dyn_unit.anvil:220` | `rs1` unmodified | not cleared |
| **`INIT`** | **`capstone_flu_unit.anvil:112`** | **`create_result_pack(...,rd,rd)`** | **duplicates** |
| `MOVC` / `SEAL` / `SHRINK` / `SPLIT` | | | conformant |

`INIT` is the severe one: with `rd != rs1` it writes the **newly created LINEAR capability** into
`x[rs1]`, where the spec mandates `cnull`. That is outright duplication rather than a missing
clear. `create_result_pack(id, ex, rs1, rd)` maps arg 3 to `cap_rs1` and arg 4 to `cap_result`
(`capstone_unit.anvilh:348-362`); nothing downstream rewrites it (`ex_stage.sv:1083` ->
`cva6.sv:1370` -> `scoreboard.sv:242` -> `commit_stage.sv:278` is pass-through apart from the
CCSRRW gate at `commit_stage.sv:385`).

**Repro, ~12 s, no board:** `verif/tests/custom/capstone/linear-clear-audit.S` in `capstone-ariane`,
registered in `testlist_capstone.yaml`. Run per the `rtl-sim` skill. Result on
`capstone-bootstrap` @ `aef2baa79`, **541 cycles, 0 exceptions** (not the 2000013 timeout):

| arm | what it does | result | |
|---|---|---|---|
| 0 | `MOVC`, LINEAR source -- **instrument control** | `Reg[15]: 0000000000000000` | source CLEARED |
| 1 | `CINCOFFSET`, NONLIN source -- **conformance control** | `Reg[17]: ... Type : 2` | source SURVIVED, correct |
| 2 | `CINCOFFSET`, LINEAR source -- **the probe** | `Reg[19]: ... Type : 1` | **source SURVIVED = DUPLICATION** |

Arms 0 and 1 are what make arm 2 readable: arm 0 proves `CAPPRINT` can see a cleared source at all,
and arm 1 proves the core does not clear unconditionally. Arms 1 and 2 differ in exactly one thing,
the source type, so that difference is the variable. **No pre-existing test covered this** --
`cincoffset.S:22` sets its source to `CAP_TYPE_NONLIN` first, so it is the negative case only.
QEMU is no reference either: `trans_csldc`/`trans_csstc` write no `cnull` at all.

**Impact.** One 1-cycle instruction, no exception, yields a second capability with the same
`revnode_id`. That defeats **exclusive access**, the first of the architecture's goals. Revocation
is NOT defeated -- there is no refcount and aliases share a `revnode_id`, so one `revoke`
invalidates every copy -- so do not offer revocation as a mitigation, and do not claim it breaks.

**Do NOT simply fix this in RTL.** Our own code depends on the non-conformant behaviour.
`tests/scan-linear-clear-exposure.py` scans built artifacts for sites a conformant clear would
change; over the 76-image `.dom` corpus it finds **292 hits in only four distinct shapes**, all in
the domain entry glue and repeated across 31 images:

```
BREAK   rule-A  x70  cincoffset s2, gp, zero    (gp minted LINEAR by split; read again at stc gp, 0x10(s1))
BREAK   rule-A  x70  cincoffset t6, t2, zero    (t2 minted LINEAR by split; read again at stc t2, 0x0(s2))
SURFACE rule-B  x70  stc gp, 0x10(s1)
SURFACE rule-B  x70  stc t2, 0x0(s2)
```

LLVM does not model the clear -- `CapstoneInstrInfo.td:2413` takes `CIncOffset`'s `rs1` as a pure
use -- so a conformant `CINCOFFSET` would null `gp` in the entry glue of essentially every domain
we build. A real fix needs paired compiler work and is its own project.

**Two implementation gotchas for whoever does fix it.** `check_fwd_rs1` (`ariane_pkg.sv:929-935`)
lists only `{SPLIT,MOVC,CJALR,CCSRRW,STC}`; a clear added without a matching entry is forwarded
around and is architecturally invisible (`SEAL` is already in that state). And `MOVC`'s
`data.rs1 == data.rd` guard must be kept, or `cincoffsetimm sp, sp, -96` would null `sp`.

**Counter-evidence, recorded rather than buried.** The spec never states explicitly that a named
instruction inside a numbered step executes with its side effects; the reading rests on the spec
using "Write `x[rs1]` to `x[rd]`" (SPLIT `:309-315`, MREV `:541-542`) exactly where the source must
survive and `MOVC rd, rs1` where it must be consumed, and on `CALL` (`ctrl-flow-insn.adoc:121`)
being absurd otherwise. Upstream `capstone-qemu` implements the clear for this family
(`op_helper.c`, commit `b23d516401`, 2023) -- **except `helper_csshrinkto` (`:833-847`), which does
not**, so the reference model is itself inconsistent on one instruction.

### R-22 — `stc` does not write `cnull` to its register source `OPEN — SPEC VIOLATION; NOT yet reported`

`capstone-spec/parts/mem-access-insn.adoc:105`: "If `x[rs2]` is a capability and `x[rs2].type` is
not `1` (non-linear), write `cnull` to `x[rs2]`." The RTL does not.

The decoder aliases `STC`'s `rd` field to `rs2` (`decoder.sv:1308-1314`), which looks like it could
be the clear -- **it is not.** `capstone_dyn_unit.anvil:419` (normal path) and `:410` (uninitialised
path) both build `create_result_pack(..., rs1_v, rs2_v)`, and `rs2_v` is the **unmodified** stored
capability. It cannot be `cnull`, because the same field is the store data handed to
`cap_store_ri.req` on the next line. So the `rd`-aliased writeback is a self-write of the same
value. The store-syncer return path preserves it (`capstone_unit.anvilh:587-591`).

Note for anyone re-deriving this: `commit_stage.sv:278` drives the second capability write port
from `commit_instr_i[0].rs1`, i.e. it targets **`rs1`, not `rs2`** -- and for `STC` that port is
load-bearing, delivering the uninitialised-capability cursor advance (`dyn_unit.anvil:409-411`).
The accurate statement is "no port targets `rs2`, and the `rd`-aliased port carries the unchanged
value", not "no port other than `rd`".

**Positive control that the intent is the opposite:** the *memory*-side counterpart of the same
rule (`LDC`, `mem-access-insn.adoc:54-55`) **is** implemented, over exactly the non-NONLIN type set
(`load_unit.sv:447-453`). One half of the rule is in, the mirror half is missing.

**Same caveat as R-21: do not fix this alone.** `CapstoneInstrInfo.td:2402` declares
`STC ... (outs)` -- empty -- so LLVM believes `rs2` survives an `stc`. The exposure scan finds 140
rule-B sites across the corpus. QEMU also omits this clear
(`trans_capstone.c.inc:172-193` -> `op_helper.c:1192-1199` touch only the memory map), so **no
in-tree model currently distinguishes conformant from non-conformant behaviour** and a fix should
land in both or note the divergence.

### R-24 — the FLU/DYN exception encoder is +1 off the spec, so every capability `mcause` from the execute path is wrong `OPEN — SPEC VIOLATION, direction now determinate; NOT yet reported`

> **Sweep 2026-09-05 — re-measured, unchanged.** `excode-base-audit.S` at the flashed commit 5097eb166 (detached worktree): still +1.

**RESOLVED 2026-08-12 against `capstone-spec`, and the answer is the opposite of the first guess
recorded below.** `capstone-academic-spec/parts/int-except.adoc:19-27` gives the authoritative table:

| Exception | spec code |
|---|---|
| Unexpected operand type | **24** |
| Invalid capability | **25** |
| Unexpected capability type | 26 |
| Insufficient capability permissions | 27 |
| Capability out of bound | 28 |
| Illegal operand value | 29 |
| Insufficient system resources | 30 |

With `ex_code` ordinals `NO_EXCEPTION = 0, UNEXPECTED_OPERAND = 1, ...`, the spec's numbering is
**base 23**. Therefore:

* `commit_stage.sv:205-228` (base 23) is **SPEC-CONFORMANT**.
* `ex_stage.sv:469` (FLU) and `cva6.sv:1360` (DYN) use **base 24** and are **+1 on every capability
  exception the execute path raises** — which is nearly all of them.
* `riscv_pkg.sv:349-353` encodes the same +1 error (`UNEXPECTED_OPERAND_TYPE = 25` where the spec
  says 24), so it corroborates the RTL's behaviour, not the spec.

**THE SQLITE DOMAIN IS MEMORY-STARVED, AND THE GOVERNING CONSTANT IS NAMED (2026-08-12).**
Two boots, control green in both, differing only in the heap/stack split inside the SAME 2 MiB
allocation:

| build | heap | stack | wedge site |
|---|---|---|---|
| `sqwedge` | 256 KiB | 320 KiB | `sqlite3FinishCoding+0x464`, `cincoffset a1, a1, a2` |
| `sqheap512` | 512 KiB | **57 KiB** | `sqlite3WalkExprNN+0x30`, `ldc a2, 0x10(a0)` |

Both are `mcause 25` `UNEXPECTED_OPERAND`, and **both are the same shape**: a pointer that should be
a capability is `NOT_CAP` — i.e. NULL — and the capability operation on it faults. In the second,
`a0` is `pWalker`, loaded by `ldc a0, 0x0(a0)` **from a stack slot at `s0-0x40`**, and
`sqlite3WalkExprNN` is **directly recursive** (and mutually recursive via `sqlite3WalkExprList` /
`sqlite3WalkSelect`). A 57 KiB stack under a recursive expression walker is textbook exhaustion, and
a corrupted stack slot is exactly what it produces.

**So the 512 KiB experiment traded a heap shortage for a stack shortage** — which is what the build's
own budget predicted (stack 320 KiB → 57 KiB) and why it was flagged as confounded before it ran.
The useful part is that the failure MOVED: the domain is short of memory overall, not misallocating
within a fixed budget.

**THE CONSTANT.** `module/capstone.c:24` sets `MODULE_HEADROOM = 4096 * 16` = 64 KiB, and
`capstone.c:83-85` computes `dom_tot = pow2pages(code_len + MODULE_HEADROOM)`. SQLite's `code_len`
is ~1.38 MB, so `dom_tot` ≈ 1.45 MB and rounds to **2 MiB (order 9)** — which must then hold code,
globals blob, cap table, heap AND stack. Raising `MODULE_HEADROOM` to ~1 MiB pushes the total past
2 MiB and rounds to **4 MiB (order 10)**, leaving roughly 2.6 MB of `dom_data` — enough for a 1 MiB
heap and a ~1.5 MB stack simultaneously, instead of trading one against the other.

That is a **kernel-module change** and needs the module plus the rootfs rebuilt. It is the first
change in this investigation that is plausibly sufficient rather than diagnostic.

**WHAT IS STILL NOT ESTABLISHED.** The `sqlite3FinishCoding` NULL at the 256 KiB configuration has
NOT been shown to be an allocation failure — that remains the leading hypothesis, not a finding.
Both observed failures are consistent with memory starvation, but only the second has a mechanism
tied to evidence (recursion + a stack-slot load + a 57 KiB stack). If the 4 MiB build still wedges,
memory is excluded and the NULL has another source.

**ARCHITECTURAL NOTE, worth keeping regardless of the outcome.** On ordinary hardware `base[index]`
with `base == NULL` computes an address and faults only at the dereference — often never reached,
because a branch catches the NULL first. On Capstone the ADDRESS COMPUTATION faults: `cincoffset`
and `ldc` reject a non-capability `rs1` outright. So a NULL that ordinary hardware tolerates becomes
a hard fault one instruction earlier, and any C that relies on forming (but not using) a pointer
from NULL will fault here and nowhere else.

**THE SQLITE WEDGE IS LOCALISED TO ONE INSTRUCTION (2026-08-12).** After the debug-mux change made
the latched trap `mepc` readable, two boots gave `0x828897FC` and `0x81E897FC` — **identical low
bits**, both offset **`0x897FC`** from a 2 MiB-aligned base, matching the domain's order-9 (2 MiB)
allocation. The fault is at a fixed image offset, deterministic across boots and load addresses.

**Mapping, and a correction.** The first attempt used the FILE offset and named the wrong
instruction. `libcapstone.c:174-187` lays the image out by **virtual address relative to
`loadable_start`**, not by file offset: `memcpy(image_base + (p_vaddr - loadable_start), ...)`. The
domain has one `PT_LOAD` at `p_vaddr = 0x10000`, so `VA = offset + 0x10000` = **`0x997FC`**.

That is **`sqlite3FinishCoding + 0x464`**:

```
997ec:  ldc            a1, 0x280(a1)     a1 <- a capability loaded from memory
997f0:  cincoffsetimm  a2, s0, -0x58     a2 becomes a capability
997f4:  lw             a2, 0x0(a2)       a2 overwritten with a plain integer
997f8:  slli           a2, a2, 0x4
997fc:  cincoffset     a1, a1, a2        <- FAULTS, mcause 25 = UNEXPECTED_OPERAND
```

`capstone_flu_unit.anvil:30` raises `UNEXPECTED_OPERAND` when
`rs1.cap_type == NOT_CAP || rs2.cap_type != NOT_CAP` — the base stopped being a capability, OR the
offset still looks like one.

**THE OFFSET DISJUNCT IS REFUTED, by directed test.** `cincoffset-lw-stale.S` reproduces the exact
shape — build a capability in `a2`, overwrite it with `lw` through itself, apply the same `slli`,
use it as `rs2`. It **PASSES**: `lw` DOES clear the capability metadata. The test carries a positive
control (a live capability used directly as `rs2`, which MUST trap 25 — it does, and it is the only
exception in the log) so a quiet probe means "cleared", not "the check never fired", and a control
arm that must not trap. This does not contradict `cincoffset-stale-metadata.S`, which refuted the
same leakage for `lui`+`addi`; a load is a different writeback path and had never been asked.

**SO THE FAULT IS THE OTHER DISJUNCT: `rs1` — `a1` — IS `NOT_CAP`**, i.e. whatever
`ldc a1, 0x280(a1)` returned is not a capability.

**AND THE FIXUP IS NOT WHAT UNTAGGED IT — refuted by directed test.** `fixup-tag-survival.S`
replays `BEEBS_CHUNK_COPY`'s exact per-granule sequence (two plain 64-bit stores, then `ldc`/`stc`
on top) over a granule holding a REAL capability, reloads the destination and asks its type with
the total query. It **PASSES**: the tag survives. Two controls make the pass meaningful — a
never-copied capability must report NONLIN (so a "1" is not vacuous), and a plain `ldc`/`stc` copy
must also preserve, so a failure would be attributable to the harness rather than the fixup. This
supersedes nothing: the earlier "stages 170/171 show tags survive" note was measured on a rung, and
this measures the sequence itself.

**WHAT THE INSTRUCTION STREAM ACTUALLY SAYS.** Read the whole chain, not just the faulting
instruction:

```
ldc         a1, 0x280(a1)   base = ptr->field_0x280
cincoffsetimm a2, s0, -0x58
lw          a2, 0x0(a2)     index
slli        a2, a2, 0x4     index * 16  == index * sizeof(void *) ON THIS TARGET
cincoffset  a1, a1, a2      &base[index]   <- FAULTS
ldc         a1, 0x0(a1)     base[index]
jalr        a2              ... and calls it
```

That is **an indexed load from an ARRAY OF POINTERS, followed by an indirect call**. A `NOT_CAP`
base is exactly what a **NULL `base`** looks like.

**THIS MAY NOT BE A DEFECT AT ALL — it may be an ARCHITECTURAL DIFFERENCE.** On an ordinary
machine `base[index]` with `base == NULL` computes an address and faults only when the LOAD
dereferences it, and plenty of C never reaches that because a preceding branch catches the NULL. On
Capstone the ADDRESS COMPUTATION ITSELF faults: `cincoffset` rejects a non-capability `rs1`
outright. So a NULL pointer that ordinary hardware tolerates until dereference becomes a hard fault
one instruction earlier.

**Leading hypothesis, NOT yet established: an allocation failure.** The domain's SQLite heap is
**256 KiB** (`build-sqlite-silicon.sh:42-44`, which records that 1 MiB "does not fit"). If a
`CREATE TABLE` allocation fails, SQLite sets `mallocFailed` and leaves pointers NULL, and the code
above then indexes off one. That also fits the fixup dependence: with the fixup OFF the schema text
is corrupt and SQLite bails early with `rc=11`, never reaching this path; with it ON the data is
correct, execution proceeds, and the allocation is actually attempted. It is consistent with the
project's own documented "S-04 phantom `SQLITE_NOMEM`".

**Test in progress:** rebuild with a 512 KiB heap and re-run. If the wedge moves or clears, the
allocation path is implicated. If it wedges at the identical offset, memory pressure is excluded
and the NULL has another source.

**FIRST BOOT ON `caplifive_12august.bit` (2026-08-12) — THE WEDGE NOW REPORTS ITS OWN `mepc`.**
Control green throughout; the debug-mux change works.

| domain | result |
|---|---|
| `k800` | `retval=4` — **boot VALID** |
| `s06lcc` | **`retval=171`** — the S-06 enabler survived synthesis. Also the bitstream-identity check: on RTL without it the plain-data query wedges |
| `sqfixoff` | entered and RETURNED |
| `sqwedge` | `SQ: G/enter`, **no return in 300 s — WEDGED** |

Debug mux at the wedge:

* `TRAP LOG = 0x99` → `seen=1`, **`mcause = 25`** — reproduced on the new RTL.
* **`trap mepc = 0x00000000828897FC`** (switches 196..203). **This is the first time the wedge has
  said WHERE.** The live `commit pc` still reads `0x2` junk, exactly as predicted.
* `rev_node_head = 425`, `overflow = 0` — pool healthy. Fifth independent confirmation.
* `privM = 1`.

**THE ENTER-PATH MARKERS SAY NOTHING NEW — I misread them, retracted the same day.** I flagged that
this run printed `ENT0`/`ENT1` and no `ENT2` (where `sqfixoff` printed all three in the same boot)
as evidence the wedge sits at or before the end of the domain switch rather than inside SQLite.
**Wrong.** `ENT2` is emitted AFTER `__domcallsaves` returns (`sbi_capstone.c:911-913`), so a domain
that never returns can never print it. The monitor's own comment states the decode:

```
ENT0 then silence -> died in this function before the switch
ENT1 then silence -> control genuinely left M-mode; the domain owns the wedge
ENT2             -> the domain returned; value is its result
```

`ENT0 + ENT1 + silence` is therefore the EXPECTED signature of a wedging domain, and what it does
tell us is the useful half: **control genuinely left M-mode and the DOMAIN owns the wedge** — the
monitor's setup path is exonerated. That is consistent with everything already recorded, and the
"SQLite runs deeper" framing is untouched by it.

Worth stating because the trap generalises: a marker that is only reached on the SUCCESS path
carries no information when the failure path is silent. Reading its absence as a location is the
same error as reading a timeout as a pass.

**THE MEPC CANNOT YET BE MAPPED TO AN INSTRUCTION, and the missing piece is small.** The monitor
never prints the domain's load base, so there is no way to convert `0x828897FC` into an offset in
`sqwedge.dom`. The only addresses in the whole transcript are the shared regions (`0x81D1_7000`..
`0x81D5_F000`), `0x8220_0000` and `0x8344_9156`; the domain base is not among them. **Printing
`base_addr` in `create_domain` is a one-line firmware change and needs no bitstream** — do that
before the next board session and the address becomes an instruction.

Until then `mcause 25` remains two-way (R-24): `UNEXPECTED_OPERAND` from the execute path, or
`INVALID_CAPABILITY` on the PC capability from `commit_stage`. The `mepc` is the discriminator and
is now readable; it just needs the base to be interpretable.

**THE WEDGE HAS NOW BEEN REPRODUCED AND IT REPORTS `mcause 25` ON THE WEDGE ITSELF (2026-08-12).**
One boot, control green, three domains in ascending order:

| domain | result |
|---|---|
| `k800` control | `retval=4` in 2 s — **boot VALID** |
| `sqfixoff` (library fixup OFF) | entered, **RETURNED** `SQLITE ERROR stage=create rc=11 message=malformed` |
| `sqwedge` (`SQLITE_LDC_HIGH_HALF_FIXUP=1`) | `SQ: G/enter`, **NO RETURN in 300 s — WEDGED** |

Debug-mux readout taken at the wedge, before releasing the board:

* `sw=255 TRAP LOG {seen, mcause[6:0]} = 0x99` → `seen=1`, **`mcause = 0b0011001 = 25`**.
* `sw=224` → `privM=1`, `flush=1`, `flu_ready=1`, `dyn_ready=1`, `lsu_ready=1`, `ex_commit.valid=1`.
* `sw=249/250` → `rev_node_head = 0x1a1 = 417`, `overflow = 0`. **Fourth independent confirmation
  that pool exhaustion is not the mechanism** (417 of 65536).
* `sw=230-237` → `commit pc = 0x2`, the usual junk sentinel. Says nothing.

Read with the measured decoding above, `mcause 25` from the execute path is `UNEXPECTED_OPERAND`.
**It is still not discriminated from `commit_stage`'s PC-capability check**, which is
spec-conformant base 23 and emits 25 for `INVALID_CAPABILITY`; `privM=1` means that check's gate is
satisfiable. Both readings remain live.

**A PLAN PREMISE OF MINE WAS WRONG, and it is the reason this run did not settle it.** I recorded
that the monitor would print `mcause`/`mepc`/`mtval` at the next wedge because the `EXCX` reporting
is committed and compiled into the firmware (verified by disassembling the ELF). **It did not fire:
zero `EXCX`, `MCAU`, `MEPC`, `MTVL` in the whole run.** A capability fault inside a capability domain
**wedges rather than trapping to `mtvec`**, so it never reaches `handle_exception` and the monitor
can never report it. The monitor's reporting covers traps that reach the monitor; this one does not.
Do not plan around it again.

**THE DECISIVE DATUM EXISTS IN HARDWARE AND IS NOT READABLE.** `cva6.sv:1083` latches
`recent_nontrivial_mepc_log_q <= pc_commit` in the same block that latches the mcause this run read
out — so the faulting PC *is* captured. It is never selected by the debug mux (declared at `:980`,
cleared at `:1005`/`:1026`, written at `:1083`, read nowhere). Exposing it is a few mux arms next to
the existing trap-log arm, and it needs a **bitstream reflash**, which is ask-first. Until then the
two readings of 25 cannot be separated and the faulting instruction cannot be named.

**Also settled this run, as a clean negative:** the COMPILER-side fixup
(`-capstone-memcpy-high-half-fixup`) is **neutral for SQLite**. A matched pair in the preceding boot
— 220 fixup sequences versus 1, verified by disassembly, control green — returned the *identical*
`stage=create rc=11 malformed` on both arms. The S-06 damage that reaches SQLite is in the LIBRARY
`memcpy`'s `BEEBS_CHUNK_COPY` (`SQLITE_LDC_HIGH_HALF_FIXUP`), which never consults the compiler's
memcpy hook. Only the library knob produces the wedge, exactly as `build-sqlite-silicon.sh:777-795`
already documented.

**BOTH CODES NOW MEASURED DIRECTLY, in one matched pair.** `excode-base-audit.S` traps each
exception on purpose and asserts the delivered `mcause`:

| arm | how it is provoked | delivered `mcause` |
|---|---|---|
| U | `CINCOFFSET` with a still-tagged capability as `rs2` (`capstone_flu_unit.anvil:30` raises `UNEXPECTED_OPERAND` and nothing else for that condition) | **25** |
| I | `LDC` through a capability whose revocation node has been revoked — `capstone_dyn_unit.anvil:337`, **the exact site the SQLite wedge was blamed on** | **26** |

It PASSES, with a control inside arm I confirming the same `LDC` does *not* fault before the
`REVOKE`, and with each arm asserting `trap_count == 1` so an arm that silently failed to fault
cannot pass on a stale count. The RTL's own `$display` corroborates independently: the log contains
exactly one `Exception: UNEXPECTED_OPERAND` and exactly one `Exception: INVALID_CAPABILITY`.
Negative-tested — changing arm I's expectation from 26 to 25 makes it FAIL at `fail_i_cause`.

**So the retraction is settled empirically, not by reading encoders:** the revocation-validity check
delivers 26, and the board wedge showed 25. It cannot be that site.

**MEASURED, not inferred.** `cincoffset-stale-metadata.S` builds a genuinely tagged `CAP_TYPE_LIN`
capability and feeds it to `CINCOFFSET` as `rs2`. `capstone_flu_unit.anvil:30` raises
`UNEXPECTED_OPERAND` and nothing else for that condition. The test's trap handler records `mcause`
and the self-check asserts **both** `trap_count == 1` (so the trap provably happened, and a vacuous
pass is impossible) **and** `observed_cause == 25`, failing to `selfcheck_fail` otherwise. It
**PASSES** on this RTL. So on silicon `UNEXPECTED_OPERAND` really does deliver `mcause 25`, where
the spec says it should deliver 24.

This also closes the one link in the retraction's evidence chain that was unverified — whether the
Anvil compiler assigns enum ordinals in declaration order from 0. It no longer matters: the
end-to-end behaviour is measured directly.

**Consequence for the SQLite blocker.** The retraction under S-06 stands and is now empirically
backed: an `INVALID_CAPABILITY` from `LDC`/`STC` (`capstone_dyn_unit.anvil:337`, `:404`) leaves the
execute path as **26**, and the wedge showed 25. The two live readings of an observed 25 are
unchanged — `UNEXPECTED_OPERAND` from the execute path, or `INVALID_CAPABILITY` from the
spec-conformant fetch-path check — and `mepc` still discriminates them.

**Do NOT fix by changing the base yet.** The RTL is wrong, but so are `riscv_pkg.sv`, the monitor's
expectations, and at least one directed test (`cincoffset-stale-metadata.S` asserts 25 and would
start failing). They have to move together, and the change alters every `mcause` software receives.
This needs the board owner, not a patch.

<details><summary>Superseded first analysis, kept because it was acted on</summary>

### two exception encoders disagree by one, so a single `mcause` value has two names

Capability exceptions reach `mcause` through two different encoders using two different bases:

| encoder | base | covers |
|---|---|---|
| `ex_stage.sv:469` (FLU), `cva6.sv:1360` (DYN) | **24** | every capability op's exception |
| `commit_stage.sv:205-228` (PC-capability check) | **23** | the fetch capability only |

`riscv_pkg.sv:349-353` agrees with base 24 (`UNEXPECTED_OPERAND_TYPE = 25`, `INVALID_CAPABLITY = 26`,
`UNEXPECTED_CAPABLITY_TYPE = 27`), so `commit_stage.sv` is the one out of step, and its own comments
state the base explicitly (`64'd25; // INVALID_CAPABILITY (23 + 2)`), so this is not a
misreading — it is written that way.

**Consequence: EVERY capability `mcause` in the overlap has two readings.** Write it out, because
this table has now produced TWO published misdiagnoses and guessing from one encoder is exactly how:

| `mcause` | execute path (base 24) | PC-capability check (base 23) |
|---|---|---|
| 25 | `UNEXPECTED_OPERAND` | `INVALID_CAPABILITY` |
| 26 | `INVALID_CAPABILITY` | `UNEXPECTED_CAP_TYPE` |
| 27 | `UNEXPECTED_CAP_TYPE` | `INSUFFICIENT_PERMISSION` |
| 28 | `INSUFFICIENT_PERMISSION` | `OUT_OF_BOUNDS` |
| 29 | `OUT_OF_BOUNDS` | — |
| 30 | `ILLEGAL_OPERAND_VALUE` | — |

**Both misdiagnoses came from assuming one encoder.** The SQLite wedge's `mcause 25` was named
`INVALID_CAPABILITY` off the stale enum comment (three investigations lost). Then a tidy-up pass
corrected `SILICON-BLOCKER.md`'s `mcause 28` from `OUT_OF_BOUNDS` to `INSUFFICIENT_PERMISSION` —
applying the execute-path encoder as if it were the only one, when `commit_stage.sv:223` emits 28
for `OUT_OF_BOUNDS` in as many words. **The original reading was live all along.** That correction
has itself been corrected; the entry there now says UNRESOLVED.

**The rule: a capability `mcause` names a fault only once you know WHICH ENCODER produced it.** The
discriminator for both is the latched trap `mepc` (switches 196..203 on the new bitstream) — under
the fetch-path reading it lands outside the domain's code bounds, under the execute-path reading it
lands on a capability instruction whose operand can be inspected.

**This is not theoretical — it has already cost.** The SQLite blocker's `mcause 25` was named
`INVALID_CAPABILITY` and three investigations were spent on the revocation subsystem before the
arithmetic was checked. See the retraction entry under S-06.

**NOT fixed here, deliberately.** Changing either base changes the `mcause` values software
actually receives, so it needs a decision rather than a patch: the monitor, any handler, and the
directed tests that check `mcause` (e.g. `cincoffset-stale-metadata.S`, which expects 25 for
`UNEXPECTED_OPERAND` and would be correct under base 24) all have to move together. The
off-by-one *comments* in the `ex_code` enum that caused the misnaming ARE fixed
(`capstone_unit.anvilh`, comment-only).

Verified against `capstone-spec`: **NOT YET.** The spec's exception numbering has not been checked
against either encoder, so which base is *correct* — as opposed to which is in the majority — is
still open. Do that before proposing a fix.

*(That check has since been done — see the top of this entry. The answer inverted the expectation:
the majority encoder is the non-conformant one.)*

</details>

### UNRESOLVED — two open questions from the same audit, recorded as questions and NOT as findings

Neither is asserted. Both need a directed test before they are anything.

**U-1: atomics are excluded from the shadow-tag update, a candidate capability-forgery path.**
`wt_axi_adapter.sv:143-145` gates the tag write on `dcache_data.rtype inside {DCACHE_LOAD_REQ,
DCACHE_STORE_REQ}`; `DCACHE_ATOMIC_REQ` (issued at `wt_dcache_missunit.sv:553`) is **absent**, so an
AMO writes DRAM without writing the shadow-tag byte, and self-invalidates the L1 line. `RVA` is
enabled in the active config. That composes to: `stc` a real capability to `X`; `amoswap.d` an
arbitrary value to `X+8`; `ldc` from `X` refills with the tag still set and the attacker's word
promoted to `ruser` (`wt_dcache_mem.sv:304`) -- a capability with chosen `cap_type`, `perm`,
`bounds` and `revnode_id`, which would be materially worse than S-06. **Unestablished:** whether the
AMO write lands at `X+8` through the FPGA interconnect; whether the `ldc` genuinely misses and
refills rather than being satisfied from the write buffer; and whether the result survives
`get_node_query_validity` (`capstone_dyn_unit.anvil:333`). Any one of the three kills it. The test
needs four arms in one image, and the arm that matters most is the positive control -- `stc` then
`ldc` with no interference, which **must** show a tag, or a clean probe arm proves only that the
instrument is dead.

**U-2: `CJALR`'s `rs1`/`rd` handling.** `capstone_flu_unit.anvil:223-232`: the `rs1 == rd` branch
writes `cnull` to both slots; the `rs1 != rd` branch passes `rs1` through **uncleared** and still
puts `cnull` in `rd`. `ctrl-flow-insn.adoc:37-39` requires `x[rd] <- pc` always and `cnull ->
x[rs1]` when `rs1 != rd`. Where (or whether) the pc capability reaches `x[rd]` for `CJALR` was not
traced, so this is not a defect claim. Separate signature -- **must not** be folded into R-21.

**Not a defect, recorded so it is not re-raised.** The two shadow-tag write paths (`wt_dcache_mem.sv`
`:419` store, `:412` refill) were suspected of using divergent predicates, with `wr_cl_user_i[7:0]`
read as the low byte of `bounds`. **REFUTED.** On the refill path those eight bits are a shadow-tag
byte of `0x00`/`0x01`: `wt_axi_adapter.sv:441-442` zeroes the word and writes one byte of
`tag_wr_value_q = is_cap_req = |dcache_data.user` (`:196`, `:402`), and `:731-734` reads exactly
that byte back. The AXI USER sideband carries nothing (`:204`). The two gates are the same
predicate over different encodings. This also withdraws the eviction hypothesis that S-06's
`FIX-PROPOSAL.md` once offered for the SQLite wedge.

**Cosmetic, non-security:** `ariane.core:48`, `Bender.yml:114` and `src_files.yml:50` all name
`core/capstone_dyn_unit.sv`, which does not exist; the file that reaches the build arrives via
`core/Flist.cva6:138` -> `core/anvil.Flist`.

---

## QEMU CORE SUITES WERE RED (2026-08-14) — ROOT-CAUSED AND FIXED: an unconditional gp carve

**RESOLVED 2026-08-14.** Fix: `capstone-sbi` `1a926b0` (carve gp only for images that declare a
globals region) + `caplifive-buildroot` `b098a39` (the build dependency that was hiding it).
Verified: smoke returns 42, coremark validates its CRC.

**The cause was NOT the compiler**, and not the S-06 guard, and not a stale LLVM build directory.
The monitor's `create_domain` split every domain's code capability at `base + gpoff`
unconditionally, and `gpoff` fell back to `GPFREE_GLOBALS_OFFSET` (0x1000) whenever the image did
not declare a globals boundary. Domains linked with `my_first_domain/link.ld` — about 78 build
scripts, i.e. the whole core tier — have no globals at 0x1000, so all of them were carved wrongly,
in two shapes that look nothing alike:

| image size | symptom |
|---|---|
| `< 0x1000` | the SPLIT is itself out of range; QEMU asserts in `helper_cssplit` and aborts before the domain exists. Measured on the 416-byte `write_42` domain: parent `[base, base+0x1a0)`, `mid = base+0x1000`. |
| `>= 0x1000` | the split succeeds and silently TRUNCATES the code cap to 4096 bytes; the entry glue's first access past it faults. Measured on coremark (27084-byte image): `bounds = [base, base+0x1000)`, access at `base+0x6278`, cause 5. |

One cause, both symptoms, whole tier.

**Why it took so long, and the two things that actually cost the time.** First, the `.c.S`
intermediate the monitor is compiled through did not depend on the monitor source
(`sbi_capstone_dom.c` is a one-line `#include`), so the first fix was applied, rebuilt, re-tested,
and produced a byte-identical failure at the identical pc — the firmware kept its exact previous
size, which is the clearest possible evidence that nothing was rebuilt. Second, `helper_cssplit`
aborted with no operands and no pc, so the failing split could not be attributed; adding
`mid`/parent-bounds/pc to it named `create_domain` in a single run. Both are fixed
(`caplifive-buildroot` `b098a39`, `capstone-qemu` `f462a68b80`).

**The superseded hypothesis, kept because it was wrong in an instructive way.** What follows was
written before any of the above and reasoned from the WRONG ONE OF THE TWO MONITOR COPIES: the
`package/capstone-sbi-domain/` copy has no gp carve at all, so its `create_domain` looked innocent.
The firmware is built from `components/opensbi/`. Checking which source the artifact was built from
would have cost a minute.

**VALIDATION AFTER THE FIX (2026-08-14, all on QEMU):**

| suite | before | after |
|---|---|---|
| `smoke` | CSSPLIT assert | PASS (`retval = 42`) |
| `coremark` | cap fault, PCC truncated | PASS (CRC validated) |
| `authority`, `revoke-on-free`, `borrow-cost`, `intra-domain-mrev` | FAIL | PASS |
| `revoke-matrix`, `hier-revoke`, `shared-region`, `linear-uninit-corpus` | PASS / not run | PASS |
| `tree-cost-O2`, `rv8` | FAIL / not run | PASS |
| `beebs` | FAIL | **52/52** — one test (`prime`) failed with `<no serial output captured>`, zero capability faults, and PASSED on a clean retry, so it was an infra flake |
| `static-cap-globals` | FAIL | still exits 1 — see below, it is an INVERTED probe and this is not a regression |

`static-cap-globals` expects its static-const arm to FAIL as a known limitation. All four of its
domains are under `0x1000`, so before the fix every arm aborted in the split and the probe could
never have reported its intended verdict at all. Now all four run and the static arm returns
`305397871` instead of faulting. The probe only checks fault-vs-no-fault, so "succeeded" may be
masking a wrong value; it needs a correct expected value before its expectation is touched.
**Deliberately left alone rather than flipped.**

**THE SAME DEFECT IS PRESENT, DORMANT, IN THE SILICON MONITOR — deliberately NOT fixed here.**
`caplifive-system/.../capstone-sbi/sbi_capstone.c:659` has the identical
`unsigned gpoff = GPFREE_GLOBALS_OFFSET;` fallback and `:882` the identical unconditional
`__split(dom_code, base_addr + gpoff)`. It does not bite today because every domain currently run
on the board is a gp-captable build that declares its globals offset, so `gpoff` is genuinely
supplied. It WOULD bite the moment a board rung is linked without globals — and note that the
existing `__pad` convention ("keep the image above 0x1000 so the monitor SPLIT is non-degenerate")
is a workaround for exactly this defect, not an independent requirement.

Not fixed in the same change because it means rebuilding board firmware, which invalidates the
bitstream/firmware pairing every current silicon measurement was taken against. It should be fixed
deliberately, with a re-baseline, not as a side effect of a QEMU repair.

---

### Superseded record (2026-08-14, pre-root-cause)

**Not caused by the S-06 granule guard.** That flag is `cl::init(false)` and is referenced by
exactly one build script (`benchmarks/sqlite/build-sqlite-silicon.sh`); no suite below passes it,
and `grep -rl guard-cap-granule` over the suite scripts returns nothing. The SQLite silicon domain
passes under QEMU with the guard both ON and OFF (hashes `c08aeaa614ac61e4` vs `f1214600d0dac351`,
`.text` differing by ~34 KB, both reaching `__CAPSTONE_SQLITE_EXTENDED_PASSED__` and
`__CAPSTONE_SQLITE_MEMORY_PASSED__`).

**How it surfaced.** `llvm/cmake-build-debug` was STALE relative to committed sources; rebuilding
`llc`/`clang` brought it up to date and the suites went red. There are no uncommitted LLVM changes,
so this is committed work whose effect was hidden by an out-of-date build directory. Anyone
trusting a green suite run from before 2026-08-14 15:05 should re-run it.

**State (core tier, `run-nightly.sh --skip-build`, incomplete):** `revoke-matrix` PASS; `lit` PASS
(48/48). FAIL: `authority`, `smoke`, `coremark`, `rv8`, `beebs`, `revoke-on-free`, `borrow-cost`,
`tree-cost-O2`, `static-cap-globals`, `intra-domain-mrev`. NOT RUN: `hier-revoke` (was still
running), `shared-region`, `linear-uninit-corpus`, and the whole `--extended` tier.
Logs: `/tmp/capstone/nightly-20260814_150736/`.

**Measured signature** (`smoke.log`): the domain reports `Segment size = 1a0` — a **416-byte**
loadable image — and QEMU then aborts in
`helper_cssplit: Assertion 'mid > rs1_v->val.cap.bounds.base && mid < rs1_v->val.cap.bounds.end'`.
So a SPLIT is being taken at a `mid` outside the code capability's bounds.

**HYPOTHESIS, not yet confirmed:** the monitor's `create_domain` splits the code capability at a
FIXED offset, and an image shorter than that offset puts `mid` past `end`. This project has already
recorded the countermeasure — "`__pad` keeps image > 0x1000 so the monitor SPLIT is non-degenerate"
— which implies the fixed offset is `0x1000` and that tiny domains rely on padding to clear it. A
committed compiler change altering code size for very small domains would then unmask exactly this.
**I did not locate the monitor's split site to confirm the offset**, so treat the mechanism as
unverified; the 416-byte image and the assertion text are the measurements.

**Cheapest next step:** confirm the monitor's split offset, then check whether the failing domains
lost padding or fell under it. If so this is a latent fragility (fixed split offset vs. variable
image size), not a miscompile, and the fix belongs in the padding or the monitor rather than in the
compiler.

## How to add an entry

**When an entry's status becomes final, move it to `ISSUES-ARCHIVE.md` in the same commit that records the finding, verbatim. IDs are never reused, so an ID absent from this file is in the archive.**

One heading per issue with: a one-line statement of the behaviour, a **runnable repro**, the
evidence note, what has been tried, and the impact. Board reproducers go in
`tests/fpga-repros/R<nn>-<slug>/` — **committed, never `/tmp`**, which loses them on reboot
and makes them unreviewable. Keep frozen `.dom` images with the package when they are small
(the exact binary that reproduced is the point); when they are megabytes, ship the source
plus the rebuild command instead. An issue without a reproducer is
a rumour — write the probe first. Every probe must be **QEMU-verified before the board** so a
board deviation is unambiguous, and must **return a diagnostic rather than hang** (a hung domain
reports nothing at all).
