# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure,
**S-n** = **unattributed** — reproducible, but origin NOT yet established (may be RTL, compiler or
software). An S-n is promoted to R-n/C-n only when the origin is demonstrated, never on suspicion.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-09-09 — the registry was split on this date: this file lists only what is still OPEN; resolved and retracted entries are in [`ISSUES-ARCHIVE.md`](ISSUES-ARCHIVE.md), verbatim and by ID. R-26 and R-27 archived 2026-09-09 (fixed in RTL, in the flashed bitstream 66c4e7517).

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

> # ⚠ RETRACTED 2026-09-10, the same day it was made. THE RULING BELOW IS WRONG AND Q-04 REMAINS A GENUINE SPEC QUESTION.
>
> **The error, plainly: I mixed two numbering systems.** The ruling argued *"`NOT_CAP` is type 0, and
> `0 != 1`, so the source is nulled."* **The spec has no `NOT_CAP` type at all.** Its table
> (`capstone-spec/parts/prog-model.adoc:177-184`) is Linear 0, Non-linear 1, Revocation 2,
> Uninitialised 3, Sealed 4, Sealed-return 5 — in the SPEC's numbering `0` is **LINEAR**. `NOT_CAP = 0`
> is the RTL's enum, which inserts it at 0 and shifts everything up; the RTL says so itself at
> `capstone_dyn_unit.anvil:182-183`: *"the spec numbers types with no NOT_CAP, the RTL inserts it at 0
> and shifts"*. So the syllogism that was the whole argument evaluates a spec sentence with RTL
> constants.
>
> **This is the exact mistake that cost boot sw39** and that the project's own memory note "RTL
> cap-type numbering ≠ spec" exists to prevent. It was made anyway, in a ruling, against the file that
> records it.
>
> **And the ambiguity the ruling denied is real, on four independent grounds** (all re-verified):
> `cap-man-insn.adoc:16` says *"**Capabilities** can be moved between registers with the MOVC
> instruction"* and its operands are annotated `(C)`; spec commit `a1db3c2` removed MOVC's
> *"`x[rs1]` is not a capability"* exception and **left the consumption clause untouched**, so the
> clause was written under a precondition that was later deleted; `mem-access-insn.adoc:45` uses the
> same `(i.e., type != 1)` gloss for a scalar-EXCLUDING condition; and `:105` writes the guard out
> longhand as *"is a capability and `x[rs2].type` is not `1`"*, which is exactly QEMU's `tag &&`.
> CINCOFFSET is even defined in terms of MOVC (`:74-78`) while requiring a capability operand (`:66`).
>
> **What stands from the ruling:** the RTL really does null a NOT_CAP source
> (`capstone_flu_unit.anvil:13-26` — anything not `CAP_TYPE_NONLIN` is nulled), and that is
> board-confirmed by this entry's own reading. What does not stand is that the spec settles the
> question, or that QEMU is simply wrong.
>
> **Q-04 is therefore what it was before: a spec question, and the lead's with the spec's owners.** The
> substantive framing is that scalar-exemption would be a RESTORATION of the clause's original
> precondition rather than an amendment — which is the opposite of what the ruling said, and which
> makes it the cheaper option rather than the more expensive one. **C-14 does NOT depend on it** (see
> that entry).
>
> The ruling is left below rather than deleted, because the reasoning that failed is the useful part.

> **RULING 2026-09-10 (board lane), from the spec text itself. SCALARS ARE NOT EXEMPT. QEMU is the
> outlier and QEMU is what changes.**
>
> `capstone-spec/parts/cap-man-insn.adoc:34-38` is the whole definition and it is not ambiguous:
>
> ```
> * If `rs1 = rd`, the instruction is a no-op.
> * Otherwise
> . Write `x[rs1]` to `x[rd]`.
> . If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to `x[rs1]`.
> ```
>
> `NOT_CAP` is type 0, and `0 != 1`, so the source is nulled. There is no scalar carve-out in the
> text, and the RTL implements it literally. The claim that the spec is "under-specified on whether
> scalars are exempt" (raised while dispositioning **C-14**) does not survive reading the paragraph:
> it is under-*motivated*, which is a different thing.
>
> **Consequences, both of which follow immediately:**
> 1. **Q-04 is a QEMU fix**, not a spec question — make `helper_csmovc` null a NOT_CAP source. Gate:
>    the tier, plus a grep for MOVC-of-scalar shapes first (the Q-05 experience: find who leans on the
>    divergence before removing it).
> 2. **C-14 is a compiler fix**, and its attribution is no longer pending. Using `movc` as a scalar
>    register copy destroys the source on spec-conformant hardware. The symptom being "gone on
>    silicon" is not the same as the codegen being correct — it means the current rungs happen not to
>    read the source afterwards.
>
> **The counter-argument, recorded because it is reasonable and is NOT being adopted here.** Nulling a
> scalar serves no security purpose: there is no linearity to preserve in a non-capability, so a
> narrower rule (`type != 1 && the operand is a capability`) would lose nothing and would make
> `movc`-as-`mv` legal. That is a genuine SPEC AMENDMENT proposal, and it is the project lead's and the
> spec's owners' to make — not a lane's, and not something to assume by leaving QEMU divergent. Until
> such an amendment exists, the implementations should agree with the text they have, because a silent
> three-way disagreement between spec, RTL and emulator is worse than either rule. Filed as the open
> question, not as a blocker.

`capstone-spec/parts/cap-man-insn.adoc` (MOVC): "If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to `x[rs1]`" — a NOT_CAP source qualifies, and the RTL does it (`capstone_flu_unit.anvil:13-26`, rtl-oracle 2026-09-04). QEMU's `helper_movc` nulls rs1 only under `rs1_v->tag && !captype_is_copyable(...)` (`op_helper.c:580-585`), so an untagged source survives a `movc` under QEMU and dies on silicon. Consequence: every copy of an integer-bridged pointer that stays live passes under QEMU and loses its value on the board (C-32, XFAIL `c32-movc-untagged-live.ll`); QEMU is a permissive oracle for that whole class until this is aligned with the spec. Fix belongs in `capstone-qemu`; the compiler side is C-32.


## Q-07 — QEMU's `INIT` requires `cursor == end` and aborts the host process otherwise; the spec and the RTL require `cursor > end` `OPEN — QEMU divergence, filed 2026-09-09 from the R-25 probe work`

> **2026-09-10, SECOND PASS — THE QUESTION IS ANSWERED, AND NONE OF THE THREE OPTIONS ON THE TABLE
> WAS THE RIGHT ONE.** The choice was posed as: amend the spec's `INIT` precondition, add a
> monitor-side reclaim that avoids `INIT`, or change what `revoke` leaves behind in the RTL. It rested
> on a false premise. See **R-30** and **R-31**: `INIT` is unreachable on this silicon for ANY UNINIT
> capability (the cursor tops out at `end`, `INIT` demands `> end`), and REVOKE's permission clause is
> inverted, so for the RW regions this path handles REVOKE returns LINEAR and the monitor's `C_INIT` is
> never even reached.
>
> **The resolution, in order, and only the first step is a decision:**
> 1. **Declare whether `end` is inclusive or exclusive.** The spec never says and the RTL is split
>    against itself. Recommended EXCLUSIVE — every access path and all of QEMU already assume it.
>    **The lead's, with the spec's owners.**
> 2. **Fix `INIT`** to match (R-30): under the exclusive reading, `capstone_flu_unit.anvil:139`
>    `<=` → `<`, with `cap-man-insn.adoc:421` changed alongside it.
> 3. **Fix REVOKE's permission clause** (R-31) — but NOT before step 2, or a silent disclosure becomes
>    a live monitor trap.
> 4. **Then QEMU aligns to the fixed RTL**, which is what Q-07 becomes: `helper_csrevoke` leaves the
>    UNINIT cursor at BASE like the RTL rather than at `end`, and `helper_csinit` raises exception 29
>    instead of asserting, using the corrected comparison. The three-way disagreement collapses to one
>    rule.
> 5. **M-5's monitor code is then reachable and can be judged on its merits.** It is not fixable before
>    the above and is not the thing to fix first.
>
> The original coupling note stands below, because it is what made the search necessary.

> **2026-09-10 — Q-07 AND M-5 ARE ONE SYSTEM, and neither can be fixed alone. Verified in the QEMU
> source, not inferred.**
>
> `helper_csinit` (`capstone-qemu/target/riscv/op_helper.c:1198-1200`) is three host `assert()`s, so a
> wrong operand `SIGABRT`s the emulator instead of raising a guest trap. All three map one-to-one onto
> spec exceptions (`capstone-spec` `parts/cap-man-insn.adoc:415-421`: 24 unexpected operand type, 26
> unexpected capability type, 29 illegal operand value), and the idiom to replace them with sits five
> lines away in the same file at `:728`. That part is mechanical.
>
> **What is not mechanical:** `helper_csrevoke` at `:920-921` places the retained handle's cursor at
> `end` *specifically to satisfy this assertion*, and says so —
> *"UNINIT … cursor at END, the canonical UNINIT form that csinit requires (csinit asserts
> cursor==end). Leaving it at base produced a handle no instruction could advance — scc rejects UNINIT
> and csinit rejects cursor!=end — so a linear borrow could never be re-lent after revoke."* It labels
> itself an *"Experimental revocation-semantics choice"*. The RTL instead leaves `cursor = start`,
> which is why **M-5** is silicon-dead: no legal instruction sequence takes UNINIT(cursor=start) to LIN
> on hardware except the `CAPTYPE` debug op, which production code cannot use.
>
> **So the order is forced and the two must land together.** Fix Q-07 alone and QEMU's own
> revoke → init → mrev flow breaks. Fix M-5 alone and there is nothing on QEMU that reproduces it.
> Aligning `csrevoke` to the RTL's `cursor = start` in the same commit converts M-5 from
> silicon-latent-and-unmeasurable into QEMU-reproducible with an existing gate — and
> `run-nullblk-all.sh` then goes red and STAYS red until M-5 is resolved, which is correct rather than
> an obstacle, but means **neither may be pushed to `dev` before M-5's resolution is chosen**.
>
> **The decision, which is the lead's with the RTL lane** (M-5 has two sites, not the one the entry
> names: `sbi_capstone.c:1196-1197` and `:1340-1341` in `share_child_region`): amend the spec's `INIT`
> precondition; add a monitor-side reclaim that does not route through `INIT`; or change what `revoke`
> leaves behind in the RTL. Ship `>` rather than `>=` whichever way it goes — spec `:421` and RTL
> `capstone_flu_unit.anvil:139` both fault on `<=`, so `>` is what silicon does today, and any `>=`
> relaxation is a spec change first.
>
> Gates when it lands: `run-linear-uninit-corpus-probe.sh` (its expectations are written around the
> current `csinit` semantics and must be rewritten with the fix) **and** `run-nullblk-all.sh`, together,
> serialized on the rootfs lock.

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

### R-28 — the revocation-node WRITE ops (DROP/REVOKE/MREV/SPLIT/DELIN) can mutate node state for an instruction that never retires `OPEN — NAMED BY AUDIT 2026-09-09, not demonstrated; directed arms exist (r28-interrupt-probe) but cannot reach it (see the box); the live untested route is in the commit stage; no fix`

> **2026-09-09 (evening), RTL lane after a claim-auditor pass — STAYS OPEN, and the interrupt route is closed
> by the RTL, not by measurement.** (1) An interrupt binds to an instruction at DECODE (`decoder.sv`:
> `instruction_o.valid = instruction_o.ex.valid`); such an instruction never reaches a functional unit
> (`issue_read_operands.sv:1268`) and is latched into the issue slot with `ex.valid` clear
> (`id_stage.sv:346,372-386`), so it cannot acquire one afterwards; the write ops additionally issue only into
> an empty scoreboard. Cite the RTL for this, not any arm. (2) The directed arms on branch
> `r28-interrupt-probe` (monotonic revocation-node ids over a counted run of MREV/SPLIT) are NOT evidence
> either way: node requests equalled retirements in every family because no arm had a measured chance of
> landing an interrupt on an in-flight op — the twelve `r28-mrev-n*` arms FAIL with their own oracle's code
> 15 (the interrupt missed), and three positive-control attempts on a scratch build with MREV removed from
> the hold set all had the flush arrive before the node request went out; the detector is unproven for an
> orphan. (3) A LIVE untested route keeps the entry open: `commit_stage.sv:205-229` runs a PC-capability
> check on the commit head after write-back, and its invalidating input is produced during a write op's own
> execution (`ex_stage.sv:1207` into `commit_stage.sv:243-246`) — a REVOKE or DROP that clears the validity
> of the node its own PC capability derives from would mutate node state in EX and then fail to retire, mepc
> on itself: R-28's shape without an interrupt. The test is a domain whose PC-capability revocation node is a
> descendant of the node the REVOKE walks. Also unresolved: single-step debug (`csr_regfile.sv:2222`) and a
> one-cycle `commit_ack` race against `flush_csr`. Oracle limits, for whoever writes the next arm: for DROP
> and REVOKE no end-state oracle can work at all — invalidation is idempotent, so an op killed after mutating
> the node and re-executed leaves exactly the state one execution leaves; allocation is the only
> non-idempotent mutation, which is why MREV and SPLIT are the two ops testable that way; DELIN is a plain
> gap. Detail: the fix-cycle history note, R-28 section.

Named by the claim-auditor while attacking the R-27 fix (same history note). The write ops are held until they
are the oldest instruction (`issue_read_operands.sv:1525-1533`), but an interrupt or a debug flush can still land
after their request has reached the node, which may by then have mutated state — a minted node, a cleared
validity bit — for an instruction that is then killed and never retires. The R-27 drain discards the orphaned
response; it cannot undo the mutation. Consequence if real: a revocation-tree entry out of step with the
architectural state after an interrupt-timed flush (a node consumed, or a capability's lineage invalidated,
with the instruction re-executed afterwards and doing it again, or not at all). What would settle it — corrected 2026-09-09: NOT an
interrupt-timed arm (an interrupt binds at decode and never reaches a functional unit, see the box) but the
commit-stage route: a domain whose PC-capability revocation node is a descendant of the node a REVOKE/DROP
walks, read against the node pool and the retired-instruction trace. Owner: the RTL
lane. Not to be conflated with R-27: that one is a deadlock, this one is a state divergence.


### R-29 — (repro folder: `tests/fpga-repros/R29-wbuffer-highword-forwarding/`) a plain 8-byte `sd` into the HIGH word of a 16-byte granule, immediately followed by a 128-bit untagged `ldc` of that granule, returns the high half ZEROED (the struct-assignment shape S-06 declared fixed; it never was) `OPEN — DEMONSTRATED 2026-09-09 on silicon (caplifive_r25r26r27_66c4e7517, boots sw46 and sw48, two firmwares) and in RTL simulation on 5097eb166, ef5a8eaf2 and 66c4e7517 (fpga-repros/S06-…/sim/s06agg-shape.S: FAIL 11 with the store adjacent, PASS with four instructions between); revision- and firmware-independent; MECHANISM SEPARATED BY WAVEFORM 2026-09-10: the wide load MISSES and takes the refill leg (`wt_dcache_mem.sv:354-358`), so `rd_user_o` is served from a line that memory does not yet hold — while the write-buffer entry holding the missing half sits RESIDENT and fully valid at that very cycle, because the overlay that would repair it (`:283/:335/:397`) is gated at WORD granularity and a word-1 entry never hits. Refill = origin, word-gated overlay = the missing repair. The store-buffer account is REFUTED by observation. Both the store-buffer account and the `.user = 0` account are REFUTED by traced arms that show their condition existed. SILICON: the window is ONE instruction wide -- boot sw49's distance ladder turns clean at a single intervening nop. Still unobserved: `wbuffer_hit_oh`/`wbuffer_be` themselves (internal combinational, inferred from port behaviour). No fix candidate; synthesis-first; W-12 stays in force`

> **FIX CANDIDATE 2026-09-10: functionally correct, FAILS THE LINT GATE. Branch
> `r29-granule-data-overlay`, commit `00e89d968`.** The granule-scoped data overlay does what it was
> written to do — `s06agg-shape` goes FAIL 11 → PASS with four controls unchanged and the predictions
> written first — but `verif/sim/rtl-lint-gate.sh` reads **UNOPTFLAT 40 → 41**, a new combinational
> loop inside `i_wt_dcache_mem`. The RTL lane predicted it would escape because it feeds `rd_user_o`
> rather than `rd_ctag_o`; it did not. That is exactly the risk `wt_dcache_mem.sv:384` warns about, and
> the third formulation in this family to take a +1.
>
> **The decision this raises is the lead's, and it is a real fork.** Precedent exists: **S-10** took
> the same +1 and was accepted, after synthesis showed its loop set unchanged and its nets clear of the
> critical path. So the options are (a) spend a synthesis to learn whether this loop is benign the same
> way — about 1 h 20 m plus the collector on a quiet machine — or (b) reformulate first and synthesise
> only a lint-clean candidate. Neither is a lane's call: (a) spends a build on a gate failure, and
> CLAUDE.md is explicit that the lint gate must pass and that adding a term to a cone on the standing
> UNOPTFLAT list is the highest-risk edit available.
>
> **What this does NOT do is hold up R-30 and R-31.** Those are the more serious defects — one is a
> data-disclosure gap on the flashed part — and R-29 is a narrow one-instruction window with `W-12`
> already in force. They go on their own branch and their own bitstream if they lint clean; R-29
> follows in its own cycle.

**Signature.** The rung `s06agg` (`tests/fpga-repros/R29-wbuffer-highword-forwarding/src/s06agg.dom`,
`249118220f8cf37a`, entry VA `0x10000`, `-O0`) copies `struct { void *p; unsigned long x, y; }` by
aggregate assignment and returns `64 + r`: bit 0 = `x` (low half of the second granule) lost, bit 1 = `y`
(high half) lost. QEMU returns 64. The board returns **66**: `y` lost, `x` intact. The directed test reads
`y` back as **zero**, not merely wrong.

**The copy site** (the committed draw, `llvm-objdump`):

    10438: sd  a0, 0x18(a2)     src.y   -- the LAST plain store to the granule
    1043c: ldc a3, 0x10(a2)     the whole granule, the very next instruction
    10440: ldc a0, 0x20(gp)
    10444: stc a3, 0x10(a0)
    10448: ldc a2, 0x0(a2)      the pointer granule, after
    1044c: stc a2, 0x0(a0)

`src.x` was stored several instructions earlier (its constant build sits between). The half that is lost is
exactly the half whose store is adjacent to the load; the half with time to leave the write buffer survives.

**Board record (2026-09-09, all on `caplifive_r25r26r27_66c4e7517`, `board-results/2026-09-05.tsv`).**
sw46 (firmware variant D, the three CCSRRW `fence.i` dropped): 66, k800 and six BEEBS rungs at their
oracles first. sw48 (the M-3 monitor, every `fence.i` present): 66, k800 = 4 and `s06copy` = 32 first. One
draw, one bitstream, two firmwares, 66 twice. **No reading of this draw on the previous bitstream exists:**
the row labelled `s06agg` in boot B1 (`32e13a81cf52d4d8`, entry VA `0x20000`, retval 15) is a different
program — 15 is outside this program's 64..67 range — and that row was the S-06 folder's cited acceptance
for the struct-assignment half (corrected in the folder and under the archived S-06 entry, 2026-09-09).

**Simulation (RTL lane, 2026-09-09, `fpga-repros/R29-wbuffer-highword-forwarding/sim/s06agg-shape.S`, delay-40 model, capability mode after
CAPENTER, the same object shape and the rung's instruction order).** Adjacent (`sd y` then `ldc` of the
granule): **FAIL 11, `y` = 0** on `5097eb166` (the silicon that flew before this flash), `ef5a8eaf2` and
`66c4e7517`, 1984 cycles each; with four instructions between the store and the load: PASS on all three,
1828 cycles. The plain `sd`/`ld` control is intact in all six runs. The test's first version had the stores
apart and read PASS ×3; that reading was retracted within the hour — the test contained the shape without
creating the triggering condition.

**What it is not.** Not the S-06 tag path: that fix's directed tests pass and the memcpy-shaped rungs
(`s06copy` 32, `s06aggcap` 15, `s06aggwide` 255) are at their oracles on `5097eb166` (sw05) and on
`66c4e7517` (sw45). Not the R-25/26/27 bitstream: present on `5097eb166` in simulation. Not firmware:
sw46 and sw48 differ only in firmware and read the same. Bare M-mode reproduces it, so no monitor,
compiler, cap table or CPMP state is needed to explain the board reading.

**Consequence.** Any code whose last plain store lands in the high word of a granule right before a
capability-grained copy of that granule — the compiler emits exactly that for a struct assignment whose
trailing scalar field is initialised just before the copy. `W-12` (`SQLITE_GRANULE_GUARD`,
`workarounds/CLASSIFICATION.tsv`) stays in force with this entry as its reason; `W-04`'s retirement
(the memcpy high-half fixup) is unaffected, the memcpy loop does not have the shape.

> **AUDIT 2026-09-09 night (RTL lane's claim-auditor): the attribution below is PLAUSIBLE-BUT-UNPROVEN;
> the header was softened the same night.** What survived: the location (`rd_user_o` is the lane carrying
> the high 64 bits of an `ldc` — `:397` → `wt_dcache.sv:385` → `:257` → `wt_dcache_ctrl.sv:104` →
> `load_unit.sv:761` → `load_store_unit.sv:624`, single beat, no alternate route; line width 128 makes
> bank 1 the granule's high word), the word-granular gating at `:283/:335/:397`, and the S-10 tag comment
> at `:286`. What is false as written: the fall-through is not only the stale array — `:354-358` has a
> refill leg where `ruser` comes from the line returning from memory, and since the source region is never
> read before the `ldc` and this dcache does not write-allocate, the `ldc` very plausibly MISSED and took
> that leg, a different race in a different unit; and the generate loop is `gen_rd_user` at `:396`. What
> is unmeasured: WHERE the store was at the read cycle — allocation with `wtag` = word 1 is confirmed
> (`wt_dcache_wbuffer.sv:728-731`, S-07's stall at `:475-476` needs a capability on one side and both stores
> are plain), residency at the read was never observed. A stronger alternative is alive: load/store
> disambiguation is word-granular too (`load_unit.sv:297` takes the page offset from `vaddr[11:0]`,
> `store_buffer.sv:279/287/293` compare `[11:3]`), so a 16-byte `ldc` at `…010` is NOT held for a pending
> store at `…018` and the `sd` may still be in the STORE BUFFER, never having reached the dcache — a
> different file, a different fix, and a window of a few cycles rather than forty, which fits
> one-instruction adjacency better. A SECOND DEFECT at the same line: a plain store's wbuffer `.user` is
> provably zero (`store_unit.sv:363`), so a resident plain WORD-0 entry sets `wbuffer_be` to all ones and
> drives all eight lanes of `rd_user_o` from `.user = 0` — a word-1 term does not fix that; the fix must
> also refuse the `.user` overlay when the entry is not a capability. Also: the apart-PASS run's log was
> overwritten (that result rests on the commit message and the records file), and no test observes the
> `ldc`'s OWN result (values are read after an `stc` into a second region, so `ldc` vs `stc` vs readback is
> undetermined). The board readings and the six-run table are untouched; only the attribution moves.

> **WAVEFORM PROBE, 2026-09-10 (RTL lane; write-up and dumps in `records/r29/PROBE-READING.md`).
> THIS IS THE VERDICT, and it supersedes both the audit box and the separation arms below.** Sampled
> at the failing read of `s06agg-shape` on `66c4e7517` (times in VCD units):
>
> | cycle | signal | value | what it means |
> |---|---|---|---|
> | 2509 | `wbuffer_q[0].data` | `0x5555666677778888` | the plain store of `y` is IN THE WRITE BUFFER |
> | 2509 | `wbuffer_q[0].valid` | `0xff` | all eight bytes, not cleared before the read |
> | 2509–2687 | store-buffer commit + speculative counts | `0` | the store buffer is EMPTY across the read |
> | 2675 | `wr_cl_vld` | `1` | the wide load **MISSED**; this is the refill leg |
> | 2675 | `rd_data_o` | `0x1111222233334444` | the granule's LOW word, from the refill |
> | 2675 | `rd_user_o` | `0` | the HIGH word lane carries **zero** |
>
> * **Account (b), the store buffer, is REFUTED for this run.** Both counters are zero across the
>   read, so the store had already left it; `load_unit.sv:297` and `store_buffer.sv:279` did not
>   produce this reading. The previous note below said (b) was what stood — that was inference from
>   two passing arms and observation overturns it.
> * **Account (c), the miss-refill leg, is HOW THE STALE DATA ARRIVES.** `wr_cl_vld` is high at the
>   delivering cycle, so `ruser` comes from the line returning from memory (`:354-358`), and memory
>   does not hold `y` yet — the write buffer writes the array only at TX return.
> * **Account (a), the word-gated overlay, is THE MISSING REPAIR, not the origin.** An entry holding
>   `y` is resident and fully valid at that read and `rd_user_o` is nevertheless zero; `:397` is what
>   should have supplied it and did not.
>
> So the correct sentence is the auditor's rather than the first one written here: *the refill brings a
> stale high half, and the write-buffer overlay that would repair it is gated at WORD granularity so a
> word-1 entry is invisible.* The original wording made the overlay the origin; those are different
> claims and only the second survives.
>
> **Why the two separation arms passed** — unexplained until now: each inserts a plain load of the
> store's own word ahead of the wide load, and that load brings the line IN, so the wide load then
> HITS and never takes the refill leg. They removed the very condition that produces the failure.
> **Residency is necessary but not sufficient: the load must also MISS.** That is the third arm in
> this investigation to pass by not creating its condition, and it is the same error each time.
>
> **The `.user = 0` account is REFUTED too (RTL lane, `b871c51d4`, 2026-09-10).** `r29-sep-userzero-miss`
> writes `y` to the high word and drains it with a `fence` so memory holds it, EVICTS the granule's set
> with twelve loads at the same-set stride (32 KiB, 8-way, 16-byte lines: index is `addr[11:4]` over 256
> sets, so twelve tags overflow eight ways), then stores plainly to the LOW word and does the wide load
> immediately. Predicted FAIL 11; read **PASS** at 2485 cycles — and this time the trace shows the
> condition existed: at cycle 4379 `wr_cl_vld = 1` (the load MISSED), `rd_data_o = 0x1111222233334444`
> and **`rd_user_o = 0x5555666677778888`, correct**. The positive control is inside the reading: `x` was
> stored one instruction earlier and never drained, so memory cannot hold it, and the only source for a
> correct low word is the write-buffer overlay — its appearing on `rd_data_o` proves a plain WORD-0
> entry was resident and hit the word-granular compare at that cycle, which is exactly the condition
> under which the `.user` account says the high lane is driven from zero. It was not. So either a plain
> entry's `.user` is not zero as `store_unit.sv:363` was read to imply, or `rd_user_o`'s overlay is not
> gated as that account assumes; the arm does not separate those and does not need to, because the
> failure does not occur. **A fix need not cover it.**
>
> The contrast is the lesson: `r29-sep-userzero` also passed, but there the load HIT and the reading was
> worth nothing. The only difference between the two arms is the eviction.
>
> **Still not observed:** `wbuffer_hit_oh` and `wbuffer_be` are internal combinational signals absent
> from the trace, so the word-gating itself is inferred from port behaviour rather than read directly.
> That needs a Verilator public-signal build — and R-10's `is_cap_req` / `st_wr_cap` are the same
> problem and the same fix, so ONE traced build with those exposed closes both. That is the next probe,
> before any fix candidate.
>
> **For a fix this is good news:** a granule-scoped data overlay would repair the observed case, since
> the entry holding `y` is right there.

> **SEPARATION ARMS, 2026-09-10 — SUPERSEDED BY THE PROBE ABOVE; kept because the reasoning that did
> not survive is the useful part.** (RTL lane, `f29465d8c`, predictions written into the test headers
> before either ran, logs under `records/r29/`.) Both overlay accounts read as DISFAVOURED and the
> store-buffer account as the one left standing. The probe shows why that was wrong: neither arm
> created the miss.
>
> * `r29-sep-forceres` attacks the `:397` overlay: a plain load of the same word the store wrote sits
>   between the store and the wide load, so `store_buffer.sv:279` holds it until the store is
>   forwardable and it brings the line in — which also retires the miss-refill leg. Past it the entry
>   should be resident in the write buffer, where the overlay is claimed blind to it.
>   **Predicted FAIL 11; read PASS, 1843 cycles**, both halves intact, and the separator load itself
>   read `y` correctly.
> * `r29-sep-userzero` attacks the second defect at the same line: `y` written first and drained with
>   a `fence`, then a plain store to the LOW word, then the separator load to force residency, then
>   the wide load. **Predicted FAIL 11 from the `.user = 0` overlay; read PASS, 2229 cycles**, both
>   halves intact.
>
> ~~What is left standing is the **store buffer's word-granular disambiguation**~~ — **REFUTED
> 2026-09-10 by the waveform probe above**: both store-buffer counters are zero across the failing
> read.
>
> **Why this is not yet a refutation.** Nothing in either arm observes where the store physically was
> at the read cycle. Both are equally consistent with the separator load having drained the store all
> the way to the array, in which case neither arm created its condition and neither PASS means
> anything — the same "a clean result from a check not shown to fire" shape that produced two
> retractions on this defect already. The next step is a **waveform probe** on the existing
> `s06agg-shape` run: write-buffer hit, byte enables, refill valid, the high-word lane and the
> store-buffer queues, sampled at the wide load's read cycle. That observes rather than infers. The
> same probe can sample `is_cap_req` (`wt_axi_adapter.sv:196`) and `st_wr_cap`
> (`wt_dcache_mem.sv:138`) and so answer **R-10**'s live secondary half in the same run.


**The account as first written (RTL lane, 2026-09-09, line numbers verified against
`core/cache_subsystem/wt_dcache_mem.sv` at `66c4e7517`; read it with the audit box above).** A cache line is one 16-byte granule (`DcacheLineWidth = 128`): a 128-bit `ldc` returns the
low word on `rd_data_o` and the high word on `rd_user_o` from bank 1 (`:279`). The write-buffer overlay
compares at WORD granularity: `wbuffer_hit_oh[k]` (`:283`) matches an entry whose `wtag` equals the load's
word address, and a granule-aligned `ldc` compares WORD 0; `wbuffer_be` (`:335`) is that entry's byte
enables. Both output words are gated on it — `rd_data_o` at `:394`, and the HIGH word at `:397`:
`rd_user_o = wbuffer_be[k] ? wbuffer_ruser : ruser`. That assumes the only writer of the high word is a
128-bit capability store, whose single entry hits at word 0 and carries the high half in `.user`. A plain
`sd` to word 1 has `wtag` = word 1, never sets `wbuffer_hit_oh`, so `wbuffer_be` is zero and the high
word falls through to `ruser`, the stale array contents (zero in the directed test, hence `y = 0`).
Adjacency is the whole trigger: the store must still be RESIDENT in the buffer when the load looks;
drained, the array is right. The file documents exactly this for the TAG (`:286`: "A granule-aligned LDC
always compares WORD 0, so a resident plain store to the granule's other word is invisible to
wbuffer_hit_oh above") and S-10 fixed the tag side with the granule-scoped `wbuffer_gran_oh` /
`wbuffer_gran_clr` (`:296-315`, into `rd_ctag_o` at `:379`); the data path was not touched. **Fix shape,
not a fix:** extend the granule-scoped term to the data path so a word-1 entry's own data and byte
enables overlay `rd_user_o`. The tag-side term already cost UNOPTFLAT 39 → 40 on `i_wt_dcache.rd_ctag`
(`:384`), and the file says any new term there joins the ring — `rd_user_o` may be a different cone but
that is for synthesis to say. Synthesis-first, one bitstream, the lead's call; not this cycle.

> **RETRACTED AS CONFIRMATION 2026-09-09 night (RTL lane, on the audit).** `r29-lowword`'s oracle is
> inverted: the claimed RTL predicts that arm FAILS (an adjacent word-0 plain store hits, sets `wbuffer_be`,
> overlays `.user = 0`), so its PASS means the entry was not resident and the arm never created the
> condition it was built for. It is also not a one-variable pair: the store and load share `[11:3]`, so
> `page_offset_matches` STALLS the load (+56 cycles on otherwise identical code) and the seeds differ. The
> "r29-lowword must still pass" acceptance line is withdrawn with it. Kept below as the record of what was
> claimed.

~~**The discriminating arm (RTL lane, 2026-09-09, prediction written before the run).**~~ `r29-lowword` is
`s06agg-shape` with one variable changed: the last store before the `ldc` targets the LOW word and the high
word is stored early. The word-0-gating account entails PASS (a resident store to word 0 does set
`wbuffer_hit_oh`, so the overlay applies); a generic "a nearby store disturbs the load" account entails
FAIL. Reading on `66c4e7517`, delay-40: `s06agg-shape` (adjacent store to the HIGH word) FAIL 11, 1984
cycles, `y` zeroed; `r29-lowword` (adjacent store to the LOW word) PASS, 1844 cycles, both intact; plain
control intact in both. The site is named and corroborated; a claim-auditor pass on the mechanism is
pending (result to be recorded here either way). Bare M-mode simulation: corroborates the mechanism,
adds no board evidence.

> **SILICON DISTANCE LADDER, boot sw49, 2026-09-10 — the window is ONE INSTRUCTION WIDE.** Four arms,
> one variable, each image's gap verified in its own disassembly rather than in its source (an arm whose
> filler had been optimised away would return 64 and read as "the defect stopped"):
>
> | arm | gap | reading | |
> |---|---|---|---|
> | `k800` | — | 4 | control, OK |
> | `s06agg` | 0 | **66** | third reading of this draw (sw46, sw48, sw49), first on the fence.i-drop firmware |
> | `s06agg_d1` | 1 | **64** | **the turn point** |
> | `s06agg_d2` | 2 | 64 | |
> | `s06agg_d4` | 4 | 64 | matches the simulation's four-apart arm |
>
> A single intervening instruction clears it. Combined with the probe, the condition is *resident AND
> the load misses*, so this ladder measures the REFILL window rather than the write buffer's: by one
> instruction later the line is already in and the load hits. It also means the exposure in real code is
> narrow — the compiler must emit the plain store immediately before the capability-grained copy, which
> is exactly what an aggregate assignment with a trailing scalar does.

**What would settle it (rewritten after the audit).** (a) Separate the three accounts in simulation with
instruments that OBSERVE, not infer: where the `sd` is at the `ldc`'s read cycle (store buffer, write buffer,
array), whether the `ldc` hit or missed, and the `ldc`'s OWN result register (not a readback after an `stc`) —
one arm per account, prediction first, logs kept. (b) A fix candidate only after (a); whatever it is, it must
also refuse the `.user` overlay for a non-capability entry (the second defect), then the sim pair, lint, the
auditor and synthesis before any board time. (c) ~~The matched board pair~~ — **done, boot sw49 above.** (d) The matched board pair:
the same rung with a `fence` (or any instruction) between the `sd` and the `ldc`, predicted 64. (c) A
fix candidate then goes through the sim pair (adjacent must PASS, apart unchanged), lint, synthesis, and
one bitstream; the rung's 66 → 64 on the board is the acceptance.


### S-14 — restoring `SQLITE_OMIT_EXPLAIN` makes the SQLite domain fault at its FIRST region share, before any of its own code runs `OPEN — DEMONSTRATED 2026-09-09 under QEMU by single-define bisection with an all-deployed control; root cause NOT established and not claimed; the define is dropped from the restore set, so nothing ships broken`

**What happens.** The `restored` SQLite silicon image built with all eight `-U` defines does not run:
it faults at `SQ: E/share1` with **cause 24** (unexpected operand type), *before* the domain enters,
on the ordinary SLT path with no probe involved. The `deployed` image passes the same corpus file in
the same run (`SQ: H/return`, `SLT-SUMMARY` present). Same host, same file, same monitor.

**Why it is one define and which.** Bisected one define at a time, each built and run through that
same path, with an all-deployed control in the same loop: **`-USQLITE_OMIT_EXPLAIN` alone** reaches
only `E/share1` with cause 24; `FOREIGN_KEY`, `UTF16`, `INCRBLOB`, `GET_TABLE`, `DEPRECATED`,
`COMPILEOPTION_DIAGS` and `UNTESTABLE` each reach `H/return`. Confirmed compositionally afterwards:
the seven together pass, all eight fail.

**Why it is filed rather than shrugged off.** Restoring EXPLAIN buys nothing here — we build from the
amalgamation, whose parser tables are pre-generated, so `SQLITE_OMIT_EXPLAIN` never removed the
grammar and EXPLAIN parses and runs today *with* the define set. So the practical answer was to drop
it (the restore set is seven), and the branch does. But **a build define that changes what the
monitor's region share does, before the domain executes a single instruction, is a finding on its
own**: nothing in the SQLite source should be able to reach `share1`. The plausible shapes — an
image-layout or alignment change that moves a capability-bearing object, or a region-count/size
change that alters what is carved — are exactly the shapes of I-03, R-11 and the region-pool limits,
and none of them has been checked here.

**Smallest reproducer today:** build `sqlite_silicon.dom` with `SQLITE_FEATURE_SET=restored` plus only
`-USQLITE_OMIT_EXPLAIN` and run it through the SLT path; the fault is at the first share. Reducing
that to something smaller than SQLite is the first task if this is picked up.

**What would settle it:** which object moved. Diff the two images' `.capstone_gp_initdesc` and region
descriptors (the capability-init slot checker already passes on **both** images, so it is not an
undersized holder), and identify what the monitor is being handed at `share1` that it rejects as an
unexpected operand type.

**Owner:** unassigned. Found by the helper lane during the SQLite stock-ness work, 2026-09-09.

### R-30 — `INIT` is UNREACHABLE on silicon: filling an UNINIT region leaves the cursor at `end`, and `INIT` faults unless the cursor is PAST `end`. The shortfall is exactly one byte, and it kills the whole reason the UNINIT type exists `OPEN — DEMONSTRATED BY READING THE FLASHED RTL 2026-09-10 (66c4e7517); not yet run as a directed test; the defect is INHERITED FROM THE SPEC, which has the same arithmetic`

> **PROVISIONAL, and not a regression — read this before citing it.** This rests on READING
> `66c4e7517`'s source. There is no directed test and no board arm yet, and a claim-auditor pass is
> attacking it. **It is not a property of the currently flashed bitstream in particular:** the same
> arithmetic and the same clause are in the previous bitstreams and, for R-30, in the spec itself, so
> nothing here argues that the 2026-09-09 flash should not have happened or should be reverted. The
> R-25/26/27 fixes that build carries are validated on silicon and stand.
>
> **INDEPENDENTLY CONFIRMED by the RTL lane, 2026-09-10**, line by line against the same revision, and
> they went at the one reading that could have collapsed R-31: whether `\<=p` is a lattice relation a
> bitmask test would get wrong. It is not a problem here — `asm_insn.h` encodes NA 0, XO 1, WO 2, WX 3,
> RO 4, RX 5, RW 6, RWX 7, so bit 1 is write and `2 \<=p perms` is exactly `(perms & 2) == 2`; and
> `existing-insn.adoc:255` uses the same operator as a store's fault condition, which the RTL
> implements as `(perm & 2) != 2` at `dyn:401`. The test is right and only the SENSE is reversed. Still
> derived from source rather than measured — two readers agreeing is not a demonstration.
>
> **FIX ON A BRANCH AND LINT-CLEAN, 2026-09-10.** `r30-r31-init-revoke` off `66c4e7517`, anvil
> regenerated before linting: **UNOPTFLAT exactly 40** with every other count at baseline (LATCH 52,
> MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 717, ANVIL_UNOPTFLAT 0).
> `capstone_flu_unit.anvil` takes `INIT`'s `<=` → `<`; `capstone_dyn_unit.anvil` takes the revoke
> polarity `==` → `!=`; each carries its reasoning, the spec citation and the ships-together constraint
> in the comment.
>
> **Why these lint clean where R-29's candidate did not**, predicted before the run and worth keeping
> as the general rule: both flip ONE OPERATOR inside a condition that is already evaluated — no new
> signal, no new term, no new comparator, so there is nothing to join the ring. R-29's candidate added
> a partner-word hit vector and a leading-zero count, which is why it cost a loop and these do not.
>
> **Predictions, written before the lint ran and before any simulation** (`records/r3031/`):
> `init-rs1-ne-rd` unchanged (it fabricates `cursor > end`, which `<` still accepts); a new
> fill-then-INIT arm goes ILLEGAL_OPERAND_VALUE → PASS; `revocation.S` and `data-sharing.S` MAY change
> and every change must be explained before it is accepted; the 88-row sweep identical except on rows
> that revoke through a write-bearing capability or INIT a filled region; and **the new revoke arm must
> show the CURSOR RESET TO BASE, not merely the type changing** — the type alone does not prove the
> disclosure is closed.
>
> **DEMONSTRATED 2026-09-10 (`a1484c6d3`), with negative controls that prove the arms fire:**
> `r30-fill-init` FAIL 11 → PASS and `r31-revoke-cursor` FAIL 11 → PASS, where on the UNFIXED tree the
> failures carry the reserved codes "INIT trapped" and "type is not UNINIT" — the two defects being
> present. The prints show the conditions directly: an UNINIT capability over `[0x80003000,
> 0x80003040)` reads cursor exactly `0x80003040` after four 16-byte `STC`s, landing ON `end`, which IS
> the shortfall; and after the revoke the capability reads **type UNINIT with the cursor back at base**,
> confirmed independently by `LCC`. That cursor reading is the one a type-only check would have missed.
> *Caveat from the author:* `r30-fill-init`'s post-INIT bounds are an artefact of passing an absolute
> address as `rs2` (INIT computes `rs2.cursor + start`, so it double-counts); it does not affect the
> claim, which is only that INIT stopped trapping.
>
> # ⚠ THE RTL FIXES ALONE WOULD TRAP THE MONITOR. They must ship WITH a firmware change.
>
> This corrects the ordering note above, which said R-31 must not ship before R-30 as though R-30
> rescued it. **It does not.** The monitor has two sites — `sbi_capstone.c:1196-1197` in
> `shared_region_annotated` and `:1340-1341` in `share_child_region` — of the form
> `if (cap_type(r) == 3) { C_INIT(r, r, 0); }`. They are DEAD today for RW regions precisely because of
> the inversion (revoke returns LINEAR, so the test is false). Fix R-31 and they go live. And R-30's fix
> does not save them: INIT then accepts `cursor >= end`, while a revoke-derived UNINIT has its cursor at
> **base** — which the `r31-revoke-cursor` arm shows directly. Base is not end for any non-empty region,
> so `C_INIT` traps on the first revoke of a writable region with a linear borrow.
>
> **And the monitor is wrong on the merits, which is why this is a fix rather than an obstacle.**
> `C_INIT` immediately after revoke is an attempt to skip the refill. Revoke hands back UNINIT-at-base
> exactly so the owner must overwrite the borrower's data before reusing the region; a monitor that
> re-inits straight through defeats the property R-31 restores. So **M-5 stops being latent and becomes
> the third piece of this change**: fill the region and then INIT, or leave it UNINIT until something
> does. The bitstream and the firmware land together, gated on the QEMU suites.
>
> `revoke_region`'s own two sites (`:1455`, `:1460`) are safe — they store the result back without
> inspecting the type. That grep is not proof of completeness and an auditor has been asked to look for
> a third consumer.

**The arithmetic, from the flashed bitstream's own source.** For an UNINIT capability over a region
`[S, E)`:

* the only instruction that advances an UNINIT cursor is `STC`
  (`load_store_unit.sv:994-996` traps every scalar store through a type-3 capability with
  `UNEXPECTED_CAP_TYPE`, so `sd`/`sw`/`sb`/`sh` cannot);
* `STC`'s bound is `rs1_end = end - 16` and it faults when `cursor > rs1_end`
  (`capstone_dyn_unit.anvil:387`, `:409`), and `imm` must be 0 for UNINIT (`:404`), so the cursor
  *is* the address;
* each accepted store advances the cursor by 16 (`capstone_dyn_unit.anvil:431-432`).

So the last permitted store is at `E - 16`, after which **cursor = E**. And `INIT` faults iff
`cursor <= end` (`capstone_flu_unit.anvil:139`). `E <= E` is true, so `INIT` raises
`ILLEGAL_OPERAND_VALUE (29)`. **Maximum reachable cursor `E`; required `E + 1`; shortfall exactly 1.**

**Every other route is closed**, checked one by one at `66c4e7517`: `CINCOFFSET`/`CINCOFFSETIMM`
reject UNINIT (`flu:32`, `:63`); `SCC` rejects it (`flu:97`); `SHRINK` clamps the cursor down, never
up (`flu:204-209`); `SPLIT` rejects it (`dyn:120`); `TIGHTEN` touches only permissions
(`dyn:221-248`); `MOVC` and `LDC`/`STC` round trips preserve type and cursor.

**One legal sequence reaches the precondition and is useless.** `SHRINKTO` accepts UNINIT
(`flu:227`), guards on `(cursor < start) || (cursor + imm > end)` (`:231-232`) and sets
`end := cursor + imm - 1` (`:237`). Any `imm <= 0` passes and yields `end < cursor`, satisfying
`INIT` — but the surviving capability has `end < start`, an inverted zero-length region. A sequence
exists; none preserves the region.

**So the generic "fill an uninitialised region, then INIT it" flow — the reason the type exists —
cannot complete on this silicon for a region of any size.** This is strictly larger than **M-5**,
which is one instance of it.

**The spec has the same defect, so the RTL inherited rather than introduced it.**
`capstone-spec/parts/mem-access-insn.adoc:93` bounds the store at `[base, end - CLENBYTES]` and
`cap-man-insn.adoc:421` faults `INIT` on `cursor <= end`. Neither spec states whether `end` is
inclusive or exclusive (`prog-model.adoc:92` defines it only as "the end memory address"), and the
arithmetic fails under BOTH readings — exclusive is dead by one byte as above; inclusive is worse,
because the largest aligned address `<= end - 16` leaves the cursor at `end - 15` and the region's
last granule is never writable at all.

**Why nothing downstream noticed:** QEMU papers over it three separate ways — `csrevoke` puts the
cursor at `end` rather than `base` (`op_helper.c:920`), `csinit` asserts `cursor == end` instead of
`> end` (`:1200`), and QEMU never advances an UNINIT cursor at all (no such write exists in the
target). See **Q-07**.

**What would settle it, and what it costs.** A directed test is cheap and should come first: fill an
UNINIT region to its bound with `STC` and then `INIT` it, predicting exception 29. The repo already
contains the admission — `verif/tests/custom/capstone/init-rs1-ne-rd.S:29-32` says INIT "raises
ILLEGAL_OPERAND_VALUE (29) unless the uninitialised capability's cursor is past its end", and can only
fire INIT by FABRICATING the operand with the Custom3 debug ops (`MKCAP(a5, CAP_TYPE_UNINIT, 512, 496)`
— `end = base - 16`). That is a test working around the defect rather than reporting it.

**The fix is one comparison, but which one depends on a decision that is not a lane's.** The `end`
convention must be declared first, because the RTL is currently split against itself — exclusive on
every access path (`lsu:1004`, `dyn:387`, SPLIT `dyn:141-145`) and inclusive in `SEAL` (`flu:163`) and
`SHRINKTO` (`flu:237`).
* **`end` exclusive** (what the access paths already assume, and what QEMU is throughout): `INIT`'s
  test becomes fault iff `cursor < end`, i.e. `flu:139` `<=` → `<`. Spec `cap-man-insn.adoc:421`
  changes with it, and `SEAL`/`SHRINKTO`'s ±1 are the stragglers to reconcile.
* **`end` inclusive**: `INIT`'s `>` is right and the STORE bound is the bug — `dyn:387` and
  `mem-access-insn.adoc:93` become `end - CLENBYTES + 1`, which also unblocks the last granule.

Recommendation: **exclusive**, because it is what the RTL's access paths and all of QEMU already do,
making it the smaller and better-tested change.

**AND IT CLOSES Q-07 AS A SIDE EFFECT, which materially changes the cost of the decision** (RTL lane,
2026-09-10). Q-07 currently records that QEMU and the RTL accept **disjoint** `INIT` operands — QEMU
asserts `cursor == end` and aborts the emulator on precisely the operand the RTL demands. Under the
exclusive convention the RTL's corrected test accepts `cursor >= end`, which is what QEMU already
implements, so the two agree for the first time and Q-07's divergence disappears rather than needing
its own fix. A spec decision that resolves a second recorded defect is a different proposition from
one that merely enables this one.

**Three independent arguments for exclusive**, all from the RTL itself: `STC`'s bound is `end - 16`,
which is exclusive arithmetic (inclusive would need `end - 15`); a full buffer leaves the cursor at
`end`, the exclusive convention's natural "one past the last byte"; and QEMU's `helper_csinit` has
asserted `cursor == end` all along, i.e. it implemented exclusive from the start.

**Independent corroboration the RTL lane supplied:** `init-rs1-ne-rd.S` was written before this defect
was named, and it fabricates its UNINIT operand with `CINCOFFSET` past `end` *precisely because filling
cannot reach the precondition* — a workaround, in the test suite, for a defect nobody had named. **This is a spec decision and belongs to the lead and
the spec's owners, not to a lane.** See **R-31**, whose fix must NOT land before this one.

### R-31 — REVOKE's permission clause is INVERTED against the spec, so revoking a linear borrow of a WRITABLE region returns a readable LINEAR capability instead of an UNINIT one — the reinitialisation step is skipped and the borrower's data is disclosed to the owner `OPEN — SECURITY-RELEVANT. VERIFIED BY READING THE FLASHED RTL 2026-09-10 (66c4e7517) against the spec; not yet demonstrated by a directed test`

> **PROVISIONAL, and not a regression — read this before citing it.** This rests on READING
> `66c4e7517`'s source. There is no directed test and no board arm yet, and a claim-auditor pass is
> attacking it. **It is not a property of the currently flashed bitstream in particular:** the same
> arithmetic and the same clause are in the previous bitstreams and, for R-30, in the spec itself, so
> nothing here argues that the 2026-09-09 flash should not have happened or should be reverted. The
> R-25/26/27 fixes that build carries are validated on silicon and stand.
>
> **INDEPENDENTLY CONFIRMED by the RTL lane, 2026-09-10**, line by line against the same revision, and
> they went at the one reading that could have collapsed R-31: whether `\<=p` is a lattice relation a
> bitmask test would get wrong. It is not a problem here — `asm_insn.h` encodes NA 0, XO 1, WO 2, WX 3,
> RO 4, RX 5, RW 6, RWX 7, so bit 1 is write and `2 \<=p perms` is exactly `(perms & 2) == 2`; and
> `existing-insn.adoc:255` uses the same operator as a store's fault condition, which the RTL
> implements as `(perm & 2) != 2` at `dyn:401`. The test is right and only the SENSE is reversed. Still
> derived from source rather than measured — two readers agreeing is not a demonstration.
>
> **FIX ON A BRANCH AND LINT-CLEAN, 2026-09-10.** `r30-r31-init-revoke` off `66c4e7517`, anvil
> regenerated before linting: **UNOPTFLAT exactly 40** with every other count at baseline (LATCH 52,
> MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 717, ANVIL_UNOPTFLAT 0).
> `capstone_flu_unit.anvil` takes `INIT`'s `<=` → `<`; `capstone_dyn_unit.anvil` takes the revoke
> polarity `==` → `!=`; each carries its reasoning, the spec citation and the ships-together constraint
> in the comment.
>
> **Why these lint clean where R-29's candidate did not**, predicted before the run and worth keeping
> as the general rule: both flip ONE OPERATOR inside a condition that is already evaluated — no new
> signal, no new term, no new comparator, so there is nothing to join the ring. R-29's candidate added
> a partner-word hit vector and a leading-zero count, which is why it cost a loop and these do not.
>
> **Predictions, written before the lint ran and before any simulation** (`records/r3031/`):
> `init-rs1-ne-rd` unchanged (it fabricates `cursor > end`, which `<` still accepts); a new
> fill-then-INIT arm goes ILLEGAL_OPERAND_VALUE → PASS; `revocation.S` and `data-sharing.S` MAY change
> and every change must be explained before it is accepted; the 88-row sweep identical except on rows
> that revoke through a write-bearing capability or INIT a filled region; and **the new revoke arm must
> show the CURSOR RESET TO BASE, not merely the type changing** — the type alone does not prove the
> disclosure is closed.
>
> **DEMONSTRATED 2026-09-10 (`a1484c6d3`), with negative controls that prove the arms fire:**
> `r30-fill-init` FAIL 11 → PASS and `r31-revoke-cursor` FAIL 11 → PASS, where on the UNFIXED tree the
> failures carry the reserved codes "INIT trapped" and "type is not UNINIT" — the two defects being
> present. The prints show the conditions directly: an UNINIT capability over `[0x80003000,
> 0x80003040)` reads cursor exactly `0x80003040` after four 16-byte `STC`s, landing ON `end`, which IS
> the shortfall; and after the revoke the capability reads **type UNINIT with the cursor back at base**,
> confirmed independently by `LCC`. That cursor reading is the one a type-only check would have missed.
> *Caveat from the author:* `r30-fill-init`'s post-INIT bounds are an artefact of passing an absolute
> address as `rs2` (INIT computes `rs2.cursor + start`, so it double-counts); it does not affect the
> claim, which is only that INIT stopped trapping.
>
> # ⚠ THE RTL FIXES ALONE WOULD TRAP THE MONITOR. They must ship WITH a firmware change.
>
> This corrects the ordering note above, which said R-31 must not ship before R-30 as though R-30
> rescued it. **It does not.** The monitor has two sites — `sbi_capstone.c:1196-1197` in
> `shared_region_annotated` and `:1340-1341` in `share_child_region` — of the form
> `if (cap_type(r) == 3) { C_INIT(r, r, 0); }`. They are DEAD today for RW regions precisely because of
> the inversion (revoke returns LINEAR, so the test is false). Fix R-31 and they go live. And R-30's fix
> does not save them: INIT then accepts `cursor >= end`, while a revoke-derived UNINIT has its cursor at
> **base** — which the `r31-revoke-cursor` arm shows directly. Base is not end for any non-empty region,
> so `C_INIT` traps on the first revoke of a writable region with a linear borrow.
>
> **And the monitor is wrong on the merits, which is why this is a fix rather than an obstacle.**
> `C_INIT` immediately after revoke is an attempt to skip the refill. Revoke hands back UNINIT-at-base
> exactly so the owner must overwrite the borrower's data before reusing the region; a monitor that
> re-inits straight through defeats the property R-31 restores. So **M-5 stops being latent and becomes
> the third piece of this change**: fill the region and then INIT, or leave it UNINIT until something
> does. The bitstream and the firmware land together, gated on the QEMU suites.
>
> `revoke_region`'s own two sites (`:1455`, `:1460`) are safe — they store the result back without
> inspecting the type. That grep is not proof of completeness and an auditor has been asked to look for
> a third consumer.

**The spec** (`capstone-spec/parts/cap-man-insn.adoc:585-592`) sets `x[rs1].type` to LINEAR if EITHER

* every invalidated capability `c` is non-linear, **or**
* `2 \<=p x[rs1].perms` does **NOT** hold — i.e. the revocation capability does **not** carry write,

and **otherwise** sets it to UNINIT with `cursor = base`.

**The RTL** (`capstone_dyn_unit.anvil:62`) is:

```
if(rsp == 1'd1 || ((rs1.metadata.perm&3'd2)==3'd2)){   // -> LINEAR
```

`rsp == 1` is the first clause, correctly implemented (`capstone_rev_node.anvil:156`, `:24-26`, `:19`:
the flag starts at 1 and is cleared when an invalidated node was linear). The second disjunct is
**inverted**: `perm & 2` is the WRITE bit — confirmed from the access checks in the same file, where
`STC` faults `INSUFFICIENT_PERMISSION` on `(perm & 2) != 2` (`dyn:401`) and `LDC` on `(perm & 4) != 4`
(`dyn:338`). So the RTL returns LINEAR when write **is** held; the spec says LINEAR when write is
**not** held.

**Consequence on silicon, and it is the security-relevant direction.** For a revocation capability
that carries write — which is *every* shared read-write region the monitor hands out — revoking a
capability whose borrow was linear returns **LINEAR with the cursor untouched**, instead of UNINIT at
base. The UNINIT step exists precisely to force the owner to overwrite the region before it can read
it again; skipping it hands the owner a directly readable capability over whatever the borrower left
there. That is an information-disclosure gap relative to the spec's intent, not merely a type
mismatch.

Conversely a revocation capability WITHOUT write yields UNINIT at base — a handle that can still be
`STC`-filled (the UNINIT path skips the write-permission check, `dyn:401` tests only LINEAR/NONLIN)
but can never be `INIT`-ed, per **R-30**.

**This corrects M-5.** M-5 records that the monitor's re-share path `C_INIT`s a revoke-derived UNINIT
and predicts a trap. On this RTL, for the RW regions that path actually handles, `cap_type(r) == 3`
is **false** — REVOKE returned LINEAR — so `C_INIT` is never reached and the path silently *appears*
to work. M-5's latency has a different cause than recorded, and the observable is a disclosure rather
than a trap.

**Ordering, and it is a hard constraint.** Fixing this inversion makes RW revokes return UNINIT, which
immediately runs into R-30's dead `INIT` and turns a silent disclosure into a live monitor trap. **R-31
must not ship before R-30 is decided and fixed.** Ship them together or not at all.

**What would settle it:** a directed test that revokes a linear borrow of a writable region and reads
back the returned type and cursor, predicting UNINIT-at-base under the spec and observing LINEAR-with-
cursor-preserved today. Neither this nor R-30 has been demonstrated by execution — both rest on
reading the flashed source, which is why they are filed as OPEN with that stated rather than as
measured defects.

### R-3 — Second domain at the same entry VA hangs within one boot `WORKED AROUND, ROOT DEFECT LIVE AND NOW UNTESTABLE (2026-09-10): the monitor still lacks the icache invalidate on domain switch, and preflight C15 refuses the same-VA staging that would exercise it, so no boot since it landed has been able to measure this issue either way`
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
- **2026-09-10 — the workaround is now machine-enforced, which also makes the defect unmeasurable.**
  `tests/preflight-board-run.sh:434` BLOCKS a boot that stages two `.dom` files at the same
  `DOMAIN_BASE_VA` (*"C15: ENTRY-VA COLLISION ... R-3 will hang whichever runs second and it will
  read as that rung's verdict"*). That is the right default and it is why no sweep has hit R-3
  since — but it means **"not seen lately" carries no information about R-3**, and the only
  post-reflash same-VA data point (2026-09-05, `rawhazard5` accidentally staged at the control's VA,
  did not stall) is explicitly recorded as NOT a verdict. Measuring R-3 on current silicon requires
  disabling C15 deliberately for one arm, as the last domain of a boot. The R-26 CCSRRW flush in
  `66c4e7517` does not touch it: that entry states *"its icache flush is irrelevant here"*.
- **Mechanism note:** the domain-boundary `fence.i` was long suspected to fix R-1 as well;
  board test #63 disproved that. It remains the right fix for **this** issue only.

### R-4 — A shared-region word is silently corrupted `RECORD ONLY 2026-09-10 — one uninstrumented sighting from 2026-07-28, three bitstreams ago, no reproducer and no image; its SYMPTOM CLASS is now covered by three characterised entries and a new sighting belongs to whichever it matches. NOT closed as 'not reproducible': nobody ever tried`
`rv8_primes` returned the *correct* result while a word of its shared region held a stray DRAM
address. Passing rungs were only ever clean where someone looked.
- **Evidence:** `ref/fpga-silicon-measurements-for-paper.md` §5

> **DISPOSITION 2026-09-10 — RECORD ONLY. The reasoning matters more than the verdict.**
>
> Three lines, no reproducer, no named image, no bitstream, and the one observation predates the
> 2026-08-04 reflash — so it is on `working-caplifive-captype-fixed.bit`, three bitstreams behind
> current silicon. The cited evidence is itself a bullet inside a *"what is NOT established"* list.
>
> **Closure as "not reproducible" was proposed and is REJECTED as overstating the record.** Nobody
> ever tried to reproduce it: the 2026-09 sweep marked it UNTESTABLE precisely because there was
> nothing to run. *"We looked and it did not recur"* and *"there was never anything to look at"* are
> different claims, and only the second is true here. Closing on absence of evidence is the failure
> this project has repeatedly paid for.
>
> **What HAS changed is that the symptom class is no longer unattributed.** "A region word holds
> something the program never wrote" now has three characterised owners, each with a reproducer:
> **R-19** (the victim holds `compress_cap(NULL) + n`, a hardware encoding the program cannot
> materialise), **R-10**, and **R-29**. **No attribution is made here.** R-4 has no artefact to match
> against any of them, and a fit is not a mechanism — this registry has paid for that confusion before.
>
> **Corrected 2026-09-10 after an audit, because the first version of this paragraph overreached in
> three specific ways and the corrections are more useful than the claim:**
> * It restated R-4's *"held a stray DRAM address"* as *"holds something the program never wrote"*,
>   which is strictly broader — and **that widening is the only thing that makes R-19 fit**. R-19's
>   victim holds `compress_cap(NULL) + n`, a 64-bit encoding constant, which is not a DRAM address.
>   R-19 is the WEAKEST of the three and fits only the widened sentence.
> * It cited the wrong sentence of **R-10**. The strong fit is not the OR-reduce inference but R-10's
>   own: *"when the high half is ZERO, `is_cap_req = 0` sets `axi_wr_blen = 0`, so only ONE beat is
>   written and **the high 8 bytes are left at whatever was in DRAM**."* That is a memory-CONTENT
>   mechanism, which is what "held" describes. It is the strongest of the three.
> * It offered no discriminator. Here is one: **R-10's half-beat write leaves stale DRAM IN MEMORY,
>   while R-29 is READ-SIDE** (`rd_user_o` served from the refill leg, memory intact). If the
>   2026-07-28 sighting came from a host-side dump of the region, R-29 cannot be the mechanism. The
>   three lines do not say how it was observed, so **R-29's membership is UNRESOLVED**, not
>   established.
>
> The routing rule below survives all of that and is the actionable part.
>
> **The actionable part.** Keeping this open as a separate ID invites a fourth parallel investigation
> of a symptom three entries already own. A NEW sighting of this shape is filed against whichever of
> R-19, R-10 or R-29 its signature matches, or gets its own ID *with an artefact* — not reopened here.
> It stays in the open registry rather than the archive because nothing about it was resolved; it is
> retained as provenance for the sighting.

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

> **2026-09-10, SECOND READING — PARTIALLY FIXED, and the split is the finding. Read this box, not the
> one below it, for the current state.** Checked against the FLASHED tip (`capstone-ariane`
> `66c4e7517`) rather than against this entry's cited lines, which are stale:
>
> * **Both sites this entry names are REPAIRED, by explicit intent.**
>   `wt_axi_adapter.sv:209` is now `assign is_cap_req = dcache_data.is_cap;` and
>   `wt_dcache_mem.sv:155` is `assign st_wr_cap = wr_is_cap_i;`. Each carries an S-06-fix comment
>   saying the flag is carried explicitly from the store opcode *"instead of inferred from |user|"*,
>   naming the old inference as defect D3/D4/D7. (The entry cites `:196`, which is now a comment.)
> * **But the PATTERN survives, twice, on the REFILL path — and nobody had looked there:**
>   `wt_dcache_mem.sv:358` `rd_ctag = |wr_cl_user_i[7:0];` and
>   `wt_dcache_mem.sv:501` `cap_tag_q[wr_cl_idx_i][j] <= wr_vld_bits_i[j] & (|wr_cl_user_i[7:0]);`
>   Both decide "this granule holds a capability" by OR-reducing the incoming line's user byte, which
>   is exactly the shape this entry describes. **The store side was repaired; the refill side was not.**
>
> So the secondary half is **PARTIALLY FIXED**: closed at the two cited sites, live at `:358` and
> `:501`. Note how it was found — by reading the tip, not by sampling. The waveform probe originally
> planned for this would have sampled `is_cap_req` and `st_wr_cap`, seen them behaving correctly, and
> closed the entry WRONGLY.
>
> **One co-location, recorded as geography and not as causation:** `:358` sits in the SAME
> `always_comb` as the refill assignment the R-29 probe caught — that block sets `rdata`, `ruser` and
> `rd_ctag` together on the `wr_cl_vld` leg. R-29 is the DATA half of that leg being stale; these two
> are the TAG half being inferred by OR-reduce on the same leg. Whether they interact is UNTESTED and
> is not asserted here. A fix touching that block should be written knowing all three live in it.

> **2026-09-10 — first reading, superseded by the box above: THE SECONDARY DEFECT BELOW IS LIVE, AND
> IT IS R-29's SHAPE. It was proposed for demotion to "a one-line RTL observation" and that proposal is
> withdrawn.** The two lines this entry
> named on 2026-07-29 decide "this store holds a capability" by OR-reducing the metadata word and
> never consult `cap_type` — the same word-versus-granule confusion that **R-29** is about, and R-29
> is demonstrated on the newest bitstream (`66c4e7517`, boots sw46 and sw48, `s06agg` = 66 twice).
> Neither line has been retracted or re-measured since. Treat R-10's secondary half and R-29 as
> **sibling accounts of one memory-subsystem question**, and let R-29's separation arms answer
> whether the OR-reduce is in the path; if it is, this half closes with R-29's fix rather than on its
> own. R-10's PRIMARY half (the `__linear` declarator bug in capstone-c) is fixed and shipped — it is
> only the RTL half below that is live.
>
> Also unchanged and still owed: the **stage-8 discriminator against the fixed firmware** proposed in
> this entry has never been run, on any bitstream; and this entry's own instruction *"Do not record
> C-13 as fixed"* was overtaken when C-13 was archived on C-14's mechanism, not on this one.

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

### C-4 — constant pools unreachable in a domain (C-4a) and the large-RO copy path in the generated glue (C-4b) `FIXED 2026-07-28, BOTH halves re-verified 2026-09-05 under QEMU (C-4a rv8_sha512 = 1390718314 = oracle; C-4b beebs_crc32big = 1703161001 = oracle). NOT MOVED TO THE ARCHIVE — see the note directly below`

> **APPLIED 2026-09-10.** Verified against this entry's own sub-entries: there is no residual. C-4a and
> C-4b are both FIXED and both re-verified, and the "remaining domain-creation bug" the old heading
> promised is the `helper_cssplit` path that C-4b's rung now returns its oracle on — though note that
> identifying that referent is INFERENCE from C-4b's provenance text, not sourced to the authoring
> commit, which predates a directory rename and could not be read directly. The compiler lane's
> caveat (recorded further down) stands and is not dropped: this rests on RECORDED sweep evidence and
> neither rung was re-run for the decision.
>
> **Why this entry is NOT in `ISSUES-ARCHIVE.md` despite a final status.** Moving it is impossible
> today without weakening or bypassing `precommit-scan`. Text further down this entry contains a phrase
> the scan's credential pattern matches (`precommit-scan.sh:163`) — a false positive; it is the
> registry's own word for a status word. The scan reads REMOVED and CONTEXT diff lines as well as added
> ones, so the commit that deletes this entry from here is blocked by the text it deletes, the commit
> that adds it to the archive is blocked too, and even editing NEAR the phrase blocks. Recorded rather
> than worked around. The fix is a scope correction to the scan — removals and context should WARN, not
> BLOCK, since content leaving the tree cannot introduce a secret — and that is a change to a release
> gate, so it is the lead's call. Until then this entry is final-but-resident, which is a cost of the
> gate and not a fact about C-4.
>
> **Correction to commit `3c013e06d99f`'s message.** It says C-4 was "archived … moved to the archive".
> It was not, for the reason above; the file has said `NOT MOVED TO THE ARCHIVE` throughout. The message
> misdescribes its own content, which is the failure the `-o/--only` rule exists to prevent, in a
> different form. Recorded here rather than by rewriting a pushed message.


> **RECOMMENDATION 2026-09-10 (compiler lane), for the lead — not applied.** Both sub-entries read
> `FIXED 2026-07-28` and both carry a "Sweep 2026-09-05 re-verified FIXED" line (C-4a via
> `rv8_sha512` = oracle, C-4b via `beebs_crc32big` = 1703161001 = oracle). The header's "a remaining
> domain-creation bug" looks stale: the residual it names is a QEMU `helper_cssplit` assertion, and
> `beebs_crc32big` is the rung built specifically for that path and now returns its oracle under QEMU.
> Proposed token: `FIXED 2026-07-28, re-verified 2026-09-05`, retitled to drop the "remaining" clause.
> **Caveat stated by the proposer:** this reads the sweep's recorded evidence; the rung was not re-run
> for this recommendation.
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

### R-12 — rev-node exhaustion DEADLOCKS the core (a deliberate stall), and the pool is 65536 nodes, not 1024 `CHARACTERISED 2026-09-10 — the wraparound/silent-corruption account below is WITHDRAWN; the threshold is ~65532 allocations, not 1025, and the failure is a visible hang, not silent id reuse. The `99.3 % consumed` board reading is WITHDRAWN 2026-09-10 (it appears only after a wedge; every healthy boot reads the sentinel, which cannot be the true head or no domain would run). No workload is known to approach the threshold; measuring one needs a monitor-side split counter, not the debug aperture`

> **CORRECTED 2026-09-10 (board lane), read against the FLASHED bitstream's own source
> (`capstone-ariane` `66c4e7517`), not against this entry's cited lines.** Both halves of the
> original account are false on this RTL, and they fail in opposite directions -- the threshold is
> 64x larger and the failure mode is *more* visible, not less:
>
> * **The head is 16 bits, and the pool is 65536 nodes.** `core/anvil_build/capstone_rev_node.anvil:74`
>   gates allocation on `*head != 16'd65535` with the comment `REVNODE_SENTINEL = (2^REVNODE_HEAD_BITS)-1`;
>   `:78-79` build the id as `#{14'd0,*head}` and bump by `16'd1`. `core/include/ariane_pkg.sv:587`
>   sizes the pool: *"Revocation-node pool: 65536 nodes * 16 bytes/node"*, and `CAP_REVNODE_MEM_BASE`
>   `0xBFF00000 + 65536*16 = 0xC0000000` abuts the tag region exactly. The 10-bit `head` this entry
>   cited is not at `capstone_rev_node.anvil:168` or anywhere else. The same correction was already
>   made on 2026-08-27 in `benchmarks/sqlite/build-sqlite-silicon.sh:943-949` against the resident
>   bitstream; it never reached this entry.
> * **Exhaustion does NOT wrap, and it is not silent.** `capstone_rev_node.anvil:99-107` is explicit:
>   *"rev-node pool exhausted ... deliberately drop this request instead of aliasing a new node onto
>   an id that wraps back into 0/1/2 or a live node. We never send `ep.init_res`, so SPLIT's
>   `recv rev_node_ep.init_res` blocks forever -- a forced, visible stall instead of silent id reuse."*
>   So the title's "SILENT CORRUPTION, not a fault" is backwards: the designed behaviour is a
>   **deadlock**, which on this RTL presents as a wedge (no trap, no return -- see M-1).
> * **Consequently the SQLite scare is void.** SQLite's entry glue does ~1,060 splits, which is
>   1.6 % of the 65532 usable ids, not a crossing of 1,024. Nothing in the ladder approaches the real
>   threshold either.
>
> **THE MEASUREMENT IS IN, 2026-09-10, and the answer is that the instrument cannot make it.** Every
> rev-node head sample the driver took on a HEALTHY boot — sw44, sw46 and sw48, all arms returning —
> reads `0xFFFF`, which the driver correctly refuses as a datum
> (`run_sqlite_stages_fpga.py:2112-2130`: it is `REVNODE_SENTINEL` and is indistinguishable from an
> all-ones dead aperture). The `65047` reading appears in exactly the two boots that carried a WEDGE,
> sw45 and sw47, and is identical in both. So there is no healthy-boot consumption datum at all.
>
> **And `0xFFFF` cannot be the true head, which is what settles it.** Per the RTL,
> `head == 16'd65535` IS the exhausted state: `capstone_rev_node.anvil:74` gates allocation on
> `*head != 16'd65535`, and `:99-107` then drops the request so SPLIT blocks forever. If the head
> really were at the sentinel, `create_domain`'s five splits would deadlock and no domain would ever
> run — yet every one of those boots created domains and returned their oracles. Therefore `0xFFFF`
> here is the dead aperture, not the register, and the whole read path is uninformative about pool
> consumption on a healthy boot.
>
> **Conclusion: `99.3 % consumed` is WITHDRAWN as a reading.** It was never a live allocation count;
> it is what the halted read path yields after a wedge, twice identically. Answering the actual
> question — how close does any workload come to 65532 — needs a different instrument, and the
> cheapest is monitor-side: `create_domain` and the entry glue already know how many splits they
> perform, so a counter reported through the existing trace tags would give an exact number with no
> aperture involved. Not scheduled; nothing currently approaches the threshold on any known workload.

*The original open question, kept for the record:* the board driver reported
`rev-node head = 65047 (99.3% of 65535)` after the wedged probe in BOTH boot sw45 and boot sw47 --
> the *identical* value in two boots that ran different stage sets. Either the pool really is 485
> allocations from a deadlock on a freshly power-cycled board (which would matter a great deal, since
> there is no slot reclamation -- `DROP` invalidates without freeing), or the read is not a datum: it
> is taken after a wedge, through a path that can serve cached values, and the driver already
> documents one way this register lies. **That check has now been made — see the box above — and the
> healthy-boot reading is the sentinel, so the answer is that this instrument cannot measure pool
> consumption at all.**

*The original account, kept because it is what the entry claimed for six weeks:*

~~The revocation-node allocator's `head` is 10 bits (`capstone-ariane/core/anvil_build/capstone_rev_node.anvil:168`), so allocation
**#1025 wraps to node id 0 and reuses live ids**.~~ `overflow_flag` reaches only a debug LED
(`cva6.sv:1185`) -- nothing traps, nothing prints. Only `SPLIT` and `MREV` allocate
(`capstone_dyn_unit.anvil:136, :91`); `ldc`/`stc`/`cincoffset` allocate nothing
(`:330-332, :399`, `capstone_flu_unit.anvil:29-44`).

~~`create_domain` does **5** splits, so this is NOT the current SQLite blocker. But SQLite's
entry glue does **1,060** splits (1 table + 1,059 globals) and will be the first domain to
cross 1,024 -- at `call_dom`, i.e. the moment after the present wedge is cleared. No
ladder rung approaches it (bigmany: 65).~~

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

### C-14 — the COMPILER uses `movc` (a MOVE) for scalar register copies `FIXED — `copyPhysReg` branches on register class (GPCR → MOVC, GPR → ADDI, `CapstoneInstrInfo.cpp:548-563`), so scalar copies no longer go through MOVC; GONE ON SILICON 2026-09-05. TWO RESIDUALS REHOMED, see the box: the untagged-in-GPCR live copy is C-32, the MOVC modelling is C-46`

> # ⚠ CORRECTED 2026-09-10 — twice in one day. Read this box, not the two below it.
>
> **The "attribution pending Q-04" status was wrong on both halves, and an auditor caught it.**
>
> 1. **The Q-04 dependency was manufactured.** This entry's own v3 box already says: *"**What is NOT
>    in doubt, through all three versions:** the mechanism, the numeric proof, and that LLVM is
>    emitting the wrong instruction. Only blame moved."* The compiler attribution was never pending on
>    a spec ruling. I invented a dependency the entry explicitly denies.
> 2. **The defect as described is already FIXED.** `CapstoneInstrInfo.cpp:548-563` branches on
>    register class — `GPCRRegClass` → `MOVC`, `GPRRegClass` → `ADDI` — so an integer copy is an
>    ordinary ALU move. The body's citation of *"`copyPhysReg` emits MOVC for every GPR-to-GPR copy
>    (`CapstoneInstrInfo.cpp:520-523`)"* is stale in both the line numbers and the claim. The rungs
>    pass on silicon **because the fix landed**, not because they happen not to re-read the source —
>    the ruling attributed to luck what this same entry attributes to a fix.
>
> **Two residuals are real and are rehomed rather than closed with the entry:**
> * **The untagged-capability-in-GPCR live copy → C-32.** `movc` with a source read afterwards is
>   still emitted for GPCR operands; the auditor reproduced it directly
>   (`movc a0, s0 ; cjalr ; movc a0, s0`) from `test/CodeGen/Capstone/c32-movc-untagged-live.ll`, which
>   is committed `XFAIL` with a positive control. Q-04's own tail already says *"the compiler side is
>   C-32"*. **That** is what a Q-04 ruling would gate, not this entry.
> * **The MOVC instruction MODELLING → C-46 (new).** `CapstoneInstrInfo.td:2479-2483` declares
>   `hasSideEffects = 0` with `$rs1` as a pure USE and no `Constraints` tying `rd` to `rs1`, unlike
>   `PseudoINIT`/`PseudoSEAL` a few definitions below. That is wrong for LINEAR capabilities on ANY
>   implementation, independent of the scalar question. It was named inside this entry and would have
>   been ORPHANED by closing C-14 — a grep for it returns exactly one hit, here.
>
> **Caveat carried from the auditor rather than hidden:** the `llc` used to reproduce the C-32 shape is
> older than two of the lowering sources, though `CapstoneInstrInfo.cpp` and `.td` are older still than
> the binary, so the emission path exercised is current. A `ninja llc` and a re-run would collapse it.

> **APPLIED 2026-09-10.** The symptom and the attribution are deliberately kept apart. "Gone on
> silicon" is not "fixed": closing this on the symptom would discard the live question of whether our
> codegen relies on behaviour the spec does not guarantee. Q-04 is the blocker and is named in the
> header so nobody re-derives it. If the ruling goes scalar-exempt this becomes a small compiler fix
> (stop using `movc` for scalar copies) and the compiler lane takes it.
>
> **THE RULING CAME THE SAME DAY, and it went the other way: scalars are NOT exempt** (see Q-04). The
> spec text has no scalar carve-out and the RTL implements it literally, so `movc` as a scalar register
> copy destroys its source on conformant hardware. C-14 is therefore a COMPILER FIX and its attribution
> is settled. The header's "pending the Q-04 ruling" is superseded by this box; the work is to stop
> emitting `movc` for scalar copies whose source is live afterwards. That the current rungs pass on
> silicon means only that they do not read the source again — not that the codegen is right.

> **RECOMMENDATION 2026-09-10 (compiler lane), for the lead — not applied.** The entry carries a
> "Sweep 2026-09-05 — GONE on silicon" line (`gpn2` = 3976364985 = oracle and RETURNED, in the boot
> where it used to wedge). But its own attribution box records that the blame was revised twice, and
> the spec is under-specified on whether scalars are exempt from the MOVC consumption rule — which is
> the open **Q-04** question, not a compiler defect anyone can close. Proposed token:
> `GONE ON SILICON 2026-09-05 — attribution pending the Q-04 spec ruling`, i.e. staying open on the
> ruling rather than on the symptom. If the ruling goes scalar-exempt it becomes a small compiler fix
> (stop using `movc` for scalar copies) and the compiler lane takes it.

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

> **2026-09-10 — THIS ENTRY'S PREMISE IS WRONG ON THIS SILICON, and the two findings that replace it
> are R-30 and R-31. Read those first.**
>
> M-5 says the monitor `C_INIT`s a revoke-derived UNINIT that `INIT` cannot accept. Two corrections:
>
> 1. **For the RW regions this path actually handles, REVOKE does not return UNINIT at all.** The RTL's
>    permission clause is inverted against the spec (**R-31**, `capstone_dyn_unit.anvil:62`), so a
>    revocation capability carrying write yields **LINEAR with the cursor untouched**. `cap_type(r) == 3`
>    at `sbi_capstone.c:1197` is therefore FALSE, `C_INIT` is never reached, and the path silently
>    appears to work — by skipping the reinitialisation the spec requires, which is a disclosure rather
>    than the trap this entry predicts.
> 2. **`INIT` is unreachable for ANY UNINIT capability, not just revoke-derived ones** (**R-30**): the
>    cursor tops out at `end` and `INIT` demands `> end`, a shortfall of exactly one byte. So M-5 is one
>    instance of a generally dead instruction, not a defect of the re-share path.
>
> **The entry stays open** because the monitor code is still wrong in both worlds — it depends on an
> `INIT` that cannot succeed — but it is no longer the thing to fix first, and it is not fixable in the
> monitor alone. The two sites remain `sbi_capstone.c:1196-1197` and `:1340-1341`, and any monitor
> change must follow the R-30 `end`-convention decision rather than precede it.

> **2026-09-10, SECOND PASS — THE QUESTION IS ANSWERED, AND NONE OF THE THREE OPTIONS ON THE TABLE
> WAS THE RIGHT ONE.** The choice was posed as: amend the spec's `INIT` precondition, add a
> monitor-side reclaim that avoids `INIT`, or change what `revoke` leaves behind in the RTL. It rested
> on a false premise. See **R-30** and **R-31**: `INIT` is unreachable on this silicon for ANY UNINIT
> capability (the cursor tops out at `end`, `INIT` demands `> end`), and REVOKE's permission clause is
> inverted, so for the RW regions this path handles REVOKE returns LINEAR and the monitor's `C_INIT` is
> never even reached.
>
> **The resolution, in order, and only the first step is a decision:**
> 1. **Declare whether `end` is inclusive or exclusive.** The spec never says and the RTL is split
>    against itself. Recommended EXCLUSIVE — every access path and all of QEMU already assume it.
>    **The lead's, with the spec's owners.**
> 2. **Fix `INIT`** to match (R-30): under the exclusive reading, `capstone_flu_unit.anvil:139`
>    `<=` → `<`, with `cap-man-insn.adoc:421` changed alongside it.
> 3. **Fix REVOKE's permission clause** (R-31) — but NOT before step 2, or a silent disclosure becomes
>    a live monitor trap.
> 4. **Then QEMU aligns to the fixed RTL**, which is what Q-07 becomes: `helper_csrevoke` leaves the
>    UNINIT cursor at BASE like the RTL rather than at `end`, and `helper_csinit` raises exception 29
>    instead of asserting, using the corrected comparison. The three-way disagreement collapses to one
>    rule.
> 5. **M-5's monitor code is then reachable and can be judged on its merits.** It is not fixable before
>    the above and is not the thing to fix first.

> **2026-09-10, THIRD PASS — M-5 IS THE FIRMWARE HALF OF THE R-30/R-31 CHANGE, and it is the sequence
> the RTL previously made IMPOSSIBLE, not a workaround for a new behaviour.**
>
> Once R-31 lands both sites go live and `C_INIT(r, r, 0)` on a cursor-at-base capability traps (R-30's
> fix accepts `cursor >= end`, not base). The monitor must change in the same cycle.
>
> **Why the shortcut exists at all**, and it reframes the entry: before R-30, "fill the region then
> INIT" was *impossible* — INIT was unreachable by filling for a region of any size. Re-initialising
> straight through was the only thing that could work. The two RTL fixes together make the correct
> sequence expressible for the first time.
>
> **A numbering trap that leads to the OPPOSITE conclusion, recorded so the next reader does not fall
> in.** The sites test `cap_type(r) == 3` while the RTL's UNINIT is **4**, which looks like the test can
> never fire and there is no blocker. It fires: `cap_type(cap)` is `__capfield((cap), 1)`
> (`sbi_capstone.c:195`), the LCC type query, and LCC returns `cap_type - 1`, so RTL UNINIT reads back
> as 3. Checked independently by two readers because the 3-versus-4 invites the wrong answer.
>
> **Options and costs:** (1) **zero-fill then INIT**, the spec's intended reclaim — walk the region with
> `STC` (16 B per store, `imm` 0, so the cursor is the address) to `end`, then `INIT`. This IS the
> security property. **Cost: one store per 16 bytes**, i.e. 65,536 stores per revoke for SQLite's 1 MiB
> region — a real number that belongs in front of the lead. (2) leave it UNINIT and refuse the re-share
> — cheap, but turns a working path into an error return and `__mrev` needs LIN. (3) fill lazily in the
> domain — matches what the type is for, largest change, alters the shared-region ABI.
> **Recommended (1)**, with the per-revoke cost measured on the board before acceptance and (3) as the
> design-level answer if it proves unacceptable.
>
> **Deliberately NOT written yet, for a gate reason rather than caution:** QEMU never advances an UNINIT
> cursor at all (Q-07), so a fill loop is a no-op under the emulator and the suites would pass a monitor
> that fills nothing. Q-07's QEMU fix must land first or the gate is worthless.
>
> The original coupling note stands below, because it is what made the search necessary.

> **2026-09-10 — M-5 AND Q-07 ARE ONE SYSTEM, and neither can be fixed alone. Verified in the QEMU
> source, not inferred.**
>
> `helper_csinit` (`capstone-qemu/target/riscv/op_helper.c:1198-1200`) is three host `assert()`s, so a
> wrong operand `SIGABRT`s the emulator instead of raising a guest trap. All three map one-to-one onto
> spec exceptions (`capstone-spec` `parts/cap-man-insn.adoc:415-421`: 24 unexpected operand type, 26
> unexpected capability type, 29 illegal operand value), and the idiom to replace them with sits five
> lines away in the same file at `:728`. That part is mechanical.
>
> **What is not mechanical:** `helper_csrevoke` at `:920-921` places the retained handle's cursor at
> `end` *specifically to satisfy this assertion*, and says so —
> *"UNINIT … cursor at END, the canonical UNINIT form that csinit requires (csinit asserts
> cursor==end). Leaving it at base produced a handle no instruction could advance — scc rejects UNINIT
> and csinit rejects cursor!=end — so a linear borrow could never be re-lent after revoke."* It labels
> itself an *"Experimental revocation-semantics choice"*. The RTL instead leaves `cursor = start`,
> which is why **M-5** is silicon-dead: no legal instruction sequence takes UNINIT(cursor=start) to LIN
> on hardware except the `CAPTYPE` debug op, which production code cannot use.
>
> **So the order is forced and the two must land together.** Fix Q-07 alone and QEMU's own
> revoke → init → mrev flow breaks. Fix M-5 alone and there is nothing on QEMU that reproduces it.
> Aligning `csrevoke` to the RTL's `cursor = start` in the same commit converts M-5 from
> silicon-latent-and-unmeasurable into QEMU-reproducible with an existing gate — and
> `run-nullblk-all.sh` then goes red and STAYS red until M-5 is resolved, which is correct rather than
> an obstacle, but means **neither may be pushed to `dev` before M-5's resolution is chosen**.
>
> **The decision, which is the lead's with the RTL lane** (M-5 has two sites, not the one the entry
> names: `sbi_capstone.c:1196-1197` and `:1340-1341` in `share_child_region`): amend the spec's `INIT`
> precondition; add a monitor-side reclaim that does not route through `INIT`; or change what `revoke`
> leaves behind in the RTL. Ship `>` rather than `>=` whichever way it goes — spec `:421` and RTL
> `capstone_flu_unit.anvil:139` both fault on `<=`, so `>` is what silicon does today, and any `>=`
> relaxation is a spec change first.
>
> Gates when it lands: `run-linear-uninit-corpus-probe.sh` (its expectations are written around the
> current `csinit` semantics and must be rewritten with the fix) **and** `run-nullblk-all.sh`, together,
> serialized on the rootfs lock.

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

### C-46 — `MOVC` is modelled as side-effect-free with `$rs1` a pure USE, so the machine model does not know it CONSUMES a linear source `OPEN — LATENT HARDENING, not a live miscompile (compiler lane verified 2026-09-10: the transforms this would license are each independently blocked today). The fix shape this entry first implied is WRONG — see the box`

> **CORRECTED 2026-09-10 by the compiler lane, on both counts that matter. The premise holds; the
> severity and the fix do not.**
>
> **Premise CONFIRMED.** `MOVC` is `hasSideEffects = 0` with `$rs1` a pure USE and no `Constraints`,
> while every sibling that consumes or modifies its source ties them — `SHRINK` (`$rd = $cap_in`),
> `DROP` (also `hasSideEffects = 1`), `PseudoINIT` and `PseudoSEAL` (`$rd = $rs1_in`). The comment at
> `:2643` even calls consumption *"the spec's MOVC step"* while MOVC's own definition does not model it.
>
> **But it is LATENT, because each transform I named is independently blocked, with a mechanism for
> each.** MOVC has **no IR pattern** — it is emitted only from `CapstoneInstrInfo.cpp:549`
> (`copyPhysReg`) and `CapstoneRegisterInfo.cpp:313` (frame-index elimination), both at or after
> register allocation, so MachineCSE and MachineSinking are pre-RA and never see it. It is not marked
> `isReMaterializable`, and `isReallyTriviallyReMaterializable` special-cases only RVV before the
> generic check. And it is not `isMoveReg`, while `isCopyInstrImpl` recognises only `isMoveReg` plus
> ADD/OR/XOR-with-X0, so MachineCopyPropagation will neither forward through it nor delete it.
> **It becomes live the moment someone adds an IR pattern for MOVC, marks it `isMoveReg`, or reaches it
> from a pre-RA pass.** Scope of that claim, stated by the auditor rather than by me: it is a static
> reading of pass ordering and instruction flags, and the post-RA scheduler, tail duplication and
> MachineLateInstrsCleanup were not audited.
>
> **THE FIX SHAPE THIS ENTRY FIRST IMPLIED IS WRONG AND MUST NOT BE APPLIED.** Mirroring
> `PseudoINIT`/`PseudoSEAL` with `Constraints = "$rd = $rs1"` would be a serious error: those are
> codegen pseudos where the tie IS the semantics, whereas **MOVC is the register-COPY primitive** —
> `copyPhysReg` builds it precisely when the destination differs from the source, so tying them defeats
> its only purpose and breaks every capability copy, including C-14's fix. The only defensible change is
> `hasSideEffects = 1` alone, and that pessimises every capability copy to insure a latent risk. **That
> is a trade for the lead, not a cleanup**, which is why no code has been changed.
>
> Also relevant and previously unrecorded: both emitters already pass a kill flag on the source, so the
> source is normally dead after the `movc`. The live-source case is the one `copyPhysReg`'s own comment
> argues the allocator never asks for on a linear capability, with the untagged case flagged as the
> residual — which is exactly where **C-32** sits.

`llvm/lib/Target/Capstone/CapstoneInstrInfo.td:2479-2483` declares

```
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def MOVC : ... (ins GPCR:$rs1)
```

with `$rs1` a pure USE and **no `Constraints`** tying `rd` to `rs1` — unlike `PseudoINIT` and
`PseudoSEAL` a few definitions below, which do exactly that. But `MOVC` writes `cnull` to `rs1`
whenever the source is not non-linear, so for a LINEAR source the instruction **destroys its own
operand** and the machine model says it does not. That licenses every transform that assumes a copy is
non-destructive: rematerialisation, CSE, sinking, and reordering across the copy.

**This is independent of Q-04.** Whether or not a SCALAR source is consumed is a spec question; that a
LINEAR one is consumed is not in dispute in any of the three implementations. So this is wrong today
regardless of how that ruling lands.

**Why it has its own ID.** It was named inside C-14 (*"The LLVM bug is bigger than the scalar case"*)
and a grep for it returned exactly one hit — that entry. C-14 is now FIXED for the scalar case, and
closing it would have taken this with it. Split out on the one-defect-one-ID rule, ID allocated with
`tests/next-issue-id.sh`.

**Not yet demonstrated.** No test shows a miscompile from the missing constraint; the argument is from
the definition. **What would settle it:** a directed case where a `movc` of a linear capability is CSE'd
or rematerialised and the source is read afterwards, predicting a fault or a null on the second read.
Until that exists this is a modelling defect by inspection, not an observed miscompile.

**Owner:** compiler lane. Related: **C-32** (the untagged-in-GPCR live copy), **Q-04** (whether a
scalar source is consumed at all).

### C-45 — the register+symbol call form `call a0, foo` (`PseudoCALLReg`) does not assemble `OPEN — found 2026-09-10 while fixing C-38; NOT a regression (the pre-fix 2026-09-04 binary rejects it identically); low priority, no known consumer`

Spun out of C-38 under the one-defect-per-commit rule. C-38 fixed the register+register form
(`call a0, a1`) by making `parseCallSymbol` decline register names; the register+**symbol** form that
`PseudoCALLReg` defines still does not assemble.

**Not a regression, established rather than assumed:** the pre-fix binary of 2026-09-04 rejects
`call a0, a1`, `call a0, a0` and `call a0, foo` identically, so this form has never worked on this
target. Documented in `cap-call-mnemonic.s` beside the C-38 case.

**No known consumer:** a search of the MC and CodeGen tests, the runtime glue and the compiler-rt
builtins found zero occurrences of `call` followed by a bare register and a symbol. So this is a hole
in what the assembler accepts versus what the instruction definitions declare, not a blocked user.

**DECISION 2026-09-10: land it this cycle, as its OWN commit on top of C-38's, never squashed with it.**

Three reasons in order of weight. (1) It is in the function C-38 just changed, so the code and its
test are open now; later it costs more, because whoever picks it up must re-derive why
`parseCallSymbol` declines register names. (2) The cost of leaving it is a latent trap — an
instruction form the definitions declare and the assembler silently refuses is what someone hits at
the worst possible moment, and today's "no consumer" search does not bind tomorrow's code. (3) The
risk is bounded: same file, same test, and a lit suite currently at 102/102 on the branch, so a
regression is immediately visible and immediately revertible.

**Separability is the condition, not a nicety.** C-38 is board-relevant (the disassembler round trip);
C-45 is not. If either has to be reverted they must come apart — which is also why keeping C-45 out of
C-38's commit was right.

**Owner:** compiler lane.

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

> **2026-09-10 — A FIX IS WRITTEN AND IS EXPLICITLY NOT GATED. Do NOT batch it into a bitstream yet.**
> RTL lane, branch `r24-excode-base` at `69658cf16` off `66c4e7517`.
>
> **Direction re-confirmed against the spec, because the RTL's own comment said the opposite.**
> `capstone-spec` gives 24/25/26/27/28/29 for unexpected operand type, invalid capability, unexpected
> capability type, insufficient permissions, out of bound and illegal operand value — i.e. `23 +
> ordinal` throughout. The two encoders emit `24 + ordinal`, so the encoders are the off-by-one and
> `commit_stage` was already conformant.
>
> **Two pieces of stale documentation went with it, and finding them is what settled the direction
> rather than assuming it.** The comment above the `ex_code` enum asserted that the encoders agreed
> with `riscv_pkg.sv` and called `commit_stage`'s base 23 *"an off-by-one in its own right"* — both
> backwards. And `riscv_pkg`'s four capability localparams carried the encoders' wrong numbers while
> being **declared and never referenced anywhere in `core/`**, which is why nothing ever caught the
> disagreement. That is the same shape as R-10's finding the same day: the authority everyone cites
> turned out to be unused.
>
> **What is NOT done, and it is most of the work:** no test updates, no lint, no auditor, no sweep.
> `excode-base-audit` must be rewritten — measuring this base is its entire purpose and it expects the
> old values *by design*, with a header explaining that the RTL is +1 against the spec — and eight
> other directed tests assert capability cause numbers and need checking. A known-good test still
> passes on the built model, which says the change does not break the core and is the ONLY gate it has
> passed. **Order when someone picks it up: the test updates FIRST**, because until
> `excode-base-audit` is rewritten the suite cannot tell a correct change from a broken one.
>
> **Consequence to carry into the same commit whenever it lands:** every capability `mcause` this
> project has recorded shifts by one, so the board driver, the wedge tracer and the historical
> readings need annotating — an old `mcause 25` reads as the new 24.
>
> ⚠️ **INSTRUMENT WARNING for that worktree.** `host-sweep.sh` returned `TIMEOUT 400013` for all nine
> tests run on it, *including* `init-rs1-ne-rd`, which passes through the container path in 6013 cycles
> on the same model. Those nine rows are a HARNESS failure, not readings; taken at face value they say
> this change hangs the core. Use the container path on that worktree until it is understood. Not
> chased.

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
