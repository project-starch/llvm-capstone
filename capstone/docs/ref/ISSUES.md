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


## Q-04 — QEMU's MOVC does not null a NOT_CAP source; the RTL does, and whether a scalar source MUST be consumed is an open spec question (the 2026-09-10 ruling that the spec settles it was RETRACTED the same day) `OPEN — QEMU/RTL divergence, filed 2026-09-05; the spec question is the lead's; 2026-09-15: masks C-32 on every emulator pass; 2026-09-24: CAPSTONE_MOVC_NULL_SCALAR=1 makes the emulator null it as the RTL does, opt-in`

> **2026-09-24: the emulator can now do what the RTL does here, on request.** capstone-qemu's
> `CAPSTONE_MOVC_NULL_SCALAR=1` makes `helper_csmovc` null an untagged source too (default off: the
> helper is unchanged). `capstone/tests/runtime-qemu/movc-null-scalar/run.sh` boots one -O2 domain
> with the switch off and then on. The stage-50 probe (two hand-written `movc` of one integer) reads
> `b=5 c=5` and then `b=5 c=0`. C-32's shape, the Sublet port's `setupLookaside` reduced to a
> function, keeps its address and then loses it to the null arm: the lookaside-off symptom, seen on
> the emulator for the first time. Against a QEMU that lacks the switch the test exits 2 rather than
> passing. With it on, the whole boot (monitor, Linux, the domain) completed in the one boot run so far. The first
> non-zero integer the switch nulled is the monitor's: `split_out_cap` passing a `region_cpmp` index
> to `read_cpmp` (pc `0x80020a50`). None of this rules the spec question. It removes the reason an
> emulator pass could not stand in for the board on it.

> **2026-09-15: this divergence MASKED C-32 on every emulator pass of the SQLite Sublet port at -O1/-O2.**
> The port's `setupLookaside` moves an integer-bridged block base through `movc` and re-reads the
> source; QEMU keeps it, the RTL nulls it, and the lookaside was silently OFF on silicon for every
> optimised-image board run (sw80b–sw82, arm A) while the emulator's counters said it was on. Found
> from the port's own counters, confirmed by `--stats` on the board and by the RTL/QEMU sources (see
> C-32's 2026-09-15 box). Whether a scalar source SHOULD be consumed remains this entry's open spec
> question, the lead's; nothing here rules it — but until the emulator agrees with the RTL on this
> point, an optimised image's emulator pass cannot stand in for the board on any path that casts
> integers to pointers.


> # ⚠ RETRACTED 2026-09-10, the same day it was made. THE RULING BELOW IS WRONG AND Q-04 REMAINS A GENUINE SPEC QUESTION.
>
> **The error, plainly: I mixed two numbering systems.** The ruling argued *"`NOT_CAP` is type 0, and
> `0 != 1`, so the source is nulled."* **The spec has no `NOT_CAP` type at all.** Its table
> (`capstone-academic-spec/parts/prog-model.adoc:177-184`) is Linear 0, Non-linear 1, Revocation 2,
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
> `capstone-academic-spec/parts/cap-man-insn.adoc:34-38` is the whole definition and it is not ambiguous:
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

`capstone-academic-spec/parts/cap-man-insn.adoc` (MOVC): "If `x[rs1]` is not a non-linear capability (i.e., `type != 1`), write `cnull` to `x[rs1]`" — a NOT_CAP source qualifies, and the RTL does it (`capstone_flu_unit.anvil:13-26`, rtl-oracle 2026-09-04). QEMU's `helper_movc` nulls rs1 only under `rs1_v->tag && !captype_is_copyable(...)` (`op_helper.c:580-585`), so an untagged source survives a `movc` under QEMU and dies on silicon. Consequence: every copy of an integer-bridged pointer that stays live passes under QEMU and loses its value on the board (C-32, `c32-movc-untagged-live.ll` — no longer XFAIL: it was removed in `46c53b7b6ae2` and the test passes); QEMU is a permissive oracle for that whole class until this is aligned with the spec. Fix belongs in `capstone-qemu`; the compiler side is C-32.


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
>    against itself. ~~Recommended EXCLUSIVE — every access path and all of QEMU already assume it.~~
>    **The lead's, with the spec's owners.**
>
>    > ⚠ **THAT RECOMMENDATION IS SUPERSEDED — see the correction further down this entry
>    > (search "THE RULING WAS MADE ON A FRAMING THAT IS WRONG IN BOTH DIRECTIONS").** The question
>    > is not one question, the three arguments for exclusive collapse to one, and the ruling commit
>    > **`d71d5007ee05`** was withdrawn the same day it was made. It is still findable by its subject
>    > line and has already been read as live by another lane once, which is why this pointer sits
>    > here rather than 500 lines below where the correction is. The surviving resolution fixes **one
>    > token on each side** and moves no convention. Nothing is decided until the lead re-rules on the
>    > corrected framing.
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
> spec exceptions (`capstone-academic-spec` `parts/cap-man-insn.adoc:415-421`: 24 unexpected operand type, 26
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
(`capstone-academic-spec/parts/cap-man-insn.adoc:446`) and the RTL does the same
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

### R-42 — a speculatively fetched I-cache MISS that a taken-branch redirect kills is never refilled, so a loop whose taken branch ends a 16-byte line pays +1 cycle EVERY iteration `OPEN — performance, not correctness; found 2026-09-24 (ladder control, E2b); identical at 054cea69b and 4ad0df694; FIX SYNTHESIZED at capstone-ariane 6cbdaeeb4 (2026-09-24): no new combinational loop, bitstream written, routed WNS -10.615 against 4ad0df694's -8.341; not on silicon, and the flash is the lead's call`

**Symptom.** Search terms: cycle anomaly, loop alignment, layout-dependent CPI, 7 vs 6 cycles
per iteration. The ladder control `ctrsanity` (identical code on both halves) read **1.167×**
on `caplifive_m1_054cea69b` when its hot loop started at `…1ac`, and **1.000×** at every other
start address. A layout pad moves it; it follows no length and no boot position (phases 6–10,
`fpga-silicon-measurements-for-paper.md` §2).

**Mechanism.** Established from a waveform, a discriminating run and the RTL source. The front
end fetches PC+4 speculatively every cycle; this target's fetch is 4 bytes and its I-cache line
is **16 bytes** (`CVA6ConfigIcacheLineWidth = 128`).
- When a loop's **taken backward branch is the last word of its 16-byte line** (address mod
  16 = 12), the PC+4 fetch lands in the next line, which the loop never executes, and misses.
- The predicted-taken redirect asserts `kill_s2` (`frontend.sv:345`: `kill_s2 = kill_s1 |
  bp_valid`). On a killed miss the I-cache FSM goes `READ→IDLE` **without raising
  `mem_data_req_o`** (`cva6_icache.sv:294-295`; the refill request at `:301` is never
  reached). So the refill is abandoned, the line is never installed, and **the miss recurs
  every iteration**. It costs one cycle each time, because the redirect is accepted a cycle
  late.
- On a hit, the FSM accepts the redirect in the same cycle (`:275-286`).

**Evidence.** All of it is in `tests/rtl-smoke/ladder-revival-2026-09-22/`, as
`sim-loopalign*.S` and `sim-loopalign.result-lines.txt`, on `capstone-ariane 054cea69b`,
Verilator, `--sv_seed 1`.
- A 5-instruction loop runs **7.009** cycles/iteration with its branch at mod 16 = 12, and
  **6.008** at mod 16 = 0, 4 and 8. It behaves **identically in plain M-mode and after
  `capenter`**, so it is not a capability effect.
- Waveform: at the slow offset the fall-through fetch shows `cl_hit=0000`, `kill_s2=1` and
  READ→IDLE every iteration. At the fast offset the same fetch shows `cl_hit=0001` and the FSM
  stays in READ.
- **Discriminator:** executing the line after the loop once, beforehand, removes the penalty,
  7.453 → 6.453 against the fast control's 6.406.
- It predicts all five board offsets by position mod 16.
- The source reading was confirmed independently by the RTL lane at `4ad0df694` (the R-35
  branch leaves both files untouched).

**Consequences.**
- **Every ladder cycle ratio can carry up to one cycle per hot-loop iteration of layout cost**,
  in either half. That is why the layout-randomised table (phase 10) exists, and why
  `bs`/`janne`/`insertsort` carry bands.
- **Any cycle comparison across a rebuild or a reflash must hold loop layout fixed.** The R-35
  lane already excludes `take_cyc`/`give_cyc` from cross-reflash comparison for this reason.
- It affects silicon performanc**Fix (capstone-ariane `6cbdaeeb4`, branch `r42-icache-killed-miss`, one commit on `4ad0df694`).**
In READ's killed-miss branch the FSM now accepts the redirected request in the same cycle,
under the hit path's guards: `!inv_q && !mem_rtrn_vld_i` raise `ready`, a pending `req` stays
in READ, and a `kill_s1` cycle still goes to IDLE, because the frontend then presents a stale
address. **The killed miss is still abandoned**: no refill and no wrong-path fetch. Only the
one-cycle IDLE detour goes.
- **This is NOT the shape first noted here** ("let a right-path demand miss complete its
  refill"), and that sentence was wrong. The killed fetch is the fall-through, which the loop
  never executes, so it is a wrong-path fetch. Completing its refill would be a wrong-path
  prefetch, blocking the cache for a miss latency and possibly evicting useful lines.
- **Sim evidence** (Verilator, `S12_MEM_DELAY=12`, seed pinned):
  - on the loop-alignment pair, the slow-offset arms go 28035 → 24036 cycles, exactly one
    cycle per taken branch (3,999), and the fast offsets are unchanged;
  - warm arm A goes 477 → 415, and B/C are unchanged;
  - the baseline arm reproduced all 13 of the board lane's numbers;
  - the neutrality sweep shows 0 differing trap counts across 92 tests;
  - `rtl-lint-gate` PASS, at baseline.
- **Synthesis (synth lane, 2026-09-24, sealed at `6cbdaeeb4`), against `4ad0df694`:**
  - **No new combinational loop.** `check_timing` reports 1 in both, and it is the SAME loop: the
    TIMING-23 arc is `ex_stage_i/lsu_i/state_q[3]_i_19` (I1→O) in both builds. So a count of 1 → 1
    is not hiding a swap. `LUTLP-1` = 0 in `drc_routed.rpt` (`CFGBVS-1` present as control).
    `write_bitstream` completed; all 241,250 routable nets routed. This settles the one question
    lint could not: `ready`'s new dependence on `kill_s2` closes no loop.
  - **Area neutral:** LUT 171,503 (+3), FF 93,812 (+2), LUTRAM unchanged. `i_cva6_icache` 736 LUT /
    136 FF (746 / 136 before).
  - **WNS −10.615 against −8.341 (Δ −2.274); failing endpoints 55.23 % against 51.70 %.**
    Pre-registered ±0.5 ns, so the prediction MISSED. The band was too narrow, and prior art said
    so: the S2 null-duplication control moved WNS by 3.74 ns with no functional change (see S2
    below). The worst path is the base's own class (`csr_regfile cpmp_q_reg` →
    `issue_read_operands fu_data_q`), with fewer logic levels and more wire. It has 0 I-cache and
    0 frontend cells, and so do all of the worst 500 (0/500 in both builds; control: the sibling
    `i_wt_dcache` appears on 500/500). The new failing endpoints fall in the D-cache and the issue
    stage, while the I-cache's count is unchanged (294 → 294). That fits placement variance, but
    ONE build cannot separate variance from a real effect.
  - Bitstream sha256 `0cd45bb099a22b0636363ce7d955766803b6fd193dccc23f24478046032b8c05`, on the
    synthesis host. Not staged anywhere. Flashing trades ~2.3 ns of WNS for one cycle per affected
    loop iteration; the board has run a flashed bitstream at −12.425 (`1bfff7776`).
- **Board acceptance:** slow offset 7.009 → ~6.008 cyc/iter, and fast offsets unchanged.

te by the RTL lane, 2026-09-24.

### R-43 — R-35's fix DENIES ON A CACHE MISS, so a live capability whose id was evicted is falsely refused; the rate under a large live-id population is unmeasured `OPEN — CONFIRMED 2026-09-25, on silicon (both R1 harness runs, caplifive_r42_6cbdaeeb4.bit) and in RTL simulation (r43-evict-live.S); safe direction (false DENY, never an authority escape) but it blocks every revocation-heavy workload on the R-35-fixed silicon; fix planned: docs/plans/r43-query-on-miss.md`

> **Scope.** The M-mode LSU revocation cache in `capstone-ariane 4ad0df694` (4 ways x 64 sets, exact
> 30-bit `{generation, index}` tag). An access whose id is not resident is refused with cause 25
> (`load_store_unit.sv`, the `!rvc_lu_hit` arm). That is fail-closed by design, and the RTL's own comment
> calls it **"an oracle rather than the shipping design"**: a miss is absence of proof of liveness,
> not proof of revocation. This entry is about the cost of that choice, not about safety.
>
> **Distinct from R-38.** R-38 is a same-cycle compare race in the CPMP tracker (S/U mode). This is a
> capacity/eviction miss in the LSU cache (M mode).
>
> **Evidence so far, and why it is weak.** On the flashed bitstream, ~43k live-alias accesses commit and
> 17/17 ladder rungs return their pre-flash values
> (`tests/fpga-repros/R35-revoked-reference-retains-authority/results/board-4ad0df694.result-lines.txt`).
> But those live ids were **recently minted**, so they barely exercise eviction. A program with more live
> revocation ids than the cache holds (SQLite) is the real test and has not run on this image.
>
> **CONFIRMED 2026-09-25 — on silicon and in simulation.**
> - **Board, boot r42b3** (image `1b7a04fe237e1580`, the R1 release-cost harness, clean in all 90
>   invocations on `054cea69b`): cause 25 on the first invocation. It was a load through the domain's
>   own globals capability (`ldc a1,0x80(gp)`), which was **allowed at `+0x4f40` and denied at `+0x4ff4`**
>   after an intervening `mrev`. The emulator runs the same invocation to completion. The RTL lane
>   re-decoded the capture and the disassembly.
> - **Board, boot r42b4** (image `46f99c7b5bf2556e`, the R1 cold harness): cause 25 on its first
>   invocation, a load in `run_series`, tval inside the domain's own block. The register was not traced.
> - **RTL simulation** (`verif/tests/custom/capstone/r43-evict-live.S`, capstone-ariane `93f509f54`,
>   on `6cbdaeeb4`): alias A reads fine after 16 new live nodes; after 512 more the same read traps 25
>   while `LCC(A) = 1`; after that `LCC` it reads fine again, because the LCC's node read re-installed it
>   through the read tap. Arms 1 and 2 differ only in the number of nodes minted.
> - Not affected: the ladder, the small m1 drop run (about 160 ids), and the live16 sweep (board, passed).
>
> **First experiments** *(written at filing; the sweep images exist, and the simulation test supersedes
> the board discriminator)*.
> 1. A SQLite workload on the fix bitstream: a correct result with no cause-25 trap bounds the rate.
> 2. The discriminator: a stale probe of the NEWEST age only (k=43295, just revoked, likely still resident
>    as dead), paired with a live alias installed more than 256 fills earlier and untouched since (a
>    probable miss). If the live one traps 25, the fix is selecting on residency, and this entry's cost is
>    real.
>
> **Remedy, parked:** query-on-miss (ask the rev-node unit on a miss). The first attempt broke the DYN
> unit's own queries by OR-ing a second requester onto the rev-node channel, invisibly to lint. A retry
> needs registered arbitration, a miss fixture first, and synthesis. **Any remedy must fail closed and
> re-pass R-35's acceptance fixture (exactly 7 traps).**

### R-44 — the CPMP tracker still ADOPTS any unseen revnode id as valid, so R-35's authority escape remains open for S/U-mode capability enforcement `OPEN — split out of R-35 on 2026-09-25 so that closing R-35 does not orphan it; LIVE IN PRODUCTION; fix deferred on a boot-kill risk`

> **The defect.** `core/pmp/src/pmp_data_if.sv:86-93` @ `4ad0df694`, inside `cpmp_revnode_tracking`:
> when a CPMP entry presents a revnode id different from the one it tracks, the tracker takes it and sets
> `cpmp_revnode_valid_d[i] = 1'b1` — *"assume the revnode is live until proven otherwise"*. This is
> R-35's mechanism, per CPMP entry instead of core-wide. R-12's own residual already said so: *"the
> same stale capability installed into a different CPMP entry is re-adopted until the next broadcast of
> its index."*
>
> **Why it is not latent.** The LSU check that R-35 fixed is gated `ld_st_priv_lvl_i == PRIV_LVL_M`; the
> CPMP check is gated `!= PRIV_LVL_M`. They are complementary, so the CPMP is **100 % of S/U-mode
> capability enforcement**. `capmode` is a sticky core-wide bit the monitor sets before the first S-mode
> instruction, so Linux and userspace run under it and cannot opt out.
>
> **Why it was deferred rather than fixed with R-35.** Copying R-35's fail-closed approach here has a
> boot-kill risk: `cpmp(0..2)` carry hardcoded ids produced with zero rev-node traffic, so a
> deny-on-miss tracker predicts a permanent deny on the first S-mode instruction fetch, i.e. an
> unbootable board. Reachability and fault-loop objections were raised and refuted; the boot-kill risk
> was not. Measured in simulation by `verif/tests/custom/capstone/r12-recl-cpmp.S` probe 4.
>
> **Sibling, latent:** `core/commit_stage.sv`, `pc_revnode_tracking`, adopts the same way for the PC
> capability ("On domain change ... assume valid"). It needs a stale CODE capability via CALL/RETURN,
> which has never been constructed.
>
> **What a fix needs:** positive evidence for the CPMP entries (R-35's taps can feed a second consumer)
> plus an explicit seed for the hardcoded `cpmp(0..2)` ids, so they are not denied at boot. It goes to
> synthesis before any board time, and a first S-mode boot is its acceptance.

## Q-08 — no capability fault path assigned `env->badaddr`, so `tval` was stale on every capability fault ever reported `FIXED 2026-09-11 in capstone-qemu cabc953e58; found while root-causing an unaligned capability store in SQLite`

**Word any restatement of this carefully.** `grep -c badaddr target/riscv/op_helper.c` returns **2**,
at `:656` and `:1002`, and a reviewer who runs the obvious grep and sees a non-zero count stops
reading. Both occurrences are prose inside comments and neither assigns anything. Every real
`env->badaddr =` in the tree is in `cpu_helper.c` (`:1217`, `:1262`, `:1287`, `:1990`), all on the
ordinary MMU paths.

**Consequence.** Neither the alignment check nor the bounds check below it ever wrote the field, so
the `cause/pc/badaddr` line the monitor prints carried whatever the last ORDINARY fault had left
there — readable-looking, plausible, and unrelated to the fault being reported. The investigated case
reported `tval = badaddr = 0x1015b0000` against a true faulting address of `0x101fb8168`.

**Verified after the fix** on the same reproducer: `tval` and `badaddr` both read `0x101fb8148`,
equal to the address the debug line prints.

**NOT a defect, recorded because both lanes inferred one from it.** The alignment path does not call
`cpu_restore_state` and the bounds path does. That asymmetry is about the PRINT, not the trap:
`riscv_raise_exception(env, excp, GETPC())` reaches `cpu_loop_exit_restore`, which restores state
whenever `pc` is non-zero (`accel/tcg/cpu-exec-common.c:75-81`), so `mepc` was always correct on both
paths. The explicit call exists only so that path's own debug print can read `env->pc`, which inside
a helper is otherwise whatever was last synced at a translation-block boundary. **An asymmetry
between two code paths is a question, not a finding, until you have read what the common callee
does.**

## Q-09 — a misaligned capability STORE raised `LOAD_ADDR_MIS` (cause 4) instead of `STORE_AMO_ADDR_MIS` (cause 6) `FIXED 2026-09-11 in capstone-qemu cabc953e58`

The 16-byte alignment check in `op_helper.c` sat above and independent of the `is_store` branch below
it, and raised `RISCV_EXCP_LOAD_ADDR_MIS` unconditionally. So every misaligned `stc` reported itself
as a misaligned load. Both constants exist in this tree (`cpu_bits.h:676` and `:678`); only the load
one was used.

Filed separately from **Q-08** because it moves a reported cause number rather than a reported
address, and separately from the diagnostic improvement that landed with it. Verified on the
reproducer: `cause = 6`.

**On reading either of these on silicon:** R-24 records the exception encoder as off by one depending
on which unit raised the fault, so a board-side cause number is not the same claim as an emulated
one, fixed or not.

## Q-10 — `CSSPLIT` with an out-of-range `mid` ABORTS the emulator instead of raising `OPEN — QEMU divergence, filed 2026-09-11; same class as Q-07, which was fixed by trapping`

`helper_cssplit` (`target/riscv/op_helper.c:1105-1131`) prints a good diagnostic and then
`assert(mid > ...bounds.base && mid < ...bounds.end)`. The assert kills the host process. A reachable
operand combination therefore takes the emulator down rather than raising a capability exception the
domain could report — which is exactly the shape Q-07 records for `INIT`, and Q-07 was resolved by
making it trap.

**How it is reached, with the arithmetic, because it is not exotic.** The domain's entry glue carves
SQLite's arena by splitting it off the **top** of a parent capability. Raise `SQLITE_HEAP_SIZE` past
that parent's span and `mid = end - heap` falls below `base`. Observed 2026-09-11 at a 2.5 MiB arena
on the speedtest1 path:

    requested split (end - mid)   2,621,440   = SQLITE_HEAP_SIZE, exactly
    parent capability span        2,507,248
    shortfall                       114,192   = how far mid falls below base

    capstone: CSSPLIT out of bounds -- mid=0x101d711f0 parent=[0x101d8d000,0x101ff11f0) (mid below base)
    qemu-system-riscv64: op_helper.c:1131: helper_cssplit: Assertion `mid > ...base && mid < ...end' failed

**NO THRESHOLD MAY BE DERIVED FROM THIS.** A run on 2026-09-10 (`qemu-run1.log`) issued the same
2,621,440 request with this assert already present and did **not** trip it, so the parent's capacity
is not a fixed number and any figure taken from one run describes that run.

**What the correct behaviour is has NOT been established here.** It should raise rather than abort,
but which exception the spec and the RTL specify for an out-of-range `CSSPLIT` is an rtl-oracle
question and is not answered by this entry.

**Two warnings about how this was found, because both are the classes this file exists to catch.**
It was first reported as an instance of **S-14** and that was wrong: S-14's signature is
`Cap mem access requires capability` followed by `cause = 24`, and `cause =` appears **zero** times in
any of the five logs from that experiment while firing in eleven others in the same directory. The
classifier that produced the misreading keyed on `SQ: E/share1` present and `SQ: G/enter` absent —
a condition satisfied by a pre-entry capability fault **and** by an emulator abort, so it could not
separate the two hypotheses on the table, with the diagnostic line sitting in the log. **Read the
diagnostic, not the marker sequence.**

**It may be what S-14's trigger (b) always was.** That trigger is recorded as "a 2.5 MiB heap arena,
no define involved", reported and withdrawn within the hour as an N=1 over-claim. It has the same
arena size and the same path. Not asserted — stated as the thing to check before trigger (b) is
treated as an S-14 instance.

## Q-11 — capstone-qemu's `ldc` untags a loaded capability whose node is revoked; the RTL forwards it unchanged, so a type read after an ancestor's revoke is 7 there and 2 on silicon `OPEN — for the collaborator (capstone-qemu); found by E1 on hardware 2026-09-14; enforcement on use agrees on both`

Three S2 cells of the nginx probe (`ngx_uaf.c` stops 7 and 8, `ngx_subpool_test.c` phase 13) read
the TYPE of a subordinate handle after its ancestor's REVOKE: the emulator reports 7, the FPGA reports
2 (r1b2: C70227 vs C70277, C80021 vs C80071; r1b3: 0D3E04 vs 0E3E00, the four `== SUBLET_TYPE_NONE`
checks of phase 13). RTL-oracle reading, quoted: both type queries (`LCC` selector 1,
`capstone_dyn_unit.anvil:206-208`; `helper_cslcc` case 1, `op_helper.c:894-896`) return the register's
stored type field; the difference is `ldc`: QEMU's writeback `helper_reg_set_cap_compressed`
(`op_helper.c:1583-1591`) calls `capstone_cap_revoked` and clears the tag of a loaded capability
whose node is invalid, so the later type read takes the untagged short-circuit (`:856-858`, returns 7);
the RTL's `LDC` (`capstone_dyn_unit.anvil:355-357`) validates only the ADDRESSING capability's node and
`check_load_data` (`capstone_unit.anvilh:583-609`) forwards the loaded value verbatim. Encoding is
ruled out (both number the post-shift types LIN 0, NONLIN 1, REV 2, UNINIT 3; the RTL's 7 is
`NOT_CAP(0) - 1` wrapped, the emulator's 7 is its untagged fallback). Enforcement on use agrees:
stop 9 offers the stale handle to REVOKE and both machines refuse it (FPGA: wedge with mcause 26
`INVALID_CAPABILITY` latched at `revoke t0` in `sublet_give_to`, mepc 0x81B03D2C − DBAS
0x81B00000 = +0x3D2C; emulator: cause 24 under its own numbering, and it aborts before capstone-qemu
PR #4). Consequences: (1) `sublet.h:175-176`'s model ("type 7 means the slot is empty") holds on
silicon, where a revoked-but-present handle (2) and an empty slot (7) are distinguishable, and the
emulator collapses both to 7; (2) `ngx_subpool_test.c` phase 13's four `SUBLET_TYPE_NONE` expectations
encode the emulator's behaviour and pass there for the wrong reason — the port's owners' fix; (3) the
paper's tab:safety hierarchy rows that rest on "reads NONE after withdrawal" are emulator evidence.
A second divergence found on the way: QEMU's `LCC` VALIDITY selector (imm 0) is a stub
(`op_helper.c:891-893`, "always valid for now") where the RTL queries the revocation node; nothing in
the ports uses selector 0 today. For the collaborator (capstone-qemu).

## Q-12 — capstone-qemu's `ldc` leaves a loaded LINEAR capability in its memory slot and its `stc` leaves a LINEAR register source intact; the deployed RTL clears both `OPEN — QEMU divergence, found on silicon 2026-09-15 (boot sw8x-f4, §7w); for the collaborator (capstone-qemu)`

The R1 harness's `--series linear` on image `55e6a187d52e5cc8`: after `ldc t0 <- slot` of a LINEAR capability
the slot reads 0 (still LINEAR) on the emulator and 7 (empty) on silicon; after `stc t0 -> slot` of a LINEAR
register, `t0` reads 0 on the emulator and 7 on silicon; the NONLIN controls read 1 on both, and
`movc`/`cincoffset`/`scc` clear the linear source on both. The spec (`mem-access-insn.adoc:54-55, 105`)
requires both clears; R-21's text already noted that `trans_csldc`/`trans_csstc` write no `cnull`. Impact: a
port that relies on a linear capability surviving in memory after a load, or in a register after a store,
passes on the emulator and loses it on silicon — the opposite polarity from Q-04/C-32. `sublet.h` stores
every loaded linear capability back and reads the base before the store, so the Sublet port is unaffected.
Related: R-21, R-22 (resolved on silicon), Q-04.


## Q-13 — an out-of-range TIGHTEN permission immediate (> 7): the RTL faults, capstone-qemu clamps it to NA `OPEN — model divergence and a spec question for the lead; found 2026-09-25 (E3 boot r42e3) through a harness encoding bug`

**The two models disagree:**
- **RTL**, `capstone_dyn_unit.anvil:262-273` @6cbdaeeb4: `(imm_v > 5'd7) || ((imm_v & rs1_perm) !=
  imm_v)` → ILLEGAL_OPERAND_VALUE, mcause 29 (`capstone_unit.anvilh:319`).
- **capstone-qemu**, `target/riscv/op_helper.c:1164-1189` (`helper_cstighten`): `perms > 7 ?
  CAP_PERMS_NA : perms`. An out-of-range immediate silently means "no access", and the operation
  proceeds.

A program that passes an invalid immediate therefore runs on the emulator and traps on silicon. The
first instance was E3's R1 harness encoding a register NUMBER (12) into the immediate field; see
R-21's box. **Which behaviour is intended is a spec question**, and neither source reads as the
specification. The RTL's refusal is the safer default.
## R-32 — the spec and the RTL still disagree by ONE on every bound taken or returned as a VALUE `OPEN — decision deferred 2026-09-10; ALL FOUR MEASURED. Only two are convention questions; SHRINKTO is an RTL off-by-one and SEAL's check is inert (S-11)`

> **This is the residue of the `end`-convention resolution, and it is deliberate rather than
> overlooked.** That ruling fixed each document's *outlier arithmetic* and moved no convention: the
> RTL keeps an exclusive `end`, the spec an inclusive one. Every **access** bound now agrees exactly
> — algebraically, for every region size and every access width, 4-byte scalars included. What it
> could not align, because aligning it would mean moving a convention, is every instruction that
> takes or returns a bound **as a value**.
>
> | | spec | this RTL | status |
> |---|---|---|---|
> | `SPLIT` at `val` | byte `val` in the **lower** half (`cap-man-insn.adoc:338`) | byte `val` in the **UPPER** half | **MEASURED** |
> | `LCC rd, rs1, 4` over a K-byte region | `base + K − 1` (`:197`) | **`base + K`** | **MEASURED** |
> | `SHRINKTO imm` | a region of exactly `imm` bytes (`:294`) | **`imm − 1` — and the RTL disagrees with ITSELF here** | **MEASURED** |
> | `SEAL` minimum size | ≥ 1024 (`:486`, `end − base + 1`) | **no minimum at all — the check is INERT (S-11)** | **MEASURED** |
>
> **⚠ `SHRINKTO` IS NOT THE SAME KIND OF PROBLEM AS THE OTHER THREE — it is an off-by-one in the
> hardware, not a convention difference.** Measured: over a 256-byte region with the cursor at base,
> `SHRINKTO 64` produces `start 0x80001000, end 0x8000103f` — **63 bytes, not 64** (406 cycles, no
> exception). `capstone_flu_unit.anvil:232-239` guards on `cursor + imm > end`, which is **exclusive**
> arithmetic, then sets `rd_end = cursor + imm - 1`, which under an exclusive `end` is one byte short.
> The guard and the effect use different conventions, so this is internal to the RTL: the spec is
> self-consistent the other way and would give exactly `imm`. Every other row here is two documents
> differing; this row is one document differing from itself, and a caller who knows the convention
> still gets the wrong size.
>
> **`SHRINK` was on this list and comes OFF it: it does not diverge.** `SHRINK` and `SHRINKTO` are
> different opcodes — `SHRINK` is R-type with `rd` an in-out capability and `rs1`/`rs2` plain integer
> bounds (`flu:178`), `SHRINKTO` is I-type `0x5B` funct3 `0b000` (`decoder.sv:1174`). The spec's
> `SHRINK` matches this RTL clause for clause: the integer-operand requirement, `base := x[rs1]`,
> `end := x[rs2]`, and all three illegal-operand conditions. Two attempts to measure "SHRINKTO" that
> trapped `UNEXPECTED_OPERAND` were passing a capability where an integer belongs — the wrong
> instruction, not a convention question.
>
> **The two measurements**, both directed tests on `capstone_cv64a6_imafdc_sv39`, both real
> completions rather than the harness's SUCCESS-at-timeout:
>
> * `verif/tests/custom/capstone/split-cut-side.S` (`capstone-ariane` `ec320921e`), 386 cycles. Base
>   `0x80001000`, split at `val = 0x80001040`: lower `[0x80001000, 0x80001040)`, upper starting **at**
>   `0x80001040`. Contiguous, no gap, no overlap — and byte `val` is in the upper half. The spec's
>   `rs1.end := val` (inclusive) with `rd.base := val + 1` is the other partition, equally gapless,
>   one byte across.
> * `verif/tests/custom/capstone/seal-minsize-boundary.S` (`f6ec6c198`), 670 cycles: a 1022-byte
>   region seals. Its negative control — sealing a NON-LINEAR capability, rejected on a different
>   branch — is the only raise in the run, which is what stops "no arm raised" from meaning "the
>   instrument never reached SEAL".
> * `verif/tests/custom/capstone/bound-value-readback.S` (`95930b8e3`), which now measures both: LCC
>   field 4 minus field 3 reads **256** over a 256-byte region — the exclusive end — and `SHRINKTO 64`
>   over that region yields **63** bytes. 406 cycles, no exception.
>
> **SPLIT is the one that is not a notation question.** The other three are readbacks: a caller that
> knows the convention can adjust. SPLIT is the same call producing a **different partition**, so code
> written from the spec that splits at `val` expecting `val` in the lower half gets it in the upper
> half on hardware.
>
> **Why this is OPEN rather than a defect, and why it is not urgent.** Nothing we build reads bounds
> from the prose: the monitor and the compiler take them from the RTL via `__capfield`, so they agree
> with the hardware, and access enforcement — the half that governs safety — agrees exactly. The
> exposure is to a future implementer, or to a reader of the spec. **The decision is the lead's,
> because whichever way it goes one of the two documents changes**, and it was deferred on 2026-09-10
> with that reasoning recorded rather than left implicit.
>
> **⚠ `SEAL` IS TWO DEFECTS STACKED, and the ±1 is currently UNOBSERVABLE.** Measured
> (`seal-minsize-boundary.S`, 670 cycles, one exception in the whole run — the control): a
> **1022-byte** region seals without raising, where the spec requires ≥ 1024. That confirms **S-11**
> (`fpga-repros/S11-seal-minsize-alignment-inert/`) on the RTL rather than from the generated
> Verilog: the Anvil check folds to `size < 1`, unreachable for any capability with `end >= start`,
> so **neither** the minimum size **nor** the 16-byte base alignment is enforced. Underneath it, the
> Anvil computes `size = end - start + 1` — an inclusive formula under an exclusive `end` — so it
> would admit 1023 bytes even once reachable. **Fixing S-11 without that `+1` ships a check that is
> wrong on its first day; they must move together.** The test's 1023-byte arm exists for exactly
> that and becomes the discriminator with no rewrite.
>
> **What would close R-32:** a ruling on the two convention rows (`SPLIT`, `LCC`). The decision
> content is reproduced below rather than left in `/tmp`, where it was written on 2026-09-10 and
> where it would not survive a reboot. The other two rows need no
> ruling at all: **`SHRINKTO` needs an RTL fix** (one token, `flu:237`), and **`SEAL` needs S-11
> fixed and the `+1` fixed in the same change**. Both should be tracked as work rather than waiting
> behind a documentation decision.
>
> ### The ruling, laid out — nothing here is urgent and nothing is broken
>
> This is a decision about which of two documents changes, and whichever way it goes, one of them
> does. Both rows are **measured**, not read off source: `split-cut-side.S` (386 cycles) and
> `bound-value-readback.S` (383 cycles), both real completions.
>
> ```
>   SPLIT at val      spec: val in the LOWER half      RTL: val in the UPPER half
>                     both partition with no gap and no overlap; they differ ONLY in
>                     which side of val the cut falls
>
>   LCC rd, rs1, 4    spec: base + K - 1               RTL: base + K
>   over K bytes      the spec returns the last byte, the RTL one past it
> ```
>
> | option | what it means | cost |
> |---|---|---|
> | **(a) spec follows the RTL** | `LCC` documented as returning an exclusive `end`; `SPLIT` documented with the cut on the upper side | cheapest — the silicon is flashed and running, and the monitor and compiler already read bounds from the RTL. But the spec's `end` stops meaning "the last byte" in these two places, so the document is no longer uniform |
> | **(b) RTL follows the spec** | change the hardware | uniform documents, but it moves silicon behaviour for an issue with **no known victim**, needs synthesis and a reflash, and every existing capability-manipulating binary wants re-checking |
> | **(c) document and change neither** | write the difference down where implementers will hit it | free and honest — and leaves the trap in place, which is exactly how this was found |
>
> **Recommendation: (a) for `LCC`, (c) for `SPLIT`, and they are deliberately different.**
>
> `LCC` is a **readback**: a caller who knows the convention adjusts once and is correct forever, so
> the spec is the cheaper thing to move and moving it costs nobody anything.
>
> `SPLIT` is not a readback. The same call with the same argument produces a **different
> partition**, so code written from the spec silently gets a one-byte-shifted cut on real hardware —
> the failure is in the data, not in a returned number, and it will not announce itself. That
> deserves a wider audience than a spec footnote before anything changes, which is why the
> recommendation stops at (c) and asks for a second opinion rather than taking (a) for both.
>
> **What would change the recommendation:** anything found that actually calls `SPLIT` and depends
> on the cut side — then it stops being a documentation question. Nothing in the monitor, the
> compiler or the LLVM backend was found to, **but that search was not exhaustive.**
>
> **If (a) is ruled**, the spec edits are small and go to `capstone-academic-spec` (branch
> `capstone-bootstrap`): `parts/cap-man-insn.adoc:197`, the `LCC` field table's `4` row; and
> `:336-339`, `SPLIT`'s operational steps if `SPLIT` is included. Prose-and-arithmetic only, no code
> impact on our side. **Note that this credential cannot push that repository** — it returns 403 on
> read as well as write — so the edit can be made here but not published from this lane.
>
> Full narrative, including the resolution this is the residue of:
> `history/10-09-2026_19-00-00_decisions-A-and-B.md`.


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

> # 2026-09-15 — THE COMMIT-STAGE ROUTE IS CONSTRUCTIBLE IN BARE METAL. Design, so nobody re-derives it.
>
> Read out of the RTL rather than reasoned: the trap the route needs is `commit_stage.sv:225-227`,
> `else if (!pc_revnode_valid_d) → cause 64'd25`, gated at `:208-209` on
> `priv_lvl_i == PRIV_LVL_M && capmode_i`. Its input is `pc_revnode_tracking` (`:231-247`), which latches
> `pc_cap.metadata.revnode_id` and **assumes valid on any change**, clearing only when
> `revnode_invalidation_id_i == pc_revnode_id_d` — the broadcast produced at `ex_stage.sv:1207-1208`
> **during a write op's own execution**. So a REVOKE whose walk reaches the PC capability's node
> invalidates it mid-flight, the commit head is the REVOKE itself, and it traps instead of retiring —
> having already mutated node state in EX. R-28's shape with no interrupt and no debug flush.
>
> **It must be a DESCENDANT, and that is a constraint not a detail.** `REVOKE_NODE`
> (`capstone_rev_node.anvil:15-31`) terminates on `node_in.depth <= *depth_bound` and invalidates only
> what lies below, so revoking a node does **not** invalidate the node itself. The PC capability's node
> has to lie in the junior run of the node being revoked.
>
> **~~Construction: CAPENTER with a child capability.~~ RETRACTED within the hour — `CAPENTER` does not
> take a capability, and the route is NOT reachable from a bare-metal test at all.**
>
> `CAPENTER` mints its capabilities with **hard-coded node ids**: `capstone_flu_unit.anvil:455` gives
> the returned revoke capability `revnode_id = 30'd1` and `:471` gives the data capability `30'd2`, and
> `commit_stage.sv:197` gives the **PC capability `30'd1`**. Nothing the program holds influences any of
> them. So "enter with the child" is not a thing `CAPENTER` can do.
>
> **And no obtainable capability is rooted above the PC's node.** `depth_bound := node_in.depth`
> (`capstone_rev_node.anvil:154`), the walk starts at `node_in.next` and stops once
> `node_in.depth <= *depth_bound` (`:18`), so a REVOKE invalidates **strictly deeper** nodes only —
> never the revoked node itself. Nodes 1 and 2 are both depth 1 (`:173`, `:175`) and are siblings
> (`node_1.next = 2`). Therefore: revoking through the node-1 capability sets `depth_bound = 1`, walks
> to node 2, finds `1 <= 1`, and terminates having invalidated nothing. MREV/SPLIT only ever produce
> **deeper** nodes, so they cannot manufacture an ancestor either. The only node above depth 1 is node 0,
> the depth-0 sentinel, which is created with `valid = 1'd0`.
>
> **This is very likely why the earlier 37 arms never reached the condition.** The entry records that
> as a timing problem — no arm had a measured chance of landing an interrupt on an in-flight op. The
> structural reading is simpler: **in the bare-metal configuration the required state cannot be
> constructed at all**, so no amount of timing would have produced it.
>
> **What the route actually needs** is a PC capability carrying a node the program can revoke from
> above, which happens on a **domain switch** — `commit_stage.sv:235` names CALL/RETURN alongside
> CAPENTER, and a CALL's PC capability comes from a sealed domain capability whose node the monitor
> created as a descendant. That is monitor-level setup, not a directed `.S`. Anyone attempting this
> should budget for it as a domain test, and the oracle design below still applies unchanged.
>
> *(Recorded as a retraction rather than an edit because the wrong construction was published first and
> may have been read. The feasibility check that refuted it cost minutes; writing the arm would have
> cost a day and produced a clean, meaningless pass.)*
>
> **THE ORACLE, and it sidesteps the idempotence problem this entry records.** The entry states that no
> end-state oracle works for DROP/REVOKE because invalidation is idempotent. That is true and it does
> not block this route, because **the observable here is a PAIR, not an end state**: (a) did the REVOKE
> trap with cause 25 — read from a `mtvec` handler that latches `mcause`/`mepc` and steps `mepc` by 4 so
> the run does not spin on itself; and (b) was the node mutated anyway — read with `LCC(rd, cap, 0)`,
> selector 0 being validity (`capstone_dyn_unit.anvil:231-240`). Mutation observed **together with**
> non-retirement is the demonstration. Neither half alone is.
>
> Expect the run to reach its report and then time out, for the usual reason: completing needs a store
> to `tohost` through an `auipc`-derived integer base, which the R-34 delivery fix refuses under capmode.
>
> **Two things to get right or the arm is void.** The handler must step `mepc`, or the trap re-executes
> the REVOKE, traps again and hangs — a hang here is indistinguishable from the wedge this entry is
> about. And validity must be sampled **before** as well as after, or "invalid" cannot be told from
> "never valid".

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


### S-15 — the speedtest1 port DE-LINEARISES a grant the monitor has already de-linearised, and on silicon that is a fault, not a no-op `ACCOUNT STRENGTHENED BY THE MATCHED PAIR 2026-09-13 (sw66 hangs at share3, sw67 with a trap vector returns from it and dies at sqlite3_initialize); MECHANISM TRACED THROUGH THE RTL AT THE BITSTREAM COMMIT (DELIN raises UNEXPECTED_CAP_TYPE on any non-LINEAR operand; the delin is the only type-sensitive instruction in the branch); MCAUSE READ BACK ON SILICON 2026-09-13 (sw69: the trap word the glue's handler wrote into the shared arena is arena0=0xF6C09D13, mcause field 27 = UNEXPECTED_CAP_TYPE, written during the share3 domcall; the run then failed with obs=0x5117BAD3 = the sqlite3_initialize failure sw67 died on); FIX PROVEN ON SILICON 2026-09-13 (sw68: the image with the delin removed and nothing else, no trap vector, passes share3 and RAN size-20 to completion at 64,732,455,367 cycles / ratio 1.194 where sw64/sw66 hang); S-15 CONFIRMED end to end — mechanism, fault named, and fix all on silicon`

> # ⚠ THE CONFIRMING ARM DID NOT CONFIRM IT (boot sw65, 2026-09-13). Read this before citing the chain below.
>
> **RESOLVED THE SAME EVENING (sw67, the proven matched pair).** sw66 re-ran sw64's exact image and
> hung at share3's `SHA5` again; sw67 ran the same image plus `INTERP_DOMAIN_MTVEC=1` (proven: the
> recipe without the flag reproduces `23da3b126a304585`) and **returned from share3** (`SHA6`),
> entered, and died at `sqlite3_initialize` -- `obs = 0x5117BAD3`, exactly sw65. The glue's own rule
> for its trap handler is "returns where the handler-less build hangs ⇒ the domain faulted", and it
> applies to the share entry -- an elimination argument, audited the same evening as broader than its
> evidence (share3 also differs in region residency and size, and the handler has no positive control
> on this bitstream), while the mechanism itself is now sourced at the RTL commit `1bfff7776`. The trap
> word is invisible because the handler writes it
> `sw t0, 0(a0)` through the region capability saved at entry -- for a share entry, the shared
> region itself, i.e. the arena's first word. Then `sqlite_arena` stays NULL, `CONFIG_HEAP(NULL, 0)`
> reverts SQLite to its default allocator (`sqlite3.c:187958`), the port's `malloc` stub returns
> NULL, and `initialize` fails. So the box below records an instrument gap, not a refutation: the
> arm asked for was run, and its answer went where nothing reads. **Owed:** the host reading `arena[0]`
> after share3 with the pair image unchanged -- predicted `0xF6C09D13` (mcause 27 at the delin).
> **sw68 (the fix, no trap vector): share3 `SHA5→SHA6`, `G/enter` -- the delin removed and nothing else
> turns the hang into a run; the region's residency and size are unchanged in that image.** Mechanism, evidence and the
> fix in §7o. **The fix** (the REGION_ARENA branch no longer delins; both comments corrected) is on
> `dev`; the silicon proof is sw68, the fixed image without a trap vector passing share3 and
> running. **Instrument fix LANDED 2026-09-13 (0f150add):** the host reads every REV_SHARED region's first word back after the share and reports a packed trap word (`SQ: share-trap=…`), stopping there; positive control sw72 (pair image → `0xF6C09D13`, mcause 27 at share3, no `G/enter`), negative control on QEMU (fix image → sentinel unchanged, oracle hash). Scope: trap-vector images only — a measurement image never returns from the faulting share; the entry watchdog bounds that case.
>
> **Boot sw66 (2026-09-13, later) closes the question from the other side.** The exact redraw of
> sw64's image stalled at share3's `SHA5` again, so sw64's stall is deterministic and lives in the
> domain's share-entry execution — but the arena branch that calls `__builtin_capstone_cap_delin`
> is code-identical between sw64's commit and `dev`, and sw65's image carried it and PASSED share3.
> So the double delin is not what sw64 hits either. The mechanism this entry names has no silicon
> evidence for it and one arm against it. §7o.
>
> The arm this entry asked for was run: the domain rebuilt with `INTERP_DOMAIN_MTVEC=1`, same
> REGION_ARENA configuration, 128 MiB arena. The glue packs a trap into the return value with **bits
> 31..28 = 0xF**. The arm returned **`obs = 0x5117BAD3`** — top nibble `0x5`, so **no trap was taken
> at all**. That value is `SQLITE_HC_ERR_INITIALIZE` (`sqlite_hostcall.h:147`), "sqlite3_initialize
> refused".
>
> **And sw64's hang did not reproduce.** `SQ: F2/share3ECSA:00000001`, then
> `SQ: G/enterENT0:00000001`, then `SQ: H/return` — share3 **completed**, the domain **entered**, and
> the run got as far as SQLite's own initialisation before failing. Control `k800 retval=4`,
> cycles 4558. Arm returned in 26 s.
>
> **So the predicted fault did not occur in a build that compiles the very branch this entry
> blames.** The REGION_ARENA branch — and therefore the `cap_delin` at `speedtest1_measure.c:722` —
> was compiled in (the build logged "arena from a REGION: 134217728 bytes, shared third"). The domain
> de-linearised a `REV_SHARED` grant and did **not** fault. Either the monitor's `__delin` at
> `sbi_capstone.c:1396-1399` did not fire for this grant, or a second delin is tolerated — and this
> entry asserted the opposite.
>
> **What this does NOT establish, stated so the entry is not over-corrected.** sw65's domain is
> **not** sw64's image — 1,684,352 bytes against 1,861,088, a different build — so non-reproduction
> is not proof that sw64's hang was something else. It removes the *general* claim, not the specific
> one. Two readings now sit open for sw64: a per-image entry/share stall (R-16's family), or a
> build-specific difference between the two domains that this lane has not identified.
>
> **A genuinely new positive, and it further undermines the retracted size story:** a **128 MiB
> `REV_SHARED` arena CAN be shared to a domain and the domain CAN enter** on this silicon. That was
> unknown before sw65 — the path had never completed at any size.
>
> **Also learned, and it blocks the instrument this entry proposed:** `trapctl`, the control the
> source requires beside any `INTERP_DOMAIN_MTVEC` build, **cannot earn a QEMU pass**. Its deliberate
> out-of-bounds `ldc` kills QEMU (register dump, EOF) rather than raising a trappable fault, so the
> staged preflight blocks it permanently — the same class of divergence as this entry's own QEMU
> caveat. sw65 ran without it on the narrower ground that trapctl disambiguates a *negative*, and a
> returning arm is self-proving.

> **The chain, each link quoted.** The host shares SQLite's arena with annotation **`REV_SHARED`**
> (`ports/sqlite/sqlite_host.c:584-586`). The monitor **de-linearises it before handing it over** —
> `sbi_capstone.c:1396-1399`, `if (cap_type(r) == CAP_TYPE_LINEAR) r = __delin(r);` — so the domain
> receives a **NONLIN** capability. The domain then de-linearises it **again**
> (`ports/sqlite/speedtest1_measure.c:722`, `__builtin_capstone_cap_delin`). This RTL's `DELIN`
> raises **`UNEXPECTED_CAP_TYPE`** for any non-LINEAR operand.
>
> **Why that presents as a hang rather than a fault — this is [[M-1]]'s first measured cost.** A
> domain runs with `mtvec = 0`, so the fault vectors to pc 0, lands outside PCC, and re-faults
> forever in silence. Boot sw64 showed `SHA5` (the monitor about to enter the domain) and never
> `SHA6`, then **eight hours** of nothing. M-1 has been `OPEN — OURS, FIX FIRST` for some time on
> principle; this is the first time it has been billed.
>
> **QEMU cannot see it, by construction.** `helper_csdelin` was patched to return early when the
> capability is already NONLIN, so the double de-linearise is silent under emulation and fatal on
> silicon. Every emulated run of this configuration is green, including the size-20 pair that was
> used to justify the board run.
>
> **The source contains the contradiction that let it ship.** `speedtest1_measure.c:705` states the
> REGION_ARENA grant "arrives already-NONLIN from a REV_SHARED share, where the delin is redundant";
> `:713`, *inside that very branch*, states "The grant arrives LINEAR". The monitor settles it in
> favour of `:705`. The wrong comment is what made a redundant call look load-bearing, so the fix has
> to correct both.
>
> **It is NOT a region-size problem, and an earlier reading of it as one is retracted in §7n.**
> `share3` is `#ifdef SPEEDTEST1_REGION_ARENA`; sw56's capture reads `share3=0`, so §7k never shared
> a region arena at all and neither did §7m's bridge. **There is no known-good point for this path at
> any size — sw64 was its first board exercise, ever.** Nothing on the `SHA5`→`SHA6` path scales with
> length: the only length-bounded loop in the monitor is the reclaim fill, gated on
> `cap_type == UNINIT` and sitting before SHA3/SHA4/SHA5, all of which printed.
>
> **What would settle it, and what would NOT.** Build the domain with `INTERP_DOMAIN_MTVEC=1` and
> re-run the same configuration at the same size: a trap vector converts the silent re-fault into a
> labelled return, and the prediction is `mcause & 0x3F == UNEXPECTED_CAP_TYPE` with
> `(mepc − _start) >> 2` landing on `:722`. **An `--arena N` size ladder would NOT settle it** —
> `--arena` shares `REV_BORROWED` (`sqlite_host.c:601`), which the monitor does not de-linearise, so
> the domain's delin is legal there and the ladder would return a clean sweep that proves nothing
> about this fault.
>
> **Fix, once the reading confirms:** delete the delin at `speedtest1_measure.c:722` and correct the
> comment at `:713`. Related: [[M-1]] (the silence), §7n (the boot), §7m (unaffected — the bridge used
> the static-heap build, two shares, and its seven pairs stand).

### S-14 — `__capstone_cap_init` reloads a capability spill slot with a scalar `ld`, so the tag is lost before the domain runs `FIX LANDED 2026-09-13 (#14) — the gp-free `ra`/`c1` 8-byte spill is now gated on FrameSetup/FrameDestroy, so only the prologue return-address save takes the integer form and an allocator spill of a capability parked in c1 keeps the 16-byte `stc`. VERIFIED BY REBUILD-AND-DIFF on the 160-test MicroPython image: __capstone_cap_init went from 14 `sd ra` 8-byte spills (13 truncating a capability = the S-14 shape) to 1 (the prologue save), reloads `ld ra`→`ldc ra`. The shipped lit test does NOT gate the defect (its 20-global shape never spills ra; byte-identical pre/post-fix) and capinit-scan under-detects the spill form; the DIRECT gate is the `sd ra` count in a high-pressure cap_init. Regression: lit + QEMU corpus green (this landing).`

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

> ## ⚠ A SECOND INSTANCE OF THIS SIGNATURE, WITH A COMPLETELY DIFFERENT TRIGGER (2026-09-10)
>
> The bench lane hit **the same `SQ: E/share1`, cause 24, before the domain enters** while sizing
> speedtest1 — on a **2.5 MiB heap arena**, with no build define involved at all. They reported it
> first as a classification rule ("that signature means the heap is too large") and then **withdrew
> that themselves within the hour**, unprompted, as an N=1 over-claim. The withdrawal is the right
> call and is why this box says what it says.
>
> **A THIRD TRIGGER, 2026-09-11 (bench lane, reported to the board lane and filed here on their
> behalf): adding `json` to the SQLite feature set.** Same `SQ: E/share1`, cause 24, before entry,
> **on every testset**. What makes this the most useful of the three is that it arrives with the
> bracketing series the entry above asked for — the same image built four ways:
>
> | build | `code_len` | `globals_off` | globals | result |
> |---|---:|---:|---:|---|
> | baseline | 1,483,608 | `0x150000` | 208 | runs |
> | + floating point | 1,551,080 | `0x160000` | 211 | runs |
> | + rtree | 1,621,624 | `0x170000` | 218 | runs |
> | **+ json** | **1,760,872** | **`0x190000`** | **234** | **FAULTS** |
>
> **Read what this does and does not settle.** It is monotone in all three quantities and the fault
> appears only at the largest, which is consistent with the geometry hypothesis and is the first
> evidence for it beyond "both triggers change the layout". It does **not** identify which of the
> three quantities matters, and it cannot: they move together across these four builds. Separating
> them needs a build that moves one without the others — the discriminator below.
>
> **What is actually known: three instances — a restored build define, an arena size, and a feature
> set — with three unrelated causes, the same signature, and NO mechanism established for any of
> them.** The tempting common factor is image or
> `dom_data` geometry — both triggers change the image's size or layout — but nobody has shown that,
> and this registry has a bad record with mechanisms that merely fit.
>
> **Consequences, both directions.** Do not read a future `E/share1` as benign because "it is just the
> heap"; and do not read S-14 as being about `EXPLAIN` specifically until something separates the two.
> If a board stage produces this signature it is an open defect, not a known quantity.
>
> **The bracketing series makes the discriminator sharper, not unnecessary.** Three triggers with
> three unrelated causes and one signature is now a pattern rather than a coincidence, and the next
> step is still to move ONE quantity.
>
> **2026-09-11 — ROOT CAUSE. A capability spill slot is written with `stc` and read back with a
> scalar `ld`, whose result is fed straight to `cincoffsetimm`.** The tag is gone by then, so the
> instruction faults with cause 24, "x[rs1] is not a capability". At the exact pc the emulator
> reports, in the json/`.bss` image:
>
>     189204: 83 30 01 26   ld            ra, 0x260(sp)
>     189208: db a0 f0 4c   cincoffsetimm ra, ra, 0x4cf      <- the logged fault pc
>     18920c: 5b 48 15 06   stc           ra, 0x70(a0)
>
> **Slot `0x260(sp)` is a capability slot accessed BOTH ways in the same function**: 33 `ldc` and 3
> `stc` against 7 `ld` and 4 `sd`. The scalar reloads are the defect. The function is
> `__capstone_cap_init`, which the compiler synthesises
> (`llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp`) and `start.S` calls before `domain_main` —
> so this is the **domain image's own code**, not the monitor's. It is straight-line, one `ret` and no
> branches, so a bad site is reached unconditionally and the fault is deterministic per image.
>
> **THE PATTERN SEPARATES FAULTING FROM RUNNING IMAGES PERFECTLY.** Counting scalar `ld` from an `sp`
> slot whose value the next instruction consumes with `cincoffsetimm`:
>
>     image            trigger                    arena          bad reloads   result
>     slt-explain      -USQLITE_OMIT_EXPLAIN      .bss 256 KiB        3        FAULTS
>     jsonbss          json feature set           .bss 2 MiB          3        FAULTS
>     slt-base         (control)                  .bss 256 KiB        0        RUNS
>     json-region      json feature set           region 6 MiB        0        RUNS
>     w3-bss           rtree                      .bss 2 MiB          0        RUNS
>     w3-region        rtree                      region 2 MiB        0        RUNS
>
> In both faulting images the FIRST such site is the pc the emulator printed.
>
> **`slt-base` against `slt-explain` retires the geometry hypothesis.** Same path, same 256 KiB `.bss`
> arena, same carve, one define apart — one clean, one with three bad reloads. **So triggers (a) and
> (c) are ONE defect, and the `dom_data` carve is not the variable.** Image size, globals count and
> carve are correlates that happened to move together in the four-build series; they are not the
> mechanism.
>
> **What the region arena actually did, stated so nobody reads it as a fix.** Removing the 2 MiB
> `sqlite_heap` array changes `__capstone_cap_init`'s codegen and the bad reloads disappear with it.
> That is **incidental**. Any change to the global set can reintroduce them, and an image can be
> clean today and faulty after an unrelated edit. **Gate on the reload pattern, not on a carve
> number.**
>
> **Two things this entry previously said that are wrong, corrected here.** It described the fault as
> being in "the monitor's entry glue" — it is the domain's own compiler-emitted `cap_init`. And it
> gives the signature as `Cap mem access requires capability` followed by `cause = 24`: **that string
> appears zero times in either faulting log.** The signature is `cincoffsetimm with an UNTAGGED rs1`
> followed by `cause = 24`. A reader applying the old wording would reject a true instance.
>
> **A classifier keyed on `cause = 24` alone is not enough either.** A `cause = 24` fault was observed
> the same day AFTER `SQ: G/enter` with no `UNTAGGED` line — post-entry, not S-14. The discriminating
> pair is the `UNTAGGED rs1` diagnostic plus the pre-entry marker state.
>
> **Still open:** which LLVM component emits the scalar reload — register allocation, spill-slot
> typing, or `CapstoneCapGlobalInit` itself. That is compiler work and is not diagnosed here.
>
> **ANSWERED AND FIXED, 2026-09-13.** Neither register allocation nor
> `CapstoneCapGlobalInit`: spill-slot typing, in `CapstoneInstrInfo::storeRegToStackSlot` and
> its mirror. Under gp-free the 8-byte `SD`/`LD` arm was selected by asking whether the
> register IS `$c1`, as a proxy for "this value is a return address". `C1` is an ordinary
> allocatable `GPCR` and sits last in the allocation order, so at `-O0` — where `RegAllocFast`
> passes the PHYSICAL register to these hooks, unlike the `-O2` spiller which passes a virtual
> one — a capability parked in `ra` under pressure was truncated to its address half. The arm
> is now gated on `FrameSetup` / `FrameDestroy`, which the frame save and restore carry and an
> allocator spill does not, so the prologue and every frame size are unchanged.
>
> **A FOURTH AND FIFTH TRIGGER, on MicroPython rather than SQLite, with a knob that flips it.**
> A domain built from the same source at seven test-table sizes: 4, 60 and 421 boot, 100 and
> 300 die inside `__capstone_cap_init` with cause 24, 160 and 200 die elsewhere and are NOT
> this defect. `capstone/tests/capinit-scan.py` finds 2 sites in the 100 image and 13 in the
> 300 image and none in the other five, and in both the FIRST site is exactly the pc the
> monitor printed, which is the property this entry claims for its own SQLite images.
>
> **THE QUIET FORM, which this entry did not describe.** When the untagged register is the
> ADDRESS of a capability access the fault is immediate and inside `cap_init`. When it is the
> DATA of an `stc` nothing faults there: an untagged capability-sized word lands in a global
> and the first read through it faults arbitrarily far away. On the 421 image that was
> `stc ra, 0x0(a7)` writing a type pointer, and the fault came 157 tests later in
> `mp_convert_member_lookup` reading `m_type->flags`. The full suite went from
> `PASS=269 FAULT=1` to `PASS=270 FAULT=0`.
>
> **WHY THE PATTERN AND NOT THE SHAPE.** This entry gives the shape it first saw, `ld <rd>,
> <imm>(sp)` consumed by `cincoffsetimm`. Both MicroPython instances differ: the slot is
> reached through a materialised `cincoffset sp, imm` and the consumer is an `stc`. A checker
> written to the stated shape reported all seven images clean. The gate keys on the invariant
> instead, that the address operand of a capability access must be a capability.
>
> **WHY THERE IS NO LIT TEST FOR THE DEFECT ITSELF.** It needs the physical register to reach
> the hook, so it needs `-O0`, and three shapes tried at `-O0` do not put a spilled capability
> in `ra`: a flat array of pointer globals up to 300 entries, a call with 24 capability
> arguments, and inline asm with 28 live capability inputs. The real `cap_init` reaches it
> because its initialisers are nested aggregates the pass walks recursively, holding one
> capability live across many leaves. `cap-gp-free-ra-spill-keeps-tag.ll` pins both arms of
> the decision so neither can be removed or widened silently, and says so in its header. The
> machine verifier cannot catch this form: an earlier fix made the instruction well-formed by
> storing the `X` half, so it is only semantically wrong.

> **2026-09-11 — THE DISCRIMINATOR RAN. Trigger (a) is CONFIRMED on its own path, and the fault now
> has a specific diagnosis. It also does NOT reproduce on the speedtest1 path.**
>
> Matched pair on the SLT path, which is the path this entry was recorded on — region 1 MiB, the
> 256 KiB default arena, no declared stack:
>
>     slt-base      RUNS   full A..H chain, `records`, guest exit 0
>     slt-explain   FAULTS at SQ: E/share1, no F/share2, no G/enter, after 8 CINCOFFSET gp traces
>
> The control is shown to have run rather than assumed. The faulting arm's diagnostic:
>
>     capstone-qemu: cincoffsetimm with an UNTAGGED rs1 -- pc=0x101d45174 rd=x1 rs1=x1
>                    val=0x101fad710 priv=3
>     [CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x101d45174, tval = 0x0
>
> **`priv=3` is M-mode (`cpu_bits.h:621`), and it corroborates "pre-entry" independently of the
> markers.** The faulting pc sits 1,331,572 bytes into a 1,450,840-byte image — the load base is
> `0x101c00000`, fixed by the glue's first cap-table access at `0x101c00034` — so domain-range code is
> executing in **machine mode**, which is the monitor's entry glue before the privilege switch, not
> the domain proper. Three independent things now agree that no SQLite instruction has run: the
> marker sequence, the absence of `G/enter`, and the privilege level. A reader checking whether this
> is really pre-entry should not have to take the marker's word for it.
>
> **Three readings, each with its warrant.** `rd` and `rs1` are both **x1**, the return-address
> register, so it is `ra` that has arrived untagged. `cause = 24` is spec-derived on this path —
> `op_helper.c:655-665` cites `cap-man-insn.adoc` for "Unexpected operand type (24): x[rs1] is not a
> capability" — so the number is trustworthy *under emulation*; R-24 remains about silicon. And by the
> instrument's own documented rule at `op_helper.c:640-646`, a **plausible non-zero** `val` means a
> capability that LOST its tag in memory — the S-07 family — where `val=0` would have meant an
> ordinary null pointer and a program bug. `val=0x101fad710` is plausible and non-zero.
>
> **Do not quote `badaddr` from this line.** That path assigns nothing to it (see **Q-08**), so it
> holds whatever the last ordinary fault left.
>
> **It does not reproduce on the speedtest1 path.** Four images — `SQLITE_FULL` off and on, each with
> and without `-USQLITE_OMIT_EXPLAIN` — all reach `H/return` with correct hashes, at region 64 KiB,
> 2 MiB arena and 1 MiB declared stack. **This does not isolate geometry.** The two paths differ in at
> least four respects: region size 16x, arena 8x, declared stack present versus absent, and
> `-DCAPSTONE_SQLITE_SLT=1`, which changes the program. A one-variable comparison between them does
> not exist yet.
>
> **A correction, recorded because it was circulated.** An earlier pass on 2026-09-11 reported trigger
> (a) as REFUTED and reported a 2.5 MiB-arena image as a third S-14 instance localising the fault to
> the `dom_data` carve. Both were wrong. The arena image had aborted QEMU on a `CSSPLIT` assert
> (**Q-10**) and never took a capability fault — `cause =` appears zero times in all five logs of that
> experiment and fires in eleven others in the same directory. The classifier keyed on `E/share1`
> present and `G/enter` absent, which a capability fault and an emulator abort both satisfy. Nothing
> from that pass reached this file.

> **The cheap discriminator nobody has run:** build the EXPLAIN-restored image and the oversized-arena
> image and compare their `.text` size, globals offset and `dom_data` carve. If those coincide where
> the working images differ, geometry is the factor and both entries collapse into one. That is
> off-board and needs no boot.

### I-6 — a userspace binary that dies **in M-mode** is probably reading a machine-mode CSR, not hitting a monitor bug `RECORDED 2026-09-10 — a diagnostic trap, not a defect; measured by the bench lane while building the speedtest1 matched baseline`

**The signature, and why it convinces.** A plain static Linux userspace binary dies, and the console
shows a fault whose `pc` is in **firmware space**:

    [CAPSTONE] Print = Scalar(0xdeadbeef), Print = Scalar(0x2)
    Cap mem access requires capability: pc = 80023318
    domain halted by capability fault: cause = 24, pc = 0x80023338

**Two counter probes, `csrr cycle` and `csrr instret`, had ALREADY SUCCEEDED in the same guest
command.** That is what makes the wrong conclusion so natural: the counters are demonstrably readable,
so the crash "cannot" be about counters, and the firmware `pc` points at the monitor.

**The cause is a U-mode read of an M-mode CSR.** The VFS clock in `capstone_sqlite_os.c` reads
**`mcycle`, CSR `0xB00`**. A capability *domain* may read it, because the monitor leaves it
domain-readable; **Linux userspace may not.** The `Scalar(0x2)` is **mcause 2, illegal instruction**.
With no libc there is no handler, so the process dies and the trap surfaces from M-mode.

**Discriminator, before suspecting the monitor: grep the binary for a machine-mode CSR.** The U-mode
mirrors are `cycle` (`0xC00`) and `instret` (`0xC02`).

**THE GENERAL SHAPE, which is the part worth carrying: the two halves of a paired measurement do NOT
run at the same privilege.** The capability arm is a domain and reads `mcycle`; the baseline arm is
ordinary userspace and must read `cycle`. Sharing a timing header between them is the obvious thing to
do and is wrong, and it fails in **the arm nobody suspects**, because the domain side is the exotic one.

**And the rule was ALREADY WRITTEN DOWN, in the right words, in the wrong place.**
`tests/rtl-smoke/ladder_base_ctl.c:17-22` states it exactly: the domain half reads `0xB00` which the
monitor leaves domain-readable, userspace cannot, the U-mode mirror is `0xC00`. It did not prevent
this, because the code that broke was written for a *domain* and only later compiled for userspace, so
nobody had any reason to open the ladder's control program. **This is a placement failure, not an
absence.** Recorded here, in the registry, because that is where someone who has just watched a
userspace binary die in M-mode will actually look; also in `fpga-debugging-recipes.md` §6.

**Corollary for board arms:** probe each counter in its **own** invocation. A gated CSR traps with no
handler, so only that invocation dies and the rest of the run still returns data — the "make every run
RETURN" rule applied to privilege rather than to control flow.

### C-32 — `MOVC` is emitted for an integer-bridged (untagged) pointer where a plain `mv` would do, and the RTL nulls its source (silently: MOVC raises nothing) `OPEN — LIVE ON SILICON 2026-09-15: the SQLite Sublet port at -O1/-O2 loses its lookaside to it (the block base is nulled by the movc that passes it, and re-read), so every optimised-image board number of that port is a lookaside-OFF run, and Q-04 hides it on every emulator pass; DESIGN A CHOSEN AND MERGED 2026-09-15 (46c53b7b6ae2, on dev at e3bb47b43680) AND MEASURED NOT TO FIX THIS SITE — the design choice is BACK WITH THE LEAD; still blocking P1's O2 arms; reproducer no longer an XFAIL, and a local reproducer of the surviving site is in capstone/tests/c32-sinkfold-repro/`

> **LEAD'S DECISION, 2026-09-25: "D′ now + prototype C".** Options were assembled by the compiler lane and
> adversarially audited before the decision.
> - **D′ (the route for P1): a port-only change.** `sqlite3MallocLinear` returns a `uptr`, and `pStart`
>   stays an integer through the `pStart = 0` merge. It is cast to a pointer only at call arguments and at
>   the store.
>   - This is sound because `setupLookaside` never dereferences `pStart`: every use needs an address and
>     none needs authority (`ports/sqlite/sublet/sublet-3530300.patch:418-472`).
>   - Handing out the linear capability from `pBlock` instead was REFUTED. It is a C-46-class hazard,
>     `sublet_carve` still needs the linear block, and a delinearised whole-pool alias would outlive the
>     per-slot revocation.
>   - **Unproven until measured:** whether GVN or CSE recombines the casts into one GPCR vreg at the PHI.
>   - **Pass criteria, fixed before the build:** `movc-cfg-scan` on a fresh cell ⑥ -O2 image shows no
>     `setupLookaside` site; **the board reads the SAME counters as that image's own emulator run**; and
>     `--stats` on silicon reads non-zero lookaside slots.
>   - **RETRACTED (2026-09-25): "its emulator counters equal arm C's (5568/37966/32565/37966/5401)".**
>     - Arm C (image `2b9e4d0d`, boot 2026-09-15) and E2's cell ⑥ (09-14) both predate `dac22bcaeca4`
>       (09-18), which rewrote the patch onto `capstone_cap_slot` and the runtime header. So they are
>       a different program, not a baseline for today's tree.
>     - Measured by the compiler lane on the rebuilt port: `split=5481 mrev=37884 delin=32575
>       revoke=37884 init=5309`, lookaside ON (25015). Close to arm C, not equal.
>     - D′ was verified against that image's own counters instead: bit-identical before and after,
>       both `setupLookaside` sites gone, and the two unrelated sites still present as the scanner
>       control.
> - **C (the class fix, prototype and cost only, no commitment):** keep a bridged integer in a GPR until
>   it is genuinely used as a capability. The audit found it is NOT "all four sites by construction":
>   - `main+0x3aabc` is not shown to be a bridged value;
>   - a bridged value stored with `stc` and reloaded with `ldc` is invisible to it;
>   - every capability use must re-bridge, or a shared GPCR vreg brings the PHI copy back.
> - **Ruled out:** A (partial sink-and-fold) reaches one site of four and changes an upstream pass's
>   contract. B (a register class) is structurally unworkable.
> - **Left as they are:** `renameResolveTrigger` and `main` are dormant under the P1 workload and are
>   present in the native cell ⑤ as well. C-32 stays OPEN as a class.
> - The paper condition in the measurements doc (§7s arm C) is re-worded to match.

> **2026-09-24: a second instance, in musl, found by measurement and out of reach of any compiler
> fix.** `CAPSTONE_MOVC_NULL_SCALAR=1` (Q-04) with `capstone/tests/runtime-qemu/movc-null-scalar/exposure.sh`
> runs the nightly and libc-test with MOVC keeping and zeroing an integer source. Across the corpus
> exactly one verdict changes, `iconv_open`, the same in two runs at the same pc. musl's
> `combine_to_from()` returns a conversion descriptor as an integer, `(void *)(f<<16 | t<<1 | 1)`.
> libc-test at -O1 keeps it in `s5` and passes it to `iconv` three times with `movc a0, s5`, and
> under the RTL's rule the second call gets `cd = 0` and faults at `iconv`'s first load (cause 24).
> The integer becomes a pointer inside `iconv_open`, so the caller sees only a returned pointer.
> No caller-side analysis can know it is an integer, and the design-A extension discussed below
> would not reach it either. The measurement and what it did not cover:
> `plans/2026-09-24-q04-movc-integer-source.md`.

> # ⚠ OBSERVED LIVE ON SILICON 2026-09-15 — in a production workload, silently, and it cost three board readings and a QEMU-vs-board hunt (§7s of the measurements doc).
>
> **The instance.** The SQLite Sublet port's `setupLookaside`, compiled at -O2 (image `c506694f9f6f6889`,
> function at 0x26518): the lookaside's 64 KiB linear block comes back from the inlined
> `sublet_take_linear` as an INTEGER base (`lcc s3, t0, 0x3` at 0x2677c — the port casts it with
> `(void*)base`), the pointer argument to `sqlite3MallocSize` is materialised by **`movc a0, s3`**
> (0x2679c), and `s3` is **re-read** afterwards for the function's `if (pStart)` test (`mv a0, s3; bnez
> a0` at 0x267f0-0x267f4) and for `a = (uptr)pStart` (0x26884). On the RTL the `movc` writes `cnull`
> into `s3` (`capstone_flu_unit.anvil:6-27`: any source that is not a tagged NONLIN capability is
> nulled; no exception is raised), so the test reads `pStart == 0` and the function takes its
> no-lookaside exit with the block already taken and never freed — the free on that path is
> `sqlite3_free(pStart)` through a second `movc a0, s3` (0x26a20/0x26a4c), which on the RTL frees a
> null, so the same nulling that disables the lookaside also leaks its block (the six unrevoked
> handles at the end of the run); on capstone-qemu the source survives (Q-04) and the lookaside is
> set up. No fault on either machine — this instance differs from
> the reproducer's fault only in what the nulled register is used for next.
>
> **How it read on the board.** The same image, same arguments, oracle hash on both machines; the
> port's own counters `split/mrev/delin/revoke/init` 8654/41293/32638/41287/8649 on silicon against
> 5568/37966/32565/37966/5401 on the emulator — decomposed, "one linear take, no slot carved, every
> small request served by memsys5"; then `--stats` read **zero lookaside slots** on the board against
> 25,010 hits on the emulator (boot sw8x-o2stats). The -O1 image diverges identically; the -O0 image
> does not (its int-to-pointer cast goes through memory: `stc` of the integer, `ld` of the cursor
> word, and no `movc` source outlives the copy). Bisection: twenty allocator functions at -O0 inside
> the -O2 image changed nothing (arm A); `setupLookaside` alone at -O0 (arm C, `2b9e4d0d3ed523c9`)
> is the pre-registered confirmation and is queued. Consequence for the corpus: sw80b, sw81, sw82 and
> arm A are lookaside-OFF Sublet runs and their ratios are not cited.
>
> **What it changes here.** The design choice above is no longer insurance against a latent case: the
> live-source read after a `movc` of an integer-bridged value is emitted by -O1 and -O2 for ordinary C
> (an `unsigned long` cast to `void*` and tested for null), and every QEMU pass of such code is blind
> to it. Until the bridge is fixed, an optimised image of any port that casts integers to pointers
> carries this risk on silicon, and its board result needs an emulator-independent check (here: the
> port's counters and `--stats`). **Related:** Q-04 (the emulator side; still the open spec question
> and NOT ruled here), C-46 (the machine model's blindness to the consumed source — its "only where a
> read of the source outlives the movc" bound is exactly this instance).

> **Design A was chosen, merged, and does not fix this site (2026-09-17).** The lead chose design A:
> `inttoptr` lowers to the rematerializable `PseudoBRIDGE_CAP` instead of a bare `INSERT_SUBREG`
> (`46c53b7b6ae2`, merged at `e3bb47b43680`). Scanned with `movc-cfg-scan.py` over the Sublet cell ⑥
> `-O2` image before (`c506694f9f6f6889`) and after (`113221f93b0ac994`): **the same four sites, same
> functions, same classifications**, offsets shifted a few bytes. It removed nothing on this workload.
> Not caused by the remat override removed in `46c53b7b6ae2` — rebuilt with it restored, the image is
> byte-identical.
>
> **Why, settled 2026-09-17.** Design A's protection is not rematerialisation at all. RA-side remat
> never runs for this pseudo: `TargetInstrInfo::isReallyTriviallyReMaterializable` refuses any
> instruction with a virtual-register use (*"Don't allow any virtual-register uses"*) and the pseudo
> has one, and Capstone's override (`CapstoneInstrInfo.cpp:244`) is the inherited RVV switch, which
> falls through to it. What removes the `movc` is `MachineSinking::PerformSinkAndFold`
> (`MachineSink.cpp:405`), pre-RA, which rewrites ISel's `$c10 = COPY %bridged` into
> `$c10 = PseudoBRIDGE_CAP %int` and so leaves no GPCR vreg for register allocation to copy. **That
> fold is all-or-nothing per def:** the first use that is not a copy chaining to a physreg of the same
> register class, and not a foldable load/store address, makes it decline for the whole def — so one
> such use leaves the other uses' copies as `movc` too. In `setupLookaside` that use is the **join
> PHI**. (The address-half read-backs — the `a = (uptr)pStart` this entry already describes, at
> `0x26888` in the post-fix image — are uses of the *joined* value, so they are not what declined the
> fold; they are what makes the nulling observable, by reading the source after the `movc` wrote
> `cnull` over it.)
>
> **So the live site is a bridged value reaching a merge — the root design A was accepted as not
> covering.** The lead's decision table (`plans/DECISIONS-WAITING-2026-09-10.md:559-561`) records
> design A as leaving "**PHI copies**, which remat cannot reach", and the live site is a PHI copy.
> What was not anticipated is that the documented gap is the live case. (The register-class
> alternative was set aside there on a structural objection of its own, not on design A being
> sufficient, so nothing measured here falsifies that objection.)
>
> **It is NOT what the lit test's `bridged_phi_residue` pins, and that is a hole in the regression
> net.** In `bridged_phi_residue` each bridge's *only* use is the PHI, so no conforming copy is lost
> to the decline and its `movc` are copies of the PHI *result*. The live site is a bridge with a
> conforming call-argument copy **and** a PHI use, which is where all-or-nothing bites.
> `setupLookaside` contains both kinds: `0x267a0` is the fold-declined one (the INT-ONLY site, the one
> the harm story runs through), while `0x26a28`/`0x26a54` are copies of the merged value and are the
> `bridged_phi_residue` kind. **A fix that removed the `0x267a0` copy would leave
> `bridged_phi_residue` still emitting its `movc` and the Capstone lit suite still green.**
> `capstone/tests/c32-sinkfold-repro/` shape 4 is currently the only thing guarding that shape.
>
> **The four sites do NOT share one cause** — an earlier statement that they did is retracted.
> `setupLookaside+0x26a28` is the same merge as the live site; `renameResolveTrigger+0x10b5b8` is the
> same *class* with its cause not established; and `main+0x3aabc` has **twelve** reaching definitions
> including three capability loads, so it is not shown to be a bridged value at all — which agrees
> with `docs/history/15-09-2026_02-40-00_c32-movc-scan-one-site-or-a-class.md:136`, "**Not
> PHI-shaped.**" The four-site table itself is correct and re-verified on the post-fix image; only its
> interpretation is corrected.
>
> **A defect in the scanner behind that table, found on the way and now fixed.** `movc-cfg-scan.py`
> read `jalr rs` as a DEF of `rs` (`defs_reads` returned operand 0), but the one-operand form is the
> pseudo for `jalr ra, rs, 0` — it defines `ra` and *reads* `rs`. So an indirect call terminated the
> backward walk on its own target register and injected a `cap` reaching definition that does not
> exist, making the reaching-def union a **lower bound**. It bit only when the `movc` source is
> **callee-saved**, because caller-saved registers are already short-circuited at the scan's `call`
> branch — which is why it survived every image anyone had run. It is a plausible cause of
> `0x26a28` reading MIXED when its two reaching defs are an integer and `cnull`. Fixed, and the
> scanner now has the positive control it never had: `capstone/tests/movc-cfg-scan-selftest.py`,
> negative-tested two-sided (the pre-fix scanner classifies the shape `cap` and the test fails; the
> fixed one classifies it INT-ONLY and it passes, while a genuine capability def stays unclassified
> in both). **Re-scanned after the fix: the table does not move.** Both scanner
> versions were run on the verified image (`113221f93b0ac994`, extracted from
> `xfer/c32-f1-2026-09-16` and hash-checked) and give byte-identical reports — same four sites, same
> classifications, same 2/2/1780 split. This is a *loaded* negative rather than a void one: the input
> contains what the fix acts on (**seven one-operand `jalr` in `setupLookaside` alone**), the two
> scanner versions differ as files, and the selftest passes in both directions on this host. The
> instrument changed and the answer did not.
>
> The operator's prediction that `0x26a28`'s MIXED was an artefact of the `jalr` defect was **wrong**,
> and the real reason was already in the record: its two reaching definitions are `0x2679c mv s3, a0`
> (integer) and `0x267c4 movc s3, zero` — the cnull arm of the branch at `0x26790`. `movc` is not an
> integer producer, so the `cap` half is genuine. The site the fix *does* touch is `main+0x3aabc`,
> where it removes a spurious `jalr s5` reaching definition; three real `ldc` capability loads remain
> among its twelve, so the classification and the conclusion drawn from it — not shown to be a bridged
> value at all — are unchanged, and are now resting only on genuine capability defs.
>
> Design A remains a real improvement for defs all of whose uses conform. Whether it can be *extended*
> is open: all-or-nothing is a property of upstream's current `PerformSinkAndFold`, not a law, and
> "fold the conforming uses, leave the non-conforming ones" would remove `0x267a0` while leaving
> `0x26a28`/`0x26a54`. Not proposed and not costed — it changes an upstream CodeGen pass's contract,
> not Capstone-local code — recorded only so the option is not dismissed by a sentence.
>
> Evidence and controls: `docs/history/17-09-2026_14-23-55_c32-design-a-sinkfold-mechanism.md`.
> **F1's confirming boot is held and should stay held:** the predicted reading is already determined —
> the four sites are unchanged and the mechanism says they must be — so a boot would report "lookaside
> silently off", which the scan established. A cycle that confirms what the evidence already implies
> is not worth board time.


**FILED 2026-09-10, LATE.** This ID has been live in a committed, tracked test —
`llvm/test/CodeGen/Capstone/c32-movc-untagged-live.ll` — with **no registry entry in either half of
the registry**. That is I-02's own failure mode and the **second** instance after C-25: an ID handed
out in a commit and never entered, so the allocator could have reissued it and no reader could look
it up. `tests/next-issue-id.sh` now reads committed content precisely to stop the next one.

**The defect.** `inttoptr` on a c128 lowers through a bare `getTargetInsertSubreg`
(`CapstoneISelLowering.cpp:8062-8064`) into an UNDEF capability carrying only an address, which is
what `inttoptr` means — the tag is deliberately clear. Register allocation may then copy that value
with `MOVC`, and the RTL faults on a `MOVC` of an untagged operand. **The spec sides with the RTL
here**, so QEMU is the permissive divergent side and **no QEMU run can observe this class** — which is
why it needs a codegen fix rather than an emulator one.

**Two designs, neither prototyped, and the choice is the lead's:**

* **A rematerializable bridge pseudo** replacing the bare `getTargetInsertSubreg` — carrying
  `isReMaterializable`/`isAsCheapAsAMove` and expanding to `ADDI` on the `sub_cap_addr` half, mirroring
  `PseudoTRUNC_CAP` (`CapstoneInstrInfo.td:2530-2552`). At `-O2` the bridge is already a single
  `mv`, so remat turns both `movc` into `mv`. **Residue it cannot remove: PHI copies.**
* **A register class distinguishing a bridged integer**, so `copyPhysReg` can choose `mv`. Weaker, and
  possibly unworkable: *untagged* is a property of the **value**, whereas a `RegisterClass` is a set of
  **physical registers**, and by the time `copyPhysReg` runs it sees only physical numbers.

Both touch the ABI of integer-bridged pointers, which is why neither is a lane's call. Analysis by the
compiler lane, 2026-09-10; entry placed by the board lane, whose path this file is.

> # ✅ SYNTHESISED 2026-09-10 — all three pre-registered falsifiers HELD. R-30 + R-31, `1bfff7776`.
>
> Built on the synth machine, exit 0, 1 h 21 m 54 s, peak 25.87 GB against a 100 GB ceiling. Scored
> against the prediction **as written**, not against a prediction adjusted afterwards:
>
> | | `1bfff7776` | `66c4e7517` | pre-registered | |
> |---|---:|---:|---|---|
> | WNS `clk_out1` | **−12.425** | −12.425 | −15.3 … −11.7 | PASS |
> | placed Slice LUTs | **169,207** (83.03 %) | 169,207 | 168.9k … 170.5k | PASS |
> | combinational loops | **29** | 29 | must not move | PASS |
> | TNS / failing endpoints | −718477.5 / 102508 | identical | — | |
> | registers | 93,131 | 93,131 | — | |
>
> **THE FIGURE THAT CARRIES THE INFORMATION IS THE BITSTREAM, NOT THE TIMING.** Same size to the byte,
> 11,443,722, and a **different** sha256 — `406e12bf…` against `b03bd967…`. Same structure, different
> bits, which is exactly what flipping two comparison operators does: LUT *initialisation contents*
> change while LUT and register counts do not. **Identical timing is the expected result here, not a
> suspicious one, and a MATCHING hash would have meant the build did not contain the change.**
>
> Every figure was compared field-for-field against the same report pulled from `66c4e7517`'s own
> artifact rather than against notes. `write_bitstream completed successfully` is present in the log
> rather than inferred from the exit code. Routed DRC: 20 rules, all Warning, no Error or Critical and
> no `LUTLP-1` — a negative that is supported, because the same extractor listed 20 real rule names
> from a populated table.
>
> **Provenance, stated because the stamp says "dirty":** `1bfff7776` plus exactly four locally modified
> files, all machine-path and guard fixes (`env.sh`, `env_cap.sh`, `fpga-env.sh`, `synth-guard.sh`).
> **No file under `core/`, `corev_apu/` or `verif/` was modified.**
>
> **WHAT THIS DOES NOT ESTABLISH: that either fix WORKS.** Every number says the change is
> timing-neutral and structurally invisible, and nothing more. The functional case rests on the
> directed tests and the one-variable control, not on this build.
>
> **~~THE FLASH IS NOT MERELY WAITING FOR A YES.~~ SUPERSEDED 2026-09-11 — BOTH DECISIONS ARE RULED
> AND NOTHING IS WAITING ON A JUDGEMENT.** This box said the flash was gated on two OPEN decisions:
> the `end`-convention re-ruling (decisions item 1, because the spec amendment ships in the same
> change) and the monitor reclaim shape (item 2, because the firmware half must land alongside).
> Both were ruled on **2026-09-10 (evening)** — `DECISIONS-WAITING-2026-09-10.md:50`, *"RULED
> 2026-09-10 (evening): adopt the resolution. Both spec checkouts edited"* — and the reclaim shape
> was implemented at monitor `0a5c3d9`, with the fill itself at `a006c63`. Left struck through
> rather than deleted, because a stale gate reads as a live one and two lanes carried this wrong
> state into 2026-09-11 from this paragraph.
>
> **The requirement it was protecting still stands and is now satisfied:** this must never ship
> RTL-only. The monitor being flashed (`2c49c41`) contains both reclaim commits as ancestors, and
> carries them by content — `RCLM`/`RCSH`/`RCPR` at `sbi_capstone.c:152-154`, the R-31 block at
> `:204-212`. That matters because `REV_TRANSFERRED`'s check is a hard `while(1)` on the FPGA
> target, so a monitor mismatch wedges rather than traps.
>
> **What remains is mechanical, and mostly not ours.** The timing row is read and scored — all three
> pre-registered falsifiers PASS (see the table above: WNS −12.425, placed LUTs 169,207,
> combinational loops 29 unmoved). The `.bit` is in-tree on the synth machine — not extracted, not
> staged, not flashed — and the console bitstream store cannot be written by our driver, so the
> upload is a GUI action for the board owner.
>
> **This bitstream is clean of R-24**, checked after R-24 was found to break exception delivery
> outright (`docs/history/11-09-2026_19-30-00_r24-debug-request-collision.md`): `r24-excode-base` is
> not an ancestor of `1bfff7776`, and `66c4e7517..1bfff7776` touches only the two anvil units and
> test files — no exception encoders.

### R-30 — `INIT` is UNREACHABLE on silicon: filling an UNINIT region leaves the cursor at `end`, and `INIT` faults unless the cursor is PAST `end`. The shortfall is exactly one byte, and it kills the whole reason the UNINIT type exists `HEADLINE SUPERSEDED, AND THE RESIDUAL IS NOW EXPLAINED AND RE-FILED. The one-byte precondition is FIXED and verified on silicon (boot sw61, 5,334 INITs on caplifive_r30r31_1bfff7776). The separate 1,728-byte shortfall this entry carried as open is NOT a fill failure at all: it is bounds re-encoding, demonstrated on silicon 2026-09-12 (boot sw62) and filed as **R-33**. Nothing about the fill remains open here`

> **2026-09-14 (sw74/sw74b):** the 1,419,584-byte reclaim that wedged sw60 with `RCSH:000006C0` now
> completes through csinit on the #3 module + monitor 4274268 (RCLM 0 → 1, no RCPR/RCSH/RCRE, twice).
> That shortfall was R-33's re-encoding, not this entry's; the audited account, the confound (the
> monitor moved too) and the owed control are under R-33.

> # ⚠ THE HEADLINE "INIT IS UNREACHABLE" IS CONTRADICTED BY SILICON, 2026-09-12. Two readings on ONE bitstream, in adjacent boots, and they do not agree.
>
> On `caplifive_r30r31_1bfff7776`:
>
> * **boot sw60** — the monitor's reclaim of a **1,419,584-byte** region reports
>   `RCSH:000006C0`: the fill stopped **1,728 bytes** (108 granules of 16) short of `end`, and halted
>   on its own designed `while(1)`. That is a shortfall, but it is **not the one-byte shortfall this
>   entry describes**.
> * **boot sw61** — the Sublet port on the same silicon reports
>   `sublet: split=5508 mrev=37899 delin=32565 revoke=37899 init=5334`. **5,334 successful INITs**,
>   every counter bit-identical to QEMU.
>
> **INIT is therefore plainly REACHABLE on this silicon, 5,334 times in one run.** Whatever sw60's
> 1,728-byte shortfall is, "INIT can never be satisfied" does not describe it, and this lane's own
> prediction — that R-30 would force `init = 0` and produce the masking signature — was written down
> before sw61 and was **wrong**.
>
> **NARROWED 2026-09-12 (RTL lane, `d1d85697d11b`, plus this lane's reading of the fill macro). The
> two boots are NOT in conflict about the FIX, and calling them one overstated it.** R-30's one-byte
> claim was about INIT's PRECONDITION: a filled UNINIT leaves the cursor AT `end`, and INIT demanded
> strictly greater, so it was unreachable by one position. The fix made equality legal — and sw61 is
> that fix working, 5,334 times. sw60 is a fill that never reached `end` at all, which is **one step
> earlier**; with the cursor genuinely short, INIT refusing is CORRECT behaviour, not the defect.
> **So what does not survive is this entry's WORDING, not the change.** sw60 is a second effect that
> appears only on a large region, and it needs its own account.
>
> **AND THE MONITOR'S OWN ARITHMETIC CANNOT PRODUCE 1,728.** `C_RECLAIM_FILL` (`sbi_capstone.c:259`)
> sets `n = (end - base) >> 4` and each `stc` advances the cursor one granule, so the largest
> shortfall the loop admits is `(end - base) mod 16`, i.e. **at most 15 bytes** — and `RCPR` not
> firing means the cursor provably started AT base, so that is measured, not assumed. 1,728 is 108
> granules and is unreachable from this source. **This also kills the allocator-rounding account
> before it costs a boot:** `n` is computed from the same `end - base`, so a region rounded UP scales
> `n` with it and the shortfall still cannot exceed 15.
>
> # ✅ CLOSED 2026-09-12 BY BOOT sw62: it was account (b), and the mechanism is R-33.
>
> Arms at granule-ALIGNED arenas (1,419,264 and 709,632) reclaimed **cleanly**, reclaim count 0 → 1 → 2.
> The unaligned arm (354,880) halted with `RCSH = 448` — the compression prediction, written down
> before the boot, against 432 for a proportional store-failure rate. `RCCU` shows the cursor reached
> the true end, so **every store advanced and none failed**; `RCEN` shows `end` reading
> `round_up(354,880, 512) = 355,328`. The shortfall is the rounding and nothing else.
>
> **So account (a) — "108 stores did not advance the cursor" — is REFUTED, and it was this lane's own
> surviving account.** The full mechanism, exposure and mitigation are in **R-33**; everything below
> this line is the trail that led there and is kept for the two retractions it records, not as live
> analysis.

> # ⚠ CORRECTION 2026-09-12 (same day, later): ACCOUNT (b) IS REINSTATED. Both refutations of it — the RTL lane's and this lane's endorsement of it — were made at the WRONG LAYER.
>
> **What was written, and why it is wrong.** (b) "`end` moved during the fill" was struck on the
> grounds that *"STC's UNINIT path advances the cursor by exactly 16 and passes the metadata carrying
> `end` through unchanged"*, and a bounds-compression fit was separately struck because *"capabilities
> here are fat, with full 64-bit start and end fields, so there is nothing to round"*. **The second
> claim is contradicted by the RTL it cites.** `ariane_pkg.sv:786` is
> `function automatic bounds_t compress_bounds(input fat_bounds_t bounds, …)` — it takes the fat
> bounds and *returns a compressed encoding*. Capabilities are fat **inside the pipeline** and
> compressed **on writeback**, which is exactly the layer the first claim does not reach. The two
> arguments and the hypothesis were never in contact.
>
> **The compressed encoding rounds, and it predicts 1,728 exactly.** Read in full at
> `ariane_pkg.sv:786-845`, and note the naming trap: `len = bounds.base - bounds.start` (`:807`), so
> in this function **`start` is the LOW bound and `base` is the HIGH one** — the opposite of the
> monitor's vocabulary. There are two schemes:
>
> * `if (bounds.start == cursor)` (`:787`) — the cursorless form, taken while the cursor is still at
>   the low bound. No round-up.
> * otherwise (`:806`+) — `E = leading_zeros - 12` (`:817`), and then
>   `if(((bounds.base >> (E+3))<<(E+3)) != bounds.base) T[11:3] += 1` (`:827-828`): **the TOP is
>   rounded UP to a `2^(E+3)` granule.** (The `E==0 && len[12]==0` sub-case at `:818-821` is exact, so
>   the effect cannot appear on small regions at all.)
>
> For a 1,419,584-byte region the highest set bit of `len` is 20, so `E = 8` and the granule is
> **2,048**; the top rounds to 1,421,312, i.e. **1,728 bytes higher — `0x6C0`, sw60's `RCSH` to the
> byte.** The round-up is applied to the ABSOLUTE top address, and sw60's `BASE:AC100000` is
> 2048-aligned, so the length arithmetic is the right arithmetic here.
>
> **The timeline fits `C_RECLAIM` exactly.** `n = (end - base) >> 4` is computed BEFORE the fill,
> while the cursor is still at the low bound and the cursorless form is in force, giving 88,724. The
> first `stc` moves the cursor off base; the writeback re-encodes in the rounding form and `end`
> decodes 1,728 higher. The post-loop `lcc(%1, %2, 4)` then reads `end - cursor = 1,728`. **On this
> account every store succeeded and nothing failed to advance — `end` moved.**
>
> **What is NOT established, stated plainly.** This is an exact numerical match, not a proof. The LSU
> writeback path that would re-compress `stc`'s cursor-advanced `rs1` has not been located in the
> RTL; `ex_stage.sv:1176-1190` shows `compress_cap` on the FLU and dyn writeback paths, not the store
> path. And at N = 1,419,584, `round_up(N, 2048)` and `round_up(N, 4096)` are the same number, so
> sw60 alone cannot separate the granule from a page-alignment story.
>
> **Accounts as they now stand.** (a) 108 stores did not advance the cursor — an ISA question;
> **(b) `end` moved during the fill — REINSTATED, and now the leading account**; (c) an instrument
> fault in the reclaim path. Kept from the earlier read because it is independently useful: STC's
> bound check puts the last legal store at `end - 16`, whose advance lands the cursor exactly ON
> `end`, so a fill of `(end - base) >> 4` stores is exactly right rather than lucky.
>
> **Boot sw62 settles it, and its predictions are pre-registered here.** A granule-ALIGNED arena needs
> no round-up, so compression predicts NO shortfall where any store-failure model still predicts one:
>
> | arena | granule | compression predicts | store-failure predicts |
> |---:|---:|---:|---:|
> | 1,419,264 | 2,048 | **0 — no halt** | ~1,728 — halt |
> | 709,632 | 1,024 | **0 — no halt** | ~864 — halt |
> | 354,880 | 512 | **448** | **432** |
>
> `RCEN` is the load-bearing reading, not `RCSH`: it reports `end - base` directly, which is the
> disputed quantity itself.
>
> **The instrument that separates the surviving accounts is now in the monitor and unbooted** — `RCEN`
> (`end - base`, the capability's true size) and `RCCU` (`cursor - base`, where the fill stopped)
> are reported alongside `RCSH`, inside the branch that already halts, so nothing measurable is
> perturbed. `RCEN - RCCU` must equal `RCSH`, which makes the reading self-checking. Pre-registered:
> `RCEN == 1,419,584` ⇒ 88,724 stores attempted; `RCEN == 1,421,312` ⇒ 88,832 attempted; either way
> 108 did not advance. `RCEN - RCCU != RCSH` ⇒ account (c).
>
> **The instrument was verified against the LINKED FIRMWARE before being believed, and it needed to
> be.** `C_RECLAIM` declares the capability input-only (`"r"(cap)`) while the hardware advances its
> cursor, so by the constraints alone the post-template reads are stale — `RCCU` would report 0 and
> `RCEN` the pre-fill end, and the self-check would fire while pointing at the wrong account.
> capstone-c's linear discipline is what saves it: a capability is written back to its home slot
> after every use. Confirmed by decoding `fw_payload.elf` at **all five** reclaim sites — each stores
> the post-fill capability home and reloads it from that exact slot before the reads. The same pass
> found a latent early-clobber hazard (either output could have been allocated over the still-live
> capability) which had not bitten but was resting on allocation; `"=&r"` on both, rebuilt, and
> re-verified byte-identical in allocation. Evidence: §4g.3, §4g.4, §4g.5, and
> `history/12-09-2026_R30-sw60-sw61-reconciliation-attempt.md`.

> # ⚠ RETRACTION 2026-09-10 (RTL lane's auditor): R-30's FIX IS A DELIBERATE SPEC DEVIATION, NOT A CONFORMANCE FIX. This is now a DECISION in front of the lead, not a correction.
>
> `cap-man-insn.adoc:421` lists `x[rs1].cursor <= x[rs1].end` as INIT's illegal-operand condition.
> **The PRE-FIX RTL was therefore SPEC-CONFORMANT.** Changing `flu:139` to `<` makes the RTL deviate,
> and `:421` must be amended in the same change or the two disagree in the other direction.
>
> **The "three independent arguments for exclusive" recorded above collapse to ONE.** The spec is
> INCLUSIVE wherever it actually speaks — `prog-model.adoc:119` defines aliasing over a closed interval
> and `:289` bounds fetch at `end-3` — and the RTL is split against itself: STC/LDC use `end-16` and
> SPLIT faults on `cursor >= end`, but SEAL computes `end - start + 1` and SHRINKTO sets
> `rd_end = inc - 1`, both inclusive. What survives is that **QEMU is exclusive throughout and the whole
> software stack is validated against QEMU.** That is a real argument and it may well be decisive, but
> it is one reason, not three, and it makes this a choice about which convention the project adopts
> rather than a defect being corrected.
>
> **AND Q-07 NARROWS RATHER THAN CLOSES.** I recorded that fixing R-30 this way would close Q-07 and
> called it the best reason to do it. That is withdrawn: post-fix the RTL accepts `cursor >= end` while
> QEMU asserts `cursor == end`, so the accepted sets still differ *above* `end`. The divergence shrinks;
> it does not disappear.
>
> **What still stands, unchanged:** the arithmetic (a full fill leaves the cursor at `end`), that INIT
> is therefore unreachable by filling, and that this makes the UNINIT type's purpose unachievable. What
> changes is that the remedy is a spec amendment plus an RTL change, decided together.

> # ⚠ AUDIT 2026-09-10: the CONCLUSION SURVIVES, the ENTRY DOES NOT. Three defects, all corrected below; read this box before citing anything in this entry.
>
> **1. PROVENANCE — the header's "reading the FLASHED RTL (66c4e7517)" is WRONG for most citations.**
> The `capstone-ariane` checkout is on `62b09ca92`, and `66c4e7517` **is not an ancestor of it**
> (`merge-base --is-ancestor` → NO). The `capstone_dyn_unit.anvil` cites are safe (that file is
> identical between the two), but the others are working-tree line numbers. Corrected against
> `git show 66c4e7517:`:
>
> | entry said | actually, at 66c4e7517 |
> |---|---|
> | `load_store_unit.sv:994-996` | `:975-977` (`:993-996` is a `tval` assignment) |
> | `lsu:1004` | `:985` |
> | `flu:227` / `:231-232` / `:237` (SHRINKTO) | `:238` / `:242-243` / `:248` |
> | `flu:163` (SEAL) | `:170` |
> | `flu:204-209` (SHRINK clamps) | `:210-217` |
> | `init-rs1-ne-rd.S:29-32` and its `MKCAP(a5, CAP_TYPE_UNINIT, 512, 496)` | **does not exist at 66c4e7517** — that file fabricates the operand as MKCAP LIN + `CINCOFFSETIMM` 272 + in-place `CAPTYPE` at `:43-47`, and contains neither the quoted sentence nor that line |
>
> **2. STEP (1)'s MECHANISM IS REFUTED, and the correct reason is stronger.** The entry said scalar
> stores cannot advance the cursor *because the LSU traps them*. That gate
> (`load_store_unit.sv:975-977`) is conditioned on `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`
> (`:947-950`) and was **measured INERT in our domains** on silicon (7 probes, 2026-08-04; a store
> through a base with no capability metadata, which trips the block's first clause, did not trap). So
> the trap is not what closes the route. **What closes it is that a scalar store has no
> capability-writeback path at all:** `scoreboard.sv:240-247` sets `cap_result` only from writeback
> ports 0 (FLU) and 4 (DYN) and `'0` for every other port, `commit_stage.sv:325` gates the capability
> regfile write on `cap_result.valid`, and `store_unit.sv:200` emits capability metadata only for `STC`
> or a domain switch. A `sd` cannot write a capability register regardless of the check.
> Also: the entry's phrase "type-3 capability" is wrong on this RTL — **type 3 is REVOKE**; UNINIT is 4
> (`capstone_unit.anvilh:278-286`). The SV uses symbolic names so the code is right; the prose was not.
> Same class as the sw39 loss.
>
> **3. THE PREDICTED OBSERVABLE IS WRONG AND WOULD VOID THE DIRECTED TEST.** The entry predicts
> "exception 29". `ILLEGAL_OPERAND_VALUE` is enum index 6 (`capstone_unit.anvilh`, whose own comment
> reads `ILLEGAL_OPERAND_VALUE, // mcause 30`), and the execute-path encoders compute `24 + code`
> (`ex_stage.sv:472` for FLU, `cva6.sv:1443` for DYN). INIT is an FLU op, so the delivered **mcause is
> 30, not 29** — which is exactly **R-24**, open in this same file. A test asserting 29 would report a
> miss and read as REFUTING R-30.
>
> **Three prose overstatements, softened:**
> * *"Neither spec states whether `end` is inclusive or exclusive"* is too strong — `prog-model.adoc:118`
>   defines aliasing over the **closed** interval `[base, end]`, and SEAL uses `end - start + 1`, so the
>   spec leans INCLUSIVE. The arithmetic still fails under both readings, but **"shortfall exactly one
>   byte" is true only under the EXCLUSIVE reading**; under inclusive it is 16.
> * *"SHRINK clamps the cursor down, never up"* — `flu:210-213` clamps a below-start cursor **up**. The
>   conclusion holds (it cannot exceed the new end, `:216-217`, `:203`), but the cited range showed only
>   the half that supported the claim.
> * *"Any `imm <= 0` passes"* for SHRINKTO — only `imm == 0`. A negative immediate wraps under the
>   unsigned compare at `flu:243` and faults.
>
> **One route the entry closed by accident rather than by argument, now explicit:** `CINCOFFSETIMM` on a
> LINEAR capability is **unbounded** (`flu:66-70`, no bounds check), so a cursor CAN legally be placed
> past `end` while the capability is LINEAR — the flashed test file does exactly that. The route to INIT
> then needs a LINEAR→UNINIT retype, and the only architectural producer of UNINIT is REVOKE, which
> resets `cursor := start` (`dyn:68`); the retype in that test is the Custom3 `CAPTYPE` **debug** op. So
> the route is closed, but by the absence of an architectural retype, not by the cursor rules.
>
> **A route the entry never considered, tested and closed by the auditor.** The register file stores
> COMPRESSED metadata (`ex_stage.sv:1215` compresses on writeback, `issue_read_operands.sv:1151` +
> `ex_stage.sv:784` decompress), so every STC advance re-round-trips the bounds and a shrunken `end`
> would make INIT reachable. A model of `compress_bounds`/`decompress_bounds` swept over 400,000 random
> `(base, size)` with `cursor == end`: `end` **never shrinks** (0 cases), so INIT still faults. Route
> closed. *Caveat: a Python model of the source, not an RTL simulation.*
> **Incidental and UNRESOLVED:** ~30 % of those cases **widened** the region on an ordinary register
> writeback (start rounds down, end rounds up) — e.g. `[0x80000010, 0x80010010)` reading back as
> `[0x80000000, 0x80010080)`. Whether that is intended compressed-capability rounding or a defect was
> not checked against design intent. **Consequence for any directed test here: read `end` back with
> `LCC` rather than assuming the requested value, or the predicted out-of-bounds point will be wrong.**

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
> # ⚠ THE RULING WAS MADE ON A FRAMING THAT IS WRONG IN BOTH DIRECTIONS. Back to the lead before anything is edited. (2026-09-10, later)
>
> The question was put as "is `end` inclusive or exclusive, project-wide". It is not one question, and
> neither of the two routes debated is correct:
>
> * **The RTL's `end` is EXCLUSIVE**, confirmed twice at `66c4e7517`: the LSU faults when
>   `lsu_ea_full + lsu_access_sz > bound_end` (`:985`), so the last valid byte is `bound_end - 1`; and
>   SPLIT sets `rs1.end := val` with `rd.start := val` (`dyn:141-142`), which partitions without overlap
>   only if `end` is exclusive.
> * **The spec's `end` is INCLUSIVE**, equally clearly: `prog-model.adoc:119` closes the interval
>   `[c.base, c.end]`, `ctrl-status-insn.adoc:79-80` sets `end = INIT_*_END - 1`, and SHRINKTO
>   (`cap-man-insn.adoc:269`) sets `end = cursor + imm - 1`.
>
> **Each document has exactly ONE instruction using the other's arithmetic, and they are DIFFERENT
> instructions.** That is why this looked like a convention dispute:
>
> | | convention | the odd one out | fix |
> |---|---|---|---|
> | **spec** | inclusive | the STORE BOUND, `[base, end - CLENBYTES]` (`mem-access-insn.adoc:93`) — exclusive arithmetic | `end - CLENBYTES + 1` |
> | **RTL** | exclusive | `INIT`'s check, `cursor <= end` (`flu:139`) — inclusive arithmetic | `<=` → `<` |
>
> **Fix one on each side, change no conventions, and the two agree.** For bytes `S..S+63` the spec says
> `end = S+63` and the RTL says `end = S+64`; both permit stores at `S, S+16, S+32, S+48`; both leave the
> cursor at `S+64`; both then accept `INIT`. QEMU is exclusive throughout and already accepts that.
>
> **AND ONE OF THE PROPOSED ROUTES IS A SECURITY REGRESSION.** Changing the RTL's STC bound to
> `end - 15` — correct for the spec's inclusive `end` — permits, under the RTL's exclusive `end`, a
> 16-byte store at `end - 15` whose last byte is at `end`: **one byte past the region, on the store
> path.** It must not be applied to the RTL. It is the right fix for the SPEC and only for the spec.
>
> **Nothing is edited until the lead rules again.** The previous ruling stands recorded below because it
> was made in good faith on the framing available, and the framing was mine.
>
> **INDEPENDENTLY VERIFIED by the RTL lane** (2026-09-10), who checked all three claims against the
> source rather than accepting them, and worked the equivalence through arithmetically. They withdrew
> their own route 2, in their words, because they *"derived the RTL's convention from the spec's instead
> of reading it"* — the same class of error being caught in others' work all night. No spec edit has been
> made by either lane.
>
> **AND ONE CONSEQUENCE IS NOW SETTLED THAT WAS PREVIOUSLY LEFT AS "NARROWS": whether Q-07 CLOSES
> DEPENDS ON A CHOICE IN THE QEMU FIX, NOT ON A FACT.**
> * If QEMU's `csinit` keeps `cursor == end`, it accepts exactly one value while the fixed RTL accepts
>   `end` and above, so the sets still differ above `end` — **narrows**, which is what was recorded.
> * If QEMU's `csinit` takes `cursor >= end` — which is what the Q-07 change written on 2026-09-10 does
>   (`op_helper.c`: fault iff `cursor < end`) — the two accepted sets are **identical** and **Q-07
>   closes**.
>
> Both implementations hold `end` exclusively, so `>= end` is the coherent choice and is what is
> written. The earlier retraction of "fixing R-30 closes Q-07" was correct against the `== end`
> proposal and is superseded by the `>=` one; recorded this way so neither statement reads as simply
> wrong.

> # ~~LEAD'S RULING 2026-09-10: `end` IS EXCLUSIVE~~ — SUPERSEDED, see the box above.
>
> The recommendation was put with its one surviving argument stated plainly — every RTL access path and
> all of QEMU already assume exclusive, while the spec leans inclusive wherever it speaks — and with the
> fact that this is a DEVIATION rather than a conformance fix. The lead ruled for exclusive.
>
> **What that authorises, in order:** amend `cap-man-insn.adoc:421` to match (`cursor < end` illegal,
> i.e. accept `cursor >= end`) in the SAME change as `capstone_flu_unit.anvil:139`; keep the revoke
> polarity fix with it; land the firmware change alongside, never RTL-only; then synthesis, then the
> lead's flash decision. The `SEAL` and `SHRINKTO` inclusive stragglers are reconciled separately and do
> not hold this up.
>
> **What it does NOT authorise:** describing R-31's fix as closing the disclosure. It restores the type
> discipline on the capability-grained route; the scalar route stays open because its check is gated on
> a privilege level domains never run at.

> # ALL FOUR GATES PASS, 2026-09-10. Branch `r30-r31-init-revoke`. What remains is a DECISION, not a gate.
>
> | gate | result |
> |---|---|
> | **lint** | PASS — UNOPTFLAT **40**, every count at baseline |
> | **functional** | both arms PASS on the fixed tree and **FAIL 11 on the same tree with only the two operators reverted** — a true one-variable pair, rebuilt without the R-29 candidate that confounded the first control |
> | **auditor** | R-31's polarity SUPPORTED against four attacks; R-30's code change defensible; **two of the author's own claim sentences refuted and corrected in the source** |
> | **sweep** | **88 of 88** runnable tests PASS, 0 failures, 0 hangs (`records/r3031/sweep-table.txt`) |
>
> Three of the 91 listed tests did not build — `rand`, `perf_tcache_random`, `perf_riscv_random` —
> because their listed sources are **absent from the tree entirely**. Generator entries with no
> checked-in source, so they would not build on any revision. Stated rather than assumed; a control run
> on those three would make it airtight and was judged unnecessary against the missing-file evidence.
>
> **TWO THINGS THAT ARE NOT GATES AND MUST TRAVEL WITH THIS TO THE LEAD:**
> 1. **R-30 is a spec DEVIATION, not a conformance fix.** The spec text must be amended in the same
>    change, which needs a ruling on whether `end` is inclusive or exclusive — not a lane's to make.
> 2. **It must NOT ship RTL-only.** The monitor traps on **five** sites once R-31 lands, and R-31
>    restores the type discipline **without closing the disclosure**, because the scalar-path check is
>    gated on a privilege level domains never run at. Firmware ships with it or neither ships.
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
> `revoke_region`'s own two sites (`:1455`, `:1460`) are safe — they store the result back without **[2026-09-11: safe for the RESULT type, which is what this box asked. The OPERAND type at those same two lines is M-6.]**
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
`capstone-academic-spec/parts/mem-access-insn.adoc:93` bounds the store at `[base, end - CLENBYTES]` and
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

### R-31 — REVOKE's permission clause is INVERTED against the spec, so revoking a linear borrow of a WRITABLE region returns a readable LINEAR capability instead of an UNINIT one — the reinitialisation step is skipped and the borrower's data is disclosed to the owner `FIXED ON SILICON — VERIFIED 2026-09-12 on caplifive_r30r31_1bfff7776, boot sw60, through the monitor's real share/revoke path`

> # ✅ FIXED AND VERIFIED ON SILICON, 2026-09-12, boot sw60.
>
> The bitstream `caplifive_r30r31_1bfff7776` was flashed this session (`nv_bitstream_sha256`
> `406e12bf…` read back from the board after the mandatory power-cycle). A host probe revoked a
> `REV_BORROWED` region and then **shared it again**, which is what reaches the monitor's reclaim —
> the guard at `sbi_capstone.c:1309` sits on the SHARE path, so an arm that revokes only at teardown
> can never trigger it. Verbatim:
>
>     RGID:00000014  AREV:00000001      region 20, REV_BORROWED
>     SHA2:00000003                     cap_type(r) = 3 = UNINIT
>     BASE:AC100000  ALEN:0015A940
>
> `SHA2` is `cap_type(r)`, emitted at `:1276` **before** the guard. It reads **UNINIT**, where the
> previous bitstream returned LINEAR — that is exactly the inversion this entry describes, and it is
> gone. **`RCPR` did not fire either**, so `cap_cursor == cap_base`: both halves of the contract hold.
>
> Corroborated in simulation with a matched negative control: `r31-revoke-cursor` FAILS at
> `66c4e7517` (tohost=11) and PASSES at `1bfff7776` (490 cycles), identical test binary, RTL the only
> variable. Details in `fpga-silicon-measurements-for-paper.md` §4g.3 and §4g.4.


> # AUDIT 2026-09-10: R-31 is SUPPORTED on all four attacks — and it is ALSO NOT SUFFICIENT. Both halves matter.
>
> **The reading that could have collapsed it does not.** `\<=p` is defined BY EXTENSION at
> `capstone-academic-spec/parts/prog-model.adoc:103-113`, and the only pairs with `2` on the left are
> `(2,2), (2,3), (2,6), (2,7)` — exactly the set with bit 1 set. `:94-95` names the encoding
> ("`2` = write-only … `6` = read-write"). So `2 \<=p perms` **is** `(perms & 2) == 2`: subset ordering
> on a 3-bit mask. Cross-checked for consistency: `mem-access-insn.adoc:36` uses `4 \<=p perms` for a
> load and `:46` uses `2 \<=p perms` for a store, which the RTL implements as `perm&4` (`dyn:338`) and
> `perm&2` (`dyn:401`). Same operator, same reading, both directions. **Both spec copies are identical**
> here, so there is no "the RTL implements the other spec" escape.
>
> **`rsp == 1` confirmed** as "no invalidated capability was linear": the flag is set on `rev_req`
> (`capstone_rev_node.anvil:156`), cleared when an invalidated node had `linear == 1` (`:24-25`), and
> returned at `:19`; nodes are born linear and cleared only by `delin`.
>
> **The disclosure consequence confirmed via a check this entry did not make.** The obvious refutation
> would be that `rev_req` also invalidates the REQUESTING node, which would make the returned LINEAR
> fault `INVALID_CAPABILITY` and disclose nothing. It does not: the walk starts at `node_in.next`
> (`:155`), bounds on the requester's own depth (`:154`) and terminates at `:18`, so the requester's node
> is never visited. `modify_cap_type` preserves perms and the revnode id, and REVOKE touches no memory.
>
> **An attack that nearly succeeded.** `asm_insn.h:13` and `CapstoneInstrInfo.td:2670-2675` both encode
> REVOKE with `rd = x0`, which would send the retyped capability to the discard register. Refuted at
> `decoder.sv:1185` (`instruction_o.rd[4:0] = instr.rtype.rs1` — the encoded rd is overridden) and
> `ariane_regfile_ff.sv:63-67, :87-94` (when both ports target one index the rd port wins).
>
> **QEMU has NO permission clause at all** (`op_helper.c:907-908`), so the RTL differs from the spec
> *and* from the emulator. That removes the "the spec sentence is probably just wrong" escape.
>
> **Premise confirmed:** the monitor's retained revocation handle really does carry write for RW regions
> — `rev = __mrev(r)` is minted (`sbi_capstone.c:1180`, `:1198`) BEFORE the `__tighten` calls that narrow
> the shared copy, MREV preserves perms, and `__tighten` can only succeed if the bits are already held.
>
> # ⚠ BUT FIXING R-31 ALONE DOES NOT CLOSE THE DISCLOSURE, and the entry implied that it would.
>
> UNINIT blocks `LDC` (`dyn:332`), but a **scalar `ld`** is type-checked only in
> `load_store_unit.sv:975-977`, and that block is gated on `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`
> (`:947-950`). **That gate was measured INERT in our domains** — the 2026-08-04 seven-probe silicon
> result recorded as `project_lsu_cap_check_inert`, where a store through a base carrying no capability
> metadata (tripping the block's first clause) did not trap. The same gate is present at `66c4e7517` and
> **has not been re-measured since**. If it is still inert, an UNINIT capability in a domain is readable
> by plain `ld` anyway, so the reinitialisation step R-31 restores is bypassable by a different route.
>
> **Consequence: R-31 is necessary and not sufficient**, and any claim that its fix "closes the
> disclosure" is unsupported until that gate is re-measured on the current bitstream.
>
> **2026-09-10 — and the reason is STRUCTURAL, not a bug in the check.** The gate requires
> `ld_st_priv_lvl_i == riscv::PRIV_LVL_M` (`load_store_unit.sv:947-950` at `66c4e7517`), i.e. it only
> applies to loads and stores issued in **M-mode**. Domains do not run in M-mode: the monitor's own
> trace markers say so in as many words — `SHA5` is *"about to leave M-mode for the domain"* and `ENT1`
> is *"about to leave M-mode INTO the domain"* (`sbi_capstone.c:126`, `:129`). So the only capability-type
> check on the SCALAR load/store path is inapplicable to domain code **by construction**.
>
> *(2026-09-15: half of the explanation — the other half is **R-34**, the LSU's exceptions being raised and dropped at every privilege; the conclusion below stands.)*
> That explains the 2026-08-04 "inert" measurement completely, and it reclassifies it: not a check that
> fails to fire, but a check that was never in scope for the code we run. **It also makes R-31's
> insufficiency structural rather than contingent** — an UNINIT capability held by a domain is readable
> with an ordinary `ld` no matter what REVOKE hands back, so the reinitialisation property cannot be
> restored by the RTL revoke fix alone. Closing it needs the scalar path type-checked outside M-mode,
> which is a design question well beyond R-31.
>
> **BOARD READING sw51 (2026-09-10): the probe ENTERED AND WEDGED — which CONTRADICTS the prediction
> above, and the run does NOT say why.** Predicted `0x310003A5` (a return, on the structural argument);
> observed no return. The domain was created and entered (`DBAS`, `DENT`, `SHA0`-`SHA6`, `ENT0`/`ENT1`,
> region 14 transferred, and the driver's own classifier says created-and-entered with no monitor tag),
> so this is a genuine in-domain wedge and not an entry stall.
>
> **But it does NOT establish that the scalar load trapped, and the parser that said it did was wrong.**
> `WEDGE_TRACER` was 0 for this boot, so there is no latched `mcause`/`mepc`. The probe executes CAPTYPE,
> then a spill/reload of the retyped capability, then `LCC`, then the scalar `ld`. A wedge at ANY of
> those looks identical from outside. My boot script printed *"the scalar load TRAPPED -> the LSU check
> is LIVE"* from the absence of a result — the exact over-claim this investigation has been cataloguing
> all night, committed by my own parser.
>
> **sw51 ATTEMPT 2 (tracer on): the wedge is in the ENTRY GLUE, not at the measurement. The run carries
> NO verdict, and the "evidence against" reading below is withdrawn.** Latched `mcause` **27**, latched
> `mepc` **0x819e00b0**, with this domain's `DBAS` at `0x819e0000` — so image offset **0xb0**, which
> disassembles to **`delin gp` in the entry glue**:
>
> ```
> d00ac: <unknown>
> d00b0: delin gp        <- the fault
> d00b4: addi  t1, t1, -0x10
> ```
>
> The probe's scalar load is at offset **0x334**. The domain entered, ran about 0xb0 bytes of startup
> glue, and faulted deriving its own `gp` — **before `domain_main` and therefore before CAPTYPE, the
> type read-back, or the measurement.** So sw51 says nothing whatever about the LSU check, in either
> direction, and the earlier "a wedge is evidence against the structural argument" line is withdrawn:
> a wedge that happens before the experiment is not evidence about the experiment.
>
> (Cause 27 here is in the ambiguous band — see **R-24** — but `delin` is an execute-path op, so it
> reads as `UNEXPECTED_CAP_TYPE`. A domain faulting in its own entry glue is an observation worth its
> own look; it is NOT attributed here, and it may be peculiar to this probe: the build reports
> *"3 global(s) (0 initialized)"* where the R-25 probes at adjacent entry VAs report 3 of 3 initialised,
> and `k800` ran to its oracle in the same boot.)
>
> **THIS PROBE HAS NOW FAILED TWICE, FOR TWO DIFFERENT REASONS, WITHOUT EVER MEASURING ANYTHING** —
> first staged under a host that never delivers the region share, then wedging in the glue. It is
> parked rather than re-run a third time: two boots have been spent and the question is better answered
> by the RTL lane's simulation, where the privilege level and the fault site are both observable
> directly, than by another draw at it on the board.
>
> # SETTLED FROM SOURCE 2026-09-10 (RTL lane, verified here): R-31 IS NECESSARY BUT NOT SUFFICIENT. This is a READING, not an open question.
>
> The capability check on ordinary loads and stores is gated on
> `CVA6Cfg.CAPSTONE_EXT && capmode_i && ld_st_priv_lvl_i == riscv::PRIV_LVL_M`
> (`load_store_unit.sv:947-950`). **Domains do not run in M-mode**, and the arithmetic is direct:
> `create_domain` sets the domain-switch context to `dom_seal[3] = (3 << 38) | (2 << 34)` =
> `0xC800000000`, whose bits **[12:11] — mstatus MPP — are ZERO**, so the `mret` into a domain lands in
> **USER mode**. The monitor says the same from the other side at `sbi_capstone.c:1838-1842`:
> *"MPP=3 -> the address IS a firmware address … MPP<3 -> it is a guest virtual address"*.
>
> **So the check CANNOT FIRE for any domain access.** A revoked region handed back as UNINIT remains
> readable by a plain `ld` inside the domain, and the reinitialisation discipline R-31 restores is
> bypassable by anything that reads with scalar loads instead of `LDC`. This matches the finding this
> project already recorded — and retracted a bounds claim over — that plain integer loads and stores are
> unchecked in our domains, filed then as a gate-or-configuration question rather than a defect.
>
> **What R-31's fix should therefore be sold as:** it restores the TYPE DISCIPLINE on the
> capability-grained route. It does **not** by itself close the disclosure. Anyone describing it as
> closing the disclosure is wrong, and that is now a statement about the source rather than a pending
> measurement.
>
> **A TRAP FOR WHOEVER TRIES TO MEASURE THIS, worth more than the finding.** A bare-metal directed test
> runs in **M-mode**, where the gate DOES fire — so such a test would report *"the check works, UNINIT is
> not readable"*, which is true of the test and false of every domain. It would have been an eighth
> instance of tonight's pattern, and I proposed exactly that route before it was caught. **A real
> measurement has to reach U or S mode**, which means a code CPMP entry for the low-privilege code; the
> one existing S-mode capability test is parked precisely because it lacks that and spins on an
> instruction-access fault. That is a day of test work, not a board slot, and it should wait for the
> `end`-convention ruling because **the answer does not change what R-30 and R-31 do.** The discriminator is one re-run with
> `WEDGE_TRACER=1`, reading the latched `mcause` against the LSU's OWN cause table
> (`load_store_unit.sv:972-990` at `66c4e7517`), which emits **raw** mcause values rather than going
> through the `24 + enum` execute-path encoder:
>
> | latched mcause | meaning |
> |---|---|
> | **26** | the type check fired — not LINEAR/NONLIN. **This is the discriminator**: the check IS live in a domain and the M-mode reading above is wrong. |
> | 24 | the operand was NOT_CAP — the capability was lost before the load, so the probe never held UNINIT |
> | 27 | a permission fault, not a type fault |
> | 28 | out of bounds — the retype corrupted the reconstructed bounds, not a type result |
> | anything else, or a wedge with no latch | the probe wedged before its measurement |
>
> **Note my first draft of this table said 27.** The LSU raises `26` for a wrong type and reserves `27`
> for permissions; I had applied the FLU/DYN `24 + enum` convention to a unit that does not use it —
> the same numbering-system confusion, a fourth time, caught before the reading rather than after.
> (That the LSU and the execute path use different conventions is itself part of **R-24**.) **Until that runs, R-31's sufficiency is UNRESOLVED in both
> directions** and neither the "inert by construction" nor the "check is live" claim should be cited.

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
> # ⚠ THE RULING WAS MADE ON A FRAMING THAT IS WRONG IN BOTH DIRECTIONS. Back to the lead before anything is edited. (2026-09-10, later)
>
> The question was put as "is `end` inclusive or exclusive, project-wide". It is not one question, and
> neither of the two routes debated is correct:
>
> * **The RTL's `end` is EXCLUSIVE**, confirmed twice at `66c4e7517`: the LSU faults when
>   `lsu_ea_full + lsu_access_sz > bound_end` (`:985`), so the last valid byte is `bound_end - 1`; and
>   SPLIT sets `rs1.end := val` with `rd.start := val` (`dyn:141-142`), which partitions without overlap
>   only if `end` is exclusive.
> * **The spec's `end` is INCLUSIVE**, equally clearly: `prog-model.adoc:119` closes the interval
>   `[c.base, c.end]`, `ctrl-status-insn.adoc:79-80` sets `end = INIT_*_END - 1`, and SHRINKTO
>   (`cap-man-insn.adoc:269`) sets `end = cursor + imm - 1`.
>
> **Each document has exactly ONE instruction using the other's arithmetic, and they are DIFFERENT
> instructions.** That is why this looked like a convention dispute:
>
> | | convention | the odd one out | fix |
> |---|---|---|---|
> | **spec** | inclusive | the STORE BOUND, `[base, end - CLENBYTES]` (`mem-access-insn.adoc:93`) — exclusive arithmetic | `end - CLENBYTES + 1` |
> | **RTL** | exclusive | `INIT`'s check, `cursor <= end` (`flu:139`) — inclusive arithmetic | `<=` → `<` |
>
> **Fix one on each side, change no conventions, and the two agree.** For bytes `S..S+63` the spec says
> `end = S+63` and the RTL says `end = S+64`; both permit stores at `S, S+16, S+32, S+48`; both leave the
> cursor at `S+64`; both then accept `INIT`. QEMU is exclusive throughout and already accepts that.
>
> **AND ONE OF THE PROPOSED ROUTES IS A SECURITY REGRESSION.** Changing the RTL's STC bound to
> `end - 15` — correct for the spec's inclusive `end` — permits, under the RTL's exclusive `end`, a
> 16-byte store at `end - 15` whose last byte is at `end`: **one byte past the region, on the store
> path.** It must not be applied to the RTL. It is the right fix for the SPEC and only for the spec.
>
> **Nothing is edited until the lead rules again.** The previous ruling stands recorded below because it
> was made in good faith on the framing available, and the framing was mine.
>
> **INDEPENDENTLY VERIFIED by the RTL lane** (2026-09-10), who checked all three claims against the
> source rather than accepting them, and worked the equivalence through arithmetically. They withdrew
> their own route 2, in their words, because they *"derived the RTL's convention from the spec's instead
> of reading it"* — the same class of error being caught in others' work all night. No spec edit has been
> made by either lane.
>
> **AND ONE CONSEQUENCE IS NOW SETTLED THAT WAS PREVIOUSLY LEFT AS "NARROWS": whether Q-07 CLOSES
> DEPENDS ON A CHOICE IN THE QEMU FIX, NOT ON A FACT.**
> * If QEMU's `csinit` keeps `cursor == end`, it accepts exactly one value while the fixed RTL accepts
>   `end` and above, so the sets still differ above `end` — **narrows**, which is what was recorded.
> * If QEMU's `csinit` takes `cursor >= end` — which is what the Q-07 change written on 2026-09-10 does
>   (`op_helper.c`: fault iff `cursor < end`) — the two accepted sets are **identical** and **Q-07
>   closes**.
>
> Both implementations hold `end` exclusively, so `>= end` is the coherent choice and is what is
> written. The earlier retraction of "fixing R-30 closes Q-07" was correct against the `== end`
> proposal and is superseded by the `>=` one; recorded this way so neither statement reads as simply
> wrong.

> # ~~LEAD'S RULING 2026-09-10: `end` IS EXCLUSIVE~~ — SUPERSEDED, see the box above.
>
> The recommendation was put with its one surviving argument stated plainly — every RTL access path and
> all of QEMU already assume exclusive, while the spec leans inclusive wherever it speaks — and with the
> fact that this is a DEVIATION rather than a conformance fix. The lead ruled for exclusive.
>
> **What that authorises, in order:** amend `cap-man-insn.adoc:421` to match (`cursor < end` illegal,
> i.e. accept `cursor >= end`) in the SAME change as `capstone_flu_unit.anvil:139`; keep the revoke
> polarity fix with it; land the firmware change alongside, never RTL-only; then synthesis, then the
> lead's flash decision. The `SEAL` and `SHRINKTO` inclusive stragglers are reconciled separately and do
> not hold this up.
>
> **What it does NOT authorise:** describing R-31's fix as closing the disclosure. It restores the type
> discipline on the capability-grained route; the scalar route stays open because its check is gated on
> a privilege level domains never run at.

> # ALL FOUR GATES PASS, 2026-09-10. Branch `r30-r31-init-revoke`. What remains is a DECISION, not a gate.
>
> | gate | result |
> |---|---|
> | **lint** | PASS — UNOPTFLAT **40**, every count at baseline |
> | **functional** | both arms PASS on the fixed tree and **FAIL 11 on the same tree with only the two operators reverted** — a true one-variable pair, rebuilt without the R-29 candidate that confounded the first control |
> | **auditor** | R-31's polarity SUPPORTED against four attacks; R-30's code change defensible; **two of the author's own claim sentences refuted and corrected in the source** |
> | **sweep** | **88 of 88** runnable tests PASS, 0 failures, 0 hangs (`records/r3031/sweep-table.txt`) |
>
> Three of the 91 listed tests did not build — `rand`, `perf_tcache_random`, `perf_riscv_random` —
> because their listed sources are **absent from the tree entirely**. Generator entries with no
> checked-in source, so they would not build on any revision. Stated rather than assumed; a control run
> on those three would make it airtight and was judged unnecessary against the missing-file evidence.
>
> **TWO THINGS THAT ARE NOT GATES AND MUST TRAVEL WITH THIS TO THE LEAD:**
> 1. **R-30 is a spec DEVIATION, not a conformance fix.** The spec text must be amended in the same
>    change, which needs a ruling on whether `end` is inclusive or exclusive — not a lane's to make.
> 2. **It must NOT ship RTL-only.** The monitor traps on **five** sites once R-31 lands, and R-31
>    restores the type discipline **without closing the disclosure**, because the scalar-path check is
>    gated on a privilege level domains never run at. Firmware ships with it or neither ships.
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
> `revoke_region`'s own two sites (`:1455`, `:1460`) are safe — they store the result back without **[2026-09-11: safe for the RESULT type, which is what this box asked. The OPERAND type at those same two lines is M-6.]**
> inspecting the type. That grep is not proof of completeness and an auditor has been asked to look for
> a third consumer.

**The spec** (`capstone-academic-spec/parts/cap-man-insn.adoc:610-617`) sets `x[rs1].type` to LINEAR if EITHER

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


**SUFFICIENCY SETTLED FROM SOURCE 2026-09-10 — the fix is NECESSARY but does NOT close the
disclosure, and no board arm is needed to say so.** The fix makes revoke return UNINIT, whose purpose
is to carry no read authority. But at the flashed revision `66c4e7517` **nothing traps a plain scalar
`ld` through an UNINIT capability in a domain.** An `rtl-oracle` sweep of every `UNINIT` occurrence at
that revision (24 non-doc hits) places all of them in one of four places, none reachable from a scalar
load:

* the **M-gated** `cap_violation_detection` block, `load_store_unit.sv:947-977`, whose UNINIT clause
  (`:975`, cause 26) sits under `ld_st_priv_lvl_i == riscv::PRIV_LVL_M`;
* **LDC-only** paths — `load_unit.sv:214-218` gated at `:231` on `operation == LDC`;
  `commit_stage.sv:331-340` gated on `fu == CAPSTONE_DYN && op == LDC`; `capstone_unit.anvilh:583-609`,
  reachable only from `capstone_dyn_unit.anvil:495`;
* **DYN/FLU-unit instructions only** (`capstone_dyn_unit.anvil`, `capstone_flu_unit.anvil`) — a
  `LOAD`-typed instruction is dispatched at `issue_read_operands.sv:1282-1284` and never reaches the
  `default:` arm at `:1291-1312` that feeds those units;
* **simulation-only**, `ex_stage.sv:808-837`, inside `` `ifndef SYNTHESIS ``.

The sweep was positive-controlled against its own blind spot: `CAP_TYPE_UNINIT` is `3'b100`
(`ariane_pkg.sv:655`), so a raw bit-slice test would be invisible to a name grep. Searching
`[30:28]` and `3'b100` finds the known sim-only assert and nothing else.

**Three of the sweep's citations were re-read at `66c4e7517` in this session rather than taken on the
subagent's word** — the CPMP block and its `cpmp_allow` term at `pmp_data_if.sv:286-294` and
`:124-134` (the novel claim), and the LDC gate at `load_unit.sv:231`. All three match verbatim.

**THE PRIVILEGE RESIDUAL IS CLOSED BY MEASUREMENT, NOT BY DERIVATION.** The source argument reduces
to "the M-gated block cannot fire in a domain", which needs the domain to run at `priv_lvl != M`;
the RTL alone does not settle that, because `priv_lvl_q` moves only on trap entry or `xRET`
(`csr_regfile.sv:1048,2155,2318,2341,2362,2376`) and the domain switcher writes `mstatus` without
touching it (`capstone_dom_switcher.anvil` has no `priv_lvl`/`mpp` occurrence at all). The
**2026-08-04 seven-probe measurement settles it directly and better**: a store through a base
carrying no capability metadata trips that same block's FIRST clause (`:972`, cause 24) and **did not
trap in a domain**. The UNINIT clause is in the same block under the same gate, so it does not fire
either. That is a measurement of the block in question rather than an inference about privilege.

**That argument has one premise, and it was checked rather than assumed.** It holds only if the
M-gate existed on the RTL that measurement ran against — otherwise the block was inert for some other
reason that need not carry forward. `caplifive_fixed_forward.bit` was built from `capstone-ariane
7aac52f93` (ISSUES-ARCHIVE.md:708), and at that revision the gate is present at
`load_store_unit.sv:934` with the NOT_CAP clause at `:958` and the UNINIT clause at `:961` — the same
block, the same gate, the same structure as at `66c4e7517`. The premise holds and the residual is
closed outright rather than conditionally.

**A LIVE, NON-M-GATED capability check on every domain load and store DOES exist, and it is not this
one.** CPMP, `pmp/src/pmp_data_if.sv:286-294`, fires exactly when `ld_st_priv_lvl_i != PRIV_LVL_M`
and is instantiated on the scalar path at `load_store_unit.sv:476-506`. It checks a **separate
16-entry CSR-mapped window table** (`:124-134`), never the load's own `rs1`, and its gating term is
`cap_type != NOT_CAP` — so an UNINIT window entry grants access exactly like a LINEAR one. It cannot
close this disclosure and is, if anything, permissive toward UNINIT.

**The `lsugate` board probe is RETIRED, not repaired.** It was built to measure this and wedged twice
in its entry glue. The question is now answered from primary sources plus an existing measurement, so
spending a third boot would confirm what the evidence already implies — the S-12 recorder mistake.
If it is ever revived, the two live mechanisms are distinguishable by signature: cause 26 with a
VIRTUAL `tval` (`load_store_unit.sv:993`) means the M-gated block fired and the domain was not in
U-mode; cause 5 with a PHYSICAL `tval` (`pmp_data_if.sv:296`) means CPMP rejected the address for
want of window coverage, which is a monitor CPMP-setup question and not a type check at all.

> # ✅ FIXED IN FIRMWARE 2026-09-12 (unbooted). The allocator now rounds, and the reclaim's postcondition no longer conflates the two failures.
>
> **The representability fix is in the KERNEL, not the monitor** — `ioctl_create_region`
> (`modcapstone/module/capstone.c`), `caplifive-buildroot` **`8da1559`**. It rounds the requested
> length up to `2^(bit_length(len)-10)` before `PAGE_ALIGN`, passes the rounded value to the monitor
> and stores it in the mirrored bookkeeping, so the allocation, the ecall and `dma_free_pages` agree
> on one number. **Rounding in the monitor would have been wrong:** `dma_alloc_pages` has already run
> with `PAGE_ALIGN` of the *unrounded* request, so a monitor-side round-up carves pages the kernel
> never handed out — silently aliasing a later allocation that falls inside the over-carve, or
> wedging on `SPLA` if one straddles it. Rounded requests are **logged**, not silent.
>
> **The rounding reserves exactly what the hardware widens to**, which is the property that makes it
> a fix rather than a mitigation: 354,880 → 355,328 and 1,419,584 → 1,421,312 are the same numbers
> `RCEN` measured. Verified standalone against those cases and an exhaustive 64-aligned sweep to
> 8 MiB (zero non-representable results, none shrinking), with every power-of-two and round-MiB size
> left untouched as the negative control.
>
> **The base half is NOT fixed and cannot be fixed here.** A region is representable only if its base
> is also a granule multiple, and the base comes from `CONFIG_CMA_ALIGNMENT` — which satisfies this
> for every region under 512 MiB on the current config, by accident of configuration rather than by
> any invariant. A misaligned base now warns.
>
> **The monitor's reclaim postcondition is split into two checks** (`capstone-sbi` **`4274268`**).
> The fill check compares the cursor against `base + 16n` — cursor reads only, so it is immune to the
> re-encoding — and keeps `RCSH` for its true meaning. A new `RCRE` tag then tests INIT's
> precondition explicitly, because dropping the `end` read alone would just let `csinit` trap
> `ILLEGAL_OPERAND_VALUE` inside the monitor, which is the uninformative fault the postcondition
> exists to prevent. On sw60's case the fill check now passes and `RCRE` reports 1,728 as the
> widening it is.
>
> **⚠ TWO BUILD HAZARDS THIS TURNED UP, both of which produce a clean `rc=0` and a stale artifact.**
> (1) **There are THREE copies of `modcapstone/module/capstone.c`**, and the FPGA image builds from
> the `caplifive-system` one via `BR2_EXTERNAL_CAPSTONE_PATH` — not the `caplifive-buildroot` one.
> Editing the wrong copy rebuilt a freshly-timestamped `.ko` without the change. The two are kept
> byte-identical; `caplifive-system-dev`'s copy is a third and differs. (2) **The board drivers'
> bake loop is `for a in linux-rebuild opensbi-rebuild`, which never rebuilds the module** —
> `modcapstone-rebuild` must run FIRST, before the pass that rolls the cpio, or the boot ships a
> stale `capstone.ko`. Any driver that needs this fix must add it. **Check the built `.ko` by
> CONTENT** (`strings` for the new message, with an existing message as the control), never by
> exit status or timestamp.
>
> **NEITHER IS BOOTED.** The resident firmware predates all of it.

> **2026-09-14 (E1, boots sw78 r1b2/r1b4/r1b5): the domain-side reading the block above said only a
> U/S-mode probe could give.** `ngx_uaf.c` stop 3 on the Sublet arm (image `103998ef04d6c342`, entry
> 0x310000) — the object touched after `ngx_destroy_pool` has revoked its pool — RETURNED on the FPGA
> with mark C30000: the byte load retired and read 0x00 (the plain arm's stop 3 reads its own 0xA0,
> `9d55128c0eb91875` → C300A0; the Sublet arm's stop 1 reads 0xA0 before the destroy). The emulator
> faults the same image at the same `lbu` (pc image+0x66cc) with cause 24 NOT_CAP, because its `ldc`
> untagged the reloaded pointer (Q-11 above). The 0x00 is the port's own `stc zero` fill
> (`sublet.h:138-142`, run when the revoke hands back UNINIT), not hardware. Whether the reloaded base
> was TAGGED on the board — the difference between "the LSU's node-validity clause
> (`load_store_unit.sv`, cause 25) never applies below M-mode" and "the base was an untagged integer, the
> 2026-08-04 NOT_CAP result re-observed" — was read by stops 10/11 (r1b4/r1b5, 2026-09-15 00:0x): **TAGGED, type 1** — s10 `CA0180` (the
> emulator: `CA0780`), s11 `CB0100` (type 1 and the byte 0x00 from the same run). So the load retired
> through a tagged capability under a revoked node: the LSU's node-validity clause does not apply below
> M-mode, measured from inside a domain. And s5 on the protected arm read the NEW occupant's byte
> (`C5005B`, r1b5) where the emulator faults.
> Audited before entry (claim-auditor, 2026-09-14): the hash-cited image, the fault pc on QEMU, the
> destroy→revoke chain (`ngx_destroy_pool → ngx_pool_slot_put → ngx_subpool_release → sublet_give_to →
> revoke`, unconditional), the `--arena-linear` guard (a non-linear arena marks FD00xx, not C3xxxx).
> N = 1 per platform; the boots' S-07 self-test reported itself unproven (irrelevant to the marks).

### R-33 — the region allocator hands out capabilities whose size is NOT REPRESENTABLE in the compressed bounds encoding, so moving the cursor WIDENS a capability's authority past its own allocation by up to one granule less a byte `OPEN — the widening is MEASURED on silicon 2026-09-12 (boot sw62, RCEN = round_up(N, granule) on caplifive_r30r31_1bfff7776) and STC's bound check is READ FROM SOURCE to consume that end; CONFIRMED 2026-09-12 to reach ORDINARY LINEAR capabilities through CINCOFFSET -- i.e. plain pointer arithmetic, not just the reclaim -- by a matched RTL-sim pair on the flashed hash; the resulting over-permissive store is **DEMONSTRATED** 2026-09-12 in RTL simulation at the flashed revision (`r33-store-past-end.S`): a representable control's store at its true end is refused OUT_OF_BOUNDS while a non-representable arm's identical store RETIRES WITHOUT FAULT. Contained by the kernel's PAGE_ALIGN below 4 MiB and NOT contained at or above it. Cause is the allocator, not the encoder; fix is to round region sizes to the granule at creation`

> # 2026-09-15 — R-33 CANNOT REACH THE PAPER'S SAFETY TABLE, and the reason is arithmetic
>
> Asked because a demonstrated over-permissive store is the one defect here that could **invalidate**
> a published claim rather than add one. The paper's `tab:safety` has exactly two cells asserting
> anything about extent — `appendices/c-validation-and-accounting.tex:41` "Inner free preserves live
> sibling & Sibling intact, block extent unchanged" and `:42` "Sibling survives uncooperative child",
> with `:66-69` making the extent claim explicit. A block rounding far enough for a store to land in
> an adjacent sibling would be "block extent unchanged" being false while the discipline reports
> success.
>
> **The siblings ARE adjacent** — that was checked first and came back the unhelpful way. The fixture
> is `capstone/ports/nginx/port/ngx_subpool_test.c`, phases 12-13, and it is deliberate:
> *"The sibling is taken FIRST and from the same arena, so that it is a neighbour of the nest rather
> than something allocated after the dust settled"* (`:359-361`). Phase 12 carves two 1024-byte
> children from one 4096-byte block with a 64-byte object in each.
>
> **Representability closes it, and the reason is STRUCTURAL rather than a property of the sizes this
> fixture happens to use.** (First stated as an enumeration of the fixture's sizes; corrected
> 2026-09-15 to the stronger form, which survives a fixture change — an enumeration would have to be
> redone every time the fixture moves.)
>
> The allocator rounds **every** request: `ports/nginx/port/ngx_subpool.c:86`
> `bytes = (bytes + 15) & ~(size_t) 15;`, with the comment explaining why — an odd carve leaves the
> arena 8-aligned and "EVERY later block is misaligned and the first capability stored in one is an
> unaligned access". The fixture asserts it directly at phase 8: a 4280-byte request yields extent
> **4288**, *"rounded, not as asked"*.
>
> So the quantity that matters is the **granted extent, not the requested size**; every granted extent
> is a multiple of 16; and for every extent below 16384 the granule is 8 or 16, both of which divide
> 16 (checked exhaustively: zero counterexamples). **Widening therefore cannot occur at these sizes
> for ANY request whatsoever**, not merely the ones this fixture chooses.
>
> **The margin is 3.8×, not the 1024× first recorded here.** That figure was the threshold at which
> the granule exceeds *page* alignment, which is a different boundary and the wrong one for this
> question. What governs is where the granule stops dividing the allocator's own 16-byte rounding:
>
> | | |
> |---|---|
> | first request whose granted extent is not a multiple of its granule | **16385** → extent 16400, granule 32, remainder 16 |
> | fixture's largest granted extent | 4288 |
> | headroom | **3.8×** |
>
> The conclusion is unchanged and 3.8× is real headroom, but the distinction matters for what may be
> said next to it: **an nginx port allocating a 16 KiB buffer sits at that boundary.** "Not a near miss
> a fixture change could tip" was supported by the wrong number; the reason to stop worrying is the
> structural argument above, not the size of the gap.
>
> **The emulator-versus-silicon question is moot here**, which is worth recording because it was the
> proposed discriminator: the precondition is a property of the sizes the fixture chooses, not of the
> platform, so it fails on both and this does not widen the emulator/silicon gap that the safety-table
> decision turns on.
>
> **R-32 drops off the same path by the same argument** — its exposure was those same two extent
> cells, and an off-by-one at a boundary cannot falsify a cell whose regions are exactly representable.
> Both remain open as soundness defects; neither threatens a claim. **This closes the SIZE half only.**
> The bases here are `sublet_base(&x) + <8-aligned offset>` and so are 8-aligned if the root arena base
> is — true of any capability-bearing arena, but that is reasoning rather than a measurement.

> **R-11 IS THE SAME CONTRACT, AND THIS FIX CLOSES IT TOO (added 2026-09-15).** R-11 is
> `compress_bounds`' OTHER branch — the cursorless one, losing an unaligned TOP past its window —
> and it is open only because nothing we ship is large enough to trigger it. Rounding region sizes
> up to the representability granule at creation makes tops granule-aligned at any size, which is
> exactly the condition R-11 needs. The two entries' containment edges agree: the granule reaches
> 8192 B at 4 MiB, so 4 KiB `PAGE_ALIGN` stops covering it there, which is this entry's stated
> 4 MiB edge derived independently from R-11's granule arithmetic. Do not work R-11 separately.

> # ⚠ RE-SCOPED 2026-09-12 (later, RTL lane `12eb7c5d21dc` + this lane's containment analysis): this is CAPABILITY SOUNDNESS, not instrumentation — and the cause is the ALLOCATOR, not the encoder.
>
> **The rounded `end` is not merely *reported* high; it is the authority bound.** `STC`'s check is
> `rs1_up > rs1.metadata.end - 16` (`capstone_dyn_unit.anvil`, flashed revision), and `rs1.metadata`
> is decompressed from the register file (`ex_stage.sv:802`). So once the first store re-encodes the
> bounds, every later decode yields the rounded end and the hardware **permits stores into it**. On
> sw62's arm 4 that authorises writes up to `base + 355,312`, whose last byte lands at
> `base + 355,328` — **448 bytes past the 354,880 the allocator reserved.**
>
> **And it is not confined to UNINIT fills — that is now DEMONSTRATED, not inferred.** `CINCOFFSET`
> moves the cursor on an ordinary LINEAR capability (`new_cursor = rs1.cursor + val`) and its result
> goes through the same FLU writeback and the same `compress_cap`. The RTL lane built the directed
> arm this entry asked for (`verif/tests/custom/capstone/r33-cincoffset-widen.S`, run on the flashed
> `1bfff7776`, 435 cycles against a 200,000 timeout with no exceptions — a real completion, not a
> SUCCESS-at-timeout):
>
> | arm | length | representable? | `End` with cursor at base | `End` after a 16-byte `CINCOFFSET` |
> |---|---:|---|---|---|
> | **R** (control) | 16,416 | yes (`% 32 == 0`) | `80008020` | `80008020` — **unchanged** |
> | **W** | 16,400 | **no** (`% 32 == 16`) | `80008010` | `80008020` — **widened by 16** |
>
> **Matched on everything that matters:** same base, same permissions, same 16-byte cursor move, and
> the same granule — both lengths have their highest set bit at 14, so `E = 2` and the granule is 32
> for both arms. Only representability differs, and the predicted widenings from the granule law
> (0 and 16) match the observations exactly.
>
> Two properties make that reading safe rather than suggestive. **The control did not move**, so this
> is representability-specific and not "CINCOFFSET always re-rounds". And **arm W's first print shows
> the exact top `80008010`**, proving the capability really was in the exact-encoded state before the
> move — so the widening is caused by the cursor moving, not pre-existing.
>
> **This is what changes R-33's class.** The scope is not the reclaim routine and not the UNINIT type:
> it is every capability whose cursor leaves its base, which is ordinary pointer arithmetic on
> ordinary linear capabilities. That is an ISA-level defect rather than a monitor-adjacent one.
>
> **Which is why the defect is the ALLOCATOR and not the encoder.** A lossy compressed-bounds format
> is standard for this class of design, and it is EXACT for a *representable* object — one whose
> size and base align to the granule. The contract that allocators hand out representable objects is
> nowhere stated or enforced here: `create_region(N)` passes `N` straight through to the monitor's
> `__split`, so a non-representable size is silently widened. **sw62 is the experiment for that
> reading and it already ran:** both granule-aligned arenas were clean and only the unaligned one
> failed. **Fix: round region sizes up to the representability granule at creation**, and reserve
> that much. The two-check postcondition then remains as what catches a breach of the contract.
>
> **CONTAINMENT, and it is size-dependent — this bounds how bad it is today.** The kernel allocates
> `PAGE_ALIGN(len)` (`module/capstone.c:232`), so the widened authority may still land inside the
> page allocation. The granule is `2^(bit_length(N)-10)`, which exceeds a page exactly when
> **N ≥ 4 MiB**:
>
> | region | granule | widened by | page slack | past the allocation |
> |---:|---:|---:|---:|---:|
> | 354,880 (sw62 arm 4) | 512 | 448 | 1,472 | none |
> | 1,419,584 (sw60) | 2,048 | 1,728 | 1,728 | **none, with zero margin** |
> | 4,194,368 | 8,192 | 8,128 | 4,032 | **4,096** |
> | 8,388,672 | 16,384 | 16,320 | 4,032 | **12,288** |
> | 136,314,944 | 262,144 | 262,080 | 4,032 | **258,048** |
>
> So **no region this project has actually used escapes its allocation** — sw60's sits exactly on the
> boundary, and the 130 MiB region of sw55 happens to be granule-aligned. The exposure is for future
> large non-aligned allocations, and a 130 MiB-class region has a 262,144-byte granule.
>
> **THE CONTAINMENT ABOVE IS ACCIDENTAL, AND IT EXPIRES ON APPROVED WORK.** (RTL lane
> `3f587843527e`; arithmetic re-derived here.) Nothing has escaped so far because every region used
> has been either small or a **round multiple of a MiB** — and at these magnitudes a round MiB count
> is granule-aligned *for free*, so it widens by exactly **zero**. 2, 64, 120, 130 and 256 MiB all
> widen by 0. That is luck of the units, not a check.
>
> **The next approved step removes that luck.** `current-next-step.md:245` records that
> `main --size 100` "needs a **measured** 120 MiB" arena — and a measured value has no reason to be
> a multiple of its granule. In the 64-aligned candidate band just below 120 MiB (granule 131,072,
> page slack at most 4,095):
>
> * **96.8 %** of candidate sizes widen past their page allocation;
> * the worst within ~256 KiB of 120 MiB escapes by **126,976 bytes — 31 pages**.
>
> The arena is sized by measurement precisely to be as tight as possible, which is the worst case
> for this. **So if the arena is re-measured to a non-round value, the representability fix belongs
> before it is used.** It is also cheapest there: rounding the request up costs at most one granule
> less a byte, under 0.1 % at these sizes, and makes a measured value safe by construction rather
> than by inspection.
>
> > **⚠ SCOPED DOWN 2026-09-12 (bench-lane audit). An earlier version of this paragraph said the fix
> > MUST land before the size-100 run, and that is not supported.** The artifacts actually built for
> > that run are **128 MiB = 2²⁷** plus two 64 KiB regions — all powers of two, and a power of two is
> > representable at ANY granule, so every delivered region widens by **zero**. R-33 therefore does
> > not gate the set in hand. The gating claim came from reading `current-next-step.md:245`'s
> > "measured 120 MiB" as the artifact size when the delivered build is 128 MiB; the 120 MiB figure
> > is the measured *need*, recorded on a branch this document could not see. The hazard is real and
> > is confined to a **re-measured, non-round** arena. Whether to land the fix first anyway is the
> > lead's call, not a consequence of this entry.
>
> *Graded below: this paragraph is DERIVED — arithmetic over the granule law — not measured.*
>
> **THE OVER-PERMISSIVE STORE IS NOW DEMONSTRATED (2026-09-12), and it was the last derived step.**
> `verif/tests/custom/capstone/r33-store-past-end.S`, run at the flashed `1bfff7776`, 512 cycles
> against a 200,000 timeout — a real completion, not a SUCCESS-at-timeout. Each arm moves its cursor
> to its OWN TRUE END and attempts a 16-byte `STC` there, which writes entirely outside the region.
> The two arms share granule 32 and granule-aligned bases, and their regions are **disjoint**, so an
> over-write lands in memory no capability owns:
>
> | arm | region | len mod 32 | decoded `End` after the move | store at its true end |
> |---|---|---:|---|---|
> | **A** (control) | `[0x80004000, 0x80008020)` | 0 — representable | `0x80008020` *unchanged* | **`OUT_OF_BOUNDS`** |
> | **B** | `[0x80010000, 0x80014010)` | 16 — **not** representable | `0x80014020` *widened* | **RETIRED, no fault** |
>
> `trap_mask = 0x1` — exactly the value pre-registered in the test header before it ran; bit 0 set is
> the control refusing, bit 1 clear is arm B being permitted. The trace carries **one** exception in
> the whole run, `OUT_OF_BOUNDS` at cycle 365 on `rs1 = 18` (arm A), with the monitor's own STC report
> naming the operands. Arm B's `STC` at `0x800000bc` appears in the retired-instruction stream with
> no exception after it.
>
> **Stated precisely: what is shown is that the store is ACCEPTED, not that the bytes were observed
> landing.** The RVFI tracer does not render 128-bit capability stores as `mem` lines — neither arm's
> `STC` produces one, and the only three `mem` lines in the run are the 32-bit `sw`s to `.data`. A
> retired `STC` that raises nothing has passed the authority check, which is the question R-33 asks;
> the byte-level write was not separately observed.
>
> **The control is what makes this readable.** Had both arms trapped, the widening would confer no
> authority and R-33's soundness framing would be refuted; had both been permitted, the test would be
> measuring something other than representability. Artifacts (test, RVFI log, sim log, testlist entry)
> committed as **`eab5b196b`** on `r30-r31-init-revoke` (tag `backup/r33-store-past-end-2026-09-12`),
> beside its sibling `r33-cincoffset-widen.S` (`cb2cd046a`); test sha256 `19840b83b5d88318`, matching
> the handover byte for byte. **Independently re-run by the RTL lane before landing** — same 512
> cycles, same single `OUT_OF_BOUNDS` on the control at cycle 365, same acceptance on arm B — so the
> result has two runs by two lanes, not one.
>
> **What is measured, read, derived and still NOT shown — kept separate deliberately, and the
> boundary has moved twice.** MEASURED on silicon: `RCEN` = `round_up(N, granule)`, i.e. the
> decompressed end really is the widened value. READ FROM SOURCE: `STC`'s bound check consumes
> `rs1.metadata.end`. **DEMONSTRATED in RTL simulation (was DERIVED until 2026-09-12):** the widening
> reaches ordinary LINEAR capabilities through `CINCOFFSET`, and a store past the region's true end is
> **accepted** — both by matched pairs with representable controls. DERIVED, and still only derived:
> the containment table, which is arithmetic over the granule law.
> *(2026-09-15, from the RTL lane's third arm at `2c59a355b`: the encoding widens at the BOTTOM too — a base that is not
> granule-aligned reads low once the cursor moves, the mirror of the top's round-up. The containment table above measured
> the TOP only; the kernel's page rounding rounds the SIZE up, not the start down, so it covers none of the bottom. Whether
> any allocator hands out a non-granule-aligned base is not established — every region tested so far was base-aligned,
> which is why it went unnoticed.)*
>
> **STILL NOT SHOWN, and worth naming precisely rather than letting the entry read as closed:** (i) an
> over-permissive store **on silicon** — both demonstrations are RTL simulation, on the flashed
> revision but not on the board; (ii) the **bytes** of such a store observed landing outside the
> region, since the tracer does not render capability stores as memory lines; (iii) the **bottom**
> truncation (`ariane_pkg.sv:825`), which no arm has exercised because every region tested had a
> granule-aligned base.
>
> **This is the mechanism behind sw60's 1,728 bytes, and it is NOT R-30.** R-30 is INIT's
> precondition, which is fixed and verified (sw61, 5,334 INITs). This is a separate defect in bounds
> re-encoding. It was found while investigating R-30 and cost two wrong accounts on the way — see the
> correction box in R-30.
>
> **Boot sw62, four arms, one image, predictions pre-registered before the boot.** Control
> `k800 retval=4`. Every arena a multiple of 64; the two "aligned" ones are exactly 693 × their own
> granule:
>
> | arm | arena | granule | aligned? | result |
> |---|---:|---:|---|---|
> | 2 | 1,419,264 | 2,048 | yes | **clean** — `RR/done`, reclaim count 0 → 1 |
> | 3 | 709,632 | 1,024 | yes | **clean** — returned `rc=1`, count 1 → 2 |
> | 4 | 354,880 | 512 | **no** | **halted**: `RCSH:000001C0` = **448** |
>
> **Arm 4 is the whole finding in two numbers.** `RCCU:00056A40` = **354,880** — the cursor reached
> the TRUE end, so all 22,180 stores advanced and **not one failed**. `RCEN:00056C00` = **355,328** —
> which is exactly `round_up(354,880, 512)`. The shortfall is the difference and nothing else:
> 355,328 − 354,880 = 448 = `RCSH`, so the instrument's own self-check holds.
>
> **The same capability reported two different `end` values in the same arm.** `ALEN` is traced on the
> share path (`sbi_capstone.c:1302`) BEFORE the fill, while the cursor is still at base: it read
> **354,880**, the exact requested size. `RCEN` read the same region after the first store moved the
> cursor: **355,328**. Nothing about the region changed in between except the cursor position.
>
> **Pre-registration, so this cannot be read as fitting after the fact.** 448 is what bounds
> compression predicts; a proportional store-failure rate predicts 432 (= 1,728 × ¼). Both numbers
> were written into the driver header and into this file before the boot ran. The reading was 448.
>
> **Mechanism — five links, each quoted, none inferred.**
>
> 1. `decoder.sv:1309` — `STC` sets `instruction_o.fu = CAPSTONE_DYN`. It is a DYN op, so there is no
>    separate store-unit path (an assumption that cost this lane an hour).
> 2. the DYN unit advances the UNINIT cursor by 16 and passes the metadata through unchanged.
> 3. `ex_stage.sv:1188` — `compress_cap(capstone_dyn_res.cap_rs1)`: the DYN result's `rs1` is
>    **re-compressed on writeback**.
> 4. `ariane_pkg.sv:873` — `compress_bounds(fat_bounds_t'{bound_start, bound_end}, cap.cursor, …)`:
>    the cursor is an input to the encoding.
> 5. `ariane_pkg.sv:787` selects an exact "cursorless" form **only while `bounds.start == cursor`**;
>    otherwise `:817` takes `E = leading_zeros - 12` and `:827-828` rounds the TOP up to a `2^(E+3)`
>    granule:
>    `if(((bounds.base >> (E+3))<<(E+3)) != bounds.base) T[11:3] += 1;`
>
> **Naming trap, and probably how this was missed twice.** Inside `compress_bounds`,
> `len = bounds.base - bounds.start` (`:807`) — so `start` is the LOW bound and `base` is the HIGH
> one, inverted from the monitor's vocabulary. Reading the fat struct's full-width fields and
> concluding "there is nothing to round" is exactly the error that produced the two retracted
> refutations; the rounding is not in the struct, it is in the function that compresses it.
>
> **The cursor is NOT affected, and that asymmetry is useful.** `fat_cap_t` (`:582-585`) carries
> `cursor` as a full-width `xlen_t`, and `cap_metadata_t` (`:637-642`) — what `compress_cap` returns —
> has **no cursor field at all**. Only the bounds are lossy. So any check written in terms of cursor
> readbacks is immune to this defect, and any check that reads a bound back after moving a cursor is
> not.
>
> **What it explains.** sw60's 1,728 = `round_up(1,419,584, 2048) − 1,419,584` exactly. sw61's 5,334
> successful INITs: the `E == 0 && len[12] == 0` sub-case at `:818-821` is exact, so small regions
> never round and the Sublet port never met this. And why INIT still fails on a large UNALIGNED region
> even at the flashed revision, where the R-30 fix made equality legal (`cursor < end` raises,
> `wt-1bfff7776` `capstone_flu_unit.anvil`): the cursor stops at the true end while `end` decodes
> higher, so `cursor < end` and INIT is refused. On an ALIGNED region the two meet and INIT succeeds —
> which is what arms 2 and 3 demonstrate, reclaim count included.
>
> **Exposure.** The reclaim's own postcondition is the only place in the monitor that currently reads a
> bound with the cursor off base — the RTL lane worked the other call sites and INIT resetting the
> cursor to base (`new_cursor = rs2.cursor + start`, with `csinit` passing `x0`) closes them, while the
> `ALEN` trace is safe by explicit ordering rather than luck. **That is a property of today's call
> sites, not a guarantee:** nothing in the API warns a caller, and the region-field query
> (`sbi_capstone.c:1755-1765`) would return a rounded `FIELD_END`/`FIELD_LEN` for any future caller
> that asks mid-fill. Note also the encoder truncates the BOTTOM without a correction term (`:825`),
> so `base` can read LOW by up to a granule on a non-granule-aligned base, and the length case
> subtracts the two.
>
> **A firmware mitigation exists and does not wait on the RTL.** The reclaim's postcondition never
> needs to read `end` back: it verified `cursor == base` before the fill and knows the store count, so
> `base + 16n` is where the cursor must land. Comparing that against the cursor readback uses two
> quantities neither of which is a re-encoded bound. Today the check cannot tell a re-encoded `end`
> from a genuinely short fill, which is precisely what cost sw60.
>
> **Related.** R-30 (INIT's precondition — fixed; this is what remained underneath it) and R-32 (the
> spec/RTL one-off on bounds taken or returned as a VALUE — a different question about a different
> quantity, but the same family of "which number is the bound really").
>
> **2026-09-13 late, boot sw72 — the module's "not representable" log line has its positive control.**
> `bigregion.user 4194304` reports `not-representable-lines=0` and `bigregion.user 1419584` reports
> `1` in the same boot (the host counts the module's dmesg line across its own create_region;
> commit 0f150add), so the line reaches the capture and fires exactly when the arithmetic says it
> must. Every earlier "zero rounding lines" reading (sw63/66/68/71) is now informative rather than
> uninformative. The arena gate (`arena-mismatch-gate.py`) now refuses a non-representable `--expect`
> up front, mirroring `capstone_repr_granule`; 1,419,584 is its negative control.
>
> **2026-09-14, boots sw74 / sw74b — the reclaim of the rounded arena completes (twice), and what that
> does and does not show (claim-auditor SPLIT verdict recorded verbatim in substance).** The
> "shortfall" sw60 (`RCSH:000006C0` = 1,728) and sw62 (`RCSH:000001C0` = 448) reported was never a
> short fill: sw62's `RCEN:00056C00 / RCCU:00056A40` show the cursor reaching the full 354,880-byte
> request while the re-encoded `end` read 448 high, exactly `round_up(354880, 512) − 354880`, and 1,728
> is the same gap for 1,419,584 — this entry's mechanism, already filed here (the R-30 header points
> at it). On the #3 module (d04bd83, which rounds the request: the monitor reports `ALEN:0015B000` =
> 1,421,312 for the 1,419,584 request) with monitor 4274268, the revoke-reshare probe on that request
> reclaims and re-shares through csinit: `released pool rc=1`, `RR/share-A … RR/done`, **RCLM 0 → 1,
> no RCPR / RCSH / RCRE** — sw74 arm 2's teardown and sw74b arm 3's, both `--speedtest1` teardowns.
> **Attribution to the rounding is inferred, not controlled:** the monitor also moved 2c49c41 → 4274268
> (RCSH now measures cursor advance; RCRE, new, reports the re-encode gap), so RCSH's disappearance is
> explained by the monitor alone; the pass through csinit is not, but RCRE has never been observed to
> fire. **The settling control (owed, one arm):** monitor 4274268 + a pre-rounding `.ko` + the same
> 1,419,584 request, predicted `RCRE:000006C0` and a wedge — it positively controls RCRE and proves
> the attribution at once. Two smaller notes: the module's "not representable" `pr_info` is not in the
> UART capture (dmesg only; `ALEN` is the on-console evidence), and host stdout is buffered, so the
> `SQ:` lines' transcript position is not time order (RCLM brackets the pool-release ecall).


### R-34 — every exception the load/store unit generates ITSELF (the five capability causes 24–28 and the misaligned causes 4/6) is LOST when the access is granted in its request cycle, and delivered only if an exception is still being presented one cycle later `FIXED IN RTL SIMULATION 2026-09-15 on r34-r24-exception-delivery (c77c65324) — the MMU exception register upstream #2528 (23355d29f) deleted is restored exactly as deleted; delivery measured end to end at the ex_stage boundary for causes 4/6/24/27/28, load and store side, each ex_o.valid one cycle wide; lint at baseline. SHIPS WITH R-24 AND CANNOT SHIP WITHOUT IT. **SYNTHESISED 2026-09-17** as part of `m1-reclaimer` `054cea69b` (exit 0, WNS -8.307, loops 1, routed Total LUTs 168,757, bitstream written) -- the pair reached that build through the Phase 0 merge `f714d2a72`, verified by CONTENT not ancestry (`misaligned_ex_q` 0 -> 5, `DEBUG_REQUEST` 24 -> 32). **Never synthesised ALONE, so no pair-specific timing or area figure exists and none may be quoted**; what IS pair-specific is lint (the merged tree measured identical to the splice baseline, 52/3/40/2/25/717) and the sweep (12 explained deltas, all the RVTEST_PASS-after-CAPENTER class). NOT on the board as of this line, and the monitor faulted in its own trap handler until sbi_capstone.S:113 was fixed — D3 CLOSED 2026-09-16, capstone-sbi d3-monitor-capability-writeback 2dcd3a5, validated on this branch by a matched pair (old shape cause 24 and store refused, replacement cause 0 and store lands, cursor restored, 1 trap in the run); the branch is held off capstone-bootstrap so the drivers keep their pinned monitor — see WHAT THE FIX EXPOSES. Originally: demonstrated at f6ec6c198, mechanism from source, timing from a waveform, AUDITED (raised 21 times, delivered once); the stock rv64mi-p-ma_addr FAILS the same way with capmode never set, so the loss predates the capability check`

> **Folder (the report):** `capstone/tests/fpga-repros/R34-lsu-exception-lost-on-immediate-grant/` — the directed
> test `lsu-mmode-gate.S`, the runner, the three runs' result lines and the waveform extract.
>
> **What it is.** In M-mode with capmode set (witnessed by a CSCRATCH round trip) and `mstatus.MPRV = 0`
> (witnessed), so that `cap_violation_detection`'s gate (`load_store_unit.sv:966-967`) is satisfied and the
> block demonstrably runs (its revnode tracking at `:985-989` updates), a plain `ld` through a WRITE-ONLY
> tagged capability, a plain `ld` at exactly `bound_end`, a plain `sd` through a READ-ONLY tagged capability, a
> misaligned `lw` and a misaligned `sw`, and a plain `ld` through an untagged base ALL retire with a value and no
> trap; the two stores LAND (the misaligned one corrupts the neighbouring bytes). The single access the cache did
> present one cycle after its request — the first of two back-to-back untagged loads after the stores, whose
> successor held `cap_exception.valid` across the boundary — had its exception delivered: cause 24, which is this
> core's `DEBUG_REQUEST` (R-24), so the core entered the debug ROM with no debug request pending and on `dret`
> the same pair re-ran as two separate pulses and completed silently. **The clauses are live (raised 21 times in
> the audited run, delivered once); delivery depends on whether the exception is still asserted one cycle later.**
>
> **Mechanism.** The exceptions are combinational on `lsu_ctrl` (`:225`; `:820-951` misaligned, `:957-1015`
> capability, merged at `:951`); on an immediate grant the load unit pops that entry in the request cycle without
> consulting `ex_i.valid` (`load_unit.sv:423-424`); the MMU forwards `lsu_exception_o = misaligned_ex_i`
> UNREGISTERED (`cva6_mmu/cva6_mmu.sv:514`) while asserting `lsu_valid_o = lsu_req_q` a cycle later (`:513`,
> `:741`); the units emit `ex_o.valid` in that later cycle only (`load_unit.sv:718` SEND_TAG — its own contract at
> `:715-717`; `store_unit.sv:349` `state_q != IDLE`), when `lsu_ctrl` is already empty. At every fire the waveform
> shows `ex_i.valid = 1`, `state_q = IDLE`, `ex_o.valid = 0` — **re-derived 2026-09-15 into a committed
> artifact** (`sim/vcd-baseline-loadunit.txt`: 21 fires up to t = 3,000 ps, every one of that shape, the single
> delivery at 1,167–1,173 being the three-cycle hold; the earlier `sim/vcd-timing.txt` selected 14 signals and
> contained none of these three, so the sentence had rested on a waveform nobody could reopen). Upstream `23355d29f` (#2528, "extracted PMP") removed
> the MMU's `misaligned_ex_q` register that used to carry the exception with the request; the non-MMU configuration
> still registers (`load_store_unit.sv:459`), the MMU one does not. **Not capability-specific:** the stock
> `rv64mi-p-ma_addr` fails on this RTL with capmode never set (TESTNUM 10, the `ld` crossing an 8-byte word,
> returns wrong data without a trap; the in-word cases pass on the cache's shifted bytes).
>
> **What it changes.** The reading in `docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md`
> that the block is "live for the trusted monitor's own accesses" is superseded: the privilege gate keeps DOMAINS
> off the block, and R-34 loses the block's exceptions everywhere else, so plain data accesses are unenforced at
> EVERY privilege on this RTL (the monitor's rdtime emulation stores through an untagged base at every Linux clock
> read and does not halt: this is why). Capability accesses (`LDC`/`STC`) are the DYN unit's path and are
> unaffected. Misaligned plain accesses silently complete with shifted data (wrong data across an 8-byte word) unless a second exception is presented one cycle later.
> For the paper: the four plain-data-access rows of the safety matrix are not supported on this configuration for
> two independent reasons, and "satisfy the gate" would not have enforced them either.
>
> **THE FIX IS SUFFICIENT AT THE `ex_stage` BOUNDARY (measured 2026-09-15, `sim/vcd-fix-boundary.txt`).** A
> sufficiency condition was raised against any fix: `ex_stage.sv:1015` masks the LSU's exception with the DYN
> load syncer's valid, whose message carries no exception, so a one-cycle lag would lose every single-cycle
> exception. On the RTL lane's `c77c65324` each one-cycle `ex_o.valid` is followed by `load_exception_o.valid`
> with the same cause and then by `csr_regfile_i.ex_i.valid` — causes 4/24/27/28 on the load side, 6/27/28 on
> the store side — because the LSU registers the exception in the same spill register as the valid
> (`load_store_unit.sv:685`). `debug_mode_q` stays 0 across four cause-24 deliveries, so **R-24's renumber and
> R-34's delivery must ship together**: unrenumbered, those four would enter the debug ROM. Scope, stated
> precisely: the block is gated `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`, so what the fix makes
> enforceable is **M-mode code only** — the monitor and the test harness — not domain code.
>
> **THE MISS PATH IS SUFFICIENT TOO, measured 2026-09-15 (`r34-coldmiss-deliver.S`, `9a7bd598c`).**
> A second sufficiency condition, one module below the `ex_stage` one: `load_unit.sv`'s delivery site
> is nested inside the `req_port_i.data_rvalid` block and SEND_TAG asserts `kill_req` when
> `ex_i.valid`, so a faulting access that MISSED would have its transaction killed, never see an
> rvalid, and drop its exception — a silent read through a capability that forbids it. Neither the
> boundary waveform nor the full sweep could answer it: `S12_MEM_DELAY` defaults to 0, so every
> measurement had a response available in the tag cycle and the miss path had never been created.
>
> **⚠ LABEL CORRECTION 2026-09-16 — `S12_MEM_DELAY` IS NOT A CYCLE COUNT.** `stream_delay.sv` (both
> copies in the tree) declares `CounterBits = 4` and `assign counter_load = FixedDelay`, so the
> parameter is **truncated to its low four bits**. `S12_MEM_DELAY=40` — the value in all 39 places it
> appears, described everywhere as "a 40-cycle memory" — realises as **40 mod 16 = 8**. Confirmed
> behaviourally as well as from source: define 12 and define 28 (28 mod 16 = 12) gave IDENTICAL rev1,
> rev2 and total cycle counts on one tree, while the artifact readback proved the builds received
> different defines. **The live trap: any value ≡ 0 mod 16 loads a ZERO counter and realises as LESS
> delay than define 2** — measured at define 16: rev2 = 51, IDENTICAL to the true-bypass run, total
> 1,004 against 708 at bypass and 1,415 at define 2. "A 32-cycle memory" gets you essentially none, and
> it reads as a clean negative. Only 0 reaches the true bypass; 1 is special-cased;
> **usable range 2..15.** **AND THE KNOB IS NOT MONOTONE**, which is the part that bites: the delay
> is a period-16 SAWTOOTH in the define, not a dial. Measured totals on one tree and test: define 0 →
> 708, 2 → 1,415, 12 → 3,427, **16 → 1,004**. Turning it UP from 12 to 16 turns latency DOWN to near
> bypass, so "larger define, more latency" yields a plausible-looking result rather than an obvious
> failure. This is a magnitude label, NOT a retraction: every finding resting on
> "non-zero latency changes the behaviour" stands, S-12, R-26 and R-34 included. Those runs had an
> 8-cycle memory. Found while calibrating the R-12 S1 ladder; detail in
> `docs/history/16-09-2026_11-54-05_r12-s1-revoke-walk-cost-ladder.md`.
>
> **Rebuilding at delay 40 and rerunning an existing test does NOT answer it either, and this is the
> trap worth keeping.** `lsu-mmode-gate` at `S12_MEM_DELAY=40` returns all thirteen readings
> IDENTICAL to the zero-latency run, value for value, only the cycle counts moving (868 → 2485).
> That reads as robustness and is not: the test hammers one buffer, so the line is resident after the
> first access and the faulting arms still HIT. Turning the delay on is not the same as making the
> access miss.
>
> A matched pair through the SAME write-only capability, one variable — whether the faulting line was
> brought in first — with every precondition witnessed in the same run, because "cold" is a claim
> about cache contents a test cannot otherwise see:
>
> | reading | value | |
> |---|---|---|
> | warming value | `0x4c535550` | the warm line really is resident |
> | warming cause | 0 | the warming access did not itself trap |
> | ARM W value / cause | 0 / `0x1b` = 27 | warm line, delivered |
> | line separation | `0x1000` | the two lines really are distinct |
> | ARM C value / cause | 0 / `0x1b` = 27 | **cold line, delivered; no data returned** |
> | total traps | 2 | one per arm, none spurious |
>
> Identical at delay 40 and at 0; only cycles differ, 1287 against 499. So R-34's fix has **no open
> sufficiency condition**: boundary measured, miss path measured, renumber measured. What stands
> between it and a bitstream is the monitor, not the RTL.
>
> **WHAT THE FIX EXPOSES, and why 12 tests are left failing on purpose.** With delivery working, the
> suite sweep (RTL lane, 2026-09-15, against baseline `4cc068572`) has 12 tests that passed before
> now timing out. They are **one pattern, not twelve bugs**: a plain load or store through an
> INTEGER-derived, untagged base while capmode is set in M-mode. Every one of the twelve calls
> CAPENTER; not one of R-24's 15 cause-assertion tests does, which is the discriminator.
>
> Ten fault on the test harness's own success convention rather than on anything the test does —
> `verif/tests/riscv-tests/env/p/riscv_test.h:239` is `sw TESTNUM, tohost, t5`, which the assembler
> expands to `auipc t5, %pcrel_hi(tohost)` then a store through that integer base. Matched pair from
> the same sweep, one variable:
>
> | | epilogue | outcome |
> |---|---|---|
> | `cap-overwrite` (no CAPENTER) | `auipc t5` / `sw gp,-334(t5)` | retires, `mem 0x80001000` written |
> | `cld` (CAPENTER) | `auipc t5` / `sw gp, off(t5)` | exception |
>
> Two fault earlier, in the test body, and are the same rule firing sooner: `data-transfer` on a
> `c.ld` immediately after CAPENTER, `cpmp-if-check` 3102 times.
>
> **Cause 24 is MEASURED here, not inferred.** `cpmp-if-check` installs `mtvec` before it faults, so
> its handler reads the value: `csrrs t0, mcause` returns `0x18`.
>
> **The monitor is the same pattern and is the one that matters.** `sbi_capstone.S:112-114` computes
> `add t5, sp, t5` and stores `sd a0, 16(t5)` through the integer result — that is the emulated-CSR
> writeback, so it runs at every `rdtime`. It is the ONLY site of this shape in the file, and the C
> half adds none (`mtime`/`mtimecmp` are capabilities minted by `split_out_cap`). The fix is three
> instructions using a macro the file already defines seven times: `CINCOFFSET(sp, sp, t5)` IN PLACE
> (rd == rs1, which sidesteps the R-25 source-consumption question entirely), the store, then the
> negated offset back.
>
> **That fix cannot be validated on the deployed bitstream**, and this is the trap to write down: on
> the current silicon the exception is never delivered, so before and after both run clean. An
> emulator pass against the flashed bitstream proves only that nothing broke, never that the change
> is correct. Only simulation of the fix branch can validate it.
>
> **2026-09-17 — R-34 AND R-24 ARE IN THE BITSTREAM QUEUED FOR FLASHING, WHICH IS LABELLED FOR
> SOMETHING ELSE.** `caplifive_m1_054cea69b.bit` is described as the splice-plus-reclaimer build and
> **also carries this pair.** Established BY CONTENT, not by its name and not by ancestry:
> `misaligned_ex_q` in `cva6_mmu.sv` goes 0 → 5 between the flashed `1bfff7776` and `054cea69b`, and
> `DEBUG_REQUEST` in `riscv_pkg.sv` goes 24 → 32. (All three R-34/R-24 commits are also ancestors, but
> ancestry cannot see a cherry-pick and content can.) **Consequence: the moment it is resident the
> monitor's unfixed writeback stops being a dropped exception and becomes a delivered one, inside the
> monitor's own trap handler, on the path taken at every `rdtime`.** The drivers bake the monitor at
> `4274268` and eight of them GATE on that commit, where the integer add is still at line 113, so the
> D3 fix below must be merged and those pins moved before the first boot on the new silicon. The
> memory map does NOT move: `CAP_REVNODE_MEM_BASE` and `CAP_TAG_MEM_BASE` are byte-identical at both
> revisions (0xBFF00000, 0xBC2D2D2D), verified independently by two lanes, so the device tree stays
> valid. Note also that the two entries' own "NOT synthesised, NOT on the board" status lines are
> about to be overtaken by a build queued for a different reason.
>
> **D3 IS CLOSED — the monitor fix exists and is validated, 2026-09-16.** `capstone-sbi`
> `d3-monitor-capability-writeback` at `2dcd3a5`: the writeback moves the stack capability's cursor in
> place (`CINCOFFSET` with `rd` = `rs1` = `sp`, the store at 16 off `sp`, the negated offset back), so the
> source-consumption question does not arise. Validated where it CAN be — against this branch, since on
> the deployed bitstream the old form and the new one both run clean. Matched pair
> `tests/monitor/d3-monitor-writeback.S` at `c77c65324`, 645 cycles: the replacement takes **cause 0** and
> its store reads back, the capability is **bit-identical before and after** the in-place move, and the old
> shape run last takes **cause 24** with its store refused — **exactly one trap in the run, and it was the
> control's.** The replacement is bounds-checked where the integer base was not, which the change
> INTRODUCES: the store lands at `frame_base + slot*8`, the `SAVE_REG`s that run first write slots
> 1..31 through the same register, so **slot 0 alone is newly checked** (reachable when the emulated
> `rd` is `x0`) — arm B0 reproduces that geometry, cursor 16 BELOW the base, and reads cause 0 with
> the value stored. **Residual:** whether the LIVE stack capability's base is at or below
> `frame_base` rather than `frame_base + 8` is runtime state no synthetic test settles; it would show
> as cause 28 on slot 0 only, and the first boot on a delivering bitstream decides it.
> Had the control not trapped, the build would not be delivering the exception and the
> replacement's clean return would have meant nothing. The branch is held OFF `capstone-bootstrap`, which
> stays at `4274268`, the commit the board drivers pin by hash. Note: `RVTEST_PASS` is the same defect —
> `sw TESTNUM, tohost, t5` expands to a store through an `auipc` integer base — which is why ten of the
> twelve sweep tests time out in their epilogue; that pair exits through a capability minted over `tohost`
> and reports SUCCESS, so the same three lines would convert those ten timeouts into readings. Full note:
> `docs/history/16-09-2026_15-00-00_d3-monitor-writeback-validated.md`.
>
> **Fixing the monitor does NOT make the plain-data-access safety rows hold.** The gate is
> `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M` and domains run in S-mode, so the check still never
> applies to domain code. It makes the M-mode arm enforceable instead of self-faulting. "R-34 is
> fixed" must not be read as "the plain-data rows now hold".
>
> **What would settle the residuals** *(2026-09-15 later: (b) CLOSED by the RTL lane at the flashed revision — a store at
> `bound_end` through the same capability the load arm uses takes no trap and the guard word past the buffer holds the
> stored value, the corrupting direction; (a) MOOT for the capability clauses — translation through MPRV needs MPP below
> M and the load-store privilege is then that MPP, so translation on takes the gate out of its satisfied state, and
> S-mode fails it anyway; what (a) still covered was the MISALIGNED causes with translation on — CLOSED by the RTL lane (dev `1e205738c667`):
> with MPRV set, MPP supervisor and an sv39 identity map witnessed working by an aligned translated load, the misaligned
> load is dropped both with and without translation (shifted value, cause 0, 0 traps in 537 cycles), so the drop is
> independent of translation, as the unregistered forward implies. The run before it had REFUSED its own precondition —
> the translated aligned load access-faulted (no PMP entry once data accesses ran at supervisor privilege) and that run's
> misaligned arm reported cause 4, which read alone would have been a clean, wrong "delivered under translation". Both
> residuals are closed; what remains is the fix decision, sequenced with R-24).*
> (a) The translation-on path (S-mode, Linux) is argued from source only
> (`cva6_mmu.sv:539` skips the translation branch on a misaligned request and leaves `:514` in force): a directed
> test with `satp` set, or a board arm. (b) A store bounds arm. (c) After an RTL fix, this test must FAIL its last
> arm by design — the untagged load then enters debug mode (R-24), and so would `RVTEST_PASS`'s own `sw` to
> `tohost` (it raised cause 24 in the run) and the debug ROM's accesses — and the write-only/bounds/misaligned
> arms must trap with 27/28/4/6; `rv64mi-p-ma_addr` must pass. (d) The `trans_id` mis-attribution the delivery
> path implies (`load_unit.sv:718-721`: the responding load's id, the next request's cause) needs two DIFFERENT
> causes back to back to show.

### R-35 — on silicon a REVOKED capability still READS AND WRITES the storage its object gave up, and the access does not trap, because the LSU's single core-wide revnode tracker RE-ADOPTS any unseen id as valid `FIXED on the M-mode LSU data path — CLOSED 2026-09-25 by the RTL lane (decision delegated by the project lead), after an adversarial audit. Fix: capstone-ariane 4ad0df694, flashed as caplifive_r35_4ad0df694.bit (2026-09-24). Two-sided in RTL simulation: acceptance fixture 4 -> 7 traps, build shown to track source; the stale read, stale store and readback trap 25; live controls return data. On silicon the exposing probe NO LONGER REPRODUCES: image 35fb3fec traps 25 at the stale lbu where 054cea69b read live data; a live alias and ~43k live accesses commit; 17/17 ladder rungs unchanged. N=1 per arm, no closing control, bitstream identified by label and behaviour. NOT shown on silicon: which cause-25 arm fired, which probe (k=0 or k=21648) trapped, the stale write. That k=0 cannot have been allowed is a SOURCE argument: an allow needs an exact 30-bit tag hit marked live; the 14-bit generation retires at 16383 and never wraps (capstone_rev_node.anvil); reuses_since <= 2706. The flashed configuration denies on a miss, which the RTL calls an oracle rather than the shipping design; its false-deny cost is R-43, and any replacement must re-pass this folder's acceptance. LDC/STC never used this tracker: they decode to CAPSTONE_DYN and are gated by the DYN unit's rev-node query (a stale STC has no fixture). NOT CLOSED BY THIS: the same optimistic adopt at the CPMP (S/U mode, live in production) -- R-44 -- and at commit_stage's PC tracker (latent). PREVIOUSLY RECORDED: ON SILICON (2026-09-24, caplifive_r35_4ad0df694.bit): the probe that exposed R-35 no longer reproduces it. The same image that read the current occupant's live data through a revoked alias on 054cea69b (is_live_data=1) now traps with cause 25 on the stale read of leaf[0]. A live alias to the same leaf, at the same address, commits without a trap, as do ~43k earlier live accesses, and 17 of 17 ladder rungs return their pre-flash values. N=1 per arm. NOT shown on silicon: WHY the stale read was denied -- cause 25 cannot separate observed revocation from the cache's deny-on-miss, which is the expected route for an id reissued thousands of times; which probe age trapped (k=0 or k=21648); the stale write; and false-deny rates under SQLite. Result lines and limits: results/board-4ad0df694.result-lines.txt. History follows. WAS (earlier 2026-09-24): FIX SYNTHESIZED AND TIMING-CLEAN, NOT YET ON SILICON: capstone-ariane 4ad0df694 routes at WNS -8.341 against the flashed base's -8.307, all five pre-registered predictions pass, bitstream sha256 8db73f8e20244438a2663fef070202e95dde29fe5b1d957b9804a7babf60382c; reflash authorized by the lead, pending the board probe. History follows. WAS: FIXED IN SIMULATION, NOT YET DEPLOYABLE. Root-caused at 054cea69b (load_store_unit.sv, the optimistic adopt) and reproduced two-sided in RTL simulation 2026-09-22. FIX: capstone-ariane f83fe9342 (branch r35-m1-revnode-cache), a tagged positive validity cache filled ONLY by passive taps on the rev-node unit's own node-memory traffic, so an access is allowed only if its exact 30-bit (generation,index) is resident and was last seen live. The acceptance fixture inverts (exactly 7 traps; the three revoked accesses trap 25; both live-alias controls still return data) and lint is at the committed baseline -- and f83fe9342 WAS SYNTHESIZED and is NOT deployable: area fixed (177,669 post-synth LUTs) but routed WNS -27.665 from +15.45 ns of route delay on a fill-side path (the rev-node's combinational write-request selector driving the cache's 256-entry write decode). Next: register the fill taps. The two hashes that WERE synthesized are not reflash candidates: Stage 0 (247b76896) routed at WNS -12.900 against base -8.307, and the first cache (079dc720a) at -14.415 with 192,642 LUTs = 94.53 % of the device because it described a crossbar. The intermediate register-file rebuild (6ee277cc3) introduced an authority escape of THIS issue's own class, closed at f83fe9342 and found only by an adversarial audit -- the fixture returned 7 traps before and after, since it cannot see same-cycle coincidences. The defect is in the BROADCAST PROTOCOL (the invalidation carries a bare index), which is why no tracker-local fix exists. Board evidence is build-contaminated and the simulation supersedes it`

> **Folder (the report):** `capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/` — the
> board probe, the RTL-simulation reproducer `src/r35-rotate-stale.S`, its 117 result lines, and the
> acceptance criteria a fix must invert. Siblings: **R-37** (the ordering defect at the same trackers)
> and **R-38** (the permanent false-deny at the CPMP tracker).
>
> **What it is.** `core/load_store_unit.sv` keeps **one core-wide** tracked `(generation, index)`
> revnode id plus a `valid` bit. Its adopt arm takes any access whose capability presents a
> **different** id and asserts `valid = 1'b1` **without asking the rev-node unit**. Cause 25
> (`INVALID_CAPABILITY`) is the **last** arm of the priority chain and tests only that bit, so once the
> tracker has been displaced by an access through any other capability the clause **cannot fire**.
> Bounds (28) and permissions (27) survive because they read the capability's own metadata.
>
> **The severity framing, which is what makes it more than a corner case:** a single core-wide tracker
> rotates on **every** access through a different capability, so for any M-mode program touching two or
> more capabilities the revocation check is effectively **VACUOUS** — cause 25 can only ever fire for a
> program that never rotates. The reproducer's arm A is exactly that never-rotating program, which is
> why it is the positive control rather than the finding.
>
> **Witnessed conditions (RTL simulation at `054cea69b`, `S12_MEM_DELAY=12`, M-mode, capmode set).**
> The data grant was `SPLIT` into halves carrying **distinct revnodes 2 and 3**, each `MREV`'d and
> `DELIN`'d to a NONLIN alias. `LCC` witnesses the revoke in-run: alias A reads **1 before** and **0
> after**, sibling B stays **1**. Then: **arm A** (no rotation) traps 25 and retires no value; **arm A2**
> (a second un-rotated load) **also** traps 25 — this is what makes the pair single-variable, since it
> proves the trap did not itself re-validate the tracker; **arm B** (one intervening access through the
> *other* revnode) returns the revoked object's data with **cause 0**; **arm C** stores `0xA5` through
> the revoked alias and **it lands**, read back through a live alias.
>
> **Unit attribution, and it does not rest on silence.** The run's fourth trap is a plain `sd` through
> an integer base retiring **cause 24**, and cause 24 for a store is emitted at exactly one place in
> `load_store_unit.sv` — so `cap_violation_detection` demonstrably reached the exception path **in this
> run** and is silent for arms B and C. Plus decode routing (a plain `ld` never enters the DYN unit) and
> a separate `commit_stage` exclusion. Counts: **4** `exception @` in the retirement trace against
> **1** `Exception:` in the console.
>
> **Why no fix has landed.** Three independent walls, each verified: (1) flipping the assumption to
> `1'b0` fails **closed** — under id rotation nearly every access would raise 25, and R-12's reclaimer
> *relies* on the adopt to re-adopt a fresh `(g+1, i)`; (2) giving the LSU a port on the rev-node query
> channel is refused on five counts, the binding one being that `capstone_rev_node` and `ex_stage` are
> both named members of the standing combinational ring; (3) **the tracker's only external input carries
> no generation** — `send_revnode_update` masks the id to 16 bits and its own comment states *"the
> generation lives in the node, never in a link or an address"*, and all 14 broadcast call sites go
> through it. So the optimistic adopt is the only thing the tracker *could* have done, and a
> DEAD-SET is provably unsound (per-index generation would cost 65,536 x 14 bits, aliasing
> false-denies the frequent generation-0 nodes, and the live-index set does not saturate — 0.529
> linear). **Record these as negative results so they are not re-derived.**
>
> **Sibling instances of the same adopt, not this defect:** `core/commit_stage.sv` for the **PC**
> capability (genuinely latent — needs a stale *code* capability via CALL/RETURN, never constructed) and
> `core/pmp/src/pmp_data_if.sv` per CPMP entry (already sim-measured by
> `verif/tests/custom/capstone/r12-recl-cpmp.S` probe 4). The two privilege gates are
> **complementary, not redundant** — the LSU block is `== PRIV_LVL_M` and the CPMP check is
> `!= PRIV_LVL_M` — so the CPMP instance is **100 % of S/U-mode capability enforcement** and is LIVE IN
> PRODUCTION, not latent: `capmode` is a sticky core-wide bit and the monitor sets it before the first
> S-mode instruction, so capability-unaware Linux and userspace run under full CPMP enforcement and
> cannot opt out. It is deferred on a **boot-kill risk** (a permanent deny on `cpmp(0..2)`, whose ids
> are hardcoded and produced with zero rev-node traffic, predicts an unbootable board on the first
> S-mode instruction fetch) — **not** on reachability, and **not** on a fault-loop argument, both of
> which were raised and refuted.

### R-36 — NUMBER WITHDRAWN BEFORE FILING; DO NOT REUSE `NOT AN ISSUE. Drafted 2026-09-21 and refuted by a claim-auditor before it was filed; the refutation was then verified independently and recorded. Kept as a stub only because the number is referenced in history/`

> **What it would have claimed.** That the RTL revoke walk never unlinks dead nodes, that this is why
> `give_cyc/n` grows superlinearly, and that implementing splicing would fix it.
>
> **Why it is wrong**, per `docs/history/15-09-2026_19-16-04_manuscript-read-directly-at-last.md` §12:
> the splice **already exists and is in the resident bitstream** (`379248185` is an ancestor of
> `054cea69b`; the draft had read `capstone_rev_node.anvil` on the side branch
> `board/r35-directed-repro` and mistaken it for the fix line); it duplicates **R-12**'s cost half; and
> the mechanism **predicts the wrong shape** — an unspliced walk at a constant ~3 cycles per dead node
> is **linear**, and it was being used to explain a **superlinear** curve. The curve is a linear walk
> multiplied by a **memory-hierarchy cliff**: the d-cache holds exactly **2,048** nodes (32 KiB / 16 B,
> = 256 sets x 8 ways, one node per line) and the 3.000 slope holds to 1,024 and breaks at 2,048.
>
> **What survived the refutation**, because a dead mechanism can leave a live variable: the independent
> variable is **per-domain minting, not table occupancy**
> (`history/15-09-2026_19-25-14_m1-revoke-cost-variable.md`), and the give-vs-take *shape* asymmetry is
> a direct RTL reading that stands on its own — `REVOKE_NODE` advances via `walk_next := node_in.next`,
> a serial **dependent** node-read chain with no memory-level parallelism, against a fixed handful of
> up-front addresses for INIT/DELIN.

### R-37 — the revnode trackers' INVALIDATE and ADOPT are mis-ordered, so an invalidation broadcast arriving in the same cycle as an adopt is LOST at the LSU and mis-applied at the CPMP `STAGE 0 IS IN THE FLASHED BUILD since 2026-09-24 (247b76896 is an ancestor of the flashed 4ad0df694): fixed in source, NOT verified on silicon; the LSU half is moot, since R-35's fix replaced the LSU tracker outright. WAS: OPEN in the flashed build, read from source at 054cea69b. Stage 0 was IMPLEMENTED at capstone-ariane 247b76896 and SYNTHESIZED 2026-09-23: routed WNS -12.900 against base -8.307, a 4.593 ns REGRESSION from 32 lines and zero new signal declarations, with all nine lint counters at the committed baseline -- so the earlier expectation that its zero lint delta made it 'a usable bisection control' held for lint and failed for synthesis. A before-audit localized the cost to the LSU half, whose post-adopt value fed cap_exception combinationally. (AUDITED, NOT MEASURED: Stage 0 changed the LSU and CPMP halves in one commit and 4.593 ns is their combined cost; no build has separated them. The split is one build -- 247b76896 with pmp_data_if.sv reverted -- and has not been run.) NOT a reflash candidate. At the LSU this issue is now moot rather than fixed: the tracker it reorders no longer exists in R-35's fix (f83fe9342), which keeps the same-cycle-invalidation property with a single comparator against the accessed id`

> **Folder:** shares `capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/`. Parent
> defect **R-35**; see also **R-38**.
>
> **What it is.** `core/commit_stage.sv` has the safe shape — it adopts first and invalidates **last**,
> comparing the **post-adopt** value. The other two sites do not:
> - **LSU:** the invalidate runs **before** the adopt, and the adopt then unconditionally asserts
>   `valid = 1'b1`. So a broadcast landing in the adopt's cycle is **overwritten and LOST** — the
>   adopt wins, which is a **false ALLOW** and therefore R-35's own defect class.
> - **CPMP:** adopt runs first and invalidate last (so the invalidate *wins*, the conservative
>   direction) but it compares the **pre-adopt** id. See **R-38** for the consequence.
>
> **Two traps for whoever fixes or reviews this.** (1) A clean directed sweep after the fix proves
> *no behaviour change*, **not** that anything was fixed — the finding is unobservable while R-35
> stands. (2) The sub-case where the lost broadcast is for the id being **evicted** behaves
> identically under both orderings, so a reader checking only that case will wrongly conclude the
> finding is bogus.
>
> **And do NOT "normalise" the two sites by moving the CPMP to the LSU's order.** Implementing
> "adopt first, invalidate last" with the LSU's *adopt-wins* semantics would leave an entry rewritten
> to `(g,i)` racing that id's invalidation reading `valid = 1` for an already-revoked node, with no
> further broadcast for it ever arriving — a **permanent false ALLOW**. The target shape is the
> CPMP's direction (invalidate-wins); the LSU must move *toward* it.

### R-38 — the CPMP tracker's invalidation compares the PRE-ADOPT id, so an entry rewritten in the same cycle as a broadcast for its old id is left tracking a LIVE capability marked permanently INVALID `STAGE 0 IS IN THE FLASHED BUILD since 2026-09-24 (247b76896 is an ancestor of the flashed 4ad0df694): fixed in source, NOT verified on silicon; the CPMP compare now tests the post-adopt id (_d). WAS: OPEN in the flashed build, read from source at 054cea69b. RETRACTED 2026-09-23: this entry said the consequence is a false DENY and 'never an authority escape'. An audit has since constructed a false ALLOW from the SAME code -- entry holds X; the CPMP presents Y so the adopt fires; the same cycle's broadcast names Y (the NEW id, not the old one); the compare is against the stale X and misses; the entry latches valid for a dead Y indefinitely. So 'never an authority escape' is withdrawn. Its REACHABILITY is UNRESOLVED -- it needs a CPMP entry write in the same cycle as a broadcast for the id being written. Stage 0's _q -> _d retarget (247b76896, carried into f83fe9342) closes both the filed false deny and this false allow. The same audit ARGUES that this CPMP half is timing-innocent (every consumer reads the registered value, so the path is flop-to-flop into 16 endpoints) and that the 4.593 ns lies in Stage 0's LSU half. (AUDITED, NOT MEASURED: Stage 0 changed the LSU and CPMP halves in one commit and 4.593 ns is their combined cost; no build has separated them. The split is one build -- 247b76896 with pmp_data_if.sv reverted -- and has not been run.) DO NOT revert it to _q as a timing measure: that reinstates the false allow. A parallel-compare restructure of it is audited bit-identical and parked`

> **Folder:** shares `capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/`. Siblings
> **R-35** (parent) and **R-37** (the ordering half).
>
> **What it is.** At `core/pmp/src/pmp_data_if.sv` the adopt runs first and the invalidate last, but
> the invalidate compares the broadcast index against the **pre-adopt** tracked id. The same-cycle race
> therefore fails in **opposite directions** depending on which id the broadcast names — which is why
> one sentence cannot describe it, and why an earlier one-sentence version of this finding was wrong:
> - broadcast names the **newly adopted** id → no match against the old value → validity **not**
>   cleared → a live-looking entry for an id just invalidated (**false ALLOW**; that is the CPMP
>   instance of **R-37**);
> - broadcast names the **displaced** id → match → validity **is** cleared → but the entry now tracks
>   the **new** id, left marked invalid with **no further broadcast for it ever arriving**, so the deny
>   is **permanent** (**false DENY**; this issue).
>
> **What the fix does and does not cover.** The `_q -> _d` retarget closes the case above. It does
> **not** help where an entry is rewritten to `(g+1,i)` and a broadcast for `(g,i)` arrives, because
> the broadcast is a bare 16-bit index and **no compare width can separate them** — however, that
> second case is believed unreachable by `>>` **sequencing** (the reclaim pop's generation advance is
> ordered before `(g+1,i)` is handed out), which is a stronger argument than a cycle-count one and is
> recorded in the plan. A separate route to a stuck entry is **not** fixed by any compare change: an
> invalidated entry cannot re-adopt **the same** capability into **the same** entry, because the adopt
> guard requires a *differing* id. Any commit must say **which** of these it closes.
>
> **Correction to the first write-up of this, recorded because it was cited circularly.** It was
> initially claimed that `swap_cpmp` reinstalls such an entry into a **fault loop**. It does not:
> `swap_cpmp` skips a region already loaded (`region_cpmp[region_id] != -1 → continue`,
> `sbi_capstone.c:1911`), falls through to the no-covering-region path, and takes a **one-shot**
> terminal exit — `print_regions()`, a `CPMX`/`CAPSTONE_NO_CPMP_REGION` report over UART, then
> `fault_return_from_domain`, which does not return. So the symptom is **a single domain kill with a
> diagnostic report**. It was also claimed that such an entry can **never** re-adopt; re-adoption does
> happen, because the round-robin eject sets `region_cpmp[ejected_region_id] = -1` and a later install
> lands in an entry whose tracked id differs. The narrow true statement is the "same capability, same
> entry, no intervening occupant" one above.

### R-39 — the rev-node unit broadcasts an invalidation for INDEX 0 during ordinary operation, unguarded, and every tracker treats a matching index as a reason to clear validity `OPEN but believed HARMLESS, on an invariant that is verified for RTL and UNRESOLVED for software. Found by a claim-auditor while auditing the R-37 ordering fix; PRE-EXISTING and not introduced by it. No exploit and no observation -- filed so the invariant is written down rather than relied on silently`

> **Folder:** none of its own; the analysis sits with **R-35**'s
> (`capstone/tests/fpga-repros/R35-revoked-reference-retains-authority/`). Siblings **R-35**, **R-37**,
> **R-38**.
>
> **What it is.** `core/ex_stage.sv` fires an invalidation broadcast on any node write whose `valid`
> bit is clear. **The wire is 30 bits wide (`node_wr_req[29:0]`) and carries a bare index** — the
> generation is masked off at source by `send_revnode_update`, so bits `[29:16]` are hard zero, and the
> **16-bit compare lives at each consumer** (e.g. `commit_stage.sv:247`), not in the broadcast.
> Calling the broadcast itself "16-bit" is the loose phrasing that makes the opposite-widths rule hard
> to apply correctly. Three writes carry `valid = 0` on a
> node that is *already* dead and stays dead — the deadness-preserving neighbour fixups. Two of them
> can name **index 0**: `capstone_rev_node.anvil:224` (INIT's next-node `prev` fixup, reachable because
> `node_2.next = 30'd0` at INIT_STAGE, so any INIT whose parent is the chain tail) and `:270` (MREV's
> prev-node `next` fixup, reachable because `node_1.prev = 30'd0`). **Only the REVOKE splice guards
> index 0** (`:42`), and its comment shows the author knew the shape but treated it as local to that
> site.
>
> **Why it is believed harmless, stated as an invariant so it can be checked rather than assumed:**
> *no live capability carries revnode index 0.* Exhausted for the RTL mint sites — root capabilities
> are `30'd1`/`30'd2` (`capstone_flu_unit.anvil`), the monitor's PC capability is hardcoded `30'd1`
> (`commit_stage.sv:197`), allocated indices are **>= 3** because `head` starts at 3 and the free-list
> push requires index > 2, the id-0 responses are *failure* responses on which the DYN unit raises
> rather than minting, and untagged ingestion forces `NOT_CAP` with `revnode_id = 0`, which every
> adopt guard excludes. The compressed `revnode_id` is a full 30-bit field, so no two indices can
> alias in a tracker.
>
> **UNRESOLVED, and it is the whole residual:** whether software — the monitor, the boot image, a
> debug/CSR path, or any residual partial-store tag-retention hole — can materialise a **tagged** word
> with `revnode_id[15:0] == 0` and a LINEAR/NONLIN type. If it can, an unguarded index-0 broadcast
> would clear a tracker holding it, producing a **false DENY** (never an authority escape). That would
> be a forgery hole in its own right rather than a defect in this broadcast, which is why this entry
> records the invariant instead of proposing a guard.
>
> **Why it is not simply guarded anyway.** A guard at `:224`/`:270` is two comparisons and would be
> cheap — but it would also make the invariant *unstated and unchecked* at the cost of looking
> addressed. The useful artefact here is the written invariant plus the software question.

### R-40 — the monitor's region ecalls have NO authorization check on the GENESIS region ids 0-2, so an unprivileged caller can aim `share` + `revoke` at the region whose capability carries the monitor's own PC rev-node `OPEN, and MATERIALLY CORRECTED 2026-09-22 after a claim-auditor REFUTED the single-ecall mechanism this entry was first filed with. VERIFIED: no id guard on either the share or the revoke path; genesis region 0's capability carries revnode 1; the ecall path has no authorization check; /dev/capstone is mode 0666 with an ioctl forwarding a caller-supplied region id. NOT VERIFIED, and now the whole open question: whether the resulting walk actually invalidates node 1 and faults the monitor's own fetch. NOT OBSERVED -- no run has been made`

> **Folder:** none of its own; analysis with **R-35**'s. Related: **R-39** (the index-0 invariant),
> **R-41** (a regression in the very guard that blocks the single-ecall form), **R-35**.
>
> ## RETRACTED 2026-09-22 — the single-ecall mechanism this entry was first filed with
>
> It claimed **one** ecall, `revoke_region(0)`, revokes the monitor's PC capability. **False, and the
> refutation is a line of source the first version stopped short of reading.** `revoke_region` contains
> **no `__mrev` at all** — the premise "`revoke_region` does `__mrev(r)` then `__revoke(rev)`"
> describes no version in the tree. It does this (`sbi_capstone.c:1669-1675`, FPGA build input):
>
> ```c
> if (region_cpmp[region_id] != -1) {
>     rev = read_cpmp(region_cpmp[region_id]);
>     if (cap_type(rev) != CAP_TYPE_REV) {   // <-- region 0 exits HERE
>         return 2;
>     }
>     r = __revoke(rev);
> ```
>
> Genesis region 0's capability is **`CAP_TYPE_LINEAR`**, not REV: `capstone_flu_unit.anvil:504-505`
> sets `revnode_id = 30'd1` and `cap_type = cap_type_t::CAP_TYPE_LINEAR` on **consecutive lines** — the
> first version quoted the revnode and not the type. The guard is on the `regions[]` arm too and is
> present in the **compiled** artifact (`sbi_capstone_dom.c.S:5628-5642`: `lcc`, compare, `bnez` ahead
> of `revoke(t0)`), so this is not a source-only argument.
>
> **It also fails at the RTL, independently of that C guard**, which makes the refutation
> firmware-age-independent: `capstone_dyn_unit.anvil:48-64` reaches `send rev_node_ep.rev_req` only
> inside the `else` of a type test, so `csrevoke` on a LINEAR capability raises **cause 26
> UNEXPECTED_CAP_TYPE at the `csrevoke` itself** and the walk never starts. **Corrected severity of the
> single-ecall form: a refused ecall, not a monitor fault.**
>
> ## What IS verified, and the mechanism that survives — it needs TWO ecalls
>
> 1. **Genesis region 0's capability carries revnode 1.** `CAPENTER`'s else branch (taken, since
>    `_cap_text_start/_end` are non-zero) emits **both** outputs at `revnode_id = 30'd1`
>    (`capstone_flu_unit.anvil:489`, `:504`). `a0` is never written between there and
>    `call cap_env_init`: only SPLITs on a1/a2, two `mv`s, a `CCSRRW`, `MOVC(a2,a3)`, and
>    `call dom_init` — which in the FPGA asm (`sbi_capstone_dom.c.S:8426-8505`) neither writes `a0`
>    nor makes any call. **Correction:** `cap_env_init` writes **`cpmp(0..2)`**, *not* `regions[0..2]`
>    (`sbi_capstone_dom.c:11-33`), so `regions[0]` is an untagged zero (`NOT_CAP`) and the live
>    capability lives in the CPMP entry. The first version said both.
> 2. **Neither path guards ids 0-2.** `revoke_region` checks only `region_id >= region_n` and
>    `region_live[region_id] == 0`; `shared_region_annotated` only
>    `dom_id >= dom_n || region_id >= region_n` plus the same liveness flag. **Zero** comparisons of
>    `region_id` against any literal on either path, and genesis regions pass by construction
>    (`region_live[0..2] = 1`).
> 3. **The ecall path has no authorization check.** `sbi_capstone.S:69-76` tests `mcause` against
>    `CAUSE_SUPERVISOR_ECALL` then calls `handle_trap_ecall`, which runs straight from its signature
>    into `switch(ext_code)`. `grep` for `MPP|mstatus` over that file returns **0**, positive-controlled
>    (the same pattern returns 3 on `sbi_capstone_init.S`, so the instrument fires).
> 4. **The caller who chooses the id is UNPRIVILEGED.** `/dev/capstone` is **mode 0666**
>    (`modcapstone/module/capstone.c:658`) and `ioctl_revoke_region` (`:390-398`) forwards a
>    user-supplied id straight into `sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_REVOKE, …)`.
>    The ecall is S-mode *by design* — S-mode is the legitimate SBI client, which is why "no privilege
>    check" was the wrong framing — but the **region id originates in U-mode**. The defect is a missing
>    **authorization** check on genesis ids.
> 5. **THE SURVIVING MECHANISM, and it is UNRESOLVED at its last step.**
>    `shared_region_annotated(dom, 0, …, REV_DEFAULT)` accepts region 0 and does
>    `__rev void *rev = __mrev(r);` then `write_cpmp(region_cpmp[0], rev)` (`:1363-1371`) — installing a
>    **genuine REV** into `cpmp(0)`. A following `REGION_REVOKE(0)` then reads a REV, **passes the type
>    guard**, and runs the walk on node 1's subtree. Everything up to the walk is verified above.
>    **NOT verified: that the walk ends with node 1 `valid = 0`, and that this faults the monitor's
>    fetch** via `commit_stage.sv:197` (`revnode_id = 30'd1`) and `:224-226`.
>
> **Cheapest settling experiment — simulation, not the board:** a directed `.S` that MREVs node 1 and
> issues `csrevoke` on the senior handle, asserting on the next monitor fetch. One run, no boot.
>
> **A wider reachability question this opened, also UNRESOLVED:** `split_out_cap`'s
> `if(base == region_base) region = mem_l;` (`:784-787`) returns the parent capability **with its
> revnode intact**, and `SPLIT` gives the fresh node only to `rd`. So an ordinary `create_region` whose
> base equals a genesis region's base may inherit **revnode 1** with no genesis id ever being named,
> which would make this reachable without ids 0-2 at all. No shipped board run is established as a
> control for that branch.
>
> ## Lineage correction — the third instance of this same error in one session
>
> The first version cited the **102,389-byte** `package/` copy and asserted the lineages were
> "byte-identical (verified: diff = 0)". **That check compared `package/` to `package/`.** The FPGA
> firmware is built from a **third** copy,
> `caplifive-system/sw/buildroot/components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c`,
> **106,935 bytes**, dated Sep 17 01:38 — matching `sbi_capstone_dom.c.S`'s timestamp and reached via
> `sbi_capstone_dom.c`, which is the single line `#include "capstone-sbi/sbi_capstone.c"`. Every line
> number in this entry is now that file's. The refutation itself is lineage-independent (the type
> guard's commit is an ancestor of both lineages); the citations were not.

### R-41 — `revoke_region`'s type-guard early return DROPS the CPMP entry it just read, because `read_cpmp` is destructive and the refusal path never writes back `OPEN, PLAUSIBLE-BUT-UNPROVEN: the read's destructiveness is traced through the RTL and the missing write-back is plain in the source, but the downstream consequence is NOT verified and no run has been made. Found by a claim-auditor while refuting R-40's first mechanism -- so this is a regression in the very guard that blocks that mechanism`

> **Folder:** none; analysis with **R-35**'s. Context: **R-40**.
>
> **What it is.** `revoke_region`'s CPMP arm does `rev = read_cpmp(region_cpmp[region_id]);` and then,
> if the type is not REV, `return 2` — **with no write-back** (`sbi_capstone.c:1669-1675`, FPGA build
> input). But `read_cpmp` is **destructive**: `C_READ_CCSR` expands to `ccsrrw(rd, cpmp(n), x0)`
> (`:16`); `capstone_csr_unit.anvil`'s `ccrw_handle_pmp` returns `3'b110`/`3'b111`, so **bit 2 is set
> unconditionally**; `commit_stage.sv:386` drives `ccsr_we_o = cap_check[2]`, and
> `csr_regfile.sv:2418-2425` writes `cpmp_d[0]`/`cpmp_tag_d[0]` from the `x0` operand. So the read
> *moves* the capability out of the entry and the refusal path leaves it cleared. The pre-guard code
> always wrote back.
>
> **Exposure:** the **direct** `REGION_REVOKE` ecall on a non-REV region — one call, no authorization
> check (R-40), reachable from the 0666 ioctl. Not exposed through the kernel module's release path,
> which pops the region afterwards.
>
> **NOT VERIFIED: the downstream consequence.** Asserting one without a run would repeat exactly the
> over-reach R-40's first version was refuted for. Settling test, no board: issue `REGION_REVOKE(0)`,
> then read `cpmp(0)` back, or make any access into `[MEM_BASE, _cap_text_start)`.
>
> **Why filed despite being unproven:** it is the *same class* the monitor already documents at
> `sbi_capstone.c:1355` — *"REV_SHARED's type test simply fails, silently skipping its `__delin` AND
> its write-back"* — a sibling branch of the same function, so the pattern has bitten this code before.

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

### R-10 — a 16-byte capability copy MANGLES plain scalar data in its high half `OPEN — PARTIALLY FIXED; root cause of C-13, board-confirmed 2026-07-29. The secondary defect is "holds a capability" decided by OR-REDUCING a user/metadata word rather than consulting `cap_type`. The STORE side is REPAIRED (`is_cap_req`/`st_wr_cap` now carry an explicit opcode-derived flag); the REFILL side is LIVE — `wt_dcache_mem.sv:358` and `:501` — on every tree checked including 66c4e7517, 1bfff7776, 4cc068572, dev's pin f6ec6c198 and 9a7bd598c. R-29's sibling account; the stage-8 discriminator is unrun. READ THE SECOND-READING BOX BELOW, NOT THIS LINE ALONE`

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
> **2026-09-15 — THAT PREDICTION CAME TRUE, and the header line is why.** The RTL lane read the two
> repaired sites, traced the explicit `is_cap` chain from issue to the AXI adapter, found it sound,
> and was about to record the secondary defect as FIXED. The refill pair was missed for a reason
> worth naming: the search was `grep is_cap`, and **neither `:358` nor `:501` contains the string
> `is_cap`** — a narrowed view cannot show what it is not keyed to, which is the mirror of sampling
> `is_cap_req` and seeing it behave. The box above was not read first; it says exactly this.
>
> The header line materially helped: it called the defect "LIVE on `66c4e7517`" while naming only the
> two sites this box records as REPAIRED on that very commit, so a reader who checks those two and
> finds them clean concludes the entry is stale. The header now carries the split. **Also checked
> while refuting the claim** — the store-side repair predates `66c4e7517`, and the refill pair is
> present at identical constructs on `66c4e7517`, `1bfff7776`, `4cc068572`, dev's pin `f6ec6c198`
> and the R-34 branch `9a7bd598c`, so no tree separates "current" from this entry's subject.
>
> Two further things the refutation turned up, recorded so the next reader is not misled the same
> way. `wt_dcache_wbuffer.sv:754` does not end where it is usually quoted: `is_cap` is STICKY across
> a merge into an already-capability granule (`| ((|wbuffer_q[wr_ptr].valid) & wbuffer_q[wr_ptr].is_cap)`),
> deliberate, but not what the one-line quote says. And the store side's "value-driven tag" is not an
> independent authority for memory-sourced capabilities: the tag on an LDC result comes from
> `req_port_i.data_rtag` ← `rd_ctag_o` ← `wt_dcache_mem.sv:358`, i.e. **the refill OR-reduce one hop
> earlier**. The inference moved upstream of the tag; it was not removed. The tag ⇔ `cap_type`
> equivalence that would close the gap is asserted only under `ifndef SYNTHESIS`
> (`ex_stage.sv:807-819`), so it is inert on silicon — and that assertion's own comment names the
> failure mode the refill OR-reduce would produce: "some path minted a tag for non-capability bits."
>
> The entry's literal wording also still holds: `grep -rn cap_type core/cache_subsystem/` returns
> nothing. The cache subsystem has never consulted `cap_type`.
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

> # 2026-09-16 — THE REVOKE-WALK SPLICE: built, measured, synthesised. R-12's COST half, not its capacity half.
>
> `r12-splice-revoked-nodes` at `f1331daed` (synthesised at `379248185`). **This addresses the cost of
> revocation, not the 65,532 ceiling** — the two are separate problems and only a reclaimer touches the
> ceiling. Do not read this as R-12 being fixed.
>
> **The defect.** `REVOKE_NODE` re-enters its FSM per visited node and `change_rev_node_validity`
> (`capstone_unit.anvilh:563-565`) preserves `prev`/`next`, so invalidated nodes stay linked and every
> later walk re-reads them — round *r* costs *r+2* dependent 16-byte reads. P1 measured the consequence
> on silicon as release cost rising ~12× within a domain while minting rose ~1.5×.
>
> **The fix is a DEFERRED ONE-SHOT splice at the walk exit, not a per-node unlink.** When the walk
> terminates, the index in hand is the first node outside the revoked subtree and its record is already
> read; everything between it and the revoked node is exactly what this revoke killed. So one splice
> unlinks the whole run in **two writes, independent of run length**, adding no reads — the revoked
> node's record is cached at entry from a read the unit already performed and discarded. A per-node
> unlink would instead add reads and writes *inside* the loop and create a read-after-posted-write on
> two different dcache ports.
>
> **Measured, both trees staleness-gated and the other four generated units confirmed byte-identical:**
>
> | | unspliced | spliced | |
> |---|---|---|---|
> | full sweep | — | **95/95, 0 status changes** | only movement: `excode-base-audit` and `r31-revoke-cursor`, **+4 cycles each** — the two extra writes, on exactly the two tests that revoke |
> | revoke that kills a run | 52 | 56 | +4, flat |
> | **revoke that walks the corpses** | 56 | **45** | **−11 on a four-corpse run; the saving scales, the cost does not** |
> | lint | baseline | **PASS, UNOPTFLAT 40** | no new synthesis hazard |
>
> **Synthesis (exit 0, 1h17m):** WNS **−9.225** against the flashed −12.425, TNS halved, 12,981 fewer
> failing endpoints, +725 LUTs (0.4 %), combinational loops **29 → 13**. Best WNS ever recorded on this
> design; first whose arithmetic reaches 20 MHz. **NOT FLASHED.**
>
> **⚠ THE TIMING GAIN IS NOT ATTRIBUTABLE TO THIS CHANGE'S PURPOSE, and the obvious sentence is false.**
> Unlinking alters what the chain holds *at runtime*; `synth_design` never sees runtime, so a dynamic
> property cannot move a loop count. Read out of the generated RTL: the splice put
> `send ep.rev_res(...)` in **two** branches instead of one, so anvil emitted a **registered endpoint
> selector** (`_ep_rev_res_valid_selector_q`) plus ~9 one-bit scheduling registers.
>
> Confirmed on the **routed netlist**, not just the generated RTL (synth lane, hierarchical
> utilisation): `capstone_rev_node` FFs **606 → 678 (+72)** and LUTs **1,052 → 1,141 (+89)**, while
> design-wide FFs fell **93,145 → 92,939 (−206)**. So the unit gained registers and the design lost
> them — about **278 flops removed outside this unit**, the fingerprint of the loops actually going.
> ~~**Roughly nine one-bit registers removed sixteen loops and bought 3.2 ns.**~~ **THE MIDDLE CLAUSE IS
> RETRACTED 2026-09-17, measured by S2** (see the S2 block below): the registered endpoint bought the
> timing and did NOT touch the loops. A semantically null duplication of the same send, with no
> unlinking, reached **WNS −8.684 with combinational loops still at 29**. The registers are worth more
> than 3.2 ns and zero loops. What survives: *"a change that duplicated a send site caused anvil to
> register that endpoint, and timing improved"* — never *"unlinking revoked nodes improved timing"*.
> The sixteen loops came from the splice commit's STRUCTURAL changes beyond the send duplication (the
> exit splice, the cached `serving_node`/`serving_idx`/`serving_next`, the reworked walk) — **not** from
> the endpoint registration, and not from the runtime fact of unlinking either: the opening warning
> above stands, a dynamic property cannot move a static loop count.
>
> *(Two register figures are in circulation and both are right: **+133 declared flop bits** summed from
> `_q` widths in the generated RTL — what the source asks for — against **+72 implemented flops** from
> the routed report. Synthesis did not map ~61 declared bits, including unused upper bits of the 94-bit
> `serving_node`. Quote the implemented figure against implemented design totals; never mix them.)*
>
> **⚠ AND AN UNEXPLAINED COST THAT BELONGS IN THE SAME BREATH.** Design-wide LUTs **rose** by 731 while
> the unit accounts for only 89 of it — so about **642 LUTs appeared OUTSIDE `capstone_rev_node`**.
> **RETRACTED IN PART 2026-09-17:** these 642 are not intrinsic to registering the endpoint. S2
> registers it and routes to 169,637 Total LUTs — **307 BELOW the splice** and only 424 above the
> flashed base — with `capstone_rev_node` itself smaller than the splice's (1,093 LUT / 607 FF against
> 1,141 / 678). So the 642 belong with the structural half, alongside the loops. Original text, kept
> because the size of the effect is still unexplained: removing sixteen loops evidently let synthesis
> restructure well beyond this unit, trading registers
> for logic somewhere, and **nobody has explained where or why**. Small against 169k, but not nothing
> and not local. Any deliberate application of this lever needs that understood first: a remedy whose
> side effects are unmeasured is an inference with one datum, not a validated technique.
>
> **AND THAT IS A LEAD WORTH MORE THAN THE SPLICE.** It converges with an independent measurement along
> an older build's worst path — 122 hops, 46.010 ns of routing, median 0.373 ns, no tail — a *depth*
> signature whose stated remedy was pipelining. Registering one endpoint is that remedy, arrived at
> accidentally from the other direction, and is **the first empirical confirmation of it** on a design
> with an open clock-rate question. The control that would settle attribution is one synthesis of a
> semantically null duplication of that send site — which must have its generated RTL read back to
> confirm anvil actually emitted the selector, or a collapsed duplicate yields a null result that reads
> as a refutation.
>
> # 2026-09-16 — S1 LADDER: the cost curve, measured on both trees at non-zero memory latency
>
> The table above measured a **four-corpse** run at the testbench's default ZERO latency and reported
> −11 cycles. That was too small a lever to read a curve off. The parameterised ladder
> (`r12-s1-walk-ladder.S`, `NROUNDS` per build) at `S12_MEM_DELAY=12` gives the curve:
>
> | N dead nodes crossed | unspliced | spliced | ratio |
> |---:|---:|---:|---:|
> | 8 | 273 | 328 | 0.8× |
> | 160 | 729 | 328 | 2.2× |
> | 1,024 | 3,321 | 328 | 10.1× |
> | 3,072 | 96,822 | 516 | **187.6×** |
>
> **Unspliced marginal cost is exactly 3.000 cycles per dead node** — `rev2 = 3N + 249` fits exactly at
> every rung from N=8 to N=1,024. **Spliced marginal cost is exactly 0.000** — 328 cycles at six
> different N, identical to the cycle. The splice pays a fixed ~79 cycles to delete a per-node 3;
> **crossover at N ≈ 26**.
>
> **The 3 is an L1-HIT cost and therefore a LOWER BOUND.** The dcache holds exactly 2,048 nodes (32,768 B
> / 128-bit lines, a node being one line); the 3.000 slope holds to 1,024 and breaks at 2,048. One SQLite
> speedtest1 run mints 43,355 nodes — **21× the dcache** — so the workload is entirely in the cold regime
> the simulation only reaches at its last rung.
>
> **The intercept is a FIT PARAMETER, not a fixed cost, and the sub-8 region has a measured cause.**
> Exactly two rungs — N=6 and N=7 — sit exactly +91 cycles above the fit; everything from N=8 up is on
> it at ten exact points. The mechanism is the write-through dcache write buffer
> (`CVA6ConfigWtDcacheWbufDepth = 8`): halving it to 4 moved the +91 pair from {6,7} to **{2,3}**, a
> shift of exactly 4, while every rung from N=8 upward stayed **byte-identical** between the two
> configurations. So the anomalous pair sits at `depth−2`/`depth−1` and the perturbation does not reach
> the fitted range. Quote `249` only as "the intercept of a fit valid for N ≥ 8", never as a fixed cost
> of revocation. The 3.000 slope, the spliced 0.000 and the N ≈ 26 crossover are unaffected.
>
> **The pair is clean by construction:** the control tree's HEAD *is* the merge-base of the two branches,
> and one source file differs. **The flat cost is a working splice, not an early exit** — a `WITNESS`
> build behind `#ifdef` (timed path byte-identical) shows the middle handle's own node going valid → 
> invalid across revoke 2 on **both** trees. Full detail, including the cold rung's internal control:
> `docs/history/16-09-2026_11-54-05_r12-s1-revoke-walk-cost-ladder.md`.
>
> **`drop_req` deliberately NOT spliced:** the walk's already-invalid branch simply advances, so dropped
> nodes sit inside the spliced run and are swept by the next revoke that crosses them. An eager unlink
> would add 2 reads and 2 writes to every DROP — which, unlike the walk, cannot amortise them.

> # 2026-09-16 — R-12's HANG is gone: pool exhaustion is an ARCHITECTURAL FAULT (cause 30). Built and measured on `m1-reclaimer`.
>
> The 65,533rd allocation used to be deliberately never answered — the unit dropped the request rather
> than alias an id, the dynamic unit blocked forever on its `recv`, and the only external signal was a
> debug LED. From software it was indistinguishable from every other wedge. **Now the unit replies with a
> two-bit status and the dynamic unit raises `INSUFFICIENT_SYSTEM_RESOURCES`** — the spec's own cause 30
> (`int-except.adoc:27`, *"up to the implementation where to raise it"*), already in QEMU as
> `RISCV_EXCP_INSUF_RESOURCES`, mapped **by name** in both encoders because no appendable ordinal reaches
> 30 arithmetically. DELIN, which rewrote a node without reading `valid` and so re-broadcast dead
> indices, now refuses with `INVALID_CAPABILITY`.
>
> **Measured, merged tree (splice + R-34/R-24), fixtures with no `CAPENTER`:** `r12-pool-exhaust` —
> **65,532 mints, then cause 30**; base still valid; second mint 30; SPLIT 30; 3 traps; three unit
> dprints. **Positive control on the pre-change tree `c49190d90`:** the same program runs to the
> 3,000,013-cycle ceiling with `tohost = 0`, no exceptions, no prints; its RVFI trace ends with
> `x18 = 65,532` and the loop jump — **the 65,533rd MREV never retires.** Same count, same instruction;
> the pair differs by exactly this change. `r12-delin-dead`: DELIN on a revoked node → cause 25.
>
> **This is R-12's capacity half made VISIBLE, not lifted.** The ceiling is still 65,532; the reclaimer
> (Part A of the v2 spec, in progress on the same branch) is what lifts it. The trap is what makes the
> ceiling *measurable* on a board, and is a prerequisite for the reclaimer regardless, since two of its
> audit corrections require an operation to be refused and no response channel could say no.
>
> **An instrument fact learned here that binds every trap-then-print fixture:** `CAPPRINT` prints at
> *execute* on the FLU; a late DYN-unit exception flushes and re-executes younger prints, so a register
> prints **twice, speculative stale value first. Last print wins.** A first-wins parser reads a clean
> zero.
>
> **THE SECOND DESIGN EXISTS (2026-09-16): `docs/plans/2026-09-16-revnode-reclamation-v2.md`. Start
> there, not from the rejected v1 — but it has been AUDITED TOO and DOES NOT SHIP AS WRITTEN EITHER.**
> v1 was incomplete, having no mechanism; v2 has mechanisms and three of them were individually wrong
> in ways that would have shipped: the answer to the fatal item relied on a broadcast that does not
> occur at reclaim, the allocator never composed the new id (so a recycled slot would be handed out at
> generation 0 and every check would pass vacuously), and the `free` bit was to be set in a helper
> shared with DROP, which would have marked still-LINKED nodes reclaimable. All are corrected in place
> and marked **[AUDIT]**; the verdict banner is at the top of the document. **Read the banner before
> the body.** One audit item is settled by measurement rather than argument: the generation's home bits
> are structurally always zero today, so the round trip had never been exercised — a node written with
> every depth bit set reads back 4294967295, all 32 bits surviving, with the high bits after an
> eviction the stated residual. It answers the reconciliation's six items and adds three the M1
> start gate does not cover — a failure encoding, the id-transplant primitive, and the closed site
> inventory. It supplies the mechanism v1 lacked: an **intrusive FIFO free list threaded through the
> node's own `next`** (zero new storage), with reuse eligibility of **unlinked, not merely invalid**.
> It differs from v1 on where the discriminator lives: v1 spent the slot padding, which the audits
> showed is **not plumbed**; v2 narrows `depth` from 32 bits to 17 and spends the freed bits as
> `generation:14 | free:1`, which **are** plumbed end-to-end (verified: `ex_stage.sv:1149` returns them
> as `data_ruser[29:0]`). The `free` bit exists because "unlinked" is **not** decidable from the node
> record — W1's splice rewrites only the boundary nodes — and it is set in a write the revoke walk
> already performs, so it costs no extra memory traffic. It does **not** claim confinement, does not claim it costs nothing, and does not
> quote a capacity figure. Its fatal item — the S/U boundary that never reads a node — is answered with
> an index-compare recycle broadcast **plus a stated monitor-side software contract**, with the residual
> named rather than hidden. **Still gated:** nothing is scheduled until the lead names the RTL owner and
> approves the algorithm and invariant. A separable Part B turns pool exhaustion from a core hang into a
> reportable fault and can be approved alone.
>
> **TWO AUDITS OF THE RECLAMATION DESIGN, RECONCILED (2026-09-15) — read before proposing a second design.** `docs/history/15-09-2026_20-39-03_revnode-reclamation-two-audits-reconciled.md`. Verdict: the design in `docs/plans/2026-09-14-revnode-reclamation-design.md` is **incomplete, not merely unsafe** — `head` is only ever incremented and there is no free list, reclaim queue or reuse scan anywhere in the unit, so **no index is ever reused and the generation is never consulted**. Separately fatal to the mechanism: the unprivileged boundary (`pmp_data_if.sv:82-97`) decides authority from a cached bit and **never reads a node**, so a split id stops matching the broadcast after the first reuse and revocation silently stops invalidating S/U-mode capabilities — invisible because everything runs at generation 0. The note also records a resolved contradiction between the two audits over the node's field packing (MSB-first; the RTL lane's measurement was wrong and is retracted in the design document), and lists six things a second design must assert. **The direction is not refuted** — the 65,532-lifetime ceiling is real and generation tagging is a sound way to lift it — but this design does not implement it.

> **2026-09-14 (boots sw74 / sw74b): the budget bit a measurement boot, twice, and the read-out is
> unambiguous.** A second full workload run of the Sublet cell (`ceeded2533a74bce`, `main --size 1`)
> in one boot entered and never returned; the wedge tracer read `rev_node_head[15:0] = 0xFFFF` at both
> wedges (sw74 arm 4 with `--stats`, sw74b arm 4 without — the discriminator that also cleared the
> `--stats` path). One Sublet run mints split + mrev = 5,481 + 37,874 = 43,355 nodes; two runs cross the
> 65,535 sentinel mid-workload. Driver rule, now in the state doc: ONE Sublet-cell workload per boot;
> memsys5 cells and the §7 measurement images mint only shares and boundary borrows (sw73 ran
> `--size 100` in one boot), so their second runs are safe. The first run's `--tail` teardown probe
> (a few mints) is not what exhausts it.


> **THE WORKLOAD THIS ENTRY SAYS IS NOT KNOWN, 2026-09-13: nginx's own request traffic crosses the
> threshold at about a thousand HTTP requests.** This entry asks for "a monitor-side split
> counter" and records that "nothing currently approaches the threshold on any known workload".
> The counter is now available from the guest side instead, and the workload does approach it.
>
> `capstone/ports/nginx` replays a recording of a real nginx worker under wrk, made by
> `experiments/a11/nginx` in the paper repository. Only SPLIT and MREV allocate, which this entry
> establishes and which the correction above does not touch, so the count is `split + mrev`:
>
> | replayed pool calls | `split` | `mrev` | allocations | of 65532 |
> |---:|---:|---:|---:|---:|
> | 1 000 | 1 012 | 1 144 | 2 156 | 3.3 % |
> | 10 000 | 10 010 | 12 714 | 22 724 | 34.7 % |
> | 25 000 | 25 012 | 32 002 | 57 014 | **87.0 %** |
>
> 2.28 allocations per pool call, so 65 532 falls at about 28 700 calls. The recording is 28
> records per HTTP request, which puts the threshold at roughly **1 000 requests**. Beside it,
> SQLite's entry glue does ~1 060 splits, 1.6 %, which is why six weeks of SQLite work never met
> this.
>
> The consequence is this entry's own: a deliberate stall, SPLIT blocking forever, a wedge with no
> trap. So a server under the discipline serves about a thousand requests and then stops, without
> a fault to say why.
>
> **What this does NOT establish.** That the port is the cheapest possible in allocations. It takes
> two handles per pool and carves every object, and a design that shared one handle across a
> pool's blocks would spend fewer. The number stands as what THIS discipline costs, which is what
> a paper claims, and not as a floor.
>
> **A SECOND WORKLOAD, 2026-09-25: tshark on the Sublet heap spends 10–13 thousand nodes per run,
> and capstone-qemu itself ran out in one boot.** From the port's heap line (`split + mrev`) for one
> full `-V -n` run: dhcp 12,596; dns_port 13,335; dns-ooo 12,261; http 10,396; arp 9,827; ntp
> 10,055. So on a bitstream without the node reclaimer the budget is about five such runs per boot,
> before the monitor's and the host's own node use.
>
> **capstone-qemu reclaims no node, and so it has the same cumulative per-boot budget.** A node's
> refcount is set to 1 and never changed. Nothing calls `cap_rev_tree_release`: its only caller,
> `cap_rev_tree_update_refcount`, is never called. `cap_rev_tree.h` says "this emulator reuses no
> node". The free-list comment at `cap_rev_tree.c:8-10` describes that dead code. It misled the
> first draft of this measurement, although a 2026-09-15 history note had already found the dead
> code (`docs/history/15-09-2026_19-16-04_manuscript-read-directly-at-last.md`).
>
> In one boot, five full runs spent 64,123 nodes and completed. Everything else up to the
> watermark spent 1,409, ntp's first heap allocations included. ntp then crossed #65,532, and QEMU
> asserted at the pool's end (`cap_rev_tree.c:56: _cap_rev_tree_dup_node_before: Assertion
> new_node != CAP_REV_NODE_ID_NULL`). The port's runner now allows at most four full runs per
> sublet boot (`ports/wireshark/app/results/2026-09-25-qemu-safety-sublet/`).

> **A diagnostic that carries the withdrawn account.** `capstone-qemu/target/riscv/cap_rev_tree.c`
> prints, at cumulative allocation 1022, that "on silicon (10-bit bump head from 3, no
> reclamation) this is where the head WRAPS to 0 and starts reusing LIVE ids". That is the account
> this entry withdrew on 2026-09-10, in a message a reader meets at exactly the moment it will
> mislead them. It misled this measurement for an afternoon. Not changed here, because the
> emulator is a shared submodule and the change is a sentence, not a fix.

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
>   made on 2026-08-27 in `ports/sqlite/build-sqlite-silicon.sh:943-949` against the resident
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

> # 2026-09-16 — THE RECLAIMER: R-12's CAPACITY half, built and PROVEN IN SIMULATION on `m1-reclaimer` (`054cea69b`). FIXED-IN-SIM; not synthesised, not flashed.
>
> Revocation nodes are reclaimed. A node the walk invalidates is pushed on a LIFO free list — folded
> into the write that already happens, so a two-node REVOKE costs **252 cycles before and after** — and
> an allocation pops the list first, handing out **(generation+1, index)** and bumping `head` only when
> the list is empty; a pop costs **6 cycles more than a bump**, a bump-path SPLIT gained 1. Every use of
> a revocation reference now requires `valid && generation == g` at the unit (query, drop, delin, mrev,
> init); the three trackers compare the invalidation broadcast on the index and re-adopt a fresh
> `(g+1, i)` through the 30-bit adopt guard. Design: `docs/plans/2026-09-16-revnode-reclamation-v2.md`
> as corrected by its audit; LIFO first (the spec recommended FIFO), flagged there.
>
> **The approval test is a PAIR: consecutive commits on the same tree with the same fixtures.** On the
> gen-blind control `1ac15c4ef` a retained stale reference **succeeds** against the slot's new owner —
> stale LDC/MREV/SPLIT/DROP all cause 0, the fresh owner's `LCC` reads 0 afterwards (destroyed), a stale
> REVOKE kills a sibling. On `d9620b907` (and identically on the corrected tip `054cea69b`) every one of those arms is **refused (cause 25)**, the fresh owner
> reads 1, the sibling reads 1, and the fresh `(1,3)` still works: 8 traps where the control had 4. Read
> with prints aligned to RVFI retirements (`capstone/tests/capprint-retired.py`); the last-wins parser
> would have hidden 12 of the 29 readings.
>
> **Also measured on `d9620b907` and re-run byte-identical on `054cea69b`:** a DROP'd node is never reissued (bump id, never the dropped index; after
> the walk sweeps it the next mint pops the revoked neighbour as `65539`); the same index reclaimed 64
> times at strictly increasing generation with 7 handles dropped mid-run and none reissued
> (`capstone/tests/freelist-check.py`); the enforcing S/U tracker (`pmp_data_if.sv`) refuses an S-mode
> load through an entry holding `(0,3)` after the **pop's** reclaim write broadcasts index 3 — the
> reading that is 0 without that broadcast — and adopts the fresh `(1,3)` installed over it; pool
> exhaustion 65,532 mints then cause 30, 3 traps -- unchanged from Part B; retirement at production width index 3 reclaimed **16,384 times at generations 0..16383**, last id (16383<<16)|3, then retired permanently; allocation falls back to the bump path and the fresh index enters the same cycle. 0 traps in 16,387 pops; the one-literal variant the retirement bound as ONE literal (14'd16383 -> 14'd3): exactly one generated line differs, `localparam logic[13:0] thread_1_wire$498 = ...` (artifact b2a1a0f15ca0f018 against production b0d5db13cd8a406c), and at NROUNDS=6 it reproduces the production-width shape at 1/4096 the scale in 3,860 cycles: index 3 at generations 0,1,2,3, retired at the bound, then a bump to index 8 and index 8 reclaimed at generations 1 and 2; LCC on the retained gen-0 sliver 0, base 1, 0 traps. It emulates a reduced BOUND, not a reduced field width.
> Reference suite identical to A2 except the one SPLIT-bearing test by +1 cycle; full sweep 65 PASS / 27 TIMEOUT / 3 NOBUILD, **zero status changes**; three passing tests have moved cycles since Part B (+1, +2, +30) and all of it is A3/A4's -- a worktree at 1ac15c4ef reads the same three numbers -- attributed to alloc_slot()'s one-cycle call boundary on the bump path, measured in the sweep's own regime;
> lint LATCH 52 / MULTIDRIVEN 3 / UNOPTFLAT **40** / BLKSEQ 2 / UNDRIVEN 25 unmoved, UNUSEDSIGNAL **733**
> with every delta attributed by (message, bit-range) shape.
>
> **WHAT THE RECLAIMER CANNOT RECLAIM, measured 2026-09-17 and a bound on its practical value.**
> `REVOKE` frees the nodes BELOW the handle; **the handle's own node is never freed** — inherent, since
> the handle stays usable after the revoke and its node must stay live. So a workload leaks one node per
> handle it mints, and the reclaimer can only recycle what a walk sweeps. Measured on the 16,386-round
> fixture (34,821 minted ids, every one printed): distinct indices ever minted is **linear in
> allocations at 0.529**, flat to three digits from 64 allocations to 34,821 — the round allocates two
> nodes and frees one. No saturation at any point. **The consequence for any capacity boundary in the
> node table is that the reclaimer MOVES it by the reciprocal of the leak fraction rather than removing
> it** (here 2,048 lines of D-cache at ~2,048 allocations becomes ~3,870, a factor of 1.89). For a
> workload whose allocations are ALL handles the fraction is 1.0 and the boundary does not move at all,
> with the reclaimer nonetheless working exactly as designed. The coefficient is a property of the
> workload's allocate-to-free ratio and must be measured per workload, never carried across.
>
> **A NODE IS RECLAIMABLE ONLY IN THE WALK THAT INVALIDATES IT** — the second half of the bound, and the
> one that is easy to state backwards. The push lives only in the walk's `node_in.valid == 1'd1` branch;
> the `else` branch, an already-invalid node, advances and does nothing. So "a reclaiming revoke frees
> what it walks" is FALSE as usually said — it frees what it walks **and finds valid**. Measured in
> `r12-recl-drop-not-free.S`: a DROP'd node is crossed by a later revoke and **never reissued**. Anything
> invalidated by DROP, or by an earlier walk that has already spliced it out of the chain, is unreachable
> to every future walk and leaks for the life of the boot. Together with the handle rule above, those two
> are the whole of what bounds the feature.
>
> **THE THREE INVALIDATION SITES, and why a no-DROP workload has a valid fraction of exactly 1.** Only
> three places set `valid = 0`: the walk's push branch (invalidates AND frees); the walk's non-push
> branch (invalidates without freeing — only for `idx <= 2` or a generation at the retirement maximum);
> and DROP, via `change_rev_node_validity`, which preserves prev/next so the node stays linked and is
> never freed. Nothing else. Since the exit splice unlinks the whole revoked run, a previous walk's
> corpses are not in any later walk's path either. **So on a spliced tree a workload that never DROPs
> has every walked node valid, and the fraction of a walk's visits that it frees is 1** — the only
> exceptions being the two sentinels and a retired index. The practical form: measuring that fraction
> tells you nothing, but ASSERTING it is free, and a deviation means a DROP nobody knew about or an
> index that has retired. It also means the leak fraction is decided entirely by the handle rule.
>
> **Cost, confirmed at N = 65,532.** The allocator's call boundary costs **+1 cycle per allocation** —
> measured on one instruction (bump SPLIT 27 → 28, bump MREV 26 → 27) and confirmed at scale by the
> exhaustion fixture, which runs +65,542 cycles against Part B while performing 65,532 allocations
> (residual 10). A *reclaiming* allocation costs a further **+6**, identical at memory delay 12 and 0
> while cold operations differ tenfold, so it is unit-side work on cache hits. The bump-path mint is
> unchanged (the 282-cycle reference mint is byte-identical to A2's). All of it is A3/A4's: A5 adds
> nothing measurable.
>
> **What the pair cannot distinguish, written down:** the PC-capability path (a stale CODE capability
> needs CALL/RETURN to install); `commit_stage`'s post-adopt compare race; the unit-side belt for
> `rev_req` (REVOKE's refusal comes from the preceding query; the dyn unit runs one operation at a
> time); FIFO-only hazards. And the tracker's **per-entry residual** is real and printed: the same stale
> capability installed into a *different* CPMP entry is re-adopted until the next broadcast of its index.
>
> **An audit refuted the first version of this entry, and the correction is the point.** With every
> gate green at `d9620b907`, an adversarial read of the diff found a composed (generation, index) id
> reaching a node's LINK: `change_rev_node_next/prev` do not mask, so the walk read a generation back
> into `revoke_index`, a push carried it into `free_head`, and the allocator's compose adder -- whose
> comment asserts the popped value carries none -- added it twice. The capability still works (the
> caller rebuilds the record from the same wrong value), so the symptom is a generation SKIP, and the
> retirement compare fires only at exactly 16383: a skip can step over it and WRAP, which is the one
> condition under which a stale reference becomes acceptable again. Fixed at both ends in `054cea69b` and
> measured as a pair on `r12-recl-composed-link.S` -- a node at generation 1 reissued as generation 3
> before, generation 2 after, same fixture, same delay, same 3,180 cycles, one differing print. **No
> fixture reached it**: five directed tests, a full sweep, a lint gate and a 16,386-round retirement run
> were all green on RTL carrying it, and the fixture header records the two shapes that miss it.
>
> **Status: FIXED-IN-SIM when written; FIXED AND VALIDATED ON SILICON 2026-09-17** — see the silicon
> block below. Both halves of R-12 are now addressed on hardware: exhaustion is an architectural fault
> rather than a deadlock (Part B), and the 65,532-node ceiling is lifted by reclamation — **bounded by
> the workload's leak fraction, not removed**, which is the qualification that must travel with the
> claim.
>
> **CANONICAL WORDING FOR M1, RULED BY THE PROJECT LEAD 2026-09-18.** The result is to be described as
> **"M1 measured; completion condition four ANSWERED rather than SATISFIED"** — never as "M1 complete".
> Three of M1's four conditions are met on silicon (permitted reuse demonstrated, stale operations never
> regain authority, pressure behaviour classified). The fourth, `occupied + free = usable capacity`, is
> **false by design rather than unmeasured**: this implementation loses nodes deliberately — a revoke
> never frees its own handle, a DROP'd node is never reclaimable, and an index retires permanently after
> 16,384 reuses — so the protocol's partition has no category for them. The shortfall is measurable at
> the exhaustion point and is reported; the equation is not made to balance. Reason for the ruling: a
> reader of "complete" assumes node accounting balances, and a capacity estimate or a paper claim built
> on that assumption would be wrong.
>
> Original status line, kept: Synthesis (S1) requested from the synth lane with the prediction written
> first — WNS no worse than the splice's −9.225, LUTs within +1 % of 169,932, loops ≤ 13; a regression
> past those numbers is a stop, not a note. No bitstream, no reflash (ask-first). History note:
> `docs/history/16-09-2026_20-30-00_m1-reclaimer-built.md`.
>

> # 2026-09-17 — S2, THE NULL-DUPLICATION CONTROL: the timing gain reproduced, the loop drop did NOT. Prediction's dichotomy was incomplete.
>
> `r12-null-dup-control` at `54ac25f97` — the unspliced tree `4cc068572` with the walk's single
> `send ep.rev_res` duplicated into two identical branches of a semantically null `if`. No unlinking, no
> functional change of any kind. Built to decide whether the splice's timing gain came from the
> registered endpoint anvil emits for a two-site send, or from the splice itself. Exit 0, 1h50m35s,
> bitstream written.
>
> | | flashed `1bfff7776` | **S2 `54ac25f97`** | splice `379248185` |
> |---|---|---|---|
> | WNS clk_out1 | −12.425 | **−8.684** | −9.225 |
> | combinational loops | 29 | **29** | **13** |
> | routed Total LUTs | 169,213 | 169,637 | 169,944 |
> | routed FFs | 93,145 | 93,140 | 92,939 |
> | `capstone_rev_node` | 1,052 LUT / 606 FF | 1,093 / 607 | 1,141 / 678 |
>
> **The experiment happened:** the synth lane's own pre-synth check found `_ep_rev_res_valid_selector_q`
> four times in the regenerated RTL (declaration, next-state, reset, clocked update); controls — a
> nonsense pattern 0, `rev_res` 30. Anvil did not collapse the duplicate.
>
> **Reading.** The registered endpoint buys the timing and nothing else. A change that does *nothing*
> reaches the best WNS ever recorded on this design, better than the splice's, while leaving the loop
> count untouched at the flashed base's 29. So the 29 → 13 loop drop is attributable to the splice
> commit's structural changes, and the timing is attributable to a duplication that could be applied to
> any tree. **The prediction written before this build offered two arms — "the splice's 3.2 ns and 29 →
> 13 reappear, so the gain is the endpoint" or "they do not, so the splice's gain is unexplained again"
> — and the outcome was neither.** Recorded because an incomplete dichotomy is the kind of thing that
> gets read as whichever arm it most resembles.
>
> **A measurement fact that came out of this and invalidates a gate, worth more than either.**
> `reports/ariane.utilization.rpt` is written after synthesis and **OVERWRITTEN after routing**, so in
> any archive of a build that routed it holds a POST-ROUTE number. S2 measured the offset directly:
> 171,620 post-synth against 169,637 post-route, a shrink of 1,983 LUTs (−1.16 %). Consequences: the
> 171,497 "highest ever routed" is post-route; the 173,337 of the build that failed to route is
> genuinely post-synth; and an RTL-lane request had proposed killing S1 before implementation if
> post-synth LUTs exceeded 171,497 — **that gate would have killed S2**, which routed to 83.24 % with
> the router converging from 117,715 overlaps to zero. It was disarmed to reporting-only before it
> fired. **And the premise does not survive the correction:** projecting 173,337 through the S2 offset
> gives 171,334 post-route, i.e. the build that FAILED to route projects 163 LUTs BELOW the build that
> routed. LUT count does not discriminate routability on this design; whatever sank `1cb22e30a` is not
> in that quantity, and no corrected threshold should be built on it.
>
> Not flashed. Open and deliberately not pursued: which sixteen loops the splice removed — the artifact
> carries only the count, and enumerating them means opening the retained routed checkpoint. M1 rests on
> none of it.
>

> # 2026-09-17 — S1: THE RECLAIMER SYNTHESISES, and every pre-registered reading is met. Loops at ONE. Attribution deliberately withheld.
>
> `m1-reclaimer` at `054cea69b`, exit 0, **03h43m25s**, synth peak 21.49 GB against a 100 GB ceiling,
> `write_bitstream completed successfully`. Bitstream sha256 `d76d2a36d919d094…`, distinct from the
> flashed `406e12bf…` and from S2's `a309323d…`, so the build carries the change. Not flashed.
>
> | | flashed `1bfff7776` | splice `379248185` | S2 `54ac25f97` | **S1 `054cea69b`** |
> |---|---|---|---|---|
> | WNS clk_out1 (ns) | −12.425 | −9.225 | −8.684 | **−8.307** |
> | TNS | −718,478 | −349,409 | −425,811 | **−312,530** |
> | failing endpoints | 102,508 | 89,527 | 91,461 | 90,379 |
> | combinational loops | 29 | 13 | 29 | **1** |
> | routed Total LUTs | 169,213 | 169,944 | 169,637 | **168,757** |
> | routed FFs | 93,145 | 92,939 | 93,140 | 93,085 |
> | `capstone_rev_node` LUT/FF | 1,052/606 | 1,141/678 | 1,093/607 | 985/740 |
>
> `capstone_dyn_unit` 2,579 LUT / 1,095 FF. Post-synth Total LUTs 170,928 (83.87 %). `LUTLP-1` = 0 in
> `drc_routed.rpt` (a real DRC table: 20 rules, all Warning, `CFGBVS-1` present as control), and independently
> confirmed by `write_bitstream` completing, since `LUTLP-1` is Error-severity and blocks bitgen.
> *Basis corrected 2026-09-24:* the control first stated here, "TIMING-4/6/7 present in the same file", was
> invalid. It was read from `methodology.rpt`, and `LUTLP-1` is a DRC rule that can never appear there. The
> same re-read gives `LUTLP-1` = 0, with `CFGBVS-1` present and a bitstream produced, for every build reported
> this way: `379248185`, `54ac25f97`, `054cea69b`, `f714d2a72`, `247b76896`, `079dc720a`, `f83fe9342`,
> `4ad0df694`.
>
> **Predictions, written before the build and all met:** WNS no worse than −9.225 → **−8.307**, the best
> ever recorded on this design; LUTs at or below 171,497 → met at both stages; loops ≤ 13 → **1**.
>
> **It is a net REDUCTION, not a passed budget.** 168,757 routed is **456 LUTs below the flashed base**,
> 1,187 below the splice and 880 below S2 — while adding an allocator, a free list, a generation check
> and the merge. The RTL lane's "+1 % band" framed the reclaimer as a cost to be bounded and that
> framing was wrong in direction as well as in units.
>
> **⚠ ATTRIBUTION IS WITHHELD ON PURPOSE, and this is the same trap as the splice's retracted mechanism
> sentence.** `m1-reclaimer` is 17 commits and **14 files under `core/`** from the splice — the
> reclaimer AND the `r34-r24-exception-delivery` merge (MMU, PMP, LSU, `csr_regfile.sv`, `riscv_pkg.sv`).
> So *"the reclaimer took loops from 13 to 1"* and any reclaimer-specific timing or area claim are
> **exactly the sentences this evidence does not support**. What the four builds jointly license, and
> no more:
>
> * registering the endpoint buys timing and **not** loops — S2, 29 loops at −8.684;
> * something in the splice commit's **structural** half takes loops 29 → 13;
> * something in the **14-file reclaimer-plus-merge** range takes them 13 → 1. **RESOLVED 2026-09-18 by
>   `f714d2a72`: it was the MERGE. The reclaimer removes no loops at all.**
>
> Which of those 14 files does it is **unmeasured**. `f714d2a72` — the merged baseline, never
> synthesised — is the only build that would split the reclaimer from the merge, and until it exists the
> joint statement is the whole of what may be said. Either no reclaimer-specific synthesis claim is ever
> made, or that build happens; the lead's call, and it gates nothing now.
>
> **Two instrument facts from this build.** (1) The loop line is **singular at 1** — *"There is 1
> combinational loop in the design"* — so a plural-only matcher returns nothing and **reads as zero**,
> the standing no-data-is-not-a-zero failure; it was caught only by the neighbouring latch-loop line.
> Match both forms and fail loudly on no match. (2) The post-synth → routed Total LUTs offset is now
> **N=2 and consistent**: S2 171,620 → 169,637 (−1.16 %), S1 170,928 → 168,757 (−1.27 %). Post-synth
> over-reads routed by roughly 1.2 %; it still does not rescue a LUT ceiling, since the build that
> failed to route projects below the highest that routed either way.
>
> **Wall-clock, recorded because nobody had this number.** S1: ~2h50m synthesis, of which **~100 minutes
> single-threaded in Timing Optimization with the log silent** — the stretch most likely to be misread as
> a hang — then ~45 minutes from `opt_design` to bitstream, the router converging 241,309 overlaps to
> zero in about 20. S2 for scale: 40 minutes synthesis, 1h50m35s end to end.
>

> # 2026-09-17 — THE RECLAIMER ON SILICON: permitted reuse demonstrated, both cost curves FLAT, the capacity knee gone.
>
> **RESOLVED 2026-09-17 by a LABEL-INDEPENDENT argument, after this lane committed the report as fact
> and then had to dispute it.** The run itself is the discriminator:
>
>     R1 m1 end arm=drop stop=target alloc=200000 minted=200031 revoked=200000
>
> Both candidate images carry the same `16'd65535` sentinel, so the pool is 65,532 usable on either — I
> checked that rather than taking it. A run that mints 200,031 nodes from a 65,532-index pool and reaches
> `stop=target` cannot have avoided exhaustion, and exhaustion is not a completed run on EITHER candidate:
> pre-Part-B it is a wedge (measured — the control ran to the 3,000,013-cycle ceiling with `tohost` never
> written), and post-Part-B it traps with cause 30. **So whatever image executed that run reclaims, and
> the deployed one does not.** No label is load-bearing in that argument, which is why it settles what a
> curve comparison could not: a flat curve against a knee is also the shape two different builds produce.
>
> **The reconciliation is a vantage limit, not an error by anyone.** The synthesis lane's "that bitstream
> never left this machine" was their honest belief and is falsified by something they cannot observe — an
> INBOUND `scp` pull performed from the board side, hash-verified as `d76d2a36…` on arrival, then uploaded
> and flashed. A pull leaves no trace at the source.
>
> **⚠ AND THE LIMITATION THIS EXPOSED IS BIGGER THAN THE INCIDENT, because it applies to every board
> result this project has ever published.** The console exposes **no digest of the resident image**:
> `flash_state` carries only `state`, `nv_bitstream_name` and `server_epoch`; the bitstream routes refuse
> GET (405) and have no download endpoint (404). So **"cite a board result by its image hash" is
> UNSATISFIABLE for a resident image on this hardware** — the label is the only identity the console
> offers. The workable form of the rule is therefore three-part, and all three parts existed here:
> hash the artifact BEFORE it leaves the build host; record the console's label; and **put a behavioural
> discriminator in the run that only the intended build can pass**. The third is what actually settled
> this, and it is the one nobody had been asking for.
>
> **The residual, stated rather than closed:** the reuse argument proves *a reclaiming build* ran, and
> three exist on this branch (`1ac15c4ef`, `d9620b907`, `054cea69b`). What pins it to `054cea69b` is the
> pre-upload file hash, not the run. Any claim resting specifically on A6's link masking rather than on
> reclamation in general inherits that gap.
>
> rule is to cite a board result by its image hash and never by its label, and it exists because a label
> has named a different program before. Until the sha256 of the image actually on the board is produced
> and matched against `d76d2a36…`, **every measurement in this block should be read as provisional** —
> if the resident image is not the one assumed, the numbers describe a different design.
>
> Recorded by this lane, which accepted a peer report and committed it as fact without verifying it
> against the primary source. That is the error, independently of which report turns out to be right.
>
> `caplifive_m1_054cea69b.bit` flashed and resident (`nv_bitstream_name` read back). Board boots, monitor
> survives, capability exception delivery live for the first time with the D3 writeback fix holding in
> practice — firmware scanned two-sided before the bake: 1 integer-derived access in capability-mode text
> with the D3 site present, 0 after, positive control 7,115 both times. Measured by the board lane.
>
> **PERMITTED REUSE, ON SILICON, AND IT NEEDED NO NODE-ID READ.** One invocation reached its target at
> **alloc = 200,000, minted = 200,031, without exhausting a 65,532-index pool.** More allocations than the
> pool holds distinct indices cannot have happened without reuse. That is M1's first completion clause,
> which had been unreachable by construction since the study began — the bitstream exposes no node-id
> read, and the way around it was to run PAST the pool and let Part B's exhaustion fault serve as the
> counter. Part B made exhaustion observable rather than fatal; this is what that bought.
>
> | | deployed `1bfff7776` | reclaimer `054cea69b` |
> |---|---|---|
> | `take_cyc/n` | floor 66.7 → 130–218, knee at alloc ~1,792 | **72.1, FLAT to 200,000** |
> | `give_cyc/n` | grew ~12× | **103.2, FLAT to 200,000** |
>
> `give_cyc/n` reads 103.2 at alloc 1,250 and 103.2 at alloc 195,000. **The 12× release growth is gone and
> the capacity knee is gone.** 175.3 cycles per allocation in the brackets, so a full 10C arm is about
> five seconds of board time.
>
> **A PRE-REGISTERED FALSIFIER THAT DID NOT FIRE, AND WHY THAT IS HONEST.** The board lane had written
> "one curve flattens, the other does not; if both flatten or neither, my account is wrong." Both
> flattened. It does not count as a refutation **only because the reason was written down in advance by
> two other lanes**: the reclaimer changes minting, so node-table growth per allocation is the free-list
> MISS rate rather than 1, the invocation's distinct-index set never approaches 2,048 lines, and the
> capacity knee therefore cannot form. A vanished knee is the capacity account's own prediction under
> reclamation. Had that not been on record first, this would have been a falsifier firing and an
> after-the-fact excuse.
>
> **THE COEFFICIENT IS BOUNDED, NOT MEASURED: c < 65,532/200,000 = 0.3277.** And a caution for anyone
> who measures it by running to exhaustion: **RETIREMENT consumes indices exactly as the handle leak
> does**, so `65,532/A` is a COMBINED index-consumption rate. An index retires after 16,384 allocations
> of itself and the free list is LIFO, so with reuse rotating over 1/2/4/16 indices the first retirement
> falls at ~16,384 / 32,768 / 65,536 / 262,144 allocations — the 200,000 pilot is already past it unless
> reuse spreads over sixteen or more. The correction is computable, since retirement is deterministic:
> `handle-leak fraction = (65,532 − retired)/A` with `retired ≈ A/16,384`. Nothing observable separates
> the two on the board — a retirement only makes the next allocation take the bump path, 6 cycles cheaper.
>
> **An instrument retired in the same breath**, by its owner: a "give per walked node" column that divides
> by an assumed walk of `alloc/M1_LIVE`. The splice unlinks the run, so that walk does not exist and the
> column decays as 1/alloc — 63.7 at alloc 32, 0.008 at alloc 195,000. Arithmetic about a denominator
> rather than a measurement. **A normalised figure carries an assumption about its denominator, and a
> change that removes the denominator's referent turns the normalisation into noise without touching the
> numerator.** The honest figure on a spliced bitstream is the flat 103.2.
>

> # 2026-09-18 — THE ATTRIBUTION BUILD: the MERGE removed the loops, the RECLAIMER returned the LUTs. "The reclaimer took loops 13 to 1" is now REFUTED, not merely unsupported.
>
> `f714d2a72`, the merged baseline, synthesised to completion (exit 0, 1h54m47s, bitstream sha256
> `45597add…`). It is the only build that separates the reclaimer from the r34-r24 merge, and it was
> built for exactly that. The split:
>
> | | loops | routed Total LUTs | WNS |
> |---|---|---|---|
> | **merge alone** `379248185`→`f714d2a72` | **13 → 1** | **+1,165** | **+1.350** |
> | **reclaimer alone** `f714d2a72`→`054cea69b` | 1 → 1 | **−2,352** | −0.432 |
>
> | | flashed | splice | S2 null | **merge base** | **S1 tip** |
> |---|---|---|---|---|---|
> | commit | `1bfff7776` | `379248185` | `54ac25f97` | `f714d2a72` | `054cea69b` |
> | WNS clk_out1 (ns) | −12.425 | −9.225 | −8.684 | **−7.875** | −8.307 |
> | TNS | −718,478 | −349,409 | −425,811 | −293,489 | −312,530 |
> | combinational loops | 29 | 13 | 29 | **1** | **1** |
> | routed Total LUTs | 169,213 | 169,944 | 169,637 | 171,109 | **168,757** |
> | routed FFs | 93,145 | 92,939 | 93,140 | 93,006 | 93,085 |
> | max freq (MHz) | 19.07 | 20.31 | 20.54 | **20.89** | 20.70 |
> | `capstone_rev_node` LUT/FF | 1,052/606 | 1,141/678 | 1,093/607 | 1,213/678 | 985/740 |
>
> **THE CLAIM THAT DID NOT GET MADE, AND WHY THAT MATTERED.** S1 came in at loops 1 against the splice's
> 13, and "the reclaimer removed twelve combinational loops" was available, striking, and would have gone
> into a paper. It was withheld on the grounds that 14 `core/` files separated the two builds. **It is now
> refuted rather than unsupported: the merge did it, and the reclaimer removes none.** The withholding was
> not caution for its own sake — the attractive claim was false.
>
> **AND THE GOOD NEWS INVERTS THE OTHER WAY.** S1's net LUT reduction, which looked like the composite's,
> is **entirely the reclaimer's**: the merge ADDS 1,165 and the reclaimer GIVES BACK 2,352, with
> `capstone_rev_node` itself falling 1,213 → 985. The RTL lane's original "+1 % of 169,932, a cost to be
> bounded" was wrong in direction for the third time — the reclaimer is a net return of 2,352 LUTs.
>
> **The reclaimer's true cost is 0.432 ns of WNS**, and the consequence should not be softened: the
> **merge base is the best-timed build this design has ever produced** (−7.875, 20.89 MHz), and the
> resident tip is not. It remains far better than the flashed −12.425, and the merge base is not a
> substitute since it has no reclaimer — but "the tip is the best build" would be false.
>
> **A SECOND DATUM THAT LUT COUNT DOES NOT DISCRIMINATE ROUTABILITY HERE.** This build came in at 173,187
> post-synth (84.98 %) — **150 LUTs BELOW `1cb22e30a`, the only build that has ever failed to route** — and
> it routed. It did so through a pattern neither previous outcome showed: overlaps 272,124 → 102,596 →
> 30,993 → 10,988 → 3,055 → 1,246 → **9,233 → 49,316** → 17,875 → 5,953 → 929 → 53 → 0, clearing at 36
> minutes against ~20 for the others. Successes fell monotonically; the known failure fell monotonically to
> a floor of 46 over five hours; this one oscillated and cleared. Post-synth → routed is −2,078, so that
> offset is now **N=3 and holds at about −1.2 %**.
>
> **The singular loop-line trap nearly bit a second time** — this artifact also reads "There is 1
> combinational loop", so a plural-only matcher would have reported 0 loops for a build that has 1. Matched
> both forms deliberately this time.
>










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
>   is committed with a positive control (its `XFAIL` was removed in `46c53b7b6ae2` and it now passes;
>   note its `bridged_phi_residue` arm does NOT pin the shape live on silicon — see C-32). Q-04's own tail already says *"the compiler side is
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

**What the spec says.** `capstone-academic-spec/parts/cap-man-insn.adoc:33-37`, MOVC:

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

> ## ⚠ THE `memset` RESIDUE IS **NOT REPRODUCED** (2026-09-10, compiler lane) — strike the second-route claim
>
> The entry carried a live residue: a non-zero-fill `memset` was said to reach the unforgeable-constant
> diagnostic because `findOptimalMemOpLowering` guards its i128 avoidance on `Op.isMemcpy()`. **It does
> not.** A 72-run grid — sizes 16 to 256, alignments 1/8/16, default / `-capstone-gp-captable` / `-O0`
> — produced **zero** forge diagnostics. A non-zero `memset` lowers to scalar stores and never selects
> i128.
>
> **The zero is trustworthy because the FIRST positive control was wrong and was caught.** That control
> was a plain `i128` return, which is not capability-typed and cannot forge — so it returned a clean
> zero, and the *control* was failing rather than the subject passing. The zero was only believed after
> an `inttoptr` of a >64-bit constant was shown to still diagnose. This is the exact shape this
> registry keeps recording, caught this time before it became a claim.
>
> **The fix that was planned would have SHIPPED A BUG**, and this lane's audit stopped it. It proposed
> widening all three `isMemcpy` guards. Guard 2 (`CapstoneISelLowering.cpp:25677`) assigns **c128**
> chunks specifically to preserve TAGS from a copy's source; a `memset` has no source and no tags, so
> widening it would have *caused* a 128-bit materialisation — the very forge being chased. Guard 3
> (`:25700`) is the only genuine "never i128, cap the unit at XLen" branch, and widening its predicate
> alone would have **asserted**: its body calls `Op.getSrcAlign()`, which is
> `assert(isMemcpy() && "Must be a memcpy")` at `TargetLowering.h:181-183`, in a Debug build.
>
> **Why nobody had opened the function.** The plan cited the guards at `:26131`, `:26157`, `:26179`.
> Those line numbers were **never valid at any tip** — they come from a stale comment at
> `CapstoneISelLowering.cpp:2921` that cites them. The real guards are at `:25651`, `:25677`, `:25700`.
> A stale in-source citation kept a function unread; **that comment is still there and should be
> fixed.**
>
> **FOR THE LEAD — close the queued "two i64 halves" lowering PERMANENTLY.** It has **no mechanism**:
> there is no ALU write to a capability's upper half, `SCC` asserts a tagged `rs1` so it cannot build
> one from `cnull`, and `CIncOffset` raises on an untagged `rs1` (`CapstoneISelLowering.cpp:8054-8060`,
> which says exactly this in its own comment). Two-halves would silently drop the high half — **worse
> than today's diagnostic**, which is correct and should stand.

> **2026-09-14 (compiler lane, note `docs/history/14-09-2026_21-00-00_b6-i128-select-not-the-blocker.md`):
> the SQLite domain BUILDS AND RUNS at `-O1` and `-O2`.** `MVT::i128` is no longer a legal type
> (`CapstoneISelLowering.cpp:201` registers c128; no `addRegisterClass` for i128), so the `SELECT_CC`
> node cannot form; the `wide_arm` reproducer now hits the forge diagnostic ("Cannot materialize
> arbitrary >64-bit constants as capabilities"), a correct refusal — a constant arm needing more than
> XLen bits is by construction carrying bounds/permission/tag bits and materialising it as two i64
> halves would forge a capability. B6 (the two-halves lowering) is therefore NOT to be written.
> Images: -O0 `6cf8edf637f72063` (692,983,497, reproduces the archived cell ④ exactly), -O1
> `50ca86aa70b3e425` (342,161,746), -O2 `ec061577fb008e18` (336,067,501), all at the oracle, in
> `~/capstone-artifacts/b6-2026-09-14/`. The board lane's -O2 builds of the P1 arms the same evening:
> cell ⑤ (lookaside on, 2 MiB) `d61c8bf784f2bbd1` 330,723,308; cell ⑥ (Sublet) `c506694f9f6f6889`
> 338,496,909 with the -O0 node counts unchanged (5,481 / 37,874); native `b36eb3814c3cefce`
> 240,654,449 (25,122 lookasides). Stakes drop from "backend crash" to "reachable from C is UNRESOLVED
> and not worth further spend".
>
> **Geometry line for the -O0 measurement image (same note):** `code_len 1,483,656 + 8 KiB + dom_data
> 2,701,904 = 4,193,752` of the order-10 ceiling's 4,194,304 — 552 bytes of headroom. Any growth of
> the -O0 arm, instrumentation above all, fails at domain CREATION with no compiler error. -O1 has
> 260,080 bytes of headroom.

### M-1 — domains run with `mtvec = 0`, so a domain fault is an unbreakable loop `OPEN — OURS, FIX FIRST. FIRST MEASURED COST 2026-09-13: it turned S-15's UNEXPECTED_CAP_TYPE into EIGHT HOURS of silent board time on boot sw64, because a fault and a hang are indistinguishable without a trap vector. The argument for fixing it is no longer only a principled one — and 2026-09-13 evening: sw64 and sw66 were the SAME fault as sw65/sw67 (S-15), read as a hang only because mtvec was 0`

> **2026-09-14 (boot sw76): the entry watchdog's LIVE positive control fired.** sw64's image
> (`23da3b126a304585`, the stall at share3) as the last arm: `SHA5` with no `SHA6`, then
> `ENTRY-STALL 781s … no SHA6 for 421s -> Aborting runner` at the configured 420 s, `TERM` to the
> runner, board released, one boot banner in the window (no reset). Until this boot the abort had only
> been replayed against sw64's log. §7q. The bound M-1 costs a measurement boot is now the watchdog's
> `ENTRY_STALL_S`, proven live, not the idle budget.

> **Question for the project lead (2026-09-14, not decided here): should MEASUREMENT images carry the
> trap vector by default now?** Boot sw69 showed the handler is reachable after a real fault on this
> RTL (the pair image `214b300efd169f03` took UNEXPECTED_CAP_TYPE at share3 and returned to the host
> with the packed trap word), and sw72 showed the host-side readback names that trap at the share.
> Against it: the measurement images of record (§7 rows, sw68's 1.194) were all taken with
> `mtvec = 0`, so a vector-carrying image is a different binary from every published row and would
> need its own bridge pair; and a vector that returns a value where the old image wedged converts an
> entry-watchdog abort into a completed-with-a-trap-word run, which the drivers must then refuse to
> read as a result. The instrument (host readback) is landed either way; this is only about the
> default the build ships.

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

> **OBSERVED 2026-09-16 from the musl probes, and it narrows the entry.** A domain that faults on
> its FIRST entry hands control back: the write probe faulted in `hc_write` at first entry, the
> host process saw the register dump, printed its markers and the guest shell continued. A domain
> that faults on RE-ENTRY, after a `domreturn` yield and a later `call_dom`, does not: libc-test's
> `mbc` faulted inside musl's `__simple_malloc` after a yield and the serial log ends at the
> register dump, twice. No `LT-END`, no shell prompt, `alarm()` in the host cannot fire because the
> process is inside the ioctl, and `kill -9` from the guest changes nothing. So the "unbreakable
> loop" is not every fault, it is the resumed-domain fault, which is worth knowing when choosing
> where a probe may fault. Batch runs order known faulters last for this reason
> (`libc-test/quarantine.txt`).

### M-5 — the `REV_BORROWED` re-share path `C_INIT`s a revoke-derived `UNINIT` that cannot satisfy `INIT` on silicon `OPEN — LATENT on silicon, monitor; QEMU-validated only`

> # ⚠ 2026-09-10: FIVE monitor sites, not two — and the widest one is not a re-share at all.
>
> My grep found the two `C_INIT` sites and I said so was not proof of completeness. It was not. The
> RTL lane's auditor found **five** consumers that would meet an UNINIT capability once R-31 lands:
>
> 1. **`split_out_cap` — the widest, and it needs no re-share whatsoever.** A revoked region is left
>    `region_live`, so an unrelated later `create_region` inside its range picks it and SPLITs an
>    UNINIT. Nothing about the revoked region's own lifecycle is involved; an unrelated allocation
>    walks into it.
> 2. **unguarded `__mrev` in REV_DEFAULT.**
> 3. **unguarded `__delin` in `delinearize_region`.**
> 4. **REV_SHARED silently skips its delin** and hands the borrower an UNINIT whose first `LDC` faults.
> 5. **REV_TRANSFERRED's type check is a hard `while(1)` wedge on the FPGA target**, not a trap — so it
>    presents as a hang rather than an error.
>
> **Checked and NOT sites:** `TIGHTEN` accepts UNINIT, the `LCC` base/end queries are fine, the CPMP
> install path cannot tell LINEAR from UNINIT, and the domain switcher never reads a type.
>
> **So the monitor change is materially larger than "fix the two `C_INIT` calls".** It has to cover all
> five, and site 1 in particular means the fix cannot be confined to the share/revoke paths.
>
> **A third QEMU divergence, and it is load-bearing for the security claim.** QEMU's `helper_csrevoke`
> puts the revoked cursor at **END**; the fixed RTL puts it at **BASE**. That is *why* this monitor code
> exists: `C_INIT(r, r, 0)` succeeds on QEMU without any rewrite, so the shortcut was never wrong under
> the emulator anyone develops against. **Consequence: the disclosure R-31 closes on silicon remains
> open on QEMU by that route until QEMU is aligned** — so "R-31 closes the disclosure" is false for the
> *(the "inert LSU" here is the privilege gate plus **R-34**'s dropped delivery, 2026-09-15)*
> emulator even after the RTL fix, which is a second reason beyond the inert-LSU one not to describe it
> that way.

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
>    **⚠ SUPERSEDED — do not act on this paragraph.** Both halves of it are wrong: the spec DOES say
>    (inclusive, `prog-model.adoc:119`, `:289`), and the three arguments for exclusive collapsed to
>    one. A ruling of EXCLUSIVE was given at commit `d71d5007ee05` and withdrawn the same day; that
>    commit is still findable by its subject line and has already been read as live once. The
>    corrected framing and the ruling actually taken are further down this file and in
>    `plans/DECISIONS-WAITING-2026-09-10.md` item 1: **each document keeps its own convention** and
>    one arithmetic form is corrected on each side.
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
> spec exceptions (`capstone-academic-spec` `parts/cap-man-insn.adoc:415-421`: 24 unexpected operand type, 26
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
(`capstone-academic-spec/parts/mem-access-insn.adoc:93,:104`), so **no** store sequence reaches
`cursor > end`, and revoke leaves `cursor = start` on the RTL. The only way to reach the required
state is the `CAPTYPE` debug instruction, which production code does not use.

The path is QEMU-validated — Q-06's null-blk flow exercises it, under an emulator that wants
`cursor == end` (see Q-07, whose accepted set is disjoint from the RTL's) — and **has never run on
the board**. On silicon the first re-share-after-revoke would trap `ILLEGAL_OPERAND_VALUE` in
M-mode.

**What would settle it:** one board boot with a host that borrows a region, revokes it, and
re-shares it. No such host exists yet, which is why this is latent rather than measured. Filed
2026-09-09 by the RTL lane; line numbers verified against the sources named above.



**REPRODUCED UNDER QEMU 2026-09-10, and the gate everyone assumed is the WRONG ONE.** With the held
Q-07 change built into `capstone-qemu/build-q07/` (never into `build/`), the corpus probe
`uninit_init_then_use_ok` **faults with cause 29** — `Illegal operand value`, from `helper_csinit`,
because revoke now leaves the cursor at `base` and `INIT` requires `>= end`:

    [CAPSTONE] domain halted by capability fault: cause = 29, pc = 0x101560264, tval = 0x0

That probe is **M-5 in seven lines**: revoke, then `cap_init`, with no rewrite of the borrower's data
in between — the same shape as `sbi_capstone.c:1196-1197` and `:1340-1341`, in a domain instead of the
monitor, off the board. It is therefore the cheapest available test bed for whichever reclaim shape is
adopted.

**`run-nullblk-all.sh` IS NOT AN M-5 GATE, and its green must not be read as one.** It was predicted
red and came back green; the pre-registered response to a green was to suspect the instrument first,
and that was right. **RETRACTED WITHIN THE HOUR: my first account of WHY was wrong.** I wrote that the
suite never executes a revoke. It does — `null_blk.c` issues `SBI_EXT_CAPSTONE_REGION_REVOKE` in
eleven places, and instrumenting `helper_csrevoke` in the scratch build counts **48 revokes in
`split-io` and 12 in `split-rmmod`**.

**The measured reason, which is more useful than the one I invented:**

    PROBE csrevoke: type_out=0 no_lin_revoked=1 has_write=1 base=101525000 end=101526000

Every one of the 60 revokes returns `type_out = 0` = `CAP_TYPE_LIN`, because `no_lin_revoked = 1` —
**nothing linear was invalidated**, so the LINEAR clause is taken and the handle never becomes UNINIT.
The monitor's `cap_type(r) == 3` guard at `sbi_capstone.c:1196` therefore never fires, and
`helper_csinit` is called **zero** times across all three tests. null_blk shares non-linearly; M-5 is a
**linear-borrow** path.

**So the fill cost in decisions item 2 attaches to LINEAR borrows only, not to every revoke** — 60
revokes here would cost nothing extra. That narrows the affected population and is a real input to
that decision.

Two incidental confirmations from the same instrumentation: `end - base = 0x1000` on every line, which
**measures** the 4096-byte region that the 256-store figure rests on; and `has_write = 1` throughout,
so it is the linearity clause and not the permission clause doing the work here.

It is a valid don't-break-the-split-path control — and it stayed green, which is a real and useful
result — but it cannot reach the condition and so carries no verdict about this entry. The plan's
§5.1 gate list named it as one and is corrected here.

> # ✅ FIXED 2026-09-10 — the lead ruled FILL THEN INITIALISE, and it is implemented and gated.
>
> Monitor `capstone-sbi` at **`a006c63`** (branch `capstone-bootstrap`, all four checkouts in sync).
> Both reclaim sites — `shared_region_annotated`'s `REV_BORROWED` branch and `share_child_region` —
> now fill the region before initialising: one capability store per 16 bytes, which overwrites the
> borrower's data **and** walks the cursor to `end`, so the two are one loop.
>
> **The fill is the security step, not a workaround for a precondition.** Before R-30/R-31 the cursor
> came back already at `end`, so `INIT` succeeded with no rewrite at all. That is what this closes.
>
> **GATES, all green, on the emulator carrying the Q-07 change (`build-q07`) and a monitor rebuilt
> from `a006c63`:**
>
> | gate | result | what it means |
> |---|---|---|
> | **`run-hostcall-all.sh`** | **12/12, 0 failures** | **THE gate.** Twelve probes revoke a region and immediately re-share it as `REV_BORROWED`, which IS the site changed. |
> | `run-linear-uninit-corpus-probe.sh` | **7/7** | the in-domain miniature. `uninit_init_then_use_ok` **failed with cause 29 this morning** on the same emulator without the fill, and passes now — the red-to-green. |
> | `run-smoke.sh` | pass | don't-break-the-monitor control |
> | `run-nullblk-all.sh` | 3/3 | **control only.** Measured: all 60 of its revokes return LINEAR, so it cannot reach this path and its green carries NO verdict here. |
>
> **Two probes were rewritten, because they encoded the OLD cursor placement**, not because the fix
> was wrong. `uninit_init_then_use_ok` now fills before initialising. `uninit_negative_offset_fault`
> read `db[-1]` *because* the cursor sat at `end`, making that address `end-1` and inside the region;
> with the cursor at `base` it addressed `base-1`, outside, and **kept passing while testing the
> opposite of what it claimed**. It now reads a well-inside offset.
>
> **Inert on the currently flashed bitstream** — verified from the RTL, not assumed: `revoke` there
> returns LINEAR for a writable region, so the guard does not fire. This can sit in the tree without
> disturbing anything already measured, and becomes live at the flash.
>
> ~~**Not pushed:** the `capstone-sbi` remote refuses this credential with a 403.~~ **CORRECTED
> 2026-09-11 — it IS pushed.** `git ls-remote capstone-sbi capstone-bootstrap` returns
> `0a5c3d9a3413…`, the same commit. The 403 was real when recorded and the line was then repeated
> for a day without a retry. **Re-try a blocker before restating it, and ask the REMOTE, not the
> local copy of what the remote last said** — counting commits against `origin/HEAD` reads a cached
> ref and a symref that need not name the branch in question, and it agreed with the stale line.
>
> ### THE RECLAIM COUNTER IS A BOARD INSTRUMENT ONLY — and it produced an unreadable zero first
>
> The fill's cost is **bytes per reclaim x reclaims per boot**, and only the first factor is known.
> A counter answers the second on a boot that is happening anyway, so one was added.
>
> **On its first run it reported ZERO reclaims across the whole 12-probe suite** — which read naively
> says the guard never fired and the fill never ran. **It is a fact about visibility, not about the
> guard.** On the emulator target `capstone_uart_flush` is a **no-op** and `capstone_report` goes
> through the trace instruction rather than the UART, so **no monitor marker of any kind reaches
> those logs.** No amount of gate work would have surfaced that; the number simply cannot be read
> there.
>
> **Measured properly instead**, by instrumenting `helper_csrevoke` and running one probe: **three
> revokes, all returning type 3 (UNINIT)** — so the guard fires ~3x per probe, ~36x across the suite,
> and the fill runs every time. Combined with the loop-bound argument above, that closes it from both
> ends.
>
> **Two design consequences, both from the RTL lane:**
>
> 1. **The count is reported UNCONDITIONALLY on every share, not inside the guard.** Inside the guard
>    the tag is *absent* pre-flash (revoke returns LINEAR, guard never fires), so there would be no way
>    to test that the number can be READ until the one boot where it matters. Reported on every share,
>    a **pre-flash boot shows `RCLM:00000000`** — zero reclaims, path proven. **That boot is not a
>    convenience; it is the only chance to test the instrument before the boot that needs it.**
> 2. **A RUNNING count beats a teardown total, and not only for readability.** Truncate the last line
>    and the earlier ones survive, and the sequence says *when* reclaims happened rather than only how
>    many. A single teardown total has no fallback: truncate it and the measurement is gone, flushed
>    or not. The flush is still there — teardown is where output gets cut, which is why
>    `REPORT_REGION_OVERFLOW` flushes too.
>
> **The general form: a counter that cannot be read is worse than no counter, because it produces a
> number rather than silence.**

>
> ### WHY THE GATE MEANS WHAT IT LOOKS LIKE — the loop bound, and a failure mode it forecloses
>
> The RTL lane asked the right question: if the emulator's `revoke` still parked the cursor at `end`,
> a revoked handle would arrive with `cursor == end`, the fill would run **zero iterations**, the
> self-check would pass trivially, the counter would increment having filled nothing, and 12/12 would
> tell us the guard fires while saying nothing at all about the loop. **That gate would be
> indistinguishable from a real one in every artifact.**
>
> **It cannot happen here, and the reason is structural rather than evidential.** The loop bound is
> computed from the BOUNDS, not from the cursor —
> `n = (cap_end(cap) - cap_base(cap)) >> 4`. So a handle arriving with `cursor == end` does not
> produce a short loop; it produces `n` attempted stores whose **first** one violates STC's
> `cursor <= end - 16` and **faults loudly on the first reclaim**. The gates pass, therefore the
> cursor arrived at `base` and the loop ran `n` times on every reclaim the twelve host-call probes
> performed.
>
> Independently: `op_helper.c:937` sets `cursor = base` unconditionally, aligned to the spec and to
> `capstone_dyn_unit.anvil`'s `create_capability(..., rs1.metadata.start)`.
>
> **Consequence for the flash, and it lowers the risk:** the fill loop has already run in anger, many
> times. What is new at the flash is the HARDWARE path — real revoke semantics, real capability stores
> against the memory system, and the five sites going live together — not the firmware logic.

>
> ### THE FLASH BOOT IS THE FIRST TIME ANY OF THIS RUNS ON HARDWARE — order the arms accordingly
>
> All five reclaim sites are **inert on the current bitstream**, so the emulator gates are the only
> evidence there will be until the flash. At the flash, **five newly-live paths become live at once,
> on a bitstream that also changes `revoke` semantics underneath them.** That is more than one unknown
> in one boot, which is exactly what this project's board rules exist to prevent.
>
> **Take them one at a time where the stage set allows:** a known-good control, then the simplest
> revoke-and-reuse path, and only then the SQLite stages, which are the ones that both share the large
> regions and exercise `split_out_cap`. A wedge in the middle of a combined boot makes everything
> after it collateral, and the five sites would not be separable afterwards.
>
> *(RTL lane, 2026-09-10. Not a reason to hold the change — it is about what the first boot can be
> made to conclude.)*
>
> ### A STUCK FILL IS RECOVERABLE — by shrinking `end` DOWN, not by moving the cursor up
>
> The constraint recorded elsewhere is that `STC` is the only instruction that advances an UNINIT
> cursor, and that `CINCOFFSET` and `SCC` both reject UNINIT — from which I concluded a fill loop that
> stops short leaves the capability permanently unusable. **That is wrong, and the RTL lane supplied
> the missing half.**
>
> **`SHRINK` accepts UNINIT.** `capstone_flu_unit.anvil` raises `UNEXPECTED_CAP_TYPE` only for types
> outside `{LINEAR, NONLIN, UNINIT}`; it takes `start` from `rs1.cursor` and `end` from `rs2.cursor`
> and clamps. So shrinking the END DOWN to where the cursor actually reached gives `cursor == end`,
> which `INIT` then accepts. The region is permanently smaller, which is the price.
>
> **THE CAVEAT IS THE LOAD-BEARING HALF: it needs at least one completed store.** `SHRINK` raises
> `ILLEGAL_OPERAND_VALUE` when `rs1.cursor >= rs2.cursor`, so it cannot rescue a capability that made
> **zero** progress. Establish which failure mode you are in before reaching for it — a partial fill
> is recoverable, a fill that never started is not.


### M-6 — `revoke_region` hands `csrevoke` a non-REV capability whenever the region was never shared with a retaining share `FIXED 2026-09-11 (monitor, QEMU-verified); the release path it unblocks then hits M-7`

> **M-6's GUARD SILENTLY DEPENDS ON AN S-06-ERA SPEC DEVIATION, and the coupling is invisible from
> either site.** The fix works by asking `cap_type(rev)` — `LCC` selector 1 — before handing the
> capability to `csrevoke`. That is safe only because the type query is **TOTAL**: it answers for a
> non-capability instead of raising. Verified in the RTL at two places:
> `capstone_unit.anvilh:469-472` never names selector 1 among the invalid multiplexings, and
> `capstone_dyn_unit.anvil:195` conditions the NOT_CAP exception on `zimm != 64'd1`.
>
> That totality is **deliberate and a documented spec deviation**, made as the S-06 enabler so
> software could ask "does this granule hold a capability?" and branch on the answer
> (`capstone_dyn_unit.anvil:171-193`; the answer for a non-capability is 7, free because the result
> path already computes `cap_type - 1`). `cap-man-insn.adoc` still needs amending for it.
>
> **So if anyone ever restores LCC's type query to the spec as written, M-6's guard begins faulting
> on exactly the path it was added to protect.** Neither the monitor source nor the RTL says this
> where the other would see it, which is why it is recorded here.

> **FIXED, ruled "proceed to the pop" by the lead.** `revoke_region` now tests the handle's type on
> **both** arms and returns a third value — `0` revoked, **`2` nothing to revoke**, `-1` refused —
> and `ioctl_release_region` treats `2` as permission to pop. Monitor `2c49c41` (`capstone-sbi`) and
> `f633a11` (`caplifive-sbi`), module in `caplifive-buildroot`. **Neither monitor remote accepts this
> credential (403), so both commits are local-only; the module and the buildroot pointers are
> pushed.**
>
> **Verified by the disappearance of the abort, which is the only positive evidence available here.**
> Before: `helper_csrevoke: Assertion 'rs1_v->val.cap.type == CAP_TYPE_REV' failed`, 3/3 at two sizes.
> After: no assertion, and execution continues past the revoke into the pop. The guard is the only
> thing that changed on that path.
>
> **It does NOT yet prove the `pre_mmap_offset` fix**, because the release path has a second defect
> behind this one — see M-7. `tests/runtime-qemu/offsetcycle` still cannot complete cycle 0, and its
> PASS path has still never executed.



**The defect, stated so a fix covers it.** `revoke_region` (`sbi_capstone.c:1584-1606`) reads the
region's handle as a `__rev` and passes it to `__revoke` **on both of its arms** — `:1595` from the
CPMP slot and `:1600` from `regions[]` — without ever inspecting the capability's type. It is not one
line: which arm runs depends on `region_cpmp[region_id]`, which `cap_env_init` presets to 0/1/2 for
the first three ids (`sbi_capstone_dom.c:19-26`) and which `swap_cpmp` assigns on any access fault
into a region (`:1933`). A guard on one arm fixes nothing.

**The handle is only ever a REV after a RETAINING share.** `create_region` stores what
`split_out_cap(base, len, 1)` returned (`:1196`, `:1210`), and that is LINEAR by construction rather
than by inference: `split_out_cap` ends in `if(linear && ty != CAP_TYPE_LINEAR) capstone_error(...)`
(`:796-798`) and `capstone_error` spins forever (`:190-191`), so the call either returns a LINEAR
capability or never returns. A `REV_DEFAULT` (`:1320`), `REV_BORROWED` (`:1342`) or
`share_child_region` (`:1521`) replaces it with a REV. **`REV_SHARED` does not** — it stores a
delinearised NONLIN at `:1354`, which reaches the same failure. `REV_TRANSFERRED` makes the slot a
hole (`:1388`), so `revoke_region` refuses at `:1590-1592` and is safe.

**Trigger surface: any `SBI_EXT_CAPSTONE_REGION_REVOKE` on a region in one of those states.**
`IOCTL_REGION_REVOKE` reaches it directly through `libcapstone.c:557-563` with no share required.
`IOCTL_REGION_RELEASE` is simply the instance that was observed, because it revokes before it pops.

**What each target does.**

| | |
|---|---|
| QEMU | `assert(rs1_v->val.cap.type == CAP_TYPE_REV)` at `capstone-qemu/target/riscv/op_helper.c:905` — **aborts the emulator**, taking the guest with it |
| silicon | `raise_exception(..., ex_code::UNEXPECTED_CAP_TYPE)` at `capstone-ariane/core/anvil_build/capstone_dyn_unit.anvil:47-48` |

**The silicon outcome is UNRESOLVED and is not "the monitor returns an error" — nothing in the
source does that.** The trap is taken while already inside the monitor's own ecall handler, and
`_cap_trap_entry` (`sbi_capstone.S:13-101`, installed by `sbi_capstone_dom.c:38`) has no
`mstatus.MPP` check: it swaps `cscratch`/`sp` a second time on an M-mode `sp` and dispatches to
`handle_exception`, whose default arm calls `fault_return_from_domain` → `return_from_domain`
(`:1658-1682`) with whatever `caller_dom`/`caller_buf` the last domain call left, into
`DOM_REENTRY_POINT` — `_dom_reentry: j _dom_reentry`, an unconditional infinite loop
(`sbi_capstone.S:4-10`). Which of {recursive trap, wedge in `_dom_reentry`, entry into a stale
domain, corrupted S-mode context} actually happens was not settled by reading, and depends on
whether `ctvec` is taken at all for an M-mode-originated capability fault on this RTL. **No mcause
number is given deliberately:** R-24 records the execute-path encoder as +1 off the spec and
`r24-excode-base` would move it.

`/dev/capstone` is created `.mode = 0666` (`module/capstone.c:553-557`), so the ecall is reachable
from unprivileged userspace. The observed run was root, so this run demonstrates reachability from
userspace, not specifically from an unprivileged one.

**A guard tests `cap_type(r) == 2`, not 3.** `cap_type()` is `__capfield(cap,1)`, which lowers to
`LCC` selector 1 (`capstone-c/src/codegen.rs:1442-1457`, `arch_defs.rs:53-63`), and the RTL's
selector-1 case returns `cap_type - 1` (`capstone_dyn_unit.anvil`) — so software sees LINEAR 0,
NONLIN 1, **REV 2**, UNINIT 3 on both targets, and QEMU's enum is already numbered that way
(`target/riscv/cap.h:27-32`). The raw RTL ordinals in `asm_insn.h:77-83` (REV 3, UNINIT 4) are the
WRITE side, what `CAPTYPE` takes; they are not what this guard would compare against.

**Why it survived, and the residual that makes it new.** `release_region` has exactly one caller in
the tree — `tests/runtime-qemu/offsetcycle/offsetcycle_host.c:80`, added today. Of the seventeen
files that call `revoke_region(`, sixteen share first and the seventeenth only mentions it in a
comment, so no existing caller has ever revoked an unshared region. It is latent by the shape of the
corpus, which is why ~48 instrumented revokes (`:3065`) never met it. Two adjacent records exist and
neither is this one: `design/sqlite-marshalling-feasibility.md:146-148` (2026-06-29) recorded the
same assert from the `REV_SHARED` antecedent and was never filed as an issue; and **M-5's own box at
`:1270` inspected these exact two lines and cleared them** — for the type of the RESULT, which is
the UNINIT question, never for the type of the OPERAND.

**N = 3, and the confirmation is in.** The original run
(`/tmp/capstone/offsetcycle-4194304.log`) plus two deliberate replications at TWO sizes —
`/tmp/capstone/m6-confirm-4194304.log` and `m6-confirm-1048576.log`, both carrying
`helper_csrevoke: Assertion` exactly once. All three abort at cycle 0 having printed the same
`id=9 mmap_offset=33947648`, which is what a defect fixed at create time and independent of size
should look like. The second size is there because a byte-identical repeat of one run confirms the
run, not the mechanism.

**It blocks the `pre_mmap_offset` proof.** `tests/runtime-qemu/offsetcycle` fails against the old
module rather than passing quietly, which is the only reason it is worth a run — and it cannot
complete a single cycle until this guard exists. The fix is deliberately NOT written yet, and for ONE
reason rather than the two first recorded here. **The push argument is withdrawn** — this entry
originally said the monitor remote refuses this credential, which was a day-stale line; `0a5c3d9`
is on `capstone-sbi`, so a monitor commit lands normally. What remains is a design question that is
not a guard: for a never-shared region the monitor holds the only capability, so "refuse the
revoke" and "nothing to revoke, proceed to the pop" are both defensible and the module's own
contract (`module/capstone.c`, the comment above `ioctl_release_region`) does not settle it.

That distinction decides whether the offset proof is unblocked at all. **Refusing** is the minimal
safe fix — it converts an M-mode capability fault into the `-1` that `ioctl_release_region` already
reports — but it leaves `release_region` unable to free a never-shared region, so
`tests/runtime-qemu/offsetcycle` would still exit VOID at its first cycle and the
`pre_mmap_offset` fix would still be unproven. Only the **proceed-to-the-pop** semantics unblock
it, and that is a behavioural change to the region lifecycle rather than a guard. Whoever rules on
it should know the verification hangs on the ruling and not on the code.


### M-7 — the first `create_region` AFTER a `release_region` faults; the release itself is clean `OPEN — PARKED 2026-09-11 with the mechanism NOT established; see "why this is parked"`

> **Not this entry, recorded so it is not mistaken for it (2026-09-14, boot sw77 arm 6):** a REGION_ARENA
> image's SECOND run in a boot stopped after `SQ: C2/mkarena` — the host's second 128 MiB arena
> `create_region` of the boot — with no `release_region` having run (that host never releases its
> arena), no trap latched, rev-node head 289. Filed in §7q as "one REGION_ARENA workload per boot";
> mechanism not established (a second 128 MiB allocation from the 256 MiB CMA area that still holds
> the first is the obvious candidate, N = 1).

> **WHY THIS IS PARKED, so a later reader does not mistake a pause for an oversight.** Nothing in the
> tree depends on it: `release_region` has exactly **one** caller, and that caller is the probe that
> found this. No workload, no board image and no gate reaches the path. It was localised as far as
> one-variable QEMU runs can take it — five of them, below — and the next step needs monitor-side
> instrumentation, which is a firmware change on a component this credential cannot publish.
>
> **What would restart it**, in order of cheapness: (1) a print of `region_cpmp[region_id]` inside
> `revoke_region` would settle which arm ran, which is the one thing the bisection could not reach;
> (2) anything in the tree gaining a second caller of `release_region` — then it stops being latent;
> (3) the region-table asymmetry below being fixed for its own sake, which would likely take this
> with it.
>
> **It does NOT block the flash, the speedtest work, or any board boot.** It blocks exactly one
> thing: proving the `pre_mmap_offset` fix, which is itself now known to be incomplete.

**Newly reachable, not newly created.** `release_region` has exactly one caller in the tree — the
`offsetcycle` probe added the same day — so the pop path had never executed. M-6 was the first defect
on it; this is the second, and it is what the probe now hits instead.

> **⚠ LOCALISED BY BISECTION, and the first filing of this entry put it in the wrong place.** It said
> "after the pop, an access to the region faults", which implied the release path. **The release path
> is clean.** A build of the same source with `-DCYCLES=1` runs create → query → release end to end
> and reports `released id=9 rc=0`; a `-DCYCLES=2` build of that same source completes cycle 0
> identically and then faults in **cycle 1's `create_region`**. So the defect is in creating a region
> after one has been released, not in releasing one.
>
> That also means **M-6's fix is verified end to end** — the whole release path, revoke returning 2
> through pop and `dma_free_pages`, completes and returns 0.
>
> **`pre_mmap_offset` is STILL unproven, and the one-cycle PASS is exactly why to be careful here.**
> With `CYCLES=1` the probe's comparison is `off[0] == off[0]`, which cannot fail, and it duly
> printed `__OFFSETCYCLE_PASSED__` and `leak_bytes=0` having tested nothing about offsets. The
> two-cycle run is the first that could fail, and it never reaches cycle 1's query — so `off[0]` is
> known and `off[1]` has never been observed. The probe now **refuses to compile** below two cycles
> (`#error`, negative-tested in both directions) so that a vacuous pass cannot be produced again.

**What is established, from one QEMU run:**

```
[CAPSTONE] Print = Scalar(0x101c00000)     <- swap_cpmp: badaddr
[CAPSTONE] Print = Scalar(0x9)             <- swap_cpmp: region_n, i.e. the pop DID happen
[CAPSTONE] Print = Scalar(0xdeadbeef)      <- CAPSTONE_ERR_STARTER
[CAPSTONE] Print = Scalar(0x2)             <- CAPSTONE_NO_CPMP_REGION, not a RISC-V cause
[CAPSTONE] Cap mem access requires capability: pc = 800239c4, rs1 = x7
[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x800239e4, badaddr = 0x101c00000
```

An access to `0x101c00000` takes an access fault; `swap_cpmp` finds **no region covering it** and
calls `fault_return_from_domain(CAPSTONE_NO_CPMP_REGION)`; that reaches `return_from_domain`, whose
`*caller_buf = retval` (`sbi_capstone.c:1696`, `fw_jump.elf:0x800239e4`, `sd s0, 0x0(t2)`) faults
because `caller_buf` is not a capability. **The probe creates no domain**, so there is nothing to
return to — this is the privilege-blind recovery path M-6's audit named, now demonstrated rather
than inferred.

**Two defects, and they should not be conflated.**

1. **The recovery path is wrong for a fault with no domain running.** `handle_exception`'s default
   arm (`:2060-2068` on the QEMU target) calls `fault_return_from_domain` unconditionally. That is
   correct for a faulting domain and invalid otherwise, and it converts a diagnosable fault into a
   second fault in the handler.
2. **Why the access was unmapped at all is NOT established**, and the bisection narrows where to
   look without settling it. The fault happens during the create that FOLLOWS a release, at the
   address the previous region occupied (`0x101c00000`), with `region_n` back down to 9. Two
   candidates, neither tested:
   - `pop_region` clears `cpmp_region[region_cpmp[region_i]]` and `region_cpmp[region_i]` but does
     not release the CPMP hardware entry, so bookkeeping and hardware disagree after a pop;
   - the popped range is never returned to the parent it was SPLIT out of, so the next
     `create_region` over the same physical pages — which the kernel hands back, having just freed
     them — finds no live parent covering it, and `split_out_cap`'s UNINIT reclaim (`:706`,
     `C_DO_RECLAIM`) **writes** to a slot whose memory is gone.

   **Both are readings of the code, not measurements, and neither should be cited as the cause.**
   The bisection makes the second more attractive, because it explains why the fault needs a
   *subsequent create* rather than firing during the release.

> ### ⚠ NARROWED AGAIN, and the narrowing exposes why the offset fix cannot work as written
>
> **OBSERVED, five QEMU runs, each a one-variable change from the last:**
>
> | cycle 0 | cycle 1 | outcome |
> |---|---|---|
> | 4 MiB | — (`CYCLES=1`) | completes, `released id=9 rc=0` |
> | 4 MiB | 4 MiB | **faults**, `badaddr = 0x101c00000` |
> | 1 MiB | 1 MiB | **faults**, `badaddr = 0x101600000` |
> | 4 MiB | 1 MiB | **completes both cycles** — `id=9` then `id=10`, offsets 33,947,648 then 39,190,528 |
>
> Two things fall out and neither needed a mechanism. **The faulting address tracks the region
> size**, so it is the region's own base rather than any fixed image address. And **the fault needs
> the second create to ask for the SAME size** — a different size completes, which points at the
> kernel handing back the same physical pages rather than at the release itself.
>
> **The different-size run is the first to reach the offset comparison at all, and it FAILS it:**
> `leak_bytes = 5,242,880`, `__OFFSETCYCLE_LEAKED__`. So `pre_mmap_offset` is still leaking with the
> restore in place.
>
> **Why, READ FROM THE CODE AND MARKED AS SUCH — this is an inference, not a measurement.** The
> region ids give it away: cycle 0 got 9 and cycle 1 got **10**, so the slot was not reused even
> though the release returned 0 ("popped and freed"). `create_region` states that it "needs up to two
> slots (a right-hand fragment, then the region)" and refuses when `region_n + 2 > MAX`;
> `ioctl_release_region` pops exactly **one**. So a create that splits a remainder leaves that
> remainder behind when its region is released, and the table does not return to its prior shape.
> The module then takes its `region_n != m_args.region_id` branch — the one that calls
> `probe_regions()` — rather than the branch that maintains `pre_mmap_offset`.
>
> **Consequence for the `pre_mmap_offset` fix, and it is the useful part:** restoring the offset on
> release is correct and cannot be sufficient. While a create may consume two monitor slots and a
> release pops one, create/release is not idempotent in the region table, and no amount of offset
> bookkeeping in the module makes it so. **The fix as committed is necessary, not complete**, and the
> probe is what showed that rather than an argument.

**N = 5 QEMU runs** — the original, the matched `CYCLES=1` / `CYCLES=2` pair that localised it, and
the size-variation pair that narrowed it further.
Silicon behaviour unknown and not inferred: the FPGA arm of `handle_exception` is a different
`#ifdef` branch with its own reporting.


### R-17 — a ~1.6 MB domain hangs after ANY perturbation of its image `OPEN — NOT ROOT-CAUSED; the title is too broad, see the 2026-09-11 box`

> **⚠ 2026-09-11 — "ANY perturbation" is refuted for at least one image family, by our own boots.**
> The seven-testset speedtest1 image is a **+176,760 byte** perturbation of the three-testset one —
> more globals, a different define set — and it returned on the board across **all seven arms of
> sw56**, and again in sw57 and sw58. The instrumented build is a further +144 bytes and returned
> 8/8 in sw57 and 16/16 in sw58. So the headline claim does not govern this family in the literal
> form it states, and quoting the nine-for-nine as a failure *probability* for a new perturbation is
> not supported. That mistake was made and corrected the same day, in the ruling on whether to add a
> `DOMAIN_BASE_VA` knob to the SQLite build.
>
> **What the entry does still establish is narrower and is the part to use.** Its own
> tested-and-excluded list names *address of the executed code* — excluded because `sqlite3Strlen30`
> sits at the **same** address in both the passing and the hanging build. So a build at a
> **differing** address is **untested, not cleared**, which is the opposite of what an exclusion
> gives you. That is the live reason to be wary of a relink, and it is a different and much weaker
> claim than the title.
>
> **And read past the headline to the residual**, which changes what the entry predicts: the
> mechanism recorded further down is **sporadic wrong `strlen` results at ~3% of calls, not
> length-dependent** — stage 16 calling `strlen` on one literal 128 times and getting 1 back from
> four of them. That reads as a machine-level sporadic fault which sometimes lands fatally, rather
> than as perturbation causing hangs. A sporadic fault landing badly in nine builds of one program
> does not transfer to a relink of another.
>
> Recovered into this entry on 2026-09-11 after a rewrite of `state/current-next-step.md` dropped
> it: it had been recorded only in that file's CURRENT block, and a conservation check over the
> rewrite showed it existed nowhere else. A correction to a registry entry belongs in the registry.

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

### R-11 — RTL truncates a capability TOP past a 2 MiB window; QEMU never does `OPEN, not yet hit — and "QEMU never does" is NOT corroboration: QEMU has no `cursorless` encoding at all (zero occurrences in capstone-qemu/target/riscv/), so it cannot exhibit the branch this entry is about. See the 2026-09-15 box`

> **RUN 2026-09-10, with the positive control the earlier attempt lacked. Still NOT HIT, and now that
> statement means something.** The 2026-09 sweep logged `check-repr.py` as NOT RUN; an audit then ran it
> on a stale corpus where it returned clean **without executing its detection branch at all**, which is
> a pass that carries no information.
>
> **What was run.** The checker over the current domain set — the acceptance images, the R-29 ladder
> variants, `lsugate`, and the SQLite silicon image — all **OK**, exit 0.
>
> **Why that alone still proves nothing, and it is worth stating.** The truncation branch is gated on
> `tot > WINDOW` (2 MiB). Every small domain reports `tot = 131072`. **The SQLite image reports
> `tot = 2097152` — exactly the window, so `>` is false and the branch does not run.** The largest real
> domain we ship sits precisely at the cliff edge, one doubling from exercising this at all.
>
> **The branch DOES execute on a large enough image.** `pass1.dom` (8.6 MB, `tot = 16777216`) takes it
> and reports OK **with the checker's own fragility note**: *"past the 2 MiB window — exact only because
> every top happens to be aligned; treat as fragile."* So the branch runs and finds nothing, rather than
> not running.
>
> **POSITIVE CONTROL on the detector itself.** `cursorless_top_exact` was driven directly: with the
> region spanning bit 21 and above and an unaligned top it **REJECTS** — `[0x0,0x200001)` E=1 losing 1
> byte, and eight more at wider spans — while the matched control at the same span with an aligned top
> **ACCEPTS**. The discriminator is real; it is not a function that can only return true.
>
> **Status therefore: NOT HIT, on a check now shown able to fire.** The condition needs a domain past
> 2 MiB whose carve produces an unaligned top. Nothing we ship is past 2 MiB; SQLite is AT it. **The
> trigger for re-running this is a domain image crossing 2 MiB**, and the SQLite corpus work is the
> likeliest thing to do it — which is what "not yet hit" should be read to mean.
>
> *(Method note: the earlier clean run is the exact shape this registry keeps paying for — a check that
> returns OK because its detection code never executed. Running it is not the same as exercising it.)*

> # 2026-09-15 — "QEMU NEVER DOES" IS AN ABSENCE OF THE FEATURE, NOT AN INDEPENDENT WITNESS
>
> This entry's title contrasts the RTL against QEMU. That contrast does not carry the weight it reads
> as carrying, because **the two do not implement the same encoding.**
>
> `grep -rn cursorless capstone/capstone-qemu/target/riscv/` returns **zero hits**. The RTL has a whole
> cursorless branch — `ariane_pkg.sv:625` `bounds_cursorless_t`, `:633` the `cursorless` flag, and
> `:666-670` the branch that reconstructs `base` from it, which is precisely the path this entry is
> about. So QEMU "never does" the truncation because **QEMU cannot**: the encoding in which it happens
> does not exist in the model. That is not a reference implementation disagreeing with the RTL; it is a
> reference implementation that does not model the field.
>
> **The two compressed layouts differ from bit 27 upward, by exactly one**, and this was found while
> auditing something else:
>
> | field | QEMU (`cap_compress.c:30-37`) | RTL (`ariane_pkg.sv:630-638`) |
> |---|---|---|
> | bounds | bits 0-26, **27 bits** | bits 0-27, **28 bits** (the extra bit is `cursorless`) |
> | type | 27-29 | 28-30 |
> | perm | 30-32 | 31-33 |
> | revnode_id | 33-63, **31 bits** | 34-63, **30 bits** |
>
> `verif/tests/custom/capstone/asm_insn.h:64`'s `NODE_ID_INVALID = ((-1) & ((1 << 31) - 1))` follows
> QEMU's 31 bits, not the RTL's 30. The two QEMU copies are byte-identical, so this is model-versus-RTL
> and not copy-versus-copy.
>
> **No artifact was found in which the two exchange a compressed metadata word, so present harm is
> UNRESOLVED** — this is recorded before someone builds on it rather than after. The check that would
> settle it: does any differential test compare a compressed metadata word, or a `CAPNODE` result,
> across QEMU and silicon? If one does, it has been comparing misaligned fields.
>
> **Two consequences worth carrying beyond this entry.** Any claim of the form "the emulator does not
> show X" is weak evidence about capability *bounds* behaviour until it is known whether the emulator
> models the encoding X lives in. And R-33's containment reasoning is unaffected — that rests on sizes
> being exactly representable, which is an allocator property and holds on both — but the general
> emulator-versus-silicon caveat that the safety-table decision turns on is sharper than it looked.

> # 2026-09-15 — THE STATED TRIGGER IS ONE DOUBLING TOO LOW, AND R-33's FIX CLOSES THIS ENTRY TOO
>
> **Re-running at a 2–4 MiB image will produce another uninformative OK**, which is the same shape the
> method note above warns about — one level further in. 2 MiB is where the truncation BRANCH starts
> executing (`tot > WINDOW`); it is not where the branch can FIND anything. The granule is
> `2^(E+3)` with `E = bit_length(len) − 13`, so:
>
> | region | granule | 4 KiB `PAGE_ALIGN` covers it |
> |---|---|---|
> | 1 MiB | 2048 B | yes |
> | **2 MiB** | **4096 B** | **yes — exactly at the edge; the branch runs and must report OK** |
> | 3 MiB | 4096 B | yes |
> | **4 MiB** | **8192 B** | **NO — the first size where an unaligned top is possible** |
> | 8 MiB | 16384 B | no |
>
> Below 4 MiB every page-aligned carve is granule-aligned by construction, so the top is exact and the
> checker cannot fire however large the image is within that range. **The informative threshold is
> 4 MiB, not 2 MiB.** That also explains the `pass1.dom` reading above without appealing to luck: at
> 8.6 MiB the branch both executes and could fire, and its "exact only because every top happens to be
> aligned" note is the checker observing page alignment, not a coincidence.
>
> **Cross-check, and the reason this is recorded rather than merely reasoned:** the 4 MiB edge derived
> here from the granule formula is the same edge R-33 states independently from its own containment
> analysis ("contained by the kernel's `PAGE_ALIGN` below 4 MiB and NOT contained at or above it").
> Two separate derivations landing on the same number is what makes this worth acting on.
>
> **So R-11 and R-33 are one contract seen from two branches of `compress_bounds`** — R-11 the
> cursorless branch losing an unaligned top, R-33 the other branch widening both ends for a
> non-representable size — and **R-33's fix closes both**: rounding region sizes up to the
> representability granule at creation makes tops granule-aligned at any size, which is precisely the
> condition R-11 needs and cannot otherwise guarantee above 4 MiB. Anyone about to work R-11
> separately should do R-33's allocator fix instead and then re-run `check-repr.py` on a 4 MiB+ image
> as its verification.

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

### M-8 — the PostgreSQL test images (`my_first_domain/start.S` glue) do not ENTER on the board monitor: `delin gp` at image+0x44 traps UNEXPECTED_OPERAND_TYPE because the image declares NO globals boundary, so the monitor delivers NO gp `OPEN — the port's fix (link with a `.capstone_gp_initdesc`, or drop the delin); found by E1 boot sw78 r1b3, 2026-09-14; mechanism corrected from source 2026-09-15`

`pg_subpool_test.dom` (`91d23730153d6a18`, r1b3 arm 4) trapped at image+0x44, the first instruction
of `test` after `_start` (`my_first_domain/start.S`): `delin gp`, mcause 25 UNEXPECTED_OPERAND_TYPE,
before any scenario. `pg_hierarchy.dom` (`7284f2f8db444e87`, r1b4, 2026-09-14 23:5x): the same trap,
mepc image+0x44, cause 25 — confirmed on a second image. Cells recorded `unsupported`, reason
implementation-unavailable. QEMU passes both images (GOOD).

**Mechanism, corrected 2026-09-15 from source (the first write-up said "gp arrives already
non-linear, the S-15 class" — an analogy, and the cause number refutes it: a capability of the
wrong TYPE raises 27 UNEXPECTED_CAPABILITY_TYPE, which is what S-15 produced; 25 is raised when the
operand is NOT A CAPABILITY, `capstone_unit.anvilh` / R-31's oracle reading).** The PostgreSQL
images have no `.capstone_gp_initdesc` section (`llvm-readelf -S`: `pg_subpool_test.dom` and
`pg_hierarchy.dom` carry only `.gct`; `ngx-uaf-p1.dom` and `sqlite_silicon.dom` carry both). The
loader (`libcapstone.c:292-296`) then packs `globals_off = 0`; the monitor
(`sbi_capstone.c:928-934`, the 2026-09-07 fix that removed the 0x1000 fallback) takes `gpoff = 0`,
and carves no gp at all for `gpoff == 0` (`sbi_capstone.c:1123-1126`, "0 means THE IMAGE DECLARES
NO GLOBALS REGION"), so nothing is stored in the cscratch top slot. The glue's `delin gp` at the
head of `test` (`start.S:41`) therefore operates on a register that holds no capability → 25. The
nginx and SQLite images are linked by `link-gpfree.ld`, which places the descriptor first in the
globals region, so they enter. `libcapstone.c:235-237`'s comment ("absent section => 0 => the
monitor keeps its historical 0x1000 default") describes the fallback the monitor no longer has and
is stale. Why the emulator enters the same image is NOT settled here: QEMU's `helper_csdelin`
(`op_helper.c:1192`, `assert(rd_v->tag)`) would abort on an untagged gp, so under QEMU gp is
tagged at that point from some other source — a register-state difference at domain entry, not
chased (the cells are `unsupported` either way). Fix is the port's: link the PostgreSQL domain
with the gp-free/cap-table glue and linker script the nginx port uses (`DOMAIN_BASE_VA` already
reaches `pgdom_link`), or drop the `delin`.

### M-9 — a linear (REV_BORROWED) region released while TOP-OF-STACK is popped, and the next `create_region` in the same boot faults inside the monitor `OPEN — emulator evidence (chain18 run 2, 2026-09-15); workaround: keep a region above it`

Found while chaining harness invocations through `sqlite_host_rr.user --speedtest1 --arena N` in one
emulator boot (the R1 harness, `capstone/sublet/r1`). Invocation 1 shares its 4 MiB arena
`REV_BORROWED` (linear to the domain), runs, returns; the host's `release_region(pool)` returns **0**
— the pool was top-of-stack, so `ioctl_release_region` revoked AND popped it (the SQLite host's
comment at `sqlite_host.c:161-168` documents the two return values: 0 = popped and gone, 1 = revoked
and the slot kept). Invocation 2 then dies at its own `F2/mkregion3`: `[CAPSTONE] Cap mem access
requires capability: pc = 800239e4, rs1 = x7, imm = 0, value = 0`, `domain halted by capability
fault: cause = 24, pc = 0x800239e4, tval = 0x0, badaddr = 0xf0300000` — the pc is the MONITOR's
(`pc_cap = C(8002006c [8001a1d0,8002ff10))`), the bad address is the NEW region's base, and the
operand the monitor used to touch it was the integer 0: the slot the pop freed hands the next
create a null where a capability should be. The same chain with a 64 KiB `--tables` region above
each pool (the SQLite cells' shape) makes every release return 1 (revoked, slot kept) and the next
create works — 14 invocations in a row (chain18 run 4), until the region table runs out.
Consequences: (1) every study that carves fresh regions per invocation must keep a region above the
linear one it releases, or the boot ends at the second invocation; (2) because the kept slots are
never reclaimed within a boot, the boot's region budget is finite: ~14 region-bearing invocations of
four regions each (metadata, payload, pool, tables) before `create_region` fails (`MAX_REGION_N 96`
in the module, `modcapstone/module/capstone.c:43`; the monitor's own table is smaller, the failures
started at the 15th invocation = 56 + the controls' regions). Board drivers cap it at 12. Not yet
reproduced on silicon (the boards ran the SQLite shape, which never pops a pool). The faulting
instruction in the emulator's monitor (`caplifive-buildroot/build/images/fw_jump.elf`, the same
bytes as the opensbi-custom generic build, `bb0d91c4ef20010f`): `800239e4: sd s0, 0x0(t2)` after
`ldc t2, 0xc0(gp); ldc t2, 0x0(t2)` — a store through a capability read from a table the monitor
keeps at `gp+0xc0`, whose entry read back as the integer 0; `nm` places it at `_return_from_domain.1
+ 0xa`. The source site is unread — next, with the RTL oracle. The SQLite cells read
`released pool rc=1 / released tables rc=1` on both platforms (sw79; c6o2-2mib), so they never took
the pop path.

### M-10 — the emulator console truncates a guest command longer than its line buffer, silently; the run looks complete and is not `OPEN — procedure; chain18 run 3, 2026-09-15`

`run-domain-smoke.py --guest-command '<text>'` types the text at the guest shell. A chained
command of 18 harness invocations (~3.4 KB) ran five of them and stopped with no error, no marker
and smoke rc 1: the shell received the line cut at the console's buffer, the cut fell inside the
sixth invocation, and everything after it was never typed. Twelve `echo R1_INVOKE` lines were
visible in the transcript (the console echoes what it received), which is what made it look like a
run that had started everything. Rule: a guest command of more than a few hundred bytes goes into a
script file in the share directory (`sh /mnt/host/run.sh`), and the run's completeness is proven by
counting the per-invocation return markers against the list, never by the presence of the command
text in the log. The board path (`run_sqlite_stages_fpga.py`) sends one stage string per invocation
and is not affected by this shape, but its stage strings are typed the same way — keep each under a
few hundred bytes.

### M-11 — the board drivers' summaries lose a result line split across two UART chunks (or by a monitor marker), so a present result reads as absent — and the control rung's own line is not exempt `GATED 2026-09-15 — every driver summary and the watchdog now read through fpga_driver/transcript.py (positive control: test_transcript.py); audited over all 85 archived transcripts: no recorded verdict or cited number was affected`

The console delivers the UART in chunks and the driver frames each as `[fpga] [uart] '...'`; a domain's
line that straddles two chunks is two lines in `driver.log`, and the monitor's share markers
(`ECSA:00000004` ...) land mid-token besides. A summary that greps the framed log line by line then
prints `0x []` for a present result (arm C's `sublet:` counters, sw8x-optC2), "no result line" for a
present mark (E1 r3b3's subpool test, 0D3E04 in the transcript), or a TRUNCATED number as the value
(`SPEEDTEST1-CYCLES 8562` in sw8x-b80s-O2's summary for 1,166,594,074). In five boots the control's
`RESULT k800 retval=4` itself was readable only after joining (sw55, sw65, sw74, sw74b, sw78 r3b4): a
summary-only reader calls those boots VOID.

> **GATED 2026-09-15 (afternoon).** The join is now one module, `capstone/tests/rtl-smoke/fpga_driver/transcript.py`:
> every `[fpga] [uart] <repr>` frame through `ast.literal_eval` (the archive holds 2,107 double-quoted frames the
> single-quote extractors silently dropped, and frames of one line are routinely separated by `[event]` lines,
> which an adjacency-only seam join misses), scoped after this run's `monitor load_image`, joined, the markers
> deleted for domain lines and KEPT for the marker rows and the stall marker, per-arm segments on the
> `[stages] --> TEST` lines, full-match `find_all`, and `require` so that no data is an error. Its positive
> control `test_transcript.py` asserts the OLD reading wrong and the new one right on a synthetic log
> (a control split with an event line between the chunks, a double-quoted frame, a marker mid-token in a cycle
> count, a SHA6 split across chunks, a mark split after four digits → the old reading `00051D`). The six
> drivers' summaries (`board-c6var/b80s/b80a/b78-w2h/r1/r1e4.sh`) read through it, the per-arm mark regex is
> anchored on its line terminator so a cut number can never print as a mark, the marker is `refused`/`failed`/
> `done` by the runner's status and the driver exits non-zero on the first two, and `board-watchdog.sh` reads
> its last SHA5/SHA6 marker through the module's CLI (a split SHA6 read as a stall before). Replayed over the
> archive: the only deltas are the predicted recoveries (arm C's `sublet:` row, sw8x-b80s-O2's 8,562 →
> 1,166,594,074, E1 r3b3's subpool mark, r3b4's control, R1's banner count, R1 boot 1's false `bad`), and a
> refused launch now prints `ERROR: no UART after this run's load_image` instead of a table of zeros. Two
> findings on the way: `boot.txt` (PROBE_SCOPED_OUT) carries no boot banner at all, so the R1 drivers' "must be
> 1" line was a structural zero on every boot; and `ENTRY-STALL` is written to `watchdog.log` only, so a
> summary grepping `driver.log` for it can never fire. The bundle generators import the module too.
>
**Rule:** the transcript is the record and a driver's summary is a view of it; before any line-based
read, join the seams (`re.sub(r"'\n\[fpga\] \[uart\] '", "", log)`), unescape the newlines, delete
the markers (`[A-Z0-9]{4}:[0-9A-F]{8}\n?`), then parse — as `capstone/sublet/r1/r1-bundle.py` and
`capstone/ports/nginx/e1-bundle/e1-bundle.py` do. A summary's `0x []` or "no result" is a prompt to read
the transcript, never a verdict. Audit of the archive (2026-09-15 08:10, §7r): no recorded verdict
changes and no cited number is a truncated fragment. Related: M-10 (the emulator console's silent
truncation of a long command), the transcript-marker note in the board-run skill.

### C-59 — `isValidInsnFormat` is defined non-`static` in BOTH the RISCV and the Capstone asm parser, so a static build of LLVM does not link `OPEN — PARTIALLY FIXED. The Capstone copy of isValidInsnFormat is static as of da5e88488080 (branch compiler/c59-odr, efe9b957d538), which removes the one collision that was actually observed. It is ONE OF SIXTEEN: a BUILD_SHARED_LIBS=OFF link still fails, with fifteen errors instead of sixteen`

> **Scope, measured 2026-09-25 by the compiler lane.** Every strong (T/D/B) defined symbol in both
> target trees was enumerated and intersected, and each pair was reproduced with `ld -r`. **Sixteen
> symbols collide** between `llvm/lib/Target/Capstone` and `llvm/lib/Target/RISCV`, not one:
>
> - twelve `isTune*Fusion`: ADDILoad, ADDLoad, AUIPCADDI, AUIPCLoad, BFExt, LDADD, LUIADDI,
>   LUILoad, SHXADDLoad, ShiftedZExtW, ZExtH, ZExtW. They are emitted into `*GenMacroFusion.inc`,
>   which is **tablegen output, so not a one-word `static` fix**;
> - three that are a one-word fix: `getPredicatedOpcode` (`CapstoneInstrInfo.cpp`),
>   `isRegImmLoadOrStore` (`CapstoneISelDAGToDAG.cpp`) and `PreferredLandingPadLabel`
>   (`CapstoneIndirectBranchTracking.cpp`).
>
> Those objects link into any tool with both targets enabled, and they are more central than the
> AsmParser object whose collision was the one observed. **The entry's original premise, one symbol,
> was incomplete when filed.** Nothing here means "the static build is fixed".

**What happens.** `llvm/lib/Target/Capstone/AsmParser/CapstoneAsmParser.cpp:3476` and
`llvm/lib/Target/RISCV/AsmParser/RISCVAsmParser.cpp:3346` each define
`bool isValidInsnFormat(StringRef, const MCSubtargetInfo &)` at external linkage. The Capstone
target began as a copy of RISCV. A static link (the LLVM default) of any tool that pulls in both
parsers fails:
`multiple definition of 'isValidInsnFormat(llvm::StringRef, llvm::MCSubtargetInfo const&)'`.
This was first seen linking `llvm-ar` in a fresh Release+Assertions build on origin/dev.

**Why nobody saw it.** The shared tree builds with `BUILD_SHARED_LIBS=ON`. Each target library is
its own `.so`, so the duplicate never meets itself at link time, and every lane's gate uses that
configuration. It is still an ODR violation: which definition a caller binds to is up to the
dynamic linker.

**Fix shape (not done).** Give the Capstone copy internal linkage (`static`, or an anonymous
namespace), or rename it. That is a one-line change, but it is in the compiler lane's area.
Verify it with a static build of `llvm-ar`, which is exactly what fails today.

### C-60 — any `-fstack-protector*` asserts in "Insert stack protectors" on capstone64, because `llvm.stackprotector` is not address-space overloaded `OPEN — COMPILER, crash; found 2026-09-24 by the tshark port; root cause from source (mechanism corrected the same day: the SLOT mismatches, not the guard); reproduced with the tree's clang`

**What happens.** `-fstack-protector`, `-fstack-protector-strong` and `-fstack-protector-all` abort
with `Calling a function with a bad signature!` (`llvm/lib/IR/Instructions.cpp:761`) in the
`Insert stack protectors` pass. `-fno-stack-protector` compiles.

    printf 'void use(char *);\nvoid f(void) { char buf[64]; use(buf); }\n' > ssp.c
    clang -target capstone64-unknown-elf -ffreestanding -O2 -fstack-protector-strong -c ssp.c

Reproduced on 2026-09-24 with `llvm/cmake-build-debug/bin/clang`:

- `-fstack-protector-strong` exits 1 with the assertion;
- `-fno-stack-protector` exits 0;
- `int main(void){return 0;}` under `-fstack-protector-strong` exits 0.

**Root cause. The mechanism was CORRECTED the same day.** The first version of this entry said both
arguments were AS200 and that the guard came from `@__stack_chk_guard`. That is wrong. Exactly ONE
argument mismatches, and it is the slot. Checked against source on origin/dev:

- `CreatePrologue` (`llvm/lib/CodeGen/StackProtector.cpp:562-566`) allocates the guard slot and
  passes it to `llvm.stackprotector`.
- The alloca's result type is `ptr addrspace(200)`, because the datalayout sets the alloca address
  space to 200 (`-A200` in `clang/lib/Basic/Targets/Capstone.h:249`). The emitted IR shows
  `alloca ..., addrspace(200)`.
- `llvm.stackprotector` is declared `[llvm_ptr_ty, llvm_ptr_ty]` (`llvm/include/llvm/IR/Intrinsics.td:937`),
  which is AS0 and not overloaded. **The slot operand mismatches.**
- The guard VALUE is fine. `CapstoneTargetLowering::getIRStackGuard` returns a value only for Fuchsia,
  Android or `-mstack-protector-guard=tls`. Otherwise it falls through (`CapstoneISelLowering.cpp:26051`)
  to `TargetLoweringBase::getIRStackGuard` (`TargetLoweringBase.cpp:2049`), which returns `nullptr`
  for everything but OpenBSD.
- With a null guard, `getStackGuard` sets `SupportsSelectionDAGSP`, calls `insertSSPDeclarations`,
  and emits `llvm.stackguard`. That intrinsic is declared `[llvm_ptr_ty]` (`Intrinsics.td:938`), so its
  result is AS0 and matches.

`StackProtector.cpp` never calls `getAllocaAddrSpace`. This is an upstream assumption that capstone64
violates, and no in-tree target with a non-zero alloca address space enables SSP.

**Why it escapes configure and CMake probes.** A protector is inserted only for a function with
something to protect, so a flag probe compiles cleanly and the flag is enabled build-wide. The
peer's measurements:

| Probe | Result |
|---|---|
| `int main(void){return 0;}` | compiles |
| an unused local array | compiles |
| `char buf[64]; use(buf);` | crashes |
| `int x; use((char*)&x);` | crashes |

The last row shows an address-taken scalar is enough, so this is not array-specific. Wireshark's
CMake adds `-fstack-protector-strong` whenever the compiler accepts it, and 82 tshark files failed.
Any autotools or CMake port that probes the flag hits the same trap.

**Workaround.** `-fno-stack-protector`. The tshark port passes it and is not blocked.

**Fix shape: not chosen, and the lead's call.** A compiler must not assert, so the crash is a defect
whichever shape wins.

- **(a)** Overload the intrinsics on address space and use `DL.getAllocaAddrSpace()` in
  `CreatePrologue`. This is upstreamable, but it changes generic CodeGen.
  - The fix must cover BOTH `llvm.stackprotector` and `llvm.stackguard`.
  - It must also decide the guard's own address space on a capability target. The guard is a
    pointer-sized cookie, and whether it should be a capability at all is a real question, not a
    mechanical retype.
- **(b)** Have Capstone decline the IR stack protector, giving a clear diagnostic or a no-op. The
  argument for (b) is that bounds already catch a stack-buffer overflow spatially, which subsumes
  most of SSP's value. That is a threat-model claim, not a compiler detail.

**Other passes: shallow probes only, not clearances.** No fix has been built. Eight files in
`llvm/lib/CodeGen` hardcode `PointerType::getUnqual`: StackProtector (this defect), LowerEmuTLS,
SjLjEHPrepare, ShadowStackGCLowering, DwarfEHPrepare, JMCInstrumenter, AtomicExpandPass and
TargetLoweringBase. The compiler lane probed each once, on dev:

| Pass | What the probe showed |
|---|---|
| LowerEmuTLS | **UNREACHABLE by design, not masked.** RETRACTED 2026-09-25: "masked by C-47, surfaces once C-47 is fixed". The C-47 fix forces `Options.EmulatedTLS = false` (`CapstoneTargetMachine.cpp:171`), so `-femulated-tls` is served by native local-exec lowering and the pass never runs. Measured: objects with and without the flag are byte-identical, with 0 `emutls` symbols. The underlying defect is real and NOT fixed (the pass's control-variable struct has integer fields where this target's pointers are capabilities, and it asserted), but nothing can reach it. Re-examine only if a domain ever needs an `__emutls` runtime. |
| DwarfEHPrepare | **CONFIRMED, filed as C-61.** The first probe, a C++ `throw` at -O2, compiled clean only because it had no cleanup and so no `resume` to rewrite. That was a false negative. |
| AtomicExpandPass | **CLEAN, and already FIXED by this project.** Both sites are guarded on `DL.isNonIntegralPointerType()`: `:1929` and `:2030`. The guards came from C-54, `c7f0de349b4b` (2026-09-23), and `ni:200` makes AS200 non-integral, so the AS0 cast is skipped. The probe was loaded: a 32-byte struct through `__atomic_load` emits a real `__atomic_load` call at -O1, re-checked here. An earlier oversized-`_Atomic` probe was VOID, because the frontend rejected it. |
| SjLjEHPrepare | No hard case was constructed. |
| ShadowStackGCLowering | Unused by this project: needs the shadow-stack GC strategy. |
| JMCInstrumenter | Unused by this project: needs `-fjmc`. |

### C-61 — any C++ with exceptions enabled crashes in "Exception handling preparation", because `_Unwind_Resume`'s type is built with an address-space-0 pointer `FIXED 2026-09-25, merged at 114f9678a670 (branch compiler/c61-eh-addrspace, 8fb7d4461eca): DwarfEHPrepare builds _Unwind_Resume's parameter and the exn.obj PHI in the exception object's own address space, guarded on isNonIntegralPointerType. C++ exceptions stay blocked behind C-63 and the absent unwind runtime`

> **Two residuals, neither a regression, recorded so they are not rediscovered.**
> 1. **An ARRAY-typed landingpad** (`landingpad [2 x ptr addrspace(200)]`) still aborts identically,
>    because `dyn_cast<StructType>` cannot match it and the guard falls back to AS0. It is valid IR
>    and clang never emits it. The fix is `isAggregateType()` plus `getContainedType(0)`.
> 2. **A module that already declares `_Unwind_Resume` with an AS0 parameter** gets that existing
>    declaration back from `getOrInsertFunction`, with nothing checking that the two agree. The call
>    and the declaration can therefore diverge **silently**, where the pre-fix compiler aborted
>    loudly. clang's C++ path never declares `_Unwind_Resume`, so this is reachable only from
>    hand-written or linked mixed-address-space IR.
>
> **Verification (compiler lane):**
> - the test is negative-tested two-sided, and the PHI-site CHECK is load-bearing at -O0 and -O2;
> - no `.ll`/`.mir` in `llvm/test` or `clang/test` carries both a `resume` and an `ni:` datalayout;
> - `llvm/test/CodeGen` + `llvm/test/Transforms` (38,745 tests) have failure sets identical to dev's
>   baseline;
> - Capstone lit passes 115/115.

**What happens.** `DwarfEHPrepare::InsertUnwindResumeCalls` aborts with
`Calling a function with a bad signature!` (`llvm/lib/IR/Instructions.cpp:761`) in the
`Exception handling preparation` pass. No `try`/`catch` is needed: a destructor in scope across a call
is enough.

    printf 'struct D { ~D(); };\nvoid g();\nvoid f(){ D d; g(); }\n' > eh.cc
    clang -target capstone64-unknown-elf -ffreestanding -fexceptions -c -x c++ eh.cc

Reproduced 2026-09-24 with `llvm/cmake-build-debug/bin/clang`:

| Case | Result |
|---|---|
| `-fexceptions -O0` | crash |
| `-fexceptions -O1` | crash |
| `-fno-exceptions -O1` | compiles |
| `void f(){ throw 1; }` at `-fexceptions -O2` | compiles (no cleanup, so no `resume` for the pass to rewrite) |

The compiler lane measured two more cases: try/catch with a rethrow crashes under `-fexceptions`, and
under `-fno-exceptions` it gets a clean frontend error rather than a crash. Exceptions are on by default
for C++, so this blocks essentially any nontrivial C++ at default flags.

**Root cause.**
- `DwarfEHPrepare.cpp:230-231` builds the rewind function's type as
  `FunctionType::get(void, PointerType::getUnqual(Ctx))`. The `_Unwind_Resume` parameter is therefore
  AS0.
- `ExnObj = GetExceptionObject(RI)` (`:243`) takes the exception object out of the `resume`. On capstone64
  that object is `ptr addrspace(200)`: the IR carries `resume { ptr addrspace(200), i32 }`.
- The call at `:250` passes the AS200 argument against the AS0 parameter, which trips the assertion.

A third instance sits on the multiple-resume path, `PHINode::Create(PointerType::getUnqual(Ctx), ...)`
at `:273`. The reproducer does not exercise that path, so it is not confirmed.

**Same class as C-60, different site. The two are less alike than they first look.** C-60 is an AS0
intrinsic *declaration*, fixed in `Intrinsics.td`. This one is an AS0 function type *synthesized inside
the pass*, so a fix for C-60 does not fix it.

**There is an in-tree remedy pattern.** C-54 (`c7f0de349b4b`, 2026-09-23) guarded
`AtomicExpandPass.cpp:1929` and `:2030` on `DL.isNonIntegralPointerType()` instead of assuming AS0, and
`CodeGenPrepare.cpp` uses the same predicate at 7 sites.
- **For C-61 it applies directly:** build `_Unwind_Resume`'s `FunctionType`, and the `:273` PHI, with
  the exception pointer's real address space. This is exactly how libatomic is assumed to be built for
  AS200.
- **C-60 resists it:** `llvm.stackprotector`'s signature is fixed in `Intrinsics.td` and cannot be
  guarded the same way.

**C-54 has no entry in this registry.** Its branch is merged and its lit tests are on dev, but the
defect and its remedy are undocumented. That is why C-60 and C-61 were found, and their fix shapes
proposed, a day later without knowledge of the pattern.

**How it was first missed.** The first probe was `throw 1` with no cleanup, recorded in C-60's audit
table as "compiled clean". It had no `resume`, so it could not have exhibited the defect. The loaded
probe is a destructor in scope across a call.

**Workaround.** `-fno-exceptions`, for code that does not use `try`/`catch` syntax. No C++ port is in
flight, but this should be known before anyone scopes one.

**Not established.**
- **SjLjEHPrepare:** capstone64 does not appear to select SjLj EH, and no case was constructed.
- **The libcall path at `AtomicExpandPass.cpp:2030`:** RESOLVED as CLEAN. It is guarded by C-54; see
  C-60's audit table.

### C-54 — a capability-valued atomic lost its tag at three layers, so every atomic pointer came back untagged `FIXED 2026-09-23 (external collaborator, c7f0de349b4b, merged): pointers in a non-integral address space stay pointers from clang through AtomicExpand to the runtime; entry backfilled 2026-09-24 by the compiler lane; claims re-checked against origin/dev when filed`

**What was wrong.** An atomic whose value is a pointer lost the tag at three separate layers:

* **clang** (`CGAtomic.cpp`): `EmitAtomicExpr` cast every `__atomic_*` operand to an `iN` pointer,
  so a `void **` load became `load atomic i128`.
* **AtomicExpand, which value to pass:** a 16-byte atomic exceeds the 8-byte lock-free limit and
  becomes a libcall; the sized `__atomic_*_16` calls pass the value in two integer registers.
* **AtomicExpand, the object pointer:** it was `addrspacecast` to address space 0 for every
  libcall, which on this target is a 64-bit integer pointer — `mv a1, a0` passed a bare address
  the callee could not dereference. **This was true of every atomic libcall, including the
  existing `__int128` ones, which nothing had implemented to notice.**

**The fix.** Operations that only move or compare a value (load, store, exchange,
compare-exchange, in the GNU, C11 and scoped spellings) keep a pointer in a non-integral address
space as a pointer; a non-integral pointer value takes the generic libcall, which passes through
memory with capability loads and stores; and the object pointer is passed as it is. Arithmetic on
pointers and all integer atomics are unchanged, and C11 `_Atomic` already kept pointers.

**THE GUARD THIS ESTABLISHED, which is the part later work needs.** `AtomicExpandPass.cpp:2030`
(and `:1929`) now read the address space instead of assuming it:

    if (!DL.isNonIntegralPointerType(PtrVal->getType()))
      PtrVal = Builder.CreateAddrSpaceCast(PtrVal, PointerType::getUnqual(Ctx));

capstone64 declares AS200 non-integral (`ni:200`), so the cast is skipped and the capability is
passed intact. This is the canonical in-tree remedy for a generic CodeGen pass that hardcodes
address space 0; `CodeGenPrepare` guards the same way at seven sites. **See C-61**, whose
`_Unwind_Resume` FunctionType should be built the same way, and **C-60**, which this pattern does
NOT reach because `llvm.stackprotector`'s signature is fixed in `Intrinsics.td`.

**Runtime.** `capstone/ports/musl-capstone/runtime/atomic_libcalls.c` provides
`__atomic_load`/`store`/`exchange`/`compare_exchange` by copying through capability loads and
stores. **It takes no lock**, which is correct only on a one-hart domain with no `clone`, where
nothing else writes the domain's memory while it runs; the file says so.

**What is deliberately left.** The ISA has LR/SC and AMOs with capability *addresses* but none
whose *data* is a capability, so a lock-free lowering of capability CAS is not available. Inline
`ldc`/`stc` for plain atomic loads and stores is left for later.

**Evidence, as recorded by the fix's author** (this lane verified the artifacts exist on dev but
has NOT re-run the suites): `clang/test/CodeGen/capstone-atomic-pointer.c` and
`llvm/test/CodeGen/Capstone/atomic-capability-value-libcall.ll` (the `movc` check fails when fed
`mv a1, a0`); `capstone/tests/runtime-qemu/capability-atomics/run.sh`, 7 checks at -O0 and -O2
where every pointer an atomic stored or returned is dereferenced, with a negative control — the
same image with a runtime copying the 16 bytes as two `long`s halts with cause 24 at the first
dereference; CoreMark CRC validated; BEEBS 76 of 81 wrappers pass, the other 5 skipped as they
fail identically on dev.

**Verified by the compiler lane 2026-09-24 when backfilling:** all four cited test artifacts are
present on `origin/dev`, `clang/lib/CodeGen/CGAtomic.cpp` carries a non-integral check, and the
`AtomicExpandPass` guards at `:1929` and `:2030` are present and dated to this commit by
`git log -L`. A loaded probe — a 32-byte struct through `__atomic_load`, whose emitted assembly
actually contains an `__atomic_load` call — compiles cleanly, which is how the guard was
rediscovered.

### C-51 — `llvm.ptrmask` on a capability crashes isel (`Shift amount is not an integer type!`), and every 8- and 16-bit atomic reaches it `FIXED 2026-09-23 on compiler/c51-ptrmask-capability (9335700f339d), merged into dev 2026-09-24 as #76 (4c5cfbec908b): ptrmask moves the capability by the masked difference; masked atomics have _CAP forms; 18 lane/operation checks pass in QEMU; lit, CoreMark and BEEBS (76/81, 5 identical on dev) green`

**What happens.** `llvm.ptrmask.p200.i64(%p, -4)` alone, five lines of IR, asserts in
`SelectionDAG::getShiftAmountConstant` called from `SelectionDAGBuilder::visitIntrinsicCall`.
AtomicExpand aligns the address of every sub-word atomic with exactly that call, so any 8- or
16-bit compare-exchange, fetch-op or exchange on a capability address crashes, while 32- and
64-bit ones (which need no alignment) compile. `__builtin_align_down` on a `char *` lowers to the
same intrinsic and crashes the same way. CPython's `PyMutex` is one byte locked by
compare-exchange, which is how 26 CPython objects reach it.

**Mechanism, read at `d030df93d4a4`.** `case Intrinsic::ptrmask` compares the mask (`i64`, the
index width) with the pointer's memory width (128), takes the branch written for AMDGPU buffer
descriptors, and pads the mask by building `SHL` on the pointer's value type, `c128`, which is not
an integer. Past the assert that branch would still AND the whole capability; on a capability
the mask has to act on the address alone, so this is a lowering to design, not a guard to add.
`d5b5f11cae8f` (capability-address 32/64-bit atomics) records the edge in its message:
"Subword and capability-valued atomics are outside this change."

**Reproducer.** `capstone/tests/compiler-repros/C51-ptrmask-on-capability/run.sh` — `ptrmask.ll`
alone, the operation × width matrix with u32/u64 as controls, CPython's `PyMutex_Lock` shape and
`__builtin_align_down`; PRESENT on `d030df93d4a4` and `d5b5f11cae8f`; on `f7b50f081ca4`
`ptrmask.ll` crashes too and the 32/64-bit controls fail, as that compiler predates `d5b5f11cae8f`.

### C-52 — the Greedy register allocator segfaults in `SplitEditor::rematWillIncreaseRestriction` on CPython's `compiler_visit_stmt` `FIXED 2026-09-23 on compiler/c52-frame-base-capability (fc987bb99d8d), merged into dev 2026-09-24 as #77 (5815527db9bf): the local-stack-slot base register was a GPR; it is a GPCR from CIncOffsetImm now; the Greedy crash was its consequence`

**What happens.** `Python/compile.c` (CPython 3.13.7) kills clang with SIGSEGV in pass `Greedy
Register Allocator` on `compiler_visit_stmt`, at `-O1 -g`, `-O2 -g`, `-O3 -g` and `-O3`. The stack
runs `RAGreedy::tryBlockSplit` → `SplitEditor::splitSingleBlock` → `enterIntvBefore` →
`defFromParent` → `rematWillIncreaseRestriction`. The basic allocator (`-regalloc=basic`) and the
fast one (`-O0`) compile the same input. It is the one object of the survey this stops, and it is
the bytecode compiler.

**Not established.** Which pointer in `rematWillIncreaseRestriction` (`SplitKit.cpp:591`, read at
`d030df93d4a4`) is null here. `d5b5f11cae8f` crashes identically and does not contain C-32's
rematerializable bridge (`46c53b7b6ae2`), so that change is not required for it.

**Reproducer.** `capstone/tests/compiler-repros/C52-greedy-regalloc-segfault/run.sh` (basic-allocator
control, verdict) on `src/compiler_visit_stmt.reduced.ll`. It needs `-Xclang -disable-llvm-passes`:
clang `-O1` on the `.ll` re-optimizes it and the crash disappears. PRESENT on `d030df93d4a4` and
`d5b5f11cae8f`.

**Root cause, found 2026-09-23 (supersedes "Not established" above).** gdb on the unfixed llc:
`getRegClassConstraintEffectForVReg` returns null and `rematWillIncreaseRestriction` dereferences
it, because the use asks for GPCR and the register is a GPR, a pair with no common class. The GPR
comes from `CapstoneRegisterInfo::materializeFrameBaseRegister`, which Capstone kept from RISC-V:
`ADDI` into a GPR vreg as the shared base LocalStackSlotAllocation gives accesses far from sp.
`-verify-machineinstrs` on the original 2.2 MB `compile.ll` stops right after that pass with four
`SD ..., %4983:gpr` (expected GPCR). In `compile.c` the far accesses are byval-copy stores whose
temporaries sit more than 2047 bytes from sp. The reduced reproducer shows it too, and it also
carries 73 `LW $x0` from `llvm-reduce` nulling byval sources, which is C-57, not this.

**Fixed** on `compiler/c52-frame-base-capability` (`fc987bb99d8d`): the base is a GPCR from
`CIncOffsetImm` whenever the frame register is a capability, which is the test the
`eliminateFrameIndex` scratch register already makes. The reproducer reports ABSENT with that
compiler and PRESENT with dev's; `compile.ll` compiles at `-O1`..`-O3` with the verifier and no
errors. lit: `CodeGen/Capstone/frame-base-register-capability.ll` (12 lines of IR: two byval copies
ahead of a 4 KiB byval temporary; fails on the unfixed llc). CoreMark validated; BEEBS 76 of 81 with the five known host-header skips.

### C-53 — an inline-asm `"m"` INPUT operand crashes isel ("Memory operands expect pointer values"); `"=m"` outputs compile `OPEN — COMPILER; found 2026-09-23 through CPython's configure; blocks no port today`

**What happens.** `__asm__ volatile("lw zero, %0" : : "m"(*p))` asserts in
`SelectionDAGBuilder::visitInlineAsm` at `-O0` and `-O1`, whether the memory is reached through a
pointer, a local or a global; `"=m"` output operands compile. CPython's `configure` reached it in
its x87 and mc68881 FPU checks, which read the crash as "no" -- right for this target by accident.

**Mechanism, read at `d030df93d4a4`, not confirmed by a fix.** `SelectionDAGBuilder.cpp:10392`
asserts the operand's type is `TLI.getPointerTy(DL)`, address space 0's `i64`; the operand is an
`addrspace(200)` capability, `c128`.

**Reproducer.** `capstone/tests/compiler-repros/C53-inline-asm-memory-input/run.sh` (the two
output shapes are controls); PRESENT on `d030df93d4a4`, `d5b5f11cae8f` and `f7b50f081ca4`.

### C-55 — two cascaded selects on a capability with a null operand put the physical `$c0` into a PHI; LiveVariables / PHIElimination assert `FIXED 2026-09-23 on compiler/c55-cascaded-select (b3fdbbe081e1), merged into dev 2026-09-24 as #79 (59a2c523632c): the cascaded path now COPYs a physical source into a vreg as the single-select path does; a RESIDUAL of d5b5de228b38`

**What happens.** `select c, a, null` followed by `select c, null, <that>` on `ptr addrspace(200)`
asserts in LiveVariables at `-O1`..`-O3` ("getVarInfo: not a virtual register") and in
PHIElimination at `-O0`. One select with null, two without null, and the same shape on `i64` all
compile. CPython's `Objects/dictobject.c` reaches it at `-Os` (`dict___contains__`), not at `-O3`.

**Mechanism.** Seen in the machine code after `finalize-isel`: `PHI $c0, %bb.0, %0, %bb.1, $c0,
%bb.2`. Read at `d030df93d4a4`, not confirmed by a fix: `EmitLoweredCascadedSelect` builds its PHI
from raw operand registers, while the general path in `emitSelectPseudo` routes physical sources
through `materializeSelectPHISource` -- the fix `d5b5de228b38` made for the single-select case,
which the cascaded path never received.

**Reproducer.** `capstone/tests/compiler-repros/C55-cascaded-select-null-capability/run.sh` (three
controls; needs `-disable-llvm-passes`, which it passes). PRESENT on `d030df93d4a4`,
`d5b5f11cae8f` and `f7b50f081ca4`.

**Fixed** on `compiler/c55-cascaded-select` (`b3fdbbe081e1`): `EmitLoweredCascadedSelect` passes
each PHI source through the same COPY-into-a-vreg as `emitSelectPseudo`, which confirms the
mechanism above. The reproducer reports ABSENT with that compiler (all three files, `-O0`..`-O3`,
including CPython's reduced `dict___contains__`) and PRESENT with dev's. lit:
`CodeGen/Capstone/cascaded-select-null-cap.ll` (fails on the unfixed llc); CoreMark validated;
BEEBS 76 of 81 with the five known host-header skips.

### C-56 — the address of an undefined weak symbol is not NULL in a domain `OPEN — TOOLCHAIN (the runtime no longer depends on it: #84, merged 2026-09-24); found 2026-09-23 when exit() from a CPython-shaped domain halted at the image base; the runtime's one dependence on it is removed (runtime/c56-weak-at-exit)`

**What happens.** `extern int hook(int) __attribute__((weak)); ... if (hook) hook(x);` with `hook`
undefined: the test is TRUE in a domain and the call jumps to the image base (cause 2). The address
is formed pc-relative and added to gp (`auipc`/`addi`, `cincoffset gp`); lld resolves the undefined
weak symbol to 0 at the LINK address, and the image runs at another base without relocation, so
the result is neither 0 nor untagged. `my_first_domain/link.ld` already works around the same class
for `__fini_array_start` by defining the markers. The runtime tested `if (__capstone_at_exit)` on
its exit path, so every `exit()` from a domain that did not define the hook faulted -- libc-test
defines it, which is why its suite never showed this.

**Fixed for the runtime** on `runtime/c56-weak-at-exit` (`427006e78950`): the hook is DEFINED weak
with a default body and called unconditionally. `tests/runtime-qemu/exit-hook/run.sh`: exit(7) with
no hook ends with status 7 and exit with a hook with 42, stdio flushed, no fault; the same no-hook
image against dev's hostcall.c halts with cause 2 at the image base.

**Open for the toolchain:** any other `if (&weak_undefined)` in a domain is still wrong. The fix
belongs in codegen or the linker (an undefined weak must produce a null capability), not in each
caller.

### C-57 — a load through the null capability selects `$x0`, a GPR, as its base `OPEN — COMPILER; verifier-only on everything seen so far; found 2026-09-23 while reducing C-52`

**What happens.** A load from `ptr addrspace(200) null`, or from null plus a constant, selects the
integer zero register as the address. A store to the same address does not (it verifies clean), so
the load path alone differs:

    define i32 @f() addrspace(200) {
      %v = load volatile i32, ptr addrspace(200) null
      ret i32 %v
    }

    %0:gpr = LW $x0, 0 :: (volatile load (s32) from `ptr addrspace(200) null`, addrspace 200)
    *** Bad machine code: Illegal physical register for instruction ***  ($x0 is not a GPCR register)

The emitted instruction is `lw a0, 0(zero)`: x0 and c0 share an encoding, so it addresses through
the null capability and traps, which is what the program asked for. Only
`-verify-machineinstrs` sees it. The same appears for a `byval` argument copied from null, which is
how `llvm-reduce` made C-52's reduced reproducer carry 73 of these (the original CPython IR has
none). What is NOT known: whether any pass after isel can act on the wrong class (a copy or spill of
`$x0` as a GPR would not be a capability), which would make it more than a verifier finding.

**Where to look.** The null capability in a load address reaches the address-mode selection
as a constant 0 and is materialized as `X0`; `lowerSELECT` already maps a null capability to the
zero capability register for selects (`d5b5de228b38`), and the address path needs the same.

### C-58 — MachineLICM hoists capability arithmetic above the NULL test that guards it, and CIncOffset of NULL traps `FIXED 2026-09-23 on compiler/c58-no-speculative-cap-arith (c1510d99aa5d), merged into dev 2026-09-24 as #80 (bb246e69f5f4): CIncOffset(Imm), LCC and SHRINK are hoisted only from blocks that run on every iteration`

**What happens.** A CPython call with no keyword arguments halted in
`_PyArg_UnpackKeywordsWithVararg` with `cincoffsetimm with an UNTAGGED rs1 ... val=0x0`: the
instruction was `cincoffsetimm s7, s2, 0x30`, `&kwnames->ob_item`, with `kwnames` NULL. The source
forms that address only after `kwnames` is known to be non-NULL, and so does the optimized IR (the
GEP follows a load of `kwnames->ob_size`). Early MachineLICM moved the `CIncOffsetImm` from that
guarded block into the outer loop's preheader, above the test (`-stop-before/-after
early-machinelicm`). On a conventional target that computes an address nobody uses; on Capstone
CIncOffset of an untagged value raises UNEXPECTED_OPERAND.

**Why LLVM does it.** MachineLICM hoists any loop-invariant instruction that is safe to move, and
only loads must also be guaranteed to execute. Pointer arithmetic is assumed not to trap. On Capstone
`CIncOffset`, `CIncOffsetImm`, `LCC` and `SHRINK` trap on an untagged operand (the QEMU helpers that
raise UNEXP_OP_TYPE for it) and are otherwise side-effect free, so the assumption is wrong for them.

**Fixed** on `compiler/c58-no-speculative-cap-arith` (`c1510d99aa5d`): `CapstoneInstrInfo::shouldHoist`
lets those four be hoisted only from a block that runs on every iteration (dominates every exiting
block -- MachineLICM's own rule for loads, computed by reachability because the hook has no
dominator tree). lit: `CodeGen/Capstone/no-speculative-cap-arith.ll`, the CPython shape as two nested
loops (the address formed in a guarded inner-loop preheader; the unfixed llc hoists it into `entry`), and
a control that is still hoisted. CPython's `getargs.c` keeps the instruction in its guarded block.

**Not covered.** IR-level speculation of a GEP on a possibly-NULL capability (SimplifyCFG,
LICM), which LLVM also considers free. Not seen yet; the same trap would follow. The SQLite fault
recorded in `docs/history/23-08-2026_00-30-00_sqlite-lost-tag-two-hypotheses-withdrawn.md`
(`cincoffsetimm a4, a4, 0xb0` = `&pWInfo->sWC` with `pWInfo` NULL, concluded "a null dereference in
software") has this shape and was not checked against it.
*2026-09-25:* the machine-level half of this was reached a second way, through MachineCSE's PRE, and
is C-66, which also moves this entry's fix out of MachineLICM's hook into a generic one. The IR-level
half named here was already covered when this was written: C-19 put the rule into
`isSafeToSpeculativelyExecute`, which SimplifyCFG, LICM and GVN's PRE all ask. The SQLite fault was
checked against C-66 and is not it (see there).

### C-62 — with `-g` at `-O1`+, Assignment Tracking asserts on every escaping local, because its offset accumulator is sized at the POINTER width (128) instead of the INDEX width (64) `FIXED 2026-09-23 (external collaborator, f8b140caa818, merged via #75). COMMITTED AS "C-50", a number already taken by an unrelated OPEN defect; renumbered here 2026-09-24`

> **Numbering.** The fix commit (`f8b140caa818`), its merge (`359fb2140caa`, #75), the reproducer
> commit (`a6486c9731ec`) and the folder `capstone/tests/compiler-repros/C50-assignment-tracking-index-width/`
> all say **C-50**. This registry's **C-50** is a different, **OPEN** defect: the by-value aggregate
> copy through an integer `addi` on the frame pointer. The FFmpeg port's build gate, plan and results
> cite that one, which is why this one moved rather than that. **A merged "C-50" branch does NOT close C-50.**

**What happens.** At `-g -O1` and above, `AssignmentTrackingAnalysis` asserts
`BitWidth == DL.getIndexTypeSizeInBits(getType())` in `stripAndAccumulateConstantOffsets`
(`Value.cpp`) on every local whose address escapes.

**Cause.** `walkToAllocaAndPrependOffsetDeref` (`llvm/lib/CodeGen/AssignmentTrackingAnalysis.cpp`, a
shared LLVM file) sized its `APInt` with the pointer's type size, which is 128 for an `addrspace(200)`
capability. `stripAndAccumulateInBoundsConstantOffsets` requires the index width, which is 64. The
pass's other accumulator, and `at::getAssignmentInfo` in `DebugInfo.cpp`, already used
`getIndexTypeSizeInBits`. On every upstream target the pointer width equals the index width, so nothing
upstream sees it.

**Impact.** This was the largest single failure in the CPython port's compile survey: 146 of 253
objects, because CPython builds with `-g -O3`. Before the fix, the port avoided it with
`-Xclang -fexperimental-assignment-tracking=disabled`.

**Fix and evidence, as recorded by the fix's author.**
- The fix sizes the accumulator with `getIndexTypeSizeInBits`.
- A new lit test, `clang/test/CodeGen/capstone-assignment-tracking-index-width.c`, covers an escaping
  array, an escaping struct holding a pointer, and a local that SROA removes, at `-g -O1/-O2`. The
  unfixed clang aborts on it.
- In DWARF, the `-g -O1` and `-O0` objects give the same `DW_AT_location`.
- CoreMark's CRC validated, and 76 of 81 BEEBS wrappers pass; the other 5 fail identically on dev.
- The suites were NOT re-run when this entry was filed. The fix and reproducer commits were checked
  to be ancestors of `origin/dev`.

**The family.** C-54, C-60, C-61 and this one are all upstream assumptions about pointer
representation that capstone64 violates. They lie on two axes:
- **address space:** C-54 (fixed), C-60 (open), C-61 (open);
- **width, pointer vs. index:** this one (fixed).

### C-63 — a `landingpad` cannot be selected: the exception pointer is zero-extended as an integer `OPEN — COMPILER, crash; found 2026-09-24 behind C-61; reproduced with the tree's llc; its fix needs an EH ABI decision; C++ exceptions are THREE layers deep (C-61, this, no unwinder)`

**What happens.** `Capstone DAG->DAG Pattern Instruction Selection` aborts with
`Assertion 'VT.isInteger() && N1.getValueType().isInteger() && "Invalid ZERO_EXTEND!"'`
(`SelectionDAG.cpp`). It needs only a landingpad, whose result need not even be used. There is no
`resume`, so DwarfEHPrepare returns early, and this is independent of C-61.

    declare void @g() addrspace(200)
    declare i32 @__gxx_personality_v0(...) addrspace(200)
    define i32 @f() addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
    entry:
      invoke addrspace(200) void @g() to label %c unwind label %l
    c:
      ret i32 0
    l:
      %0 = landingpad { ptr addrspace(200), i32 } cleanup
      ret i32 1
    }

Reproduced 2026-09-24: `llc -mtriple=capstone64-unknown-elf` aborts at both `-O0` and `-O2` (rc 134).

**Root cause.**
- `SelectionDAGBuilder::visitLandingPad` (`SelectionDAGBuilder.cpp:3539-3544`) copies
  `ExceptionPointerVirtReg` out at `getPointerTy()`.
- It then coerces the result with `DAG.getZExtOrTrunc` to the landingpad's first element type, which
  here is the capability, `c128`.
- `getZExtOrTrunc` requires integer types. A capability can be neither zero-extended nor truncated
  into place.

**Family.** This is the fifth upstream pointer-representation assumption, alongside C-54, C-60, C-61
and C-62. Its axis is a pointer coerced with an integer operation.

**The fix is not mechanical: it needs an EH ABI decision.** All of the following were checked
against source on origin/dev.
- `CapstoneTargetLowering::getExceptionPointerRegister` returns `Capstone::X10`
  (`CapstoneISelLowering.cpp:25557-25559`). That is an INTEGER register, inherited from RISCV. On a
  capability target the exception pointer is a capability and belongs in `C10`.
- `visitLandingPad` copies the register at `getPointerTy()`, whose default is AS 0. That gives `i64`,
  because Capstone's override returns `c128` only for AS 200 (`CapstoneISelLowering.h:324-328`).
- The ZERO_EXTEND therefore comes from `i64` to `c128`.
- `getZExtOrTrunc` (`SelectionDAG.cpp:1513-1517`) is a bare `bitsGT ? ZEXT : TRUNC` with no
  equal-type short-circuit. Merely making the widths agree would emit a TRUNCATE instead.
- So the coercion has to be AVOIDED for a capability, not satisfied. The register is a runtime
  contract, and it has to agree with whatever hands the pointer back.

**C++ exceptions are three layers deep, not one.**
1. **C-61** (fixed, unmerged) corrects the address space of `_Unwind_Resume`'s parameter.
2. **This defect** blocks instruction selection, and its fix needs the ABI decision above.
3. **No unwinder exists for a domain.** Nothing under `capstone/` outside docs and tests defines
   `_Unwind_Resume` or `_Unwind_RaiseException`, and no unwind runtime is built
   (`LLVM_ENABLE_RUNTIMES` is empty). Even with C-61 and C-63 fixed, `_Unwind_Resume` would be an
   undefined symbol at link time.

The runtime is absent in general, not only `_Unwind_*`. No `__gxx_personality_v0`, `__cxa_throw` or
`__cxa_begin_catch` is defined under `capstone/` either; the grep does fire, since it finds the names
in this file. **The link failure cannot be demonstrated from C++ source until C-63 is fixed**, because
clang produces no object. The evidence is the absence of any definition, which is weaker than a link
but sufficient. Hand-written IR calling `_Unwind_Resume` directly could force a link; that has not
been done. **`-fno-exceptions` is not a workaround pending a fix. It is the only
configuration in which C++ compiles today.** C-61 alone does not make C++ "nearly work".

**Fix: none yet. The ABI decision is the lead's.**

### C-65 — musl-capstone's `pthread_cond_t` cannot hold its own fields: `_c_tail` lies 32 bytes past the 48-byte object `OPEN — LIBC ABI (musl-capstone); found 2026-09-24 by the tshark port; WORKED AROUND for GLib only (ports/wireshark/app/deps/patches/glib-0008); source read in musl 1.2.5 as prepare-musl-capstone.sh prepares it, at 93860ed`

**What happens.** On capstone64, `pthread_cond_t` (`include/alltypes.h.in:88`) is `int __i[12]`,
48 bytes, and its pointer view `__p[12*sizeof(int)/sizeof(void*)]` holds three 16-byte pointers.
musl's internal macros (`src/internal/pthread_impl.h:92-98`) assume 8-byte pointers:
- `_c_head` is `__p[1]`;
- `_c_tail` is `__p[5]`, at +80, 32 bytes past the object;
- `_c_shared` (`__p[0]`) overlaps `_c_seq` (`__vi[2]`);
- `_c_lock` (`__vi[8]`) lies inside `__p[2]`.

`pthread_cond_init` zeroes 48 bytes. `pthread_cond_signal`/`broadcast` then walk the waiter list
from whatever follows the object, single-threaded or not. On level0's heap that is the next block's
header, whose `free` flag, 1, sits at +80: `cincoffsetimm a4, a0, 0x20` with `a0 = 1`, cause 24.
The mutex, barrier and rwlock macros fit their types (`_m_prev`/`_m_next` are `__p[3]`/`__p[4]`
of a 5-pointer union, and `_b_inst` is `__p[3]` of 4); only the condition variable overruns.

**Repro** (QEMU): `ports/wireshark/app/tests/runtime-gaps/run.sh c65`, a heap
`pthread_cond_t`, init, broadcast. It halts with cause 24 in `__private_cond_signal`
(`pthread_cond_timedwait.c`) with `x10 = 1` (log `qemu-oracle-20260925-002510-T6wb`). Natively,
glibc's broadcast returns.

**Evidence.**
- tshark's second boot halted identically in `epan_init`: GLib's `gthread-posix.c` broadcast,
  n = −1, `a0 = 1`.
- An audit matched the instruction in the archived `libc-capstone.a` (`ldc a0, 0x50(s3)` reads
  `_c_tail`).

**Worked around (GLib in the tshark port).** In a domain, GCond signal and broadcast are no-ops,
since one thread has no waiter, and waits abort. Other code calling `pthread_cond_*` is still
exposed. A survey of 219 recent domain images found no `pthread_cond` symbol outside tshark.

**The fix** is an ABI change for the lead: size `pthread_cond_t`/`cnd_t` for 16-byte pointers and
relayout the `_c_*` macros. Every port relinks against the new libc.

**Proposed, not landed (2026-09-25):** local branch `musl/c65-pthread-cond` (`85972d2`), not
pushed. It adds two overlay files in `ports/musl-capstone/arch-capstone64/` and leaves the
upstream tree untouched:
- `bits/alltypes.h.in` declares a 64-byte `pthread_cond_t`/`cnd_t`, which the generic TYPEDEFs
  then skip;
- `pthread_impl.h`, found before `src/internal`, `#include_next`s musl's own header and moves only
  the seven `_c_*` macros: the pointers at `__p[0..2]`, the ints at bytes 48–63.

With a private musl build, `run.sh c65` returns with `sizeof = 64` and "broadcast returned". The
musl build fails the same 6 objects as without the patch. The wait paths need a futex and were not
exercised (`docs/history/25-09-2026_01-30-00_c64-i11-runtime-fix.md`).

## Infrastructure / procedure

### C-66 — MachineCSE's PRE moves capability arithmetic above the test that guards it: a second hoisting path the C-58 fix does not see `FIX on compiler/c66-machinecse-pre-trapping-cap-arith; found 2026-09-25 by the CPython pymalloc corpus run inside the interpreter`

**What happens.** The CPython interpreter image of #107 (no Sublet, no adapter) halts with cause 24 in
`_PyArg_UnpackKeywords` on any call without keyword arguments that reaches its keyword walk:
`cincoffsetimm with an UNTAGGED rs1 -- rd=x16 rs1=x13 val=0x0`, at ELF `0x33e210` =
`_PyArg_UnpackKeywords+0x500`, `cincoffsetimm a6, a3, 0x30`. `a3` is `kwnames`, NULL. It ended the
control arm of **6 of the 13** interpreter-corpus candidates (cases 1, 5, 9, 10, 11, 19) before
their defect ran, and the Sublet arm of three of them (9, 11, 19), so those pairs could not be
decided at all.

The arithmetic is `&kwnames->ob_item` (`0x30` is `offsetof(PyTupleObject, ob_item)` with 16-byte
pointers) from `find_keyword` (`Python/getargs.c:2022-2046`), inlined twice. Each inlined loop forms
it in its own preheader, and each loop is entered only when `nkwargs > 0` -- which implies
`kwnames != NULL`. The guard is that correlation, not a NULL test, so no known-non-NULL reasoning
on the operand can see it.

**Where it moves.** `-print-before/-print-after=machine-cse` on `getargs.c`, compiled as the port
compiles it:

    before  bb.29.for.body.lr.ph.i.us       %37:gpcr  = CIncOffsetImm %101:gpcr, 48
    before  bb.73.for.body.lr.ph.i424.us    %76:gpcr  = CIncOffsetImm %101:gpcr, 48
    after   bb.125.for.end                  %678:gpcr = CIncOffsetImm %101:gpcr, 48
            (%101:gpcr = COPY $c13 -- the fourth argument, kwnames)

`MachineCSEImpl::ProcessBlockPRE` (`llvm/lib/CodeGen/MachineCSE.cpp:822`) finds the two copies in
blocks where neither dominates the other, duplicates the instruction into their nearest common
dominator (`TII->duplicate(*CMBB, CMBB->getFirstTerminator(), MI)`, debug location erased -- which
is why `addr2line` gives `getargs.c:0` for it), and lets CSE delete the originals. `isPRECandidate`
refuses loads and asks nothing else about speculation. `for.end` lies above the loops' zero-trip
test, so the copy runs when `nkwargs == 0`.

**Why the C-58 fix did not catch it.** C-58 is the same trap reached through MachineLICM, and it was
fixed inside MachineLICM's own hook, `TargetInstrInfo::shouldHoist(MI, FromLoop)`, with the
guaranteed-to-execute test re-derived by reachability because that hook gets no dominator tree. The
knowledge was right -- `trapsOnUntaggedOperand` lists the ten opcodes that raise on an untagged
operand -- but only one pass could ask for it. `shouldHoist` occurs zero times in `MachineCSE.cpp`.
Measured, not inferred: `early-machinelicm` leaves all three copies in their guarded blocks on this
file; it is `machine-cse` that moves one.

**The asymmetry this is an instance of.** At the IR level the same rule is in ONE predicate that
every speculating pass asks: C-19 made `isSafeToSpeculativelyExecute` answer false for a GEP on a
non-integral pointer whose base is not known non-NULL (`llvm/lib/Analysis/ValueTracking.cpp`), and
SimplifyCFG, LICM, GVN's scalar PRE (`GVN.cpp:3072`, the IR counterpart of this pass),
SpeculativeExecution and InstCombine's select folds all consult it. At the machine level there was
no such predicate, so each speculating pass had to be taught separately, and one was.

**The fix: the IR design, one level down.** A generic hook, `TargetInstrInfo::canTrap(MI)` -- true
when the instruction can trap for some operand values although it neither accesses memory nor has
unmodeled side effects; false by default, so no other target changes. Every machine pass that runs
an instruction on a path where it did not run before asks it, the way it already treats loads:

- MachineLICM (`IsLICMCandidate`, both before and after register allocation): a trapping
  instruction is hoisted only from a block that is guaranteed to execute -- its own dominator-tree
  test, the one it applies to loads. That test caches its answer, and the cache is now keyed by
  loop as well as by block (below).
- MachineCSE (`isPRECandidate`): a trapping instruction is not PRE'd. Ordinary CSE, which replaces
  an instruction by a dominating copy, is untouched.
- EarlyIfConversion (`canSpeculateInstrs`): not speculated. Inert on Capstone today -- the target
  does not enable the pass and cannot insert a select -- and there so that the rule does not depend
  on that staying true.

Capstone answers once, with the existing `trapsOnUntaggedOperand`. Its `shouldHoist` override and
`executesOnEveryIteration` are removed: MachineLICM applies its own guaranteed-to-execute test
instead, with the real dominator tree.

**The speculation cache, and what moving C-58's question into it broke.** An earlier revision of
this branch, never pushed, said C-58's test (`no-speculative-cap-arith.ll`) shows that the removed
override and MachineLICM's test agree. They agree only for the loop asked about first.
`IsGuaranteedToExecute` caches its answer per block (`SpeculationState`, reset once per block in
`HoistOutOfLoop` and in the post-RA walk). `HoistOutOfLoop` asks about a block for the outermost loop
first. When the instruction cannot leave that loop, it asks again for each subloop in turn, and it
got the outermost loop's answer back. A block can run on every iteration of the outer loop that
reaches its exit without running on every iteration of the subloop. The increment then went into the
subloop's preheader, above the subloop's own test. C-58's hook recomputed per loop and never did
this. Routing the question through the cache did: in `c66-machinelicm-subloop-speculation.ll`,
`CIncOffsetImm %1, 48` lands in `bb.1.outer` without the change below. It stays in `bb.3.body` with
the change, and with C-58's hook on `a378789289cd`. This was found because the LICM hoist count
moved when the prediction said it would not. The cause was then read in the code and the test built
from it.

The cache now also remembers which loop it answered for (`SpeculationLoop`) and recomputes when
asked about another. That is not a Capstone-only fix. The cache was written for loads, and it moves a
load the same way on any target. On x86 the load of `p+48` in the same shape goes above `p == NULL`
into the subloop's preheader under this fork's unmodified MachineLICM (drift base `b3a1c7778245`; not
checked against upstream main). `c66-machinelicm-subloop-load-x86.ll` FAILS there: on
`a378789289cd`, whose `MachineLICM.cpp` is the drift base's and dev's, `MOV64rm %3, 1, $noreg, 48`
lands in `bb.2.outer`. With the change the load stays in `bb.4.body`. Both
tests carry a control: the same instruction in the subloop's header, which does run on every
iteration, still goes to the subloop's preheader. So what they show is the rule, not an end to
subloop hoisting. The negative control was run for the Capstone test too: with only the keyed cache
reverted and `llc` rebuilt, it fails (`CHECK-NOT: excluded string found`). That run is the only
evidence for the Capstone test failing: the intermediate binary is gone, and on dev the test passes,
since C-58's hook is exact. In the lit gate it guards against that intermediate state coming back.

The stale answer also cut the other way, and that changes code on every target. When the outer loop
can exit before the subloop starts, a load in the subloop's header is not guaranteed for the outer
loop. That "no" was handed to the subloop, where the load does run on every iteration, and it
stayed put. Keyed by loop, it goes to the subloop's preheader. This is the third function of the
x86 test: on dev `MOV64rm %2, 1, $noreg, 48` stays in `bb.3.ihead`, and with the change it moves to
`bb.2.pre`. The X86 and Generic CodeGen directories were run with the final compiler: 5470 tests,
5421 pass, and no failure is new. 17 need tools this build does not have (`llvm-dwarfdump`,
`llvm-profdata`). The other 5 are the TLS tests `emutls*.ll` and `tls-android.ll`, which fail
identically on dev without C-66.

An adversarial audit re-ran both tests on dev and on `a378789289cd` and confirmed the x86 case is a
miscompile rather than a legal speculation. The load's memory operand carries no `dereferenceable`
or `invariant` flag, and the final assembly puts `movq 48(%r9)` ahead of `testq %r9, %r9`. The audit
also found no caller of `IsGuaranteedToExecute` that asks about a block other than the one whose
iteration last reset the state. There are three callers: loads, `canTrap`, and `AvoidSpeculation`.
All three pass the instruction's own block, and the post-RA walk sees top-level loops only.

**What a PRE refused costs.** PRE merges two copies that lie on one path; refusing it costs at most
one extra capability increment on the paths through both blocks, and nothing on any other path.
How many it refuses on CPython is measured below, after the end-to-end pair.

**If the ISA changes.** `plans/2026-09-24-scc-cincoffset-untagged.md` asks whether an untagged
`CINCOFFSET`/`SCC` should compute, as CHERI's do. If it does, those opcodes leave
`trapsOnUntaggedOperand` and every pass above follows from that one line. That memo also says
"no program we build needs the change now"; this issue is a program that does.

**Not covered.** A machine pass added later that speculates without asking `canTrap`. The passes
that can place an instruction on a new path were inventoried for this entry (MachineLICM,
MachineCSE's PRE, EarlyIfConversion; MachineSink only moves an instruction to a subset of its
paths, BranchFolding hoists only what every successor already runs); a new one has to be added to
that list by hand.

**Checked and ruled out: the SQLite fault C-58 lists as unchecked.** `cincoffsetimm a4, a4, 0xb0`
(`&pWInfo->sWC`, `pWInfo` NULL) in `sqlite3WhereCodeOneLoopStart`
(`docs/history/23-08-2026_00-30-00_sqlite-lost-tag-two-hypotheses-withdrawn.md`) has this shape but
not this cause, on two grounds that history records. It was a SILICON fault and "QEMU runs the
identical path tagged": a code-motion defect is deterministic compiler output, and QEMU traps on an
untagged `cincoffsetimm` as the silicon does, so a moved increment would have faulted on QEMU too.
And `pWInfo` is a parameter that `sqlite3WhereBegin` allocates -- never NULL in a correct run --
whereas this issue needs a value that is legitimately NULL on the path the copy was moved onto. So
that fault stays what history concluded it was not: a value lost on silicon.

**End to end, one variable.** There are two CPython images from one tree: the port as merged on dev,
patches 0001-0013, 250 of 250 objects compiled, strict link with 0 undefined symbols.
- Image A (`c3e07424`) is built entirely by this branch's final compiler.
- Image B' (`7a2477ef`) is identical except `Python/getargs.o`. That object was recompiled by dev's
  compiler at `01ec8b0d322a`, the fix's own parent, and relinked. So the two differ in C-66 and in
  nothing else, and `getargs.o` does differ.

Corpus cases 1 and 9, whose control arm used to end at this trap:

    B', case 1 and case 9  cincoffsetimm with an UNTAGGED rs1 -- rd=x16 rs1=x13 val=0x0,
                           _PyArg_UnpackKeywords+0x510                                  (this issue)
    A,  case 9             BEGIN, ARMED ... CPY-CASE-END -- runs through
    A,  case 1             no untagged cincoffset; faults later, in find_name_in_mro+0xdc,
                           loading through a register that holds the integer 1

The trap follows `getargs.o` and nothing else. An earlier revision of this paragraph built B from
`a378789289cd`. That compiler also lacks C-46, C-47 and C-61, so the pair differed in more than
C-66. It was redone with the fix's parent, and the result is the same.

What case 1 does next is not this issue, and it is worth recording. gh-146613 compares through a
pointer to a freed key. The block has been reused, and where `ob_type` stood there is now integer
data. That value has no tag, so the capability machine stops the use-after-free on its own, without
Sublet. It could not be seen before, because the control arm died here first.

**What refusing the PRE costs, measured.** CPython's 143 core sources were compiled with
`-mllvm -stats`, with the port's flags, once by dev's compiler (`01ec8b0d322a`) and once by the final
one. 137 compile under both; the 6 that do not are platform files. The counts, dev against final:

    machine-cse  PRE (partial redundancy made full)   1056 ->   847   (-209)
    machine-cse  common subexpressions eliminated     64037 -> 63484  (-553, the copies PRE would have merged)
    machinelicm  hoisted out of loops                24709 -> 24811   (+102, 16 of 115 files)
    machinelicm  hoisted in low register pressure    13126 -> 13193   (+67)
    getargs.c    PRE                                    15 ->     7

The LICM change is the keyed cache, and it goes up because the stale cache had also refused
legitimate subloop hoists. A first comparison against `a378789289cd` put it at +2119. That
comparison was not one variable: the rest of the difference came from C-46, C-47 and C-61.

**Lit.** The Capstone CodeGen directory, 108 tests, with the final compiler: 107 pass. The one
failure is `shared-patches-present.test`, the manifest guard, and not on account of this branch.
The four shared files patched here are in `llvm/utils/capstone-shared-patches.txt` (MachineLICM.cpp
at +16 -1 with the keyed cache; nothing else moved). What it still reports is `CodeGenDAGPatterns.cpp: diff is +40 -6, manifest
says +41 -7` -- a file this branch does not touch, whose manifest entry this branch does not touch.
It fails on dev as it stands, and whether a line of that TableGen patch was lost is a question of
its own.

**QEMU gates, with the final compiler.**
- `run-hostcall-all.sh`: 28 of 28.
- The nightly core tier (`--skip-build`) is 13 of 17 green. The four red suites were each checked
  against dev without C-66, under identical conditions, and none of them is caused by C-66:
  - `lit`: the `shared-patches-present` drift described above.
  - `lit-generic`: the two `dwarf-*` tests need `llvm-dwarfdump`, which is not built.
  - `beebs`: 5 of 81 benchmarks fail to compile. Freestanding `<string.h>` resolves to the host's
    glibc header (`bits/libc-header-start.h` not found), and dev's compiler fails on the same line.
  - `linear-uninit-corpus`: `uninit_init_then_use_ok` stores 16 bytes at its region's end
    (`va 0x660`, cause 7). Dev's compiler gives the identical signature, so this is the emulator in
    use (built 2026-09-17), not the code.

**Test.** `llvm/test/CodeGen/Capstone/c66-machinecse-pre-trapping-cap-arith.ll`: two guarded blocks
form `&kw->field` in sequence, and their nearest common dominator is `%entry`. On `a378789289cd`
(no fix) the test FAILS as it must -- `CIncOffsetImm %0, 48` is in `bb.0.entry` after
`machine-cse`. Its control, the same shape with an integer multiply, is PRE'd into `%entry` both
with and without the fix, which is what shows the test can see PRE at all.
`c66-machinelicm-subloop-speculation.ll` and `c66-machinelicm-subloop-load-x86.ll`: the speculation
cache, above, each with its control and each failing without the keyed cache.

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


### I-8 — under `-capstone-gp-captable` an image built from SEVERAL translation units gets a capability table for the globals of ONE of them `FOUND 2026-09-13 on the nginx port; GATED, capstone/tests/gp-initdesc-blocks.py`

**What happens.** `.capstone_gp_initdesc` carries a 32-byte header, `(built, count)`, followed by
24 bytes per global. The linker concatenates one such block per translation unit that has globals.
The entry glue and `domdata-budget.py` both read the FIRST header and stop. So an image with more
than one block gets a table carved for whichever object the linker put first, every global past
that has no slot, and the first access to one walks off the end of `gp`.

**What it looks like.** Not a link error, not a warning. The nginx port's image linked, and every
gate the MicroPython build accumulated passed: `cjalr=0  gp-accesses=76  gp_table sections=1
gct_end=1`. The domain then faulted inside the entry glue before a line of the program ran:

```
[CAPSTONE] Cap mem access OOB: insn = 0401b5db, pc = 101583644, pcc_base = 101580000,
           va = 3644, rs1 = x3, cursor = 1015bffc0, imm = 64, addr = 1015c0000
[CAPSTONE] domain halted by capability fault: cause = 5, tval = 0x101600000
```

`cursor + 0x40` is exactly the end of what was carved. The section held three blocks of 4, 2 and 9
globals; fifteen were in use and four had slots.

**Why no existing port hit it.** They all amalgamate. SQLite uses the upstream amalgamation, and
the MicroPython build says so in its own words, "amalgamating py/ + the port into one translation
unit". That reads as a convenience about build time. It is a requirement of this ABI, and until
now it was written nowhere.

**What to do.** Build the image as one translation unit with globals. The gate
`capstone/tests/gp-initdesc-blocks.py` refuses an image that is not, names every block and its
count, and says what the glue will carve. Run it after linking, beside the gp-table header check.

**What is NOT established.** Whether the glue could be taught to walk every block instead, which
would remove the constraint rather than gate it. That is a change to the entry glue and to
`domdata-budget.py` together, since the two must agree about what a domain asked for, and it is
not attempted here.

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


### I-9 — fourteen core QEMU suites hardcode `$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot`, so they cannot run from a git worktree `OPEN — found 2026-09-24 while gating PRs #75–#87 from a scratch worktree`

`capstone-test-env.sh` exports `CAPSTONE_BUILDROOT_DIR` precisely so the buildroot tree can live
elsewhere, and the newer probes honour it. Fourteen core suites do not, and neither do the 12 older
`build-hostcall-*.sh` probes. They reach the buildroot through a path fixed relative to the repo
root, either in the shell script or as a literal
`#include "../../../caplifive-buildroot/.../libcapstone.h"` in the guest C source:
- authority, smoke, coremark;
- rv8, beebs (through the shared `beebs_simple_host.c`);
- revoke-matrix, revoke-on-free, borrow-cost, tree-cost-O2, static-cap-globals, intra-domain-mrev,
  hier-revoke, shared-region, linear-uninit-corpus.

In a worktree the submodule dir is empty, so every one of these dies at
`cc1: fatal error: … No such file or directory`. That is a **BLOCKED** suite, which a careless
reading takes for a failure, or, worse, skips past.

**Workaround used on 2026-09-24:** symlink the worktree's `capstone/caplifive-buildroot` to the
shared tree's. After that, all of them ran and passed.

**Fix shape (not done):** route every one of those includes and paths through
`$CAPSTONE_BUILDROOT_DIR`: an `-I"$CAPSTONE_BUILDROOT_DIR/package/modcapstone/userspace/lib"` on the
compile, and `#include <libcapstone.h>` in the guest source.

### I-10 — `toolchain-fresh.py` misreports a FRESH build dir in both directions `OPEN — gate defect, reported 2026-09-24 by the compiler lane; the target list was checked here, the two outcomes were not re-run`

The gate checks only `DEFAULT_TARGETS` (`capstone/tests/toolchain-fresh.py:56-62`), and that causes
two opposite errors on a fresh build dir.

- **It passes (rc 0) when lit cannot run.** On such a build dir, `count`, `llvm-readelf`,
  `llvm-stress` and `llvm-config` are not in the list, and lit dies on the first of them.
  `llvm-readelf` is left out **deliberately**, with the reason at `:63`: it is a regenerated symlink
  that the classifier misreads. So the fix is not simply "add it".
- **It returns 2 ("cannot check") whenever `llvm-ar` is built.** `llvm-ar` is outside the list, and
  the thread-local probe's log carries that warning.

The rc=2 cost is already recorded in the comment at `:58-61`: every caller warns on rc 2 and nothing
acts on it. **The fix belongs to the gate's owner. Until then, read rc 0 on a fresh build dir as "the
listed tools are fresh", NOT as "lit can run".**

### I-12 — QEMU guests stall before any domain starts, and a runner's time budget decides how long the shared lock is held `OPEN — infra; counted 2026-09-25 on the tshark runner; cause unknown`

Some QEMU guest boots stop, or crawl, before any domain starts. The tshark safety campaign
(`ports/wireshark/app/results/2026-09-25-qemu-safety/stall-classes.txt`) had 28 boots, and 7 of
them never reached a domain. All ran on the port's PRIVATE rootfs, so the shared rootfs's ext4
corruption is not the cause. FFmpeg's hardening round counted 9 such boots: 7 before login, 1 at
login, 1 in the 9p copy. The shapes, told apart by a setup watchdog in
`ports/wireshark/app/host/run-qemu.sh`:
- **A slow 9p copy, 6 of the 28.** The guest's copy of the domain images from the 9p share
  normally takes about 30 s, and here took about 5 minutes. Each time, `cp` was waiting in
  `p9_virtio_zc_request`, a 9p zero-copy read, while its file kept growing. 3 of these boots
  completed. The other 3 were powered off by earlier watchdog versions that did not check for
  progress.
- **The guest wedged in setup, 1 boot.** The watchdog's `sleep` never woke, so userland stopped
  too.
- **Stopped between init and login, 2 boots.**
- **1 more stopped in setup before the watchdog existed**, so its kind is unknown.

During these stalls the host's load average was about 2.3, and the stalled QEMU was at 100% CPU.
So a guest that crawls is the common factor, not 9p alone.

**The cost is the shared lock, not the boot.** Nothing is lost, since a stalled boot is retried and
never counted. But the QEMU holds the lock for the whole guest budget. At 960 s per section, one
such boot held it for 85 minutes against three other lanes' queued work. What the tshark runner does now:
- size the budget from measured run times;
- a setup watchdog: at 300 s it reports where the copy stands (files, `cp`'s kernel stack, memory
  and CMA), then powers the guest off only if nothing moved in the next 30 s;
- 600 s for setup in the outer budget, so a slow copy can still finish.
No other runner has the watchdog.

**What would settle the cause:** whether the TCG vCPU or the 9p server is starved during a slow
copy. The QEMU monitor's `info registers`, or a host-side stack of the QEMU threads, taken during a
stall, would show which. No runner collects either yet.

## Compiler / toolchain (ours)

### C-50 — an integer-valued pointer in an aggregate passed BY VALUE (union or struct) is copied through an address formed with integer `addi` on the frame pointer, which faults `OPEN — COMPILER, caller side (byval copy, CapstoneISelLowering.cpp:24316); found 2026-09-23 by the FFmpeg app port on QEMU, reduced to 12 lines, proven from disassembly and -debug-only=isel; worked around in the port; audited and CORRECTED the same day, see the box`

> **Not the "C-50" in commit `f8b140caa818` / PR #75.** That merged fix is Assignment Tracking's
> index width, renumbered **C-62**. It does not touch this defect, which stays OPEN.

> **CORRECTED 2026-09-23 after an adversarial audit. The text below the box stands, but four
> things in it were wrong or too narrow.**
>
> 1. **It is not about unions.** A plain `struct { void *p; }` triggers it too. So do a
>    single-member union, a `uintptr_t`-valued pointer, and a 32-byte struct whose int-valued
>    pointer is at offset 0 (`addi a4, s0, -88`). The trigger is a **by-value aggregate whose
>    first 16-byte chunk the optimiser knows holds a NON-CONSTANT int-to-pointer value**, at
>    -O1 and above. It does not trigger at -O0, for a plain `void *` argument, for a constant
>    `(void *)0` or `(void *)5`, for an int-valued pointer at offset 16 of a 32-byte struct,
>    or when the union is written through a `long` member.
> 2. **The root cause is identified, not just the symptom.** Caller-side byval lowering creates
>    the local copy's frame index with the DEFAULT pointer type:
>    `SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));`
>    (`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp:24316`), which is `i64`, not `c128`.
>    That frame index then feeds `DAG.getMemcpy` (`:24318`). `-debug-only=isel` shows the copy as
>    `store (s128) into %stack.1 ... FrameIndex:i64<1>`. Once the known-integer value is split
>    into two i64 stores, the +8 half's address is an i64 `add`, which selects to `addi`. The
>    other `getFrameIndex` sites in that file (`:10544`, `:24149`, `:23921`) should be checked
>    for the same slip.
> 3. **Prior art: the same defect class, at a different site.** Archived **C-18**'s second
>    exposure (`ISSUES-ARCHIVE.md:2573-2593`): the S-06 memcpy hook built chunk addresses with
>    `ISD::ADD`, giving `addi a5, s0, -0xd8; sd a4, 0x0(a5)`. That hook is flag-gated off and is
>    not involved here. What is shared is an integer add on a frame-derived capability inside a
>    memcpy expansion, and C-50 reaches `getMemcpy` too.
> 4. **The prose below mislabels one detail.** The `cincoffsetimm` + `shrink` capability in `a1`
>    addresses the compound-literal TEMPORARY, whose two stores are `sd a0, 0(a1); sd zero,
>    8(a1)`. The ARGUMENT slot is the other object: its first half is `sd a0, -64(s0)`, with
>    the offset folded into the immediate, so it is fine. Only its second half goes through
>    `addi a2, s0, -56`. The listing is right; "the first address ... the second" is not.
>
> **Scan.** "The unpatched image gave exactly this one hit" **cannot be re-verified**: the
> unpatched image and its disassembly were not preserved. The first scan could also be
> evaded: by a `.L` label in between, by the register being used as a store SOURCE first, and
> by the register form `add`. It was rewritten to track taint through writes. The new
> version's positive controls are this reproducer plus those three layouts, and all four
> hit. On the patched FFmpeg image it reports 0 hits in 354,666 instructions. Separately,
> that image has no integer add off a capability frame pointer anywhere: its 82
> `addi …, s0, …` all sit in functions where `s0` is an ordinary integer.

**Symptom.** Capability fault **cause 24** (untagged operand) on a store to the caller's own stack
slot. FFmpeg's `ff_mpv_alloc_pic_pool` faulted in the first `avformat_find_stream_info` of the
mpeg4 decoder:

```
[CAPSTONE] Cap mem access requires capability: pc = 102480548, rs1 = x12, imm = 0, value = 1027ff2f8
[CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x102480548
   x2 (sp) = C(1027ff2f0 [102774400,102800000) type 1)      x12 = 1027ff2f8   <- sp + 8, an INTEGER
```

**Shape.** A union of two pointers, `typedef union { void *nc; const void *c; } T;` (FFmpeg's
`AVRefStructOpaque`), passed **by value**, whose value is an **integer cast to `void *`**. The caller
materialises the 16-byte argument on its stack as two 8-byte stores (value, then a zero high half).
The first address is a proper bounded capability (`cincoffsetimm` + `shrink`). **The second is
formed with integer `addi` off `s0`/`sp`**, so the store faults.

**Reproducer** (`-target capstone64-unknown-elf +m +a -ffreestanding -fno-builtin -O1`):

```c
#include <stdint.h>
typedef union { void *nc; const void *c; } opaque_t;
void *sink(unsigned long size, unsigned flags, opaque_t o, void *cb);
void *repro(int x)       { return sink(64, 1, (opaque_t){ .nc = (void *)(uintptr_t)x }, 0); }
void *control(void *p)   { return sink(64, 1, (opaque_t){ .nc = p }, 0); }
```

`repro` compiles to:

```
	cincoffsetimm	a1, s0, -48
	...
	shrink	a1, a3, a2
	addi	a2, s0, -56          <- integer address off the capability frame pointer
	sext.w	a0, a0
	sd	a0, 0(a1)
	sd	zero, 8(a1)
	sd	zero, 0(a2)          <- faults: cause 24
	sd	a0, -64(s0)
```

**The control does not reproduce it.** `control`, the same shape with a REAL pointer, stores with
`stc a0, 0(a1)` and `stc a0, -64(s0)`: two capability stores and no `addi` on the frame pointer. So
the trigger is the **integer-valued** pointer, which the backend splits into scalar halves, and
the defect is in how it forms the second half's address.

**How common, measured in one real image.** The FFmpeg domain image (354,657 instructions) was
scanned for the signature: an `addi rX, sp|s0, k` whose result is then used as a load/store base
within 7 instructions. The unpatched image gave **exactly this one hit**, which also serves as the
scan's positive control. After the port's workaround it gives 0. The scan covers only
`sp`/`s0`-based `addi` used as a base within 7 instructions; other shapes are not covered.

**Workaround (FFmpeg port, `ffmpeg-app` branch, patch 0003).** Pass a real pointer to a static mode
value instead of smuggling the int through `void *`. Unrelated to the source-level
`-Wcapstone-pointer-roundtrip` class: that warning flags this site too
(`mpegpicture.c:94`), but correctly, as harmless *if lowered right*. The integer in a `void *` is
never dereferenced by the program. The compiler's own store is what faults.


### C-49 — WITHDRAWN. The `shrink` that trapped in libc-test's `setjmp` was fed by a `sigsetjmp` that called `setjmp` instead of being it `WITHDRAWN 2026-09-17, same day it was filed — NOT a compiler defect`

**What it looked like.** libc-test's `setjmp`, run as a domain, trapped with cause 29,
`ILLEGAL_OP_VAL`, at `shrink s6, a0, s9`. The register dump had a code address in the size
operand:

```
x10 (a0) = 10157f9c0
x22 (s6) = C(10157f9c0 [10156e500,101580000) type 1)
x25 (s9) = 202ae3ba4        ; a0 + 0x1015641e4, and 0x1015641e4 is in .text
```

and `s9` is written in exactly three places in the function: `li s9, 0x80` and two
`add s9, a0, s9`. Reading only that, the bounded-pointer materialisation looked like it was
destroying its own length operand and then reading it again.

**What it actually was.** `runtime/sigsetjmp.c` in the musl port implemented `sigsetjmp` as a C
function that **calls** `setjmp`. The buffer then records that wrapper's frame: its `sp`, its
frame pointer, and whatever it left in the callee-saved registers. `siglongjmp` restores those,
so the caller resumed with a frame pointer belonging to a function that had already returned,
and every value the caller held in a callee-saved register was someone else's. The code address
in `s9` was one of them. musl's own `riscv64/sigsetjmp.s` **tail-calls** `setjmp` for exactly
this reason; the port's version is now the same instructions as `setjmp`, reached by a second
label rather than a call, and `siglongjmp` likewise.

**Measured.** With that one change and nothing else, `setjmp` goes from FAULT to a clean run.
No compiler change was involved, and the branch opened to hold one was deleted.

**What to take from it.** A capability fault inside compiler-generated bounds code is not
evidence that the compiler generated it wrongly. The bounds sequence is where a corrupted
register file first becomes visible, because it is the only place that checks. Before filing a
codegen defect from a fault dump, account for how the function was entered: anything that
restores a register file, `setjmp`, `longjmp`, a domain re-entry, moves the suspicion to
whoever saved it. The minimal reproduction that never fell out was the signal that this was
not codegen.

### C-48 — outgoing stack-passed varargs are packed at 8 bytes, `va_arg` reads them at 16: every seventh-and-later integer vararg is lost `FIXED 2026-09-16 — COMPILER, caller side; found the same day by libc-test's inet_pton, proven from disassembly`

**What happens.** A variadic callee's `va_arg` advances by the 16-byte slot stride, which is what
`lowerVAARG` documents as intended (`CapstoneISelLowering.cpp:10550`, "advance by ... the 16-byte
slot stride") and what the register save area uses (`:24117`, `CXLenInBytes` per register, written
with `stc` to keep tags). The **caller** places the varargs that overflow the argument registers on
the stack at an 8-byte stride. Both sides of one ABI, disagreeing.

Proof, no emulator needed. Caller with three fixed arguments and eight `int` varargs:

```
  li  t0, 0x88 ; li t1, 0x77 ; li t2, 0x66
  sd  t2, 0x0(sp)      # vararg 6
  sd  t1, 0x8(sp)      # vararg 7
  sd  t0, 0x10(sp)     # vararg 8
```

Callee, `va_arg(ap, int)` in a loop:

```
  ld  a0, 0x0(a0)
  cincoffsetimm a1, a0, 0x10     # next vararg: +16
```

So vararg 6 is read correctly, vararg 7's slot is skipped, vararg 8 is read as the seventh, and
the eighth read lands past the caller's frame.

**What it looks like.** musl's `inet_ntop` formats an IPv6 address with one `snprintf` of eight
`%x`: `::1` came out as `::1:0` and `1:2:3:4:5:6:7:0` as `1:2:3:4:5:6::`, i.e. the seventh
hextet gone and the eighth in its place. Any `printf` family call whose integer varargs overflow
the registers is affected; with three fixed arguments that is the sixth vararg onward.

**The fix** is in `CC_Capstone`'s generic path, `CapstoneCallingConv.cpp`: a variadic argument that
gets no register takes `AllocateStack(16, Align(16))` instead of XLen/8, so the caller's slots sit
where `va_arg` was already looking. Fixed arguments keep their XLen slots, they are never read
through a `va_list`, and capability varargs already took 16-byte slots. Nothing changes for a
callee: `LowerFormalArguments` derives the stack-vararg base as before. Changing `lowerVAARG` to 8
instead would have broken capability varargs, which need the 16-byte, 16-aligned slot to keep
their tag.

**Second symptom, a hang rather than a wrong answer.** musl's `getmntent_r` parses a line with one
`sscanf` carrying ten pointer varargs, eight `%n` and two `%d`, behind two fixed arguments, so four
of them are stack-passed and mis-stepped. Its retry loop, `while (linebuf[n[0]] == '#' ||
n[1]==len)`, keys on those `%n` results and never terminates. libc-test's `mntent` therefore spins,
and a spinning domain holds the only hart, so the guest never runs again: the batch it was in lost
every test after it. That is the shape to expect from this bug wherever a loop condition depends on
a vararg beyond the registers, and it is why `mntent` sits in `libc-test/quarantine.txt`.

**Gate.** `llvm/test/CodeGen/Capstone/c48-vararg-stack-stride.ll` checks the caller's stack slots
directly, `sd` at 0, 16 and 32 and none at 8, and needs no emulator; it fails on the tree before
the fix. The integration gate is libc-test's `inet_pton` and `mntent` under
`capstone/ports/musl-capstone/`, which fail without the fix for exactly this reason.

### C-47 — `__thread` cannot be lowered (`Cannot select: c128 = GlobalTLSAddress`), and it is NOT what blocks a libc `FIXED 2026-09-24 on compiler/c47-tls: local-exec on tp's capability, a PT_TLS segment in my_first_domain/link.ld, and the block runtime/tls.c builds from it`

> **2026-09-24: FIXED on `compiler/c47-tls`.** The "capability TLS model" this entry called a design
> question has a narrow answer for domains: one static image, so every thread-local is local-exec.
> The compiler builds `lui %tprel_hi` / `addi %tprel_lo` as an integer and applies it to tp's
> CAPABILITY with `cincoffset` (the RISC-V sequence ADDs tp as an integer and drops the tag), then
> narrows the result to the variable as a sized global is. `my_first_domain/link.ld` gives the image
> a PT_TLS segment (empty, and nothing moved, for every existing domain), and `runtime/tls.c` builds
> the block -- struct pthread below tp, the copied template at tp -- that musl's `__init_tls` would,
> since a domain has no auxv. -femulated-tls is served by the same lowering (the emutls pass asserted
> here). A thread-local initialised with a global's address is now a compile error: the
> capability-initializer pass used to skip it and leave an untagged value. Test:
> `tests/runtime-qemu/thread-local/`, 9 checks at -O0 and -O2 with an overrun control and an
> old-runtime control; the -O2 arm is what exposed C-46 live, so this branch sits on that fix.
> Still not covered: libc-test's `tls_*` tests need threads or DSOs. With this fix `tls_init` and
> `tls_local_exec` build and run, and both fault in `pthread_create`, because a domain has no
> threads (`tls_local_exec`'s main thread reports no failed check before that). `tls_init_dso` does
> not build: it initialises a thread-local with a global's address, which is the new error.

> **The registry's note on this fix, written 2026-09-24 while it sat on an unmerged branch, is
> reconciled with the box above at merge, as the note asked.** What it checked independently (the
> `cincoffset …, tp` CHECK lines, the emulated-TLS removal, the pinned old-runtime control) is what
> that box describes. On the workarounds it named: CPython runs its port's `checks.py` 6/6 in a
> domain with patch 0006 deleted and this fix; the tshark port's single-thread define has not been
> tried against it. Dropping either is that port's own change.
>
> **Scope of the defect before the fix, measured 2026-09-24.**
> - **What dies.** Any reference to a `__thread` variable that SURVIVES to instruction selection
>   dies with `Cannot select: c128 = GlobalTLSAddress`. That covers `int`, pointer-typed,
>   struct-typed, address-taken, written, or read through external linkage, at `-O0` through `-O2`.
>   Neither the type nor the optimisation level is the discriminator.
> - **What compiles, and neither case is TLS support:**
>   - a `__thread` that is **declared and never referenced**;
>   - an **internal-linkage (`static`) `__thread` that is never written**, whose reads fold to the
>     initial value at `-O1` and above. The same source fails at `-O0`.
> - **So a sample that "compiled" proves its reference was optimised away**, not that the type or
>   the access pattern is supported. A `static` one that compiles at `-O2` fails the moment anything
>   writes it or takes its address.
>
> | `__thread` | `-O0` | `-O1` | `-O2` |
> |---|---|---|---|
> | `static`, read only (with or without initialiser) | C-47 | ok | ok |
> | `static`, read + written | C-47 | C-47 | C-47 |
> | `static`, address taken | C-47 | C-47 | C-47 |
> | external, read only | C-47 | C-47 | C-47 |
>
> Measured by the compiler lane and reproduced here, on a C-50…C-58 build without C-47.
>
> **Correction, recorded because the first draft of this block was wrong and nearly landed.** It said
> the only compiling case is an unreferenced `__thread`. Two matrices supported that: the compiler
> lane's, and a "reproduction" here. **Both used external linkage in every case**, so neither
> contained the construct that read-folding needs. The static cases, found by the tshark lane, are
> the counterexample. A negative result is void unless the input contained the thing that could
> change it.

**What happens.** A `__thread` variable dies in isel with `Cannot select: c128 = GlobalTLSAddress`,
and `-femulated-tls` does not help. No capability TLS relocation or captable-style TLS slot exists.

**Why it is recorded as off the critical path.** It is tempting to read this as "a libc needs TLS, so
a libc needs this fixed". musl's core uses the `__thread` KEYWORD zero times. Every hit a text search
finds is the variable `__thread_list_lock`. musl reaches errno, the locale and the cancellation state
through the thread POINTER and a plain struct, not through thread-local storage, so a static
single-threaded domain needs `tp` and needs nothing from this issue.

> **2026-09-17, from libc-test in a domain.** Five sources of the suite fail to build on this,
> not four: `tls_align_dso`, `tls_init`, `tls_init_dso` and `tls_local_exec` die in isel, and
> `tls_align` links against a `__thread` variable its partner DSO would have defined, so it
> comes back as `undefined symbol: t`. They are the only five of the 77 that do not build.

**What actually blocked it, measured 2026-09-16**, all four the same defect in different places, a
capability sent through something that carries 64 bits:

| where | was | is |
|---|---|---|
| `arch/riscv64/pthread_arch.h` | `__get_tp()` returns `uintptr_t`, reads with `mv` | port overlay returns `char *`, reads with `movc` |
| `src/thread/riscv64/__set_thread_area.s` | `mv tp, a0` | `movc tp, a0` in the port's runtime |
| domain startup | never called anything | calls musl's own `__init_tp` |
| `runtime/start-musl.S` | `__capstone_yield` saves ra, gp, s0-s11 | also saves `tp` |

The last one is the one that would have been guessed last. The yield's own comment says the registers
ARE the suspended computation and that which of them survive the boundary is unverified; `tp` is the
answer to that question and it stood open for two ports because nothing in a domain had used it.
`write(1,...)` completed through the yield and the `write(7,...)` after it halted with `cause = 24` at
the `cincoffsetimm` of `__pthread_self`, with `tp` reading 0.

**What this issue still blocks.** Anything that genuinely uses `__thread`: real pthreads, and any
application (rather than libc) source that declares thread-local variables. Those need a capability TLS
model, which is a design question and not a missing pattern.

**Evidence.** `capstone/ports/musl-capstone/write-probe/`, its negative control built with
`-DMUSL_WRITE_PROBE_WANT_BADFD`, which reads `errno` after a refused fd and so cannot pass unless the
thread pointer survives.

### C-46 — `MOVC` is modelled as side-effect-free with `$rs1` a pure USE, so the machine model does not know it CONSUMES a linear source `OPEN — OBSERVED LIVE 2026-09-24 for direct-call targets (the direct-call instance fixed at 563e0765953e, merged 2026-09-25 at 3979abd8e9a3); the MOVC modelling itself is unchanged. The fix shape this entry first implied is WRONG — see the box`

> **What the merged fix does NOT guard.** `llvm/test/CodeGen/Capstone/c46-call-target-nonlinear.ll` is
> a codegen-SHAPE test: it checks that the target is built non-linear, not that a program survives.
> The runtime reproducer lives on the stacked C-47 commit: the thread-local QEMU test, whose -O2 arm
> is how the fault was found. So `563e0765953e` alone ships no runtime regression guard, and reverting
> the C-47 work would take C-46's runtime coverage with it.
>
> **libc-test, measured 2026-09-25 by the helper lane.** Same compiler binary for each tree, clean
> rebuilt musl, EXT4 errors uniform across all runs.
> - Before, dev `da4c9a5`: 37/6/6 + 1 NOBOOT, with the six C-46 tests faulting cause 24.
> - After, `3979abd8e9a3`: 43 PASS / 7 FAIL / 2 FAULT. fwscanf, memstream, string, strtod_simple and
>   tgmath now pass. setjmp runs and FAILs on an unserved `rt_sigprocmask`. The 2 FAULTs are
>   tls_init and tls_local_exec, which C-47 now lets build; both halt in `pthread_create.c`, because a
>   domain has no threads.
> - **Among tests that were already building, FAULT went 6 → 0.** "FAULT 2" does NOT mean C-46
>   underperformed: both faults are new arrivals that C-47 made buildable. The contributor's
>   43 PASS / 7 FAIL is exact for the population it described.
> - The musl README's 43/7/0 for 2026-09-17 predates a later regression of dev. It was not a wrong
>   baseline.

> **2026-09-24: OBSERVED LIVE under QEMU, and the "What would settle it" case below now exists.**
> A direct call's target was built as a bare `cincoffset rd, gp, off` (selectCall), which is LINEAR
> (QEMU prints `type 0` = `CAP_TYPE_LIN`, `cap.h:27`) and pure, so MachineCSE merged the targets of
> the nine calls `main` makes to one static function into one register. Under the extra register
> pressure of a thread-local test at -O2, the allocator copied it -- `movc s8, s11` at image offset
> 0x14290 -- and called through the SOURCE next: `cjalr ra, 0(s11)` at 0x14294, cause 24, "cs.cjalr
> requires capability in rs1", with the dump showing `x24 = C(... type 0)` and `x27 = 0`.
> `helper_csmovc` nulls a source that is not copyable (`op_helper.c`), which is exactly this entry's
> premise. The same program without the thread-locals has the same linear target register but no
> copy, so the trigger is register pressure, not TLS. A straight-line scan of the 168 images dev's
> baseline built found three more call-target candidates, all in SQLite (sqlite3BtreeDropTable,
> sqlite3AlterRenameTable, sqlite3WindowCodeStep) -- candidates, since the scan ignores branches.
> **It was also behind every libc-test fault on dev:** fwscanf, memstream, setjmp, string,
> strtod_simple and tgmath all halt with "cs.cjalr requires capability in rs1"; with only the fix
> below, none faults (five pass, setjmp fails its signal-mask check, which a domain cannot serve),
> and the 27 tests their chunks hid run: 38 pass / 6 fail / 6 fault becomes 43 / 7 / 0. The
> straight-line scan above found none of these six, so its count is a floor.
> **Fixed for call targets** by building them with PseudoCapGlobalBase (cincoffset + delin as one
> instruction, non-linear), as selectLGA builds a global's base for the same reason; test
> `c46-call-target-nonlinear.ll`. The MOVC definition is untouched: the trade in the box below
> (`hasSideEffects = 1` for every capability copy) is still the lead's, and any other LINEAR value
> the allocator can copy with a live source is still exposed.

> **⚠ CAVEAT on "not a live miscompile" (2026-09-24). A reproduced fault contradicts it.** The fault,
> its evidence and the fix for call targets are in the 2026-09-24 box above. (This caveat was first
> written while the fix sat on an unmerged branch; it is reconciled with that box here, as it asked
> to be at merge.) The 2026-09-10 "latent" verdict below should not be read as "C-46 is not live".
>
> **Why the 2026-09-10 argument did not catch it** (compiler lane, re-read at source 2026-09-24).
> The argument says MOVC has no IR pattern and is emitted only by `copyPhysReg` and frame-index
> elimination, at or after register allocation. So MachineCSE and MachineSinking never see a MOVC.
> That is **correct about MOVC and answers the wrong question.** The fault needs no pre-RA pass to
> touch a MOVC. A pre-RA pass (MachineCSE) **extended a LINEAR value's live range**, and the
> allocator's own copy of it IS the MOVC. So reasoning about the copy instruction's reachability
> cannot establish latency. What has to be bounded is **which LINEAR (or untagged) values can reach
> register allocation with a live range the allocator may split**. It is the same pre-RA/RA
> boundary C-32 turned on (there, `PerformSinkAndFold` deciding whether a value reaches RA as a
> copyable capability). The box above agrees: "any other LINEAR value the allocator can copy with a
> live source is still exposed."

> **2026-09-15: the read-after-copy precondition this entry defers to C-32 is OBSERVED on silicon** —
> `movc a0, s3` of an integer-bridged base with `s3` live afterwards, emitted by -O1 and -O2 in the
> SQLite Sublet port's `setupLookaside`, and the machine model saw a plain copy. See C-32's box.

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
> ADD/OR/XOR-with-X0.
>
> **⚠ THIS BOX WAS ITSELF WRONG AND IS RE-CORRECTED. `isMoveReg` IS the load-bearing thing, the
> withdrawn pin WAS guarding something real, and C-46's exposure is LARGER than this entry said.**
>
> The sequence, because the mistake is more instructive than the conclusion. I wrote that marking MOVC
> `isMoveReg` would expose it to MachineCopyPropagation. The compiler lane reported that flipping the
> flag changed nothing and gave a structural reason: copy propagation finishes before
> `ExpandPostRAPseudos` creates a MOVC. **I checked the two generic instances, found both run earlier,
> and wrote "both copy-propagation runs finish before a MOVC exists." That is a completeness claim
> about a pipeline any target may extend, and I never looked at whether this one does.** It does.
>
> **Verified here at the source, this time including the target:**
>
> * `CapstonePassConfig::addPreEmitPass()` (`CapstoneTargetMachine.cpp:593`) adds
>   `createMachineCopyPropagationPass(true)` at **`:601`** — a THIRD instance, on by default, and
>   `addPreEmitPass()` is called at `TargetPassConfig.cpp:1214`, **after** `ExpandPostRAPseudos` at
>   `:1182`. It holds MOVCs.
> * `setMachineOutliner(true)` and `setSupportsDefaultOutlining(true)` at
>   `CapstoneTargetMachine.cpp:166-167`. **The outliner is opted in, not absent** — the opposite of
>   what was recorded.
>
> Only the post-RA-scheduler reason survives.
>
> **Why their evidence fooled them, and it is the sharpest instance of this family yet.**
> `-debug-only=machine-cp` printed only `$c8 = COPY $c10` and never a MOVC — but with `isMoveReg`
> unset, `isCopyInstr` returns nothing for a MOVC, **so the pass never PRINTS one**. That log is
> byte-identical whether the pass holds a MOVC and ignores it or never sees one: it cannot separate the
> two hypotheses on the table. And "flipping the flag changed nothing across five shapes" was
> unsupported rather than negative — none of the five contained a copy-propagation-removable MOVC
> pattern, so nothing could have changed. **No positive control, and nobody noticed, myself included.**
>
> **The discriminating test, now committed** as `test/CodeGen/Capstone/c46-movc-not-a-copy.mir` with the
> integer arm as its positive control: a matched pair through the real pre-emit instance, one variable.
> Flag unset, the integer arm's copies are deleted and the capability arm **survives**. Flag set, the
> integer arm is unchanged and the capability arm **loses both MOVCs**. So `isMoveReg` is the only thing
> standing between copy propagation and a MOVC.
>
> **NET: exposure is larger, not smaller.** A live copy-propagation pass and a live outliner both
> reasoning about an instruction whose source-destroying write they cannot see. The BOUND is unchanged
> and is C-32's case — they can only do harm where a read of the source outlives the `movc`. Still no
> code change; the `hasSideEffects` trade remains the lead's.
>
> **The post-RA half of that caveat is now CLOSED (compiler lane, 2026-09-10), by auditing the target's
> actual pass list rather than a guess at it.** What runs after register allocation, from
> `CapstoneTargetMachine.cpp`: `CapstoneRedundantCopyElimination`, `CapstonePostRAExpandPseudo`, KCFI,
> `CapstoneLoadStoreOpt`, `MachineCopyPropagation`, `CapstoneLateBranchOpt`, IndirectBranchTracking,
> BranchRelaxation, `CapstoneMakeCompressibleOpt`, then the machine outliner and a feature-gated post-RA
> scheduler.
>
> The dangerous-looking one is safe, and it was checked by PREDICATE rather than by name:
> `CapstoneRedundantCopyElimination` is a target pass that deletes copies post-RA, exactly where MOVC
> lives, so it was the obvious candidate. It cannot touch MOVC — it matches the **generic COPY** opcode
> with source `X0`, guarded by a BEQ/BNE against `X0` (`:76`, `:79`, `:121`); MOVC is a real target
> instruction, not a generic COPY, and `X0` is an integer register, not a GPCR.
>
> **What genuinely remains is narrower:** the post-RA scheduler and the machine outliner can reorder or
> factor around MOVC, and because the source-nulling write is unmodelled they would treat a later read
> of `rs1` as independent of it. That is a real hazard in principle — but it needs a READ OF THE SOURCE
> AFTER the `movc`, which is the live-source case both emitters already mark against with a kill flag
> and which `copyPhysReg`'s own comment argues the allocator never asks for on a linear capability. So
> every remaining path funnels through the same precondition, and **that precondition is C-32's**.
>
> **BOTH LIMITS NOW CLOSED (compiler lane, 2026-09-10), and the residual is INERT IN OUR CONFIGURATION.**
> The post-RA scheduler is a *tune* feature, `TunePostRAScheduler`, carried by named cores only;
> `GenericTuneFeatures` at `CapstoneProcessors.td:89` is `[TuneOptimizedNF2SegmentLoadStore]` and
> `GENERIC_RV64` takes that list, and our build scripts pass no `-mcpu` at all, so we get the generic
> model and the feature is absent. The machine outliner is not opted in either —
> `CapstoneTargetMachine.cpp` does not override `enableMachineOutliner`, which defaults to false, so it
> runs only under an explicit flag. Re-verified here: 14 named cores carry the tune feature, the generic
> list does not, there is no outliner override, and no build script passes `-mcpu`.
>
> **So the watch-list changes shape: the trigger is a BUILD-CONFIGURATION change, not a source change.**
> C-46 becomes reachable if someone selects a tuned `-mcpu` or turns the outliner on — and even then
> only in the read-after-copy case that **C-32** owns. Remaining limit, stated: this is a reading of
> predicates and feature lists, not a case constructed to make a pass misbehave.
>
> **A near-miss worth recording, self-caught by the auditor.** Their first check grepped for
> `FeaturePostRAScheduler` and returned empty, which reads as "no CPU enables it" — the right conclusion
> for the wrong reason, because the token is `TunePostRAScheduler`. They noticed and redid it, which is
> how the named-core list above exists. That is the **third** instance in one session of the same
> failure: matching a name or a value instead of asking what defines it (the others being the Q-04
> retraction and the 3-versus-4 type-numbering trap in M-5). Two of the three were caught; the one that
> was not cost a retracted ruling.
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

### C-45 — the register+symbol call form `call a0, foo` (`PseudoCALLReg`) does not assemble `FIXED 2026-09-10 (compiler lane, `d7514ed41f2c` on `compiler-validation-plan`) by making the capability/integer operand coercion IDEMPOTENT — the reverse arm restores ONLY the operands the forward arm actually rewrote. Found 2026-09-10 while fixing C-38; NOT a regression (the pre-fix 2026-09-04 binary rejects it identically). ⚠ "low priority, no known consumer" is WITHDRAWN 2026-09-10: CODEGEN ITSELF emits `PseudoCALLReg`, so `-S` output cannot be reassembled`

Spun out of C-38 under the one-defect-per-commit rule. C-38 fixed the register+register form
(`call a0, a1`) by making `parseCallSymbol` decline register names; the register+**symbol** form that
`PseudoCALLReg` defines still does not assemble.

**Not a regression, established rather than assumed:** the pre-fix binary of 2026-09-04 rejects
`call a0, a1`, `call a0, a0` and `call a0, foo` identically, so this form has never worked on this
target. Documented in `cap-call-mnemonic.s` beside the C-38 case.

> ## THE FIRST FIX WIDENED THE ASSEMBLY LANGUAGE. It was narrowed on the lead's ruling — do not re-litigate.
>
> `07bf18d4f20c` made the coercion idempotent by mirroring the forward arm's full class set, and as a
> **side effect** the assembler began accepting a capability spelling in **every integer slot in the
> ISA** — `add c10, c11, c12`, `sd c10, 0(c11)`, `ld c10, 0(c11)` all assembled where they had been
> errors. Nobody set out to change the accepted language.
>
> **The lead ruled: restore ONLY what was mutated** (`d7514ed41f2c`). A "was coerced" bit on the
> parser's operand, cleared language, bug still fixed. The deciding argument was that **no prior
> intent covered the widening**: the one-register-file comment (`CapstoneAsmParser.cpp:1332-1338`)
> justifies the FORWARD direction only — a *capability* operand written with its integer name — and
> `cap-regnames.s` scopes `cN` to "a **capability operand**". No test anywhere asserted `cN` in an
> integer slot, so keeping it would have extended a decision nobody took. A third option, narrowing to
> the call classes alone, was rejected by both lanes because it leaves the identical bug latent behind
> any future two-operand capability pseudo.
>
> **The flag survives across match trials for exactly the reason the bug exists:** the generated match
> loop (`:9971`) dereferences the operand vector directly (`:10006`) and on a failed candidate simply
> continues — it never copies, clears or rebuilds it.
>
> **Negatives that discriminate, not merely pass:** `cap-invalid.s` carries the four narrowed forms,
> and every one of them ASSEMBLED under `07bf18d4f20c`'s build and errors under `d7514ed41f2c`. Five
> call forms still assemble, including `call c10, c11`, which is a capability slot and legitimately
> takes `cN`.
>
> ### The forward arm has a SOUNDNESS GAP, and it is UNREACHABLE — checked, not assumed
>
> The compiler lane flagged that the forward arm returns `Match_Success` for its narrowed classes
> **without re-checking that the register is in the class it just claimed to satisfy**: coercing `x10`
> for `MCK_GPCRC7` returns success with `c10`, and `C7` was what was asked for. Only `GPCRNoC0` is
> guarded (`if (Idx == 0 && Kind == MCK_GPCRNoC0) break;`). They did not touch it, correctly, and
> asked whether it is reachable. **It is not**, from the generated match table:
>
> * **`MCK_GPCRC7`, `MCK_GPCRJALRNonC7`, `MCK_GPCRTCNonC7` appear in NO `MatchTable` row at all** —
>   only in the class enum, the name table, and a register-class lookup array (`:4700`, `:4724`, which
>   are not match rows). No instruction ever requests them, so the looseness cannot fire.
> * **`MCK_GPCRNoC0` IS reachable — via `mrev` (`:8129`, `{ MCK_GPCRNoC0, MCK_GPCR }`) — and it is
>   precisely the one case the forward arm guards.**
> * The wrong-row hazard needs **two competing rows**; `mrev` has **exactly one**, so there is no
>   other candidate to take.
>
> **So: real in the code, dead in practice, and it becomes live the moment any instruction is given one
> of those three classes.** Left unfixed deliberately. The compiler lane's restore verifies class
> membership rather than mirroring the looseness, so the fix does not inherit it — which is why this is
> a note and not a defect.

> ## ⚠ RETRACTED — I ANALYSED THE WRONG TREE. The box below is kept because its reasoning is right about `dev` and wrong about where the fix lands.
>
> **Measured on `compiler-validation-plan` after the arm was built** — which is what I asked for, and
> the prediction did **not** hold:
>
>     call a0, a1               ->  [0x5b,0x95,0x05,0x40]  CAP_CALL, NO relocation
>     call c10, c11             ->  same encoding
>     call a0, foo              ->  auipc a0 + jalr a0, R_Capstone_CALL_PLT foo
>     call t0, __riscv_save_12  ->  auipc t0 + jalr t0, R_Capstone_CALL_PLT
>
> **Why.** I read `parseCallSymbol` on `dev`. **C-38 is not on `dev`** — and I had established that
> myself the same day, when I qualified C-38's archive entry for citing a commit that is not an
> ancestor of `dev`. C-38 adds a guard declining a REGISTER NAME as a call symbol *in any operand
> position*, so on that branch the trailing operand stays a register, `CAP_CALL` matches first and
> `PseudoCALLReg` is never reached. My own proof case is the tell: on `dev` `call c10, c11` fails; on
> the branch it assembles.
>
> **So the mechanism was right and the tree was wrong**, which is the more embarrassing of the two — I
> was holding the fact that would have caught it. **Read the branch a fix lands on, not the branch you
> happen to be standing in.**
>
> The four conditions were met anyway and are worth keeping: the predicted failure is now a standing
> pin (three register/register calls must emit NO relocation, so a degradation adds a third
> `R_Capstone_CALL_PLT` and fails loudly), the XFAIL file was rewritten so the suite can see a change,
> and the class set is mirrored in full.
>
> ### ~~THE OBVIOUS FIX TURNS A LOUD ERROR INTO SILENT WRONG OBJECT CODE~~ — true on `dev`, prevented by C-38 on the branch.
>
> The proposed fix makes the operand coercion idempotent: add a reverse arm to
> `validateTargetOperandClass` so a `GPCR` register at an `MCK_GPR` slot is rewritten back to its `X`
> form. The root cause it addresses is real and precisely located — the forward coercion mutates the
> operand IN PLACE at `CapstoneAsmParser.cpp:1344` and `:1352`, the matcher never restores it when that
> candidate fails, and the later `PseudoCALLReg` trial then sees `C10` where it wants a `GPR`.
>
> **But `CAP_CALL` is unreachable from the assembler for a SECOND, independent reason**, and the fix
> does not address it. `parseCallSymbol` (`CapstoneAsmParser.cpp:2246-2248`) returns `NoMatch` **only
> when the identifier is not last**, and the generated matcher registers that custom parser for `call`
> operands **0 and 1** (`CapstoneGenAsmMatcher.inc:10392-10393`). So the trailing operand of
> `call a0, a1` is **always** consumed as a call symbol and is never a register. Proof that this is not
> the mutation bug: `call c10, c11` needs no coercion at all and still fails —
>
>     $ llvm-mc -triple capstone64 -show-encoding   # 'call c10, c11'
>     error: invalid operand for instruction
>
> **The consequence, reproduced on stock `riscv64` where nothing mutates operand 0.** Once the reverse
> arm restores operand 0 to `X10`, `PseudoCALLReg {MCK_GPR, MCK_CallSymbol}` has BOTH operands
> satisfied and matches:
>
>     $ llvm-mc -triple riscv64 -show-encoding   # 'call a0, a1'
>     call a0, a1   # fixup A - offset: 0, value: a1, kind: fixup_riscv_call_plt
>
> That is a PLT call to an undefined symbol **literally named `a1`**. And `call a0, a1` is **exactly
> the text our disassembler emits for the CAP_CALL encoding** (`0x5b 0x95 0x05 0x40` disassembles to
> `call a0, a1`, verified). So after the fix, disassemble-then-reassemble silently produces different
> object code where today it errors out.
>
> **THE TEST SUITE WOULD NOT NOTICE.** `test/MC/Capstone/cap-call-mnemonic.s` is `XFAIL: *` and its
> `CHECK` demands the CAP_CALL encoding. After the fix the file still fails, on the encoding, so it
> still reports XFAIL and the delta is invisible. Its own promise at `:11-12` — "reports XPASS the
> moment it lands" — does not come true either.
>
> **This is NOT a reason to reject the fix.** `call a0, foo` and the real codegen shape
> `call t0, __riscv_save_12` both fail today (verified) and must assemble. Land it, but: pin
> `call <reg>, <reg>` so it cannot silently become a symbol call; rewrite the XFAIL file so the suite
> can see the change; and record that **C-38 stays open with its failure mode changed from loud to
> silent**, which is worse than the failure it replaces if nobody is told.
>
> **Two more findings from the same sweep.** (a) The reverse arm as described handles only `MCK_GPR`,
> one of twenty integer-side classes in the match table, while the forward arm covers seven capability
> classes — so `cN` would be accepted at `MCK_GPR` and still rejected at `MCK_GPRNoX0`, `MCK_GPRC`,
> `MCK_SP` and the rest. **Inconsistent acceptance is worse than either consistent choice.** (b) The
> arm's blast radius is not the mutation: `cN` spellings are user-writable and committed as such
> (`test/MC/Capstone/cap-regnames.s` asserts `movc c10, c11` assembles), so the arm would newly accept
> `add c10, c11, c12` and `lw c10, 0(c11)` across all `MCK_GPR` slots. Whether that widening is wanted
> is a design question — `CapstoneAsmParser.cpp:1332-1338` documents a one-register-file model where
> `cN` and `xN` are the same register — but it should be decided, not acquired.
>
> *(Found by an adversarial sweep of the generated match table, 2026-09-10; every claim above
> re-verified here against the primary source or by running `llvm-mc`. The sweep also REFUTED the
> premise that `call` is the only mnemonic with competing GPCR/GPR definitions — twelve others have
> them, `fld flh flq flw fsd fsh fsq fsw sb sd sh sw`. They are safe only because their GPR row sorts
> FIRST, on operand count. `CAP_CALL` and `PseudoCALLReg` tie at two operands. So the invariant being
> relied on is real but narrow, and a future two-operand capability pseudo on an existing integer
> mnemonic re-creates this exactly.)*

**~~No known consumer~~ — WITHDRAWN 2026-09-10 (compiler lane), and I verified both sites myself
rather than taking the report.** Codegen emits `PseudoCALLReg` on two live paths:

* **the spill libcall**, `CapstoneFrameLowering.cpp:2190` —
  `BuildMI(MBB, MI, DL, TII.get(Capstone::PseudoCALLReg), Capstone::X5).addExternalSymbol(SpillLibCall, ...)`
* **the machine outliner**, `CapstoneInstrInfo.cpp:3710` —
  `BuildMI(MF, DebugLoc(), get(Capstone::PseudoCALLReg), Capstone::X5).addGlobalAddress(...)`

So `-S` output containing `call t0, __riscv_save_12` **cannot be reassembled**, which makes this a
round-trip defect on compiler-generated text rather than a hand-written-assembly curiosity. The
original search looked for a consumer in tests and glue and did not look at what the backend itself
emits — the same shape as reading a name rather than the defining predicate.

The superseded original follows.

~~No known consumer: a search of the MC and CodeGen tests, the runtime glue and the compiler-rt
builtins found zero occurrences of `call` followed by a bare register and a symbol. So this is a hole
in what the assembler accepts versus what the instruction definitions declare, not a blocked user.~~

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

### R-21 — `cincoffset`/`scc`/`tighten`/`shrinkto` do not consume their LINEAR source, and `init` DUPLICATES it `PARTLY RESOLVED — `cincoffset` and `scc` CONFORMANT ON SILICON 2026-09-15 (boot sw8x-f4, six readings each with the instrument and conformance controls; cincoffset already noted gone at 5097eb166); `tighten`/`shrinkto` untested on silicon; the INIT half is R-25 (fixed on silicon 2026-09-09); nothing to report to the hardware side`

> **`tighten` and `shrinkto` COPY on silicon: R-21 CONFIRMED for both (2026-09-25, boot r42e3b,
> `caplifive_r42_6cbdaeeb4.bit`, image `cbf8cb41eb56c477`).** R1 `--series linear`:
> - **arm 8 `tighten-LIN` reads 0** and **arm 10 `shrinkto-LIN` reads 0**: the linear source survives
>   in both, i.e. a duplicate.
> - capstone-qemu reads 7 for `tighten` (it moves) and 0 for `shrinkto` (it copies).
> - Controls 9/11 read 1, and there was no trap.
> - Arms 2/3 (`cincoffset`/`scc`) read 7, as this entry's 2026-09-15 resolution says.
> - Arms 4/6 (`ldc`/`stc`) read 7 on silicon against QEMU's 0, which is Q-12.
>
> **CORRECTION to this box's first version (4c7e6b9).** It read the first boot, r42e3, as "`tighten`
> traps cause 29 on a linear source". The trap was a HARNESS ENCODING BUG, found by the RTL lane's
> reading of the RTL:
> - `.insn r` put an `"r"` operand in TIGHTEN's rs2 FIELD, which is the permission immediate. That
>   encoded the allocated register's number, a2 = 12, as imm 12.
> - The RTL refuses imm > 7 with cause 29, and QEMU clamps it to NA. Both are Q-13.
> - Fixed in `r1_slots_pools.c` with the literal `x6` (6 = RW). The linear source was never the
>   issue.
>
> Result lines: `tests/rtl-smoke/ladder-revival-2026-09-22/r42-e3-linear.result-lines.txt`, both boots.

> **On silicon 2026-09-15 (boot sw8x-f4, §7w of the measurements doc):** through the R1 harness's `--series
> linear` on `caplifive_r30r31_1bfff7776`, the LINEAR source of `cincoffset` and of `scc` reads cleared (7)
> after the operation in six readings each, the `movc` instrument control reads cleared and the NONLIN
> conformance control survives — the spec's behaviour, so the table's first three rows do not describe the
> deployed bitstream. `tighten` and `shrinkto` were not exercised on silicon and stay open by this entry's
> source reading. The emulator agrees on these three; its own deviations are on `ldc`/`stc` (Q-12).
> **Origin (RTL lane, from the history, 2026-09-15):** commit `b047f32eb` (2026-08-12, "Fixed some linearity
> enforcement issue"), an ancestor of the deployed `1bfff7776`, added the three cnull writebacks on the
> CINCOFFSET paths (with `cincoffsetimm` and `scc`) and shipped `cincoffset-linear-clear.S`,
> `cincoffsetimm-linear-clear.S`, `scc-linear-clear.S`. This entry's source table was read before that commit and
> not re-read after — a month stale by the time the board read it; the same may hold for other source tables here.


> **Sweep 2026-09-05 — cincoffset half GONE at 5097eb166; the INIT half is R-25.** `linear-clear-audit.S` arm 2 prints NOT_CAP (the linear source is consumed); R-25 (INIT with rd ≠ rs1 duplicates the source) is confirmed by directed test, see its entry.

**A linear capability can be copied.** `capstone-academic-spec/parts/intro.adoc:58-61` states the invariant
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

**2026-09-23 — re-read at the deployed RTL, the monitor checked, and the silicon probe built (not yet
run).**
- **Deployed RTL, `capstone-ariane 054cea69b`.** Both functions are still copies: `SHRINKTO`
  (`core/anvil_build/capstone_flu_unit.anvil:268-296`) and `TIGHTEN`
  (`core/anvil_build/capstone_dyn_unit.anvil:262-289`) always return the unmodified `rs1`. Neither has
  the `rd == rs1` / `LINEAR` → `cnull` branch that `CINCOFFSET` (`:29-52`) now carries, and no shared
  helper exists that they could route through. The line numbers above predate this revision.
- **QEMU is inconsistent in its own way.** It copies on `shrinkto`, like the RTL, but on `tighten` it
  moves (`op_helper.c:1176-1180`), like the spec. QEMU's `tighten` also delinearises a LINEAR result
  that has lost write permission (`:1185-1188`). The RTL does not do that; its downstream effect is not
  examined here.
- **Which code can hit it.** The duplication needs `rd != rs1`. The monitor image booted in the
  2026-09-22/23 ladder runs contains 9 `tighten` and 1 `shrinkto`, and all 10 are in place
  (`rd = rs1 = x5`). The check: `fw_payload.elf`, GNU objdump, 16,984 32-bit instructions decoded,
  2,481 of them opcode 0x5b. The scanner was positive-controlled on an image with one known
  `rd != rs1` tighten. So **the monitor never takes the duplicating path**, and what remains exposed
  is compiled or hand-written domain code that tightens or shrinks a linear capability into a
  *different* register.
- **Silicon probe.** Arms 8-11 of `sublet/r1/r1_slots_pools.c --series linear`: `tighten` to RW and
  `shrinkto 64`, each on a LINEAR source and a NONLIN control. Predicted from the RTL source:
  - on silicon, arms 8 and 10 read **0**, i.e. the linear source survives;
  - under QEMU, arm 8 reads 7 and arm 10 reads 0.

  **Emulator pass, 2026-09-24** (rootfs repaired): image `5852fb06f92394d0`, `R1_RC=0`, 2 reps,
  identical. Arm 8 (`tighten`, LINEAR) reads **7**, so QEMU moves. Arm 10 (`shrinkto`, LINEAR)
  reads **0**, so **QEMU duplicates the linear capability**. The NONLIN controls (arms 9 and 11)
  read 1, and arms 0–7 read as they always have. So QEMU matches the RTL on `shrinkto` (both
  copy) and the spec on `tighten`. The silicon reading is the open item: the arms ride the next
  board boot as a separate H1 record.

### R-22 — `stc` does not write `cnull` to its register source `RESOLVED ON SILICON 2026-09-15 for the deployed bitstream (boot sw8x-f4): the register after `stc` of a LINEAR capability reads cleared in six readings, the NONLIN control unchanged; fixed by b047f32eb (2026-08-12), an ancestor of the deployed RTL — the entry's analysis predates it; the emulator still omits the clear (Q-12); nothing to report`

> **On silicon 2026-09-15 (boot sw8x-f4, §7w):** `ldc t0 <- slot` (LINEAR), `stc t0 -> other slot`, then the
> type of `t0`: **7** — the register source IS nulled by `stc` on `caplifive_r30r31_1bfff7776`, six readings
> in two domains, with the NONLIN control reading 1 and the `movc` instrument control proving the read can see
> a clear. The memory-side half (`ldc` moving a LINEAR capability out of its slot) reads 7 too.
> **Origin (RTL lane, from the history, 2026-09-15):** commit `b047f32eb` (2026-08-12, "Fixed some linearity
> enforcement issue"), the only commit that ever touched `rs2_cleared` in the DYN unit: it builds a cnull and writes
> it back in place of the stored capability on both the UNINIT and the non-UNINIT STC path (eighteen lines), and
> shipped `stc-register-clear.S` and `r20-stc-ld-x10.S`. The `ldc` slot clear is NOT in that commit and stays
> unidentified. The source analysis below was read before that commit and never re-read after it landed — the
> entry stayed OPEN for a month on stale source, which is why the board's pre-registration followed it and lost.


`capstone-academic-spec/parts/mem-access-insn.adoc:105`: "If `x[rs2]` is a capability and `x[rs2].type` is
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

### R-24 — the FLU/DYN exception encoder is +1 off the spec, so every capability `mcause` from the execute path is wrong `FIXED IN RTL SIMULATION 2026-09-15 on r34-r24-exception-delivery (c77c65324, residuals e97b7e7ab) — the REFUTATION below stands and was answered rather than overturned: the collision is real, and the fix is to move the DEBUG_REQUEST sentinel off 24 (to 32) so the capability causes can sit on the spec's 24..29. Measured, not argued: debug_mode_q stays 0 across four cause-24 deliveries. SHIPS WITH R-34 AND CANNOT SHIP WITHOUT IT. **SYNTHESISED 2026-09-17** inside `m1-reclaimer` `054cea69b`, never alone -- no pair-specific timing or area figure exists; lint and sweep for the merge ARE pair-specific and are recorded under R-34. NOT on the board as of this line`

> # 2026-09-15 — THE RULING THE ENTRY ASKED FOR, AND THE FIX THAT FOLLOWS FROM IT
>
> The refutation below is correct and is not withdrawn: base 23 does put `UNEXPECTED_OPERAND` on 24,
> and 24 was `DEBUG_REQUEST`. What was missing was that **`DEBUG_REQUEST` is an internal sentinel on
> the exception bus, never an architectural `mcause`** — `csr_regfile.sv:2019` gates it out of the
> normal trap path and `:2208` consumes it into dcsr/dpc. Nothing requires it to be 24. So the
> collision is resolved from the other side: the sentinel moves to **32** (the block ends at 31, and
> no encoder ordinal reaches it), and the capability causes take the spec's **24…29**.
>
> That makes the two execute-path encoders agree with everything else in the tree for the first
> time. The four-way agreement on base 23 was already recorded here on 2026-09-10 and is unchanged;
> the only correction to it is that the LSU block and `commit_stage.sv:216-226` are **one authorial
> decision** (same author, both 2026-05-10), not two independent witnesses — the genuinely
> independent pair is the spec text and QEMU's `cpu_bits.h`.
>
> **R-24 AND R-34 CANNOT BE SPLIT.** `DebugEn` is 1 in
> `capstone_cv64a6_imafdc_sv39_config_pkg.sv:146`, so on an unrenumbered tree a delivered LSU
> `NOT_CAP` — cause 24, the commonest one — enters the debug ROM instead of trapping. It has never
> bitten only because R-34 means those exceptions are almost never delivered (raised 21 times,
> delivered once, in the audited run). Fix delivery without moving the sentinel and every one of
> them becomes a debug-mode entry. The waveform is the counterfactual made visible: `debug_mode_q`
> stays 0 across all four cause-24 deliveries on the renumbered tree.
>
> **The suite, swept against the baseline** (`4cc068572`; `core/load_store_unit.sv` byte-identical
> to the flashed `1bfff7776`). 27 tests regress, in exactly two groups and nothing left over:
>
> * **15 are cause-number assertions** and ALL FIFTEEN now pass, at cycle counts **bit-identical**
>   to baseline — 686=686, 617=617, 967=967, 673=673, 31043=31043 and so on, the three untouched
>   controls included. Identical counts, not merely green: the renumber changes nothing about how
>   they execute, only the number they assert. `c7b616b6e`'s updates applied unchanged, plus one
>   line that commit missed (`init-rs1-ne-rd.S:164`, code constant left at 25 while its header
>   comment was moved to 24 — it failed with its own "unexpected cause" exit code 20).
> * **12 are R-34's new enforcement**, not R-24's: see that entry.
>
> A third group of 15 was rechecked at a 10x timeout, because a shared 50,000-cycle budget can MASK
> a regression. Two were never broken — `s07-ldc-chain-forward` 276,242 and `stc-counter-pair`
> 215,359, identical on both trees — and the other 13 sit at 500,013 on both. The three groups are
> pairwise disjoint; the two of size 15 are a coincidence of count.
>
> **Residuals closed in `e97b7e7ab`:** `INSUFFICIENT_SYSTEM_RESOURCES` was left at 31 leaving a hole
> at 30 (spec and QEMU both say 30) — the same "moved four, left two behind" miss this entry
> criticises below, made again; `capstone_unit.anvilh`'s `ex_code` comments still read base 24; and
> that file's NOTE calling `commit_stage`'s base 23 "an off-by-one in its own right" is resolved the
> other way round — `commit_stage` was right.
>
> **One claim corrected in this entry's own favour, and one against.** The elaboration assertion
> added in `c77c65324` is **simulation-only** — it sits inside `//pragma translate_off`, so
> synthesis strips it and it does not protect the bitstream; the commit message that called it an
> elaboration check overstated it. And "the LSU is a newly found independent witness" was a
> re-derivation: this registry had it from 2026-09-10, and it was read again from scratch because
> the registry was not searched first.

> **THE FIX CANNOT SHIP AS WRITTEN. Base 23 puts `UNEXPECTED_OPERAND` — ordinal 1, the most common
> capability exception in the whole directed suite — on mcause 24, and `riscv_pkg.sv:348` already
> has `DEBUG_REQUEST = 24`.** The trap logic dispatches on that value: `csr_regfile.sv:2019` gates
> the path to `mtvec` on `cause != DEBUG_REQUEST`, and `:2208` takes it into debug mode. So after
> R-24 the exception never reaches the handler at all — it becomes a debug-mode entry.
>
> **Matched pair, one variable** (RTL lane). Pre-R-24 at `1bfff7776`: `cincoffset-linear-clear`
> PASS 686 cycles, `excode-base-audit` PASS 617. On the R-24 tree with expectations decremented to
> match the new base, those two and six more hang at the 400013 timeout — and the trace is what
> settles it: the test's own trap handler never retires an instruction while the core spins in the
> low debug-ROM addresses reading `dscratch0`.
>
> **A second defect in the same commit, independent of the first.** The localparam block it edited
> moved four of the six capability causes and left two behind, so `UNEXPECTED_OPERAND_TYPE` and
> `INVALID_CAPABLITY` are now both 25, and `ILLEGAL_OPERAND_VALUE` is still 30 where it should be
> 29. The stated reason for touching that block was to stop its constants misleading the next
> reader.
>
> **What narrows the ruling, and neither is a lane's call.** The collision exists *only* at 24 —
> every other cause under base 23 lands in 25…29 and hits nothing. And R-24 has no firmware half:
> the monitor's trap entry dispatches only on the interrupt bit and supervisor-ecall and never
> branches on 24…30, which is also why nothing on the firmware side would have caught this.
>
> **THE FLASH IS UNAFFECTED, verified by content from this side rather than taken.**
> `git diff 1bfff7776 69658cf16 -- core/cva6.sv core/ex_stage.sv` reads `-64'd24 +64'd23` once per
> encoder: the bitstream being flashed carries the **old base 24**, so ordinal 1 lands on mcause 25
> and misses `DEBUG_REQUEST` entirely. `r24-excode-base` is also not an ancestor of `1bfff7776`.
>
> **Tests were updated anyway** at `c7b616b6e` on `r24-excode-base`, anchored
> `backup/r24-tests-collision-2026-09-11`. Full note:
> `docs/history/11-09-2026_19-30-00_r24-debug-request-collision.md`.
>
> **⚠ A RETRACTION THAT GENERALISES PAST R-24.** That commit recorded nine host-sweep timeouts as a
> harness failure rather than as readings. They were readings, and they were this defect. The
> standing rule is that a surprising CLEAN result should make you suspect the instrument; this is
> the same rule run the other way — **a surprising FAILURE earns the same suspicion, and has to be
> discharged with a control rather than assumed away.** That direction is the likelier one to go
> wrong in, because a failure that is "obviously the harness" feels like it needs no evidence at
> all, whereas a suspicious pass at least prompts a check. Two tells were available before any
> rerun and both were cheap: the failure was **not uniform** (`capsbi-init` kept passing), and the
> trace showed a **mechanism**, which a harness fault does not have.

> # 2026-09-10 — R-24 IS NOT "THE RTL DEVIATES FROM THE SPEC". THE RTL DISAGREES WITH ITSELF, TODAY, ON THE FLASHED PART.
>
> Found sideways, while fixing the discriminator for an unrelated board probe. The **load/store unit
> emits capability causes RAW** — it does not go through the execute path's `24 + enum` encoder — and
> every one of its numbers is the spec's (`load_store_unit.sv:974-990` at `66c4e7517`, read directly):
>
> | LSU condition | LSU emits | spec name and number |
> |---|---|---|
> | `NOT_CAP` operand | 24 | Unexpected operand type — 24 |
> | invalid revocation node | 25 | Invalid capability — 25 |
> | not LINEAR/NONLIN (wrong type) | 26 | Unexpected capability type — 26 |
> | load without read permission | 27 | Insufficient capability permissions — 27 |
> | store without write permission | 27 | same |
> | address out of bounds | 28 | Capability out of bound — 28 |
>
> **Six for six.** So the LSU is a **THIRD independent witness** that the spec base is `23 + ordinal`,
> alongside the spec text itself and `commit_stage`'s PC-capability check. Against those three stand the
> two execute-path encoders, which add 24 and land one high on every cause.
>
> **The consequence is sharper than a spec deviation and much harder to argue with: the same logical
> fault reports a DIFFERENT NUMBER depending on which unit raised it.** A capability-type error is
> **26** from the load/store path and **27** from the execute path. Insufficient permission is **27**
> from one and **28** from the other.
>
> **WHICH HISTORICAL VALUES ARE ACTUALLY AMBIGUOUS — precisely, so past readings can be AUDITED rather
> than written off.** The LSU emits 24-28; the execute path emits `24 + ordinal`, i.e. 25-30. The
> overlap is 25 through 28 and only that:
>
> | latched mcause | raised by | reading |
> |---|---|---|
> | **24** | LSU only | `NOT_CAP` operand — **unambiguous** |
> | **25** | BOTH | LSU invalid revocation node **or** execute `UNEXPECTED_OPERAND` |
> | **26** | BOTH | LSU wrong capability type **or** execute `INVALID_CAPABILITY` |
> | **27** | BOTH | LSU insufficient permission **or** execute `UNEXPECTED_CAP_TYPE` |
> | **28** | BOTH | LSU out of bounds **or** execute `INSUFFICIENT_PERMISSION` |
> | **29** | execute only | `OUT_OF_BOUNDS` — **unambiguous** |
> | **30** | execute only | `ILLEGAL_OPERAND_VALUE` — **unambiguous** |
>
> So a blanket "historical causes are ambiguous" would be wrong and needlessly destructive: **24, 29 and
> 30 pin down their unit**. Only 25-28 need the raising instruction to disambiguate. **One filed claim
> is affected — see R-25 in the archive**, whose header names a cause in the ambiguous band.
>
> **This has already cost readings and will cost more.** Anyone correlating a board wedge's latched
> `mcause` in 25-28 against the RTL must first know which unit raised it, and **the wedge tracer does
> not record that**.
>
> **Where the tracer fix does NOT belong: with R-24.** Recording the raising unit means a new signal
> into the tracer path, and that is precisely the kind of addition that took the R-29 candidate from
> UNOPTFLAT 40 to 41. R-24's whole virtue is that it is two operators with no new term; putting a cone
> risk into the one change that currently carries none would be a bad trade. It belongs with the DRIVER
> work, where it can be done in **software** — record which instruction faulted and infer the unit from
> its opcode — with no RTL at all. It is also exactly why the `lsugate` probe's discriminator is 26 rather than 27 — I wrote 27
> first, by applying the execute path's convention to a unit that does not use it, and caught it only
> because the value was about to be read off a board.
>
> After R-24 the number stops depending on the unit. That is the argument for the change, and it is a
> better one than conformance: **an implementation that reports one fault two ways is unreadable
> regardless of which numbering anyone prefers.**
>
> *Nothing on the R-24 branch changes for this — it remains the ungated one-liner with its test updates
> owed. This strengthens the case, not the patch.*

> **2026-09-10 — A FIX IS WRITTEN AND IS EXPLICITLY NOT GATED. Do NOT batch it into a bitstream yet.**
> RTL lane, branch `r24-excode-base` at `69658cf16` off `66c4e7517`.
>
> **Direction re-confirmed against the spec, because the RTL's own comment said the opposite.**
> `capstone-academic-spec` gives 24/25/26/27/28/29 for unexpected operand type, invalid capability, unexpected
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

**RESOLVED 2026-08-12 against `capstone-academic-spec`, and the answer is the opposite of the first guess
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

> **A FOURTH INDEPENDENT WITNESS, from a different implementation (2026-09-10).** The reclaim work
> gave one for free. `uninit_init_then_use_ok` faulted under QEMU with **cause 29** for an `INIT`
> whose cursor had not reached `end` — the spec names *Illegal operand value* **29**, and the RTL's
> execute path would emit `24 + ordinal` = **30** for the same fault.
>
> So the spec text, `commit_stage`, the load-store unit **and now the emulator** all agree the base
> is **23**, against the two execute-path encoders. Four witnesses from three implementations is a
> harder case to argue with than three from two, and this one was not constructed for the purpose —
> it fell out of an unrelated gate. *(Noticed by the RTL lane, recorded here.)*

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

Verified against `capstone-academic-spec`: **NOT YET.** The spec's exception numbering has not been checked
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
exactly one build script (`ports/sqlite/build-sqlite-silicon.sh`); no suite below passes it,
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

> **Source-read entries carry the commit they were read at.** An entry whose evidence is "I read the RTL /
> the emulator / the compiler and it does X" has a shelf life nothing tracks: when the source moves, the entry
> stays as written, and an OPEN status reads as caution rather than as staleness — the safe-looking direction.
> R-21/R-22 sat OPEN for a month after `b047f32eb` had fixed them (2026-09-15: the board's pre-registration
> followed the stale status and lost). So: name the commit the source was read at (`read at <sha>`), and when a
> board reading contradicts an entry, re-read the source at the deployed revision before anything else.

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
