# S-07 — handover to the RTL lane

> ## UPDATE 2026-08-16 (LATER, after four boots on `caplifive_s07diag.bit`) — READ THIS FIRST
>
> This supersedes the block below it, which was written before your diagnostic bitstream was
> flashed. Where the two disagree, this one wins; the older block is kept for its history.
>
> ### 1. THE ASK: `mtval` is written and unreadable. Put `tval` in the LATCH.
>
> Your instrument is correct — and the dump you told us to read it from never runs. A capability
> fault inside a capability domain **wedges at exception commit** instead of trapping to `mtvec`,
> so the monitor's `EXCX/MCAU/MEPC/MTVL` block is never reached. Your own RTL states it at
> `core/cva6.sv:1228-1231`. Matched pair, same capture code, same board:
>
> | latched cause | `EXCX` | `MCAU` | `MTVL` | other live monitor markers |
> |---|---|---|---|---|
> | mcause **8** (3 runs) | 1 | `00000008` | 1 | 3 |
> | mcause **25** (6 runs) | **0** | — | **0** | 6–18 |
>
> The mcause-8 rows are the fired positive control: that path works and does print `MTVL`. The
> mcause-25 rows still emit 6–18 monitor markers, so the console was live — the handler simply
> never runs. Wedge state agrees: `excommit=1`, `flush=1`, `privM=1`. The debug latch cannot
> supply it either — `cva6.sv:994-996` / `:1097-1099` carry `trap_seen`/`mcause`/`mepc` and **no
> tval**.
>
> **Please latch `tval` in the same `always_ff` block that already captures mcause and mepc, and
> expose its 8 bytes on the free slots of debug bank `3'b110`.** The mechanism is already proven:
> that latch captured mcause=25 *and* mepc at these very wedges, so a `tval` sibling register
> latches identically. It composes with your `capstone_dyn_ftval` routing, which already delivers
> the cursor to `tval`. This covers every site at once and needs no reproducer.
>
> ### 2. Your "the failing workload IS the probe" reasoning is measured-false
>
> Four boots; **no wedging run has ever reported anything**, and the reason is structural:
>
> | build | site instrumented | wedged at |
> |---|---|---|
> | `S7C` | `sqlite3OsRead` | `pagerFreeMapHdrs+0x4c` |
> | `S7P` | `pagerFreeMapHdrs` | `sqlite3OsRead+0x4c` |
> | `S7B` | BOTH | `sqlite3BackupRestart+0x5c` |
>
> Each build dies at whichever vulnerable site the previous one covered, so a software probe can
> never be the thing that fires. And a wedge discards the retval *and* everything `output_text`
> buffered — the host reads that buffer only when the domain RETURNS. A wedging run has no
> reporting channel at all. That is the argument for the latch.
>
> ### 3. THE INVARIANT — one instruction shape, three unrelated functions
>
> Three wedges, three builds, three functions, three addresses, byte-for-byte the same shape:
> **two ADJACENT `ldc`s where the second's rs1 is the first's rd.**
>
> ```
> sqlite3OsRead+0x4c   (S7P)      pagerFreeMapHdrs+0x4c (S7C)     sqlite3BackupRestart+0x5c (S7B)
>   3a8d0: ldc a0, 0x0(a0)          43368: ldc a1, 0x0(a0)          40bc0: ldc a0, 0x0(a1)
>   3a8d4: ldc a4, 0x0(a0)
>   3a8d8: ldc a4, 0x20(a4)  <==    4336c: ldc a1, 0x40(a1) <==     40bc4: ldc a0, 0x70(a0) <==
> ```
>
> In every case the value **produced by the immediately preceding `ldc`** arrived NOT_CAP. This is
> the back-to-back dependent capability-load pair. You refuted A-1 overwrite and hit-under-miss on
> the grounds that the dyn unit serialises cap loads at issue — that refutation is about the
> *tracker*; it does not by itself cover "the second load's rs1 operand is consumed as NOT_CAP",
> which is the LOAD_WB erasure consequence you CONFIRMED. Worth re-examining with this shape in
> hand.
>
> ### 4. H1/H2: H2 refuted at one site, H1 NOT established anywhere
>
> At `pagerFreeMapHdrs+0x4c`, H2 dies by control flow: the loop condition two instructions earlier
> reads the SAME stack slot with a plain integer `ld` and branches away if zero, with `a0`
> rederived from `s0` and no intervening store — so reaching the fault proves the cursor was
> non-zero.
>
> But **H1 is not established, and we are not claiming it.** In that build the field can hold no
> legitimate capability: `SQLITE_MAX_MMAP_SIZE` is 0 (`sqlite3-capstone.c:16156`; the `__OpenBSD__`
> arm at `:16137` is dead — do not cite it), and `pagerAcquireMapPage`, the only setter of
> `PGHDR_MMAP`, is **absent from the binary**. The writer at `:63849` *is* compiled in and callable,
> gated only by that flag bit, so the accurate phrase is **"no reachable writer under intact
> data"**. A non-zero cursor in such a slot means **the memory held wrong data** — not the same as
> a lost tag, which leaves the payload intact and clears only the tag. Surviving readings:
>
> 1. the field held untagged garbage on entry (zeroing failure / heap corruption);
> 2. a corrupted `flags` bit let `pagerReleaseMapPage` really run — the only reading under which
>    the S-07 tag-loss story is right;
> 3. the `ld` itself returned a wrong non-zero over a true zero, making the `ldc` a *correct* null
>    deref and the defect an integer-load-path one — note this is the same class as the `ldc`
>    delivery failure at the other two sites, so it is not an exotic alternative;
> 4. a wild-but-tagged `pPager` reading unrelated memory at `+0xf0`.
>
> Latched `tval` discriminates (1)/(2)/(4) directly.
>
> ### 5. Solid
>
> * selftest control returned its exact PASS value `0x57070703` on **every** boot — the `ld`-based
>   instrument is proven on this bitstream, zero-reads included;
> * `sqlite3OsRead` is **never called in a clean run** (`calls=0`, full extended workload passing):
>   reachable only after an upstream error, so it was always the SECOND fault;
> * clean reps are NOT evidence of suppression — at the observed rate 3 clean is p≈0.30.
>
> ### 6. Operational, since you can drive the board
>
> * `split_out_cap`'s unimplemented exact-fit case caps a boot at **~4 domains**; the 5th spins at
>   `SPLB` with no `SQ: A/dom-ok`. The monitor's own comment records this once "manufactured a
>   confident, entirely false localization of a SQLite function that never executed".
>   Discriminator: `SQ: G/enter` present, plus the latch cause.
> * `XU.dom` (`f1214600`) is byte-identical to the historical `XF` reproducer.
> * `run_sqlite_stages_fpga.py` now attempts a GDB `mtval` read at every wedge, accepted **only**
>   if gdb's own mcause/mepc match the latched pair. Stopgap, not a substitute for the latch.
>
> ### 7. Retractions that stand — please do not let us re-assert them
>
> 1. `cincoffset` has a two-armed guard; only `ldc` sites and `sqlite3_strnicmp+0x134` are
>    rs1-unambiguous.
> 2. Image composition did not suppress the defect.
> 3. The hostcall/domain-boundary claim is false (`:memory:`, no crossing).
> 4. "Every synthetic shape excluded" is void (no `_SELFTEST` ran on that bitstream; `s07evict`
>    assumed a 64 B line where `DcacheLineWidth` = 128 **bits** = 16 B, write-through
>    no-write-allocate; rungs bounded p at only 6–19%).
> 5. NEW — **"the site wanders" is withdrawn.** Of 8 mcause-25 wedges, 6 have a recoverable image
>    VA and five of those six are `sqlite3OsRead+0x4c` in five *different builds*. Earlier wording
>    (ours and an auditor's) compared raw `mepc` across builds and misread link addresses as
>    wandering.
> 6. NEW — **"a real capability arrived NOT_CAP" was never established.** See §4.

> ## UPDATE 2026-08-16 (EARLIER, pre-diagnostic-bitstream) — superseded by the block above
>
> ### 1. One question is left, and one observation on your side settles it
>
> Every wedge is `ldc a4, 0x20(a0)` where `a0` was just reloaded from a stack slot. `ldc`'s guard is
> rs1-only, so `a0` is genuinely NOT_CAP. **What we cannot determine is which NOT_CAP it is:**
>
> * **H1 — a real capability that lost its tag.** S-07 as advertised, your problem.
> * **H2 — `pMethods` is legitimately NULL.** Then mcause 25 is the architecturally correct
>   rendering of a NULL dereference, this site is **not a defect at all**, and S-07's real fault is
>   somewhere upstream.
>
> **The discriminator is the cursor of the faulting register: zero → H2, non-zero → H1.** You can
> read that directly. I cannot — three probes failed to, for reasons in §3.
>
> ### 2. Why this site is a symptom, not an origin — this is the big reframing
>
> The database is `sqlite3_open(":memory:")`. On a memory database **a clean run executes ZERO
> `sqlite3OsRead` calls**: every call site on the main file is unreachable (`pMethods` is 0 forever,
> the only `sqlite3OsOpen` sits behind `if(zFilename && zFilename[0])`), and the only reachable ones
> go through **in-RAM memjournal playback during rollback**. Rollback only happens after an error.
>
> **So reaching `sqlite3OsRead` at all means SQLite had already failed upstream, inside
> `sqlite3_step`. The wedge we have all been looking at is the SECOND fault of the run.** That also
> means "the same construct in two binaries" shows the same *death site* — which is the same
> *defect* only under H1.
>
> Correction you should apply to your own answer: it cites the site as reached "through
> `pMethods->xRead` (a hostcall)" and proposes the domain boundary as the thing to look at first.
> **There is no hostcall and no boundary crossing** — it is in-RAM memjournal playback, entirely
> inside the domain. That claim was mine originally and I withdrew it; sorry for the detour.
>
> ### 3. What I tried, and the exact reason software cannot close it
>
> Three probes, each fixing the previous one's flaw, none sufficient:
>
> * LCC field 1 answers **7 for a lost tag and 7 for integer 0** — cannot separate H1 from H2. Added
>   a null-cursor check.
> * **A checked value does not stay checked.** At `-O0` the compiler spills and reloads between the
>   check and the dereference. My point-of-use guard *is* emitted — the `SQLITE_IOERR_READ` early
>   return is in the disassembly — and the guarded path still does `ldc a0, 0x0(a0)` from the stack
>   before the deref. The value I verify is never the value that faults.
> * **A wedged domain cannot report.** Counters come back only at end of run.
>
> Closing it from software needs the check and the deref in **one inline-asm block** with no spill
> between. Buildable. Say the word and I will build it — I am not doing so unprompted because you
> can read the register directly and that is one observation against my several boots.
>
> ### 4. What I retracted since the last message — do not act on the old versions
>
> * **"Every synthetic shape is excluded" — VOID.** No `_SELFTEST` arm has run on this bitstream, so
>   every `65535` came from an instrument never shown able to return anything else; `s07evict`
>   assumed a 64-byte line when it is **16** (`DcacheLineWidth = 128` bits) so it evicted at most 4 of
>   16 slots, and the cache is write-through **no-write-allocate** so the spill never allocates
>   anyway; and 16-48 samples bound the rate only at p > 6-19% while the defect lives near 10⁻⁶.
> * **"Bisect SQLite downward" — withdrawn.** The failure is not deterministic: identical `L2.dom`
>   passed, passed, then wedged. A one-run-per-stage bisection would call failing stages clean about
>   two times in three.
> * **`L2.dom` does not contain the extended workload** — it is a `CREATE_LADDER=2` build. The
>   smallest artifact that has ever wedged is: `config(HEAP)` → `initialize` → `open(":memory:")` →
>   prepare + step one `CREATE TABLE` → return.
> * An `s06spill` run I reported as an R-16 entry stall was a **missing artifact** (`open .dom
>   failed`); our classifier called it an RTL phenomenon.
>
> ### 5. What I can still do, on request
>
> Board runs, domain builds, the inline-asm probe, or a properly-powered rung (~10⁷ iterations with
> its selftest arm in the same boot — the existing rungs are five orders of magnitude short). The
> board is serialized between us; tell me when you want it and I will stay off.


You root-caused and fixed S-08 in a few hours off a report whose mechanism was wrong, so this is
written to a peer who can build, boot, debug and measure, not to someone who only edits RTL. Where
something needs doing rather than deciding, I say so and hand it over rather than asking a question.

## 1. First, thank you, and two confirmations you may want

* **Your S-06 fix passes on silicon.** `s06agg` 5 → **15**, `s06aggcap` 7 → **15**, `s06aggwide`
  237 → **255**, on the *unfixed* rungs with no software workaround in the build, in a
  control-validated boot (`k800` = 4). That is the exact criterion `ref/S06-WORKAROUNDS-TO-REVERT.md`
  set as decisive.
* **Your S-08 fix works.** Domains run again; `EXCX:0000E002` is 0 where it was 4-of-4. Verified on
  `caplifive_s06fixs08fix.bit`.
* **The workarounds are gone.** I reverted §1 — the `-capstone-guard-cap-granule-copies` pass, its
  intrinsic and pattern, its lit test, and the library `memcpy` fixup. lit passes; 14 of 15 QEMU
  suites pass (the 15th is a pre-existing inverted probe). SQLite now completes on silicon **through
  finalize with nothing compensating for the hardware** — first time.

Also: your note that no green directed test drives the dom-switcher, and that CAPENTER doesn't drive
it either, is recorded on our side as a coverage gap independent of S-08.

## 2. What is blocking now

**S-07 survives your fix and is easier to hit than before.** Two wedges on the current bitstream:

```
L2.dom  (sha fd0445cf…)   3a2f0: ldc a0,0x0(a0)  3a2f4: ldc a4,0x0(a0)  3a2f8: ldc a4,0x20(a4)  <== mcause 25
XF.dom  (sha f1214600…)   3a834: ldc a0,0x0(a0)  3a838: ldc a4,0x0(a0)  3a83c: ldc a4,0x20(a4)  <== mcause 25
```

Different builds, different addresses, **same function `sqlite3OsRead+0x4c`, byte-identical triple**.
So it follows the source construct, not the image layout. `L2` is a truncation arm that returned in
every pre-fix transcript we have and had never wedged.

**The fault is at an `ldc`, whose guard is rs1-only** (`capstone_dyn_unit.anvil:327-330`, and `LDC`
contains exactly one `UNEXPECTED_OPERAND` raise — rs2 is bound but never tested). So unlike every
earlier `cincoffset` instance there is no second arm to explain it away: **a capability produced by
an `ldc` arrived untagged**, and the immediately dependent `ldc` raised on it.

That is the whole solid core. Everything else in the folder is weaker than it, and I have tried to
mark it as such.

## 3. What of mine you should NOT trust

I burned three retractions on this issue today. Rather than let you rediscover them:

* **Both minimal reproducers are void.** `s07chase` (dependent chase) and `s07indep` (meant to be
  independent loads) return 0 on silicon — but neither ever puts two capability loads in flight.
  Dependent loads can't overlap by construction, and at `-O0` `s07indep` spills every result
  immediately (18 of 43 `ldc`s consumed one instruction later). Both had *firing positive controls*,
  which proved the detector worked and said nothing about whether the trigger existed.
* **Therefore A-1 is UNMEASURED, not downgraded.** I briefly wrote it up as weakened. Withdrawn.
* **An earlier `cincoffset`-based reading of this defect was ambiguous** (that guard has two arms).
  Only the `ldc` instance and `sqlite3_strnicmp+0x134` are rs1-unambiguous.
* The old **23% rate table is baseline-invalid** — different bitstream. We have two post-fix wedges
  and deliberately quote no rate.

## 4. What I would ask you to take

You can do all of this yourself, which is why I'm handing over tasks rather than questions.

1. **The board-free A-1 assertion.** Assert that no `LOAD_WB` writeback ever carries an `ldc`'s
   `trans_id`, negative-tested by forcing the bypass so you know it can fail. The bypass chain is
   quoted and line-verified in `MECHANISMS-AND-PATCH-PROPOSAL.md` (`scoreboard.sv:322-324`,
   `:241-247`, `commit_stage.sv:279`, `issue_read_operands.sv:1578`). It needs no reproduction and no
   board, and it either kills our leading candidate or hands you the defect.
2. **A rung that actually creates the condition** — inline asm, a burst of independent `ldc`s with
   nothing between them, **disassembly-verified before spending a boot**. Ours were not, twice. If
   you'd rather I build it, say so and I will; I'm offering it to you because you have the simulator
   to check the condition exists before it ever reaches silicon.
3. **What separates what wedges from what doesn't — still open, and we no longer have a candidate
   we believe.** We have re-verified all four exclusion rungs on the CURRENT bitstream (bounds,
   stores-through, scalar-load, and spill/reload across three redraws): all still `65535`. So the
   `stc` → `ldc` stack round trip the board localized is **necessary but not sufficient**. What is
   left that differs: a much larger working set and cache footprint, a capability chain rooted in
   heap structures rather than a static array, and far more capability traffic. We are testing
   cache/working-set pressure next.

   *(An earlier version of this item claimed the faulting path crosses the domain boundary via a
   hostcall VFS. It does not — the database is `:memory:`. Withdrawn; see the box below.)*

> ### CORRECTION 2026-08-15 — the "domain boundary / hostcall VFS" claim is WITHDRAWN
>
> We wrote that `sqlite3OsRead` reaches a hostcall-based VFS and therefore crosses the domain
> boundary, and offered that as the ingredient the rungs lack. **That is wrong.** The database is
> opened with `sqlite3_open(":memory:")` — SQLite's in-memory backend, entirely inside the domain.
> There is no file I/O and no boundary crossing on that path at all.
>
> So the distinguishing ingredient is **unknown**, not "the boundary". What remains different
> between the failing site and the passing rungs: a much larger working set and cache footprint, a
> capability chain rooted in heap-allocated structures rather than a static array, and far more
> capability traffic overall. Cache/working-set pressure is the leading remaining candidate and is
> the next thing we will test.

## 5. Division of labour I'd suggest

You have the Verilator model and the RTL; I have the compiler, the domains and the board harness.
Concretely: **you take the simulation questions (1) and any RTL change; I take domain-side
instrumentation and board runs on request.** If you want a specific domain built — a particular
instruction sequence, a particular workload, a bisect — ask and I'll bake and boot it. The board is
serialized between us, so tell me before you take it and I'll stay off.

## 6. UPDATE — every synthetic shape is excluded; the reproducer is still SQLite

Since writing the above I ran five more rung experiments on the current bitstream, all
control-validated, while the SQLite domain wedges reliably on the same silicon:

* `s06spill` re-run across **three redraws** (the first hit an R-16 entry stall and carried no
  verdict): **65535** each — 48 spill/reload round trips, every one tagged;
* `s06bnds`, `s06wr`, `s06pld`: **65535** — so the whole exclusion table is now re-verified on the
  silicon that actually fails, rather than inherited from a dead bitstream;
* **`s07evict`: 65535.** This is the important one. The board localized the fault to a spill and a
  reload, `s06spill` said that shape alone is sound, so the obvious missing ingredient was cache
  pressure. `s07evict` adds exactly that — a 48 KiB walk between store and reload — and the walk
  was **verified in the disassembly to sit in that window** before the boot was spent, which is the
  check whose absence made the two earlier rungs void. Still clean.

**No construct we can build in isolation reproduces S-07.** We are stopping rung construction: five
shapes excluded plus two void attempts is enough to say the approach is not converging.

**What we think is worth doing instead**, offered rather than assumed: bisect the SQLite workload
*downward* — cut the failing domain back until it stops wedging — rather than building synthetic
shapes upward. Every reduction step is a measurement on the one artifact that does fail. We can run
that here; it is board work and it is ours under the split. Say if you would rather we did something
else with the board time.

## 7. What I'm doing next unless you'd rather I didn't

* Rebuilding the reproducer properly in inline asm, with the emitted burst verified by disassembly
  before any boot.
* Tightening artifact identity: our driver doesn't print the hash of the `.dom` it actually loads
  from the initramfs, which is a weaker standard than this folder demands elsewhere.

Everything is on `capstone-bootstrap`. The rest of this folder is the evidence; `00-README.md` is the
entry point and `MECHANISMS-AND-PATCH-PROPOSAL.md` has the ranked mechanisms with what would kill
each.
