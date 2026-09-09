# SQLite stock-ness, and where the benchmark effort goes next

*Plan for the lead, approved 2026-09-09. The work it describes lands on a branch, is audited by the
board lane, is validated on silicon, and only then merges (see Delivery). Everything below rests on
read-only research; at the time of writing nothing was built and nothing was run. Line references are
against dev tip `0148e06a8d59`.*

## Context

Two questions were asked: does SLT passing on silicon mean SQLite works on silicon, and should the
next effort go into SQLite or into other benchmarks.

The answer to the first is no, and the reason is the build, not the corpus. Our SQLite carries
seventeen `SQLITE_OMIT_*` defines (`benchmarks/sqlite/build-sqlite-capstone.sh:92-120`). The
measurements doc already says so (`docs/ref/fpga-silicon-measurements-for-paper.md:921-926`:
*"17 features omitted, floating point among them, so the R column type is never compared"*). More
test records do not shorten that list. So: shorten the list cheaply, cost the expensive part
honestly, and put the remaining effort into breadth.

Decisions already taken by the lead, which this plan follows: restore the cheap features now; cost
floating point honestly but do not start it without a separate go; and **postpone the benchmark
breadth work** (2026-09-09). The survey behind it is kept at the end of this document so it does not
have to be redone, but nothing in it is scheduled. Active scope is Part 1, with Part 2 costed and
waiting on one decision.

Five facts the plan rests on, each verified today:

- **The oracle harvests the define list out of the build script's text.**
  `build-slt-native.sh:44-46` runs `sed -n '/^SQLITE_DEFINES=(/,/^)/p'` over
  `build-sqlite-capstone.sh` and greps `-D...` out of it, minus six deliberately excluded
  OS/allocator defines (`:43`). That is why the domain and its oracle have stayed in step, and it is
  the first thing any gate must not break.
- **There is no env hook into the amalgamation on the QEMU build path**, but there is one on the
  silicon path: `DOMAIN_EXTRA_DEFS` lands after `$SQLITE_DEFINES` on the amalgamation command line
  (`build-sqlite-silicon.sh:2665-2668`). It is invisible to the oracle, which makes it a fine
  *measurement* tool and an unsafe *correctness* tool.
- **Size headroom must be computed, not eyeballed.** `.text` is 1,331,080 B today; the undeclared
  ceiling is 1,376,256 B; the domain therefore declares `.capstone_domreq`, after which two separate
  allocations must each fit, and the order limit is read from the kernel config rather than assumed
  (`tests/runtime-qemu/silicon-ladder/domdata-budget.py:26-51`; it reads the ceiling from
  `caplifive-buildroot/configs/kernel.config` and refuses to guess, so a worktree run needs
  `CAPSTONE_BUILDROOT_DIR` pointed at the main checkout). The globals offset is recomputed from measured
  `.text` on every build (`build-sqlite-silicon.sh:2808-2831`), so a bigger image relocates globals
  by itself.
- **Floating point is cheaper than it looks, and its risk is elsewhere.** The soft-float runtime
  already exists and is validated by eleven floating-point BEEBS domains
  (`docs/design/capstone-softfloat-libm.md:1-8`), and its builtin set
  (`benchmarks/beebs/build-beebs-softfloat-common.sh:19-30`) already includes `divdf3`, which the
  two SQLite builds do not link. The real hazards are the one-module globals rule under
  `-capstone-gp-captable` (`build-sqlite-silicon.sh:15-18`, and the hand-written globals-free
  `__floatdidf` that exists because of it) and a silent failure: the silicon build appends builtin
  objects with `2>/dev/null && BUILTIN_OBJS+=(...)` (`:2785-2787`), so a builtin that fails to
  compile is dropped without a word.
- **JSON has no floating-point guard at all.** Its block (`sqlite3.c:212849-218463`) uses
  `sqlite3_value_double` and `%!0.17g` unconditionally and compiles under the omission only because
  SQLite substitutes `#define double sqlite_int64` (`sqlite3.c:634-637`). Re-enabling JSON without
  floating point would produce a build whose "reals" are integers. JSON therefore waits for floats.

## Part 1 — the cheap tier: "stock minus what the platform cannot provide"

### Step 0, before any code change: measure the size cost in minutes

Build the silicon image once with
`DOMAIN_EXTRA_DEFS="-USQLITE_OMIT_EXPLAIN -USQLITE_OMIT_FOREIGN_KEY -USQLITE_OMIT_UTF16 -USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_GET_TABLE -USQLITE_OMIT_DEPRECATED -USQLITE_OMIT_COMPILEOPTION_DIAGS"`
and run `tests/runtime-qemu/silicon-ladder/domdata-budget.py` on the resulting `.dom`. This answers the only question that can
kill the step, whether the image still fits its allocations, without touching a tracked file. If it
does not fit, stop and report; everything below assumes it does.

### The gate, designed around the harvest

Keep `SQLITE_DEFINES` exactly as it is, so the existing sed harvest keeps working and the default
build stays byte-identical to every recorded board result. Add a **second literal array** beside it,
`SQLITE_RESTORE=( -USQLITE_OMIT_EXPLAIN … )`, appended to `COMMON_FLAGS` only when
`SQLITE_FEATURE_SET=restored`. Then teach `build-slt-native.sh` to harvest that second block under
the same variable, and widen its grep from `-D` to `-[DU]`. Its existing sanity guard
(`(( ${#DEFS[@]} > 10 ))`, `:47`) stays and is what catches a block that has moved or been renamed.

Restored (all code-size choices, not platform limits): `OMIT_EXPLAIN`, `OMIT_FOREIGN_KEY`,
`OMIT_UTF16`, `OMIT_INCRBLOB`, `OMIT_GET_TABLE`, `OMIT_DEPRECATED`, `OMIT_COMPILEOPTION_DIAGS`, and
`UNTESTABLE` (dropping the last also restores the PRNG and allocator hooks an OOM ladder would need
later). Note `UNTESTABLE` is one of the six the oracle deliberately excludes, so removing it needs a
matching edit to the `EXCLUDE` list at `build-slt-native.sh:43` or the two sides silently diverge.

Kept, and the write-up says why: `OS_OTHER`, `THREADSAFE=0`, `TEMP_STORE=3`, `ZERO_MALLOC` with
`ENABLE_MEMSYS5`, `OMIT_WAL`, `OMIT_MMAP`, `OMIT_SHARED_CACHE`, `OMIT_TEMPDB`, `OMIT_LOAD_EXTENSION`,
`OMIT_LOCALTIME`, `OMIT_AUTOINIT`. Those are a domain with no OS, no filesystem, no threads and no
host allocator. `DQS=0` also stays: it is stricter than stock, not weaker.

### Then, in order

1. Build both sides restored. Record `.text`, image size and the `domdata-budget.py` verdict against
   today's numbers. A change of region or heap class stops the step, because it invalidates every
   staged image and its `qemu-pass` record.
2. Re-baseline the oracle. `benchmarks/sqlite/slt/check-negative-control.sh` asserts an exact
   per-arm tally and is the gate against a comparator that has stopped discriminating. Re-derive the
   aggregate-function file's tally; it uses `avg()` and real-typed rows
   (`slt_lang_aggfunc.test:34,87,101,138`), so some of its ten known query failures are artefacts of
   the omissions and are expected to move. Put both old and new tallies in the commit message.
3. QEMU: `SLT_TEST=select1.test bash benchmarks/sqlite/run-sqlite-slt.sh`, plus the negative control
   and the aggregate-function file, against the rebuilt native oracle.
4. The board boot, after the audit; see Delivery below for the sequence it sits in. Expected
   identical to the QEMU readings.
5. Rewrite the caveat list at `docs/ref/fpga-silicon-measurements-for-paper.md:921-926` with the new
   count and a split between platform limits and remaining choices. That sentence is the deliverable.

Cost: about half a day plus one board boot.

## Part 2 — floating point: costed, and cheaper than assumed

Not started without an explicit go, but the estimate in the earlier draft was too pessimistic and is
corrected here.

The runtime exists. Eleven BEEBS domains already build with floating point enabled through
`build-beebs-softfloat-common.sh`, whose builtin set includes the `divdf3` that SQLite's
`rB /= rA;` (`sqlite3.c:98553`) needs and that neither SQLite build links today. The work is:

1. **Fix the silent drop first** (`build-sqlite-silicon.sh:2785-2787`): make a builtin that fails to
   compile fail the build. This is a gate that cannot currently fire, and Part 2 is exactly the
   change that would walk into it.
2. Add the missing builtins from the BEEBS set, screening each for file-scope constants: any that
   owns globals needs the `capstone_floatdidf_noglobals.c` treatment, because two translation units
   owning globals collide positionally on the single gp cap-table.
3. Remove the `#define double sqlite_int64` ripple and check the `sqlite3AtoF` patch, which lives in
   the omission's `#else` arm and becomes dead text (harmless: the gates grep source, not
   preprocessed output).
4. Re-baseline every SLT expectation carrying a real-typed row, then JSON becomes available; JSON
   also needs virtual tables, which are not omitted, so nothing else blocks it.

Payoff: one of SQLLogicTest's three result types stops being unreachable, JSON becomes possible, and
the caveat list loses its largest item. Decide it on one question only: does the paper need to say
"with floating point" or "with JSON".

## Part 3 — breadth on silicon: POSTPONED 2026-09-09, survey retained

**Not scheduled.** The lead postponed this half on 2026-09-09. What follows is the survey result,
kept so the ranking and the declines do not have to be re-derived when it is picked up. No boots are
budgeted for it, and nothing in Verification or Risks below covers it.

The reachable set is narrow: no file VFS, no threads, one hart. Lua and mruby are already declined
by their own plans. When it resumes, this is the order I would spend boots in:

1. **The SQLite CVE corpus on silicon.** Nineteen rows exist as QEMU harnesses with matched
   fault/no-fault control pairs (`run-sqlite-row2.sh` is the model: a fault variant, and a
   `-DROW2_NO_REVOKE` control that must return). **Zero have run on the board**, while the master
   plan's T7 asks for exactly this (`docs/plans/ndss-pivot-master-plan.md:220`). It is the paper's
   security axis, it has no silicon evidence at all, and the vehicle is the best-proven thing we
   have. Work: build each row as a staged domain image the way
   `tests/rtl-smoke/slt-corpus/build-slt-corpus-images.sh` does, then small batches, control first,
   the fault variant last in its boot because it halts the domain by design.
2. **Full CoreMark and full RV8 instead of kernel slices.** Today the board has run CoreMark's
   matrix phase and RV8's `primes` only, and the doc itself says the slice
   *"should NOT be called CoreMark without scaling"* (`…measurements…:18-20`). The harnesses exist
   under `benchmarks/`; none of them has a silicon runner, and `tests/rtl-smoke/slt-corpus/` is the
   template for writing one. This removes a caveat from every row of the performance table.
3. **busybox, the genuinely new application.** Flagged in the master plan as the good small next
   target and unscoped. It needs the HostCall file service, whose design documents already exist
   (`docs/design/hostcall-file-service-v0-wire-spec.md`, `stable-file-service-subset.md`). Its own
   plan, after 1 and 2.

Declined, recorded so they are not re-proposed: the TCL suite (interpreter port plus a file VFS),
`mptester` and `threadtest3` (no threads), I/O-error and crash tests (no VFS with durability), Lua
and mruby (dropped by `docs/plans/capstone-column-xlang.md` and `cheri-baseline-xlang.md`).

Two stale statements found while surveying. They are **not** part of the postponement: fix them in
the same pass as Part 1 step 5, since they are wrong today and cost nothing to correct.
`docs/ref/paper-bug-inventory.md:200-207` still says silicon correctness "has not been run", which
§7c superseded; and the paper's own text reportedly still says SQLite has not run on the board
(`…measurements…:564-567`). The second is the lead's to change, not this lane's.

## Delivery: branch, audit, silicon, merge

The work does not land on `dev` directly. Sequence, fixed by the lead on 2026-09-09:

1. **Branch.** Fork `sqlite-stockness` from the current `dev` tip and do every Part 1 commit there:
   the two build-script edits, the re-baselined fixture tallies, and the measurements-doc caveat
   rewrite. Step 0's size probe changes no tracked file and needs no branch. Each commit scanned with
   `capstone/tests/precommit-scan.sh` by absolute path and made with `git commit -o` on its own paths.
2. **Audit with the board lane.** Hand them the branch, not a description. The three things to attack
   specifically, named so the audit is not generic: does the native harvest still produce exactly the
   domain's define list for both values of `SQLITE_FEATURE_SET`; is every changed fixture tally
   explained arm by arm rather than merely different; and does `domdata-budget.py` still say the image
   fits with the allocation numbers quoted. The branch does not move until they answer.
3. **Silicon.** One boot on the audited branch, `k800` control first, then restored `select1`, the
   negative control and the aggregate-function file. Readings into `tests/board-results/*.tsv` from
   the run's own transcript segment. A boot whose control fails is void and rerun.
4. **Merge.** Only after the silicon readings match the QEMU ones. Fast-forward `dev`, telling the
   board session first, since it commits to `dev` in the main checkout.

If the audit or the boot rejects the change, the branch stays unmerged and the default build is
untouched, which is the reason for keeping `SQLITE_FEATURE_SET=deployed` as the default in the first
place.

## Verification

- Step 0's `domdata-budget.py` verdict is recorded before any tracked file changes, together
  with the same verdict for an unmodified build of the same day, so the comparison is like for like.
- **Part 1 is not done until `check-negative-control.sh` passes on the restored native build.** A
  clean `select1` with a comparator that no longer discriminates is the failure mode this project
  keeps paying for.
- The native harvest is proven still matched: after the edit, print the harvested define list from
  `build-slt-native.sh` and diff it against the domain's compile line for both values of
  `SQLITE_FEATURE_SET`.
- Board readings from the run's own transcript segment, `k800 = 4` first, written into
  `tests/board-results/*.tsv`; a boot whose control fails is void and rerun.

## Risks

- Restoring UTF-16 and EXPLAIN grows `.text`; step 0 measures it before anything is committed.
- The aggregate-function tally changing is expected, not a regression, but must be explained arm by
  arm or it reads as one.
- A gate designed carelessly would desync the oracle silently, which is why the harvest and its
  `EXCLUDE` list are named explicitly above.
- The one board boot Part 1 needs queues behind the RTL lane's bitstream work and the lead's flash
  decision.
