# S-07 — handover to the RTL lane

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

## 6. What I'm doing next unless you'd rather I didn't

* Rebuilding the reproducer properly in inline asm, with the emitted burst verified by disassembly
  before any boot.
* Tightening artifact identity: our driver doesn't print the hash of the `.dom` it actually loads
  from the initramfs, which is a weaker standard than this folder demands elsewhere.

Everything is on `capstone-bootstrap`. The rest of this folder is the evidence; `00-README.md` is the
entry point and `MECHANISMS-AND-PATCH-PROPOSAL.md` has the ranked mechanisms with what would kill
each.
