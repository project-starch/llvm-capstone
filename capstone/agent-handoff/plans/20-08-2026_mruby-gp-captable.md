# mruby in the silicon-shaped ABI (gp-captable)

## The problem this solves

mruby runs today only because QEMU FABRICATES gp. Measured on 2026-08-20: at
least 6000 fabrications in a single probe run, identical with and without that
day's glue changes, so it is not a regression but the standing state.

The chain, established from sources rather than inferred:

- `gp` has no architectural role. The spec names `x3`/`gp` once, in the register
  table (`prog-model.adoc:234`), and nothing establishes it.
- The monitor does not set it: no reference to gp anywhere in `sbi_capstone.c`.
- Our backend nonetheless reaches every global through it. 9,688 of the 9,690
  `delin` in the mruby image follow a `cincoffset ..., gp, ...`.
- QEMU papers over the gap with what the other line's own commit calls "our
  non-canonical patch": `gp = PCC(cursor 0)`, fabricated AT EVERY USE SITE,
  because "the RTL never establishes gp that way" and those bounds are not
  representable under capability compression.
- WHY at every use site, which the measurement adds: `C_GEN_CAP` (the monitor's
  mint) and `gencap` are the same opcode and produce LINEAR capabilities, and PCC
  is type 0. A linear capability is not copyable, so
  `cincoffset rd, gp, ...` with `rd != gp` NULLs the source
  (`op_helper.c`: `*rs1_v = CAPREGVAL_NULL`). The fabricated gp destroys itself
  on first use and is fabricated again by the next global access.

So mruby as built today cannot run on the board. Not "might fault": it depends on
a capability that cannot exist there.

## Why gp-captable and not the gp-free cscratch experiment

All three silicon-shaped workloads already use the SAME configuration:

```
-mllvm -capstone-gp-captable
tests/runtime-qemu/silicon-ladder/start-gp-captable-interp.S
tests/runtime-qemu/gp-free-domain/link-gpfree.ld  +  gct-section-end.S
```

SQLite (`build-sqlite-silicon.sh`), micropython (`:149,:293`) and jerryscript.
The glue is mature: 1015 lines, descriptor-driven, O(1) `.text` regardless of
global count, and `cjalr` count zero. The kernel module already delivers the
`.capstone_gp_initdesc` descriptor into the front of dom_data.

`gp-free-domain/README.md` calls the cscratch route "Experiment A" and it needs a
`create_domain` change that the README itself keeps as a local experiment, not
committed to submodule source. It would also need either that monitor rebuild or
`CAPSTONE_GP_STANDIN=1`, which lives on the other QEMU line. Both of those are
blocked on decisions that are not this lane's. gp-captable is blocked on nothing.

## What mruby uniquely adds

It is the only workload with a REAL LIBC and HOSTCALL SYSCALLS. SQLite uses a
hand-written libc subset (`capstone_sqlite_libc.c`) and a VFS skeleton;
micropython and jerryscript are freestanding. None of them yields.

So the new ground is exactly the resumable yield, and that is where the work is.

## Steps

Each step either costs no boot or answers one question.

- [x] **1. Compile musl and mruby with `-capstone-gp-captable`. No boot. DONE.**
      **The flag costs nothing at compile time.** A/B over the same file set,
      because comparing against the survey's 40 would have compared different
      sets:

          musl   1393 sources   1343 ok / 50 fail   both arms, IDENTICAL files
          mruby    33 sources     32 ok /  1 fail   both arms

      The one mruby failure is `hash.c`, and it is the sweep's omission rather
      than the flag's doing: the real build compiles that file with `-U__GNUC__`.
      Identical in both arms either way.

      `scan-cap-base.py` over the gp-captable assembly: no findings, with its
      self-test passing first, so the zero is a measurement and not a silent
      instrument.

      So the "hours or weeks" question answers HOURS for the compile stage. This
      says nothing about whether the code is CORRECT, nor about the gct and glue
      side; that is steps 2 to 5.

- [x] **2. Transplant the yield into the gp-captable glue. No boot. DONE 2026-08-20.**
      **The plan as written above was wrong in its central assumption and the
      correction is the useful part of this step.** It said to add a
      `__capstone_dom_ret` slot, copying `start-musl.S`, which reaches that slot
      by gp-relative addressing. In gp-captable `gp` is the cap TABLE, so that
      addressing mode is not available, and three successive designs for a
      replacement slot each died on a detail: end-relative and frame-relative
      offsets do not alias, the exit path does not reset `sp` to the region end,
      and after the carve `sp` no longer covers the descriptor blob at all.

      The exit from that loop was to stop designing and read how the glue itself
      already parks a capability. **It has the whole mechanism already:** `test:`
      does `stc(ra, sp, 48)` -- the entry return capability, in its own frame --
      and `.Ldomain_returned` reads exactly that slot to `domreturn` through.
      Under `INTERP_DOMAIN_MTVEC` it also publishes the frame in `cscratch`. So
      no new slot exists in the final change; the yield reads `cscratch` to reach
      the frame, takes the capability from `+48`, and refreshes that same slot on
      resume. One stash location, so there are no two copies that can disagree.

      The edit is three hunks: the `ccsrrw(x0, cscratch, sp)` frame publish moves
      out of the `INTERP_DOMAIN_MTVEC` guard into
      `INTERP_DOMAIN_MTVEC || CAPSTONE_GLUE_YIELD` (both gates want the identical
      store, for reasons that compose); the matching zeroing before
      `.Ldomain_returned` gets the same condition; and `__capstone_yield` is added
      under `#ifdef CAPSTONE_GLUE_YIELD`.

      Also learned, from the experiment that ended the design loop: linking the
      probe against the unmodified glue reports **exactly one** undefined symbol,
      `__capstone_yield`. `domain_main` and the cap-init range already resolve.
      One link settled what three rounds of source reading had not.

      Checks run, all before any boot:
      * the glue object is **byte-identical** to the pre-edit one at
        `(no flags)`, `INTERP_DOMAIN_MTVEC`, `INTERP_PEEK_SP` and
        `INTERP_RETURN_PRECALL`, so SQLite, micropython and jerryscript are
        untouched including in layout. Positive control: the same comparison run
        **with** `CAPSTONE_GLUE_YIELD` reports a difference, so it can fire.
      * with the define: links, `__capstone_yield` defined, `cjalr` 0. The
        `cjalr` count was itself controlled -- the disassembly was confirmed to
        cover the new routine (47 instructions, one `<unknown>` = `domreturn`)
        rather than silently rendering nothing there.
      * the built probe image makes 9 `ldc`-through-`gp` and 2 `cincoffset gp`
        accesses and carries exactly one `.capstone_gp_table`, so it really does
        exercise the cap table rather than merely link against the glue.
      * 6172 loadable bytes, above the 0x1000 the monitor SPLIT needs.

      `YIELD_PROBE_GPCT=1` on `build-yield-probe.sh` selects this variant. The
      probe source, host program, run script and success markers are shared with
      the musl-glue build on purpose: a difference in the result is then a
      difference between the two glues and nothing else.

- [ ] **3. yield-probe on the new glue. One boot.**
      The smallest thing that actually exercises the new path, and the reason it
      goes before mruby: step 2 touches the entry and return edge, which is
      where clobbering `ra` silently broke the yield once already on 2026-08-20.

- [x] **3. yield-probe on the new glue. One boot. DONE 2026-08-20. PASSES.**
      Run twice, the second time after `link-gpfree.ld` gained the init/fini
      markers (see step 4). Both runs identical and all three discriminators say
      resume rather than restart:

          yield-probe: round 1 before yield
          yield-probe: round 2 AFTER RESUME, stack intact
          yield-probe: DONE after 2 serviced request(s), domain entered domain_main 1 time(s)
          __CAPSTONE_YIELD_PROBE_PASSED__

      Message 1 once, message 2 once, entry counter 1, no MARKER-LOST. So the C
      frame, the local variable set before the yield, and the cap table all
      survive a domain round trip on the gp-captable glue.

- [ ] **4. mruby probe, gp-captable. BLOCKED -- and the blocker is upstream of
      mruby.** Everything needed to BUILD the image now exists and works; the
      image cannot run, for a reason that belongs to the gp-captable ABI itself.

      **What was built and verified:**
      * `MUSL_CAPSTONE_EXTRA_CFLAGS` on the musl survey, appended to the flag list
        rather than replacing it, and a strict no-op when unset (checked against
        `HEAD`'s `--print-flags`).
      * The archive under the gp-captable ABI: **1321 of 1361 compiled, exactly
        the default-ABI baseline**, so at archive scale the flag still costs
        nothing. Controlled: 392 `.capstone_gp_initdesc` sections in the
        gp-captable archive against 0 in the default one, and `heap_fallback`
        0x100000 against 0x40000, so both switches demonstrably took effect.
      * `MRUBY_GPCT=1` on `build-mruby-probe.sh`: the flag, the gp-captable glue
        with `CAPSTONE_GLUE_YIELD`, `gct.o`, and the provisional-link pass that
        measures `.text` to place the globals region. The default path still
        produces a **byte-identical** `.dom`.
      * No malloc change was needed. `malloc.c` already picks its heap at RUNTIME
        on the tag of `__capstone_dom_data`; the gp-captable glue never publishes
        it, so the static `heap_fallback` is chosen -- and under `link-gpfree.ld`
        that array is exactly what the glue carves out of dom_data. The
        dom_data-heap mechanism is the *non*-gp-captable workaround.
      * One real defect found and fixed on the way: `stack_reserve.o` was linked
        only into the SECOND link. Under the default ABI that is invisible (it
        overrides a weak archive definition), but under gp-captable it is a new
        global, a new descriptor record, and the descriptor is the first thing in
        the globals region -- so everything behind it shifted and the
        no-loaded-byte-moved check failed naming `domreq.S`, which had done
        nothing. It is now in both links, so the two differ by exactly the one
        thing the check is about.

      **THE BLOCKER: the descriptor is per translation unit, and the glue reads
      exactly one of them.** Source, `start-gp-captable-interp.S`:
      `cincoffsetimm(t0, s1, 32)` puts record 0 immediately after the header at
      offset 0, and the record count comes from that same header. There is no
      concept of a second fragment.

      Measured in the built images:

          micropython (runs)   1 fragment,  count 232
          yield-probe (runs)   1 fragment,  count   7
          mruby                39 fragments, first count 5, total 2670

      So mruby gets storage for **5 of 2670 globals** and the rest are silently
      uninitialized -- no fault, which is precisely the failure mode
      `domdata-budget.py` was written for, and it is why that script reports
      "VERDICT: fits" on this image. The budget line "cap table 80 (5 globals)"
      is the blocker printing itself, correctly, for anyone who reads it.

      Every gp-captable workload that runs today is a single amalgamated object
      with globals; mruby plus a 1346-member archive is not, and that difference
      has not been exercised before.

      **Stated as open, not as known:** which fragment lands first is a link-order
      accident. Whether the compiler numbers cap-table slots globally or per TU
      was NOT established -- the obvious measurement, "max slot index", returns
      127 for micropython and mruby alike because 127 is what the load's immediate
      field holds (2032/16), not what the code asks for. My earlier reading of 127
      as evidence about numbering was wrong and is withdrawn. That question
      decides whether the fix is "make the glue walk all fragments" or "make the
      linker renumber slots", which is a much larger, compiler-side change.

      Also noted, separately and smaller: 10 sites in the image still use the old
      `scc <rd>, gp, <rs>` data-base addressing, all under `.Lpcrel_hi*` labels,
      so a few objects are not being compiled with the flag. Latent faults, not
      the blocker.

      **The open question is now CLOSED, and the answer is the expensive one.**
      `tests/runtime-qemu/gp-free-domain/multi-tu-slot-collision.sh` reproduces it
      in two files: two TUs with three globals each, and after linking BOTH
      address slots 0,1,2. The objects carry **no relocation at all** for the gp
      offsets -- the slot index is an immediate baked in at compile time -- so the
      linker has nothing to renumber. Merging the descriptor fragments alone
      therefore fixes nothing; every TU would still address the same low slots.
      A relocation for the slot index is needed, which is compiler and linker work.

      **One cheaper candidate, tried and NOT yet working: LTO.** The descriptor is
      emitted per MODULE by `CapstoneAsmPrinter::emitGpCaptableInitDesc`, so a full-LTO
      link would present one module and one descriptor with globally unique slots --
      a build-flag change instead of a new relocation. First trial: compiled with
      `-flto -mllvm -capstone-gp-captable` and linked with
      `--plugin-opt=-capstone-gp-captable`, the link succeeds but the image has an
      EMPTY descriptor and `reada`/`readb` make no cap-table access at all, i.e. the
      pass produced nothing rather than producing something wrong. Whether that is
      flag plumbing into the LTO backend or the pass behaving differently there is
      NOT established. Worth one focused investigation before committing to the
      relocation, because it would avoid it entirely.

- [ ] **4b. mruby probe, gp-captable. One boot.** S1 to S5. After 4's blocker.

- [ ] **5. Core mrbtest suite, gp-captable. One boot.**
      678 assertions, comparable against today's 674 OK / 2 skipped / 2 KO.

## Acceptance criterion

**The gp fabrication counter reads zero.**

The instrument already exists (`capstone_note_gp_fabricated` in `op_helper.c`,
uncommitted). It prints the first few and every thousandth. Today it reports 6000
per run; on success it reports nothing at all. That single number is the
difference between "mruby runs" and "mruby runs on a platform that lies about
gp".

## Risks, named

- Step 2 is assembly at the entry/return edge. The mitigation is step 3, and the
  symbol and `cjalr` checks that precede any boot.
- Step 1 will surface codegen cases. That is its purpose, and it costs no boot.
- The descriptor delivery depends on the buildroot module commit, which is local
  only: `caplifive-buildroot` refuses this account's push with 403. We can build
  it; we cannot share it.
