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

- [ ] **4. mruby probe, gp-captable. One boot.** S1 to S5.

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
