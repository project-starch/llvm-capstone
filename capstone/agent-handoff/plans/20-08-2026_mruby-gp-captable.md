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

- [ ] **1. Compile musl and mruby with `-capstone-gp-captable`. No boot.**
      Counts what breaks. musl is far larger than anything built gp-captable so
      far, so this is where new codegen cases surface. Also run
      `libc-ext/scan-cap-base.py` over the output. Decides whether the rest is
      hours or weeks.

- [ ] **2. Transplant the yield into the gp-captable glue. No boot.**
      `start-gp-captable-interp.S` has `cscratch` (15), `domreturn` (8) and a
      reentry path, and lacks `__capstone_dom_ret` and `__capstone_tls`. The
      yield in `start-musl.S` is 31 lines to the resume label plus the restore
      half, and needs exactly `__capstone_dom_ret`, `cscratch`, `domreturn`.
      Add the slot, the three-instruction stash at entry, and the yield.
      Check before any boot: symbols resolve, and `cjalr` stays 0.

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
