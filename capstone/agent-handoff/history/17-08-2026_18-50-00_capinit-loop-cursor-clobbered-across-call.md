# The cap-init dispatch loop kept its cursor in a caller-saved register

**2026-08-17. Root cause FOUND and FIXED, QEMU-verified, audited.**
`capstone/tests/runtime-qemu/silicon-ladder/start-gp-captable-interp.S`, macro `RUN_CAP_INIT`.

## The defect

The loop that walks `.capstone_cap_init` and calls each initializer kept its table cursor in
**`a7`** across the `jalr`:

```
70:  ldc(a6, sp, 16)              /* cursor capability: parked and reloaded -- correct        */
     ld    t0, 0(a6)
     add   t0, a7, t0            /* a7 is ALSO the base for the entry's relative delta       */
     jalr  ra, 0(t0)             /* <-- ordinary compiled C runs here                        */
     ldc(a6, sp, 16)             /* a6 reloaded: "the callee may have clobbered it"          */
     cincoffsetimm(a6, a6, 8)
     stc(a6, sp, 16)
     addi  a7, a7, 8             /* <-- a7 NOT reloaded. It is caller-saved.                 */
     lla   t4, __capstone_cap_init_end   /* t4 recomputed each iteration -- also correct     */
     bltu  a7, t4, 70b
```

`a6` is parked and reloaded, and `t4` is recomputed every iteration, so the author was plainly
aware that the callee clobbers registers. `a7` (x17) is caller-saved too, and
`__capstone_cap_init` is ordinary compiled C: **36 instructions write `a7`** in one MicroPython
domain (`llvm-objdump` over `0x9d524..0xa5184`, counting `a7` in the destination position only --
a naive `grep -c 'a7,'` gives 61 because 25 of those are `stc a7, off(rX)`, where it is the
source. That miscount was caught by the auditor and is exactly the `grep -c` class of instrument
error already in this project's incident list).

So the loop's termination test compares a **garbage** cursor against the end marker, and whether
it stops after the right number of entries is decided by whatever the generated initializer
happened to leave in `a7`.

## Why that is not a theoretical worry

Measured on a MicroPython domain built with `MICROPY_PY_UCTYPES=0`:

* the last instruction in `__capstone_cap_init` that writes `a7` is
  `auipc a7, 0xffff8 ; addi a7, a7, -0x5e8` at `0xa4fd0`, giving `a7 = 0x9c9e8` --
  which is exactly `<mod_random_random>`;
* `+8` gives `0x9c9f0`, and `bltu 0x9c9f0, 0xc3ae8` (`__capstone_cap_init_end`) is **taken**;
* the loop therefore runs one iteration too many over a table that has exactly ONE entry
  (`start 0xc3ae0`, `end 0xc3ae8`, resolving correctly to `__capstone_cap_init`), reads 8 bytes
  past it, and `jalr`s to `0x9c9f0`;
* a QEMU `-d in_asm` trace shows the next block executed is **precisely `0x9c9f0`**, the third
  instruction of `mod_random_random`'s prologue -- an address no branch inside that function
  targets, so only an indirect jump can create a translation block there.

`mod_random_random` then allocates a float before `gc_init` has run: `m_malloc` ->
`m_malloc_fail` -> `mp_raise_msg_varg` -> `nlr_jump`, which finds no NLR handler and lands in
`nlr_jump_fail`, whose tail is an infinite `j .` at domain VA `0x2f524`. **`domain_main` never
executes at all** -- `-d exec,nochain -dfilter` shows the loop-exit path at `0x102c4` running
**zero** times.

The matched control explains why the same tree worked with `MICROPY_PY_UCTYPES=1`: there the
last `a7` write is `ldc a7, 0x0(a7)`, a GCT capability pointing into the blob, which the monitor
places above the image, so `a7` came back *greater* than the end marker and the loop exited by
luck. The failing and passing arms differ in exactly one thing: whether the initializer's final
`a7` write is a PC-relative image address or a blob capability.

## The fix

Park `a7` in the frame exactly as `a6` already was: frame `-32` -> `-48`, `sd a7, 32(sp)` after
`mv a7, t3`, `ld a7, 32(sp)` at the loop head, and reload/increment/store after the `jalr`. `a7`
is an integer here (it comes from `mv a7, t3` where `t3` is an `lla`), so a plain `sd`/`ld` is
safe -- there is no tag to drop, and the file already records the board rejecting `ld` *through*
`a7` as a base, which is direct evidence it is untagged.

The three `cincoffsetimm(sp, sp, 32)` restores became 48, and the `INTERP_PEEK_CAPINIT_TARGET`
diagnostic's `ldc(a2, sp, 112)` became 128 -- its own comment derives that offset from the frame
size, and an audit caught it still reading 112 after the frame grew. Diagnostic-only, but it
would have cost a wasted run the next time that knob was set.

## Verification

* The previously-hanging arm (`MICROPY_PY_UCTYPES=0`) now returns `0x4D500000` -- `MPY_MARK`
  base with `rc = 0`, i.e. parse + compile + execute completed with no exception -- **3/3 calls**.
  The `=1` control returns the same low word 3/3. (`retval` prints as a 64-bit `sbiret.value`
  from a 32-bit `*res`, so the control's upper half reads `0x1_4D500000`; only the low word is
  domain-written.)
* MicroPython chunk 000_199 with the fixed glue: 195 PASS / 2 FAIL / 3 UNSCORED, **identical
  test-for-test** to before -- 200 domain calls, no regression.
* The production `.dom` is byte-identical before and after the comment and `#ifdef`-guarded
  diagnostic edits, so that regression run still applies to the committed source.

Suite sweep over the other consumers of this shared glue:

| suite | result |
|---|---|
| lit `CodeGen/Capstone` | 54/54 PASS |
| silicon ladder, `DOMAIN_GLUE=interp` (`beebs_prime`, `gpn64`) | 2/2 PASS, correct oracles |
| MicroPython chunk 000_199 | 195/2/3, identical test-for-test |
| SQLite QEMU gate | build + static gates PASS; runtime fails, **pre-existing** |
| BEEBS QEMU suite | does not use this glue — `grep` finds no reference |
| gp-free-domain | does not use this glue (`start-gpfree-cscratch.S`); no clean run obtained |

**State the weak spot rather than bank the passes.** Both ladder rungs also pass with the
PRE-fix glue, so they are a mechanical regression check on the frame change and *not* a positive
control for the bug: their generated initializers are too small for the register allocator to
pick `a7` as scratch. The only positive control is the MicroPython `__capstone_cap_init`, where
the failing arm is reproducible and the fix flips it.

Two pre-existing problems surfaced by the sweep, neither caused by this change and neither fixed
here:

* the SQLite QEMU gate fails at `create_dom` (`obs=18446744073709551615`) **before the domain is
  entered**, so it never reaches `RUN_CAP_INIT` at all. A/B'd by stashing this fix: byte-identical
  failure without it.
* `run-sqlite-silicon.sh` does not propagate `OUT_DIR` to `build-sqlite-host.sh`, so the host
  binary is written to `sqlite-build` while the run step looks in `sqlite-silicon`.

## Why this matters beyond MicroPython

This glue is shared by **every** gp-captable domain -- SQLite and BEEBS included. Any of them is
one register-allocation change away from the same failure, and the symptom is a silent hang with
no fault, before `domain_main`, that moves when anything about the image moves.

That is the shape of `tests/fpga-repros/S01-image-perturbation-hang` ("nine structurally
different perturbations of `uc` were built; every one hangs, and only unmodified builds return").
**It is NOT claimed to be S01**: S01's README reports `dp0.dom` as *correct under QEMU*, whereas
this bug reproduces under QEMU, so they are probably different. The S01 folder ships sources and
hashes but no built `.dom`, so the cheap static check -- does the last `a7` write in that image's
`__capstone_cap_init` land below `__capstone_cap_init_end`? -- needs those images rebuilt first.
Worth 30 seconds whenever someone next has them. The frozen copies under
`fpga-repros/S01-.../src/` and `R16-entry-stall/src/` still carry the pre-fix loop by design;
R16 is resolved by a bitstream A/B and is unaffected.

## Method notes

* The reproducer shrank from a 200-test image to a **one-line program** in a non-runner build,
  which is what made `MPY_STAGE` staging and per-second QEMU tracing affordable.
* Four hypotheses were raised and **refuted by measurement**, not argument: `.text`/globals
  overlap (readelf: `.text` ends `0xaf52c`, globals at `0xb0000`), insufficient slack (forcing a
  spare 64 KiB gave 68,308 bytes and hung identically), carve bounds representability
  (`check-repr.py` reports OK for both arms), and the const-folding module table at
  `py/parse.c:663` (`MICROPY_COMP_MODULE_CONST=0` did not fix it).
* One verification run was **VOID and briefly misread as a pass**: a zsh word-splitting bug put
  the build flags into `DOM_NAME`, so the `.dom` filenames contained spaces, the loader printed
  `Failed to open the file.` and returned `-1` for both arms. It was caught only because the
  retval was not the expected marker. `Ok, good file.` versus `Failed to open the file.` in the
  boot log is a genuine positive/negative control on the loader and should be grepped for
  explicitly, not assumed.
