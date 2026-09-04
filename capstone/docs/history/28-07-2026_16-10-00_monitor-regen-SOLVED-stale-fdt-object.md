# C-11 SOLVED: the monitor rebuild was inheriting an FPGA device tree from a stale object

**Date:** 2026-07-28
**Status:** **FIXED.** A rebuilt `fw_jump.elf` boots and runs a domain correctly.
**Supersedes:** `plans/large-ro-delivery-completion-task-A.md` §1-STATUS v2/v3, and
`history/28-07-2026_14-30-00_monitor-regen-boot-hang-cause-not-established.md` (which
correctly refuted the old cause but did not find the real one).

## The one-line answer

`build/build/opensbi-custom/build/platform/generic/firmware/fw_jump.o` was compiled on
**2026-07-22 17:38 for the FPGA firmware, where embedding a device tree is mandatory**.
`make ... A=opensbi-rebuild` only **relinks**; it never recompiles that object. So every
QEMU monitor rebuild silently linked in an **FPGA device tree**, and

```asm
#ifdef FW_FDT_PATH
    /* Override previous arg1 */
    lla a1, fw_fdt_bin        /* firmware/fw_base.S:217 */
#endif
```

made OpenSBI **discard the DTB QEMU passes in `a1`** and use the FPGA one instead. Wrong
memory map, wrong UART — the console is never initialised, so the failure is a hang with
**zero serial output**, before any banner. Exactly the reported symptom, every time.

**The compiler was never involved.** Neither was the generated assembly.

## Why this hid for four days

Three things pointed the wrong way at once.

1. **The good monitor is a PREBUILT.** It was produced by a clean, no-FDT build elsewhere
   and copied into `build/images/`. So "only the prebuilt boots" was true, and looked like
   evidence about the *toolchain* rather than about *this build tree's object files*.
2. **A real, reproducible codegen difference existed and was mistaken for the cause.**
   Regenerating `sbi_capstone_dom.c.S` does change register allocation (good `s0–s6`/frame
   −368 versus `s0–s11`/−464). That is genuine — but it is confined to `create_domain`,
   which does not run at boot. Documented in the 14-30 note.
3. **The obvious experiment was never run**, because "never rebuild the monitor" had become
   a hard rule. The rule was justified — a failed rebuild silently overwrites the
   `fw_jump.elf` both lanes depend on, and the only good copy lived in `/tmp` — but its cost
   was that nobody could take the one measurement that would settle it.

## How it was actually found — the sequence that worked

The decisive move was to **stop varying the compiler and hold every generated input fixed**:

1. Installed the **known-good** `sbi_capstone_dom.c.S` (md5 `b7baff6f`) and touched both
   `.c.S` files so make could not regenerate them. Verified with `-nt` tests that no
   regeneration would occur.
2. Rebuilt, and it **still boot-hung with zero serial**. That single result exonerates
   capstone-c completely: byte-identical assembly in, broken firmware out.
3. Diffed the two ELFs **section by section** rather than reading either. `.text`, `.data`,
   `.bss`, `.got`, `.rela.dyn` were all identical in size; **`.rodata` alone grew by 3,112
   bytes**.
4. Diffed the symbol tables: exactly one symbol was new — **`fw_fdt_bin`**.
5. Dumped the first 16 bytes of `.rodata` from the file rather than inferring:
   `d00dfeed 00000c27 …` — FDT magic, totalsize 3,111. The good monitor's `.rodata` starts
   with ordinary strings (`2d2d2d00…` = `"---"`).
6. Grepped every makefile and the build invocation for `FW_FDT_PATH`: **nowhere**. The
   invocation was `CROSS_COMPILE=… PLATFORM=generic make -j113 -C …/opensbi-custom`, no FDT.
   That contradiction is what pointed at a stale object rather than a stale setting.
7. Checked `fw_jump.o`'s mtime — **07-22 17:38**, while the rebuild ran 07-28 15:57 — and
   confirmed the object itself carries the FDT (`fw_fdt_bin` in its symtab, `d00dfeed` at
   object offset `0x690`, `.rodata` size `0xc27`).

**One wrong turn, corrected before it propagated:** at step 3 the claim was "the grown
`.rodata` overlaps `.dynsym`/`.rela.dyn` and destroys `__rel_dyn_start`". That was wrong —
it compared the *rebuild's* `.rodata` size against the *good* file's `.dynsym` address. The
sections relocate cleanly and `__rel_dyn_start` is present in both. Re-checking with exact
`readelf` output on both files killed it immediately. Worth recording because the wrong
version was a *better* story (a memory-corruption mechanism) than the right one.

## The fix

Delete the stale firmware objects so they are recompiled without `FW_FDT_PATH`:

```bash
cd capstone/caplifive-buildroot
D=build/build/opensbi-custom/build/platform/generic/firmware
rm -f $D/fw_jump.o $D/fw_jump.elf $D/fw_jump.bin $D/fw_dynamic.o $D/fw_payload.o
make build A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../capstone-c)"
```

Verification, in order of strength:

- `readelf -sW $D/fw_jump.o | grep -c fw_fdt_bin` → **0**
- `readelf -SW build/images/fw_jump.elf | grep .rodata` → **`002de8`**, byte-identical in
  size to the known-good prebuilt (was `003a10`)
- Boot: `run-ladder-qemu.sh beebs_aha_mont64` → exit **0**, OpenSBI banner present,
  `__CAPSTONE_LADDER_BEEBS_AHA_MONT64_PASSED__ (retval = 2185097489)` = its oracle.

New monitor md5 `9cbf50681afc1e80bb38246aa80be978`. The prebuilt (`6724bcb3`) is preserved
at `~/capstone-b-artifacts/monitor-known-good/fw_jump.elf.good` alongside the known-good
`.c.S` (`b7baff6f`).

## Standing hazard this leaves

**The same build tree serves both the QEMU monitor and the FPGA firmware, and the FPGA one
requires `FW_FDT_PATH`.** So this trap re-arms every time the FPGA firmware is built here:
the next QEMU rebuild will relink an FPGA object unless the firmware objects are removed
first. Until the two are separated (different `O=` build dirs would do it), **treat
`rm -f $D/fw_*.o` as part of the QEMU monitor rebuild recipe, not as a troubleshooting
step**, and check `readelf -sW build/images/fw_jump.elf | grep -c fw_fdt_bin` → 0 before
trusting a rebuilt monitor.

Note also that the checked-in `components/opensbi/lib/sbi/sbi_capstone_dom.c.S` was the
*broken-regen* copy (md5 `6dfe662a`); the known-good `b7baff6f` is now installed in the
tree. That is a submodule-source change and stays uncommitted per the repo rule, but it is
mirrored under `capstone/tests/vendor-patches/` so it cannot be lost again.

## What this unblocks

1. **Large-`.rodata` delivery → SQLite on silicon.** The monitor copy step (C-4b) can now
   be implemented, built and QEMU-tested. This was the whole reason C-11 mattered.
2. **The `fence.i` domain-boundary fix.** That is a monitor change too, and it is the real
   fix for R-3 — the reason every rung currently costs a full power-cycle plus a ~2 min
   firmware reload. Fixing it attacks the single largest consumer of board time we have.
3. **Any future monitor change at all.** The monitor was effectively frozen; it is not now.
