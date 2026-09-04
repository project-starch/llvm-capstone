# Task 017 — patched UP built-in `fw_payload` built + QEMU-regression-validated (fence.i fix)

**Date:** 2026-07-18
**Branch:** capstone-bootstrap-b
**Scope:** rebuild our `fw_payload` with the domain-switch `fence.i` monitor patch;
offline QEMU regression validation. No board run (held for a watched session). No
RTL / `capstone-ariane` / submodule-source commits; no SPI/flash.

## What this delivers

The patched board image is built, staged, and regression-clean under QEMU. The
only remaining step is the (watched, flaky) board confirmation.

**Image: `fw_payload_up_builtin_fence.bin`**
**sha256 `9c53ffd8e353b0e41b30d33a046f95d86042015181b766103573aa4b933cca0b`**
(15,367,704 B). Staged durably at `~/capstone-b-artifacts/` and
`/tmp/capstone-b/fpga-image/`. Supersedes `1d7a02e4` (same UP built-in Image, adds
the fence.i monitor fix).

## How it was built (no external checkout; reused the existing Image)

`fw_payload = OpenSBI + embedded Linux Image`. Only the OpenSBI monitor changed
(the fence.i patch), so there was no need to rebuild the kernel:

1. Applied the staged patch to the OpenSBI capstone-sbi submodule working tree
   (`opensbi-capstone-sbi-domain-switch-fence-i.patch`, 9 `fence.i`), mirrored it
   into the rsynced build copy `build/build/opensbi-custom/`.
2. Extracted the UP built-in Linux Image from the prior image `1d7a02e4` at offset
   `0x200000` (`MZ` RISC-V Image magic) → `Image_up_builtin` (13,270,536 B).
3. Rebuilt `PLATFORM=fpga/ariane` OpenSBI with `CROSS_COMPILE` (buildroot gcc 12.3)
   and `FW_PAYLOAD_PATH=Image_up_builtin`.

Verification of the wrap: the embedded Image in the new payload is **byte-identical**
to the extracted source (compared the first 13,270,536 B; the 16 B tail is zero
pad). The monitor patch is really in the binary: `objdump -d` shows the `fence.i`
instructions in both the generic `fw_jump.elf` (12) and the fpga/ariane
`fw_payload.elf` (22, incl. the payload kernel's own).

## QEMU regression validation (fence.i is a no-op under QEMU — regressions only)

QEMU models no icache, so it **cannot** exercise the fence fix — that is precisely
why the board is needed. What QEMU *can* prove is that the patched monitor + image
did not regress. Both pass:

1. **Booted the patched UP built-in Image directly** under `virt-capstone` with the
   patched `fw_jump.elf` (`-bios fw_jump.elf -kernel Image_up_builtin`,
   `-icount shift=0,sleep=off`):
   - boots to a **root shell** ✓
   - **`/dev/capstone` present at boot** as a char device, **no insmod** (built-in) ✓
   - **borrow `.dom` → `RESULT cycles/op raw=2 borrow=6`** — identical to the
     reference `RESULTS.md` ✓
2. **Standard revoke-cost sweep** (`run-revoke-cost-fpga-qemu.sh`) through the
   patched `fw_jump.elf` → **bump 7 / norevoke 60 / revoke 65, +5 revoke-at-free** —
   identical to the reference. No regression.

## Driver wiring (additive, dry-run green)

`fpga_driver/run_rtl_smoke.py`: added `--dtb` (board DTB for the gdb boot) and
`--builtin` (skip insmod for a built-in-capstone image → `load_module=False`), both
wired into the `main` → `run_smoke` call; updated `--image` help to name the
patched image + sha. `test_dryrun.py` still green ("dry-run passed").

## Board confirmation — held for a watched session

Recipe unchanged from the boot note: GDB-boot `fw_payload_up_builtin_fence.bin`
(`monitor reset halt` → `monitor load_image` fw + `caplifive.dtb`@0x82200000 → set
`$pc/$a0/$a1` → `continue` → root login), throttle the lossy UART, detach GDB
before the `.dom`s. Run **borrow-cost first** (the CALL that stalled): if it now
returns and prints `RESULT`, the fence.i fix works → run the full sweep →
`--parse-uart`. Driver invocation:

```
run_rtl_smoke.py --url <token'd> --boot-method=gdb --builtin \
    --image ~/capstone-b-artifacts/fw_payload_up_builtin_fence.bin \
    --dtb ~/capstone-b-artifacts/caplifive.dtb
```

Expected: the RTL cycle number may sit slightly above QEMU's +5 (real silicon pays
the icache-fence cost QEMU doesn't model — legitimate, not a discrepancy). If it
still stalls with fences in, next lever is the secondary MIE/interrupt difference
between the lineages; only if that also fails is it genuinely RTL.

See also `18-07-2026_14-38-40_fpga-domain-call.md` (root cause) and
`patches/opensbi-capstone-sbi-domain-switch-fence-i.patch`.
