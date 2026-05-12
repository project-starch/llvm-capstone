# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is:

> restore a **Capstone-enabled OpenSBI firmware rebuild path** and rebuild `fw_jump.elf`, because the current local image behaves like a stock OpenSBI path where host-side `SBI_EXT_CAPSTONE` calls return `error=-2`

In short: **the latest diagnostics show that the local tree had lost `capstone/caplifive-buildroot/build/local.mk`, so Buildroot was no longer overriding OpenSBI with the local Capstone-enabled source tree. Raw runtime logs now show host-side `DOM_CREATE`, `DOM_CALL_WITH_CAP`, and `REGION_CREATE` returning `error=-2, value=0`. Therefore the next step is to restore the firmware build path, not to keep iterating on guest-side shared-region protocol guesses under the current image.**

## Why this is the right next step

The narrowest active blocker has changed.

The decisive source-backed findings are now:

- `capstone/caplifive-buildroot/build/.config` expects `BR2_PACKAGE_OVERRIDE_FILE="$(CONFIG_DIR)/local.mk"`,
- the local `capstone/caplifive-buildroot/build/local.mk` had been deleted and has now been restored,
- that file is exactly what points Buildroot at the local Capstone-enabled source trees:
  - `components/linux`
  - `components/opensbi`
- raw runtime diagnostics show host-side Capstone SBI calls failing immediately:
  - `DOM_CREATE ... error=-2 value=0`
  - `DOM_CALL_WITH_CAP ... error=-2 value=0`
  - `REGION_CREATE ... error=-2 value=0`
- the baseline `null_blk` path still works,
- the split `null_blk` path no longer crashes after adding error checks, but now fails cleanly during region setup.

So the narrowest remaining uncertainty is no longer:

> "which guest shared-region API variation should we try next?"

It is now:

> **how do we get the local image back onto the Capstone-enabled OpenSBI firmware path so that host-side `SBI_EXT_CAPSTONE` calls stop failing immediately with `error=-2`?**

## Concrete form of the next step

1. Keep `capstone/caplifive-buildroot/build/local.mk` present so the Buildroot override file once again points at:
   - `components/linux`
   - `components/opensbi`
2. Restore or regenerate the missing local OpenSBI generated assembly files:
   - `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
   - `components/opensbi/lib/sbi/capstone_int_handler.c.S`
3. Rebuild the firmware image so `build/images/fw_jump.elf` is produced from the Capstone-enabled OpenSBI override instead of a stock OpenSBI source path.
4. Rerun the userspace shared-region probe first.
5. If host-side `DOM_CREATE` / `REGION_CREATE` no longer return `error=-2`, rerun `null_blk` split.
6. Only after the firmware path is genuinely fixed does it make sense to continue with higher-level guest shared-memory protocol debugging.

## What not to jump to yet

Do **not** jump straight to:

- full hosted `capstone64-unknown-linux-gnu` sysroot compatibility,
- `glibc` / `musl` / `picolibc` porting,
- speculative yield/resume trap ABIs,
- FFmpeg/sqlite/libpng/SPEC directly.

The point of this step is to restore the actual Capstone-enabled firmware/runtime baseline first. Until host-side `SBI_EXT_CAPSTONE` calls stop failing with `error=-2`, guest protocol experiments do not meaningfully test the intended runtime.

