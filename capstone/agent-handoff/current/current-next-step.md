# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The previous runtime blocker has been removed. The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is now:

> implement the first real split host/service proof on top of the restored shared-region baseline: a tiny HostCall-style `WRITE_STDOUT` / `puts` request-response experiment

In short: **restoring `capstone/caplifive-buildroot/build/local.mk`, rerunning `make build A=opensbi-rebuild`, and rebuilding `capstone-null-blk` resolved the wrong-firmware problem. The shared-region probe now passes, baseline `null_blk` passes, and split `null_blk` now loads, completes I/O, and unloads successfully. That makes the smallest real next milestone a narrow HostCall proof rather than more firmware triage.**

## Why this is now the right next step

The previously active blocker was:

- the local image behaving like stock OpenSBI,
- host-side `SBI_EXT_CAPSTONE` calls returning `error=-2`,
- shared-region mutations not becoming visible,
- and split `null_blk` either crashing or failing to create the device.

That blocker is now gone.

The source-backed validated results are:

- `build/local.mk` is present again and points Buildroot at:
  - `components/linux`
  - `components/opensbi`
- `make build A=opensbi-rebuild` regenerated:
  - `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
  - `components/opensbi/lib/sbi/capstone_int_handler.c.S`
- the OpenSBI rebuild log shows `opensbi custom Syncing from source dir .../components/opensbi`
- the shared-region probe now reaches:
  - `word after call 1 = 0x1111111111111111`
  - `word after call 2 = 0x2222222222222222`
  - `success`
- baseline `null_blk` works,
- split `null_blk` now:
  - creates `/dev/nullb0`,
  - completes the marker-based I/O path,
  - and completes `rmmod null_blk` successfully.

So the current bottleneck is no longer the firmware/runtime bring-up path itself.
The next unknown is one layer higher: validating the first real host-service protocol on top of the now-working shared-region mechanism.

## Concrete form of the next step

1. Keep `capstone/caplifive-buildroot/build/local.mk` present so future rebuilds continue to use the local Capstone-enabled Linux/OpenSBI overrides.
2. Treat the OpenSBI/runtime bring-up path as restored.
3. Use the working shared-region probe and working baseline/split `null_blk` wrappers as the new runtime baseline.
4. Build the narrowest real host/service experiment on top of that baseline:
   - domain writes a payload into a shared region,
   - domain requests `WRITE_STDOUT` (or equivalent) through a tiny fixed-width metadata ABI,
   - host services the request in ordinary Linux userspace,
   - host re-enters the domain and the domain verifies the response.
5. Only after that proof is green should the project broaden into richer host calls, micro-libc work, or larger applications.
6. When changing `components/opensbi`, keep using the validated rebuild sequence:
   - `make build CAPSTONE_CC_PATH=... A=opensbi-rebuild`
   - then rebuild any kernel modules/packages whose vermagic must match the active kernel.

## What not to regress

Do **not** accidentally drop back to stock OpenSBI by deleting or bypassing:

- `capstone/caplifive-buildroot/build/local.mk`

If that file disappears again, the earlier wrong-firmware symptoms are expected to come back.

