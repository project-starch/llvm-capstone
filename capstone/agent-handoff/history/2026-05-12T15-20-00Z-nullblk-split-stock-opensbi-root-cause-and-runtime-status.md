# `null_blk` split path diagnosis: stock OpenSBI root cause and current runtime status

Timestamp: 2026-05-12T15:20:00Z

## Executive summary

The current local `null_blk` split failure is **not** a `null_blk`-specific logic bug.

The decisive finding from this session is:

> the current runtime image is effectively booting a **stock OpenSBI path without the Capstone SBI extension**, so host-side Capstone SBI calls return `error=-2, value=0`; several callers were previously ignoring `sbiret.error`, which created false-success symptoms (`domain ID = 0`, `region ID = 0`) and eventually led to the earlier kernel crash.

That means:

- the earlier `null_add_dev()` crash was a **secondary symptom** of ignored SBI errors,
- the newly added checks in the split `null_blk` path now stop the driver cleanly instead of letting it crash in `__memcpy`,
- the deeper blocker is still the runtime/firmware path, specifically the missing Capstone-enabled OpenSBI override in the current build.

## Why the firmware diagnosis changed

Two source-backed facts now line up:

1. `capstone/caplifive-buildroot/build/.config` contains:

   - `BR2_PACKAGE_OVERRIDE_FILE="$(CONFIG_DIR)/local.mk"`

2. `capstone/caplifive-buildroot/build/local.mk` had been deleted in the local tree.

The committed content of that file is:

```makefile
LINUX_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/linux
OPENSBI_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/opensbi
```

So without `build/local.mk`, Buildroot does **not** override OpenSBI with the local Capstone-enabled source tree and instead falls back to the ordinary package source.

This also matches the earlier observation that `build/build/opensbi-1.2` did not obviously contain the Capstone-specific sources, while the runtime behavior itself showed every Capstone SBI call failing.

## Raw runtime evidence

### 1. Userspace shared-region probe with raised guest loglevel

Command shape used:

```bash
cd /home/alexey/dev/llvm-capstone
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/build-shared-region-probe.sh "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share"
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-shared-region-probe-verbose-after-domdiag.log" \
  --guest-command "dmesg -n 8 && cp /mnt/host/shared_region_probe.user /tmp/shared_region_probe.user && chmod 0755 /tmp/shared_region_probe.user && /tmp/shared_region_probe.user /mnt/host/shared_region_probe.smode"
```

Key serial lines:

```text
[    2.219924] ioctl_create_dom: paddr=101580000 code_len=179072 tot_size=40000 entry_offset=13f8c error=-2 value=0
[    2.220528] ioctl_create_dom: dom_call_with_cap dom_id=0 s_paddr=1011a6000 s_size=8192 s_entry=1011a6324 error=-2 value=0
[    2.223245] ioctl_create_region: len=4096 vaddr=ff60000081222000 paddr=101422000 error=-2 value=0
shared-region-probe: created domain ID = 0
shared-region-probe: created shared region ID = 0
shared-region-probe: call 1 retval = 0
shared-region-probe: word after call 1 = 0x0000000000000000
shared-region-probe: stage 1 sentinel mismatch (observed=0x0000000000000000)
```

Interpretation:

- `domain ID = 0` and `region ID = 0` are **false-success artifacts** caused by callers reading `.value` while the real status is already in `.error = -2`.
- the domain was not genuinely created/initialized in a working Capstone runtime sense,
- the shared region was not genuinely created in a working Capstone runtime sense,
- therefore the unchanged sentinel is expected under the current firmware path.

### 2. `null_blk` split path after adding kernel-side error checks

Command shape used:

```bash
cd /home/alexey/dev/llvm-capstone
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir /tmp/capstone/capstone-runtime-qemu-share \
  --log-file /tmp/capstone/capstone-runtime-qemu-nullb-split-after-fix.log \
  --guest-command "dmesg -n 8 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && rmmod null_blk"
```

Key serial lines:

```text
SBI domain created with ID 0
[    2.314468] null_blk: Failed to create shared region: error=-2 len=4096 pa=0x00000001014aa000
[    2.342238] null_blk: Failed to create shared region: error=-2 len=4096 pa=0x00000001014cd000
insmod: can't insert '/nullb/capstone_split/null_blk.ko': Invalid argument
```

Interpretation:

- the old split crash moved from a late `__memcpy` fault to an **early, explicit failure** in region setup,
- this confirms the earlier kernel oops was downstream from ignored SBI errors,
- the underlying runtime problem is still present.

### 3. Baseline control still passes

Command shape used:

```bash
cd /home/alexey/dev/llvm-capstone
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir /tmp/capstone/capstone-runtime-qemu-share \
  --log-file /tmp/capstone/capstone-runtime-qemu-nullb-baseline-after-fix.log \
  --guest-command "dmesg -n 8 && modprobe configfs && cd /nullb/baseline && insmod ./null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && rmmod null_blk"
```

Observed result:

```text
QEMU smoke passed.
[    2.148832] null_blk: disk nullb0 created
```

Interpretation:

- the baseline control still works,
- this remains consistent with the problem being in the Capstone split/runtime path rather than ordinary `null_blk` mechanics.

## Code changes made in this session

### 1. Restored Buildroot override file

Restored:

- `capstone/caplifive-buildroot/build/local.mk`

with:

```makefile
LINUX_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/linux
OPENSBI_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/opensbi
```

This does **not** by itself fix the already-built image, but it restores the intended Buildroot override mechanism for future rebuilds.

### 2. Made split `null_blk` fail safely on region-create/query errors

Updated:

- `capstone/caplifive-buildroot/package/capstone-null-blk/capstone_split/module/null_blk.c`

Behavioral change:

- `REGION_CREATE` and `REGION_QUERY` errors are now checked,
- invalid region-base setup no longer proceeds into later `memcpy`-based corruption/crash paths,
- split module init returns failure instead of producing a kernel oops.

### 3. Made `modcapstone` stop reporting false success

Updated:

- `capstone/caplifive-buildroot/package/modcapstone/module/capstone.c`

Behavioral change:

- `DOM_CREATE` now returns `dom_id = -1` on SBI error instead of blindly using `.value`,
- `REGION_CREATE` now returns `region_id = -1` on SBI error instead of blindly using `.value`,
- failed create operations are logged explicitly and allocated pages are freed on region-create failure,
- `DOM_CALL_WITH_CAP` errors are logged explicitly.

This is a correctness improvement even before the firmware path is fixed because it prevents misleading fake IDs.

### 4. Added a package-local workaround for `capstone-sbi-domain` rebuilds

Updated:

- `capstone/caplifive-buildroot/package/capstone-sbi-domain/Makefile`
- added `capstone/caplifive-buildroot/package/capstone-sbi-domain/sbi.dom.c.S`
- updated `capstone/caplifive-buildroot/package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c`

Reason:

- the current local Capstone-C regeneration path for the OpenSBI-style domain wrapper is unstable for this workflow,
- a checked-in/generated `sbi.dom.c.S` allows the package to rebuild without re-triggering the failing generator path.

Important caveat:

- this does **not** solve the root firmware issue by itself,
- it only keeps the `capstone-sbi-domain` package buildable in the local tree.

## What still blocks a full runtime fix

A real end-to-end fix still requires a **Capstone-enabled OpenSBI firmware image** to be built and used by QEMU.

At the moment, that path is blocked by two local facts:

1. `build/local.mk` had been deleted, so the override path was not active.
2. the local generated OpenSBI helper assembly files are currently missing from the working tree:
   - `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
   - `components/opensbi/lib/sbi/capstone_int_handler.c.S`

and the attempted wrapper regeneration path currently fails in the local Capstone-C toolchain.

So the present state is:

- the diagnosis is strong,
- the callers are now safer and more honest,
- but the final firmware rebuild path still needs the missing/generated OpenSBI `.S` pieces restored or regenerated successfully.

## Best current next step

The next highest-value step is now much narrower than before:

1. restore/recover a valid local `components/opensbi/lib/sbi/sbi_capstone_dom.c.S` and `capstone_int_handler.c.S` pair,
2. keep `build/local.mk` present,
3. rebuild the firmware image so `build/images/fw_jump.elf` actually comes from the Capstone-enabled OpenSBI source override,
4. rerun the shared-region probe first,
5. then rerun `null_blk` split.

Until that is done, every host-side Capstone SBI call in this environment should be treated as suspect if the code ignores `sbiret.error`.

