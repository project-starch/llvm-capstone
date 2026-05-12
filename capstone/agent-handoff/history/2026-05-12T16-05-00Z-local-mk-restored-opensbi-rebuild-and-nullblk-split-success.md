# Restoring `build/local.mk` fixed the firmware path: OpenSBI rebuild succeeded and `null_blk` split now works

Timestamp: 2026-05-12T16:05:00Z

## Summary

This session confirmed the suspected root cause and then verified the actual fix end to end.

The key point is:

> restoring `capstone/caplifive-buildroot/build/local.mk` and rerunning the author-recommended `make build A=opensbi-rebuild` path brought back the Capstone-enabled OpenSBI runtime path.

After that:

- the previously missing generated OpenSBI wrapper files were recreated,
- the shared-region probe passed end to end,
- `null_blk` split stopped failing on `DOM_CREATE` / `REGION_CREATE`,
- the remaining `invalid module format` issue was only a stale `vermagic` mismatch and was fixed by rebuilding `capstone-null-blk` against the active `linux-custom` kernel,
- the split null-block path then completed successfully.

## Author confirmation that unblocked the fix

The runtime author replied that:

- the deleted `build/local.mk` was likely the real problem,
- without it Buildroot grabs upstream OpenSBI,
- the intended rebuild command is:

```bash
make build A=opensbi-rebuild
```

This matched the earlier diagnosis.

## Evidence that the correct OpenSBI path was restored

### 1. `build/local.mk` was present again

Contents:

```makefile
LINUX_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/linux
OPENSBI_OVERRIDE_SRCDIR = $(BR2_EXTERNAL_CAPSTONE_PATH)/components/opensbi
```

### 2. The OpenSBI rebuild regenerated the missing wrapper assembly files

Before rebuild, these files were absent:

- `components/opensbi/lib/sbi/sbi_capstone_dom.c.S`
- `components/opensbi/lib/sbi/capstone_int_handler.c.S`

After running:

```bash
cd /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot
make build CAPSTONE_CC_PATH=/home/alexey/dev/llvm-capstone/capstone/capstone-c A=opensbi-rebuild
```

both files existed again.

### 3. The rebuild log showed the local override path was actually used

Important rebuild log lines:

```text
Running `target/debug/capstone-c --abi capstone .../components/opensbi/lib/sbi/sbi_capstone_dom.c ... > .../sbi_capstone_dom.c.S`
Running `target/debug/capstone-c --abi capstone .../components/opensbi/lib/sbi/capstone_int_handler.c ... > .../capstone_int_handler.c.S`
>>> opensbi custom Syncing from source dir /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot/components/opensbi
```

This is the source-backed confirmation that the build switched back to the Capstone-enabled OpenSBI override path rather than stock OpenSBI.

## Shared-region runtime result after the OpenSBI rebuild

Command wrapper used:

```bash
cd /home/alexey/dev/llvm-capstone
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
```

Result:

```text
QEMU smoke passed.
```

Key serial markers:

```text
shared-region-probe: created domain ID = 0
shared-region-probe: created shared region ID = 11
shared-region-probe: initial word = 0x0000000000000000
shared-region-probe: region shared via annotated path
shared-region-probe: call 1 retval = 257
shared-region-probe: word after call 1 = 0x1111111111111111
shared-region-probe: call 2 retval = 514
shared-region-probe: word after call 2 = 0x2222222222222222
shared-region-probe: success
```

Meaning:

- the host-side Capstone SBI runtime path is no longer behaving like stock OpenSBI,
- domain creation and shared-region mutation are now genuinely working,
- the earlier `error=-2, value=0` behavior was indeed the wrong-firmware symptom.

## `null_blk` split result after the OpenSBI rebuild

### First post-fix run: stale module/kernel mismatch

After the successful OpenSBI rebuild, the first split/baseline null-block rerun failed with:

```text
null_blk: version magic '6.1.26 SMP mod_unload riscv' should be '6.1.0 SMP mod_unload riscv'
insmod: ... invalid module format
```

This was not a runtime regression. It was a build-artifact mismatch:

- active kernel release: `6.1.0`
- stale `null_blk.ko` vermagic: `6.1.26`

That mismatch was confirmed directly by checking:

```bash
make -C build/build/linux-custom --no-print-directory -s kernelrelease
strings build/build/capstone-null-blk-1.0/.../null_blk.ko | grep '^vermagic='
```

### Fix for the module mismatch

Rebuilt the null-block packages after the OpenSBI rebuild:

```bash
cd /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot
LD_LIBRARY_PATH="" make -C buildroot BR2_EXTERNAL="$PWD" O="$PWD/build" capstone-null-blk-dirclean
LD_LIBRARY_PATH="" make -C buildroot BR2_EXTERNAL="$PWD" O="$PWD/build" capstone-null-blk-rebuild
LD_LIBRARY_PATH="" make -C buildroot BR2_EXTERNAL="$PWD" O="$PWD/build"
```

After that, both baseline and split `null_blk.ko` had:

```text
vermagic=6.1.0 SMP mod_unload riscv
```

matching the active `linux-custom` kernel.

## Final `null_blk` runtime validation

### Baseline control

Command used:

```bash
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-baseline-final2.log" \
  --guest-command "dmesg -n 8 && modprobe configfs && cd /nullb/baseline && insmod ./null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && rmmod null_blk"
```

Observed result:

```text
QEMU smoke passed.
[    2.584597] null_blk: disk nullb0 created
```

### Split path

Two final split validations were used.

#### A. Split I/O path with explicit success marker

Command used:

```bash
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-marker.log" \
  --guest-command "dmesg -n 8 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=1 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C && echo __SPLIT_DONE__" \
  --success-marker "__SPLIT_DONE__"
```

Observed result:

```text
QEMU smoke passed.
```

Key serial markers:

```text
SBI domain created with ID 0
[    2.694017] null_blk: disk nullb0 created
[    2.694306] null_blk: module loaded
0+1 records in
0+1 records out
1+0 records in
1+0 records out
__SPLIT_DONE__
```

#### B. Split unload path with explicit success marker

Command used:

```bash
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --buildroot-dir "$PWD/capstone/caplifive-buildroot" \
  --qemu-binary "$PWD/capstone/capstone-qemu/build/qemu-system-riscv64" \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-rmmod.log" \
  --guest-command "dmesg -n 8 && modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && echo __BEFORE_RMMOD__ && rmmod null_blk && echo __AFTER_RMMOD__" \
  --success-marker "__AFTER_RMMOD__"
```

Observed result:

```text
QEMU smoke passed.
```

Key serial markers:

```text
SBI domain created with ID 0
[    2.689899] null_blk: disk nullb0 created
[    2.690062] null_blk: module loaded
__BEFORE_RMMOD__
__AFTER_RMMOD__
```

## Interpretation

At this point the original blocker is resolved:

- the wrong-firmware/stock-OpenSBI problem is fixed,
- the shared-region runtime path works,
- `null_blk` split module creation works,
- split I/O reaches completion,
- split unload also works.

The earlier crash and false-success states were downstream symptoms of the missing Buildroot override file and the resulting stock OpenSBI runtime path.

## Practical conclusion

The original problem is no longer:

- shared-region calls returning fake IDs with `error=-2`,
- split `null_blk` crashing in kernel space,
- or split `null_blk` failing to create the device.

Those issues were resolved by:

1. restoring `build/local.mk`,
2. rerunning `make build A=opensbi-rebuild`,
3. rebuilding `capstone-null-blk` so module vermagic matched the active kernel.

The project can now move on from this blocker to the next intended runtime or hosted-software milestone.

