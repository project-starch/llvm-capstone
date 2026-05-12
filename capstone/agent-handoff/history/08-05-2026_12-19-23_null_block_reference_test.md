# Null block reference test

Timestamp: 2026-05-08T12-19-23Z

## Goal

Test the official null block reference path suggested by the architecture/runtime implementer, to determine whether a known intended shared-region consumer works in the current QEMU/Buildroot environment.

## Guest command executed

The QEMU harness ran the following guest-side sequence after boot and after the harness itself had already loaded `/capstone.ko`:

```sh
modprobe configfs && \
/null_blk.user && \
insmod /nullb/capstone_split/null_blk.ko && \
test -b /dev/nullb0 && \
echo hello-world | dd of=/dev/nullb0 bs=1024 count=10 && \
dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C
```

## Observed result

The setup helper itself succeeded far enough to print:
- `SBI domain created with ID 0`

But loading the split null block driver crashed:
- kernel `Oops [#1]`
- `Modules linked in: null_blk(O+) capstone(O)`
- `epc : null_add_dev+0x38/0x740 [null_blk]`
- `Segmentation fault`

The command therefore exited with code `139`.

## Logs

- wrapper log: `/tmp/capstone/capstone-runtime-qemu-nullb-wrapper.txt`
- full QEMU log: `/tmp/capstone/capstone-runtime-qemu-nullb.log`

## Meaning

This is important because it shows that the official reference path does not cleanly succeed in the current local environment either.

That means the current blocker is no longer just:
- a custom probe misunderstanding,

but may also involve:
- runtime implementation issues,
- environment drift,
- or a broader bug in the split shared-region path used by the null block case study itself.

## Practical conclusion

The failure of the custom shared-region probe and the failure of the null block reference test together strengthen the case that this is now a runtime-path investigation, not merely a guest-helper API misuse issue.

