# Shared-region probe and null block reference diagnostic snapshot

Timestamp: 2026-05-12T06-32-05Z

## Why this note exists

This file preserves the current runtime-diagnostics state in one place so a future session can:

1. recover the immediately previous diagnosis without re-running the whole investigation, and
2. see the next diagnostic result appended in the same history artifact.

## Terminology: what “synthetic block device” likely means here

There are two nearby block-device concepts in this workspace, and both matter:

### 1. QEMU `virtio-blk-device`

In the QEMU harness, Linux boots with:

- `-append 'root=/dev/vda ro'`
- `-device virtio-blk-device,drive=hd0`

So `virtio-blk-device` is the emulated block device that exposes the Buildroot root filesystem to the guest as `/dev/vda`.

Why it must be brought up:
- without it, the guest cannot mount the root filesystem named in `root=/dev/vda ro`,
- the guest would fail before login, before `insmod /capstone.ko`, and before any Capstone runtime probe can run.

### 2. Linux `null_blk` (“synthetic” / virtual block device)

The more likely meaning for the current diagnosis is Linux `null_blk`.
Kernel documentation in `capstone/caplifive-buildroot/components/linux/Documentation/block/null_blk.rst` says the null block device (`/dev/nullb*`) is used to benchmark block-layer implementations and emulates a block device without real hardware.

Why it matters here:
- it is an in-tree reference workload for the Capstone split-domain/shared-region path,
- it exercises a realistic block-driver control flow without needing physical storage hardware,
- if it comes up cleanly, that is strong evidence that Capstone region-sharing and domain-call plumbing are working in a real kernel/module scenario,
- if it crashes, that points to a deeper runtime/shared-region problem than a one-off custom probe bug.

## What happens in the `null_blk` reference path

From `capstone/caplifive-buildroot/package/modcapstone/userspace/null_blk.c`:

- `/null_blk.user` initializes the userspace Capstone library,
- creates a domain from `/test-domains/sbi.dom` and `/nullb/capstone_split/nullb_split.smode.ko` via `create_dom_ko(...)`,
- prints `SBI domain created with ID ...`,
- and exits.

From `capstone/caplifive-buildroot/package/capstone-null-blk/capstone_split/sdom/nullb_split.smode.c`:

- the S-mode side repeatedly queries Capstone region metadata with `SBI_EXT_CAPSTONE_REGION_COUNT` and `SBI_EXT_CAPSTONE_REGION_QUERY`,
- reads a function code from the metadata region,
- handles specific null-block operations,
- writes results into shared/borrowed regions,
- then returns with `SBI_EXT_CAPSTONE_DOM_RETURN`.

## Capstone’s role

Capstone is the isolation/runtime mechanism that wires the split flow together:

- userspace creates domains (`create_dom`, `create_dom_ko`),
- userspace calls into them (`call_dom`),
- userspace creates and shares memory regions (`create_region`, `map_region`, `shared_region_annotated`),
- the domain/S-mode side queries those regions via the Capstone SBI extension,
- and control returns via `SBI_EXT_CAPSTONE_DOM_RETURN`.

So Capstone is not the block device itself. It is the runtime boundary and shared-memory transport that the split block-device path depends on.

---

## Current diagnostic result: shared-region sentinel probe

### Why this is treated as the current diagnostic

`capstone/agent-handoff/current/current-next-step.md` frames the active blocker as:

> diagnose why the attempted `sbi.dom + .smode` shared-region sentinel probe still left the host-visible shared word unchanged

So the current diagnosis was refreshed by re-running the dedicated probe.

### Command executed

```bash
cd /home/alexey/dev/llvm-capstone
source capstone/tests/capstone-test-env.sh
export LOG_FILE="$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-shared-region-probe-2026-05-12.log"
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-shared-region-probe-wrapper-2026-05-12.txt" 2>&1
```

### Observed result

Exit status: `1`

Key output:
- `shared-region-probe: created domain ID = 0`
- `shared-region-probe: created shared region ID = 0`
- `shared-region-probe: initial word = 0x0000000000000000`
- `shared-region-probe: region shared via annotated path`
- `shared-region-probe: call 1 retval = 0`
- `shared-region-probe: word after call 1 = 0x0000000000000000`
- `shared-region-probe: stage 1 sentinel mismatch (observed=0x0000000000000000)`

### Meaning

The current local rerun reproduces the same blocker as the earlier 2026-05-08 notes:

- the guest helper successfully creates the domain and shared region,
- the annotated sharing path is reached,
- but after `call_dom()` the host-visible mapped word is still unchanged,
- therefore the narrow runtime/share-region visibility problem remains present.

### Logs

- build log: `/tmp/capstone/build-shared-region-probe-2026-05-12.txt`
- wrapper log: `/tmp/capstone/run-shared-region-probe-wrapper-2026-05-12.txt`
- QEMU log: `/tmp/capstone/capstone-runtime-qemu-shared-region-probe-2026-05-12.log`

---

## Next diagnostic test executed: official `null_blk` reference path

### Why this is the next test

The current notes already recommend comparing the custom `sbi.dom + .smode` probe with a known in-tree S-mode consumer, and the existing history explicitly uses the null block path as that reference workload.

### Command executed

```bash
cd /home/alexey/dev/llvm-capstone
source capstone/tests/capstone-test-env.sh
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-2026-05-12.log" \
  --guest-command "modprobe configfs && /null_blk.user && insmod /nullb/capstone_split/null_blk.ko && test -b /dev/nullb0 && echo hello-world | dd of=/dev/nullb0 bs=1024 count=10 && dd if=/dev/nullb0 bs=1024 count=1 | hexdump -C" \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-wrapper-2026-05-12.txt" 2>&1
```

### Observed result

Exit status: `1`

Harness failure summary:
- `RuntimeError: guest command failed with exit code 139`

Key guest/kernel lines:
- `SBI domain created with ID 0`
- `[    2.323607] Oops [#1]`
- `[    2.323738] Modules linked in: null_blk(O+) capstone(O)`
- `[    2.324654]  ra : null_add_dev+0xd6/0x81e [null_blk]`
- `Segmentation fault`

### Meaning

This rerun again shows that the official null block reference path does not currently reach a clean working `/dev/nullb0` flow in the local environment.

That matters because it strengthens the same high-level conclusion as before:

- the custom shared-region probe failure is not isolated enough to blame only that probe,
- a known in-tree reference workload also fails,
- the active issue still looks like a runtime/shared-region / split-path problem rather than just guest-helper misuse.

### Logs

- wrapper log: `/tmp/capstone/capstone-runtime-qemu-nullb-wrapper-2026-05-12.txt`
- QEMU log: `/tmp/capstone/capstone-runtime-qemu-nullb-2026-05-12.log`

---

## Consolidated diagnosis after both runs

The two refreshed diagnostics point to the same practical conclusion:

1. basic guest boot and Capstone module loading are working well enough to start probes,
2. the custom shared-region sentinel still is not visible after `call_dom()`,
3. the official `null_blk` reference path still crashes while loading `null_blk.ko`,
4. so the current bottleneck remains in the runtime/shared-region split path itself.

## Best current next debugging direction

The highest-value next investigation is still to compare the custom `sbi.dom` wrapper path against the working assumptions encoded by in-tree split consumers such as `nullb_split.smode.c` and determine which of these is false in the current environment:

- the custom `.smode` payload executes as assumed,
- the shared region is installed into the state later queried by `SBI_EXT_CAPSTONE_REGION_QUERY`,
- the region capability/base returned to S-mode is writable and aliases the same memory view,
- or the reference runtime path itself is already broken before the custom probe logic matters.

