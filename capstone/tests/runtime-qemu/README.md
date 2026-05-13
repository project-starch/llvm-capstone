# Capstone runtime QEMU smoke tests

This directory contains runtime smoke/probe helpers for the **current validated domain ABI path**.

It intentionally does **not** try to validate the broader hosted Linux user-space flow yet.
That hosted flow is still blocked earlier in the toolchain/sysroot integration.

## What the validated baseline currently covers

In one QEMU boot it verifies that:

1. a tiny domain can be built by the in-tree LLVM Capstone toolchain,
2. the guest boots,
3. the host-shared `9p` directory can be mounted inside the guest,
4. the `capstone` kernel module loads,
5. `/capstone-test.user` accepts and executes the domain from the shared directory,
6. the expected success markers appear.

The restored runtime baseline also includes the following QEMU guest-command
regressions:

7. the shared-region proof succeeds on the Capstone-enabled OpenSBI path,
8. the first HostCall-style `WRITE_STDOUT` request/response proof succeeds over shared metadata + payload regions,
9. baseline `null_blk` loads, performs I/O, and unloads,
10. split `null_blk` loads, performs I/O, and unloads.

The key property is that the domain is provided through a host-shared directory, so the test does **not** rebuild `rootfs.ext2` for each iteration.

## Files

- `build-domain.sh` — generic helper to build a Capstone domain ELF from a tiny `domain_main()` source file.
- `domains/write_42.c` — the initial tiny smoke domain.
- `run-domain-smoke.py` — QEMU + guest automation harness.
- `run-smoke.sh` — one-command entry point that builds the tiny domain and runs the smoke test.
- `run-shared-region-probe.sh` — restored shared-region proof.
- `build-hostcall-stdout-probe.sh` — cross-build helper for the first HostCall stdout proof.
- `run-hostcall-stdout-probe.sh` — first HostCall-style `WRITE_STDOUT` / `puts` regression wrapper.
- `run-nullblk-baseline.sh` — baseline `null_blk` regression wrapper.
- `run-nullblk-split-io.sh` — split `null_blk` I/O regression wrapper.
- `run-nullblk-split-rmmod.sh` — split `null_blk` unload regression wrapper.

The QEMU harness now also supports a validated exploratory mode:

- `run-domain-smoke.py --guest-command '...'`

That mode was exercised successfully while probing the `sbi.dom + .smode` runtime path.
It is useful when you want to boot QEMU once, mount the shared `9p` directory,
load `/capstone.ko`, and then run an arbitrary guest-side command without creating
another dedicated harness script first.

## Quick run

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/run-smoke.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt" 2>&1
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt"
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
```

- `capstone-runtime-qemu-smoke-wrapper.txt` is the stdout/stderr of `run-smoke.sh` itself.
- `capstone-runtime-qemu-smoke.log` is the full normalized guest serial/QEMU log.

## Current runtime regression bundle

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh

bash capstone/tests/runtime-qemu/run-shared-region-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-shared-region-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-stdout-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-baseline-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-nullblk-split-io.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-io-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-rmmod-wrapper.txt" 2>&1
```

Inspect the resulting serial logs:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-shared-region-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-stdout-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-baseline.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-io.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-nullb-split-rmmod.log"
```

## Extending it

To add another tiny domain smoke case, add a new `domains/*.c` file with the same shape:

```c
void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = 42;
}
```

Then either:
- build it with `build-domain.sh` and pass the resulting `.dom` to `run-domain-smoke.py`, or
- extend `run-smoke.sh` to build and run it in the same shared directory / QEMU boot.

For guest-side runtime probes that are **not** just `/capstone-test.user <domain>`, you can also use:

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-probe.log" \
  --guest-command "/sbi-dom.user"
```

This is a **runtime probe facility**, not by itself proof that a new architecture
milestone is validated. Only promote a new probe to the validated baseline once it
passes consistently and is documented in the handoff notes.


