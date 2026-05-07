# Capstone runtime QEMU smoke tests

This directory contains a minimal runtime smoke test for the **current validated domain ABI path**.

It intentionally does **not** try to validate the broader hosted Linux user-space flow yet.
That hosted flow is still blocked earlier in the toolchain/sysroot integration.

## What this smoke test validates

In one QEMU boot it verifies that:

1. a tiny domain can be built by the in-tree LLVM Capstone toolchain,
2. the guest boots,
3. the host-shared `9p` directory can be mounted inside the guest,
4. the `capstone` kernel module loads,
5. `/capstone-test.user` accepts and executes the domain from the shared directory,
6. the expected success markers appear.

The key property is that the domain is provided through a host-shared directory, so the test does **not** rebuild `rootfs.ext2` for each iteration.

## Files

- `build-domain.sh` — generic helper to build a Capstone domain ELF from a tiny `domain_main()` source file.
- `domains/write_42.c` — the initial tiny smoke domain.
- `run-domain-smoke.py` — QEMU + guest automation harness.
- `run-smoke.sh` — one-command entry point that builds the tiny domain and runs the smoke test.

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


