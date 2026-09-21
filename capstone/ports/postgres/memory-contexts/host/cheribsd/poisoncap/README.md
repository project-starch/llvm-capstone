# PoisonCap PostgreSQL: build, link and run

The PoisonCap arm of the PostgreSQL memory-context port. It reuses the platform the
FFmpeg arm prepares — see that port's `host/cheribsd/poisoncap/README.md` for fetching
and building the SDK, kernel, image and QEMU; fetched sources, SDKs, images and logs
stay outside this repository.

Sibling arms of the same port: the `spatial` CheriBSD arm (`host/cheribsd/run.sh`) and
the Capstone domain arm. The plan, the predicted readings and the acceptance gates are
in [`docs/plans/poisoncap-postgres.md`](../../../../../../docs/plans/poisoncap-postgres.md).

## Build

```sh
export CHERI_SDK="$WORK/sdk"
export CHERI_SYSROOT="$WORK/output/rootfs-riscv64-purecap"
bash build.sh "$BUILD" \
  -DPG_CORPUS_SRC="$REPO/capstone/bug-corpora/postgres/mmgr-repros/shared/defects.c"
```

`PG_POISONCAP=ON` selects the `sublet` PostgreSQL variant — the managers' lifetime
hooks must exist — and compiles `src/cheribsd/poisoncap.c` as their backend. Without
`PG_CORPUS_SRC` the build succeeds and simply omits the `defects` program, so check
that `$BUILD/bin/defects` exists before spending a boot on a defect suite.

## Run

```sh
python3 run.py "$BUILD" "$OUT" \
  --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
  --image "$WORK/output/cheribsd-riscv64-purecap.img" \
  --disable-default-revocation
```

`--stage platform` runs the ABI and bounds controls alone; `defects` (the default) adds
the four direct-link manager examples and all sixteen defect arms; `replay` adds the
A11 replay in both modes and needs `--recording`, which
`tests/make-contexts-trace.py OUT.bin` produces.

Automatic libc revocation is off — `--runtime-revocation off` inside the guest — while
the adapter's own sweeps stay on. `--disable-default-revocation` additionally clears the
guest default before SSH starts. That combination is the documented platform
workaround, not a kernel fix.

## Reading the result

`matrix.json` is the verdict; `guest/summary.json` is the per-case record the shared
runner writes, and `guest/<case>/stdout.txt` holds the raw supervisor report.

A defect arm runs under `supervise`, so the signal and trap PC come from the kernel
rather than from the program under test. **A mode-1 arm that faults is not by itself a
result.** The same exit status, 162, is produced by a fault at the labelled stale access
and by a fault anywhere else — an adapter bug, for instance, that hands out a dead
capability at the *next* allocation. `matrix.json` therefore reports `fault_at_probe`
separately from `protected_faulted`, and a pair counts only when the mode-0 arm reached
the same site and completed.

Case 0 is the exception: its stale access is a second `pfree`, so there is no probe and
its oracle accepts a fault anywhere. The matrix marks it `probe_required: false`.

The suite runs every arm even after one fails its oracle. A predicted reading that turns
out wrong must not cost the remaining fifteen arms of the boot, and "the other arms
rejected the same input too" is information a suite that stops at the first failure
cannot produce.

## Prerequisite that is easy to miss

The shared runner drives QEMU through `pexpect`, which the system interpreter here does
not have. Use the platform virtual environment — `$WORK/../venv/bin/python3` in the
pilot layout — and invoke `run.py` with it; the shared runner is spawned with
`sys.executable`, so it inherits the same interpreter.
