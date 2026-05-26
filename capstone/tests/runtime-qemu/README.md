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
8. the first HostCall-style `WRITE_STDOUT` request/response proof succeeds with shared metadata plus a stricter borrowed payload region,
9. a second HostCall-style `WRITE_GUEST_TMPFILE` request/response proof succeeds on the same metadata ABI and borrowed payload discipline,
10. a reverse-direction HostCall-style `READ_GUEST_TMPFILE` request/response proof succeeds with helper-produced borrowed response payload bytes,
11. a first helper-managed file-handle lifecycle proof succeeds for `FILE_OPEN` followed by `FILE_CLOSE` on one domain invocation,
12. a first handle-based `FILE_WRITE` proof succeeds for `FILE_OPEN -> FILE_WRITE -> FILE_CLOSE` on one domain invocation,
13. a first handle-based `FILE_READ` proof succeeds for `FILE_OPEN -> FILE_READ -> DONE` on one domain invocation,
14. a first handle-based `FILE_SYNC` proof succeeds for `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE` on one domain invocation,
15. a first handle-based `FILE_STAT_BASIC` proof succeeds for `FILE_OPEN -> FILE_STAT_BASIC -> FILE_CLOSE` on one domain invocation,
16. a first handle-based `FILE_TRUNCATE` proof succeeds for `FILE_OPEN -> FILE_TRUNCATE -> FILE_STAT_BASIC -> FILE_CLOSE` on one domain invocation,
17. a first combined file-object proof succeeds for `FILE_OPEN -> FILE_WRITE -> FILE_SYNC -> FILE_CLOSE -> FILE_OPEN -> FILE_READ -> FILE_CLOSE`,
18. baseline `null_blk` loads, performs I/O, and unloads,
19. split `null_blk` loads, performs I/O, and unloads,
20. a first Capstone-compiled SQLite VFS skeleton builds against the official
    `sqlite3.h` amalgamation header and links as a `.dom`; a dedicated QEMU
    wrapper exists for it, but runtime revalidation is currently blocked by the
    same sample-domain assertion that now also affects `run-smoke.sh`.

The current helper-side HostCall proofs also snapshot the shared metadata request
(and, where applicable, the borrowed request payload) immediately after the first
`call_dom()` return before performing host-side work. That keeps each validated
proof from depending on repeated reads of mutable shared state while servicing the
round.

The currently validated HostCall baseline now includes two shapes:

- the earlier single-request **two-round** proofs:
  - one `HC_V0_RET_PENDING` round,
  - host-side servicing from a snapped request,
  - one completion round,
- and a first helper-managed file-object lifecycle proof with:
  - an `OPEN` request round,
  - helper-side response plus explicit revoke-before-reborrow,
  - a `CLOSE` request round,
  - one final completion round.

That does **not** yet mean every arbitrary multi-request flow is validated, but it does
mean the tree now has one working handle-based multi-request path that obeys the
confirmed borrowed-payload lifecycle rule.

The key property is that the domain is provided through a host-shared directory, so the test does **not** rebuild `rootfs.ext2` for each iteration.

## Files

- `build-domain.sh` — generic helper to build a Capstone domain ELF from a tiny `domain_main()` source file.
- `domains/write_42.c` — the initial tiny smoke domain.
- `run-domain-smoke.py` — QEMU + guest automation harness.
- `run-smoke.sh` — one-command entry point that builds the tiny domain and runs the smoke test.
- `run-shared-region-probe.sh` — restored shared-region proof.
- `build-hostcall-stdout-probe.sh` — cross-build helper for the first HostCall stdout proof.
- `run-hostcall-stdout-probe.sh` — first HostCall-style `WRITE_STDOUT` / `puts` regression wrapper.
- `build-hostcall-filewrite-probe.sh` — cross-build helper for the second HostCall filewrite proof.
- `run-hostcall-filewrite-probe.sh` — second HostCall-style `WRITE_GUEST_TMPFILE` regression wrapper.
- `build-hostcall-fileread-probe.sh` — cross-build helper for the reverse-direction HostCall fileread proof.
- `run-hostcall-fileread-probe.sh` — reverse-direction HostCall-style `READ_GUEST_TMPFILE` regression wrapper.
- `build-hostcall-file-open-close-probe.sh` — cross-build helper for the first helper-managed file-handle lifecycle proof.
- `run-hostcall-file-open-close-probe.sh` — helper-managed `FILE_OPEN` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-file-handle-write-probe.sh` — cross-build helper for the first handle-based `FILE_WRITE` proof.
- `run-hostcall-file-handle-write-probe.sh` — helper-managed `FILE_OPEN` / `FILE_WRITE` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-file-handle-read-probe.sh` — cross-build helper for the first handle-based `FILE_READ` proof.
- `run-hostcall-file-handle-read-probe.sh` — helper-managed `FILE_OPEN` / `FILE_READ` regression wrapper.
- `build-hostcall-file-handle-sync-probe.sh` — cross-build helper for the first handle-based `FILE_SYNC` proof.
- `run-hostcall-file-handle-sync-probe.sh` — helper-managed `FILE_OPEN` / `FILE_WRITE` / `FILE_SYNC` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-file-handle-stat-probe.sh` — cross-build helper for the first handle-based `FILE_STAT_BASIC` proof.
- `run-hostcall-file-handle-stat-probe.sh` — helper-managed `FILE_OPEN` / `FILE_STAT_BASIC` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-file-handle-truncate-probe.sh` — cross-build helper for the first handle-based `FILE_TRUNCATE` proof.
- `run-hostcall-file-handle-truncate-probe.sh` — helper-managed `FILE_OPEN` / `FILE_TRUNCATE` / `FILE_STAT_BASIC` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-combined-file-object-probe.sh` — cross-build helper for the first composed file-object proof.
- `run-hostcall-combined-file-object-probe.sh` — helper-managed `FILE_OPEN` / `FILE_WRITE` / `FILE_SYNC` / `FILE_CLOSE` / `FILE_OPEN` / `FILE_READ` / `FILE_CLOSE` regression wrapper.
- `build-hostcall-second-pending-probe.sh` — metadata-only diagnostic for whether a domain may return `PENDING` twice from one invocation.
- `run-hostcall-second-pending-probe.sh` — wrapper for that metadata-only second-`PENDING` diagnostic.
- `build-hostcall-second-pending-payload-probe.sh` — narrower diagnostic for reusing one borrowed output payload region across two successive `PENDING` rounds.
- `run-hostcall-second-pending-payload-probe.sh` — wrapper for that payload-reuse diagnostic.
- `build-hostcall-second-pending-payload-revoke-probe.sh` — workaround/discipline variant that explicitly revokes the borrowed payload region before re-sharing it.
- `run-hostcall-second-pending-payload-revoke-probe.sh` — wrapper for that explicit-revoke payload-reuse diagnostic.
- `run-nullblk-baseline.sh` — baseline `null_blk` regression wrapper.
- `run-nullblk-split-io.sh` — split `null_blk` I/O regression wrapper.
- `run-nullblk-split-rmmod.sh` — split `null_blk` unload regression wrapper.
- `build-sqlite-vfs-skeleton.sh` — downloads the pinned SQLite 3.53.1 amalgamation
  if needed, then builds a Capstone-compiled SQLite-facing VFS skeleton domain.
- `run-sqlite-vfs-skeleton.sh` — executes that SQLite VFS skeleton domain in the
  existing QEMU smoke harness.
- `build-static-cap-globals-probe.sh` — builds a reduced static/global capability
  diagnostic bundle: direct-use control, runtime-side materialization POC,
  descriptor-driven materialization POC, and a file-scope `static const`
  reproducer.
- `run-static-cap-globals-probe.sh` — runs that reduced static/global capability
  diagnostic and currently expects the direct-use control case, the runtime-side
  materialization POC, and the descriptor-driven materialization POC to succeed
  while the `static const` reproducer triggers the known capability failure.
- `static-cap-typed-load-repro/run.sh` — runs a more standalone reduced diagnostic
  that isolates the narrower trigger: a capability-typed runtime load from a
  one-field file-scope static object.
- `static-cap-globals-probe/PAUSE_POINT.md` — records the current SQLite/static-
  capability stopping point and the recommended re-entry step after benchmark work.
- `BENCHMARK_SWITCHOVER.md` — records the recommended transition from the current
  SQLite/static-global line to CoreMark/BEEBS/RV8 bring-up.

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

bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-filewrite-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-fileread-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-open-close-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-write-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-read-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-sync-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-stat-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-truncate-probe-wrapper.txt" 2>&1

bash capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-combined-file-object-probe-wrapper.txt" 2>&1

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
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-filewrite-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-fileread-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-open-close-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-write-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-read-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-sync-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-stat-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-file-handle-truncate-probe.log"
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-hostcall-combined-file-object-probe.log"
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

One such targeted exploratory probe now exists for the current LLVM-generated-domain
blocker around static/global capability-bearing objects:

- `run-static-cap-globals-probe.sh`
  - direct-use control case succeeds,
  - runtime-side materialization POC succeeds,
  - descriptor-driven materialization POC succeeds,
  - file-scope `static const` reproducer currently fails with
    `[CAPSTONE] cs.cjalr requires capability in rs1`,
  - kept as the documented pause point before switching effort to benchmark bring-up.
- `static-cap-typed-load-repro/run.sh`
  - reduces the trigger further to one-field objects,
  - shows a function-pointer typed load failing with
    `[CAPSTONE] cs.cjalr requires capability in rs1`,
  - shows a string-pointer typed load failing with
    `[CAPSTONE] Cap mem access requires capability`,
  - and verifies the smallest manual runtime-materialization workarounds, the
    next-step descriptor-driven path, and a prototype LLVM-IR-generated bridge to
    that materialization shape.
  - the same directory now also fixes the exact candidate `.gct` metadata layout
    for the first LLVM-path compiler-side emission proof of concept.
  - a first in-tree backend-side POC now emits non-empty `SCAP` metadata into
    `.gct` for the narrowed failing static-cap cases; inspect with
    `static-cap-typed-load-repro/inspect-gct-emission.sh`.

## Current targeted diagnostics around multi-round HostCall

Two narrower diagnostic wrappers now exist for the current runtime question:

- `run-hostcall-second-pending-probe.sh`
  - metadata only,
  - no payload reuse,
  - currently shows that a second successive `PENDING` can work in this environment.
- `run-hostcall-second-pending-payload-probe.sh`
  - metadata plus one borrowed output payload region,
  - helper tries to re-share that same payload for the next round,
  - currently reproduces the QEMU assertion in `helper_csmrev`.
- `run-hostcall-second-pending-payload-revoke-probe.sh`
  - same payload-reuse shape,
  - but helper explicitly calls `revoke_region(payload_region_id)` before the second borrowed re-share,
  - currently succeeds.

So the current evidence is more precise than "multi-PENDING is unsupported":

- metadata-only second-`PENDING` works,
- reusing/re-sharing the same borrowed output payload across successive rounds without revoke triggers the current `helper_csmrev` assertion,
- explicitly revoking that region before the next borrowed re-share succeeds.

This now matches the intended runtime rule confirmed by the runtime/QEMU author:

> if a region is already borrow-shared, it must be revoked before anything else can
> be done to it, including borrow-sharing it again.


