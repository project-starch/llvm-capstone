# Current Capstone state

This is the smallest durable snapshot that a future session should read first.
It is intentionally short.

## Read this first

If you only need the current baseline and the next milestone, read:

1. `capstone/agent-handoff/README.md`
2. `capstone/agent-handoff/current/current-state.md`
3. `capstone/agent-handoff/current/current-next-step.md`

Read deeper documents only when the task actually needs them.

## Current verified baseline

The following is already implemented and verified in the current tree:

- the in-tree LLVM Capstone backend builds the sample domain,
- in-tree `ld.lld` links the sample as native `EM_CAPSTONE`,
- the Buildroot userspace loader accepts the domain without the old manual ELF-header rewrite hack in the default path,
- `capstone/caplifive-buildroot/build/local.mk` is present and keeps the image on the local Capstone-enabled Linux/OpenSBI override path,
- rerunning `make build CAPSTONE_CC_PATH=... A=opensbi-rebuild` restores the intended OpenSBI/runtime path,
- `capstone/tests/runtime-qemu/run-shared-region-probe.sh` passes,
- `capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh` passes,
- `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh` passes,
- `capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh` passes,
- `capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh` passes,
- `capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh` passes,
- baseline `null_blk` passes,
- split `null_blk` creates `/dev/nullb0`, completes I/O, and unloads successfully after rebuilding against the active kernel.

## What the current HostCall proofs establish

The current runtime proofs now cover:

- shared metadata across two `call_dom()` rounds,
- a tighter borrowed payload discipline,
- helper-side request snapshotting before servicing the current working HostCall proofs,
- reuse of one metadata ABI across more than one coarse service,
- a first helper-managed file-handle lifecycle path,
- a first helper-managed file-handle byte-write path,
- both payload directions:
  - domain -> helper output-style payload,
  - helper -> domain input-style payload.

This means the immediate unknown is no longer "can one more toy read/write proof work?".

The currently validated HostCall shapes are now:

- the original single-request path:
  - one `PENDING` return,
  - helper-side service from a snapped request,
  - one completion return,
- and a first handle-lifecycle multi-request path:
  - `FILE_OPEN` request,
  - helper response plus explicit payload revoke-before-reborrow,
  - optional later `FILE_WRITE` request on the returned token,
  - helper response plus explicit payload revoke-before-reborrow,
  - `FILE_CLOSE` request,
  - one final completion return.

A more precise diagnostic result is now available:

- a minimal metadata-only probe shows that a second successive `PENDING` from one
  domain invocation **does work** in the current environment,
- but a narrower probe that reuses/re-shares the same borrowed output payload across
  that next round reproduces the QEMU assertion
  `helper_csmrev: Assertion \`rs1_v->val.cap.type == CAP_TYPE_LIN\` failed.`

That limitation is now explained by an authoritative runtime/QEMU answer:

- if a region is already borrow-shared, it must be explicitly revoked before doing
  anything else with it, including borrow-sharing it again,
- the current "borrowed output" behavior is a C-mode SBI abstraction layer rather
  than a guarantee that repeated re-share without revoke is supported automatically.

So the current blocker is not simply "second `PENDING` unsupported". The precise rule is:

> repeated borrowed-output payload reuse across rounds is supported only if the
> helper first calls `revoke_region()` on that payload region before re-sharing it.

This matches the current probes:

- metadata-only second `PENDING` works,
- payload reuse without revoke reproduces `helper_csmrev`,
- payload reuse with explicit `revoke_region()` before the second borrowed re-share passes.

## Important distinction

The validated path today is still the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace.

Concretely:

- the helper is still ordinary guest Linux userspace,
- the domain is still a Capstone-loaded domain,
- the working service boundary is shared regions + synchronous multi-round HostCall,
- this is not yet evidence that a normal hosted Capstone libc/sysroot stack is ready.

## Current hosted Linux status

The current Buildroot sysroot is still a normal `riscv64 + glibc + lp64d` userspace world.
That does **not** match the current Capstone capability-oriented pointer model.

So the hosted path is still blocked by at least:

- libc header / ABI mismatch,
- dynamic-loader naming / ABI mismatch.

See `current/hosted-libc-os-analysis.md` only if the task is specifically about the hosted Linux path.

## Current next milestone in one sentence

The next meaningful step is to stop adding one-off proof opcodes and define a first small,
stable, reusable file-service subset on top of the already validated HostCall v0 boundary.

## Fast runtime entry points

The current most useful runtime wrappers are:

```bash
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh
bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh
bash capstone/tests/runtime-qemu/run-nullblk-split-io.sh
bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh
```

## Files to read only on demand

- `current/testing-matrix.md` — compact test map and when to run which layer.
- `current/capstone-agent-test-instructions.md` — practical command cookbook.
- `current/split-host-enclave-strategy.md` — source-backed architectural detail.
- `current/stable-file-service-subset.md` — first reusable file-service proposal.
- `current/hostcall-file-service-v0-wire-spec.md` — practical opcode/payload/state-machine spec for the next file-service implementation.
- `current/hosted-libc-os-analysis.md` — why hosted Linux user-space is still blocked.
- `history/README.md` — historical index and what is still worth reading.

