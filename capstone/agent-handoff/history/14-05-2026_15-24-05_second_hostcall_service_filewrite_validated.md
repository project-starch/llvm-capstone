# Second HostCall service filewrite validated

## Summary

This step implemented the next planned HostCall milestone after the tightened stdout
proof: validate that the same HostCall v0 metadata ABI and the same borrowed output-
style payload discipline can support a second coarse host service.

The chosen second service was a tiny fixed-path guest tmpfile write:

- domain writes payload bytes into the borrowed payload region,
- domain sets `opcode = HC_V0_OP_WRITE_GUEST_TMPFILE`,
- helper returns from `call_dom()` round 1 and performs ordinary guest Linux file I/O,
- helper writes `result/error/phase` back into shared metadata,
- helper enters the domain again,
- domain validates the response and finishes.

## Why this step mattered

Before this change, the HostCall ABI had only one validated service proof:
`WRITE_STDOUT`.

That was enough to prove control transfer and ownership tightening, but not enough to
show that the metadata ABI was reusable rather than special-cased for one demo.

The filewrite proof establishes that:

- the same fixed-width metadata contract works for a second opcode,
- the same `OUT + BORROWED` payload discipline still works,
- helper-side servicing is not limited to stdout and can use ordinary guest Linux file I/O,
- the wrapper can validate a guest-visible side effect, not just serial output.

## Files added or updated

New files:

- `capstone/tests/runtime-qemu/hostcall-filewrite-probe/hostcall_filewrite_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-filewrite-probe/hostcall_filewrite_probe.smode.c`
- `capstone/tests/runtime-qemu/build-hostcall-filewrite-probe.sh`
- `capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh`

Updated files:

- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h`
- `capstone/tests/runtime-qemu/README.md`
- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/new-chat-prompt.md`
- `capstone/agent-handoff/current/capstone-agent-test-instructions.md`
- `capstone/agent-handoff/current/testing-matrix.md`
- `capstone/agent-handoff/current/current-next-step.md`
- `capstone/agent-handoff/current/split-host-enclave-strategy.md`

## Validation

Shell syntax checks:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash -n capstone/tests/runtime-qemu/build-hostcall-filewrite-probe.sh \
  capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
```

Build validation:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/build-hostcall-filewrite-probe.sh
```

Runtime validation of the new proof:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
```

Repeatability re-run:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
```

Regression checks around the shared ABI/substrate:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
```

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
```

Observed success markers included:

- `hostcall-filewrite-probe: metadata shared, payload borrowed-out`
- `hostcall-filewrite-probe: first call retval = 1`
- `hostcall-filewrite-probe: servicing HC_V0_OP_WRITE_GUEST_TMPFILE`
- `hostcall-filewrite-probe: second call retval = 0`
- `hostcall-filewrite-probe: success`
- `__HOSTCALL_FILEWRITE_OK__`

## Resulting next recommendation

With two output-style HostCall services now validated, the next smallest meaningful
step is no longer “add one more opcode on the same data-flow direction”.

The next step should be to prove the reverse payload direction on the same ABI:

- helper populates payload bytes,
- payload is shared as a borrowed input-style buffer,
- domain consumes those bytes after re-entry,
- metadata still carries the request/response state machine.

