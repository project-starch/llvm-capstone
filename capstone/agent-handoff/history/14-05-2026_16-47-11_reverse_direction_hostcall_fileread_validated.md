# Reverse-direction HostCall fileread validated

## Summary

This step implemented and validated the next planned HostCall milestone after the
stdout and filewrite proofs: a reverse-direction payload proof on the same HostCall
v0 metadata ABI.

The new proof keeps the request domain-initiated but flips the response payload
producer:

- round 1: the domain publishes a read-like request in metadata only,
- helper returns from `call_dom()`,
- helper reads bytes from a fixed guest-side tmp file,
- helper shares the payload region back into the domain as borrowed input,
- helper writes `result/error/phase = RESP` into metadata,
- round 2: the domain consumes the helper-produced payload and validates it.

## Why this step mattered

Before this step, the local runtime baseline had already proved:

- a first write-like HostCall service (`WRITE_STDOUT`),
- a second write-like HostCall service (`WRITE_GUEST_TMPFILE`),
- tighter `OUT + BORROWED` ownership for domain-produced payloads.

That still left one important gap: both proofs used the same payload direction
(domain -> helper).

The reverse-direction fileread proof now establishes that:

- the same fixed-width metadata ABI supports a domain-initiated read-like request,
- the helper can produce response payload bytes for the domain,
- the payload can be re-shared as borrowed input immediately before round 2,
- the same two-round control-transfer shape still works.

## Files added or updated

New files:

- `capstone/tests/runtime-qemu/hostcall-fileread-probe/hostcall_fileread_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-fileread-probe/hostcall_fileread_probe.smode.c`
- `capstone/tests/runtime-qemu/build-hostcall-fileread-probe.sh`
- `capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh`

Updated files:

- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h`
- `capstone/tests/runtime-qemu/README.md`
- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/new-chat-prompt.md`
- `capstone/agent-handoff/current/capstone-agent-test-instructions.md`
- `capstone/agent-handoff/current/testing-matrix.md`
- `capstone/agent-handoff/current/current-next-step.md`
- `capstone/agent-handoff/current/split-host-enclave-strategy.md`
- `capstone/agent-handoff/current/runtime-terms-glossary.md`

## Validation

Shell syntax checks:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash -n capstone/tests/runtime-qemu/build-hostcall-fileread-probe.sh \
  capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
```

Build validation:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/build-hostcall-fileread-probe.sh
```

Runtime validation of the new proof:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
```

Repeatability re-run:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
```

Regression checks around the shared ABI/substrate:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
```

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
```

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
```

Observed success markers included:

- `hostcall-fileread-probe: metadata shared`
- `hostcall-fileread-probe: first call retval = 1`
- `hostcall-fileread-probe: servicing HC_V0_OP_READ_GUEST_TMPFILE`
- `hostcall-fileread-probe: payload shared as borrowed-in response`
- `hostcall-fileread-probe: second call retval = 0`
- `hostcall-fileread-probe: success`
- `__HOSTCALL_FILEREAD_OK__`

## Resulting next recommendation

With multiple HostCall services validated in both payload directions, the next
smallest meaningful step is no longer another proof-only opcode.

The next step should be to define the first small stable coarse service subset:

- one buffered write-like service family,
- one read-like response service family,
- one explicit rule for what remains local inside the domain runtime/libc layer.

