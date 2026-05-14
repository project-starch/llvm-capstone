# HostCall stdout borrowed-payload follow-up validated

## Summary

This step tightened the first HostCall stdout runtime proof so the metadata region
stays shared while the payload region uses a stricter borrowed one-direction
handoff.

The concrete code change was:

- keep metadata as `INOUT + SHARED`,
- change payload from broad shared `INOUT + SHARED` to `OUT + BORROWED`,
- keep the same two-round `call_dom()` / `DOM_RETURN(...)` protocol,
- keep the same stdout-visible success behavior.

## Why this step mattered

The earlier stdout proof already showed that a domain could request one host-side
service and validate the response.

That was still slightly too permissive because the payload buffer stayed broadly
shared. This follow-up checked that the same flow still works when the payload is
owned more tightly:

- domain writes the bytes,
- helper reads them after return,
- the runtime applies borrowed revoke semantics to the domain-facing share.

That makes the proof closer to a realistic service ABI without yet widening the
service surface.

## Files changed

- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.h`
- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe_guest.c`
- `capstone/tests/runtime-qemu/hostcall-stdout-probe/hostcall_stdout_probe.smode.c`
- `capstone/tests/runtime-qemu/README.md`
- `capstone/agent-handoff/current/current-next-step.md`
- `capstone/agent-handoff/current/runtime-terms-glossary.md`
- `capstone/agent-handoff/current/split-host-enclave-strategy.md`
- `capstone/agent-handoff/current/testing-matrix.md`

## Validation run

Build validation:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/build-hostcall-stdout-probe.sh
```

Runtime validation:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
```

Re-run for repeatability:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
```

Adjacent baseline re-check:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
```

Observed success markers in the HostCall serial log included:

- `hostcall-stdout-probe: metadata shared, payload borrowed-out`
- `hostcall-stdout-probe: first call retval = 1`
- `hostcall-v0 payload from domain`
- `hostcall-stdout-probe: second call retval = 0`
- `hostcall-stdout-probe: success`

## Resulting recommendation

With the stricter payload handoff now validated, the next smallest meaningful step
is no longer ownership-tightening for the first stdout opcode.

The next step should be:

- keep the same metadata ABI,
- keep the same two-round flow,
- keep the same tighter payload discipline,
- add one second tiny coarse-grained host service,
- revalidate the same runtime wrapper style.

