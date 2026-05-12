# Post-fix follow-up: repro commands, docs refresh, cleanup decisions, and commit planning

Timestamp: 2026-05-12T17:05:00Z

## What the immediately previous step did

This note records the follow-up work that happened **after** the main runtime fix had already been validated.

That follow-up step did four things:

1. wrote down exact reproducible commands for the restored runtime baseline,
2. refreshed the handoff/docs so future sessions use the shared-region + `null_blk` baseline rather than the older assumptions,
3. cleaned obvious junk while avoiding destructive cleanup of ambiguous local helper files,
4. grouped the staged work into a proposed commit plan.

## Runtime verification commands prepared in that step

The commands written down for immediate re-check were:

- `capstone/tests/runtime-qemu/run-shared-region-probe.sh`
- baseline `null_blk` through `run-domain-smoke.py`
- split `null_blk` I/O through `run-domain-smoke.py` with `__SPLIT_DONE__`
- split `null_blk` unload through `run-domain-smoke.py` with `__AFTER_RMMOD__`

Those commands matched the already validated local baseline from the successful fix session.

## Documentation/handoff updates made in that step

The previous step refreshed these durable notes so they reflected the restored runtime baseline:

- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/new-chat-prompt.md`
- `capstone/agent-handoff/current/capstone-agent-test-instructions.md`
- `capstone/agent-handoff/current/testing-matrix.md`
- `capstone/agent-handoff/current/current-next-step.md`
- `capstone/agent-handoff/current/split-host-enclave-strategy.md`

It also removed a no-longer-needed draft note from `current/`.

## Cleanup decisions made in that step

The previous step deleted only files that looked clearly disposable, such as:

- Python `__pycache__` directories,
- temporary `.orig` / generated sample artifacts,
- transient Meson/QEMU wrapper-lock artifacts,
- temporary exported/generated files created only for the debugging session.

It intentionally did **not** make destructive decisions for ambiguous files such as:

- `capstone/CapstoneSpecification.pdf`
- `capstone/capstone-utils/pack-capstone-files.sh`
- `capstone/utils/run-qemu.sh`
- ignored Meson-subproject content under `capstone/capstone-qemu/subprojects/`

## Important caveat discovered afterwards

That follow-up cleanup also added top-level `.gitignore` rules that hid nested component repositories from the outer workspace view.

That turned out to be a bad ergonomic choice for this workspace because the nested repositories must stay visible/editable in the IDE.

The current step should therefore treat that ignore decision as superseded and keep the nested repos visible.

## Commit-planning output from that step

The previous step also drafted a commit split separating:

1. runtime probe/test harness files,
2. sample/build cleanup,
3. handoff/docs/history,
4. buildroot docs,
5. qemu ignore cleanup.

That commit plan was only a proposal, not yet the final split.

