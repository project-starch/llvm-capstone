# Current Capstone state

Minimal snapshot. Read first in every session.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- `run-shared-region-probe.sh`
- `run-hostcall-stdout-probe.sh`
- `run-hostcall-filewrite-probe.sh`
- `run-hostcall-fileread-probe.sh`
- `run-hostcall-file-open-close-probe.sh`
- `run-hostcall-file-handle-write-probe.sh`
- `run-hostcall-file-handle-read-probe.sh`
- `run-hostcall-file-handle-sync-probe.sh`
- `run-hostcall-file-handle-stat-probe.sh`
- `run-hostcall-file-handle-truncate-probe.sh`
- `run-hostcall-path-access-probe.sh`
- `run-hostcall-path-delete-probe.sh`
- `run-hostcall-combined-file-object-probe.sh`
- `run-nullblk-baseline.sh`
- `run-nullblk-split-io.sh`, `run-nullblk-split-rmmod.sh`
- `run-coremark.sh` — all three algorithms, "Correct operation validated."

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

Five LLVM backend bugs identified during CoreMark bring-up. All worked around in
`capstone/benchmarks/coremark/build-coremark-capstone.sh`. No domain code is currently
blocked. Details: `plans/backend-compiler-fixes.md`.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
