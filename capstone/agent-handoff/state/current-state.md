# Current Capstone state

Minimal snapshot. Read first in every session.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, `run-nullblk-split-rmmod.sh`
- `run-coremark.sh` — all three algorithms, "Correct operation validated."

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

Five LLVM backend bugs identified during CoreMark bring-up. All worked around in
`capstone/benchmarks/coremark/build-coremark-capstone.sh`. The prologue bug is the
highest-priority fix — it requires a per-domain hand-written assembly entry, blocking
benchmark porting. Details: `plans/backend-compiler-fixes.md`.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
