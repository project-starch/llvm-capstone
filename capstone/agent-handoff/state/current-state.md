# Current Capstone state

Minimal snapshot. Read first in every session.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, `run-nullblk-split-rmmod.sh`
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Four remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
