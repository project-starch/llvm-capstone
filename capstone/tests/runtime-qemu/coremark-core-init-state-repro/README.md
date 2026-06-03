# CoreMark `core_init_state()` runtime reproducer

This directory contains a very small Capstone domain that does only one thing:

- call upstream CoreMark `core_init_state()` on a stack buffer.

On the current Capstone runtime/toolchain path, that is enough to reproduce the
narrower blocker observed while chasing the broader CoreMark bring-up:

- `qemu-system-riscv64: ../target/riscv/op_helper.c:591: helper_cscincoffset: Assertion 'rs1_v->tag' failed.`

Use the wrappers in the parent directory:

- `build-coremark-core-init-state-repro.sh`
- `run-coremark-core-init-state-repro.sh`

