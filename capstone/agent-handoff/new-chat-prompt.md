# Prompt for continuing this Capstone work in a new chat

Use the following prompt as the opening message in a fresh chat.

---

I am continuing work on the Capstone architecture support in the repository:
- `/home/alexey/dev/llvm-capstone`

## Important working style / constraints
1. If you run terminal commands, **always redirect output into files under `/tmp/alexey/`**, then read those files. Do **not** rely on directly captured terminal output.
2. Be iterative and conservative.
3. Prefer the **smallest meaningful next step** toward the real goal.
4. The main goal is **not** just toy backend work; it is to make the toolchain/runtime capable of compiling and running serious software such as:
   - SPEC-like tests
   - FFmpeg
   - sqlite
   - libpng
5. We currently care about the **SelectionDAG** path, not GISel.
6. When you edit files, preserve existing style and avoid unrelated refactors.
7. After edits, run focused tests and verify behavior.
8. Keep the handoff files in `capstone/agent-handoff/` up to date whenever the validated baseline or workflow changes.

## Read these handoff/context files first
Please read these files before proposing changes:

- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/README.md`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-agent-test-instructions.md`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-backend-status-for-llm.md`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/current-next-step.md` (current recommendation only; do not treat it as immutable)

Also use these logs as verified evidence of the latest native path:
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-my-domain-build-native.txt`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-my-domain-readobj-native.txt`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-qemu-native-proof.txt`
- `/home/alexey/dev/llvm-capstone/capstone/agent-handoff/capstone-lld-lit.txt`

## Current verified state
The following is already implemented and verified:

1. The LLVM Capstone backend can compile the `my_first_domain` sample.
2. Native `ld.lld` support for `EM_CAPSTONE` was added in a minimal way by aliasing Capstone to the existing RISC-V ELF behavior where needed.
3. `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c` was updated to accept both `EM_RISCV` and `EM_CAPSTONE`.
4. `capstone/my_first_domain/build.sh` now defaults to the in-tree `ld.lld`; the old EM_RISCV rewrite shim is only used if `HOST_LD` is overridden to a non-`ld.lld` linker.
5. A focused LLD regression test was added:
   - `lld/test/ELF/emulation-capstone.s`
6. This native flow was validated end-to-end in QEMU:
   - sample domain linked as `EM_CAPSTONE`
   - loader accepted it
   - `/capstone-test.user /test-domains/my_domain.dom` succeeded
   - QEMU no longer hits the old `env->priv < PRV_C` assert in this path

## Very important distinction
The `my_first_domain` flow is a **domain runtime sample**, not yet the same thing as a general hosted user-space program flow.

The next objective should be chosen accordingly.

## What to avoid spending time on right now
Unless it blocks the currently chosen milestone, please postpone:
- pretty disassembly / `llvm-objdump` polishing
- GISel support
- cosmetic cleanups
- changing the harness to print `42` from the sample result buffer
- large speculative refactors

## Expected workflow in the new chat
1. Read the handoff files.
2. Summarize the verified current state in a few bullets.
3. Determine the next smallest meaningful milestone from the current repository state.
4. Probe the existing repository/runtime for that path.
5. If a minimal patch is justified, implement it.
6. Rebuild and test using `/tmp`-redirected logs.
7. Update the handoff files if your changes modify the validated baseline or recommended workflow.
8. Explain exactly what changed and what remains blocked.

When responding, be concrete, cautious, and prefer proven facts over assumptions.


