# Capstone backend/compiler status memo (for another LLM)

Repository state inspected at `HEAD`:
- branch: `capstone-bootstrap`
- current head from local log at the time this memo was refreshed: `8b7450001ade` (`Add agent handoff documentation for Capstone backend/toolchain`)

This memo distinguishes between:
- **implemented in source**,
- **verified by tests or runtime evidence**, and
- **still missing / still provisional**.

---

# 1. High-level status

## Overall status
The Capstone LLVM backend is already beyond a toy prototype. It currently has a **working PureCap-oriented codegen path** for a meaningful subset of C code, including:
- 128-bit capability pointers in AS200,
- capability arithmetic and comparisons,
- capability loads/stores,
- ordinary scalar loads/stores through capability pointers,
- capability-specific intrinsics,
- domain-crossing / world-switching intrinsics,
- PureCap-safe frame lowering,
- dynamic alloca lowering,
- capability-safe aggregate copies for aligned 16-byte memcpy/memmove,
- basic external symbol calls through the PureCap call materialization path,
- a demonstrated end-to-end example where in-tree LLVM `clang` compiles a small domain that executes successfully inside the Capstone QEMU/Buildroot environment.

However, the system is **not yet a native full hosted toolchain bring-up** because:
- the minimal sample-domain flow now works natively with `ld.lld` and `EM_CAPSTONE`,
- the Buildroot userspace loader now accepts both `EM_RISCV` and `EM_CAPSTONE`,
- but the broader `clang -> crt/startfiles -> libc/sysroot -> hosted program -> runtime` path is still incomplete,
- and large real-world applications still require more runtime/toolchain integration work.

So the most accurate summary is:

> **Backend/codegen is already meaningfully functional and VM-validated for a small domain example, including a native `ld.lld + EM_CAPSTONE + loader` path for that sample, but the broader hosted toolchain/runtime remains incomplete.**

---

# 2. Implemented frontend/compiler surface

## 2.1 Clang target builtins for Capstone
Implemented in:
- `clang/include/clang/Basic/BuiltinsCapstone.td`
- `clang/lib/CodeGen/TargetBuiltins/Capstone.cpp`

Builtins that are present in the compiler surface include at least:
- capability manipulation builtins already present earlier,
- domain crossing / world switching / CSR builtins:
  - `__builtin_capstone_cap_call`
  - `__builtin_capstone_cap_return`
  - `__builtin_capstone_cap_enter`
  - `__builtin_capstone_cap_exit`
  - `__builtin_capstone_cap_ccsrrw`

### Verified evidence
- `clang/test/CodeGen/capstone-builtins.c` checks that:
  - `__builtin_capstone_cap_return()` lowers to `llvm.capstone.cap.return.*`
  - `__builtin_capstone_cap_exit()` lowers to `llvm.capstone.cap.exit.*`
  - and Clang emits `unreachable` afterward for the noreturn builtins

So the Clang frontend integration for at least the domain-return / exit builtins is directly regression-tested.

---

## 2.2 LLVM IR intrinsics for Capstone
Implemented in:
- `llvm/include/llvm/IR/IntrinsicsCapstone.td`

Implemented intrinsic families include:

### Capability manipulation intrinsics
Verified from source/tests:
- `llvm.capstone.cap.get.tag`
- `llvm.capstone.cap.set.bounds`
- `llvm.capstone.cap.tighten`
- `llvm.capstone.cap.split` may or may not still be incomplete elsewhere; do not assume full lowering without re-checking current sources
- `llvm.capstone.cap.delin`
- `llvm.capstone.cap.lcc`
- `llvm.capstone.cap.scc`
- `llvm.capstone.cap.shrink.to`
- `llvm.capstone.cap.mrev`
- `llvm.capstone.cap.seal`
- `llvm.capstone.cap.drop`
- `llvm.capstone.cap.revoke`
- `llvm.capstone.cap.ccsrrw`

### Domain / world-switching intrinsics
Implemented in source and tests:
- `llvm.capstone.cap.call`
- `llvm.capstone.cap.return`
- `llvm.capstone.cap.enter`
- `llvm.capstone.cap.exit`
- `llvm.capstone.cap.ccsrrw`

### Important semantic fixes already present
- `cap.return` and `cap.exit` are modeled as **noreturn** intrinsics on the frontend side (and are used with `unreachable` in tests)
- memory side-effect modeling was corrected so TableGen no longer rejects the intrinsic definitions

---

# 3. Implemented backend instruction selection / lowering

## 3.1 SelectionDAG path is the supported path
The user explicitly does **not** care about GISel at this stage.
The code and tests added in this effort are centered on **SelectionDAGISel**.

Relevant files include:
- `llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp`
- `llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.h`
- `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`

---

## 3.2 Domain crossing / world switching / CSR instruction selection
Implemented in source:
- `CAP_CALL`
- `CAP_RETURN`
- `CAPENTER`
- `CAPEXIT`
- `CCSRRW`

Instruction definitions exist in:
- `llvm/lib/Target/Capstone/CapstoneInstrInfo.td`

Selection functions exist in:
- `selectCapCall`
- `selectCapReturn`
- `selectCapEnter`
- `selectCapExit`
- `selectCCSRRW`

### Verified evidence
`llvm/test/CodeGen/Capstone/cap-control-flow.ll` checks:
- `llvm.capstone.cap.call` lowers to `call a0, a0`
- `llvm.capstone.cap.enter` lowers to `capenter`
- `llvm.capstone.cap.return` lowers to `return a0, a1`
- `llvm.capstone.cap.exit` lowers to `capexit a0, a1`
- no extra `cjalr zero, 0(ra)` is emitted after return/exit
- MIR also checks that `CAP_CALL` / `CAPENTER` carry the call-preserved register mask operand (`csr_ilp32_lp64`) and that `CAPENTER` result is truncated through `PseudoTRUNC_CAP`

This means the backend already models these as meaningful control-flow boundaries in SelectionDAG lowering without requiring GISel.

### Practical semantic notes
- `CAP_CALL` / `CAPENTER` currently avoid `isCall = 1` because that would require a standard LLVM callee operand layout and would interact badly with debug/callsite machinery.
- Instead, the implementation uses a **register mask operand** so register allocation/clobber semantics are modeled well enough for the current fast-path bring-up.
- `CAP_RETURN` / `CAPEXIT` are treated as real terminators/returns on the backend side.

---

## 3.3 Capability-specific base instructions
Implemented and/or tested capability instructions include:
- `LDC`
- `STC`
- `CIncOffset`
- `CIncOffsetImm`
- `MOVC`
- `LCC`
- `SCC`
- `DELIN`
- `TIGHTEN`
- `SHRINKTO`
- `MREV`
- `SEAL`
- `DROP`
- `REVOKE`

### Verified evidence
- `llvm/test/CodeGen/Capstone/intrinsics.ll`
- `llvm/test/CodeGen/Capstone/load-store.ll`
- earlier commit history visible in local log

These collectively show that core capability-manipulation operations are present and lowering into named Capstone instructions.

---

# 4. PureCap pointer model and ordinary C memory access

## 4.1 PureCap pointer representation
The current compiler path is clearly set up around:
- **AS200** pointers,
- 128-bit capability pointer semantics,
- PureCap-oriented lowering choices.

This is visible from tests using:
- `ptr addrspace(200)`

and from the entire family of i128 capability lowering rules.

---

## 4.2 Capability loads/stores themselves
Verified in:
- `llvm/test/CodeGen/Capstone/load-store.ll`

Directly checked behaviors:
- capability store lowers to `stc`
- capability load lowers to `ldc`

Examples from the test:
- `store ptr addrspace(200) ...` → `stc`
- `load ptr addrspace(200) ...` → `ldc`

---

## 4.3 Ordinary scalar memory access through capability pointers
This is one of the important PureCap milestones already implemented.

Verified in:
- `llvm/test/CodeGen/Capstone/load-store.ll`

Checked behaviors:
- `load i32` through `ptr addrspace(200)` lowers to `lw ... (capability base)`
- `store i32` through `ptr addrspace(200)` lowers to `sw ... (capability base)`
- GEP offsets can fold into the ordinary load/store immediate when small
- large offsets use `cincoffset` plus a zero-offset load/store

Examples directly checked:
- `lw a0, 0(a0)`
- `sw a1, 0(a0)`
- `lw a0, 16(a0)`
- `sd a1, 32(a0)`
- large offset case: materialize offset → `cincoffset` → `lw 0(...)`

This is crucial because it means ordinary C code like:
- `*p`
- `p[i]`
- `struct->field`

already has a backend path when `p` is a 128-bit capability pointer.

---

## 4.4 Folding `CIncOffset` into memory addressing
Implemented in SelectionDAG address selection.

Verified in:
- `llvm/test/CodeGen/Capstone/load-store.ll`

Meaning:
- small capability-pointer offsets are folded into `simm12` for standard scalar load/store instructions,
- so GEP on capability pointers does not always expand into a separate instruction.

This is already an important quality-of-codegen improvement and avoids unnecessary extra instructions for common field/index accesses.

---

# 5. Control flow and comparison-related support for capabilities

## 5.1 i128 / capability comparisons and conditions
From the local commit history and current code state, the backend already contains work for:
- `SETCC` with i128 support
- `BR_CC`/condition handling for capability-related comparisons
- null capability select lowering fix
- `SELECT` handling for i128/capability values

### What is verified directly
The current tree definitely contains:
- a recent commit `d5b5de228b38 Fix null capability select lowering`
- `CapstoneISelLowering.cpp` contains `setOperationAction(ISD::SELECT, MVT::i128, Custom);`
- `setOperationAction(ISD::SELECT_CC, MVT::i128, Expand);`

So pointer/capability selects are no longer obviously in the “Cannot select” state that existed earlier in development.

### Confidence level
- **Implemented in source:** yes
- **Direct dedicated regression evidence from the files inspected right now:** moderate, but not as strong as for load/store or control-flow intrinsics unless more tests are read

---

# 6. PureCap-safe stack and frame lowering

This area has already received substantial implementation work and is one of the major backend milestones already completed.

## 6.1 Capability-safe fixed frame adjustment
Implemented in frame lowering / register info so that stack pointer adjustments do not use integer `ADD/ADDI` on capability registers.

Verified in:
- `llvm/test/CodeGen/Capstone/frame-lowering.ll`

Checked behaviors include:
- stack allocation with `cincoffsetimm sp, sp, -16`
- stack deallocation with `cincoffsetimm sp, sp, 16`
- saving `ra` with `stc`
- reloading `ra` with `ldc`
- large frame adjustment via synthesized offset register + `cincoffset`
- CFI / CFA offset emission is still preserved in the large-frame case

This is important because it means normal prologue/epilogue generation no longer strips capability tags from stack-related registers.

---

## 6.2 Fixed stack realignment
This was initially planned as a failure path, but the current local history shows a later commit:
- `36823b63a78a Support fixed PureCap stack realignment in frame lowering`

Verified in:
- `llvm/test/CodeGen/Capstone/frame-realign.ll`

The test checks a realignment sequence shaped like:
- allocate frame with `cincoffsetimm`
- keep the old stack in `s0`
- read the cursor with `lcc`
- align the integer cursor with `andi`
- write the aligned cursor back with `scc`
- restore stack capability on exit

So **fixed stack realignment is already implemented**, not merely rejected.

---

## 6.3 Dynamic stack allocation / VLA lowering
This is also already implemented now.

Verified in:
- `llvm/test/CodeGen/Capstone/dynamic-alloca.ll`

The checked code pattern is:
- round requested size up to alignment using integer ops on the size,
- negate the size,
- apply it to the capability stack pointer via `cincoffset`,
- update `sp` via `movc`

Specifically, the test expects:
- `addi`
- `andi`
- `neg`
- `cincoffset a0, sp, a0`
- `movc sp, a0`

So **runtime-sized alloca/VLA support is already present** for the supported cases.

---

## 6.4 Dynamic alloca + unsupported realignment failure path
Verified in:
- `llvm/test/CodeGen/Capstone/dynamic-alloca-realign-fail.ll`

This test checks that the backend fails closed with:
- `LLVM ERROR: Stack realignment is not supported yet in Capstone PureCap`

for a dynamic alloca requiring large alignment.

Interpretation:
- ordinary dynamic alloca is supported,
- but the combination of dynamic alloca with the not-yet-supported form of realignment is intentionally rejected.

That is a reasonable current-state compromise for a bring-up backend.

---

# 7. Capability-sized aggregate copy / tag-preserving mem intrinsics

This area is already implemented and regression-tested.

## 7.1 16-byte memcpy/memmove preservation
Verified in:
- `llvm/test/CodeGen/Capstone/mem-intrinsics.ll`

The test explicitly checks that aligned 16-byte copies in AS200 lower to:
- one `ldc`
- one `stc`

rather than generic scalar splitting.

This matters because splitting a capability-bearing aggregate into 64-bit integer loads/stores would destroy the tag semantics.

The current implementation includes:
- a backend definition of `findOptimalMemOpLowering(...)`
- behavior consistent with choosing capability-sized copies for aligned PureCap memory operations

So small aligned aggregate copies already have a tag-preserving lowering path.

---

# 8. Safety hardening for capability constants and offsets

This is already implemented and regression-tested.

## 8.1 Reject forging of arbitrary >64-bit capability constants
Implemented in SelectionDAG isel.

Verified in:
- `llvm/test/CodeGen/Capstone/cap-constants-invalid.ll`

The backend now explicitly crashes with a controlled fatal error for invalid attempts to materialize arbitrary >64-bit capability constants, with diagnostics such as:
- `Capstone PureCap: Cannot materialize arbitrary >64-bit constants as capabilities; capabilities are unforgeable`

This is consistent with the Capstone/CHERI-style capability security model: non-zero capabilities are not supposed to be forged from arbitrary integers.

---

## 8.2 Reject invalid wide GEP / address displacements
Also verified in:
- `llvm/test/CodeGen/Capstone/cap-constants-invalid.ll`

The backend explicitly rejects cases where:
- `CIncOffset` displacement exceeds signed 64-bit,
- folded address displacement exceeds signed 64-bit,
- folded load/store displacement exceeds signed 64-bit.

Representative diagnostics checked by the test:
- `CIncOffset displacement must fit in signed 64-bits`
- `Address displacement must fit in signed 64-bits`
- `Folded load/store displacement must fit in signed 64-bits`

This is good hardening: the backend fails closed rather than silently truncating or generating nonsense.

---

# 9. Call lowering improvements for external symbols

Implemented and tested.

Verified in:
- `llvm/test/CodeGen/Capstone/external-calls.ll`

The current lowering path for external-symbol calls already does:
- materialize a PC-relative offset (`auipc` + `%pcrel_lo`)
- derive a callable capability from `gp` using `cincoffset`
- perform `cjalr`

The test explicitly checks:
- `auipc`
- `addi`
- `cincoffset ... gp ...`
- `cjalr ra, 0(cap)`

So compiler-generated ExternalSymbol calls no longer rely on a broken or non-PureCap path.

---

# 10. Runtime-validated example using in-tree LLVM clang

This is beyond pure llc-level testing.

Implemented example in:
- `capstone/my_first_domain/start.S`
- `capstone/my_first_domain/main.c`
- `capstone/my_first_domain/build.sh`
- `capstone/my_first_domain/README.md`

## What is validated
A small domain example now works in the actual Capstone QEMU + Buildroot environment using:
- in-tree LLVM `clang` for code generation,
- a handwritten ABI wrapper in `start.S`,
- the existing Buildroot linker/loader runtime with the temporary `EM_RISCV` shim.

### Observed runtime evidence from the validated runtime logs and handoff proof files
The captured run shows:
- the domain ELF is accepted by the userspace loader,
- a domain is created successfully,
- the domain executes,
- the domain returns without triggering the earlier QEMU assertion.

Observed lines include:
- `Created domain ID = 0`
- `Called dom (1-th time) retval = 0`

This is very important because it confirms that the backend is no longer only “passing llc tests”; it can already compile code that runs inside the current Capstone VM environment.

### Important caveat
This sample path is now a **native Capstone ELF flow** for the validated domain example:
- in-tree `clang` compiles the code,
- in-tree `ld.lld` links it as `EM_CAPSTONE`,
- the userspace loader accepts `EM_CAPSTONE`,
- and QEMU runtime execution has been revalidated.

So this now validates **code generation, native sample linking, and runtime ABI correctness** for the sample-domain path, but **not yet the broader hosted toolchain/runtime path**.

---

# 11. What is still missing / still provisional

The following items are still not “done” enough to claim a full Capstone native toolchain.

## 11.1 Native linker / ELF bring-up is partially complete
Current evidence now shows:
- `ld.lld` can link the validated sample-domain path as native `EM_CAPSTONE`,
- the Buildroot userspace loader accepts `EM_CAPSTONE`,
- the old machine-header rewrite workaround is no longer needed in the default sample flow.

What remains is the broader hosted-toolchain/link/runtime bring-up, not the minimal sample-domain native ELF path.

---

## 11.2 Disassembler support is incomplete
`llvm-objdump` still prints many custom Capstone instructions as `<unknown>`.
That does not block execution, but it does mean disassembly/debug ergonomics are incomplete.

---

## 11.3 GISel is intentionally not the target right now
The current effort is centered on SelectionDAGISel.
That is acceptable for the stated near-term goal.
But it also means one should not assume GlobalISel support is complete or relevant.

---

## 11.4 Full runtime/sysroot/libc bring-up remains separate work
Even with the backend in much better shape now, a full FFmpeg bring-up still requires work beyond the backend itself, including:
- native linker support,
- runtime/sysroot consistency,
- allocator/libc support,
- likely additional validation for atomics and broader library code.

This is outside the narrow “backend implemented?” question, but relevant for roadmap planning.

---

# 12. Suggested verified-vs-unverified summary

## Verified now (strong confidence)
These have direct code/test/runtime evidence in the inspected tree:
- Clang builtins for `cap_return` / `cap_exit` with `unreachable` emission
- LLVM intrinsics for domain crossing / world switching / CSR path
- SelectionDAG lowering for `CAP_CALL`, `CAPENTER`, `CAP_RETURN`, `CAPEXIT`, `CCSRRW`
- capability manipulation instructions such as `MREV`, `SEAL`, `DROP`, `REVOKE`
- capability loads/stores via `LDC` / `STC`
- ordinary scalar loads/stores through capability pointers
- folding of `CIncOffset` into scalar memory-address immediates
- PureCap-safe fixed frame lowering
- fixed stack realignment
- dynamic alloca lowering (supported subset)
- fail-closed rejection of unsupported dynamic alloca + realignment case
- tag-preserving aligned 16-byte memcpy/memmove lowering
- fail-closed handling for forged >64-bit capability constants and invalid wide displacements
- external-symbol call lowering through the PureCap capability-materialization path
- runtime execution of `my_first_domain` compiled by in-tree LLVM `clang` inside the current QEMU/Buildroot environment

## Implemented in source, but re-check before making strong claims
These likely exist but were not fully re-audited in this pass:
- complete breadth of all older capability intrinsics beyond the ones directly touched/tested in this work
- completeness of all compare/select corner cases for every optimization level / IR shape
- completeness of any GISel-side analogues

## Definitely still incomplete / provisional
- broader hosted `clang + crt/startfiles + libc/sysroot + runtime` bring-up
- polished disassembly support
- full toolchain/runtime path needed for large applications like FFmpeg without workarounds

---

# 13. Bottom-line answer for another LLM

If another LLM needs a single concise status statement, use this:

> The Capstone backend is already functionally substantial: SelectionDAG-based PureCap codegen for AS200 capability pointers works for scalar/capability memory accesses, capability intrinsics, domain-crossing intrinsics, PureCap-safe frame lowering, dynamic alloca, and tag-preserving aligned aggregate copies, with both regression tests and a VM-validated `my_first_domain` example compiled by the in-tree LLVM clang and linked natively through `ld.lld` as `EM_CAPSTONE`. The biggest remaining gap is no longer the minimal sample-domain ELF path, but the broader hosted runtime/sysroot/toolchain work needed for serious real-world applications.

