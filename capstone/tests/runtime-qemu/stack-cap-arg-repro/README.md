# Stack-passed capability argument repro (backend ABI bug)

**Status: FIXED (2026-06-24).** `CapstoneTargetLowering::LowerCall` now derives
the outgoing stack-argument slot address with a capability `CIncOffset` (using the
alloca/capability pointer type) instead of an integer `ISD::ADD`, so stack-passed
capability arguments keep their tag. Regression test:
`llvm/test/CodeGen/Capstone/stack-cap-arg.ll`. This domain returns
`BEEBS_RET_CORRECT` and RV8 `norx` now passes. Kept as a runtime repro.

## Symptom

A function with more than 8 arguments whose 9th+ arguments are pointers
(capabilities) receives those stack-passed arguments **untagged**. Dereferencing
one traps:

```
[CAPSTONE] Cap mem access requires capability: pc = ..., rs1 = x.., imm = 0
qemu-system-riscv64: riscv_cpu_do_interrupt: Assertion `env->priv < PRV_C' failed.
```

## Root cause

RISC-V passes the first 8 integer/pointer args in `a0`-`a7`; further args go on
the stack. On Capstone a pointer is a 128-bit capability, so a stack-passed
pointer argument must be passed in a **16-byte tagged capability slot**
(`stc`/`lc`), not a plain 8-byte `sd`/`ld`. The backend currently passes/loads
such stack arguments as plain 8-byte integers, so the tag is lost and the callee
gets an untagged pointer.

This is the same class as the already-fixed `va_list` capability-tag-loss bug
(capabilities travelling through the calling convention's memory path), and is
the root cause of the deferred RV8 `norx` benchmark: `cf_norx32_encrypt` has 10
parameters and its `ciphertext`/`tag` (args 9-10) are stack-passed capabilities.

`stack_cap_arg_domain.c` is the minimal reproducer: `f()` takes 8 `long`s (filling
`a0`-`a7`) then two `int *` (the 9th/10th args → stack); writing through them
faults. Confirmed by bisection from norx (encrypt-only, then `nbytes=0`, then this
reduced repro), all faulting at the same instruction.

## Fix location (proposed)

Backend calling-convention lowering for Capstone
(`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` `LowerFormalArguments` /
`LowerCall`, and/or `CapstoneCallingConv*`): stack-passed arguments of capability
(addrspace-200 pointer / 128-bit) type must use 16-byte capability load/store with
16-byte stack-slot size and alignment, mirroring how the `va_list` fix handles
capability slots. Both caller (store outgoing arg) and callee (load incoming arg)
sides must agree.

## Build / run

Build like the other reduced `runtime-qemu` domains (`start.S` + `link.ld` +
`beebs_simple_domain.c` harness); run under `run-domain-smoke.py`. Expected once
fixed: `BEEBS_RET_CORRECT` (`g8==11 && g9==22`).
