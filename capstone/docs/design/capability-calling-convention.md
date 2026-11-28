# The capability calling convention: how arguments are passed

The Capstone target's calling convention is RISC-V's (lp64, a0-a7, then the stack), with the rules below for values
that are or carry capabilities. They are implemented in `llvm/lib/Target/Capstone/CapstoneCallingConv.cpp`
(`CC_Capstone`) and `CapstoneISelLowering.cpp`. Each rule names the issue that set it and the lit test that pins it.

## Capability arguments and results

A capability (`c128`, a pointer in address space 200) takes one argument register, named as its capability register
(`C10`..`C17`, the same hardware registers as `a0`..`a7`, from `Capstone::getArgGPCRs`). When no register is left it
takes one 16-byte, 16-aligned stack slot. An integer register would carry only the cursor and drop the tag.

## Variadic arguments (C-48)

A variadic argument that overflows the argument registers takes a 16-byte, 16-aligned stack slot, whatever its type.
`va_arg` advances by that stride, and the register save area holds one capability-sized slot per register. Fixed
arguments keep their XLen-sized slots.

## Split integers passed indirectly (C-68)

A scalar wider than two XLen words (i256, `_BitInt(>128)`) is passed by address: the caller stores its parts in a
stack temporary. Under a capability alloca address space that address is a capability and is passed as one, in a
`C` register or a 16-byte, 16-aligned stack slot. The callee loads the parts through it. Before 2026-11-20 the
address went in an X register (`XLenVT`), and no caller compiled. Pinned by
`llvm/test/CodeGen/Capstone/c68-split-integer-indirect-capability.ll`.

## Exceptions (C-63)

Under a capability personality (address space 200) the exception pointer arrives at a landing pad in `C10` and the
selector in `X11`. Under an integer personality the pointer is `X10`. A domain's unwinder has to honour this; none
exists yet. Pinned by `llvm/test/CodeGen/Capstone/c63-landingpad-capability-exception-pointer.ll`.

## Image flags

`e_flags` bits of a Capstone ELF object or image, beyond RISC-V's (`EF_CAPSTONE_RVC` 0x0001, float ABI 0x0006, RVE
0x0008, TSO 0x0010):

| Bit | Name | Meaning |
|---|---|---|
| 0x0100 | `EF_CAPSTONE_CRVC` | Decoded in the compressed-capability mode, stage 2 option B (CSRUNV options bit 2). Allocated 2026-11-21 for the prototype; see `plans/2026-11-21-capability-rvc-stage2-prototype.md`. |

An object with the flag contains none of the five 16-bit forms that the mode reinterprets: C.FLD, C.FSD, C.FLDSP,
C.FSDSP and C.LUI x0. Two rules follow:
- the linker refuses to combine it with an object that has `EF_CAPSTONE_RVC` but not `EF_CAPSTONE_CRVC`;
- an adapter that cannot admit a context in the mode refuses an image with the flag.
