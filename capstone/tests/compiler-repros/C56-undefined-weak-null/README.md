# C-56: the address of an undefined weak symbol is not null in a domain

Registry entry: `docs/ref/ISSUES.md` C-56. No sibling issue shares this folder.

`src/weak.c` and `src/weak2.c` form one program. The checks run in this order:
- a control (a defined global reads 5);
- a weak object and a weak function that `weak2.c` defines (non-null, read 9, return 2);
- an undefined weak function's address;
- an undefined weak object's address, stored to memory and reloaded;
- the same object's address tested directly.

The undefined-object checks come last, because on a broken toolchain forming that address faults.

    SDK=<application SDK> KIT=<QEMU kit> ./run.sh    # exit 0 ABSENT, 1 PRESENT, 2 no result

## The two halves

Under the application SDK's ABI (`-capstone-gp-free -capstone-image-gp`) a global's address is a pc-relative
`auipc`/`addi` pair, then `scc rd, gp, addr` and, for a sized object, a SHRINK to the object.

1. **Linker.** lld resolved an undefined weak symbol to link address 0, but the image runs at a load base
   without relocation, so the pair formed the load delta. `if (hook)` was true, and a call went to the image
   base.
   - Fix: on Capstone, lld resolves such a pair to the absolute value and rewrites the `auipc` as a `lui`
     (`lld/test/ELF/capstone-undefined-weak-pcrel.s`).
2. **Compiler.** The SHRINK of an undefined object's base runs on a cursor outside gp's bounds and faults
   before any test: cause 29 with the old linker, 24 on QEMU with the new one, where `scc` of an
   out-of-window cursor clears the tag.
   - Fix: the base of an extern-weak global is formed only when its address is not 0 (PseudoCapWeakGlobal,
     `llvm/test/CodeGen/Capstone/c56-extern-weak-null.ll`). An undefined symbol yields the integer 0, an
     untagged null.

## Readings (QEMU kit-unq, one boot, 2026-11-20)

| compiler + lld | -O0 and -O2 |
|---|---|
| new + new | all six checks ok, failures=0 |
| new compiler + dev's lld | undefined function FAIL (address = the image base); then a fault at the object (cause 26) |
| dev's compiler + new lld | undefined function ok; a fault at the object's SHRINK (cause 24) |
| dev's compiler + dev's lld | undefined function FAIL; a fault at the object (cause 29) |

The defined-weak checks pass in every arm. lld binaries: new `b9f671e793ab869d`, dev's `bdccf3e574f669bd`
(sha256 prefixes).
