# C-14: the reaching-def rule calls a live capability a scalar, at -O1 and above

**Status: root-caused to one rule in one pass, reproducible in seconds with `llc`.
Not fixed.**

## What it looks like

mruby halts with `cause = 24` at a capability store whose base register is
untagged. Same signature in two different ABIs:

    default ABI + LTO     mrb_method_search_vm +0x3d0 (TB start; store at +0x468)
    gp-captable + LTO     mrb_method_search_vm +0x464
    both: rs1 = x10, imm = 16, and both stop at exactly TICK 256

The store is mruby's method-cache fill, `mc->m = m` on `&mrb->cache[h]`.

## Why it looked like an LTO bug and is not

The working mruby build compiles at **-O0**, so it never runs the optimising
codegen. `-flto` defers code generation to the linker, which runs it at its own
level. So LTO did not introduce the bug, it merely stopped hiding it.

Measured on the extracted function, no LTO involved:

    llc -O0   mv=0  movc=6      correct
    llc -O1   mv=4  movc=1      broken
    llc -O2   mv=4  movc=1
    llc -O3   mv=4  movc=1

    llc -O1 -regalloc=fast      mv=0  movc=3    correct

`-O0` uses the fast allocator, which is why the level appears to be the variable
when the allocator is.

## The rule that does it

`CapstonePostRAExpandPseudoInsts.cpp` rewrites `movc` to `mv` (ADDI) wherever it
can prove the source is a plain scalar. `mv` drops the tag, so a wrong proof is a
miscompile. Three rules can vote; the pass prints all three:

    C14: movc dst=$x9 src=$x10 Decided=1 ScalarUse=0 scalarAddr=0 rdaScalar=1

`$x9` is s1 and `$x10` is a0. The next-use rule says nothing, the whole-function
materialisation rule says nothing, and **`isScalarByReachingDef` claims scalar**.
The value is `mrb`, a live capability.

The obvious explanation is wrong and was checked: the rule does NOT vacuously
accept a live-in. It has `if (Defs.empty()) return false; // live-in, or not
visible: prove nothing`.

The copy sits at the head of a loop body (`in Loop: Header=BB0_4 Depth=1`), so the
reaching defs come in over a back-edge as well as the entry path. Which def chain
the query accepted is the open question, and it is the next thing to look at.

## Reproduce

    llvm-extract -func=mrb_method_search_vm <mruby LTO obj>/mrb_class.o -o msv.bc
    llc -O1 -mtriple=capstone64-unknown-elf -mattr=+m,+a msv.bc -o - | grep -E '^\s+(mv|movc)\s'
    llc -O1 -debug-only=capstone-postra-expand ... 2>&1 | grep C14

Any mruby build with `-flto` produces a suitable `mrb_class.o`; the ABI does not
matter, the bug reproduces under both.

## What did NOT reproduce it

A small standalone case -- pointer argument, loop, call forcing it into a
callee-saved register -- gives `mv=0` at both -O0 and -O1. The trigger is
narrower than that shape, so do not assume the naive reduction is equivalent.

## Consequences

* mruby runs fully (S1-S5) under gp-captable + LTO with `-DMRB_NO_METHOD_CACHE`,
  which is the workaround in use.
* `--plugin-opt=O0` does NOT help: tried, still `mv=4`. lld's LTO does not route
  that to the codegen level.
* This is the same C-14 the `capstone-scalar-copy-live-src` option was disabled
  for. That option was a blanket flip of `copyPhysReg`; the current rules are the
  narrower replacement, and one of them is now doing what the blanket flip did.
* Anything built above -O0 is exposed, not just LTO builds. That matters for the
  standing wish to build at -Os.
