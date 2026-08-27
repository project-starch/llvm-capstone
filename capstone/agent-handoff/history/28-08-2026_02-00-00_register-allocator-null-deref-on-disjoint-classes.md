# The register allocator segfaults where two register classes are disjoint

Found 2026-08-28 while bringing mruby up as a domain. The first program in this
tree large enough to reach it.

## The crash

```
Running pass 'Greedy Register Allocator' on function '@mrb_vm_exec'
 #7 llvm::TargetRegisterClass::hasSubClassEq(...) TargetRegisterInfo.h:133
 #8 llvm::TargetRegisterClass::hasSubClass(...)   TargetRegisterInfo.h:127
 #9 llvm::SplitEditor::rematWillIncreaseRestriction(...) SplitKit.cpp:619
```

`mrb_vm_exec` is mruby's whole bytecode interpreter in one function, so it is where
register pressure and live-range splitting are at their worst here.

## The defect

`llvm/lib/CodeGen/SplitKit.cpp`, in `rematWillIncreaseRestriction`. The function
computes two register-class constraints and null-checks only the first:

```c
const TargetRegisterClass *DefConstrainRC =
    DefMI->getRegClassConstraint(DefOperandIdx, &TII, &TRI);
if (!DefConstrainRC)
  return false;                                   /* checked */
...
const TargetRegisterClass *UseConstrainRC =
    UseMI->getRegClassConstraintEffectForVReg(DefReg, SuperRC, &TII, &TRI, true);
return UseConstrainRC->hasSubClass(DefConstrainRC);   /* NOT checked */
```

`getRegClassConstraintEffectForVReg` ends in `getRegClassConstraintEffect`, whose
last line is

```c
CurRC = TRI->getCommonSubClass(CurRC, OpRC);
return CurRC;
```

and `getCommonSubClass` returns **null** when the two classes have no common
subclass. Nothing forbids that; it is the ordinary answer for classes that do not
overlap.

## Why this target reaches it and others do not

On most targets the register classes in play at a split are all integer or all
float, and a common subclass exists. Here a **capability class can meet an integer
class**, and those are genuinely disjoint -- that is the point of the ISA. So the
null that the code already anticipates for `DefConstrainRC` also occurs for
`UseConstrainRC`, and the dereference is a segfault inside the register allocator
rather than a diagnostic.

This is upstream code, not ours: the function came in with
`8476a5d48030 SplitKit: Fix rematerialization undoing subclass based split (#122110)`.
The fix is the same three lines the function already applies to the other constraint.

## Evidence

Clang saved its own reproducer: `/tmp/capstone/crash/mruby_all-267859.c`, 98,776
lines, with the `.sh` beside it. Not reduced; a lit test for this would need
`llvm-reduce` against a build with the fix reverted, which has not been done. The
crash log with the stack above is the evidence, and the fix is a null check
symmetric with one three lines away, so the risk of it being wrong is low and the
risk of leaving the crash is a compiler that cannot build large capability programs.

## What it blocks, and what it says

mruby cannot be compiled at all without this, so the whole nested-allocator
blind-spot corpus depends on it. More generally: any sufficiently large function on
this target can hit it, and the trigger is not exotic -- it is the ordinary
consequence of having a register class that is not an integer class. WAMR's
interpreter did not reach it; mruby's, which is bigger, did.
