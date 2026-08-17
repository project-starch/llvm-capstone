# Capstone i128 (capability) lowering fixes

Independent fixes to the Capstone LLVM backend's i128 / capability handling, each
surfaced by compiling or running real software on the capability target, and each
guarded by a lit regression test.

On `capstone64`, `i128` is the 128-bit **tagged capability** register type while
64-bit values are plain scalars. Most bugs below are the same family: code that
mistook a scalar `i128` for a capability (or vice-versa) and emitted a capability
instruction on an untagged value, which faults at runtime.

## Round 1: lowering (SQLite, Lua 5.4.7)

Four fixes, all in `CapstoneISelLowering.cpp`.

| fix | symptom | test |
|-----|---------|------|
| i128 shift-by-≥XLen | wrong result for an `i128` shift ≥ 64 (surfaced by Lua `ltable`) | `cap-i128-or-undef.ll` |
| i128 capability `select` | a capability dropped by a `select` guarded by `x != C` | `cap-i128-select-capability.ll` |
| variable-offset pointer arithmetic | `p - (n+1)` lowered via ptrtoint / scalar / inttoptr → **untagged** pointer | `cap-i128-ptr-arith-variable.ll` |
| pointer-difference + constant | `p - (q+1)` → `CIncOffset(scalar, -k)`, a cincoffset on the untagged scalar difference; blocked every Lua C-API/base call through `lua_gettop` | `cap-i128-ptr-diff-const.ll` |

## Round 2: constants, bitwise arithmetic, and initializer size (MicroPython)

Surfaced by a census that compiles the MicroPython core, 133 files, for the
capability target. These reach wider than round 1: two are outside
`CapstoneISelLowering.cpp` entirely, and one is outside the target.

| fix | symptom | test |
|-----|---------|------|
| absolute constant in a capability slot | an `inttoptr` ConstantExpr in a 16-byte slot fell past the relocatable carve-out into `emitValue(ME, 16)` and tripped `emitIntValue`'s `1 <= Size <= 8`. Zero-extended, not sign-extended, so `[2^63, 2^64)` keeps a zero high word | `const-capability-initializer-absolute.ll` |
| constants wider than 64 bits in immediate predicates | `getSExtValue()` called before checking there was one. The same call appears in 45 generated TableGen predicates, so the width guard is emitted by `CodeGenDAGPatterns` rather than written 45 times. In a release compiler this class was a silent **miscompile**, not a crash | `cap-shrink-logic-imm-wide-const.ll` |
| zero-extended capability constant rejected | `MP_OBJ_NEW_SMALL_INT(-1)` is `inttoptr (i128 0xFFFFFFFFFFFFFFFF)`; only the sign-extended spelling was accepted, though `inttoptr i64 -1` already named the same register. Bits above the low 64 are still refused | `cap-constants-zero-extended.ll`, `cap-constants-invalid.ll` |
| static cap table pointing at an extern | `collectStaticCapReducedObject` guarded the holder against having no initializer but not the target it points at, and a table entry naming an extern object is a declaration | `static-cap-gct-extern-target.ll` |
| bitwise arithmetic on a capability | `gc_init` aligning a pointer down, `pairheap` stealing a low bit, `bound_meth_unary_op` hashing two pointers: ordinary C through `uintptr_t`, arriving as i128 AND/XOR that nothing could select. Lowered by reading the address with the same `lcc rd, rs, 2` a pointer difference uses, operating at XLen, and returning an **untagged** value | `cap-i128-and-capability-mask.ll` |
| large synthetic capability initializer | hundreds of pointer leaves in one basic block put every holder and target capability in one DAG. Capstone shares the GPR class between scalars and capabilities, so the allocator's spill is an `sd` that drops the tag and the `ldc` reload yields an untagged base. Bounded to 32 stores a block | `static-cap-global-init-large.ll` |

`CodeGenDAGPatterns` is shared LLVM code. For every target whose integers stop at
64 bits the emitted guard is always true; where it does fire it turns a wrong
immediate into a non-match.

Each commit message carries the full root-cause and fix detail, including two
approaches that were tried and reverted (declaring i128->i64 truncation free, and
rewriting the align-down in the source), which are recorded because they cost the
time.

Verified: Capstone lit **54/54**. This branch is a clean delta on
`capstone-bootstrap` and touches `llvm/` only. The application work that surfaced
these lives on sibling branches, not here: real Lua on the gp-captable ABI on
`capstone-bootstrap-xlang`, and the MicroPython port and its census on
`nested-allocators`.
