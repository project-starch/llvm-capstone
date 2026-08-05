# Capstone i128 (capability) lowering fixes

Four independent fixes to the Capstone LLVM backend's i128 / capability lowering
(all in `CapstoneISelLowering.cpp`), each surfaced by running real software (SQLite,
Lua 5.4.7) on the capability target, and each guarded by a lit regression test.

On `capstone64`, `i128` is the 128-bit **tagged capability** register type while
64-bit values are plain scalars. Every bug below is the same family: lowering that
mistook a scalar `i128` for a capability (or vice-versa) and emitted a capability
instruction on an untagged value, which faults at runtime.

| fix | symptom | test |
|-----|---------|------|
| i128 shift-by-≥XLen | wrong result for an `i128` shift ≥ 64 (surfaced by Lua `ltable`) | `cap-i128-or-undef.ll` |
| i128 capability `select` | a capability dropped by a `select` guarded by `x != C` | `cap-i128-select-capability.ll` |
| variable-offset pointer arithmetic | `p - (n+1)` lowered via ptrtoint / scalar / inttoptr → **untagged** pointer | `cap-i128-ptr-arith-variable.ll` |
| pointer-difference + constant | `p - (q+1)` → `CIncOffset(scalar, -k)` — a cincoffset on the untagged scalar difference; blocked every Lua C-API/base call through `lua_gettop` | `cap-i128-ptr-diff-const.ll` |

Each commit message carries the full root-cause and fix detail. Verified: Capstone
lit **47/47**. This branch is a clean delta on `capstone-bootstrap`; the application
work that surfaced these (real Lua on the gp-captable ABI) lives on the sibling
`capstone-bootstrap-xlang` branch, not here.
