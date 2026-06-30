# Plan: tag capability pointers in local aggregate initializers (SQLite gap 2)

*Status: SCOPED, not implemented. Pairs with the global-side fix
(`CapstoneCapGlobalInit`, gap 1). This is the clang-frontend half that unblocks
SQLite's terminal fault. Implement only after the gap-1 benchmark gate is green
and gap 1 is committed (shared LLVM build dir — no compiler rebuild before then).*

## The gap
SQLite aborts in `sqlite3RegisterBuiltinFunctions()` with
`helper_cscincoffset: Assertion 'rs1_v->tag' failed` inside a `memcpy`. Cause: a
**local** aggregate initialized from a large constant is lowered by clang to a
`memcpy` from a private `unnamed_addr constant` template. On Capstone the
template's capability-pointer fields are untagged (static image), and a bytewise
`memcpy` cannot carry the out-of-band tag, so the copied-in pointers fault on use.

This is distinct from gap 1: gap 1 (`CapstoneCapGlobalInit`) materializes
**named globals**; this is the **local var init** site.

## Exact location
`clang/lib/CodeGen/CGDecl.cpp`, `CodeGenFunction::emitStoresForConstant` (~1189):
- scalar/pointer constant → single store (a lone cap pointer is fine — isel tags it);
- mostly-zero / pattern → memset (+ a few stores);
- `shouldSplitConstantStore` (line 1028: only `-O>0` **and** size ≤ 64) → recurse
  per struct/array element (lines 1252–1280);
- **else → `CreateMemCpy` from `createUnnamedGlobalForMemcpyFrom` (line 1283)** —
  the faulting path. SQLite's builtin table is > 64 bytes, so it lands here even
  at `-O2`; at `-O0` every aggregate lands here.

## Minimal fix
The per-element recursion already does the right thing (each cap-pointer leaf
becomes a single store → `cincoffset gp`/`delin`/`stc`, i.e. tagged). We just need
to take it whenever the constant contains a capability, regardless of size/opt:

1. Add a recursive predicate (mirrors `CapstoneCapGlobalInit`'s `isCapPtr` +
   `needsMaterialization`):
   ```cpp
   // True if the constant holds an addrspace(200) pointer to a global/function
   // at any depth (i.e. a capability that a bytewise memcpy would leave untagged).
   static bool constantContainsCapability(llvm::Constant *C);
   ```
2. Widen the split trigger at line 1252:
   ```cpp
   if (shouldSplitConstantStore(CGM, ConstantSize) ||
       constantContainsCapability(constant)) { ... }
   ```
   Because the element loop calls `emitStoresForConstant` recursively, the
   predicate is re-evaluated per level: cap-containing subtrees keep splitting
   down to tagged single stores, while non-cap subtrees still fall to the
   efficient `memcpy`. So the change is localized and only affects cap-bearing
   aggregates (self-gating to Capstone — no other target forms addrspace(200)
   pointer-to-global constants).

Edge to handle: the struct/array recursion blocks (1253/1268) guard on
`STy == Loc.getElementType()`; if a cap-containing constant fails that guard it
would still fall through to memcpy. The common SQLite case matches the guard, but
the implementation should ensure cap-containing constants never reach the memcpy
(e.g. recurse unconditionally for the cap case, or assert).

## Validation
- **Unit lit:** a local `struct { fn_ptr; str_ptr; } t = {...};` (and an array of
  such) at `-O0` and `-O2` → FileCheck that the cap fields are emitted as `stc`
  stores, not a `memcpy` from a private global.
- **Reproducer:** a domain that does the local-aggregate-copy shape (extend the
  `nested-cap-global` family with a *local* copy) → runs without fault in QEMU.
- **SQLite:** `capstone/benchmarks/sqlite/run-sqlite-memory.sh` should advance past
  `sqlite3RegisterBuiltinFunctions()` (and, ideally, print the three rows +
  `__CAPSTONE_SQLITE_MEMORY_PASSED__`). Some of the agent's source-level
  adaptations (e.g. "initialize memsys5's methods table at runtime", "move
  built-in pUserData out of static initializers") may become unnecessary once
  gaps 1+2 are in — revisit them.
- **Regression:** the same CoreMark/BEEBS-82/RV8-7 gate as gap 1 (this touches
  generic local-init codegen, so re-run before landing).

## Relationship to gap 1
Gaps 1 and 2 are the two halves of "capability pointers in aggregate initializers
load untagged": gap 1 = named globals (backend pass), gap 2 = local var inits
(frontend). Together they should remove the need for SQLite-specific static-init
rewrites. Both are general C1 capability-globals infrastructure, not
SQLite-specific.
