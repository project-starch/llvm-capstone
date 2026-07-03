# Dynamic-alloca capability narrowing (C1 stack slice, increment 2)

*Status: 2026-07-03. Extends stack narrowing (task #77) to runtime-sized
(dynamic) allocas. Prior increment narrowed fixed frame objects
(`narrowToFrameObjectBounds`, via FrameIndex); dynamic allocas never reach a
FrameIndex, so they were left with whole-stack bounds. Now the pointer returned
by a dynamic alloca is narrowed to its allocated region, while the real stack
pointer keeps broad bounds. Gated on the same `-capstone-shrink-stack` flag
(default off).*

## Gap (measured)

Compiling a varargs + dynamic-alloca + fixed-object test with
`-capstone-shrink-stack=true` showed: fixed object → `shrink`; varargs save area
→ `shrink` (already covered via the fixed-object path); **dynamic alloca → 0
`shrink`**. The dynamic alloca lowers through `ISD::DYNAMIC_STACKALLOC` to
`NewSP = sp + (-alignedSize)`, a register capability that never becomes a
FrameIndex, so `narrowToFrameObjectBounds` (keyed on FrameIndex) never sees it.

## Change

`CapstoneISelLowering.cpp` `lowerDYNAMIC_STACKALLOC` (i128 path): the value copied
to X2 (the real `sp`) stays the un-narrowed `NewSP` — it must keep broad bounds
for later allocations — but the pointer *returned to the program* is narrowed to
`[cursor, cursor+size)`:
`cursor = get_cursor(NewSP)`; `end = cursor + SizeXLen`;
`shrink(NewSP, cursor, end)`. Implemented with `int_capstone_cap_get_cursor` +
`int_capstone_cap_shrink` DAG nodes (selected to `lcc`/`add`/`shrink`). The
narrowing size is the *aligned* allocation size (the region actually reserved),
so it is not over-tight. The `-capstone-shrink-stack` flag (defined in
`CapstoneISelDAGToDAG.cpp`) was made non-`static` and `extern`-declared in
`CapstoneISelLowering.cpp` so both sites share one flag.

Resulting `-O1` codegen (flag on):
```
cincoffset a1, sp, a1   ; a1 = alloca base (broad)
lcc  a2, a1, 2          ; cursor(a1)
add  a0, a2, a0         ; end = cursor + alignedSize
shrink a2, a3, a0       ; a2 = narrowed returned pointer
movc sp, a1             ; sp = a1  (UN-narrowed, broad) -- stack discipline intact
... use a2 (narrowed) for the alloca'd storage
```

## Validation

- **lit:** new `llvm/test/CodeGen/Capstone/cap-shrink-dynalloca.ll` — flag on emits
  `lcc`/`shrink` and `movc sp, <un-narrowed base>`; flag off emits no `shrink`
  (two-alloca case checks two independent shrinks). Full Capstone CodeGen lit
  **36/36**.
- **Inert by default:** with the flag off, `narrowAllocaResult` returns the pointer
  unchanged — byte-identical to prior behavior; no default-build regression risk.
  Authority suite (default flags) re-run green after the compiler rebuild.
- **Runtime OOB (dedicated probes):** authority `stack_dynalloca_inbounds`
  (`ok`, retval `1560281128`) and `stack_dynalloca_oob` (`bounds-fault`,
  `Cap mem access OOB`) — a runtime-sized `alloca(48)` where an in-bounds access
  returns cleanly and `buf[200]` traps. These build at `-O0` only after the size
  fix below. Full authority suite green with them added.

## Pre-existing limitation found + FIXED (memory-sourced alloca size)

`lowerDynamicAllocaSizeToXLen` rejected dynamic-alloca **size expressions sourced
from memory**, emitting
`fatal error: Unsupported dynamic alloca size expression in Capstone PureCap`.
At `-O0` a non-constant size is always spilled/reloaded and materializes as an
`i32`→`i128` **extending load**; the helper's structural recursion had no case for
a load leaf → `default` → `SDValue()` → fatal. So *any* `-O0` dynamic alloca with
a non-constant size failed to compile, independent of `-capstone-shrink-stack`
(reproduced flag-off). Register-sized allocas (size from a parameter at `-O1`)
were fine — which is why the suites, not using memory-sized `-O0` dynamic allocas,
were unaffected.

**Fix:** the `default` case now materializes any *scalar-integer* size leaf into
an XLen register via `getZExtOrTrunc` (truncate an i128 carrier's low bits;
zero-extend a narrower value — sizes are non-negative). This is consistent with
the arithmetic cases, which already rebuild i128 size nodes in XLen: the whole
size cone is a scalar byte count, never a dereferenceable capability, so
extracting the low XLen bits is correct. This is a **general** fix (not gated) —
it strictly expands what compiles (the recognized-opcode paths are unchanged; only
the previously-fatal `default` now returns a value). It unblocks general `-O0`
dynamic allocas *and* enabled the runtime probes above. lit
`cap-shrink-dynalloca.ll` gains a `dynalloca_memsize` case (a loaded size) that
previously failed to compile.

## Task #77 status

- Increment 1 (fixed objects incl. interior/load-store base): done (earlier).
- Increment 2 (dynamic alloca): **this change**.
- Varargs save area: already narrowed via the fixed-object path (confirmed).
- Remaining before default-on: the `-O0` memory-sized-alloca compile limitation
  (above) is worth fixing for robustness; then a full clean default-on matrix.
  Default stays `-capstone-shrink-stack=false`.
