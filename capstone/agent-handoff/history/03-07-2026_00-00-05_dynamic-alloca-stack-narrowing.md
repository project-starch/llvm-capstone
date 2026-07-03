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
- **Runtime OOB:** the narrowing reuses the `SHRINK` primitive already proven at
  runtime by the `stack_oob`/`global_oob` authority probes (an OOB access past a
  SHRINK-narrowed object faults). A dedicated `-O0` dynamic-alloca runtime probe
  is **blocked by a pre-existing, orthogonal limitation** (below), so it is not
  added to the suite; the lit codegen test + the runtime-proven primitive cover
  the mechanism.

## Pre-existing limitation found (not caused by this change)

`lowerDynamicAllocaSizeToXLen` rejects dynamic-alloca **size expressions that
come from memory** (a `LOAD`), emitting
`fatal error: Unsupported dynamic alloca size expression in Capstone PureCap`.
At `-O0` the size is always spilled/reloaded, so *any* `-O0` dynamic alloca with a
non-constant size fails to compile — independent of `-capstone-shrink-stack` (it
reproduces with the flag off). Register-sized dynamic allocas (e.g. size from a
function parameter at `-O1`) compile fine. This is why the suites (which do not
use memory-sized dynamic allocas at `-O0`) were unaffected, and why a synthetic
`-O0` runtime probe cannot be built. **Follow-up:** teach
`lowerDynamicAllocaSizeToXLen` to handle a `LOAD`/other size by materializing it
into an XLen register (spill/reload or a copy), which would also unblock general
`-O0` dynamic allocas — a small, separate enhancement.

## Task #77 status

- Increment 1 (fixed objects incl. interior/load-store base): done (earlier).
- Increment 2 (dynamic alloca): **this change**.
- Varargs save area: already narrowed via the fixed-object path (confirmed).
- Remaining before default-on: the `-O0` memory-sized-alloca compile limitation
  (above) is worth fixing for robustness; then a full clean default-on matrix.
  Default stays `-capstone-shrink-stack=false`.
