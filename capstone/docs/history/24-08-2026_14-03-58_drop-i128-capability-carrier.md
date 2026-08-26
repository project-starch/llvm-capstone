# i128 stops being the capability carrier

Completes the `captype` work. A capability is `MVT::c128` in its own register
class `GPCR`; `i128` is an ordinary illegal integer type on RV64, legalized
generically exactly as on any other target. Net **-1300 lines** in the backend.

## Why this is a deletion and not a feature

Almost everything removed here existed to answer one question at a point where
the answer was not recoverable: *is this register holding an integer or a
capability?* One register class held both, so every consumer had to guess.

The guessing machinery, all now gone:

| Removed | What it was guessing |
|---|---|
| `fixupDestructiveCopies` + 3 dataflow proofs (380 lines) | whether a `movc`'s source was a scalar |
| `capstone-scalar-copy-live-src`, `capstone-int-zero-for-zero-copy` | same question, as flags |
| `PseudoSCALAR_COPY_I128`, `PseudoTRUNC_CAP` | integer-vs-capability move |
| `isZExtFree(i64, i128)` | whether widening touched metadata |
| 11 `lowerScalarI128*` functions (535 lines) | which i128 ops were addresses |

## `fixupDestructiveCopies` was not merely redundant -- it had become WRONG

Its three strategies all searched for a `MOVC` whose source is provably a scalar.
Post-RA it asks `J->readsRegister(Src, TRI)`, which is **aliasing-aware**, and
`C10` and `X10` are the same hardware register. So an `addi x11, x10, 4` reads
`X10`, overlaps `C10`, and the pass concluded a genuine capability copy was
scalar.

Positive control, on the split build before the pass was removed:

```
$c11 = MOVC $c10   ->   $c11 = ADDI $c10, 0
```

A dropped tag, and an `ADDI` with C-register operands that the machine verifier
rejects. It scanned 0/59 lit files without firing, which is exactly why the
MIR control mattered: **the corpus was silent because it lacked the shape, not
because the pass was harmless.** After removal the same control leaves the
`MOVC` alone.

## A real crash, found by a real-IR test

`i128-sext-inreg-int-index.ll` is captured domain output. It aborted with
`"Copying to an illegal type!"` in `getCopyToParts`, from `visitInlineAsm`.

Cause: `GPRRegisterClass` still listed `[i128, XLenVT, ...]` with a 128-bit
`XLenRI`. Generic code takes a class's FIRST legal type as representative, so
every `"r"` inline-asm operand was i128 -- illegal since this change. The class
comment said `drop-i128 will have to answer this properly`; this is that answer.
`GPR` is now `[XLenVT, XLenFVT, i32, i16]` at 64 bits, and the spill/reload arms
lost their `Size == 128 -> STC/LDC` branches with it.

## `inttoptr` needed a lowering, and got a better one than before

With `i128` illegal the generic expansion round-tripped through a stack slot --
`shrink`, two `sd`, an `ldc` -- to assemble a 128-bit value whose metadata half
is not data. Now `BITCAST` with an `i128` **operand** is `Custom` (deliberately
an action on an illegal type: that is how `ExpandIntegerOperand`'s
`CustomLowerNode` finds us) and lowers to `INSERT_SUBREG` on `sub_cap_addr`.

Neither capability instruction can do this job, and both were checked against the
model rather than assumed:
* `helper_csscc` asserts a **tagged** `rs1`, so `scc cnull, addr` is not legal;
* `helper_cscincoffset` raises `UNEXP_OP_TYPE` on an untagged `rs1`.

Writing the address half, which clears the tag, is the correct and only answer --
and it is what `inttoptr` means.

## Front end: `intptr_t` is address-sized

`CodeGenModule` built `IntPtrTy` (and, through a union, `SizeTy` and `PtrDiffTy`)
from `getMaxPointerWidth()` = 128. Ordinary C pointer subtraction therefore ran
at 128 bits and, once `i128` was illegal, came out as a **`__divti3` call** --
measured on `p - q` over a 48-byte struct. It now uses the DataLayout **index**
size. No-op for every target where the two agree.

`ptrtoint` is a subregister read (`EXTRACT_SUBREG`), so it costs nothing:

| | before | after |
|---|---|---|
| `ptrtoint` alone | `mv a0, a0` | -- |
| `p - q` | `mv; mv; sub` | `sub a0, a0, a1` |
| `p - q` over 48 bytes | `call __divti3` | inline multiply-by-inverse |
| `(uintptr_t)p & ~31` | `mv; andi` | `andi a0, a0, -32` |

## Unforgeability moved from a refusal to the register file

`cap-constants-invalid.ll` refused to compile at four sites. Three were about
`p + 2^64`, and they are gone: integer arithmetic cannot reach the metadata half
any more, so the value truncates -- which is what `inttoptr` specifies, and what
the same file's own `gep-truncates.ll` case already documented as the defined
answer. The file used to give one expression two answers depending on spelling.

Materializing a capability from an arbitrary 128-bit **constant** is still
refused, because no instruction can do it; that case is also the file's proof
that the refusal machinery still fires.

## Tests

59 -> 56 files, all passing. Three deleted outright (their machinery is gone),
one split so its three capability cases survive as `cap-stack-addressing.ll`,
five rewritten to the i64 shapes the front end now emits. Each rewrite gained a
**control** -- a logical shift next to the arithmetic ones, a no-offset pointer
difference next to the offset ones, single-signedness multiplies next to the
mixed one -- because several of these assert `CHECK-NOT` and a `CHECK-NOT` that
cannot fire is not a test. `cap-i128-and-capability-mask.ll`'s
`--implicit-check-not` gate was negative-tested against an unchecked pattern.

## Inline asm was the half only the QEMU tier could see

`"r"` returned GPR unconditionally. That needed no thought while GPR held both
kinds; once it was integer-only, a capability operand reached
`getCopyFromParts` with nothing to reconcile -- **"Unknown mismatch in
getCopyFromParts!"**, which is how CoreMark failed to build. lit was green.

`"r"` and `"cr"` now return `GPCRNoC0` for a capability VT. That is not a new
convention: `CAPSTONE_DELIN` passes a pointer through `"+r"` and then runs
`delin`, which faults on an untagged register, so `"r"` has always had to mean a
capability register when the value is one.

The register then has to PRINT as its X name -- `.insn` and hand-written asm know
only those, and C and X encode identically -- so `CapstoneAsmPrinter` maps GPCR
through `sub_cap_addr` the same way the InstPrinter does. Pinned in the new
`cap-inline-asm-r.ll`, with an integer control so "capabilities go to GPCR"
cannot be satisfied by a backend that sends everything there.

## A stale i128 in the clang builtins

`__builtin_capstone_cap_shrink`, `_scc` and `_init` zero-extended their
integer arguments to i128 to match an intrinsic signature that had already been
changed to `llvm_i64_ty`. Any translation unit using them aborted in
`CallInst::init` -- "Calling a function with a bad signature!". `cap_heap.c` does,
so every rv8 and beebs benchmark did. Bounds and cursors are addresses; they are
XLen.

## An unaligned capability access goes via the stack

`expandUnalignedLoad`/`expandUnalignedStore` split a fat pointer by
reinterpreting it as a same-size integer. With i128 illegal, the store crashed in
operation legalization -- **"Unexpected illegal type!"** -- on
`tagged_cap_memcpy_misaligned.c` at -O0. Only the store: the load already tested
`isTypeLegal(intVT)` and fell through.

Both now gate on `!VT.isInteger()` and share the generic stack route: copy the
value through an aligned stack slot with integer loads and stores. That is the
only path built entirely from legal types, and it produces exactly what this
oracle requires -- the capability that comes back carries no tag, because its
bytes were written by integer stores.

The stack route then hit a second one, `getMemBasePlusOffset`'s
`Offset.getValueType().isInteger()`: it materialised its pointer increments in
the POINTER type, which is c128. Offsetting a capability by a capability is not a
thing. They go through the `TypeSize` form, which builds the offset in the index
type.

## Three capability instructions still declared integer operands

`DROP`, `CAPEXIT` and `CCSRRW` build `c128` nodes at selection but their operand
classes said `GPR`. The allocator therefore had to copy the capability into an
integer register first -- an ADDI on the address half, which clears the tag. So
`__builtin_capstone_cap_drop` was untagging the very handle it exists to consume.

`linear_drop_use_fault` found it: carve a linear statement handle, drop it,
dereference the result. It must fault at the DEREFERENCE, and it was faulting at
the DROP. **Both are cause 24.** A check on the cause code alone would have
called that a pass -- the probe compares the diagnostic string too, and that is
the only reason this was visible at all. It is the same discipline as
`linear_no_drop_ok`, which runs the identical sequence without the drop precisely
because cause 24 under-determines.

Deterministic at -O0 and -O2, and the suite had passed at 02:10 the same day, so
it bisected to this work rather than to anything older. All 21 probes (7 x 3
optimisation levels) pass now, each with its expected reason.

## The tag scanner was checking the wrong operand

`scan-tag-stripped-caps.py` had `'shrink':[1]`. SHRINK is
`Constraints = "$rd = $cap_in"` and prints as `shrink $rd, $rs1, $rs2`, so the
capability is the **first** printed operand and `$rs1`/`$rs2` are plain integer
bounds. Index 1 is a register that can never legitimately be tagged.

It therefore did both halves of being broken at once: it reported eight `mv` into
a bounds register as tag strips, and it never checked the operand that matters.
Corrected to `[0]`, with three controls -- a strip into the capability operand
fires, a `mv` into a bounds operand does not, and the previously-reported
`mv rd, rs` before `delin` still fires. 391 binaries scan clean.

It also lost a false-positive class on calls: an argument set up before a `cjalr`
and a return value used after it are not the same value, and nothing pending
survives a call.

**The uncomfortable part is the control.** This detector already had one, recorded
in its own docstring: eight hits on a known-bad CoreMark domain. Those eight were
the false positives. A control harvested from an observed failure can be
satisfied by the wrong mechanism, and this one was -- for weeks. A control built
from the instruction's operand list, which is what the three above are, could not
have been.

## Known gap, not a defect

A mixed-signedness widening multiply is four instructions where upstream riscv64
picks one `mulhsu`. The value is the same -- verified against riscv64 on the same
IR -- so this is a missed selection, not a wrong answer.
