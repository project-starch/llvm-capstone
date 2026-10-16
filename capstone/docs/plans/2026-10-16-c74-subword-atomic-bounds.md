# C-74 — sub-word atomics overrun a capability bound: corrected scope, and the three fixes costed

**Status: a decision is wanted, nothing is implemented.** The choice between the options below
trades lock-freedom against object layout and overrun precision, which is the project lead's call,
not a lane's. Everything here is static analysis plus compiler output; there is no runtime run, and
the trap itself remains the observation recorded in `ref/ISSUES.md` under C-74.

Measured with the frozen dev build `clang version 22.0.0git (… 330014ead628392cc1b9ed32625dddae9a9915e3)`,
identity printed in the same command as every reading. Instrument: `clang -S -emit-llvm` (which
carries the `align` that `AtomicExpandPass` reads) then `llc -mtriple=capstone64 -mattr=+a`, with
a positive control on every run — `lr.w` must appear in the file, and the arms must differ from
each other. Code citations are resolved on `origin/dev`, not in a working tree.

## The registry's scope is too narrow — correction, with the matched pair that shows it

C-74's entry says "any object **smaller than 4 bytes** whose capability is bounded to it alone …
cannot be the target of an 8- or 16-bit atomic", and a first pass of this investigation wrote, in
the same spirit, that a byte inside a larger object is safe.

Both are wrong. A byte inside a larger object faults whenever it lies in that object's final
*partial* word. Two arms whose instruction sequences are identical but for one immediate:

    static unsigned char arr5[5] __attribute__((aligned(4)));   atomic on &arr5[4]
      li a1, 5  ; add a1, a2, a1 ; shrink a0, a2, a1     bound  = [b, b+5)
      cincoffsetimm a0, a0, 4                            cursor = b+4
      lr.w.aqrl a3, (a0)                                 reads  [b+4, b+8)  -- 3 bytes PAST the bound

    static unsigned char arr8[8] __attribute__((aligned(4)));   atomic on &arr8[4]
      li a1, 8  ; add a1, a2, a1 ; shrink a0, a2, a1     bound  = [b, b+8)
      cincoffsetimm a0, a0, 4                            cursor = b+4
      lr.w.aqrl a3, (a0)                                 reads  [b+4, b+8)  -- exactly fits

**The condition, stated properly.** A sub-word atomic on byte `k` of an object bounded `[b, b+s)`
overruns iff the 4-byte-aligned word containing `b+k` is not contained in `[b, b+s)`. That happens
when `s` is not a multiple of 4 and `k` falls in the final partial word — for **any** `s`, not only
`s < 4` — or when `b` is not 4-aligned, where the `andi -4` rounds the cursor below `b`. The
"object smaller than 4 bytes" case in the registry is the special case where every byte is in the
final partial word.

Why the first pass missed it: the control was `&arr[16]` of a 64-byte array, an offset taken from
the comfortable interior of the range the claim generalised over. A control has to sit at the
**edge** of that range; `arr[16]` of `arr[64]` could not have failed.

## Two further corrections to how the mechanism is described

**Alignment is not the fix, and `andi -4` is not the symptom.** Clang propagates an object's
alignment onto the `cmpxchg`, and with `align 4` `AtomicExpandPass` takes its "low bits known zero"
path and skips the `ptrmask` entirely — but `getMinCmpXchgSizeInBits()` is 32, so `PMV.WordType`
stays `i32` and the access stays word-wide:

| object | `cmpxchg align` | `andi -4` | bound | access |
|---|---|---|---|---|
| `unsigned char`, align 1 | 1 | yes | `[b, b+1)` | `lr.w` after rounding |
| `unsigned char`, `aligned(4)` | 4 | **no** | `[b, b+1)` | `lr.w` at `b` |

So aligning a lone byte removes the rounding and keeps the overrun, and the fault address then
equals the object's own address with no rounding visible in the output — harder to diagnose, not
safer. Conversely the arms with `andi -4` (`&arr8[7]`, `&arr5[1]`) are both **safe**: the rounding
is not the defect, the widening is.

**The storage classes.** Globals: the bound is `getTypeAllocSize(GVar->getValueType())` in
`selectLGA`, `llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp:1717`, SHRINK at `:1736`, under
`-capstone-shrink-globals` (`cl::init(true)`). Thread-locals: the same expression in
`getCapabilityTLSAddr`, `CapstoneISelLowering.cpp:9901`. Stack locals: **confirmed**, which scopes
in the registry's untested "or one allocated alone" clause — a lone `unsigned char` local whose
address is taken gets `li s2, 1 ; add ; shrink` and then the same rounding and `lr.w`.

**The heap case is UNRESOLVED and must not be quoted either way.** The exactly-`n` narrowing that
would have settled it (`sh_narrow`, "Exactly n bytes when that is representable") is in
`capstone/ports/musl-capstone/runtime/sublet_heap.c`, which `7901153f10fa` ("runtime: retire
superseded allocator backends and campaign paths") **deleted**. The live path is
`capstone/runtime/virtual/heap.c`: slot parents are shrunk to a musl group `stride` (`:66-69`) and
the child is derived as `capstone_cap_derive(parent->parent, …, 16 + stride*count)` (`:108-109`).
Whether a heap byte can overrun therefore depends on musl's stride geometry, which this pass did
not establish.

## Option C — native byte/halfword atomics. The compiler side is already written.

`CapstoneISelLowering.cpp:835-839`:

    if (Subtarget.hasStdExtZabha() && Subtarget.hasStdExtZacas())
      setMinCmpXchgSizeInBits(8);
    else
      setMinCmpXchgSizeInBits(32);

At a granule of 8, `createMaskInstrs` returns at its `PMV.ValueType == PMV.WordType` early exit:
no `ptrmask`, no widening, the access exactly as wide as the object. `shouldExpandAtomicCmpXchgInIR`
(`:25421-25422`) and the RMW path (`:25318`, `:25324`) are already gated on the same two features.

**So the entire cost of option C is RTL and silicon** — Zabha and Zacas in the core, and a
bitstream. There is nothing to write on the compiler side. If those features are on the table for a
future bitstream, A and B below are both transitional.

## Option A — make the object own its word (pad and align), and let the bound follow

Both halves are necessary, and the two matched pairs above are what prove it: `align ≥ 4` alone
leaves the widening (the lone-byte pair), and `size` rounded up to a multiple of 4 alone leaves the
`andi -4` rounding below an unaligned base (the `arr5`/`arr8` pair shows the size half; the base
half follows from the same arithmetic).

Sites: `CapstoneISelDAGToDAG.cpp:1717` (globals), `CapstoneISelLowering.cpp:9901` (thread-locals),
the `CapstoneShrinkStack` path for allocas, and whatever the live heap question resolves to.

**The irreducible cost.** The last ≤3 bytes of any object whose size is not a multiple of 4 stop
being caught as overruns. Where the compiler pads an object it controls — a global or an alloca
given 4 bytes of its own — nothing else is reachable and this is genuine padding, not a weakening.
Where it does not, as with `arr5`, it is a weakening: an overrun into `arr5[5..7]` is no longer
caught. That trade cannot be engineered away; either the atomic cannot run, or those bytes are
reachable.

**The expensive question A has to answer is which objects.** "Can be a sub-word atomic target" is
not decidable for a global whose address escapes the module. The conservative answer reaches every
address-taken byte-indexable object, which includes ordinary byte arrays and strings. The narrow
answer is "only where a sub-word atomic is visible in the module" — and full LTO, already this
project's answer to the gp-captable multi-TU slot collision, makes that test program-wide.

## Option B — route sub-word atomics to the generic libcall. Needs an IR rewrite, not a hook.

`setMaxAtomicSizeInBitsSupported(XLen)` is a **maximum**: it is what sends 128-bit capability
atomics to the generic calls under C-54, and it cannot send sub-word ones anywhere. Returning
`AtomicExpansionKind::None` for a sub-word `cmpxchg` leaves an `i8 cmpxchg` that ISel cannot
select, there being no byte LR/SC without Zabha. B therefore needs a pre-AtomicExpand rewrite to
the generic `__atomic_compare_exchange(size, ptr, …)`.

What makes B attractive anyway: that implementation already exists, is already linked into every
application (`capstone/runtime/cmake/Application.cmake:41`), already holds a leaf spin lock safe
under the one-hart preemption model, and already moves values through memory in `size`-byte copies
— so it is bound-exact by construction, for every storage class at once, including whatever the
heap turns out to do.

What it costs: a sub-word atomic stops being lock-free, which is observable through
`__atomic_always_lock_free` and the `ATOMIC_*_LOCK_FREE` macros. CPython's `PyMutex` — the live
consumer C-74 names — becomes a lock around a lock. Correct, and slow.

**Incidental, worth fixing independently of this decision:** the runtime defines only the *generic
unsized* calls. `capstone/ports/musl-capstone/runtime/atomic_libcalls.c:43-46` renames
`capstone_atomic_{load,store,exchange,compare_exchange}` onto `__atomic_*` with
`#pragma redefine_extname`, and nothing defines `__atomic_compare_exchange_1`. A build without
`+a` emits a call to exactly that symbol — observed while setting up the probe above — so such a
build cannot link against this runtime. That is a confusing failure mode for anyone who drops the
atomics feature by accident.

## Recommendation

If a compiler fix is wanted now: **option A in its narrow form** — pad and align only objects with
a sub-word atomic visible in the module. The urgent consumer, a standalone one-byte `PyMutex`, is
covered completely; the cost is ≤3 bytes per such object; and LTO already makes the visibility test
program-wide. Keep B written down as the backstop for a sub-word atomic reached through a pointer
whose provenance the compiler cannot see, and resolve the heap question before either lands.

If Zabha and Zacas are plausible for a coming bitstream, the cheapest path is to wait for them,
because the compiler work is already done and both A and B would then be removed again.

## Registry text, for whoever owns `ref/ISSUES.md`

Replacing the entry's scope paragraph ("So any object smaller than 4 bytes whose capability is
bounded to it alone …"):

> So a sub-word atomic on byte `k` of an object bounded `[b, b+s)` faults iff the 4-byte-aligned
> word containing `b+k` is not contained in `[b, b+s)`: whenever `s` is not a multiple of 4 and `k`
> lies in the object's final partial word — **at any object size, not only `s < 4`** — or whenever
> `b` is not 4-aligned, where the rounding reaches below `b`. A lone object smaller than 4 bytes is
> the special case in which every byte lies in the final partial word. Measured: a byte at offset 4
> of a 5-byte 4-aligned array faults, the same byte of an 8-byte array does not, with identical
> code but for the bound immediate. Confirmed for globals, thread-locals and stack locals;
> the heap is unresolved (the allocator this was first costed against has since been retired).
> Raising alignment alone does **not** fix it — it removes the `andi -4` and keeps the word-wide
> access — and the `andi -4` is not the symptom: the arms that have it are the safe ones.
> Options and a recommendation: `docs/plans/2026-10-16-c74-subword-atomic-bounds.md`.

## Reproduce

No headers, no runtime, no board. Two stages, because the `align` that `AtomicExpandPass` reads is
put on the `cmpxchg` by clang and the expansion happens in `llc`:

```c
/* c74-edge.c */
typedef unsigned char u8;
static u8 lone1;                                 /* lone byte, align 1          */
static u8 lone4 __attribute__((aligned(4)));     /* lone byte, align 4          */
static u8 arr5[5] __attribute__((aligned(4)));   /* size NOT a multiple of 4    */
static u8 arr8[8] __attribute__((aligned(4)));   /* control: size is a multiple */
int l1(void){ u8 e=0; return __atomic_compare_exchange_n(&lone1,  &e,1,0,5,5); }
int l4(void){ u8 e=0; return __atomic_compare_exchange_n(&lone4,  &e,1,0,5,5); }
int e1(void){ u8 e=0; return __atomic_compare_exchange_n(&arr5[1],&e,1,0,5,5); }  /* safe  */
int e4(void){ u8 e=0; return __atomic_compare_exchange_n(&arr5[4],&e,1,0,5,5); }  /* FAULTS */
int c4(void){ u8 e=0; return __atomic_compare_exchange_n(&arr8[4],&e,1,0,5,5); }  /* safe  */
int c7(void){ u8 e=0; return __atomic_compare_exchange_n(&arr8[7],&e,1,0,5,5); }  /* safe  */
```

```sh
clang --version | head -1                       # print the identity beside the readings
clang -target capstone64-unknown-elf -ffreestanding -fno-builtin -fno-jump-tables \
      -O2 -S -emit-llvm c74-edge.c -o c74-edge.ll
grep cmpxchg c74-edge.ll                        # l4/e4/c4 carry align 4; l1/e1/c7 align 1
llc -mtriple=capstone64 -mattr=+a c74-edge.ll -o c74-edge.s
grep -c 'lr\.w' c74-edge.s                      # POSITIVE CONTROL: must be 6, not 0
```

`-mattr=+a` is load-bearing. Without it every atomic becomes a libcall, no `lr.w` is emitted at
all, and the run reads as "the defect is absent" — which is how this probe first went void. The
`grep -c` above is the control that catches it.

Then read each arm's bound immediate against its cursor offset and access width: `e4` is
`li a1, 5` with `cincoffsetimm a0, a0, 4` and `lr.w`, and `c4` is the same with `li a1, 8`. That one
immediate is the whole finding.
