# Three codegen fixes: two silicon-ladder rungs genuinely unblocked; RV8 at −O1/−O2 now COMPILES but still does not RUN

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** zero board time. All four failures were **build-time**, so none needed QEMU or hardware to diagnose.
**Status:** builds + QEMU parity green; regression sweep green with 6 verified-pre-existing failures. RV8 correctness unverified.

---

## Summary

| target | was | now |
|---|---|---|
| `beebs_crc32` (ladder rung 6) | would not build at −O1+ | **builds −O0/−O1/−O2, QEMU-correct** (oracle 1703161001) |
| `beebs_insertsort` (ladder rung 7) | crashed clang at −O1 | **builds −O0/−O1/−O2, QEMU-correct** (oracle 271779359) |
| RV8 suite at −O1/−O2 | 0/7 built | 5/7 **build** — but **0/10 pass at runtime** (see §Validation) |

The silicon ladder is now **5 rungs buildable and QEMU-correct**, up from 3 — that part is real
and validated. The two extra rungs are the cheapest available additions to the paper's perf table
and the highest-value use of the next board window.

**The RV8 result is NOT a win and must not be quoted as one.** The fixes moved RV8's −O1/−O2
failure from *compile time* to *run time*: five benchmarks now compile and then fault or hang.
Compiling is not passing.

---

## 1. `beebs_crc32` — not a compiler bug at all

The kernel deliberately generates its CRC table **at runtime** (see its header comment) precisely
to avoid shipping a large initialized global. At −O1+ **LLVM constant-folds the entire generator
loop** and re-materialises the result as a 2048 B *private* constant `.L.crctable`.

That defeats the design, and our gp cap-table glue cannot deliver it:
- 2048 B overflows the unrolled 12-bit store-offset path (`> 2040`), and
- the **large-RO copy path requires a linkable, non-`.L` symbol** — it computes
  `lla <sym> - __gpfree_globals_base` from the *glue*, a separate translation unit, so a private
  local constant is invisible to it.

Hence the generator's hard error: `global 2: 2048 B of *initialized* data overflows the 12-bit
store offset and is not copy-eligible (sym='.L.crctable')`.

**Fix (source, one line):** make the polynomial opaque to the optimizer.

```c
UNS_32_BITS poly = 0xedb88320UL;
__asm__("" : "+r"(poly));      /* defeats constant-folding of the whole table */
```

No runtime operation changes; the table stays a 1 KiB `.bss` array at every −O level.

**Worth keeping in mind:** any benchmark that hand-rolls a table to dodge the large-RO limit can
be silently undone by the optimizer at −O1+. The general cure is the large-RO delivery path
learning to handle private constants (it needs a linkable name), which is the same mechanism
SQLite's big const tables will need.

## 2. `beebs_insertsort` — two real LLVM bugs, the second one substantive

**(a) The reported crash** was generic LLVM:
`APInt::getSExtValue()` asserting `getSignificantBits() <= 64`, from
`SelectionDAGAddressAnalysis.cpp` `matchLSNode` (via `DAGCombiner::getUniqueStoreFeeding`).
The accumulator there is an `int64_t`, so a constant wider than 64 significant bits cannot be
represented at all — bailing out is the correct behaviour, not a workaround. Guarded at all
three constant sites in that function.

**(b) The bug that crash was hiding.** With the guard in, the DAG showed the real defect:

```
t77: i128 = CapstoneISD::CIncOffset t5, Constant:i128<18446744073709551612>
t57: store<... into %ir.sunkaddr ...>
```

`18446744073709551612` = `0xFFFFFFFFFFFFFFFC` = **−4 zero-extended** into the i128 pointer
carrier. (The same DAG also contains a correct `Constant:i128<-4>`, so only one path was wrong.)

Root cause in **CodeGenPrepare**'s address sinking: `AddrMode.BaseOffs` is `int64_t`, but
`ConstantInt::get(Type*, uint64_t, bool IsSigned = false)` **defaults to zero-extend**. For any
target whose `IntPtrTy` is ≤ 64 bits the bit pattern is identical, so this is invisible
everywhere else; for a 128-bit capability pointer a negative offset becomes a huge positive one.
Fixed at all three `BaseOffs` sites by passing `/*IsSigned=*/true`.

**This is a latent bug for any wide-pointer target** (CHERI included), and it was producing a
*wrong address* that only our backend's `getSignedI128ValueOrFatal` guard turned into a hard
error instead of a silent miscompile. It plausibly accounts for other −O1+ failures.

## 3. `i128 = and` — RV8's −O1 blocker

`Cannot select: i128 = and`. `lowerScalarI128And` only handles a **constant mask** that fits in
XLen; with two non-constant operands it returns an empty `SDValue`. The dispatch arm did
`return lowerScalarI128And(...)` **unconditionally**, so that bail left the node unlowered
instead of falling through to `lowerScalarI128Logical` — the general path OR/XOR already use,
which narrows both operands to XLen and re-extends.

**Fix:** try the constant-mask fast path, then fall through.

```cpp
case ISD::AND:
  if (Op.getSimpleValueType() == MVT::i128)
    if (SDValue V = lowerScalarI128And(Op, DAG, Subtarget))
      return V;
  [[fallthrough]];
case ISD::OR:
case ISD::XOR:
```

## What is still blocked (deliberately not attempted)

`rv8_qsort` (`i128 = xor`) and `rv8_miniz` (`i128 = or`) fail one level up, inside
`lowerScalarI128Logical`, which bails when the two operands are not *matching* extends (both
sign or both unsigned). Closing that requires deciding what the **high 64 bits** mean in the
mixed case — capability metadata vs a genuine 128-bit integer value. That is a semantics call
with miscompile risk, not a mechanical fix, and it was **left alone under deadline pressure**.
It is the obvious next compiler task if RV8 at −O1/−O2 is wanted in full.

---

## Validation

Full sweep run, then the one load-bearing uncertainty in it was settled directly.

| suite | baseline | actual | status |
|---|---|---|---|
| Capstone backend lit | 39-40/40 | **41/41** | PASS (re-run against the final binary) |
| clang `cap-ptr-compare.c` | 1/1 | 1/1 | PASS |
| generic LLVM lit — RISCV (full) | 2257 | 2256 pass, 1 fail | 1 fail, **pre-existing** |
| generic LLVM lit — X86 (full) | 5269 | 5246 pass + 18 XFAIL, 5 fail | 5 fails, **pre-existing** |
| CoreMark (QEMU) | validated | `__COREMARK_PASSED__` | PASS |
| BEEBS (QEMU) | 82/82 | **82/82** | PASS |
| authority suite (QEMU) | 26/26 | **32/32** (suite has grown) | PASS |

**The 6 failures are `emutls*` / `tls-android`** — emulated-TLS shadow-variable initializer
emission (`.long 4` where the test wants a zero placeholder). Nothing to do with address-mode
offset construction or `matchLSNode`.

**Verified pre-existing, not merely argued.** The three LLVM changes were stashed, `llc` rebuilt
at that baseline, and the six tests re-run: **all six fail identically without the changes.** The
working tree was then restored and confirmed byte-identical to the saved patch (same md5), and
`clang`/`llc` rebuilt. So the "unrelated" conclusion rests on an actual before/after, not on
code-path reasoning.

**Why the generic-LLVM changes are safe for other targets:** both are unreachable for pointers
<= 64 bits — `fitsInOffset` (`getSignificantBits() <= 64`) is always true there, and
`ConstantInt::get`'s single-word path applies `clearUnusedBits()` only, so `IsSigned` cannot
change the bit pattern. The full X86 + RISCV lit runs are the empirical confirmation.

### RV8 correctness at −O1/−O2 — RUN, and it FAILS 10/10

Never previously tested (the code did not compile). Now tested, with −O0 controls:

| benchmark | −O0 | −O1 | −O2 |
|---|---|---|---|
| `primes` | PASS | **silent hang** | **silent hang** |
| `aes` | PASS | **silent hang** | **silent hang** |
| `dhrystone` | PASS | **silent hang** | **silent hang** |
| `sha512` | PASS | **cap fault** | **cap fault** |
| `norx` | PASS | **cap fault** | **cap fault** |

Two distinct signatures, each identical at −O1 and −O2 (same PC), so they are deterministic
codegen faults, not flakes:

```
sha512: [CAPSTONE] Cap mem access OOB: pc = 10158025c, rs1 = x19, cursor = 1015ffe38,
        imm = 192, addr = 1015ffef8, size = 8, bounds = (1015ffdf8, 1015ffec0)   cause = 5
norx:   [CAPSTONE] Cap mem access requires capability: pc = 10158072c, rs1 = x12, imm = 2
        cause = 24
```

`primes`/`aes`/`dhrystone` emit **no** fault at all — the domain loads and then produces no
further serial output until the harness timeout. Boot and login completed in every case, so this
is not the `boot-login` infra flake.

**These are not regressions** — you cannot regress a program that previously failed to compile.
They are pre-existing −O1/−O2 codegen defects that the fixes newly *exposed*. The old note's
"RV8 is 0/7 at −O1+" therefore still stands in substance; only the failure mode moved.

**Keep the `i128 AND` fall-through anyway**: leaving a node unlowered was a real bug, the fix is
strictly additive (it can only affect code that previously failed to compile), and the whole
−O0 corpus is green with it in.

**Still owed:**
- Root-cause the two RV8 −O1/−O2 failure classes (an OOB through a stack-derived capability with
  bounds visibly too small, and an untagged capability reaching a load). Both are the shape of a
  bounds/provenance codegen bug at −O1+, and the OOB one is well localized: fixed PC, known
  cursor/bounds/imm. This is the next real compiler task.
- Board measurement of the two new ladder rungs, when hardware frees up.

## Files touched

- `llvm/lib/CodeGen/CodeGenPrepare.cpp` — 3 sites, `/*IsSigned=*/true`.
- `llvm/lib/CodeGen/SelectionDAG/SelectionDAGAddressAnalysis.cpp` — `fitsInOffset` guard, 3 sites.
- `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp` — AND falls through to the general logical path.
- `capstone/tests/runtime-qemu/silicon-ladder/beebs_crc32_kernel.h` — opaque polynomial.

Nothing committed.
