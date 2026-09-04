# Fixed: `-O2`/`-O1` Capstone backend ICE on capability-value selects

*2026-07-15. Root-cause + fix for the codegen-lane ICE flagged in COORDINATION.md
(2026-07-14, "new `-O2` Capstone-backend ICE"). Done in A-lane's working tree by
explicit user request (normally a B-lane `llvm/` item). This is the ICE that forced
the paper's Capstone BST tree-cost arm to build at `-O0`
(`tests/runtime-qemu/revoke-cost-probe/`, `history/15-07-2026_00-20-00_cheri-capstone-perf-comparison.md`).*

## Symptom

Deterministic clang backend crash at `-O2`/`-O1` (`-O0` fine). Assertion in a
`+assertions` build:

```
isa<> used on a null pointer   (Casting.h:109)
  #20 CapstoneTargetLowering::lowerSELECT(...)::$_1 (lowerBranchSelect lambda)  CapstoneISelLowering.cpp:10296
  #21 CapstoneTargetLowering::lowerSELECT                                       CapstoneISelLowering.cpp:10409
```

Minimal trigger — a conditional store of a capability into two distinct named
globals:

```c
static void *g_a, *g_b;
void store2(void *arg, int first){ if (first) g_a = arg; else g_b = arg; }
```

(The rtl-smoke workaround was the array-indexed store `regions[i&1] = arg`, which
never forms the offending node.)

## Root cause (two layered bugs; the second was masked by the first)

`GlobalMerge` + DAGCombine strength-reduce the two-named-globals store into
`store arg, (add GlobalAddr<_MergedGlobals>, (select cond, i128 32, i128 16))`
— an **i128 (capability-typed) `select` whose two arms are non-null constants**
(the merged-global byte offsets). `lowerSELECT` handles that as:

1. **Null-deref (the crash).** `lowerCapabilitySelect` captured the outer
   `TrueV`/`FalseV` **by reference** and did
   `TrueV = materializeCapabilitySelectOperand(TrueV)`. For a non-null constant
   arm that helper returns a null `SDValue` (its "bail, fall back" signal), so the
   assignment **clobbered the shared `TrueV`/`FalseV` to null**. It then returned
   `SDValue()` to fall through to `lowerBranchSelect()` (line 10409), which
   immediately did `isa<ConstantSDNode>(TrueV)` on the now-null value → assertion.

2. **`Cannot select` (was hidden behind #1).** Even with the operands preserved,
   the branch-based fallback can't lower this node: `Select_GPRCAP`
   (`SelectCC_GPR_rrirr<GPR,i128>`) requires its true/false values **in registers**,
   and a bare `Constant:i128` matches no pattern in that slot (i128-constant
   materialization only exists via the i64 `li` path, wrong type for the operand).
   So the "bail to `lowerBranchSelect`" path was never actually viable for a
   non-null constant arm — it only reached the null-deref first.

## Fix (`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`, `lowerSELECT`)

1. **Rematerialize non-null constant arms into registers instead of bailing.**
   `materializeCapabilitySelectOperand`, for a non-null constant that fits in
   XLen, forces it into a capability register via `CopyToReg`/`CopyFromReg` so it
   materializes as a plain integer load (`li`) — an untagged integer value in a
   capability register — which makes `Select_GPRCAP` match. Null arm still → `X0`;
   wider (>XLen) constants still bail.

   **Runtime-caught correction (do not regress):** the first attempt used
   `CIncOffset(X0, constant)` (emitting `cincoffsetimm dst, zero, K`). It passed
   compile + lit but **faults at runtime** — `cincoffsetimm` asserts a *tagged*
   source (`helper_cscincoffsetimm: rs1_v->tag`), and X0/zero is untagged. Caught
   by actually booting the `-O2` tree probe under QEMU. The `li` form (via
   CopyToReg, the same path the return lowering uses for an i128 integer constant)
   is the correct untagged-integer materialization. lit now `CHECK-NOT`s the
   `cincoffsetimm ..., zero,` form and `CHECK-DAG`s the `li` form.
2. **Use local `CapTrueV`/`CapFalseV` copies** so any bail leaves the shared
   `TrueV`/`FalseV` pristine for the fallback (kills the null-deref for good).
3. **Guard the "two constants differing by 1 → ISD::ADD/SUB" transform in
   `lowerBranchSelect` with `VT != i128`.** That path is capability-only (it is
   reached solely from the `VT == MVT::i128` caller), and materializing a
   capability select result via an integer ADD/SUB is unsound — matches the
   function's own stated invariant ("avoids integer/bitwise transforms not valid
   for capabilities").

Emitted code for the trigger is now a correct branch-based capability select:
`beqz` on the condition; each arm `li aN, {16,32}` (a plain untagged integer,
not `cincoffsetimm` from the null register); then `cincoffset` the merged-globals
base by the selected offset and `stc`.

## Validation

- **Compiles (all `-O2`, previously ICE):** the minimal repro; the real
  `tests/rtl-smoke/borrow_cost_fpga.c` reverted to named globals; the BST
  `cur = cond ? cur->l : cur->r` shape; and the Capstone tree probe
  `tests/runtime-qemu/revoke-cost-probe/revoke_cost_tree.c` at **`-O2` and `-O1`**.
- **Array-indexed workaround still compiles** (no regression to the shipped file).
- **Lit:** Capstone backend **39/39** (incl. a new regression case
  `select_cap_const_arms` in `llvm/test/CodeGen/Capstone/select-cap.ll`), clang
  Capstone **6/6**.
- **No codegen change to existing workloads:** the new paths fire only on the
  i128-select-with-constant-arms case (which previously could not produce code),
  so RV8/BEEBS/CoreMark/authority codegen is byte-identical (not re-run — nothing
  in their lowering reaches the changed code).

## Runtime validation (QEMU)

- **Tree probe re-measured at `-O2`** (`DOMAIN_OPT_LEVEL=-O2 run-tree-cost-probe.sh`,
  3 boots clean, no assertion). Needed one extra fix: at `-O2` the shared
  workload's key generation lowers to a hardware multiply, so the tree domain
  build now enables the `M` extension (`+m`, as CoreMark does) — else `ld.lld`
  fails on `__muldi3`. Result: bump 230, norevoke 24,300, **revoke−norevoke =
  +5 instr/op** — the O(1) revoke-at-free op, now matching the microbench's +5
  exactly (the `-O0` build measured +10). Paper `tab:perftree` updated to the
  `-O2` +5. Build-script comment refreshed (ICE no longer pins it to `-O0`).

## Follow-ons (not done here)
- Rebuilt the `cmake-build-debug` tree (what `$CAPSTONE_CLANG` and the domain
  build scripts use). The separate `llvm/build` tree, if used by any codegen path,
  needs the same one-file rebuild.
- COORDINATION.md B-lane item (2026-07-14) marked resolved.
