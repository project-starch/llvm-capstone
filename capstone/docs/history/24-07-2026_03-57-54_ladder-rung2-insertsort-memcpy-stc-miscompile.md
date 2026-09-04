# Silicon ladder rung 2 (BEEBS insertsort): two bugs found + fixed

Date: 2026-07-24. Context: climbing the silicon benchmark ladder
(`plans/sqlite-on-silicon-scoping.md`) past rung 1 (matmult-int) and init_probe.
Rung 2 = BEEBS `insertsort` (a *found* benchmark), added as
`capstone/tests/runtime-qemu/silicon-ladder/beebs_insertsort_{kernel.h,app.c,host.c}`.
Single-TU, integer, real init->sort->verify call graph, a global int array
(`is_a[11]`, .bss) plus a function-local const array (`expected[11]` in
`is_verify`). Ran through the generic harness (`run-ladder-qemu.sh beebs_insertsort`)
in the silicon config (`-capstone-gp-captable` + gp-free call/ret + shrink off + `+m`).

First run FAILED: domain ran cleanly (no fault) but returned `255001740` vs the
native oracle `271779359`. Decoding the FNV checksum against candidate array/ok
states identified the exact failure mode: **array sorted correctly, but `ok=0`**
(`is_verify` returned 0), i.e. `expected[]` read wrong in the domain.

## Bug 1 — generator: `.L`-prefixed initializer symbols dropped (harness-side)

`gen-gp-captable-glue.py` picks each initialized global's template symbol from the
`.capstone_gp_table` ADD/SUB reloc pair. Old heuristic: "skip `.L`-prefixed
symbols" (meant to skip the `.L0` SUB anchor). But a **function-local** const
array is promoted by the compiler to a private symbol named
`.L__const.<func>.<var>` -- itself `.L`-prefixed -- so BOTH reloc symbols were
`.L`, the initializer was dropped, and the global was wrongly zero-filled.
init_probe passed earlier only because its `LUT` was file-scope (plain name).

Fix: select the initializer by **reloc TYPE**, not symbol name -- the ADD half
(`R_RISCV_ADD8/16/32/64` = 33..36) targets the initializer; the SUB half (37..40)
targets the anchor. Decode the type from the Info field (`parts[1]` low 32 bits)
in the GNU-style readobj output. After this fix the generator reported
"1 initialized" and materialized the correct bytes into gp[1] -- **but the retval
did not change**, proving the domain never reads gp[1] for `expected[]`. That
exposed the real bug.

## Bug 2 — compiler: sub-capability memcpy `stc`-packing miscompile (backend)

At `-O0` a local `int expected[11] = {...}` is initialized by copying its 44-byte
const template to the stack. `CapstoneTargetLowering::findOptimalMemOpLowering`
(`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`) already had a fix for the
tag-safe 16-aligned case (i128/ldc-stc chunks) AND a fix for the sub-cap case
**only when `size % 8 == 0` and 8-aligned** (i64 chunks). A 44-byte copy is
`44 % 16 = 12` and `44 % 8 = 4`, so it matched **neither** branch and fell to the
generic lowering, which still picked an i128 (capability) unit for its leading
16-byte chunks because the *destination* stack slot was 16-aligned. The
misaligned-*source* i128 load (template is only `.p2align 2`) is (mis)legalized
into a single i64 load zero-extended to i128 while the paired i128 store stays a
16-byte `stc`, so the **upper 8 bytes of every 16-byte unit are silently dropped**:

    ldc  a0, 16(gp)          ; source template
    lwu  a3, 0(a0) ; lwu a2, 4(a0) ; slli a2,a2,32 ; or a2,a2,a3   ; a2 = (w1<<32)|w0  (64-bit int)
    stc  a2, 0(a1)           ; 128-bit capability store of a 64-bit reg -> words 2,3 = 0

Result: `expected[] = {0,2,0,0,5,6,0,0,9,10,11}` instead of `{0,2,3,4,5,...}`, so
`is_verify` mismatched at index 2 and returned 0. Config-independent: reproduces
identically in the plain PureCap ABI (only source addressing differs). Minimal
repro: `void f(void){ int e[11]={0,2,3,4,5,6,7,8,9,10,11}; sink(e); }` at `-O0`.
This is the same corruption class the existing 8-multiple workaround documents
(by-value 16-byte struct returns, `range = MakeRange(...)`), just uncovered for
sizes that are not a multiple of 8, or copies only 4-aligned.

Fix: generalize the sub-cap branch to fire for **any** copy that is not
(16-aligned on both ends AND a 16-multiple size) -- i.e. any copy that cannot
carry an in-place tagged capability -- and decompose into scalar (tag-stripping)
chunks sized to the copy's *actual* alignment (largest power-of-two unit <= XLen
dividing `min(src,dst)` align), never i128. After the fix the 44-byte copy lowers
to 11 scalar `sw` immediate stores (const-folded), fully correct.

## Verification

- Minimal repro: no `stc` into the copy dest; 11 correct `sw` stores.
- Ladder QEMU (serial, silicon config): rung 2 `beebs_insertsort` **PASS**
  (retval 271779359 == oracle); rungs 1 `matmult_int` (774662735) and `init_probe`
  (4093668916) still PASS -- no regression.
- `llvm/test/CodeGen/Capstone` lit: 41/41 pass (no golden-asm breakage).

## Status / open

- The compiler change (`findOptimalMemOpLowering`) is **unconditional** (affects
  all Capstone codegen), so it must pass the full regression gate (CoreMark /
  BEEBS 82 / authority 26 / RV8 7 / SQLite -- serialize the QEMU suites, shared
  rootfs.ext2 write-lock) before commit. Lit is green; QEMU corpus gate pending.
- Also seen (separate, NOT fixed here): `-O2` on the same kernel crashes clang
  with an APInt `getSExtValue` assertion in DAGCombiner store->load forwarding
  (`ForwardStoreValueToDirectLoad` / `matchLSNode`, 128-bit offset). The ladder
  builds at `-O0`, so it does not block rung 2, but it is a real bug to track.
- Consider a lit regression test asserting the sub-cap local-const-array copy
  emits no `stc` from an integer register.
