# Decision: how to tag capability globals (constructor-codegen vs GCT-consumer)

Status: **decided (architecture), partially implemented.** 2026-06-24.

## Problem

On Capstone, a capability tag is out-of-band metadata that cannot be encoded in a
static ELF image. So any file-scope global that holds a capability — a pointer
global, a pointer table like BEEBS dtoa's `char *nums[]`, a function-pointer
table — loads with the **address bits but no tag**. The first runtime dereference
of such an element faults:

```
[CAPSTONE] Cap mem access requires capability: ... rs1 = x.., imm = 0
```

The compiler emits these initializers as plain `R_Capstone_64` relocations (an
8-byte address in a 16-byte slot, no tag mechanism). There is no capability
relocation type and no `__cap_relocs`-style runtime fixup in the toolchain.

This is the root cause of BEEBS `dtoa` blocker #1 (see
`plans/beebs-deferred-benchmarks.md` §15). It is general: it affects any program
with initialized capability globals, not just benchmarks.

## Two candidate architectures

1. **GCT metadata + runtime consumer.** The compiler emits a `.gct` section
   (`SCAP` records: objects, slots, template bytes) describing each capability
   global; a runtime consumer walks it and rebuilds the capabilities. This is the
   pre-existing in-tree research line (`tests/runtime-qemu/static-cap-typed-load-repro/`),
   with a working single-slot consumer POC and (as of 2026-06-24) compiler
   emission extended to arrays (`CapstoneAsmPrinter.cpp`,
   `collectStaticCapReducedObject`; lit test `static-cap-gct-array.ll`).

2. **Constructor-codegen.** The compiler instead emits, per module, an
   initializer function that stores each capability global's value at runtime
   with ordinary stores (`nums[0] = "...";`). Normal codegen materializes each
   target as a properly bounded, tagged capability and `scc`-stores it **in
   place**. The init function runs before `domain_main`.

## What was validated (spike, 2026-06-24)

Two reduced array-shaped domains in
`tests/runtime-qemu/static-cap-typed-load-repro/` (wired into `build.sh`/`run.sh`):

- `fail_str_array_load.c` — statically-initialized `const char *kTable[3]`, read
  at runtime → **faults** with `Cap mem access requires capability` (reproduces
  the array form of the blocker).
- `fix_str_array_runtime_materialize.c` — non-const array, each element assigned
  at runtime (`gTable[i] = "..."`), then read back through the array →
  **succeeds, returns 336** ('o'+'h'+'y' first chars).

This proves the key runtime semantics the constructor-codegen path depends on:
an ordinary C store of a string-literal/global address into a writable capability
global produces a **tagged** slot, and a later normal load + dereference of that
slot works — entirely in place, with no metadata parsing and no hand-written
capability-derivation assembly.

## Decision

**Adopt constructor-codegen as the resolution path** for capability-global
tagging. Rationale:

- **Simplicity / low risk.** No runtime metadata parser, no `.gct` walking, and —
  critically — no hand-written capability-derivation asm (`setaddr`/set-bounds/
  `scc` from a root). The compiler's normal lowering already produces correctly
  bounded, tagged capabilities; we just emit the stores.
- **In place by construction.** It writes the *original* global, so all existing
  code that accesses `nums[i]` works unchanged. The GCT-consumer path, by
  contrast, rebuilds into a *parallel* object and would additionally need a
  format change to carry a relocated back-reference to the holder global plus a
  general in-place consumer — strictly more moving parts.
- **No firmware/ABI dependence on a loader.** Materialization is just domain code.

The compiler-emitted `.gct` metadata (including the new array support) is **kept**
as a useful, inspectable description of capability globals (tooling, the
descriptor-driven POCs, and as a possible data source for the auto-generated
constructor), but the *consumer* of record becomes compiler-emitted init code
rather than a runtime `.gct` walker.

## Remaining work to productionize (not yet done)

1. **Compiler**: auto-emit a per-module `__capstone_init_cap_globals` that stores
   every capability-global initializer at runtime (reuse the
   `collectStaticCapReducedObject` analysis to find them); zero the static-image
   slots (or keep them, they are overwritten).
2. **Placement**: ensure such globals land in writable `.data` even when `const`
   (const string/function tables currently go to `.rodata`; a runtime store there
   would fault). dtoa's `nums[]` is already non-const.
3. **Wiring**: run the initializer before `domain_main` — either a fixed symbol
   called from `start.S`, or `.init_array` processing in the domain crt. This is
   the only domain-startup-ABI touch and should be reviewed with the runtime owner.
4. **Function-pointer slots**: same store pattern; confirm a function capability
   is materialized correctly for an indirect-call target.

## Important caveat

Resolving capability-global tagging fixes blocker #1 only. BEEBS `dtoa` also has
blocker #2 — an untagged-base capability load (`helper_cslcc`) in David Gay's
`char[]` bigint arena on complex inputs — which is independent and would remain.
So this decision is justified as **general capability-globals infrastructure**,
not as a `dtoa`-specific fix.
