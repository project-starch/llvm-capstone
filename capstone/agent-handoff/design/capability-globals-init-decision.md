# Decision: how to tag capability globals (constructor-codegen vs GCT-consumer)

Status: **implemented & validated (single-module).** 2026-06-24.

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

## What was implemented (2026-06-24)

1. **Compiler** — `llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp`, a
   ModulePass run in `addIRPasses` at all opt levels. For each capability global
   it can reduce (single-field struct / array of addrspace(200) pointers, with
   GlobalVariable/Function targets), it synthesizes a per-module
   `void __capstone_cap_init(void)` whose body stores each element in place; isel
   lowers each store to a tagged capability store (`cincoffset gp` + `delin` +
   `stc`). Intrinsic/metadata globals (appending linkage, `llvm.*`,
   `llvm.metadata`) and thread-locals are skipped.
2. **Wiring** — `capstone/my_first_domain/start.S` calls `__capstone_cap_init`
   before `domain_main` (same `auipc`/`cincoffset gp`/cjalr pattern). A **weak
   no-op** `__capstone_cap_init` in `start.S` is used when a domain has no
   capability globals; the compiler-emitted **strong** symbol overrides it.
3. **Test** — `llvm/test/CodeGen/Capstone/static-cap-global-init.ll`; the runtime
   acceptance suite `tests/runtime-qemu/static-cap-typed-load-repro/` (the three
   previously-faulting `fail_*` cases now succeed unchanged: 111 / 111 /
   305419896, across string-struct, array, and function-pointer shapes).

### Key implementation decisions

- **Volatile stores, static initializer left intact.** The stores are `volatile`
  so they are never elided as redundant-with-initializer; the untagged static
  bytes are harmless (overwritten before first use). This keeps the AsmPrinter
  `.gct` emission unchanged (it still reads the real initializer), so the GCT
  metadata and its existing consumer tests are untouched.
- **Weak default in `start.S` (not "always emit from compiler").** Always
  emitting `__capstone_cap_init` from every module would collide at link in
  multi-module domains; the weak-default + strong-override avoids that for the
  common (single cap-global module) case.
- **Const placement (former Stage 3) is moot on the domain.** A `const` capability
  table goes to `.rodata`, but the domain maps `.rodata` writable at runtime, so
  the in-place `stc` succeeds (proven: the `const`-qualified `fail_str_*` cases
  pass). Only a future hosted/`mprotect` target would need writable re-sectioning.

## Remaining work

- **Multi-module**: if capability globals live in >1 linked object, the two
  strong `__capstone_cap_init` definitions collide. Generalize to a PC-relative
  **offset table** (`.capstone_init`, KEEP'd, bounded by `__capstone_init_*`)
  with `start.S` iterating it. Not needed for current single-module domains
  (incl. `dtoa nums[]`).
- **Hosted**: a hosted crt must call `__capstone_cap_init` (only `start.S` does
  today); harmless dead symbol otherwise.

## Important caveat

Resolving capability-global tagging fixes blocker #1 only. BEEBS `dtoa` also has
blocker #2 — an untagged-base capability load (`helper_cslcc`) in David Gay's
`char[]` bigint arena on complex inputs — which is independent and would remain.
So this work is **general capability-globals infrastructure**; it resolves
`dtoa`'s blocker #1 (the `nums[]` shape is the validated array case) but does not
by itself make `dtoa` pass.
