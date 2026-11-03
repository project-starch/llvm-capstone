# Decision: how to tag capability globals (constructor-codegen vs GCT-consumer)

Status: constructor path implemented in 2026-06; virtual table path qualified
on `compiler/cap-init-table` in 2026-11. The sections after the table contract
record the original constructor decision and its historical validation.

## Virtual image-gp table path

The virtual image-gp ABI can replace the per-leaf constructor code with scalar
records and `__capstone_cap_init_table_v1` in `runtime/virtual/start.S`. Each
module retains a small constructor wrapper in the existing PC-relative
`.capstone_cap_init` call table. The helper symbol is required at link time,
so objects using this path require an SDK with that helper. The physical and
gp-captable ABIs retain constructor stores. The diagnostic cap-init limit and
print options also retain that path and its original leaf order.

Records occupy the read-only `.capstone_cap_relocs` section, grouped for
sequential reads. Each is four little-endian 32-bit words, aligned to four bytes:

| Word | Meaning |
| --- | --- |
| 0 | Signed displacement from the record to the destination capability slot |
| 1 | Signed displacement from the record to the target symbol |
| 2 | Object size in bytes; zero preserves the image capability's bounds |
| 3 | Signed cursor offset, applied after narrowing |

The compiler converts initialization after GlobalMerge, preserving the merged
containers and offsets that normal lowering sees. Sized data globals use their
allocation size when global narrowing is enabled. Functions, block addresses,
aliases and zero-size/incomplete data use zero. Nulls need no record. TLS
holders with capability initializers remain rejected; TLS targets and sizes or
cursor offsets outside the compact fields retain constructor stores. The
virtual linker script bounds the image below the signed-displacement limit.

The helper uses SCC from the explicitly granted image `gp`, SHRINK when the
size is nonzero, CINCOFFSET for the cursor and STC for the slot. It preserves
the root's non-linear type, permissions and lifetime; it does not seal,
delinearize, mint or revoke. The image-gp ABI already requires NONLIN `gp`.
Stores are still eager and holders remain writable. This reduces executable
initializer code and its spill frames, but does not avoid dirtying data pages.
The record pages themselves also contribute to residency. Lazy initialization
would need a separate page-admission and synchronization design.

`-capstone-cap-init-table=false` selects constructor code for comparison.
`static-cap-init-table*.ll` covers the encoding and fallback; the virtual
`cap-init-contract.c` compares initialized pointers against ordinary lowering,
including tags, cursors, bounds, type, permissions, aliases and labels.
After sourcing `capstone/tests/capstone-test-env.sh`, run
`python3 capstone/runtime/virtual/run-cap-init.py --sdk SDK --platform PLATFORM
--work NEW_DIRECTORY`. It checks both optimization levels, constructor mode,
and a diagnostic omitted-store control that must fail a field check.

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
2. **Wiring (multi-module safe)** — each module's `__capstone_cap_init` has
   **internal** linkage (no cross-module collision) and the AsmPrinter emits a
   **PC-relative entry** `.quad __capstone_cap_init - entry` into a
   `.capstone_cap_init` table (`CapstoneAsmPrinter::emitCapGlobalInitTableEntry`).
   `link.ld` bounds the linker-concatenated table (`__capstone_cap_init_start/_end`);
   `capstone/my_first_domain/start.S` iterates it before `domain_main`, computing
   each initializer's runtime address as `entry_runtime + offset` and calling it
   via `cincoffset gp`. PC-relative offsets are required because the domain is
   loaded at a runtime base and processes **no load-time relocations**, so a
   standard `.init_array` of absolute addresses is stale; a fixed external
   `__capstone_cap_init` symbol would also collide across modules (CoreMark).
   Empty table (no capability globals) ⇒ no-op.
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
- **Internal init + PC-relative offset table (not a fixed symbol or `.init_array`).**
  A fixed external `__capstone_cap_init` collides across modules; a standard
  `.init_array` of absolute function addresses is stale under the domain's
  no-load-time-relocation model. The internal-function + `(initfn - entry)`
  offset table is both collision-free and position-independent. (Validated on
  multi-module CoreMark, which previously regressed to a duplicate-symbol link
  error.)
- **Const placement (former Stage 3) is moot on the domain.** A `const` capability
  table goes to `.rodata`, but the domain maps `.rodata` writable at runtime, so
  the in-place `stc` succeeds (proven: the `const`-qualified `fail_str_*` cases
  pass). Only a future hosted/`mprotect` target would need writable re-sectioning.

## Remaining work

- **Multi-module**: DONE — internal init + PC-relative `.capstone_cap_init`
  offset table iterated by `start.S` (validated on CoreMark, RV8, BEEBS).
- **Hosted**: a hosted crt would need to iterate the same table (only the domain
  `start.S` does today); harmless otherwise.

## Important caveat

Resolving capability-global tagging fixed blocker #1 for BEEBS `dtoa`. The
independent arena-alignment blocker was fixed later in commit `216396dbafd4`
by building with `-DOmit_Private_Memory` and a 16-byte-aligned, 16-byte-rounded
arena allocator. Together those fixes make `run-beebs-dtoa.sh` pass end to end.
The capability-global infrastructure remains broader than `dtoa`: it is the
general mechanism for materializing initialized capability globals, pointer
tables, string tables, and function-pointer tables in domain code.
