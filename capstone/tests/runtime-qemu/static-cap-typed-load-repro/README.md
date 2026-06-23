# Static capability typed-load standalone repro

This directory is a **standalone reduced diagnostic** for the current LLVM-generated
Capstone issue around loading capability-valued fields from file-scope static
objects.

## What this isolates

The important trigger is narrower than just "static const exists":

- simple direct use may optimize away and succeed,
- but a **runtime capability-typed load** from a file-scope static object still fails.

These two one-field repros isolate that behavior:

- `fail_fn_struct_load.c`
  - one-field struct with a function-pointer field,
  - takes the address of the static object and forces a runtime load via
    `const volatile struct holder *p = &kHolder;`,
  - currently expected to fail with:
    - `[CAPSTONE] cs.cjalr requires capability in rs1`

- `fail_str_struct_load.c`
  - one-field struct with a string-pointer field,
  - likewise forces a runtime load from static object memory,
  - currently expected to fail with:
    - `[CAPSTONE] Cap mem access requires capability`

## Minimal runtime-side fixes in the same bundle

Two pairs of positive cases show the smallest working workaround currently known:
**materialize the capability-valued field at runtime into writable storage**.

- `fix_fn_runtime_materialize.c`
  - assigns `gHolder.fn = helper;` before use,
  - expected result: `305419896`

- `fix_str_runtime_materialize.c`
  - assigns `gHolder.name = "ok";` before use,
  - expected result: `111`

- `descriptor_fn_runtime_materialize.c`
  - uses a tiny object descriptor plus one function-slot descriptor,
  - copies a raw template into writable storage, then patches the slot with a
    live function capability,
  - expected result: `305419896`

- `descriptor_str_runtime_materialize.c`
  - uses a tiny object descriptor plus one string-slot descriptor,
  - copies raw templates for both the holder object and the backing string,
    then patches the holder slot with a live pointer to the materialized string,
  - expected result: `111`

## Prototype bridge toward compiler-emitted metadata

The next step in this directory is a tiny **automatic bridge from LLVM IR to
runtime-materialization source** for the same reduced one-slot cases.

- `generate_runtime_materialize_from_ir.py`
  - reads the LLVM IR for `fail_fn_struct_load.c` or `fail_str_struct_load.c`,
  - extracts the reduced static object shape plus its one capability-valued target,
  - emits a standalone descriptor-driven materialization domain source.

- generated positive cases
  - `autogen_fn_runtime_materialize.dom`
  - `autogen_str_runtime_materialize.dom`
  - both are built by `build.sh` from generated C sources under the temporary
    output directory,
  - expected results stay `305419896` and `111`.

This does **not** solve the general compiler/runtime problem. It demonstrates the
current minimal end-to-end workaround shape:

1. do not rely on capability-typed fields surviving in static image data,
2. rebuild those fields at runtime from live capability values,
3. then use the rebuilt writable object.

The descriptor-driven variants are the next step beyond the manual fixes:

1. keep a tiny policy-neutral metadata shape for the object and its capability slots,
2. rebuild from raw templates plus descriptors at runtime,
3. use that as the bridge toward a future general LLVM-emitted path.

The LLVM-IR-generated variants are one step further again:

1. let the compiler produce the reduced object/target information in LLVM IR,
2. extract a minimal metadata/template view from that IR automatically,
3. emit the same runtime-materialization shape without hand-writing the descriptors.

## Fixed minimal LLVM-emitted contract for the next in-tree step

This directory now also records the exact **candidate compiler-emitted metadata
layout** for the first LLVM-path proof of concept:

- `llvm_emitted_metadata_layout.h`
  - concrete record layout for a proposed `.gct` section,
  - section header + object descriptors + slot descriptors + raw template bytes
- `llvm_emitted_metadata_contract.md`
  - explains the intended semantics,
  - maps the reduced function/string cases onto that emitted shape,
  - captures the recommended next compiler-side POC scope.

## First in-tree compiler-side POC now present

The current LLVM Capstone backend now includes a first local proof of concept:

- the reduced failing one-slot cases cause the backend to emit a **non-empty**
  `.gct` section,
- that section carries:
  - `SCAP` header metadata,
  - emitted object descriptors,
  - emitted slot descriptors,
  - raw template bytes.

This POC does **not** yet make the runtime consume `.gct` automatically.
It proves the compiler-side emission half of the contract for the narrowed cases.

### Array holders now emitted (2026-06-24)

`collectStaticCapReducedObject` (`llvm/lib/Target/Capstone/CapstoneAsmPrinter.cpp`)
was extended from the one-field-struct holder to also reduce an **array of
addrspace(200) capability pointers** (`const char *tbl[]` / a function-pointer
table).  The array yields one holder object with `NumSlots == N`, one slot per
element (string or function), and one target object per distinct string, with
correct per-target template offsets.  The single-field struct case is emitted
byte-identically to before (the repro's `consume_emitted_gct_string_domain`
still passes).  Regression coverage: `llvm/test/CodeGen/Capstone/static-cap-gct-array.ll`.

This unblocks the *emission* side for string tables such as BEEBS `dtoa`'s
`char *nums[]`, whose pointers otherwise load untagged.  See "Remaining runtime
half" below for what is still needed to make such a table usable at runtime.

### Resolution path chosen: constructor-codegen (decided 2026-06-24)

Two architectures were on the table for making ordinary code that accesses the
*original* global (`tbl[i]`) see tagged capabilities:

1. **GCT metadata + general runtime consumer** — generalize the consumer POC and
   add a relocated holder back-reference to the format so it can patch the
   original global in place. More moving parts (format change, a runtime `.gct`
   walker, and hand-written capability-derivation for the in-place store).
2. **Constructor-codegen** — have the compiler emit per-module init code that
   stores each capability global at runtime with ordinary C stores
   (`tbl[i] = "...";`); normal codegen materializes a tagged, bounded capability
   and `scc`-stores it in place.

**Decision: constructor-codegen.** It is simpler and lower-risk (no metadata
parser, no hand-written cap-derivation asm, in place by construction). The
compiler-emitted `.gct` metadata is kept as an inspectable description / possible
data source, but the consumer of record becomes compiler-emitted init code.
Rationale and remaining productionization steps:
`capstone/agent-handoff/design/capability-globals-init-decision.md`.

This was validated end-to-end by the array-shaped cases below.

### Array-shaped (`dtoa nums[]`-style) cases

- `fail_str_array_load.c` — statically-initialized `const char *kTable[3]`, read
  at runtime → **faults** (`Cap mem access requires capability`): the array form
  of the blocker.
- `fix_str_array_runtime_materialize.c` — non-const array, each element assigned
  at runtime (`gTable[i] = "..."`), read back through the array → **succeeds**,
  returns `336` ('o'+'h'+'y' first chars).

Together these prove the runtime semantics the chosen path depends on: an ordinary
C store of a literal/global address into a writable capability global produces a
**tagged** slot that later normal loads + dereferences use successfully — in place,
no metadata parsing, no cap-derivation asm.

## First runtime-side consumer POC for emitted `.gct`

This directory now also includes the first reduced domain that **reads the
compiler-emitted `.gct` section at runtime** and uses it to rebuild a live
writable object:

- `consume_emitted_gct_string_domain.c`
  - triggers backend `.gct` emission through a reduced failing-style static object,
  - walks `__capstone_gct_start .. __capstone_gct_end`,
  - validates the `SCAP` header,
  - copies the emitted raw template bytes,
  - patches the reduced string/object capability slot,
  - expected result: `111`.

For now, this first end-to-end consumer POC is intentionally limited to the
**string/object slot** case.
The reduced function-slot case still needs a clean runtime rule for turning the
emitted relocated symbol reference into a live function capability.

## Run

```bash
bash /home/alexey/dev/llvm-capstone/capstone/tests/runtime-qemu/static-cap-typed-load-repro/run.sh

bash /home/alexey/dev/llvm-capstone/capstone/tests/runtime-qemu/static-cap-typed-load-repro/inspect-gct-emission.sh
```

## Files

- `metadata_contract.h` — local minimal descriptor shape for one reduced object and
  its capability-bearing slots
- `llvm_emitted_metadata_layout.h` — exact candidate layout for the first
  compiler-emitted `.gct` metadata records
- `llvm_emitted_metadata_contract.md` — design note fixing the minimal LLVM/runtime
  contract for the reduced one-slot cases
- `runtime_materialize_helpers.h` — tiny helper layer for template copying, slot
  patching, and zeroing
- `generate_runtime_materialize_from_ir.py` — prototype automatic extractor that
  lowers reduced LLVM IR into a generated descriptor-driven materialization domain
- `inspect-gct-emission.sh` — builds the reduced domains and verifies that the
  first backend-side `.gct` emission POC produces non-empty `SCAP` metadata for
  the narrowed failing static-cap cases
- `build.sh` — builds the reduced domain binaries, including the generated ones
  and the first emitted-`.gct` consumer domain
- `run.sh` — builds, runs, and checks both expected failures, both manual fixes,
  both descriptor-driven fixes, both LLVM-IR-generated fixes, and the first
  emitted-`.gct` consumer fix for the string/object case



