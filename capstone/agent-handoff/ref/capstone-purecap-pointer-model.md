# Capstone PureCap pointer model

Short reference for onboarding developers and agents. This describes the current
validated benchmark/runtime path, not a complete language-lawyer specification.

## Core rule

In Capstone PureCap domain code, an ordinary C pointer is intended to be compiled
as a capability pointer from the start. It is not generally created by converting
a raw integer address into a capability at the point of use.

In LLVM IR this is represented as:

```llvm
ptr addrspace(200)
```

The current data layout uses 128-bit pointers for address space 200.

## Where capability authority comes from

Capability authority must come from an existing valid capability:

- stack pointers are derived from the runtime-provided stack capability;
- globals and functions are derived from a program/global capability, currently
  materialized through `gp` plus a PC-relative offset;
- shared HostCall/runtime buffers are provided by the runtime as region
  capabilities;
- normal pointer arithmetic such as `p + offset` derives a new pointer from the
  original valid pointer.

Authority is not restored just because an integer contains address-looking bits.

## Bounds, permissions, tag, and cursor

Ordinary C pointer operations usually inherit metadata from the source capability:

- **tag**: valid capability state; lost if a capability is stored/loaded as a
  scalar instead of with capability load/store instructions;
- **bounds**: generally inherited from the source capability, such as the stack
  frame, program/global area, or shared region;
- **permissions**: inherited from the source capability, or assigned by runtime
  sharing operations for shared regions;
- **cursor**: adjusted by capability arithmetic such as `cincoffset`.

The compiler narrows bounds for *selected* object materializations, not yet for
every C object. As of the C1 work it emits `SHRINK` to the object size for common
sized global materializations (`selectLGA`, `-capstone-shrink-globals`, default
on), two benchmark allocators (`rv8_malloc.c` and dtoa, `cap_shrink`), and
fixed-size stack frame objects (`-capstone-shrink-stack`, opt-in). Coverage is
partial: subobjects, unions, non-zero global offsets, function/code capabilities,
dynamic `alloca`/varargs, and the broad `gp`/`sp` roots themselves are not
narrowed, so ordinary pointer expressions should still be understood as
derivations from an existing (often broad) capability. See
`../design/granularity-provenance-discussion.md` and the `2026-06-29` audit
(`../history/29-06-2026_15-08-22_granularity-provenance-audit.md`) for the
coverage matrix and the open items.

## Loads, stores, and tags

Capability values must be copied with capability operations:

- `ldc` loads a capability and preserves its tag;
- `stc` stores a capability and preserves its tag.

Scalar operations such as `ld`/`sd` move integer bits. They do not preserve a
capability tag. If a pointer is accidentally stored with `sd` and reloaded with
`ld`, later dereference is expected to fail because the base register is no
longer a valid tagged capability.

Scalar data loads/stores may still use ordinary instructions such as `lw`, `sw`,
`ld`, or `sd`, but their base register must be a valid tagged capability.

## Integer/pointer casts

Pointer-to-integer casts may preserve numeric address bits, but they do not
preserve provenance or dereference authority.

Integer-to-pointer casts may produce a pointer-shaped value. Unless the
implementation has a specific valid provenance-restoration rule, the resulting
pointer must not be assumed safe to dereference or call through.

Safe pattern:

```c
int *q = p + 3;
```

Unsafe pattern:

```c
uintptr_t raw = (uintptr_t)p + 12;
int *q = (int *)raw;
return *q;  /* not generally valid on Capstone PureCap */
```

For benchmark bring-up, do not fix failures by fabricating authority from raw
integers. Keep memory accesses derived from valid capabilities, or add a focused
backend/runtime test if a real benchmark exposes this pattern.

## Linear capability note

Capstone capabilities are linear by default in the current path. Some capability
arithmetic forms, such as `cincoffset rd, rs1, rs2` with `rd != rs1`, can consume
the source linear capability in `rs1`.

Benchmark-local wrappers sometimes use `CAPSTONE_DELIN` before repeated pointer
derivations. This does not create authority. It changes the linear/non-linear
usage mode of an already valid capability so the same base can be reused safely.

