# Static capability globals metadata contract note

This directory first introduced the **policy-neutral descriptor idea** for
runtime-side materialization of static/global capability-bearing objects.

At that stage, the important point was conceptual:

- an object needs raw template bytes,
- it needs a list of capability-bearing fields,
- and each such field needs enough target information for runtime repair.

The matching reduced C header in this directory captures that compact runtime-side
shape:

- `metadata_contract.h`

## What this note means now

The newer standalone reduced bundle under:

- `../static-cap-typed-load-repro/`

has since narrowed the blocker further and fixed the next missing layer:

- an exact **compiler-emitted** metadata contract for the reduced one-slot cases.

See there:

- `../static-cap-typed-load-repro/llvm_emitted_metadata_layout.h`
- `../static-cap-typed-load-repro/llvm_emitted_metadata_contract.md`

So the relationship is now:

1. this directory records the original runtime-side descriptor concept,
2. the standalone typed-load bundle records the narrowed reproducer,
3. the standalone typed-load bundle now also records the first exact candidate
   section/layout contract that an LLVM-side emission POC should target.

## Minimal shape retained here

Even with that newer exact contract, the original three logical ingredients stay
the same:

### 1. Object descriptors
- object id
- object size
- object alignment
- template byte location
- slot range

### 2. Slot descriptors
- owning object id
- field offset inside the object
- slot kind
- target payload for that kind

### 3. Raw template bytes
- the non-live object image copied into writable storage before slot repair

## Why this older note is still useful

This note remains the right place to remember the architecture-neutral idea:

> runtime-side repair needs object descriptors + slot descriptors + raw template
> bytes.

The newer standalone note is where the tree now fixes the exact emitted section
layout for the first LLVM-path proof of concept.

