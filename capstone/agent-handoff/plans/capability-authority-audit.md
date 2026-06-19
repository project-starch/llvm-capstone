# Capability authority audit plan

This report consolidates the concern that Capstone must not allow programs to
manufacture capability authority from integer data, and turns it into an audit
track with concrete next steps.

## Summary

Capstone's security model relies on capability provenance: memory authority must
come from an existing tagged capability, not from address-looking integer bits.
The hardware tag model should prevent the simple failure mode where a program
casts an integer to a pointer and dereferences it as if it carried authority.

The implementation is not yet security-audited end to end. The compiler uses
`i128` as the carrier for address-space-200 pointers, while ordinary widened
integers can also become `i128`. That shared carrier is useful during bring-up,
but it is exactly where mistakes can accidentally reinterpret scalar data as a
capability base. Treat this as a first-class compiler/runtime audit track,
separate from BEEBS benchmark bring-up.

The policy target is:

- valid capability constructors are explicitly enumerated;
- scalar integer bits never become dereference or call authority by accident;
- capability tags are preserved only across capability copies (`ldc`/`stc`) and
  intentionally lost across scalar copies;
- backend and runtime tests cover common laundering paths, not only direct
  `inttoptr`.

## Motivation

The browser/interpreter concern is real. Modern sandbox escapes often start from
a logical type confusion, not from a direct spatial memory bug. A runtime may
store integers and raw pointers/capabilities in similarly shaped object slots
and rely on isolated type metadata to decide how to interpret those slots. If a
metadata bug lets attacker-controlled code read an integer as a pointer, or a
pointer as an integer, it can create exploit primitives.

The referenced CODE BLUE 2024 V8/Wasm talk shows this concretely: a WasmGC type
confusion path can produce primitives analogous to `addrOf`, `fakeObj`, and
caged arbitrary read/write. That is not the same architecture as Capstone, but
it is the right threat class to keep in mind:

<https://archive.codeblue.jp/2024/files/cb24-U25_WebAssembly_Is_All_You_Need-Exploiting_Chrome_and_the_V8_Sandbox_10_times_with_WASM_by_Seunghyun_Lee.pdf>

## Current Capstone position

The local pointer model already states the correct invariant:

- ordinary C pointers in domain code are intended to be `ptr addrspace(200)`;
- authority comes from stack, globals/functions, shared runtime regions,
  capability loads, or derivation from an existing capability;
- integer-to-pointer casts may create pointer-shaped values, but must not restore
  authority;
- scalar `ld`/`sd` copies do not preserve capability tags, while `ldc`/`stc` do.

Relevant local reference:

- `capstone/agent-handoff/ref/capstone-purecap-pointer-model.md`
- `capstone/agent-handoff/plans/backend-compiler-fixes.md`

The backend already has some targeted protections. In
`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp`,
`isCapstoneCapabilityValue()` recognizes only definite capability-producing
forms such as full-width non-extending `i128` loads, `CIncOffset`, and `LGA`.
The `ADD` lowering also intentionally excludes `CopyFromReg i128` from the
"definitely capability" set because scalar integers may live in `i128`
registers. This is the right direction, but it is a heuristic, not yet a
complete proof.

## Main risk

The dangerous bug is not "integer and pointer casts exist" by itself. The
dangerous bug is any lowering, ABI, or runtime path where a value with only
scalar provenance is later consumed as a capability base.

High-risk surfaces:

- direct `ptrtoint` / integer arithmetic / `inttoptr` / load-store-call;
- PHI and SELECT nodes that merge capability-derived and scalar-derived values;
- function arguments and returns, especially `i128` values;
- aggregate copy and struct-by-value ABI paths;
- `memcpy`, `memmove`, `memset`, and compiler-generated bulk copies;
- varargs and `va_list`;
- inline assembly boundaries;
- jump tables and indirect calls;
- laundering through memory, especially scalar stores/reloads of capability
  carriers;
- interpreter object slots that can contain either handles/integers or raw
  capabilities.

## Critical review of proposed additions

The proposed additions are sound, with two refinements.

First, negative tests for "almost capabilities" are necessary, but should not
all be compile-time rejects. C permits many casts syntactically. For Capstone,
the expected result may be either a compiler diagnostic for a deliberately
unsupported construct or a runtime tag fault when the invalid pointer is used.
The test should state which behavior is expected for each case.

Second, a full provenance lattice is useful as an audit model, but implementing
it as a complete compiler dataflow verifier may be too broad for the next patch.
Start with an explicit authority-constructor inventory and focused MIR/SDAG
assertions or lit checks for the known dangerous consumers. Promote that into a
real verifier only after the first tests expose the practical shape of the
problem.

The interpreter threat model item is important and should be documented before
claiming language-runtime isolation. Hardware tags prevent integer bits from
becoming new authority, but they do not fix a runtime that stores powerful real
capabilities inside attacker-controlled object memory and then lets a type
confusion select them.

## Authority constructor policy

The intended legal constructors are:

- runtime-provided stack capability;
- materialized global/function/program capability;
- imported shared-region capability from the trusted runtime boundary;
- `ldc` from memory that actually contains a tagged capability;
- capability arithmetic derived from an existing tagged capability, such as
  `cincoffset`;
- explicit, reviewed runtime intrinsics that manipulate capability metadata
  without inventing new authority.

Everything else is scalar unless proven otherwise. In particular:

- `ptrtoint` preserves cursor bits only, not authority;
- scalar arithmetic on those bits remains scalar;
- `inttoptr` does not regain tag, bounds, permissions, or provenance;
- scalar stores/reloads of capability-shaped values must not preserve tags;
- `i128` type alone is not proof of capability provenance.

## Proposed provenance lattice

Use this as the audit vocabulary:

- `Unknown`: provenance not established.
- `ScalarOnly`: integer or untagged value; may contain address-looking bits.
- `LostTag`: value originated from a capability but crossed a scalar copy or
  other tag-destroying boundary.
- `CapabilityLoadedTagged`: value loaded by `ldc` from tagged capability memory.
- `StackDerived`: value derived from the runtime-provided stack capability.
- `GlobalMaterialized`: value derived from a program/global/function capability.
- `SharedRegionImported`: value imported from the trusted HostCall/runtime
  boundary.
- `CapabilityDerived`: value derived from another valid capability by permitted
  capability operations.

Forbidden transitions:

- `ScalarOnly` -> `CapabilityDerived` without an explicit legal constructor;
- `LostTag` -> `CapabilityLoadedTagged` unless reloaded by `ldc` from memory
  that still has a valid tag;
- `Unknown` -> capability-consuming instruction without a local proof;
- mixed PHI/SELECT result consumed as capability unless all incoming values are
  valid capabilities under the same authority rule.

## Next-step plan

### 1. Write the authority-constructor inventory

Create a short design/reference file that lists every legal capability authority
creation site in the current compiler/runtime path and points to the relevant
code. Start with stack, globals/functions, shared-region imports, `ldc`, `LGA`,
and `CIncOffset`.

Expected output:

- new or updated handoff reference doc;
- short list of code locations that may legally create or preserve tags;
- explicit statement that `inttoptr` is not a constructor.

### 2. Add direct negative tests

Add focused tests for integer-to-pointer use where the integer contains
address-like bits but no tag/provenance.

Coverage should include:

- direct `uintptr_t` -> pointer -> load/store;
- pointer -> integer -> arithmetic -> pointer -> load/store;
- pointer -> integer -> pointer -> indirect call, if representable;
- constant integer -> pointer -> load/store.

Expected behavior:

- either compile-time rejection for constructs the target chooses not to support,
  or runtime trap/fault when dereferenced;
- no codegen path may produce `ldc`/`stc`/`cincoffset` with a scalar-only base as
  if authority existed.

### 3. Add laundering tests

Test that capability authority cannot be laundered through representation
changes. Cover both compiler IR and runtime behavior where possible.

Required patterns:

- capability -> `ptrtoint`/`i128` -> scalar memory -> reload -> pointer use;
- capability-containing aggregate -> scalar or by-value copy -> pointer use;
- capability -> PHI -> dereference, with one scalar-only incoming value;
- capability -> SELECT -> dereference, with one scalar-only arm;
- capability through function return and argument passing;
- capability through `va_list`;
- capability across inline asm boundary.

For each test, define whether the expected result is a compiler diagnostic,
SelectionDAG/MIR pattern, or runtime tag fault.

### 4. Audit backend capability consumers

Audit every lowering path where an `i128` value is interpreted as a capability
base or preserved as a tagged capability.

Initial target list:

- `ADD`, `SUB`, `SETCC`, `SELECT`, PHI-related lowering;
- loads/stores and aggregate copy;
- calls, returns, indirect calls, and tail calls;
- `va_start` / `va_arg` / `va_list`;
- libcalls and compiler-generated `memcpy`/`memmove`/`memset`;
- jump tables.

For each site, classify inputs with the lattice above. Add narrowly scoped
assertions, comments, or lit checks where the code currently depends on a
heuristic such as "this `i128` load is definitely an `ldc`".

### 5. Fix known high-value compiler bugs in this area

Prioritize currently known issues that are already symptoms of inconsistent
pointer/scalar representation:

- capability pointer/scalar arithmetic regressions similar to the fixed `miniz`
  ptrdiff and `or disjoint` cases;
- `va_list` storing an argument pointer through scalar `sd` rather than
  capability `stc`;
- compiler-generated bulk copy choosing `stc` for pure integer data or scalar
  copy for capability-bearing data;
- jump-table lowering that uses scalar table addresses where capability bases
  are required.

Do not solve these by benchmark-local source workarounds unless the purpose is
only to keep an unrelated benchmark moving. Root fixes should live in Clang,
SelectionDAG, MIR lowering, or runtime ABI code as appropriate.

### 6. Define the interpreter/sandbox threat model

Write a small design note for language runtimes/interpreters on Capstone:

- where object slots live;
- whether object slots may contain tagged capabilities;
- who can write those slots;
- which capabilities are too powerful to store in attacker-controlled memory;
- whether handles/indices and raw capabilities can be confused;
- which metadata is part of the TCB;
- what happens when a logical type-confusion bug reads a slot under the wrong
  type.

Recommended default: attacker-controlled object memory stores handles or indices,
not raw powerful capabilities. Capability tables and type metadata should live
outside the attacker-controlled cage.

### 7. Add a lightweight verifier/checker

After the first tests and audit identify the actual compiler patterns, add a
lightweight checker in the most practical layer:

- SelectionDAG debug assertion for `CIncOffset` and capability memory-operation
  bases;
- MIR verifier extension for machine instructions that require tagged
  capability operands;
- or a lit-driven MIR/codegen test suite if an always-on verifier is too broad.

The first version does not need whole-program provenance. It should catch
obvious `ScalarOnly` -> capability-consuming transitions in the backend.

## Delegation prompt

Use this prompt for a coding agent:

```text
Work on the Capstone capability-authority audit track.

Start by reading:
- capstone/agent-handoff/README.md
- capstone/agent-handoff/state/current-state.md
- capstone/agent-handoff/state/current-next-step.md
- capstone/agent-handoff/ref/capstone-purecap-pointer-model.md
- capstone/agent-handoff/plans/capability-authority-audit.md

Task:
Add the first focused negative tests proving that integer bits do not become
capability authority on Capstone. Prefer a small compiler/runtime test set over
a broad rewrite.

Focus on:
- direct uintptr_t/inttoptr-like pointer fabrication;
- pointer -> integer -> scalar memory -> pointer laundering;
- PHI or SELECT with one valid capability arm and one scalar-only arm;
- one function return or argument laundering case if it is easy to isolate.

For each case, decide and document the expected behavior: compile-time reject,
codegen pattern check, or runtime tag fault. Do not add source workarounds that
make fabricated pointers valid. Do not claim security completeness.

Before any build/test command, run:
source capstone/tests/capstone-test-env.sh

Useful gates:
"$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone

If you add runtime tests, also run the specific runtime script and at least:
bash capstone/tests/runtime-qemu/run-coremark.sh

Update capstone/agent-handoff/plans/capability-authority-audit.md with what was
covered and what remains. Commit only coherent source/test/docs changes. Do not
commit debug notes, temporary logs, or unrelated dirty files.
```

## Success criteria

The audit track is useful when:

- there is an explicit list of legal authority constructors;
- direct `inttoptr` and common laundering paths have negative coverage;
- backend capability consumers no longer rely on ambiguous `i128` shape alone;
- known scalar/capability ABI bugs have focused tests before fixes;
- interpreter designs have a written rule against placing broad raw
  capabilities in attacker-controlled object slots.

Until then, Capstone can reasonably claim that the hardware tag model is aimed at
preventing integer-to-authority fabrication, but it should not claim complete
protection against interpreter type-confusion or compiler laundering bugs.
