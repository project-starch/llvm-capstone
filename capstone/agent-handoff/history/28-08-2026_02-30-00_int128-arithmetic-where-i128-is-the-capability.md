# `unsigned __int128` asserts in the legalizer, because i128 is the capability

Found 2026-08-28, bringing mruby up. The second compiler defect mruby surfaced;
the first is in
`28-08-2026_02-00-00_register-allocator-null-deref-on-disjoint-classes.md`.

## The crash

```
clang: llvm/lib/CodeGen/SelectionDAG/LegalizeDAG.cpp:1352:
  Assertion `(Res.getValueType() == Node->getValueType(0) ||
              Node->getValueType(0) == MVT::Glue) &&
             "Type mismatch for custom legalized operation"' failed.

Running pass 'Capstone DAG->DAG Pattern Instruction Selection' on function '@mpz_gcd'
```

A **custom** legalisation in our backend returns a value whose type is not the
node's type. The assertion is generic; the custom hook is ours.

## Reproducer

`mrbgems/mruby-bigint/core/bigint.c` from mruby master, alone -- about 5000 lines
rather than the 98776-line amalgamation clang saved. The command is in the build
script's flag set; `-O1 -mllvm -capstone-gp-captable` is enough.

## Why this file and not others

`bigint.c` is built on `unsigned __int128`:

```c
unsigned __int128 acc = 0;
acc += (unsigned __int128)rp[i] + (unsigned __int128)s1p[i] * (unsigned __int128)limb;
```

On this target **i128 is the capability width**. `CapstoneISelLowering.cpp:205`
makes `ISD::BITCAST` on `MVT::i128` Custom for exactly that reason, and the
neighbouring recent work ("AS200 carries a 64-bit address inside a 128-bit pointer",
"Give the i128 logical lowering a case for constants") is in the same area. So a
program doing ordinary 128-bit *integer* arithmetic collides with the type the
target reserves for capabilities.

That is the interesting shape of it: not a bug about capabilities, but about what
happens to code that wanted a wide integer on a machine where that width means
something else.

## Not fixed, and why

The nested-allocator corpus measures mruby's GC and its object heap. Multi-precision
integers are not part of any specimen, and 64-bit integers cover all of them. So the
gem is dropped from the gembox in
`benchmarks/mruby/mruby_build_config_capstone.rb`, with the reason written beside
the line, and the defect is recorded here rather than worked around silently.

Fixing i128 arithmetic on a capability-width integer type is a separate piece of
work with its own scope: every custom hook that can see an i128 has to agree on
whether it is an integer or a capability, and the answer probably differs per
operation.

## What would settle it

The assertion names neither the node nor the opcode. The next step is to build with
`-mllvm -print-before=finalize-isel` or to instrument `LegalizeOp` to print
`Node->getOpcode()` before the check, run it on `bigint.c`, and read off which
custom hook returns the wrong type. That is cheap and has not been done.
