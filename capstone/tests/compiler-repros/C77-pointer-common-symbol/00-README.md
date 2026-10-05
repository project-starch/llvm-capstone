# C-77 — a pointer-typed common symbol crashes the capstone64 backend

**OPEN — COMPILER**, observed 2026-10-05 with the toolchain `compiler-stack-68c75ed3`
(68c75ed3e1ad46767cce002a434917fb7b9d16fa) while building the pointer benchmarks of the
llvm-test-suite (Olden, Ptrdist, MallocBench) for `capstone/experiments/revnode-cache`.

## What happens

A file-scope variable of pointer type without an initializer, compiled with `-fcommon`:

    char *shared_name;

stops clang in code generation:

    UNREACHABLE executed at llvm/lib/CodeGen/TargetLoweringObjectFileImpl.cpp:631!
    ... getELFSectionNameForGlobal ... selectELFSectionForGlobal ...

`getSectionPrefixForGlobal` handles Text, ReadOnly, BSS, ThreadData, ThreadBSS, Data and
ReadOnlyWithRel and falls through for every other section kind; a common symbol
(`SectionKind::Common`) is one of the kinds left. An integer common symbol (`int n;`) compiles,
so the backend treats a common capability-holding global differently from a common integer,
and the capability path reaches the unique-section naming the generic code cannot name.

Seven of seventeen benchmark programs stopped on it: espresso (getopt.c: `char *optarg;`),
cfrac, anagram, bc, yacr2, bh and voronoi. `-fno-common` compiles all of them; bh then needs
`-Wl,--allow-multiple-definition`, since it defines one variable in several files.

## Reproducer

`./run.sh` compiles `src/common_ptr.c` with `-fcommon` and reports PRESENT when the
signature appears; two controls must compile first (`-fno-common`, and an integer common
symbol under `-fcommon`). Exit 1 = PRESENT, 0 = ABSENT, 2 = a control failed.

## Workaround

Build with `-fno-common`, the default of GCC 10 and clang 11 and later. Programs that rely on
common symbols to share a variable defined in several files link with
`-Wl,--allow-multiple-definition` (bh), which the SDK's `capstone-cc` passes through.

## Impact

Any C of the pre-2020 style built with `-fcommon` and holding a pointer at file scope. No port
in `capstone/ports/` is affected today: they build with the compiler's default (`-fno-common`).
