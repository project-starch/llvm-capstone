# PoisonCap's published SQLite memsys5, against Sublet

A side comparison, not an arm of the allocator study. PoisonCap's code runs as published; everything here
is the harness around it.

| folder | what | results |
|---|---|---|
| `platform/` | their platform (LLVM, QEMU, CheriBSD) built in a pinned container, the NODEBUG kernel with their FPGA kernel's options, a smoke test | — |
| `corpus/` | the SQLite engine corpus's temporal cases on their memsys5: build, run, score | `bug-corpora/sqlite/engine-repros/results/2026-10-24-poisoncap/` |
| `speedtest1/` | their own benchmark, built and run as published | `results/2026-10-25-speedtest1/` |
| `reuse/` | reuse distance at memsys5 on both systems: instrument, qualification, runs, plots | `results/2026-10-25-reuse-pilot/`, `results/2026-10-25-reuse-size100/` |

Shared with the Sublet side: `../sublet/` (the Sublet arms) and `../repro322/` (the corpus build and runner).
