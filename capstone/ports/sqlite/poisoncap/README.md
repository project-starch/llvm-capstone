# PoisonCap's published SQLite memsys5, against Sublet

A side comparison, not an arm of the allocator study. PoisonCap's memsys5 and C library run as published,
and the SQLite amalgamation takes only the two source fixes every arm compiles (`corpus/build-cases.py`,
`FIXES`). Everything here is the harness around it.

| folder | what | results |
|---|---|---|
| `platform/` | their platform (LLVM, QEMU, CheriBSD) built in a pinned container, the NODEBUG kernel with their FPGA kernel's options, the guest every runner boots (`poisoncap_guest.py`), a smoke test | — |
| `corpus/` | the SQLite engine corpus's temporal cases on their memsys5: build, run, score | `bug-corpora/sqlite/engine-repros/results/2026-10-24-poisoncap/` (1 MiB heap, the first comparison), `2026-10-26-poisoncap-heap256/` (their own configuration, the current result), `2026-10-27-poisoncap-libc/` (C library revocation on, stopped) |
| `backport/` | a backported lookaside use-after-free, not a corpus case; proven natively, platform runs not made | — |
| `speedtest1/` | their own benchmark, built and run as published | `results/2026-10-25-speedtest1/` |
| `reuse/` | reuse distance at memsys5 on both systems: instrument, qualification, runs, plots | `results/2026-10-25-reuse-pilot/`, `results/2026-10-25-reuse-size100/`, `results/2026-10-27-reuse-block-port/` |

Shared with the Sublet side: `../sublet/` (the Sublet arms) and `../repro322/` (the corpus build and runner).
