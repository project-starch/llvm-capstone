# Fuzz findings — index

One folder per finding from the llvm-stress / csmith campaigns (`tests/fuzz/`), each with a
README, the reduced reproducer and its crash signature. `known-signatures.txt` in the parent dir
lists the signatures the campaign runner treats as already filed; a fixed finding loses its line
there and keeps its folder.

| folder | status (2026-09-05 sweep) |
|---|---|
| `F01-vector-elt-pointer-zext` | FIXED as C-39 (cycle 1) |
| `F02-vector-elt-load-recreated` | FIXED 2026-09-05 — `getAddressSpaceForPseudoSourceKind` (target) + `InferPointerInfo` keeps the capability address space on a value-less pointer info (shared code, manifest updated); pinned by `fuzz-f02-f03-vector-elt-stack-temp.ll` |
| `F03-vector-elt-store-recreated` | FIXED 2026-09-05 with F02 (same root cause, same pin) |
| `F04-csmith7-O0-wedge` | RETRACTED as a compiler finding 2026-09-05; the phenomenon is Q-03 (a domain wedges by position in the boot). Reproduced again in the sweep: 3 wedges of 34 items (`cs7-O0` at both its positions, `cs2-O2` at one of two) |
