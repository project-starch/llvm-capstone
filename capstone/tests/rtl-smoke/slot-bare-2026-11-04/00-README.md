# slot-bare-2026-11-04

What a capability copy through one stack slot costs on silicon. The SPEC lane estimated that devmk, which removes the
compiler's "stc then ldc of the same slot" pairs, saves 23-36 silicon cycles per removed pair. RETRACTED 2026-11-05: that
divided the cycle drop by half the instret drop, but devmk replaces each pair with one movc (net −1 instruction per pair);
the corrected figure is about 13.7 cycles per pair on bitstream 3 (imagick), 14-18 across the runtime lane's eval set (imagick, gcc, xz) on bitstream 3. See PREREG.md. The
simulator gives about 8 cycles (dev bf2fc1c9d6bd, `capstone/docs/history/04-11-2026_03-13-48_stc-ldc-pair-cost.md`). This kit reads the
per-op costs on the board.

The test is capstone-ariane `stc-ldc-slot.S`, the same blob (868111d4cd6e) on three trees:
- c5a-eval-ctl 3afd5c9e2, bitstream 3's tokens;
- objb-eval d463206ac, ObjBounds 603528b8b plus the test;
- c5a-eval 08c9fbd22, C.5a 1fc2cd5aa plus the test.

The test's header holds the hypothesis and the arms. Each image runs 512 iterations of `<body>; addi; bnez` and records
five values: the mcycle delta, the minstret delta, a value, the trap count and the last cause.

| image | body | what it isolates |
|---|---|---|
| slot-9 | (none) | the bare loop: the control, run first |
| slot-6 | sd A; ld A+0x10 | the plain pair, no hazard |
| slot-5 | sd A; ld A | the plain pair through one slot: the page-offset wait |
| slot-7 | ldc A | one LDC per iteration |
| slot-8 | stc A | one STC per iteration |
| slot-1 | stc A; ldc A | the compiled copy pattern |
| slot-2 | stc A; ldc A+0x10 | the pair, another doubleword |
| slot-3 | stc A; ldc A+0x40 | the pair, another cache line |
| slot-4 | stc A; ldc A+0x1000 | the pair, same page offset in another page |

Every image returns on bitstream 3, ObjBounds and C.5a, so they can run in any order. The recommended order puts slot-9
first. Each image is a single bare boot (`SUP_IMG`, `SUP_REC_ADDR` = its `.rec`, all 80008000) with the runner
`sup-bare-2026-10-03-ladder/run_sup_bare.py`.

How the kit is built and graded:
- `build.sh` rebuilds the images from `tests/`, and checks the test against `VP_SRC` when given.
- The wrapper is `hpm-bare-2026-11-03`'s `inc/`.
- `images/SHA256SUMS` pins the images.
- `PREREG.md` holds the vectors, simulated from these images on all three trees (`vectors.txt`), and the grading rule.
- `grade/slot-grade.py <run dir> <bs3|objb|c5a>` grades a run.

Sibling: `hpm-bare-2026-11-03` reads the LDC/STC issue stall with the stock counters.
