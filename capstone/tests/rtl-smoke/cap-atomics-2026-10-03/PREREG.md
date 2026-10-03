# Atomics through a capability address on silicon -- pre-registered 2026-10-03, committed and pushed BEFORE the boot

Part 4a of the plan to run memcached on the board, done before any memcached work starts.
- **What memcached needs.** memcached.dom (mc19-level0, the QEMU SDK image) links 171 atomics: 80 `lr.w`, 80 `sc.w`
  (musl's `a_cas` loops), 4 `amoadd.w`, 5 `amoadd.d` and 2 `amoswap.w`. If they fail on silicon, every musl lock
  fails.
- **Nothing covers them on silicon yet.** No image proven there contains any atomic: speedtest1 e6ee5255 and k800 have
  zero AMO-opcode words. No RTL directed test runs one through a capability in capability mode:
  - r43-evict-live and s06sec-amo-no-resurrect use integer addresses.
- **Not tested here: the tag half.** An AMO onto a granule holding a capability leaves its tag set: invariant I4,
  ISSUES.md "AMO over a capability granule". It is a known RTL residual, and the RTL lane's expected-FAIL repro
  `verif/tests/custom/capstone/s06sec-amo-no-resurrect.S` (capstone-ariane 36a641e0b) records it. This test asks only
  whether the operations WORK.

## The test
`tests/cap-atomics.S`, built bare by `build.sh` with the ladder's committed board harness
(`../sup-bare-2026-10-03-ladder`):
- Image: `images/cap-atomics.bin`, sha256 5fe7e5338c90992d. board_rec is at 0x80004040.
- Setup: CAPENTER over its own text, then one NONLIN RW capability over a 64-byte buffer. Every atomic takes that
  capability, or one offset from it (s1 / s2 = +32 / s3 = +48), as its address. The disassembly shows 13 atomics, all
  with rs1 in {s1, s2, s3}, and 0 compressed instructions in the test.
- Readings are architectural, with nothing timing-dependent. In CAPPRINT order:

| # | operation | expected |
|---|---|---|
| 1 | sd 0x1111, ld (plain control through the capability) | 0x1111 |
| 2, 3 | amoadd.d +5: old, then ld | 0x1111, 0x1116 |
| 4, 5 | amoswap.d 0xABCD: old, then ld | 0x1116, 0xABCD |
| 6, 7 | sw 0x7fffffff; amoadd.w +1: old, then lw (32-bit wrap, sign-extended) | 0x7fffffff, 0xffffffff80000000 |
| 8, 9 | amoswap.w 0x12345678: old, then lw | 0xffffffff80000000, 0x12345678 |
| 10, 11, 12 | lr.w; sc.w 0x55: rc; lw | 0x12345678, 0, 0x55 |
| 13, 14 | sc.w 0x66 with no reservation (the success consumed it): rc; lw | non-zero (CVA6 writes 1), 0x55 |
| 15, 16 | lr.w on +32, sc.w 0x77 on +48: rc; lw +48 (preset 0xB0B0) | non-zero, 0xB0B0 |
| 17, 18, 19 | lr.d; sc.d 0x1234: value, rc; ld | 0xABCD, 0, 0x1234 |
| 20, 21 | musl-style a_cas loop (expect 0x55, store 0x99): attempts; lw | 1, 0x99 |
| 22 | end marker | 0xA70D |

- Any trap ends the test with mcause, mepc, mtval and 0xDEAD recorded, and then the report.
- `compare.py` gives the verdict:
  - PASS = exact on every reading, and non-zero at 13 and 15;
  - "no data" is NO-RESULT, never a pass.
  It was checked on synthetic transcripts: an exact one passes, a planted miss at 3 and 13 is caught, and a run with
  no end marker is NO-RESULT.

## The board session
- Bitstream caplifive_supcall_36a641e0b (identified by csnodefree 0xFFCD). Bare M-mode, one image per power cycle
  (run_sup_bare.py: JTAG load_image at 0x80000000).
- Order: the CONTROL first, `call-retpc` (sup-bare-2026-10-02 images/call-retpc.bin, PASS exact at N=3 on this
  bitstream, readings [0, 0x12, 0x21, 0, 0x51, 0, 0, 0]); then cap-atomics.
- A failed control makes the session VOID.

## Readings that would change the plan
- **A trap at an atomic.** Its mcause says which kind: a capability check refusing an AMO address, or illegal.
  Musl's locks cannot run on silicon until it is fixed, and that blocks memcached at the root.
- **A wrong value.** A real silicon defect in the atomic path through capabilities. It is bisectable by reading
  number.
- **Reading 13 or 15 = 0** (an SC that should fail succeeds). LR/SC is not a valid lock on silicon.
