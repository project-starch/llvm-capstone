# Supervised-CALL ladder on silicon, bare M-mode -- pre-registered 2026-10-03 19:18:59, committed and pushed BEFORE the boot

Bitstream caplifive_supcall_36a641e0b.bit (identified by csnodefree 0xFFCD). Tests: capstone-ariane 36a641e0b, unchanged
through 7564c0945 (the RTL lane confirmed: no test changed since, and the RTL there equals the synthesized hash).
Harness: the v2 harness, with recorder registers chosen per image from the registers the test never names (build.sh).
One board session per image. images/SHA256SUMS lists the 17 images.

Expected vectors (expected-vectors.json): CAPPRINT readings only. They were extracted from the RTL lane's all7
simulation and checked to be an order-preserving subsequence of its committed reference (sup-call b83d67ea1,
sup-vectors-7564c0945.txt), the extras being the trap handler's csrr mcause reads.
- EXACT: the board vector equals the simulation vector.
  - sup-fullswitch (17): its data address 0x80003000 = context_start in both layouts;
  - sup-arm (13), sup-armclose (14);
  - sup-sealsize-64/1023/off8 (2 each): S-11 refuses, and the last reading is the trap cause 0x1d = 29;
  - sup-sealsize-1024/2048/off16/cur960 (11 each);
  - sup-guards (84).
- EXACT EXCEPT one REPORTED position: sup-mtip (15). Reading 1 is the reference spin count until the timer interrupt, a
  timing quantity (board mtime is 1 MHz). Invariant: below 4,000,000 (0x3d0900, "it never came"). The CLINT addresses
  hard-coded in the test (mtimecmp 0x02004000, mtime 0x0200BFF8) are the board's too: platform.c CLINT 0x2000000 +
  MTIMER 0x4000, mtime +0x7ff8.
- EXPLORATORY, invariants only: sup-strip and its STRIP_DELAY variants d0/d8/d32/d128 (5 each). Their readings depend
  on when the CLINT msip write lands relative to the CALL, which is board timing. The detector that decides the hazard
  in simulation is a sim-only $display. Invariants: the run completes; every mcause reading is 0 or
  0x8000000000000003; every mepc reading is 0 or inside the image's text.
- REPEATS, toward N = 3, on the existing images:
  - sup-escape v2 (b45cd0aab8fe97ad): exact, 72/72; saved mcause 0 in all five arms.
  - sup-quantum v2 q16 (57f1d4cd9e959232) and q64 (2e4747f346eaad85): invariants. Ladder 0x1f64f7f55a20a182, x1
    0x1234, mie 0x888, mcause 0, every escape reading (3, 0x8000000000000010). The preemption count is REPORTED (board
    q64 read 74, q16 74).
  - call-retpc v1 (a05ca464c4f683cd): exact, 0, 0x12, 0x21, 0, 0x51, 0, 0, 0.
