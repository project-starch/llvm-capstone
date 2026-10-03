# Supervised CALL on silicon, first runs -- pre-registered 2026-10-02 19:14:36, before any boot

Bitstream caplifive_supcall_36a641e0b.bit (identified by the csnodefree discriminator, 0xFFCD). Three bare M-mode
images, one board session each (power-cycle, reset halt, JTAG load at 0x80000000). The tests are the RTL lane's
verif/tests/custom/capstone/{sup-quantum,sup-escape,call-retpc}.S at 36a641e0b, byte-identical. Only the harness differs
(inc/): CAPPRINT -> board_rec[], CAP_PASS -> UART report, and an integer-mode prologue.
Images (sha256[:16]): sup-quantum 254aa761a0a1ba87, sup-escape 784b8d64c920c1c2, call-retpc a05ca464c4f683cd.

## Instrument readings that must appear first (else the run says nothing about supervised CALL)
- "SUPTEST BEGIN <Q|E|R>" (integer-mode UART), then "SB1" (prologue done, entering the test).
- "SR <n>" with n = the number of readings, then n "SV" lines, then "SUPTEST END".
- If SB1 appears but no SR: the test or the report wedged, or trapped in a loop. The GDB readout (pc, mcause, mepc,
  mtval, board_rec[]) names where.

## sup-quantum (step 4, the preemption mechanism the delegated runtime needs)
- Readings, in order: (csupstatus 3, csupcause 0x8000000000000010) repeated N times, then csupstatus 1 (the return),
  then 0x190 (400: every addi ran exactly once), 0x1234 (the domain's x1 survived every resume), N, 0xA5 (monitor
  x30 survived), 0x888 (monitor mie), 0 (mcause: no trap).
- N >= 3. Simulation read N = 52 at quantum 64. Board N may differ: DDR latency differs from the testbench's delay
  model and the quantum counts domain cycles. "Exactly 400" may NOT differ.
- REFUTED (the mechanism is broken on silicon) if: the count != 400; the 0x1234/0xA5/0x888 survivors differ; or the
  run ends in the trap handler (last reading a cause, mcause != 0).

## sup-escape (step 3: synchronous faults escape to the monitor)
- Per arm: status 0, csupstatus 5, then the cause:
  - ecall: 11;
  - illegal: 2;
  - csrw mepc: 2;
  - misaligned load: 4, with tval 0x3001;
  - mret: 2.
- Then: epc = the faulting pc (0 relative), the instruction bits (0x73 for ecall, 0xffffffff for illegal, the csrw's bits), seal back SEALED (4), 0xA5, 0x707, 0x1D1D, 0, 0x888, 0, and csupstatus 4 after the first read.
- Pass = the simulation sequence exactly.

## call-retpc (R-47: CALL parks the right return pc)
- #1..#7 = 0, 0x12, 0x21, 0, 0x51, 0, 0; #8 = 0.

## Board-only caveats (not refutations of the RTL)
- Node validity for ids 1/2 (CAPENTER's and CAPCREATE's nodes) on a bare boot is asked of the rtl-oracle. A cause-25
  trap at the first capability access would be THAT, not supervised CALL.
- Stale shadow tags in DDR are cleared by the prologue's ld/sd rewrite of .data, if an integer store clears a tag
  (asked of the rtl-oracle).
