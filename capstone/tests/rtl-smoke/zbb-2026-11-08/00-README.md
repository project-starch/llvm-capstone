# zbb-arm: Zbb/Zba execution on stage-B (C.5a's decoder), for the compiler lane's Zbb SDK profile

`gen.py` writes `zbb-arm.S` and `expected.txt` (2,015 cases): andn/orn/xnor, min/max/minu/maxu, rol/ror/rolw/rorw,
sh1add/sh2add/sh3add, add.uw, clz/ctz/cpop and their W forms, sext.b/sext.h/zext.h, rev8, orc.b, rori/roriw, over edge
operands (0, 1, 0xff, sign boundaries, all ones, two patterns; shift amounts 0/1/31/32/63/64), one CAPPRINT per result,
plain M mode. Expected values are computed on the host in Python (64-bit masked). Runs in capstone-ariane's verif tree
(copy the .S to `verif/tests/custom/capstone/`, the testlist to `verif/tests/`).

Reading (8 Nov 2026): capstone-ariane fe4810dd1 (stage-B = C.5a 1fc2cd5aa + the stage-B batch; decoder.sv identical to
C.5a's), dirty=tests: `zbb-grade.py <log.iss> expected.txt` -> **2015 values read, 2015 expected, 0 bad**. Negative
control: one expectation changed to 0xdead -> 1 bad, exit 1. A first grader version read the print line AFTER each
commit and misaligned every value by one; the print line precedes the CAPPRINT's commit in this trace.

Source side (rtl-oracle, re-checked): no Capstone decode condition reaches the OP/OP-32/OP-IMM/OP-IMM-32 arms of
decoder.sv at 1fc2cd5aa; Zbb results write back as untagged integers like add. Not covered here: inside a virtual
context (the source read covers it: no context input in those arms), Zbc clmul* (routed to MULT; a shared FLU
write-back port hazard not traced), Zbkb.
