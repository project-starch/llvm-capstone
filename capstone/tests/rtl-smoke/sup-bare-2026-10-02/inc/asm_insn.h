/* Board harness for the RTL lane's supervised-CALL directed tests (capstone-ariane 36a641e0b,
 * verif/tests/custom/capstone). The tests are byte-identical to the simulated ones; only this header
 * differs: it is found BEFORE the original on the include path and redefines the two testbench hooks.
 *   CAPPRINT(r) -- in simulation a trace print -- appends r to board_rec[] through a fresh CAPCREATE'd
 *                  capability (count in board_rec[0]).
 *   CAP_PASS(c) -- in simulation a tohost write -- prints board_rec[] over the UART and spins.
 * Both expand INLINE: CAPENTER(_start, _end_of_text) bounds the PC capability to the test's own code,
 * so out-of-line harness code would be unreachable. Scratch registers x24, x25, x31 (and, in the
 * report, x23 and x26..x31 once the test is over) are unused by all three tests. */
#ifndef BOARD_ASM_INSN_H
#define BOARD_ASM_INSN_H
#include "asm_insn_orig.h"
#define BOARD_UART      0x10000000
#define BOARD_REC_MAX   500

#undef CAPPRINT
#define CAPPRINT(rs1) \
    CAPCREATE(x25); li x24, CAP_TYPE_LIN; CAPTYPE(x25, x24); \
    lla x24, board_rec; lla x31, board_rec_end; CAPBOUND(x25, x24, x31); \
    li x24, CAP_PERM_RW; CAPPERM(x25, x24); \
    ld x24, 0(x25); li x31, BOARD_REC_MAX; bgeu x24, x31, 99f; \
    addi x24, x24, 1; sd x24, 0(x25); slli x31, x24, 3; CINCOFFSET(x25, x25, x31); sd rs1, 0(x25); \
    99: li x25, 0; li x24, 0; li x31, 0

/* one character (immediate) / one character (register x30) to the 16550, polling THRE (LSR bit 5, reg 5 << 2) */
#define BPUTI(ch) li x30, ch; 81: lw x31, 20(x25); andi x31, x31, 32; beqz x31, 81b; sw x30, 0(x25)
#define BPUTR      82: lw x31, 20(x25); andi x31, x31, 32; beqz x31, 82b; sw x30, 0(x25)
#define BCRLF      BPUTI(13); BPUTI(10)
#define BHEX(r) \
    li x23, 60; 83: srl x30, r, x23; andi x30, x30, 15; li x31, 10; blt x30, x31, 84f; \
    addi x30, x30, 55; j 85f; 84: addi x30, x30, 48; 85: BPUTR; addi x23, x23, -4; bgez x23, 83b

#undef CAP_PASS
#define CAP_PASS(creg) \
    CAPCREATE(x25); li x24, CAP_TYPE_LIN; CAPTYPE(x25, x24); \
    li x24, BOARD_UART; li x31, BOARD_UART + 0x1000; CAPBOUND(x25, x24, x31); li x24, CAP_PERM_RW; CAPPERM(x25, x24); \
    CAPCREATE(x27); li x24, CAP_TYPE_LIN; CAPTYPE(x27, x24); \
    lla x24, board_rec; lla x31, board_rec_end; CAPBOUND(x27, x24, x31); li x24, CAP_PERM_RW; CAPPERM(x27, x24); \
    BPUTI(83); BPUTI(82); BPUTI(32); ld x28, 0(x27); BHEX(x28); BCRLF; \
    li x26, 0; \
    71: bge x26, x28, 72f; addi x26, x26, 1; CINCOFFSETIMM(x27, x27, 8); ld x29, 0(x27); \
    BPUTI(83); BPUTI(86); BPUTI(32); BHEX(x29); BCRLF; j 71b; \
    72: BPUTI(83); BPUTI(85); BPUTI(80); BPUTI(84); BPUTI(69); BPUTI(83); BPUTI(84); BPUTI(32); \
    BPUTI(69); BPUTI(78); BPUTI(68); BCRLF; \
    73: j 73b

#endif /* BOARD_ASM_INSN_H */
