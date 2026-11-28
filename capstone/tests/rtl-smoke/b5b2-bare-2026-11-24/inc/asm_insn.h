/* Board harness for the RTL lane's supervised-CALL directed tests (capstone-ariane 36a641e0b,
 * verif/tests/custom/capstone). The tests are byte-identical to the simulated ones; only this header
 * differs: it is found BEFORE the original on the include path and redefines the two testbench hooks.
 *   CAPPRINT(r) -- in simulation a trace print -- appends r to board_rec[] through a fresh CAPCREATE'd
 *                  capability (count in board_rec[0]).
 *   CAP_PASS(c) -- in simulation a tohost write -- prints board_rec[] over the UART and spins.
 * Both expand INLINE: CAPENTER(_start, _end_of_text) bounds the PC capability to the test's own code,
 * so out-of-line harness code would be unreachable. v2 (tests at capstone-ariane 7564c0945): the recorder's scratch
 * registers are x16, x17, x8, because the repaired sup-escape keeps its seal aliases in s8/s9/x31. (And, in the
 * report, x23 and x26..x31 once the test is over) are unused by all three tests. */
#ifndef BOARD_ASM_INSN_H
#define BOARD_ASM_INSN_H
#include "asm_insn_orig.h"
#define BOARD_UART      0x10000000
#define BOARD_REC_MAX   500

#undef CAPPRINT
/* Recorder scratch registers: a per-test build option (build.sh), chosen from the registers the test never names.
 * Defaults are the v2 choice. */
#ifndef BOARD_R_CAP
#define BOARD_R_CAP x16
#endif
#ifndef BOARD_R_T1
#define BOARD_R_T1 x17
#endif
#ifndef BOARD_R_T2
#define BOARD_R_T2 x8
#endif
/* s1-revoke-gap (2026-10-14): a test that revokes the base's node must keep its record capabilities on a node of
 * their own. With -DBOARD_REC_NODE=<reg> every capability the wrapper creates is attached to the node id the test
 * keeps in that register (alloc + capnode); without it the macros are the ladder's, byte for byte. */
#ifdef BOARD_REC_NODE
#define BOARD_ATTACH_NODE(cap) CAPNODE(cap, BOARD_REC_NODE)
#define BOARD_ATTACH_NODE_FROM(cap, src) CAPNODE(cap, src)
#else
#define BOARD_ATTACH_NODE(cap)
#define BOARD_ATTACH_NODE_FROM(cap, src)
#endif
/* sublet-bare (2026-10-15): CAP_PASS attaches its UART capability (x25) from BOARD_REC_NODE and its record capability
 * (x27) from x25, because CAPCREATE(x27) runs before x27's attach and BOARD_REC_NODE may be x27 itself (a test that
 * keeps its surviving-node capability in s11). */
#define CAPPRINT(rs1) \
    CAPCREATE(BOARD_R_CAP); li BOARD_R_T1, CAP_TYPE_LIN; CAPTYPE(BOARD_R_CAP, BOARD_R_T1); \
    lla BOARD_R_T1, board_rec; lla BOARD_R_T2, board_rec_end; CAPBOUND(BOARD_R_CAP, BOARD_R_T1, BOARD_R_T2); \
    li BOARD_R_T1, CAP_PERM_RW; CAPPERM(BOARD_R_CAP, BOARD_R_T1); BOARD_ATTACH_NODE(BOARD_R_CAP); \
    ld BOARD_R_T1, 0(BOARD_R_CAP); li BOARD_R_T2, BOARD_REC_MAX; bgeu BOARD_R_T1, BOARD_R_T2, 99f; \
    addi BOARD_R_T1, BOARD_R_T1, 1; sd BOARD_R_T1, 0(BOARD_R_CAP); slli BOARD_R_T2, BOARD_R_T1, 3; CINCOFFSET(BOARD_R_CAP, BOARD_R_CAP, BOARD_R_T2); sd rs1, 0(BOARD_R_CAP); \
    99: li BOARD_R_CAP, 0; li BOARD_R_T1, 0; li BOARD_R_T2, 0

#ifdef BOARD_REC_PLAIN
/* b5b2-bare (2026-11-24): a recorder for a test that names every GPR but tp. Plain M stores (M is integer mode under
 * Sublet), physical addresses; it uses x4 and spills rs1, x5 and x6 to board_save, so rs1 may be x5 or x6. */
#undef CAPPRINT
#define CAPPRINT(rs1) \
    lla x4, board_save; sd rs1, 0(x4); sd x5, 8(x4); sd x6, 16(x4); \
    lla x5, board_rec; ld x6, 0(x5); li x4, BOARD_REC_MAX; bgeu x6, x4, 98f; \
    addi x6, x6, 1; sd x6, 0(x5); slli x6, x6, 3; add x5, x5, x6; lla x4, board_save; ld x6, 0(x4); sd x6, 0(x5); \
    98: lla x4, board_save; ld x5, 8(x4); ld x6, 16(x4); li x4, 0
#endif

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
    li x24, BOARD_UART; li x31, BOARD_UART + 0x1000; CAPBOUND(x25, x24, x31); li x24, CAP_PERM_RW; CAPPERM(x25, x24); BOARD_ATTACH_NODE(x25); \
    CAPCREATE(x27); li x24, CAP_TYPE_LIN; CAPTYPE(x27, x24); \
    lla x24, board_rec; lla x31, board_rec_end; CAPBOUND(x27, x24, x31); li x24, CAP_PERM_RW; CAPPERM(x27, x24); BOARD_ATTACH_NODE_FROM(x27, x25); \
    BPUTI(83); BPUTI(82); BPUTI(32); ld x28, 0(x27); BHEX(x28); BCRLF; \
    li x26, 0; \
    71: bge x26, x28, 72f; addi x26, x26, 1; CINCOFFSETIMM(x27, x27, 8); ld x29, 0(x27); \
    BPUTI(83); BPUTI(86); BPUTI(32); BHEX(x29); BCRLF; j 71b; \
    72: BPUTI(83); BPUTI(85); BPUTI(80); BPUTI(84); BPUTI(69); BPUTI(83); BPUTI(84); BPUTI(32); \
    BPUTI(69); BPUTI(78); BPUTI(68); BCRLF; \
    73: j 73b

/* b5b2-bare (2026-11-24): the B5b-2 tests end in RVTEST_PASS, a tohost store, which prints nothing on the board (the
 * vector reaches the UART only through CAP_PASS). Every test includes this header after riscv_test.h, so the pass becomes
 * the board report. The board CAP_PASS does not read its argument. */
#undef RVTEST_PASS
#define RVTEST_PASS CAP_PASS(x0)

#endif /* BOARD_ASM_INSN_H */
