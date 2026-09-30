/* Bare-metal Capstone test macros for capstone-qemu (-M virt, -bios none).
 *
 * Encodings follow target/riscv/insn32.decode of capstone-qemu: opcode 0x5B,
 * funct3 1, funct7 selects the operation. The RTL test header
 * verif/tests/custom/capstone/asm_insn.h uses the same numbering. Where the
 * hardware reads a register FIELD as an immediate (tighten, lcc), pass a
 * register whose number is the value: x4 means 4.
 */
#ifndef CAPSTONE_TEST_H
#define CAPSTONE_TEST_H

#define REVOKE(rs1)                  .insn r 0x5B, 0x1, 0x00, x0, rs1, x0
#define SHRINK(rd, rs1, rs2)         .insn r 0x5B, 0x1, 0x01, rd, rs1, rs2
#define TIGHTEN(rd, rs1, permsreg)   .insn r 0x5B, 0x1, 0x02, rd, rs1, permsreg
#define DELIN(rd)                    .insn r 0x5B, 0x1, 0x03, rd, x0, x0
#define LCC(rd, rs1, selreg)         .insn r 0x5B, 0x1, 0x04, rd, rs1, selreg
#define SCC(rd, rs1, rs2)            .insn r 0x5B, 0x1, 0x05, rd, rs1, rs2
#define SPLIT(rd, rs1, rs2)          .insn r 0x5B, 0x1, 0x06, rd, rs1, rs2
#define SEAL(rd, rs1)                .insn r 0x5B, 0x1, 0x07, rd, rs1, x0
#define MREV(rd, rs1)                .insn r 0x5B, 0x1, 0x08, rd, rs1, x0
#define INIT(rd, rs1, rs2)           .insn r 0x5B, 0x1, 0x09, rd, rs1, rs2
#define MOVC(rd, rs1)                .insn r 0x5B, 0x1, 0x0a, rd, rs1, x0
#define DROP(rs1)                    .insn r 0x5B, 0x1, 0x0b, x0, rs1, x0
#define CINCOFFSET(rd, rs1, rs2)     .insn r 0x5B, 0x1, 0x0c, rd, rs1, rs2
#define CAPENTER(rs1, rs2)           .insn r 0x5B, 0x1, 0x0d, x0, rs1, rs2
/* Stage-1 mapping instructions (M1). */
#define MAPCREATE(rd, rs1, rs2)      .insn r 0x5B, 0x1, 0x0e, rd, rs1, rs2
#define MAPPOPULATE(rs1, rs2)        .insn r 0x5B, 0x1, 0x0f, x0, rs1, rs2
#define MAPDETACH(rd, rs1)           .insn r 0x5B, 0x1, 0x10, rd, rs1, x0
#define MAPUNMAP(rd, rs1)            .insn r 0x5B, 0x1, 0x11, rd, rs1, x0
#define MAPDESTROY(rs1)              .insn r 0x5B, 0x1, 0x12, x0, rs1, x0
#define CALL(rd, rs1)                .insn r 0x5B, 0x1, 0x20, rd, rs1, x0
#define RETURN(rd, rs1, rs2)         .insn r 0x5B, 0x1, 0x21, rd, rs1, rs2
#define DEBUGGENCAP(rd, rs1, rs2)    .insn r 0x5B, 0x1, 0x40, rd, rs1, rs2
#define DEBUGPRINT(rs1)              .insn r 0x5B, 0x1, 0x43, x0, rs1, x0
#define CINCOFFSETIMM(rd, rs1, imm)  .insn i 0x5B, 0x2, rd, imm(rs1)
#define LDC(rd, rs1, imm)            .insn i 0x5B, 0x3, rd, imm(rs1)
#define STC(rs1, rs2, imm)           .insn s 0x5B, 0x4, rs2, imm(rs1)
#define CCSRRW(rd, ccsr, rs1)        .insn i 0x5B, 0x7, rd, ccsr(rs1)

#define CCSR_CTVEC 0
#define CCSR_CIH   1
#define CCSR_CEPC  2

/* Permission values (pass as register numbers to TIGHTEN). */
#define PERM_XO x1
#define PERM_WO x2
#define PERM_RO x4
#define PERM_RW x6

/* Capstone exception causes (cpu_bits.h). A trap reports 0x40 + cause. */
#define CAUSE_LOAD_ACCESS   5
#define CAUSE_STORE_ACCESS  7
#define CAUSE_UNEXP_OP_TYPE 0x18
#define CAUSE_INVALID_CAP   0x19
#define CAUSE_UNEXP_CAP_TYPE 0x1a
#define CAUSE_INSUF_PERMS   0x1b
#define TRAP_CODE(cause) (0x40 + (cause))

/* virt's test device: 0x5555 exits 0, (code << 16) | 0x3333 exits with code. */
#define TEST_DEV 0x100000
#define PASS  li t1, 0x5555; sw t1, 0(s10); 99: j 99b
#define FAIL(code) li t1, (((code) << 16) | 0x3333); sw t1, 0(s10); 99: j 99b

/* Enter capability mode, keep the low genesis capability in s10 with its
 * cursor on the test device, and install an execute-only trap vector that
 * reports 0x40 + mcause. After this a0 is null; a1 is the genesis capability
 * over [_code_end, 2^56) (2^63 before the E2 fix), RWX, linear. Tests must
 * not touch s10 or s11.
 */
#define TEST_PROLOGUE                          \
    .section .text.start, "ax";                \
    .globl _start;                             \
_start:                                        \
    lla a0, _start;                            \
    lla a1, _code_end;                         \
    CAPENTER(a0, a1);                          \
    MOVC(s10, a0);                             \
    li t0, TEST_DEV;                           \
    SCC(s10, s10, t0);                         \
    lla t0, _trap_handler;                     \
    lla t1, _trap_handler_end;                 \
    DEBUGGENCAP(t2, t0, t1);                   \
    TIGHTEN(t2, t2, PERM_XO);                  \
    CCSRRW(x0, CCSR_CTVEC, t2);                \
    j _test_body;                              \
    .align 4;                                  \
_trap_handler:                                 \
    csrr t0, mcause;                           \
    andi t0, t0, 0x3f;                         \
    addi t0, t0, 0x40;                         \
    slli t0, t0, 16;                           \
    li t1, 0x3333;                             \
    or t0, t0, t1;                             \
    sw t0, 0(s10);                             \
98: j 98b;                                     \
_trap_handler_end:                             \
    .align 4;                                  \
_test_body:

/* Scratch pages for tests, above the code and below 256 MiB of RAM. */
#define PAGE_A 0x80200000
#define PAGE_B 0x80201000
#define PAGE_C 0x80202000
#define PAGE_D 0x80203000

#endif
