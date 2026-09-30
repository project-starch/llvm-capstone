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
 * reports 0x40 + mcause. A test that expects a fault names the faulting
 * instruction with EXPECT_FAULT_AT(label); a fault anywhere else then exits
 * with 0x3f instead of the cause, so a refusal test cannot pass on a fault in
 * its setup. After the prologue a0 is null; a1 is the genesis capability over
 * [_code_end, 2^56), RWX, linear. Tests must not touch s10 or s11.
 */
#define EXPECT_FAULT_AT(label) lla s11, label
#define WRONG_SITE 0x3f
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
    li s11, 0;                                 \
    j _test_body;                              \
    .align 4;                                  \
_trap_handler:                                 \
    CCSRRW(t2, CCSR_CEPC, x0);                 \
    LCC(t3, t2, x1);                           \
    li t4, 7;                                  \
    beq t3, t4, 97f;                           \
    LCC(t2, t2, x2);                           \
97: beqz s11, 96f;                             \
    beq t2, s11, 96f;                          \
    li t0, (0x3f << 16) | 0x3333;              \
    sw t0, 0(s10);                             \
95: j 95b;                                     \
96: csrr t0, mcause;                           \
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
#define PAGE_E 0x80204000
#define PAGE_F 0x80205000
#define PAGE_G 0x80206000
#define PAGE_H 0x80207000
#define PAGE_I 0x80208000

/* Build a sealed synchronous context in the page at `page`: saved PC over
 * [entry, entry_end) (RWX), a fresh execute-only trap vector over the
 * prologue's handler, mstatus kept with privilege 3. Leaves the SEALED handle
 * in capreg. Clobbers t0, t1, t2. */
#define MAKE_CONTEXT(capreg, page, entry, entry_end)                   \
    li t0, page; li t1, (page) + 4096; DEBUGGENCAP(capreg, t0, t1);    \
    lla t0, entry; lla t1, entry_end; DEBUGGENCAP(t2, t0, t1);         \
    STC(capreg, t2, 0);                                                \
    lla t0, _trap_handler; lla t1, _trap_handler_end;                  \
    DEBUGGENCAP(t2, t0, t1); TIGHTEN(t2, t2, PERM_XO);                 \
    STC(capreg, t2, 0x10);                                             \
    csrr t0, mstatus; li t1, 3; slli t1, t1, 38; or t0, t0, t1;        \
    sd t0, 0x30(capreg);                                               \
    SEAL(capreg, capreg)

/* Mint one page as a linear RWX capability into capreg. Clobbers t0, t1. */
#define MINT_PAGE(capreg, page) \
    li t0, page; li t1, (page) + 4096; DEBUGGENCAP(capreg, t0, t1)

/* CREATE operands: a1/a2 = [2^57 + offset, +size), a0 = id, a3 = perms,
 * a4 = delivery register number. Clobbers t0. */
#define CREATE_ARGS(id, offset, size, perms, reg)                       \
    li a0, id; li a1, 1; slli a1, a1, 57; li t0, offset; add a1, a1, t0; \
    li a2, size; add a2, a1, a2; li a3, perms; li a4, reg

/* Monitor-side setup shared by the POPULATE tests: a sealed context in
 * PAGE_C, mapping id 0 over [2^57, 2^57 + 1 MiB) with rights `perms`,
 * delivered into x28 (t3); root page PAGE_A. Leaves the detach handle in s2. */
#define SETUP_MAPPING(perms)                          \
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end);         \
    MINT_PAGE(s1, PAGE_A);                            \
    CREATE_ARGS(0, 0, 0x100000, perms, 28);           \
    MAPCREATE(s2, s1, s0)

/* POPULATE page `offset` of mapping s2 with frame `page`; `treg` is the
 * register number holding a table page, or 0. Clobbers t0, t1, a0, a5. */
#define POPULATE(framereg, page, offset, treg)        \
    MINT_PAGE(framereg, page);                        \
    li a0, 1; slli a0, a0, 57; li t0, offset; add a0, a0, t0; \
    li a5, treg;                                      \
    MAPPOPULATE(s2, framereg)

/* Return from a called domain to the monitor; the domain resumes at `label`
 * on the next call. x1 holds the sealed-return capability from the call. */
#define DOM_RETURN(label) lla t0, label; RETURN(x1, t0, x0)

/* Two populated pages (PAGE_E at offset 0 with leaf table PAGE_F, PAGE_G at
 * 0x1000) under SETUP_MAPPING. */
#define POPULATE_TWO_PAGES        \
    MINT_PAGE(s4, PAGE_F);        \
    POPULATE(s3, PAGE_E, 0, 20);  \
    POPULATE(s3, PAGE_G, 0x1000, 0)

#endif
