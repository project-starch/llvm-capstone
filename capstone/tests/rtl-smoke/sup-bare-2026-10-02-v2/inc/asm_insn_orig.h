#ifdef USE_UNCOMPRESSED
#define REG_SIZE  0x20
#define REG_SIZE_LOG2 5
#else
#define REG_SIZE  0x10
#define REG_SIZE_LOG2 4
#endif

#define EXIT_STUB li a0, 0; \
    li a7, 93; \
    ecall

#define REVOKE(rs1)                     .insn r 0x5B, 0x1, 0x0, x0, rs1, x0
#define SHRINK(rd,rs1,rs2)              .insn r 0x5B, 0x1, 0x1, rd, rs1, rs2
#define TIGHTEN(rd,rs1,rs2)             .insn r 0x5B, 0x1, 0x2, rd, rs1, rs2 //rs2 is actually an immediate - .insn doesn't have the type we need
#define DELIN(rd)                       .insn r 0x5B, 0x1, 0x3, rd, x0, x0
#define LCC(rd,rs1,rs2)                 .insn r 0x5B, 0x1, 0x4, rd, rs1, rs2 //rs2 again immediate
#define SCC(rd,rs1,rs2)                 .insn r 0x5B, 0x1, 0x5, rd, rs1, rs2
#define SPLIT(rd,rs1,rs2)               .insn r 0x5B, 0x1, 0x6, rd, rs1, rs2
#define SEAL(rd,rs1)                    .insn r 0x5B, 0x1, 0x7, rd, rs1, x0
#define MREV(rd,rs1)                    .insn r 0x5B, 0x1, 0x8, rd, rs1, x0
#define INIT(rd,rs1,rs2)                .insn r 0x5B, 0x1, 0x9, rd, rs1, rs2
#define MOVC(rd,rs1)                    .insn r 0x5B, 0x1, 0xa, rd, rs1, x0
#define DROP(rs1)                       .insn r 0x5B, 0x1, 0xb, x0, rs1, x0
#define CINCOFFSET(rd,rs1,rs2)          .insn r 0x5B, 0x1, 0xc, rd, rs1, rs2
#define CINCOFFSETIMM(rd,rs1,simm12)    .insn i 0x5B, 0x2, rd, simm12(rs1)

#define LDC(rd,rs1,simm12)              .insn i 0x5B, 0x3, rd, simm12(rs1)
#define STC(rs1,rs2,simm12)             .insn s 0x5B, 0x4, rs2, simm12(rs1)

#define CJALR(rd,rs1,simm12)            .insn i 0x5B, 0x5, rd, simm12(rs1)
#define CBNZ(rd,rs1,simm12)             .insn i 0x5B, 0x6, rd, simm12(rs1)

#define CALL(rd,rs1)                    .insn r 0x5B, 0x1, 0x20, rd, rs1, x0
#define RETURN(rd, rs1, rs2)             .insn r 0x5B, 0x1, 0x21, rd, rs1, rs2

#define CAPENTER(rs1, rs2)              .insn r 0x5B, 0x1, 0xd, x0, rs1, rs2
#define CSSUPERVISE(rd,rs1,rs2)         .insn r 0x5B, 0x1, 0x22, rd, rs1, rs2   // supervised CALL on silicon: arm (rs1 = seal, rs2 = the save area, LINEAR RW >= 1 KiB; rs1 = x0 forgets; the quantum is CSR csupquantum)
#define CCSRRW(rd, ccsr, rs1)           .insn i 0x5B, 0x7, rd, ccsr(rs1)


// CCSRs
#define CCSR_CTVEC              0x0
#define CCSR_CIH                0x1
#define CCSR_CEPC               0x2
#define CCSR_CSCRATCH           0x4
#define CCSR_CPMP(n) (0x10 | n)

// Test instructions for node ops
#define QUERY(reg)              .insn r 0x5B, 0x0, 0x0, x0, reg, x0
#define DROPT(reg)              .insn r 0x5B, 0x0, 0x1, x0, reg, x0
#define RCUPDATE(rs1,rs2)       .insn r 0x5B, 0x0, 0x2, x0, rs1, rs2
#define ALLOC(rd,rs1)           .insn r 0x5B, 0x0, 0x3, rd, rs1, x0
#define REVOKET(reg)            .insn r 0x5B, 0x0, 0xa, x0, reg, x0
#define QUERYDBG(reg)           .insn r 0x5B, 0x0, 0xf, x0, reg, x0

// Test instructions for capability ops
#define CAPCREATE(reg)          .insn r 0x7B, 0x0, 0x4, reg, x0, x0
#define CAPTYPE(rd,rs1)         .insn r 0x7B, 0x0, 0x5, rd, rs1, x0
#define CAPNODE(rd,rs1)         .insn r 0x7B, 0x0, 0x6, rd, rs1, x0
#define CAPPERM(rd,rs1)         .insn r 0x7B, 0x0, 0x7, rd, rs1, x0
#define CAPBOUND(rd,rs1,rs2)    .insn r 0x7B, 0x0, 0x8, rd, rs1, rs2
#define CAPPRINT(rs1)           .insn r 0x7B, 0x0, 0x9, x0, rs1, x0

#define NODE_ID_INVALID ((-1) & ((1 << 31) - 1))

// Capability-related constants
#define CAP_PERM_NA 0
#define CAP_PERM_XO 1
#define CAP_PERM_WO 2
#define CAP_PERM_WX 3
#define CAP_PERM_RO 4
#define CAP_PERM_RX 5
#define CAP_PERM_RW 6
#define CAP_PERM_RWX 7

#define NOT_CAP 0
#define CAP_TYPE_LIN 1
#define CAP_TYPE_NONLIN 2
#define CAP_TYPE_REV 3
#define CAP_TYPE_UNINIT 4
#define CAP_TYPE_SEALED 5
#define CAP_TYPE_SEALEDRET 6
#define CAP_TYPE_EXIT 7

// Registers
#define RET x1  // or `ra`
#define ARG x10 // or `a0`

#define INIT_RWX_CAP(reg) \
    CAPCREATE(reg);\
    li a1, CAP_TYPE_LIN;\
    CAPTYPE(reg, a1);\
    li a1, NODE_ID_INVALID;\
    ALLOC(a2, a1);\
    CAPNODE(reg, a2);\
    li a1, CAP_PERM_RWX;\
    CAPPERM(reg, a1);

#define CHK_START addi x0, a0, 0
#define CHK_END   addi x0, a1, 0
#define CHK_M     addi x0, a2, 0

/* Exit for tests in capability mode: RVTEST_PASS/FAIL store tohost through an integer base, which capmode refuses
   (cause 24), so the harness runs to its cycle ceiling. CAP_PASS writes tohost through a capability instead. */
#define CAP_PASS(creg) \
    CAPCREATE(creg); li a0, CAP_TYPE_LIN; lla a2, tohost; lla a3, tohost + 64; li a4, CAP_PERM_RWX; \
    CAPTYPE(creg, a0); CAPBOUND(creg, a2, a3); CAPPERM(creg, a4); \
    fence; li a0, 1; sw a0, 0(creg); 1: j 1b
