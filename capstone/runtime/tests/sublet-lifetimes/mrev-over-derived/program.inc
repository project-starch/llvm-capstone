/* MREV over a node that already has CDERIVE children, reached through a LINEAR duplicate.
 *
 * MREV needs a LINEAR operand and CDERIVE a NONLIN one, so the shape needs two copies of one
 * node. capstone-qemu's SHRINKTO with rd != rs1 leaves its LINEAR source in place (ISSUES R-21's
 * box: the emulator's arm 10 reads 0), which supplies them. Run by run.sh through the QEMU
 * submodule's tests/sublet-lifetimes/probe.S as SUBLET_PROGRAM. */
#define DERIVE(rd, parent, size) .insn r 0x5b, 1, 0x51, rd, parent, size
#define LCC(rd, src, field) .insn r 0x5b, 1, 4, rd, src, field
#define DELIN(reg) .insn r 0x5b, 1, 3, reg, zero, zero
#define MOVC(rd, src) .insn r 0x5b, 1, 0x0a, rd, src, zero
#define SHRINKTO(rd, src, size) .insn i 0x5b, 0, rd, src, size
#define MREV(rd, src) .insn r 0x5b, 1, 8, rd, src, zero
#define REVOKE(src) .insn r 0x5b, 1, 0, zero, src, zero
    LCC(t1, s4, x5)
    li t2, 6
    bne t1, t2, app_fail     /* s4 is minted RW, without MANAGE */
    li t0, 0x5ec
    sd t0, 0(s4)
#ifdef IN_PLACE
    SHRINKTO(s4, s4, 1024)   /* control: in place, then MOVC consumes the LINEAR source */
    MOVC(a4, s4)
#else
    SHRINKTO(a4, s4, 1024)   /* rd != rs1: the emulator copies, s4 survives */
    LCC(t1, s4, x1)
    bnez t1, app_fail        /* s4 is still LINEAR (type 0) */
#endif
    DELIN(a4)                /* a NONLIN copy of the same node */
    li t0, 64
    DERIVE(a5, a4, t0)       /* a5: a CDERIVE child of that node */
    ld t1, 0(a5)
    li t0, 0x5ec
    bne t0, t1, app_fail
#ifdef IN_PLACE
fault_site: MREV(s9, s4)     /* nothing LINEAR is left to name the node: 26 */
#else
    MREV(s9, s4)             /* a handle over a node that already has a derived child */
    LCC(t1, s9, x1)
    li t2, 2
    bne t1, t2, app_fail     /* s9 is a REV handle */
    ld t1, 0(a5)             /* MREV does not end the derived child */
    bne t0, t1, app_fail
#ifndef NO_REVOKE
    REVOKE(s9)               /* legacy REVOKE of the new handle */
fault_site: ld t1, 0(a5)     /* ends the derived child: 25 */
#else
fault_site: ebreak
#endif
#endif
app_fail:
    .word 0
