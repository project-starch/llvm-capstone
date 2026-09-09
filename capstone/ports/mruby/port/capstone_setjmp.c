/* setjmp/longjmp for a Capstone domain.
 *
 * MICROPY_NLR_SETJMP=1, so this IS MicroPython's exception mechanism, not a stub: every
 * `raise` and every nlr_push in the interpreter goes through it.
 *
 * The ordinary implementation will not do. `ra` and `sp` are capabilities here, so saving them
 * with `sd` keeps the address and drops the tag, and the restored `sp` then faults on its first
 * use. Every slot is therefore 16 bytes and saved with `stc`/`ldc`, which post-S-06 is a verbatim
 * 128-bit copy for untagged values too and is correct for the integer registers as well.
 *
 * The register set is the one py/nlrrv64.c saves (ra, s0-s11, sp), at 16 bytes per slot instead
 * of 8. Proven as the `nlrjmp` ladder rung: it returns its native oracle in a domain under QEMU
 * with the jump crossing several frames.
 *
 * stc/ldc have no assembler mnemonics; the encodings are the ones start-gp-captable-interp.S uses.
 */
#include "capstone_setjmp.h"

#define CJ_STC(reg, off) ".insn s 0x5b, 0x4, " #reg ", " #off "(a0)\n"
#define CJ_LDC(reg, off) ".insn i 0x5b, 0x3, " #reg ", " #off "(a0)\n"

#define CJ_SAVED_REGS(M) \
    M(ra, 0) M(s0, 16) M(s1, 32) M(s2, 48) M(s3, 64) M(s4, 80) M(s5, 96) \
    M(s6, 112) M(s7, 128) M(s8, 144) M(s9, 160) M(s10, 176) M(s11, 192) M(sp, 208)

__attribute__((naked, noinline)) int setjmp(jmp_buf env) {
    __asm volatile(
        CJ_SAVED_REGS(CJ_STC)
        "li a0, 0\n"
        "ret\n");
}

__attribute__((naked, noinline)) void longjmp(jmp_buf env, int val) {
    /* a0 is the buffer and stays the base until the last load, so the return value is put in
       a0 afterwards rather than restored from the buffer. setjmp's contract says a zero
       argument comes back as 1. */
    __asm volatile(
        CJ_SAVED_REGS(CJ_LDC)
        "mv a0, a1\n"
        "bnez a0, 1f\n"
        "li a0, 1\n"
        "1:\n"
        "ret\n");
}
