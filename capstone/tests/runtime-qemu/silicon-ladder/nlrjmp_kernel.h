/* A capability-aware setjmp/longjmp, and a kernel that exercises it across three
   frames. This is the primitive MicroPython's exception handling rests on: nlr.h
   falls back to MICROPY_NLR_SETJMP on any architecture it does not recognise, and
   capstone64 is one, so the freestanding domain has to provide setjmp/longjmp or
   nothing that raises can run.

   Why it cannot be the ordinary one: `ra` and `sp` are capabilities here, so saving
   them with `sd` keeps the address and drops the tag, and the restored `sp` is then
   an untagged value that faults on first use. Every slot is therefore 16 bytes and
   saved with `stc`/`ldc`, which post-S-06 is a verbatim 128-bit copy for untagged
   values too -- so the same code is correct for the integer registers.

   The host build uses the C library's setjmp, which is the oracle: the domain has to
   produce the same number. */
#ifndef NLRJMP_KERNEL_H
#define NLRJMP_KERNEL_H

#ifdef __capstone

/* 14 slots: ra, s0, s1, s2-s11, sp -- the same set nlrrv64.c saves, at 16 bytes
   each instead of 8. */
typedef struct {
    void *regs[14];
} cj_buf_t[1];

/* stc/ldc have no mnemonics in the assembler; the glue encodes them by hand and so
   do we, with the same encoding as start-gp-captable-interp.S. */
#define CJ_STC(reg, off) ".insn s 0x5b, 0x4, " #reg ", " #off "(a0)\n"
#define CJ_LDC(reg, off) ".insn i 0x5b, 0x3, " #reg ", " #off "(a0)\n"

#define CJ_SAVED_REGS(M) \
    M(ra, 0) M(s0, 16) M(s1, 32) M(s2, 48) M(s3, 64) M(s4, 80) M(s5, 96) \
    M(s6, 112) M(s7, 128) M(s8, 144) M(s9, 160) M(s10, 176) M(s11, 192) M(sp, 208)

__attribute__((naked, noinline)) static int cj_setjmp(cj_buf_t env) {
    __asm volatile(
        CJ_SAVED_REGS(CJ_STC)
        "li a0, 0\n"
        "ret\n");
}

__attribute__((naked, noinline)) static void cj_longjmp(cj_buf_t env, int val) {
    /* a0 is the buffer and stays the base until the last load, so it is restored
       last -- as the return value rather than from the buffer. A zero argument must
       come back as 1, which is what setjmp's contract says. */
    __asm volatile(
        CJ_SAVED_REGS(CJ_LDC)
        "mv a0, a1\n"
        "bnez a0, 1f\n"
        "li a0, 1\n"
        "1:\n"
        "ret\n");
}

#define CJ_SETJMP(b) cj_setjmp(b)
#define CJ_LONGJMP(b, v) cj_longjmp(b, v)

#else /* host oracle */

#include <setjmp.h>
typedef jmp_buf cj_buf_t;
#define CJ_SETJMP(b) setjmp(b)
#define CJ_LONGJMP(b, v) longjmp(b, v)

#endif

/* NLRJMP_DEPTH parameterises how many frames the jump crosses. -1 jumps from the
   same frame that called setjmp, which is the smallest case that can work at all. */
#ifndef NLRJMP_DEPTH
#define NLRJMP_DEPTH 3
#endif

static cj_buf_t nlrjmp_env;

/* Recursion so the jump crosses real frames rather than returning in place, and so
   the restored sp has to be the one setjmp saw and not the one at the jump site. */
/* NLRJMP_NOJUMP=1 keeps the recursion and drops the jump: the matched arm that says
   whether a failure is about longjmp or about the frames it crosses. */
#ifndef NLRJMP_NOJUMP
#define NLRJMP_NOJUMP 0
#endif

static void nlrjmp_deep(int n) {
    if (n == 0) {
#if !NLRJMP_NOJUMP
        CJ_LONGJMP(nlrjmp_env, 42);
#endif
        return;
    }
    nlrjmp_deep(n - 1);
    /* Keep the recursion from being a tail call, so the frames are real. */
    __asm volatile("" ::: "memory");
}

static unsigned nlrjmp_compute(void) {
    volatile unsigned acc = 0xBAD;
    int v = CJ_SETJMP(nlrjmp_env);
    if (v == 0) {
#if NLRJMP_DEPTH < 0
        CJ_LONGJMP(nlrjmp_env, 42);
#else
        nlrjmp_deep(NLRJMP_DEPTH);
#endif
#if NLRJMP_NOJUMP
        return 1042; /* the no-jump arm returns the same value by construction */
#endif
        return 0xDEAD; /* unreachable: the jump above does not return here */
    }
    acc = 1000u + (unsigned)v;
    return acc;
}

#endif /* NLRJMP_KERNEL_H */
