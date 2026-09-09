/* See capstone_setjmp.c. 14 slots of 16 bytes: ra, s0, s1, s2-s11, sp. */
#ifndef CAPSTONE_SETJMP_H
#define CAPSTONE_SETJMP_H
typedef struct { void *regs[14]; } jmp_buf[1];
int setjmp(jmp_buf env);
__attribute__((noreturn)) void longjmp(jmp_buf env, int val);
#endif
