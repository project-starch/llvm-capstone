#pragma once
/* census placeholder: the real NLR does not use setjmp (see plan stage 3) */
typedef __attribute__((aligned(16))) unsigned long jmp_buf[64];
int setjmp(jmp_buf);
void longjmp(jmp_buf, int);
