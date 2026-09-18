#ifndef A11_SETJMP_H
#define A11_SETJMP_H
/* The manager never longjmps: its only use of the error path ends the
   process. A jmp_buf has to exist because elog.h declares one. */
typedef long sigjmp_buf[32];
typedef long jmp_buf[32];
int __sigsetjmp(sigjmp_buf, int);
#define sigsetjmp(b, s) __sigsetjmp((b), (s))
void siglongjmp(sigjmp_buf, int);
#endif
