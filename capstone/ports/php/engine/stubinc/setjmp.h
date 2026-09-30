/* zend_bailout is the engine's only fatal path and it MUST link and be correct.
 *
 * 16-BYTE ALIGNMENT IS LOAD-BEARING: capstone_setjmp.S saves ra/sp/s0-s11 with `stc`,
 * and stc drops the tag on a non-16-aligned address. A misaligned jmp_buf would save an
 * untagged ra and fault only when a longjmp is finally taken -- far from the cause. */
#ifndef _SETJMP_H
#define _SETJMP_H 1
typedef struct { unsigned long __opaque[28]; } __attribute__((aligned(16))) jmp_buf[1];
int  setjmp(jmp_buf);
void longjmp(jmp_buf, int) __attribute__((noreturn));
#define _setjmp  setjmp
#define _longjmp longjmp
#define sigsetjmp(b, s)  setjmp(b)
#define siglongjmp(b, v) longjmp(b, v)
#endif
