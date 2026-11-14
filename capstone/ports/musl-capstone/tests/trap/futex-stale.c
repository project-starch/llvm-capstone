#include <errno.h>
#include "syscall.h"

static long sample(int *word)
{
    /* The live word differs from the expected value, so this never sleeps. */
    return __syscall(SYS_futex, word, 128, 99, 0, 0, 0);
}

int main(void)
{
    int word = 7;
    int *old, *fresh;
    __asm__ volatile(".insn r 0x5b, 1, 0x51, %0, %1, %2"
        : "=r"(old) : "r"(&word), "r"(4UL) : "memory");
    if (sample(old) != -EAGAIN) return 1;
    __asm__ volatile(".insn r 0x5b, 1, 0x54, %0, %1, zero"
        : "=r"(fresh) : "r"(old) : "memory");
    *fresh = 8;
    if (sample(old) != -EFAULT) return 2;
    if (sample(fresh) != -EAGAIN) return 3;
    /* WAKE_OP with SET(0): a stale second word must fail without writing. */
    if (__syscall(SYS_futex, fresh, 128 | 5, 0, 0, old, 0) != -EFAULT) return 4;
    if (*fresh != 8) return 5;
    if (__syscall(SYS_futex, fresh, 128 | 5, 0, 0, fresh, 0)) return 6;
    return *fresh != 0;
}
