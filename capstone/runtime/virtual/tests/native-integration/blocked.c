#include <unistd.h>
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
int main(void)
{
    char byte = 'R';
    if (__capstone_native_syscall(64, (void *)1, &byte, (void *)1, 0, 0, 0) != 1) return 1;
    if (__capstone_native_syscall(63, 0, &byte, (void *)1, 0, 0, 0) != 1) return 2;
    return __capstone_native_syscall(64, (void *)1, &byte, (void *)1, 0, 0, 0) == 1 ? 0 : 3;
}
