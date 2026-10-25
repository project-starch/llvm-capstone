/* Hosted main: reaching the closing brace returns 0 (C11 5.1.2.2.3). The work
 * before it leaves a non-zero value in the return register, so a freestanding
 * build, where main is an ordinary function, exits with that value instead. */
#include <stdio.h>

int main(void)
{
    printf("HOSTED_NORETURN:%d\n", 41);
}
