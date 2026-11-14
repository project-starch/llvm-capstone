/* Three large allocations live together, an aligned one, and a printed double: mmap must place
 * each block where its bounds are exact, and the FPU must be on. Build it like the libc's trap
 * smoke programs (ports/musl-capstone/build-trap-smoke.sh) and pass the .dom to run.sh. It
 * prints "big-alloc: ok, (0.1 + 0.2) * 1e17 = 30000000000000004" and exits 0. */
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* One line with a decimal number, without printf (whose long double needs compiler-rt). */
static void say(const char *text, long n)
{
    char digits[24];
    int at = sizeof digits;

    digits[--at] = '\n';
    do {
        digits[--at] = '0' + n % 10;
        n /= 10;
    } while (n);
    write(1, text, strlen(text));
    write(1, digits + at, sizeof digits - at);
}

int main(void)
{
    size_t sizes[3] = {16UL << 20, 64UL << 20, 256UL << 20};
    char *p[3], *q;
    volatile double a = 0.1, b = 0.2;

    for (int i = 0; i < 3; ++i) {
        p[i] = malloc(sizes[i]);
        if (!p[i]) {
            say("big-alloc: malloc failed, errno ", errno);
            return 1 + i;
        }
        p[i][0] = 1;
        p[i][sizes[i] - 1] = 2;
    }
    q = aligned_alloc(64UL << 20, 64UL << 20);
    if (!q) {
        say("big-alloc: aligned_alloc failed, errno ", errno);
        return 4;
    }
    q[0] = 3;
    say("big-alloc: ok, (0.1 + 0.2) * 1e17 = ", (long)((a + b) * 1e17));
    return 0;
}
