/* SPDX-License-Identifier: MIT */
#include <unistd.h>
int main(void)
{
    volatile unsigned long value = 1;
    if (write(1, "R", 1) != 1) return 1;
    /* No services after readiness: scheduling/signals must interrupt compute. */
    for (;;) value = value * 6364136223846793005UL + 1;
}
