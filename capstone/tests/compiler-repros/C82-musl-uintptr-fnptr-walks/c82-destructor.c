/* C-82 positive control: an image with a non-empty .fini_array.
 *
 * The defect is musl's libc_exit_fini walking .fini_array through a uintptr_t, which untags the
 * entry pointer and makes the LDC of the entry a cause-24 fault. Every image shipped so far has an
 * EMPTY .fini_array, so the site is present and never executed -- which means the decode gate's
 * "0 refused" on those images proves nothing about the rule. This image populates the array with
 * one entry, so the walk actually runs.
 *
 * It must print its marker BEFORE exit, so a fault at exit is distinguishable from a fault earlier:
 * MAIN_OK present + DTOR_RAN absent + a fault at exit = the defect; MAIN_OK absent = something
 * else and the run says nothing about C-82.
 */
#include <stdio.h>

__attribute__((destructor)) static void dtor(void)
{
    puts("C82_DTOR_RAN");
    fflush(stdout);
}

int main(void)
{
    puts("C82_MAIN_OK");
    fflush(stdout);
    return 0;
}
