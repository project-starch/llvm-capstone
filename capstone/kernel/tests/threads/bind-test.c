/* Exercise the actual step 5 binder with the step 8 extensions. */
#include <stdio.h>
#include <stdlib.h>
#include "bind.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

int main(void)
{
    struct sublet_bind_args a = { .arg = {0, 0x10000, 0x20000, 0x30000, 0x40000}, .tagged = 0x1e };
    CHECK(!sublet_bind_core(__NR_clone, &a));
    for (int i = 1; i < 5; ++i)
        CHECK(a.arg[i] == sublet_handle(i * 0x10000, i + 1));
    a = (struct sublet_bind_args){ .arg = {0x10000, 0, 7, 0x20000}, .tagged = 9 };
    CHECK(!sublet_bind_core(__NR_futex, &a));
    CHECK(a.arg[0] == sublet_handle(0x10000, 1));
    CHECK(a.arg[3] == sublet_handle(0x20000, 4));
    /* Requeue's fourth argument is a count, even if it exceeds the address mask. */
    a = (struct sublet_bind_args){ .arg = {0x10000, 3, 0, ~0UL, 0x10000}, .tagged = 0x11 };
    CHECK(!sublet_bind_core(__NR_futex, &a));
    CHECK(a.arg[3] == ~0UL);
    CHECK(a.arg[0] != a.arg[4]);
    CHECK((a.arg[0] & SUBLET_HANDLE_ADDR_MASK) == (a.arg[4] & SUBLET_HANDLE_ADDR_MASK));
    a = (struct sublet_bind_args){ .arg = {sublet_handle(0x10000, 4), 0} };
    CHECK(sublet_bind_core(__NR_futex, &a) == -EFAULT);
    a = (struct sublet_bind_args){ .arg = {0x10000, 6}, .tagged = 1 };
    CHECK(sublet_bind_core(__NR_futex, &a) == -ENOSYS); /* PI is not admitted. */
    a = (struct sublet_bind_args){ .arg = {0x10000}, .tagged = 1 };
    CHECK(!sublet_bind_core(__NR_set_tid_address, &a));
    CHECK(a.arg[0] == sublet_handle(0x10000, 1));
    puts("PASS: clone references, futex timeout/count distinction, forged handle refusal, exit word");
    return 0;
}
