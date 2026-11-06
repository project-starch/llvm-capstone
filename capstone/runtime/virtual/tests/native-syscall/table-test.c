/* SPDX-License-Identifier: MIT */
#include <capstone/native-syscall.h>
#include <stdio.h>
#include <string.h>
int main(void)
{
    size_t count = 0;
    const struct ns_descriptor *d;
    while ((d = ns_descriptor_at(count))) {
        if (ns_descriptor(d->nr) != d) return 1;
        for (size_t j = 0; j < count; ++j)
            if (ns_descriptor_at(j)->nr == d->nr) return 1;
        if (d->route != NS_GRANT && !capstone_delegate_shape(d->nr)) return 1;
        ++count;
    }
    if (count != 109 || ns_descriptor(UINT64_MAX) ||
        ns_descriptor(CAPSTONE_SYS_dup3)->route != NS_RESOURCE ||
        ns_descriptor(CAPSTONE_NR_IOCTL_BUF)->route != NS_REFUSE ||
        ns_descriptor(NS_MAP)->route != NS_GRANT ||
        ns_descriptor(CAPSTONE_SYS_readv)->route != NS_VECTOR) return 1;
    printf("PASS descriptor inventory: %zu unique rows, unknown numbers refused\n", count);
    return 0;
}
