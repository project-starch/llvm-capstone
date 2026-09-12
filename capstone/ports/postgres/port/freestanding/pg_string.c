/* The two string functions the repository's freestanding set does not carry.
 *
 * The set is capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c,
 * and the domain build links it: memcmp, memcpy, memmove, memset, strcmp,
 * strcpy and strlen. It is used and not reimplemented because its copies
 * preserve capability tags, which is not a detail a second implementation
 * would get right by accident: a byte loop copies the address bits of a
 * pointer and drops its out-of-band tag, so the copy comes back untagged and
 * the next dereference faults. That file copies the aligned middle one
 * capability at a time and carries the measured workarounds for the silicon
 * defects on that path.
 *
 * This one was that second implementation until the first run, where the
 * memory manager's own realloc of a block moved the block's prev and next
 * pointers with a byte loop and the store through the copy faulted with an
 * unexpected operand type (aset.c:1246, cause 24).
 *
 * strcat and strnlen are what is left: the census named eleven libc symbols
 * and the set covers nine of them.
 */
#include <stddef.h>

size_t
strnlen(const char *s, size_t limit)
{
    size_t n = 0;

    while (n < limit && s[n])
        n++;
    return n;
}

char *
strcat(char *dst, const char *src)
{
    char *d = dst;

    while (*d)
        d++;
    while ((*d++ = *src++))
        ;
    return dst;
}
