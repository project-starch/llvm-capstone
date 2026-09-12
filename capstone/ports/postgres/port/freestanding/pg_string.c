/* The eleven symbols PostgreSQL's memory manager takes from libc, minus the
 * four the level below provides.
 *
 * Seven functions, and the census named them rather than a guess doing it:
 * memcpy, memset, strlen, strcmp, strcpy, strcat, strnlen. memmove is here
 * too, because the compiler may lower a structure copy to it whether or not
 * the source says so.
 *
 * Byte at a time, on purpose. A word-at-a-time copy reads past the end of the
 * last word, which on a capability machine is a fault and not a harmless read,
 * and the manager's copies are short. If they ever show up in a measurement,
 * that is the moment to make them wider, with bounds that say why.
 */
#include <stddef.h>

void *
memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *d = dst;
    const unsigned char *s = src;

    while (n--)
        *d++ = *s++;
    return dst;
}

void *
memmove(void *dst, const void *src, size_t n)
{
    unsigned char *d = dst;
    const unsigned char *s = src;

    if (d == s || n == 0)
        return dst;
    if (d < s) {
        while (n--)
            *d++ = *s++;
    } else {
        d += n;
        s += n;
        while (n--)
            *--d = *--s;
    }
    return dst;
}

void *
memset(void *dst, int c, size_t n)
{
    unsigned char *d = dst;

    while (n--)
        *d++ = (unsigned char) c;
    return dst;
}

int
memcmp(const void *a, const void *b, size_t n)
{
    const unsigned char *x = a, *y = b;

    while (n--) {
        if (*x != *y)
            return *x < *y ? -1 : 1;
        x++; y++;
    }
    return 0;
}

size_t
strlen(const char *s)
{
    const char *p = s;

    while (*p)
        p++;
    return (size_t) (p - s);
}

size_t
strnlen(const char *s, size_t limit)
{
    size_t n = 0;

    while (n < limit && s[n])
        n++;
    return n;
}

int
strcmp(const char *a, const char *b)
{
    while (*a && *a == *b) {
        a++; b++;
    }
    return (int) (unsigned char) *a - (int) (unsigned char) *b;
}

char *
strcpy(char *dst, const char *src)
{
    char *d = dst;

    while ((*d++ = *src++))
        ;
    return dst;
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
