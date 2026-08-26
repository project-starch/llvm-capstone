/* strncmp and strchr, the two the MicroPython core needs that
   benchmarks/beebs/adapted/beebs_freestanding_string.c does not provide.
   They live here rather than being added there because that file is shared with every BEEBS
   rung and with the SQLite domain, and those artifacts are what published measurements were
   taken on; adding symbols to it would relink all of them.

   Written as pointer arithmetic throughout: on this target a char* is a capability, and
   walking it with `++` keeps the tag, while indexing through an integer would not. */
#include <stddef.h>

int strncmp(const char *a, const char *b, size_t n) {
    while (n--) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        if (ca != cb) {
            return (int)ca - (int)cb;
        }
        if (ca == 0) {
            break;
        }
    }
    return 0;
}

char *strchr(const char *s, int c) {
    const char target = (char)c;
    for (;; ++s) {
        if (*s == target) {
            return (char *)s;
        }
        if (*s == 0) {
            return NULL;   /* NUL matches only if it is what was asked for, handled above */
        }
    }
}
