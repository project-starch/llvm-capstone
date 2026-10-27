/* The eight symbols zend_hash.c needs, so it can be linked WITHOUT the rest of the engine.
 * Keeps the reproducer small enough to hand to a compiler-repro directory. */
#include <zend.h>

void *malloc(unsigned long);
void  free(void *);
void *calloc(unsigned long, unsigned long);
void *realloc(void *, unsigned long);

ZEND_API void *_emalloc(size_t size)                  { return malloc(size); }
ZEND_API void  _efree(void *p)                        { free(p); }
ZEND_API void *_ecalloc(size_t n, size_t sz)          { return calloc(n, sz); }
ZEND_API void *_erealloc(void *p, size_t sz, int af)  { (void)af; return realloc(p, sz); }
ZEND_API char *_estrndup(const char *s, unsigned int len)
{
    char *p = (char *) malloc((unsigned long)len + 1);
    if (p) { unsigned int i; for (i = 0; i < len; i++) { p[i] = s[i]; } p[len] = 0; }
    return p;
}
ZEND_API void zend_error(int type, const char *fmt, ...) { (void)type; (void)fmt; }

/* These are function POINTERS in zend.h:490-491, and HANDLE_BLOCK_INTERRUPTIONS only calls
 * them when non-NULL, so NULL is the correct "no interrupt blocking" value. */
ZEND_API void (*zend_block_interruptions)(void);
ZEND_API void (*zend_unblock_interruptions)(void);
