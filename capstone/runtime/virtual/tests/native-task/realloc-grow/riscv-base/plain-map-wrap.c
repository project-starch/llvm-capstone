/* MEASUREMENT ONLY: riscv-base mcf relinked with --wrap=__plain_map: each mapping of 1 MiB or more, and whether
 * plain_reuse.c served it from its cache (its hit counter rose), written to stderr at exit. */
#include <stdio.h>
#include <stddef.h>
extern unsigned long __plain_reuse_hits, __plain_reuse_puts, __plain_reuse_evicted;
void *__real___plain_map(size_t);
static struct { size_t len; int hit; } big[256]; static unsigned nbig;
void *__wrap___plain_map(size_t len)
{
    unsigned long h = __plain_reuse_hits;
    void *p = __real___plain_map(len);
    if (len >= (1u << 20) && nbig < 256) { big[nbig].len = len; big[nbig].hit = __plain_reuse_hits != h; ++nbig; }
    return p;
}
__attribute__((destructor)) static void pr_report(void)
{
    fprintf(stderr, "PLAIN-REUSE hits=%lu puts=%lu evicted=%lu big=%u\n", __plain_reuse_hits, __plain_reuse_puts, __plain_reuse_evicted, nbig);
    for (unsigned i = 0; i < nbig; ++i) fprintf(stderr, "PLAIN-BIG %zu %s\n", big[i].len, big[i].hit ? "hit" : "mmap");
}
