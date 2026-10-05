#ifndef NATIVE_SURVEY_OBSERVER_H
#define NATIVE_SURVEY_OBSERVER_H
#include <stddef.h>

enum ns_family {
    NS_SQLITE_LOOKASIDE = 1, NS_SQLITE_MEMSYS5,
    NS_PG_ASET, NS_PG_GENERATION, NS_PG_SLAB, NS_PG_BUMP,
    NS_PYMALLOC, NS_MRUBY_GC, NS_PERL_SV,
    NS_AVBUFFER, NS_AVREFSTRUCT,
    NS_WMEM_SIMPLE, NS_WMEM_STRICT, NS_WMEM_BLOCK, NS_WMEM_BLOCK_FAST,
    NS_MC_SLAB, NS_MC_CACHE, NS_FAMILY_COUNT
};

/* Null results are failed allocations, not successful object issues. */
void ns_alloc(unsigned family, const void *instance, const void *p, size_t size);
void ns_free(unsigned family, const void *instance, const void *p);
void ns_noop_free(unsigned family);
/* Only call after success. A null result with positive size preserves oldp. */
void ns_resize(unsigned family, const void *instance, const void *oldp,
               const void *newp, size_t size, int retires_old);
void ns_bulk(unsigned family, const void *instance);
void ns_destroy(unsigned family, const void *instance);
/* For arenas obtained without malloc. Do not register overlapping ranges. */
void ns_backing_acquire(const void *p, size_t size);
void ns_backing_release(const void *p);
void ns_report(void);
#endif
