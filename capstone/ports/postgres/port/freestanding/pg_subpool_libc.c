/* malloc and its family, under the Sublet port, and they all refuse.
 *
 * The port replaces aset.c's calls to the level below with calls to
 * pg_subpool.c, so nothing in the ported path reaches these.  The other three
 * context types in the manager still call malloc, and mcxt.c's method table
 * names their functions, so the link needs them to exist.
 *
 * They fail loudly rather than quietly, and that is the whole point of the
 * file.  A stub that returned a pointer into some scratch would give a
 * context type the discipline does not cover a heap it could use, and the
 * measurement would then report a protected run that was partly unprotected.
 * A stub that was simply absent would be a link error naming a symbol, which
 * says less than a message naming the reason.
 *
 * The recorded pgbench rungs create 23 096 and 5 123 contexts and every one of
 * them is an aset, so this file is a guard against a workload that is not one
 * of those rather than a gap in the ones measured.  Both numbers are in
 * experiments/a11/postgres in the paper's repository.
 */
#include <stddef.h>

void pg_domain_text(const char *s);
__attribute__((noreturn)) void pg_subpool_refuse(const char *what);

static void
refuse(const char *what)
{
    pg_domain_text("pg-sublet: ");
    pg_domain_text(what);
    pg_domain_text(" was called.  Only aset contexts are ported to Sublet;\n"
                   "  a generation, slab or bump context would need its own "
                   "sub-pool discipline.\n");
    pg_subpool_refuse(what);
}

void *malloc(size_t n) { (void) n; refuse("malloc"); return NULL; }
void *calloc(size_t k, size_t n) { (void) k; (void) n; refuse("calloc"); return NULL; }
void *realloc(void *p, size_t n) { (void) p; (void) n; refuse("realloc"); return NULL; }
void free(void *p) { (void) p; refuse("free"); }
