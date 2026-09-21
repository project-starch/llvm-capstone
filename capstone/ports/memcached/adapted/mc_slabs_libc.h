/* The eleven system headers slabs.c includes, reduced to what the patched
 * allocator still calls. Included by slabs.c ONLY: it renames stdio, getenv
 * and exit, which a hosted program linking the allocator must not see.
 * malloc, realloc and free are not declared: the port's second patch replaces
 * every call, and one it missed must fail to compile. */
#ifndef MC_SLABS_LIBC_H
#define MC_SLABS_LIBC_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void *memset(void *, int, size_t);

/* util.h:19, reached only through getenv, which answers nothing here: the
 * test suite's T_MEMD_INITIAL_MALLOC knob is not consulted. */
bool safe_strtoll(const char *str, int64_t *out);
#define getenv mc_getenv
char *mc_getenv(const char *);

/* An allocator that starts talking has left the measured path: every message
 * and every exit ends the run with a code instead, hosted as in a domain. */
typedef struct mc_stream FILE;
#define stderr ((FILE *)0)
#define fprintf mc_fprintf
int mc_fprintf(FILE *, const char *, ...);
#define exit mc_exit
_Noreturn void mc_exit(int);
#endif
