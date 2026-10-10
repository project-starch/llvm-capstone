/* Initialize the existing inner allocator before entering the application. */
int __real_main(int, char **);
#ifdef EXP_PG_CONTEXT_SUBLET
#include <stddef.h>
void *__capstone_region(unsigned);
unsigned long pg_subpool_arena(void *, unsigned long);
#endif
int __wrap_main(int argc, char **argv) {
#ifdef EXP_PG_CONTEXT_SUBLET
  if (pg_subpool_arena(__capstone_region(1), 64UL << 20)) return 126;
#endif
  return __real_main(argc, argv);
}
