/* Per-request scratch. Bump supports bulk reset/delete, not pfree/repalloc. */
#include "client.h"

int client_run(MemoryContext parent) {
  MemoryContext scratch =
      BumpContextCreate(parent, "request scratch", 0, 2048, 8192);
  unsigned total = 0; /* a value, not a pointer into the reset context */
  int failed = 0;

  for (unsigned request = 0; request < 3; ++request) {
    unsigned *values = MemoryContextAlloc(scratch, 16 * sizeof(*values));
    for (unsigned i = 0; i < 16; ++i)
      values[i] = request + i;
    for (unsigned i = 0; i < 16; ++i)
      total += values[i];
    failed |= values[15] != request + 15;

    MemoryContextReset(scratch); /* all pointers from this request expire */
    values = NULL;
    /* The next request reuses the context, with fresh allocation authority. */
  }
  failed |= total != 408;
  MemoryContextDelete(scratch);
  return failed;
}
