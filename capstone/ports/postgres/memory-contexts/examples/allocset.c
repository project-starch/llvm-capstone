/* Variable-sized request data, with both explicit and current-context APIs. */
#include "client.h"

int client_run(MemoryContext parent) {
  MemoryContext request =
      AllocSetContextCreate(parent, "request", ALLOCSET_DEFAULT_SIZES);
  int failed = 0;

  /* Explicit ownership: allocation does not depend on CurrentMemoryContext. */
  unsigned *values = MemoryContextAlloc(request, 4 * sizeof(*values));
  for (unsigned i = 0; i < 4; ++i)
    values[i] = i + 10;

  /* repalloc may move the object. Keep its return value, not an old alias. */
  values = repalloc(values, 8 * sizeof(*values));
  for (unsigned i = 0; i < 4; ++i)
    failed |= values[i] != i + 10;
  for (unsigned i = 4; i < 8; ++i)
    values[i] = i + 10;
  failed |= values[7] != 17;
  pfree(values);
  values = NULL;

  /* palloc uses the current context; always restore it before deleting one. */
  MemoryContext previous = MemoryContextSwitchTo(request);
  char *label = palloc(6);
  memcpy(label, "hello", 6);
  failed |= strcmp(label, "hello") != 0;
  MemoryContextSwitchTo(previous);

  MemoryContextReset(request); /* releases label; request itself survives */
  label = NULL;
  label = MemoryContextAlloc(request, 4);
  memcpy(label, "new", 4);
  failed |= strcmp(label, "new") != 0;
  MemoryContextDelete(request); /* releases all remaining allocations */
  return failed;
}
