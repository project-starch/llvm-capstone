/* A complete client main linked only to PostgreSQL::MemoryContexts. */
#include "postgres.h"
#include "utils/memutils.h"
#include <stdio.h>

int main(void) {
  MemoryContext root = AllocSetContextCreate(NULL, "root", ALLOCSET_DEFAULT_SIZES);
  TopMemoryContext = CurrentMemoryContext = root;
  MemoryContext child = AllocSetContextCreate(root, "request", ALLOCSET_DEFAULT_SIZES);
  char *text = MemoryContextAlloc(child, 32);
  memcpy(text, "preserved", 10);
  char **holder = MemoryContextAlloc(child, sizeof(*holder));
  *holder = text;
  holder = repalloc(holder, 1024);
  if (strcmp(*holder, "preserved"))
    return 1;
  pfree(holder);
  MemoryContextReset(child);
  text = MemoryContextAlloc(child, 32);
  memcpy(text, "next", 5);
  if (strcmp(text, "next"))
    return 1;
  MemoryContextDelete(root);
  TopMemoryContext = CurrentMemoryContext = NULL;
  printf("ALLOCATOR_EXAMPLE postgres PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
