#include "client.h"
#include <stdio.h>

int main(void) {
  MemoryContext root =
      AllocSetContextCreateInternal(NULL, "example root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = root;
  int result = client_run(root);
  MemoryContextDelete(root);
  TopMemoryContext = CurrentMemoryContext = NULL;
  printf("PG_CLIENT " PG_CLIENT_NAME " RESULT %d\n", result);
  return result ? 1 : 0;
}
