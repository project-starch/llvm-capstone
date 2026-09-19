#include "contexts-workload.h"
#include <stdio.h>
int main(void) {
  for (unsigned kind = 0; kind < 3; ++kind) {
    unsigned result = contexts_workload(kind);
    printf("context %u result %u policy %lu\n", kind, result, policy_hash);
    if (result)
      return 1;
  }
  return 0;
}
