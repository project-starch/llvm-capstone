/* Full decoder, with its actual pool leases observed in-process. */
#include <capstone/capability.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "trace.h"

#define POOL_BYTES (256UL << 20)

extern void *__capstone_region(unsigned);
extern int ffdecode_main(int, char **);
static capstone_cap_slot grant, remainder;

int main(int argc, char **argv) {
  if (argc != 4) return 64;
  unsigned mode = (unsigned)atoi(argv[1]);
  if (mode != 0 && mode != 2) return 64;
  ff2_set_mode(mode);
  capstone_cap_store(&grant, __capstone_region(0));
  size_t available = capstone_cap_end(&grant) - capstone_cap_base(&grant);
  if (capstone_cap_type(&grant) != CAPSTONE_CAP_LINEAR || available < POOL_BYTES)
    return 74;
  if (available > POOL_BYTES)
    capstone_cap_split(&grant, capstone_cap_base(&grant) + POOL_BYTES, &remainder);
  void *payload = grant.c;
  grant.c = NULL;
  ff2_payload_init(payload, POOL_BYTES);
  int result = ffdecode_main(argc - 1, argv + 1);
  struct ff2_header h = {0};
  ff2_memory_report(&h);
  ff2_reuse_report();
  fprintf(stderr, "EXP-POOL mode=%u payload=%llu split=%llu mrev=%llu "
          "delin=%llu revoke=%llu init=%llu init_bytes=%llu\n", mode,
          (unsigned long long)h.payload_used, (unsigned long long)h.split,
          (unsigned long long)h.mrev, (unsigned long long)h.delin,
          (unsigned long long)h.revoke, (unsigned long long)h.init,
          (unsigned long long)h.init_bytes);
  return result;
}
