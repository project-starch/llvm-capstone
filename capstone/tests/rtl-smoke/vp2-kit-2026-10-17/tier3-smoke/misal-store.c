/* Q-34's store side: a misaligned PLAIN store inside a virtual context (2026-10-26).
 *
 * One 32-byte malloc object; every access below stays inside it, so no capability check can fire. The access width
 * and offset are the only variables:
 *   aligned-sw   sw at +4   (control: returns on every platform)
 *   aligned-sd   sd at +8   (control: returns on every platform)
 *   misal-lw     lw at +1   (Q-34's load, the positive control: silicon cause 4, the twin completes)
 *   misal-sw     sw at +1   (predicted: silicon cause 6, the twin completes)
 *   misal-sd     sd at +1   (predicted: silicon cause 6, the twin completes)
 *   misal-sd4    sd at +4   (4-aligned, not 8-aligned; predicted: silicon cause 6, the twin completes)
 * A step that completes prints "MISAL <which> RETURNED <value read back>" and exits 0; one that faults prints only its
 * mark, and capstone-exec reports the domain fault.
 *
 *   misal-store.dom aligned-sw | aligned-sd | misal-lw | misal-sw | misal-sd | misal-sd4
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline, used)) void misal_store_w(volatile uint32_t *p, uint32_t v) { *p = v; }
__attribute__((noinline, used)) void misal_store_d(volatile uint64_t *p, uint64_t v) { *p = v; }
__attribute__((noinline, used)) uint32_t misal_load_w(const volatile uint32_t *p) { return *p; }

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *which = argc == 2 ? argv[1] : "";
  unsigned char *b = malloc(32);
  if (!b) return 75;
  memset(b, 0x5a, 32);
  unsigned long long got;
  printf("MISAL %s mark\n", which);
  if (!strcmp(which, "aligned-sw")) {
    misal_store_w((volatile uint32_t *)(b + 4), 0x11223344u);
    got = misal_load_w((const volatile uint32_t *)(b + 4));
  } else if (!strcmp(which, "aligned-sd")) {
    misal_store_d((volatile uint64_t *)(b + 8), 0x1122334455667788ull);
    got = *(volatile uint64_t *)(b + 8);
  } else if (!strcmp(which, "misal-lw")) {
    got = misal_load_w((const volatile uint32_t *)(b + 1));
  } else if (!strcmp(which, "misal-sw")) {
    misal_store_w((volatile uint32_t *)(b + 1), 0x11223344u);
    got = b[1] | b[2] << 8 | b[3] << 16 | (unsigned long long)b[4] << 24;
  } else if (!strcmp(which, "misal-sd")) {
    misal_store_d((volatile uint64_t *)(b + 1), 0x1122334455667788ull);
    got = 0;
    for (int i = 8; i >= 1; i--) got = got << 8 | b[i];
  } else if (!strcmp(which, "misal-sd4")) {
    misal_store_d((volatile uint64_t *)(b + 4), 0x1122334455667788ull);
    got = 0;
    for (int i = 11; i >= 4; i--) got = got << 8 | b[i];
  } else {
    fprintf(stderr, "usage: %s aligned-sw|aligned-sd|misal-lw|misal-sw|misal-sd|misal-sd4\n", argv[0]);
    return 75;
  }
  printf("MISAL %s RETURNED 0x%llx\n", which, got);
  return 0;
}
