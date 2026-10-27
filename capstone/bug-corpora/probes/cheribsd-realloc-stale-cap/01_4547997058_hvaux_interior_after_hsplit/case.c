/* Perl Gv_AMupdate stale HvAUX, upstream fix 4547997058 (RT #121362). Standalone reduction of the
 * allocator shape; the measured multi-platform instrument is ../probe.c (CheriBSD) and
 * ../probe-capstone.c (Capstone virtual), scenario perl-hsplit-aux-grow. See PROVENANCE.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const unsigned long first = 200, second = 224, interior = 192;
  unsigned char *base = malloc(first);
  if (!base) return 2;
  memset(base, 0x5a, first);
  unsigned char *aux = base + interior;        /* HvAUX(stash): interior into the bucket tail */
  unsigned char *grown = realloc(base, second); /* S_hsplit's Renew */
  if (!grown) return 2;
  /* pre-fix: write the flag through the stale interior aux; the fix re-reads HvAUX(stash). */
  volatile unsigned char *target = aux;
  *target = 0xa5;                              /* stale interior store; faults under revocation */
  printf("same_addr=%d aliased=%d\n", grown == base, grown[interior] == 0xa5);
  free(grown);
  return 0;
}
