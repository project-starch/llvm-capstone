/* Complete-interpreter glue for the existing pymalloc PoisonCap component.
 * The allocator's payload and metadata remain separate; one binary selects
 * spatial or explicit nested revocation with PYM_POISONCAP_MODE=0/1. */
#include "port.h"
#include <cheri/cheric.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define PAYLOAD_BYTES (64UL * 1024 * 1024)
#define METADATA_BYTES (16UL * 1024 * 1024)

static void report(void) {
  struct pym_header counts = {0};
  pym_backing_stats(&counts);
  fprintf(stderr, "PYM_INTERPRETER_METADATA bytes=%llu\n",
          (unsigned long long)counts.metadata);
}

_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "PYM_INTERPRETER_FAIL code=%u\n", code);
  _Exit(code ? (int)(code & 255) : 1);
}

__attribute__((constructor)) static void init_pymalloc_lifetimes(void) {
  const char *choice = getenv("PYM_POISONCAP_MODE");
  if (!choice || (choice[0] != '0' && choice[0] != '1') || choice[1])
    pym_fail(731);
  void *base = mmap(NULL, PAYLOAD_BYTES + 65536, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
  void *metadata = mmap(NULL, METADATA_BYTES, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED || metadata == MAP_FAILED)
    pym_fail(732);
  void *payload = __builtin_align_up(base, 65536);
  if (cheri_getlen(payload) < PAYLOAD_BYTES)
    pym_fail(733);
  pym_lifetime_init(payload);
  pym_backing_init(metadata, NULL);
  pym_set_mode((unsigned)(choice[0] - '0'));
  if (atexit(report))
    pym_fail(734);
  fprintf(stderr, "PYM_INTERPRETER_POLICY mode=%c payload_reservation=%lu "
          "metadata_reservation=%lu\n", choice[0], PAYLOAD_BYTES,
          METADATA_BYTES);
}
