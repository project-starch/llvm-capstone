#include "port.h"
#include <string.h>
#ifdef PYMALLOC_POISONCAP
#include <stdio.h>
#endif
#define CHECK(x, n)                                                            \
  do {                                                                         \
    if (!(x))                                                                  \
      pym_fail(n);                                                             \
  } while (0)
static volatile unsigned char *held;
__attribute__((noinline)) static unsigned
read_probe(const volatile unsigned char *p) {
#ifdef PYMALLOC_POISONCAP
  return *p;
#else
  unsigned long value;
  __asm__ volatile(".globl pym_probe_read\npym_probe_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
#endif
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
#ifdef PYMALLOC_POISONCAP
  *p = 93;
#else
  unsigned long value = 93;
  __asm__ volatile(
      ".globl pym_probe_write\npym_probe_write:\nsb %0, 0(%1)\n" ::"r"(value),
      "r"(p)
      : "memory");
#endif
}
static void mark(unsigned id) {
#ifdef PYMALLOC_POISONCAP
  printf("PYM_PROBE case=%u ready\n", id);
  fflush(stdout);
#else
  extern void pym_probe_read(void), pym_probe_write(void);
  unsigned long value = 0xcf13000000000000UL | id;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(value),
                   "r"(pym_probe_read), "r"(pym_probe_write)
                   : "memory");
#endif
}
void pym_replay(const struct pym_header *input, struct pym_header *out,
                void *scratch) {
  (void)scratch;
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->mode = mode;
  out->magic = PYM_MAGIC;
  out->count = 1;
  const struct pym_event *e = (const void *)(input + 1);
  unsigned test = e->id;
  CHECK(input->magic == PYM_MAGIC && input->count == 1 && test <= 8, 601);
  size_t size = test == 6 ? 513 : test == 7 ? 16 : 64;
  unsigned char *a = pym_malloc(size),
                *sibling = pym_malloc(test == 7 ? 32 : size);
  CHECK(a && sibling, 602);
  held = a;
  a[0] = 17;
  sibling[0] = 41;
  CHECK(read_probe(held) == 17, 603);
  uintptr_t address = (uintptr_t)a;
  if (!test) {
    pym_free(a);
    a = pym_malloc(size);
    CHECK((uintptr_t)a == address && sibling[0] == 41, 604);
    /* A capability-bearing payload must survive moved realloc. */
    void **container = pym_malloc(32);
    container[0] = sibling;
    container = pym_realloc(container, 1024);
    CHECK(container && ((unsigned char *)container[0])[0] == 41, 605);
    CHECK(pym_realloc(a, PYM_ARENA_BYTES) == NULL && sibling[0] == 41, 606);
    pym_free(container);
  } else if (test == 5) {
    mark(test);
    (void)read_probe(held + size);
  } else {
    if (test == 4) {
      a = pym_realloc(a, size - 1);
      CHECK((uintptr_t)a == address && a[0] == 17, 607);
    } else {
      pym_free(a);
      a = NULL;
      if (test != 1) {
        unsigned rounds = test == 8 ? 2000 : 1;
#ifdef PYMALLOC_POISONCAP
        if (test == 8 && e->size)
          rounds = e->size;
#endif
        for (unsigned i = 0; i < rounds; ++i) {
          a = pym_malloc(test == 7 ? 512 : size);
          CHECK((uintptr_t)a == address, 608);
          a[0] = 59;
          CHECK(a[0] == 59 && sibling[0] == 41, 609);
          if (i + 1 < rounds)
            pym_free(a);
        }
      }
    }
    CHECK(sibling[0] == 41, 610);
    mark(test);
    if (test == 2)
      write_probe(held);
    else if (test == 3) {
      pym_free((void *)held);
      a = NULL;
    } else
      (void)read_probe(held);
  }
  pym_free(a);
  pym_free(sibling);
  if (!test)
    mark(test);
  out->completed = 1;
  pym_backing_stats(out);
}
