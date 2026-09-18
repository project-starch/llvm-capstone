#include "contexts-workload.h"
#include "domain-runtime.h"
#ifdef PG_CONTEXTS_SUBLET
#include "pg_subpool.h"
#else
void pg_level0_init(void *, size_t);
#endif

static unsigned long arena_type;
static const volatile unsigned *selection;
static unsigned char *volatile held;
static unsigned char *volatile interior;

_Noreturn void pg_subpool_refuse(const char *why) {
  fail(why);
  give_up(0xbad80001);
}
static void check(int condition) {
  if (!condition)
    give_up(0xbad80002);
}
__attribute__((noinline)) static unsigned
probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl pg_context_probe\npg_context_probe:\nlbu %0, 0(%1)"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  __asm__ volatile(
      ".globl pg_context_write\npg_context_write:\nsb %0, 0(%1)" ::"r"(93UL),
      "r"(p)
      : "memory");
}
static void mark(unsigned kind, unsigned test) {
  extern void pg_context_probe(void), pg_context_write(void);
  unsigned long code = 0xcf17000000000000UL | (kind << 8) | test;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(code),
                   "r"(pg_context_probe), "r"(pg_context_write)
                   : "memory");
}

static void lifetime(unsigned kind, unsigned test) {
  MemoryContext root =
      AllocSetContextCreateInternal(NULL, "root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = root;
  MemoryContext c = make_context(kind, root);
  MemoryContext sibling = make_context((kind + 1) % 3, root);
  unsigned char *p = MemoryContextAlloc(c, 64);
  unsigned char *q = MemoryContextAlloc(c, 64);
  unsigned char *s = MemoryContextAlloc(sibling, 64);
  p[0] = 17;
  p[16] = 19;
  q[0] = 23;
  s[0] = 29;
  held = p;
  interior = p + 16; /* form before revocation, not at the later probe */
  check(probe(held) == 17);
  if (test == 1 || test == 5) {
    check(kind != 2);
    pfree(p);
    check(q[0] == 23);
    unsigned char *fresh = MemoryContextAlloc(c, 64);
    fresh[0] = 31;
    if (kind == 1)
      check((uintptr_t)fresh == (uintptr_t)held);
  } else if (test == 2 || test == 6) {
    MemoryContextReset(c);
    ((unsigned char *)MemoryContextAlloc(c, 64))[0] = 37;
  } else if (test == 3) {
    MemoryContextDelete(c);
    c = make_context(kind, root);
    ((unsigned char *)MemoryContextAlloc(c, 64))[0] = 41;
  } else if (test == 4) {
    MemoryContext child = make_context((kind + 1) % 3, c);
    held = MemoryContextAlloc(child, 64);
    held[0] = 43;
    MemoryContextDelete(c);
    c = NULL;
  } else if (test == 7) {
    /* OOM must not revoke or change existing live allocations. */
    check(MemoryContextAllocExtended(c, (Size)1 << 29, MCXT_ALLOC_NO_OOM) ==
          NULL);
    check(q[0] == 23);
  } else if (test == 8) {
    check(kind == 0);
    unsigned char *grown = repalloc(p, 256);
    check(grown[0] == 17 && grown[16] == 19 && q[0] == 23);
  } else if (test == 11) {
    check(kind != 2);
    pfree(p);
    pfree(q); /* Generation now recycles the entire current block. */
    ((unsigned char *)MemoryContextAlloc(c, 64))[0] = 53;
  } else if (test == 12) {
    check(kind != 1);
    held = MemoryContextAlloc(c, 40000);
    held[0] = 59;
    if (kind == 0)
      pfree(held);
    else
      MemoryContextReset(c);
  } else if (test == 13) {
    check(kind == 2);
    held = MemoryContextAlloc(c, 0);
    check(held != NULL);
  } else if (test == 14) {
    unsigned count;
    for (count = 0; count < 9000; ++count) {
      unsigned char *next =
          MemoryContextAllocExtended(c, 64, MCXT_ALLOC_NO_OOM);
      if (!next)
        break;
      next[0] = 61;
    }
#ifdef PG_CONTEXTS_SUBLET
    check(count > 0 && count < 9000);
#endif
    check(q[0] == 23 && probe(held) == 17);
    MemoryContextReset(c);
    held = MemoryContextAlloc(c, 64);
    held[0] = 67;
  } else if (test == 9) {
    /* Exactly one byte beyond the aligned 64-byte allocation. */
    held = p + 64;
  }
  check(s[0] == 29);
  mark(kind, test);
  if (test == 5 || test == 6)
    write_probe(interior);
  else
    (void)probe(held);
  if (c)
    MemoryContextDelete(c);
  MemoryContextDelete(root);
}

void pg_domain_entry(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    switch (shares++) {
    case 0:
      meta = (void *)res;
      break;
    case 1:
      payload = (void *)res;
      break;
    case 2:
#ifdef PG_CONTEXTS_SUBLET
      arena_type = pg_subpool_arena(res, PG_REPLAY_ARENA_SIZE);
#else
      pg_level0_init(res, PG_REPLAY_ARENA_SIZE);
#endif
      break;
    case 3:
      selection = (void *)res;
      break;
    }
    return;
  }
  domain_result = res;
  check(shares >= 4 && meta && payload && selection && arena_type == 0);
  meta->length = 0;
  pg_domain_payload((char *)payload, (unsigned long *)&meta->length,
                    PG_REPLAY_PAYLOAD_SIZE);
  unsigned kind = selection[0], test = selection[1];
  check(kind < 3 && test <= 14);
  if (test == 10) {
    unsigned result = contexts_workload(kind);
    pg_domain_text("PG_CONTEXT_RESULT ");
    pg_domain_uint(result);
    pg_domain_text(" POLICY ");
    pg_domain_uint(policy_hash);
    pg_domain_text("\n");
    check(result == 0);
  } else {
    lifetime(kind, test);
  }
  pg_domain_text("__CAPSTONE_PG_CONTEXTS_GOOD__\n");
  *res = 0;
}
