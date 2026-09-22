#include "port.h"
#include "scopes.h"
#include <string.h>
#define CHECK(x, n)                                                            \
  do {                                                                         \
    if (!(x))                                                                  \
      wm_fail(n);                                                              \
  } while (0)
__attribute__((noinline)) static unsigned
read_probe(const volatile unsigned char *p) {
  unsigned long x;
  __asm__ volatile(".globl wm_probe_read\nwm_probe_read:\nlbu %0, 0(%1)\n"
                   : "=r"(x)
                   : "r"(p)
                   : "memory");
  return x;
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long x = 93;
  __asm__ volatile(
      ".globl wm_probe_write\nwm_probe_write:\nsb %0, 0(%1)\n" ::"r"(x), "r"(p)
      : "memory");
}
/* Print the stage and the three labelled access sites, so the oracle can
 * require the exact faulting PC rather than any fault at all. */
static void mark(unsigned id) {
  extern void wm_probe_read(void), wm_probe_write(void), wm_widen_probe(void);
  unsigned long x = 0xcf15000000000000UL | id;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %3, x0\n" ::"r"(x),
                   "r"(wm_probe_read), "r"(wm_probe_write), "r"(wm_widen_probe)
                   : "memory");
}
void wm_replay(const struct wm_header *in, struct wm_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = WM_MAGIC;
  out->mode = mode;
  out->count = 1;
  const struct wm_event *e = (const void *)(in + 1);
  unsigned test = e->arg;
  CHECK(in->magic == WM_MAGIC && in->count == 1 && test <= 12, 501);
  wm_scopes_init();
  wm_enter_file_scope();
  /* The per-dissection pool (block_fast) and the file scope (block). */
  wmem_allocator_t *packet = wm_packet_pool_acquire();
  wmem_allocator_t *file = wm_file_scope();
  unsigned char *p = wmem_alloc(packet, 64);
  CHECK(p, 502);
  p[0] = 17;
  p[16] = 29;
  /* Materialize aliases while live. Volatile pointer slots keep the stale
   * accesses at their labelled instructions instead of faulting earlier
   * while deriving an interior pointer from already revoked authority. */
  unsigned char *volatile held = p;
  unsigned char *volatile interior = p + 16;
  uintptr_t address = (uintptr_t)p;
  unsigned char *q = wmem_alloc(file, 64);
  CHECK(q, 503);
  q[0] = 41;
  unsigned char *volatile held_q = q;
  uintptr_t address_q = (uintptr_t)q;
  CHECK(read_probe(held) == 17 && read_probe(held_q) == 41, 504);
  switch (test) {
  case 0: {
    /* Live controls. Resetting one pool leaves the other's objects intact,
     * and both allocators reissue the same storage after a reset: that reuse
     * is the property under study, asserted here rather than assumed. */
    wm_packet_pool_reset(packet);
    CHECK(read_probe(held_q) == 41, 505);
    unsigned char *r = wmem_alloc(packet, 64);
    CHECK(r && (uintptr_t)r == address, 506);
    r[0] = 59;
    CHECK(read_probe(held_q) == 41, 507);
    wm_leave_file_scope();
    wm_enter_file_scope();
    unsigned char *s = wmem_alloc(file, 64);
    CHECK(s && (uintptr_t)s == address_q, 508);
    s[0] = 61;
    unsigned char *t = wmem_alloc(file, 100);
    t[0] = 1;
    t = wmem_realloc(file, t, 300);
    CHECK(t[0] == 1, 509);
    t = wmem_realloc(file, t, 50);
    CHECK(t[0] == 1, 509);
    wmem_free(file, t);
    mark(test);
    break;
  }
  case 1:
    /* The reported shape: a packet-scope reset, then a read through a
     * pointer kept across it. Unprotected, the old byte is still there. */
    wm_packet_pool_reset(packet);
    mark(test);
    CHECK(read_probe(held) == 17, 510);
    break;
  case 2: {
    /* The same storage is reissued, and a stale interior write lands in the
     * new object. Unprotected, the new object is silently corrupted. */
    wm_packet_pool_reset(packet);
    unsigned char *r = wmem_alloc(packet, 64);
    CHECK(r && (uintptr_t)r == address, 511);
    r[16] = 7;
    mark(test);
    write_probe(interior);
    CHECK(r[16] == 93, 512);
    break;
  }
  case 3:
    /* The recycler allocator's reset retains and reinitializes its block.
     * Its free-list node is written into the freed chunk's data, so the
     * unprotected read returns allocator metadata, not the old byte. */
    wmem_free_all(file);
    mark(test);
    (void)read_probe(held_q);
    break;
  case 4:
    /* Documented limit: an individual recycler free returns the chunk to a
     * free list inside a live block and ends no epoch. Neither mode faults;
     * the unprotected read again sees the free-list node. */
    wmem_free(file, q);
    mark(test);
    (void)read_probe(held_q);
    break;
  case 5:
    /* Bounds: one byte past the request faults in both modes. */
    mark(test);
    (void)read_probe(held + 64);
    break;
  case 6: {
    /* The strict allocator gives every object its own region; the debug
     * arm under which upstream observed its reports. */
    wmem_allocator_t *strict = wmem_allocator_new(WMEM_ALLOCATOR_STRICT);
    unsigned char *s = wmem_alloc(strict, 64);
    CHECK(s, 515);
    s[0] = 73;
    unsigned char *volatile held_s = s;
    wmem_free(strict, s);
    mark(test);
    (void)read_probe(held_s);
    wmem_destroy_allocator(strict);
    break;
  }
  case 7:
    /* Two thousand dissection epochs on one retained block. */
    for (unsigned i = 0; i < 2000; ++i) {
      wm_packet_pool_reset(packet);
      unsigned char *r = wmem_alloc(packet, 64);
      CHECK(r && (uintptr_t)r == address, 516);
      r[0] = 59;
      held = r;
    }
    wm_packet_pool_reset(packet);
    mark(test);
    (void)read_probe(held);
    break;
  case 8:
    /* Destroying the pool releases its block to the system. */
    wmem_destroy_allocator(packet);
    packet = NULL;
    mark(test);
    (void)read_probe(held);
    break;
  case 9: {
    /* A jumbo object lives in its own block and is released by the reset. */
    size_t jumbo = 2u << 20;
    unsigned char *j = wmem_alloc(packet, jumbo);
    CHECK(j, 517);
    j[0] = 83;
    j[jumbo - 1] = 84;
    unsigned char *volatile held_j = j;
    wm_packet_pool_reset(packet);
    mark(test);
    (void)read_probe(held_j);
    break;
  }
  case 10:
    /* Leaving the file scope also collects, returning the unused block. */
    wm_leave_file_scope();
    mark(test);
    (void)read_probe(held_q);
    break;
  case 11: {
    /* A second block of the packet pool is returned, not retained. */
    size_t big = 1u << 20;
    unsigned char *b1 = wmem_alloc(packet, big);
    unsigned char *b2 = wmem_alloc(packet, big);
    CHECK(b1 && b2 && (uintptr_t)b2 - (uintptr_t)p > (2u << 20), 518);
    b2[0] = 91;
    unsigned char *volatile held_b2 = b2;
    wm_packet_pool_reset(packet);
    mark(test);
    (void)read_probe(held_b2);
    break;
  }
  case 12:
    /* A stale pointer handed back to the allocator is refused at the
     * allocator's own probe. Only the protected arm can run this: the
     * unprotected recycler would corrupt its free lists silently. */
    wmem_free_all(file);
    mark(test);
    if (mode == 1)
      wmem_free(file, held_q);
    break;
  }
  if (packet)
    wm_packet_pool_release(packet);
  if (wmem_in_scope(file))
    wm_leave_file_scope();
  wm_scopes_cleanup();
  out->completed = 1;
}
