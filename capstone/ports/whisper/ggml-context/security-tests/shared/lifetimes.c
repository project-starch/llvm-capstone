#include "ggml.h"
#include "port.h"
#include <string.h>
#define CHECK(x, n)                                                            \
  do {                                                                         \
    if (!(x))                                                                  \
      wg_fail(n);                                                              \
  } while (0)
__attribute__((noinline)) static unsigned
read_probe(const volatile unsigned char *p) {
  unsigned long x;
  __asm__ volatile(".globl wg_probe_read\nwg_probe_read:\nlbu %0, 0(%1)\n"
                   : "=r"(x)
                   : "r"(p)
                   : "memory");
  return x;
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long x = 93;
  __asm__ volatile(
      ".globl wg_probe_write\nwg_probe_write:\nsb %0, 0(%1)\n" ::"r"(x), "r"(p)
      : "memory");
}
static void mark(unsigned id) {
  extern void wg_probe_read(void), wg_probe_write(void);
  unsigned long x = 0xcf14000000000000UL | id;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n" ::"r"(x),
                   "r"(wg_probe_read), "r"(wg_probe_write)
                   : "memory");
}
void wg_replay(const struct wg_header *in, struct wg_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->magic = WG_MAGIC;
  out->mode = mode;
  out->count = 1;
  const struct wg_event *e = (const void *)(in + 1);
  unsigned test = e->arg;
  CHECK(in->magic == WG_MAGIC && in->count == 1 && test <= 9, 501);
  wg_select_buffer(0);
  struct ggml_init_params params = {
      4096, test == 3 ? NULL : wg_borrow_buffer(0, 4096), true};
  struct ggml_context *ctx = ggml_init(params);
  unsigned char *p = wg_object_alloc(ctx, 0, 64);
  CHECK(p, 502);
  p[0] = 17;
  p[16] = 29;
  /* Materialize aliases while live. Volatile pointer slots prevent the
   * optimizer from sinking/hoisting pointer arithmetic across revocation;
   * each stale test must reach its labeled memory access, not fault earlier
   * while deriving an interior pointer from already revoked authority. */
  unsigned char *volatile held = p;
  unsigned char *volatile interior = p + 16;
  uintptr_t address = (uintptr_t)p;
  struct ggml_init_params other = {4096, wg_borrow_buffer(1, 4096), true};
  struct ggml_context *sibling = ggml_init(other);
  unsigned char *q = wg_object_alloc(sibling, 0, 64);
  q[0] = 41;
  CHECK(read_probe(held) == 17, 503);
  if (test == 0) {
    void **links = wg_object_alloc(ctx, 2, sizeof(void *));
    links[0] = q;
    ggml_free(ctx);
    ctx = NULL;
    CHECK(read_probe(held) == 17 && ((unsigned char *)links[0])[0] == 41, 504);
  } else if (test == 9) {
    void *zero = wg_object_alloc(ctx, 2, 0);
    CHECK(zero != NULL, 510);
    mark(test);
    (void)read_probe(zero);
  } else if (test == 5) {
    mark(test);
    (void)read_probe(held + 64);
  } else if (test == 3) {
    ggml_free(ctx);
    ctx = NULL;
    CHECK(q[0] == 41, 505);
    mark(test);
    (void)read_probe(held);
  } else if (test == 6) {
    unsigned char *descriptor = (void *)ctx;
    ggml_free(ctx);
    ctx = NULL;
    CHECK(read_probe(held) == 17 && q[0] == 41, 506);
    mark(test);
    (void)read_probe(descriptor);
  } else if (test == 8) {
    /* Failing allocation must leave all earlier objects live. */
    CHECK(wg_object_alloc(ctx, 2, 4096) == NULL && read_probe(held) == 17, 507);
  } else {
    if (test == 2 || test == 7) {
      ggml_free(ctx);
      CHECK(read_probe(held) == 17, 508);
      params.mem_buffer = wg_borrow_buffer(0, 4096);
      ctx = ggml_init(params);
    } else
      ggml_reset(ctx);
    unsigned rounds = test == 7 ? 2000 : 1;
    for (unsigned i = 0; i < rounds; ++i) {
      p = wg_object_alloc(ctx, 0, 64);
      CHECK((uintptr_t)p == address && q[0] == 41, 509);
      p[0] = 59;
      p[16] = 61;
      if (i + 1 < rounds)
        ggml_reset(ctx);
    }
    mark(test);
    if (test == 4)
      write_probe(interior);
    else
      (void)read_probe(held);
  }
  ggml_free(ctx);
  ggml_free(sibling);
  if (test == 0 || test == 8)
    mark(test);
  out->completed = 1;
}
