/* What the lifetime controls share: the block allocator's own reference to
 * its block, the controls' labelled accesses, and the lines a run records.
 *
 * Not cases, and not the arm controls one directory up: these are the
 * parent, sibling and successor controls of the pre-registration
 * docs/plans/2026-10-11-nested-lifetime-controls.md. They show that patch
 * 0001 leaves PERMITTED accesses alone. Each runs on a fresh
 * wmem_allocator_new(WMEM_ALLOCATOR_BLOCK) and releases through wmem_free,
 * the release operation the patch revokes at (wmem_block_sublet_free).
 * block_fast is not used: its free is a no-op.
 *
 * WHICH REFERENCE IS "THE PARENT". The block allocator keeps its blocks on
 * allocator->block_list as wmem_block_hdr_t pointers: the capability the
 * system allocator returned for the 8 MiB block (wmem_block_new_block,
 * `block = wmem_alloc(NULL, WMEM_BLOCK_SIZE)`). Every consumer function of the
 * allocator works through it: the chunk headers and the free-list links are
 * reached by address arithmetic from the block, and patch 0001's
 * wmem_sublet_release hands upstream's free `block + offset`, i.e. this
 * capability, not the caller's. That is the reference control 00 uses first.
 * Patch 0001 keeps two more in the block header (wmem_sublet_block_t): `life`,
 * CDERIVEd from the block, and `generation`, CDERIVEd from life; every object
 * is a CDERIVE child of `generation`, and wmem_free is
 * CREVOKE(generation, object). So `generation` is the object's DIRECT parent
 * lifetime; control 00 uses it second, on the protected build only, since the
 * release build has no such field.
 *
 * The layouts below restate wmem_allocator_block.c at the 4.6.8 pin
 * (wmem_block_hdr_t, wmem_block_chunk_t, wmem_block_allocator_t) and patch
 * 0001's addition to the block header. The protected build is recognised by
 * the patched wmem_allocator.h, the only header in the build that includes
 * <capstone/capability.h>; each control also checks it at run time against
 * the bounds of the pointer wmem handed out, and refuses (exit 75) on a
 * mismatch, so a protected verdict cannot come from a release image.
 */
#ifndef WM_LIFETIME_CONTROLS_H
#define WM_LIFETIME_CONTROLS_H

#include "corpus.h"
#include "wmem_allocator.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef CAPSTONE_CAPABILITY_H
#define WM_CTL_PROTECTED 1 /* patch 0001's wmem_allocator.h */
#else
#define WM_CTL_PROTECTED 0
#endif

/* The size of A, B and C: one 16-byte-aligned chunk, larger than the
 * free-list links (two pointers) the allocator writes into a free chunk. */
#define WM_CTL_N 64UL

struct wm_ctl_block_hdr {
  struct wm_ctl_block_hdr *prev, *next;
#if WM_CTL_PROTECTED
  wmem_sublet_block_t sublet; /* patch 0001: life, generation */
#endif
};
struct wm_ctl_chunk {
  uint32_t prev;
  uint32_t last : 1;
  uint32_t used : 1;
  uint32_t jumbo : 1;
  uint32_t len : 29;
};
struct wm_ctl_block_allocator {
  struct wm_ctl_block_hdr *block_list;
  struct wm_ctl_chunk *master_head;
  struct wm_ctl_chunk *recycler_head;
};
#define WM_CTL_ALIGN (2 * sizeof(size_t))
#define WM_CTL_CHUNK_HEADER                                                    \
  ((sizeof(struct wm_ctl_chunk) + WM_CTL_ALIGN - 1) & ~(WM_CTL_ALIGN - 1))

#if defined(__riscv) || defined(__CAPSTONE__)
#define WM_CTL_CURSOR(p) ((unsigned long)__builtin_capstone_cap_get_cursor((void *)(p)))
#define WM_CTL_BASE(p) ((unsigned long)__builtin_capstone_cap_get_base((void *)(p)))
#define WM_CTL_END(p) ((unsigned long)__builtin_capstone_cap_get_end((void *)(p)))
#else /* a native build checks the sequence only; it has no bounds */
#define WM_CTL_CURSOR(p) ((unsigned long)(uintptr_t)(p))
#define WM_CTL_BASE(p) 0UL
#define WM_CTL_END(p) 0UL
#endif

/* The block allocator's own reference to the (only) block of `al`. */
static inline struct wm_ctl_block_hdr *wm_ctl_block(wmem_allocator_t *al) {
  struct wm_ctl_block_hdr *block =
      ((struct wm_ctl_block_allocator *)al->private_data)->block_list;
  CHECK(block && !block->next, 0xc7100001);
  return block;
}

static inline void wm_ctl_show(const char *name, const void *p) {
  printf("WM_CTL ref %s addr=%#lx base=%#lx end=%#lx\n", name, WM_CTL_CURSOR(p),
         WM_CTL_BASE(p), WM_CTL_END(p));
}

/* The build is the one it says it is: on the protected build wmem hands out a
 * pointer bounded to the request, on the release build the block's. */
static inline void wm_ctl_arm(const void *object, unsigned long n) {
  printf("WM_CTL arm=%s\n", WM_CTL_PROTECTED ? "protected" : "release");
#if defined(__riscv) || defined(__CAPSTONE__)
  int bounded = WM_CTL_BASE(object) == WM_CTL_CURSOR(object) &&
                WM_CTL_END(object) == WM_CTL_CURSOR(object) + n;
  if (bounded != WM_CTL_PROTECTED) {
    fprintf(stderr, "CONTROL-FAILED the image's wmem is not the %s build\n",
            WM_CTL_PROTECTED ? "protected" : "release");
    exit(75);
  }
#else
  (void)object, (void)n;
#endif
}

/* The controls' labelled accesses: one byte each, so a fault names the access
 * by its pc. wm_probe and wm_write_probe (shared/driver.c) are the corpus's
 * labels for a STALE access; these are the labels for a PERMITTED one, and
 * they carry the value. */
#if defined(__riscv) || defined(__CAPSTONE__)
__attribute__((noinline)) unsigned wm_ctl_load(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl wm_ctl_load_insn\nwm_ctl_load_insn:\nlbu %0, 0(%1)"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
}
__attribute__((noinline)) void wm_ctl_store(volatile unsigned char *p,
                                            unsigned value) {
  __asm__ volatile(".globl wm_ctl_store_insn\nwm_ctl_store_insn:\nsb %0, 0(%1)" ::"r"(
                       (unsigned long)value),
                   "r"(p)
                   : "memory");
}
#else
__attribute__((noinline)) unsigned wm_ctl_load(const volatile unsigned char *p) {
  return *p;
}
__attribute__((noinline)) void wm_ctl_store(volatile unsigned char *p,
                                            unsigned value) {
  *p = (unsigned char)value;
}
#endif

/* Each access prints what it is about to do BEFORE doing it, so a fault is
 * named by the last `access` line without a `done`. The site is the file and
 * line of the call, by base name (WM_CTL_AT), so no build path is printed. */
#define WM_CTL_AT(file) file, __LINE__
static void wm_ctl_read(const char *what, const char *via, const void *p,
                        unsigned long n, unsigned char *out, const char *file,
                        int line) {
  printf("WM_CTL access read %s via=%s addr=%#lx len=%lu probe=wm_ctl_load_insn "
         "site=%s:%d\n",
         what, via, WM_CTL_CURSOR(p), n, file, line);
  for (unsigned long i = 0; i < n; i++)
    out[i] = (unsigned char)wm_ctl_load((const volatile unsigned char *)p + i);
  printf("WM_CTL done read %s\n", what);
}

static void wm_ctl_write(const char *what, const char *via, void *p,
                         unsigned long n, unsigned (*pattern)(unsigned long),
                         const char *file, int line) {
  printf("WM_CTL access write %s via=%s addr=%#lx len=%lu probe=wm_ctl_store_insn "
         "site=%s:%d\n",
         what, via, WM_CTL_CURSOR(p), n, file, line);
  for (unsigned long i = 0; i < n; i++)
    wm_ctl_store((volatile unsigned char *)p + i, pattern(i));
  printf("WM_CTL done write %s\n", what);
}

/* Compare a range read back with the pattern it should hold; print, never
 * exit, so the host judges the line rather than an exit code. */
static int wm_ctl_expect(const char *what, const unsigned char *got,
                         unsigned long from, unsigned long n,
                         unsigned (*pattern)(unsigned long)) {
  unsigned long bad = 0, first = n;
  for (unsigned long i = from; i < n; i++)
    if (got[i] != (unsigned char)pattern(i)) {
      if (!bad++)
        first = i;
    }
  printf("WM_CTL check %s bytes=%lu..%lu mismatches=%lu%s\n", what, from, n - 1,
         bad, bad ? "" : " ok");
  if (bad)
    printf("WM_CTL check %s first_mismatch=%lu got=%#x want=%#x\n", what, first,
           got[first], (unsigned char)pattern(first));
  return bad == 0;
}

/* Patterns: A, B, C and the controls' overwrites are pairwise distinct at
 * every index below WM_CTL_N, so a range that reads back one of them says
 * whose bytes it holds. */
static unsigned wm_ctl_pattern_a(unsigned long i) { return 0xa0 ^ (i & 0x1f); }
static unsigned wm_ctl_pattern_b(unsigned long i) { return 0x40 + (i & 0x3f); }
static unsigned wm_ctl_pattern_c(unsigned long i) { return 0xc0 + (i & 0x1f); }
static unsigned wm_ctl_pattern_w(unsigned long i) { return 0x01 + (i & 0x1f); }
static unsigned wm_ctl_pattern_v(unsigned long i) { return 0x21 + (i & 0x1f); }

static void wm_ctl_fill(void *p, unsigned long n,
                        unsigned (*pattern)(unsigned long)) {
  for (unsigned long i = 0; i < n; i++)
    ((unsigned char *)p)[i] = (unsigned char)pattern(i);
}

/* LIFETIME CONTROL 3's sequence, shared by its two programs so they differ
 * only in what follows it. A is allocated from a fresh BLOCK allocator with a
 * guard after it (so the free cannot merge A into the master's tail), filled,
 * and released with wmem_free; then C is allocated at A's size until wmem
 * returns A's start address, at most WM_CTL_TRIES times, every miss kept live.
 * C is written and read back through C. Returns C; *stale is the old A. */
#define WM_CTL_TRIES 64
static unsigned char *wm_ctl_successor(unsigned char **stale) {
  wmem_allocator_t *al = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
  CHECK(al, 0xc7300001);
  unsigned char *a = wmem_alloc(al, WM_CTL_N);
  CHECK(a, 0xc7300002);
  wm_ctl_arm(a, WM_CTL_N);
  unsigned char *guard = wmem_alloc(al, WM_CTL_N);
  CHECK(guard, 0xc7300003);
  wm_ctl_fill(a, WM_CTL_N, wm_ctl_pattern_a);
  wm_ctl_show("A", a);
  wm_ctl_show("guard", guard);
  wm_ctl_show("block", wm_ctl_block(al));
  unsigned long a_at = WM_CTL_CURSOR(a);

  wmem_free(al, a); /* patch 0001: CREVOKE(generation, A), then upstream's free */
  printf("WM_CTL released A via wmem_free\n");

  unsigned char *c = NULL;
  int tries = 0;
  while (tries < WM_CTL_TRIES) {
    c = wmem_alloc(al, WM_CTL_N);
    CHECK(c, 0xc7300004);
    tries++;
    printf("WM_CTL alloc try=%d addr=%#lx\n", tries, WM_CTL_CURSOR(c));
    if (WM_CTL_CURSOR(c) == a_at)
      break;
  }
  int same = c && WM_CTL_CURSOR(c) == a_at;
  printf("WM_CTL successor A=%#lx C=%#lx tries=%d equal=%s\n", a_at,
         c ? WM_CTL_CURSOR(c) : 0UL, tries, same ? "yes" : "no");
  CHECK(same, 0xc7300005); /* without equality the control says nothing */
  wm_ctl_show("C", c);

  unsigned char got[WM_CTL_N];
  wm_ctl_write("C", "C", c, WM_CTL_N, wm_ctl_pattern_c, WM_CTL_AT("lifetime.h"));
  wm_ctl_read("C", "C", c, WM_CTL_N, got, WM_CTL_AT("lifetime.h"));
  wm_ctl_expect("C-written", got, 0, WM_CTL_N, wm_ctl_pattern_c);
  *stale = a;
  return c;
}

#endif
