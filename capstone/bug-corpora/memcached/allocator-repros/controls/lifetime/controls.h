/* The parent, sibling and successor controls of the 2026-10-11 pre-registration
 * (docs/plans/2026-10-11-nested-lifetime-controls.md), on memcached's slab allocator.
 *
 * Each control is one program, `ctl-NN VARIANT`, driven through slabs.c's own API: slabs_alloc,
 * slabs_free and, for the page, slabs_peek_page -- the call the page mover makes to get the page
 * reference slabs.c keeps in slab_list[0]. No ledger call, no hook, no ISA test.
 *
 * Every access the control makes goes through one of three noinline functions whose single data
 * access is the access being measured, and every call is announced on stdout (unbuffered) BEFORE
 * it runs and confirmed after it, with the call site's file:line, the probe, the reference used and
 * its address:
 *
 *     ACCESS-BEGIN name=<what> kind=read|write via=<reference> addr=0x... site=<file>:<line> probe=<fn>
 *     ACCESS-DONE  name=<what> value=0x..
 *
 * so a fault is tied to one access by the last ACCESS-BEGIN with no ACCESS-DONE, and by the pc the
 * launcher reports, resolved against the image's symbols. The stale access of control 93 is the
 * corpus's own labelled probe, read_probe (shared/corpus.h), as in the security table's cases.
 *
 * Under patch 0006 (MC_CAPSTONE_SUBLET) byte 0 of every chunk holds the generation it is a child of
 * (item.sublet_parent), live or free, and slabs.c reads it at every slabs_free and slabs_alloc. No
 * control writes the first sizeof(item) bytes of a chunk: the free-list link (next, prev), it_flags
 * and slabs_clsid live there too. Writes go to the item's data[] only. */
#ifndef MC_LIFETIME_CONTROLS_H
#define MC_LIFETIME_CONTROLS_H
#include "corpus.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* One slab class with few chunks per 1 MiB page, so a second page -- which slabs_peek_page requires
 * before it answers -- costs a handful of allocations: about 8 chunks of about 120 KiB. */
#define CTL_REQUEST (96u * 1024u)
#define CTL_MAX_PERSLAB 64u
#define CTL_DATA ((size_t)sizeof(item)) /* first byte of data[], past every slabs.c field */
#define CTL_SPAN 16u                    /* bytes of data[] each control reads or writes */

extern const int ctl_number;
void ctl_body(const char *variant);

/* The address, as slabs.c's patch 0006 computes it ((uintptr_t)p). Taken only from live references. */
#pragma clang diagnostic ignored "-Wvoid-pointer-to-int-cast"
static inline unsigned long ctl_addr(const volatile void *p) { return (unsigned long)(uintptr_t)p; }

/* The permitted accesses. Each function's one load or store is the access. */
static const int ctl_read_line = __LINE__ + 2;
__attribute__((noinline, used)) static unsigned char
ctl_read_u8(const volatile unsigned char *p) { return *p; }
static const int ctl_write_line = __LINE__ + 2;
__attribute__((noinline, used)) static void
ctl_write_u8(volatile unsigned char *p, unsigned char v) { *p = v; }
/* A capability-sized load, for the generation in a chunk's slot 0 (protected arm only). */
static const int ctl_load_ptr_line = __LINE__ + 2;
__attribute__((noinline, used)) static void *
ctl_load_ptr(void *volatile *p) { return *p; }

__attribute__((unused)) static unsigned char ctl_read_at(const char *name, const char *via, const volatile unsigned char *p,
                                 const char *file, int line) {
  printf("ACCESS-BEGIN name=%s kind=read via=%s addr=%#lx site=%s:%d probe=ctl_read_u8@controls.h:%d\n",
         name, via, ctl_addr(p), file, line, ctl_read_line);
  unsigned char v = ctl_read_u8(p);
  printf("ACCESS-DONE name=%s value=%#x\n", name, v);
  return v;
}
__attribute__((unused)) static void ctl_write_at(const char *name, const char *via, volatile unsigned char *p, unsigned char v,
                         const char *file, int line) {
  printf("ACCESS-BEGIN name=%s kind=write via=%s addr=%#lx site=%s:%d probe=ctl_write_u8@controls.h:%d value=%#x\n",
         name, via, ctl_addr(p), file, line, ctl_write_line, v);
  ctl_write_u8(p, v);
  printf("ACCESS-DONE name=%s\n", name);
}
#define CTL_READ(name, via, p) ctl_read_at(name, via, (const volatile unsigned char *)(p), __FILE__, __LINE__)
#define CTL_WRITE(name, via, p, v) ctl_write_at(name, via, (volatile unsigned char *)(p), (v), __FILE__, __LINE__)

/* Read CTL_SPAN bytes of data[] at p through one reference and require every byte to be `want`; write
 * CTL_SPAN bytes. Only the first and last bytes are announced as accesses (with the CALLER's
 * file:line); the bytes between go through the same probe. */
__attribute__((unused)) static void ctl_expect_span_at(const char *name, const char *via,
                                                       const volatile unsigned char *p, unsigned char want,
                                                       int code, const char *file, int line) {
  for (unsigned i = 0; i < CTL_SPAN; ++i) {
    unsigned char got = (i == 0 || i == CTL_SPAN - 1) ? ctl_read_at(name, via, p + i, file, line)
                                                      : ctl_read_u8(p + i);
    if (got != want) {
      printf("CHECK-FAILED name=%s byte=%u got=%#x want=%#x\n", name, i, got, want);
      mcp_fail(code);
    }
  }
  printf("CHECK name=%s bytes=%u value=%#x ok\n", name, CTL_SPAN, want);
}
__attribute__((unused)) static void ctl_fill_span_at(const char *name, const char *via, volatile unsigned char *p,
                                                     unsigned char v, const char *file, int line) {
  for (unsigned i = 0; i < CTL_SPAN; ++i) {
    if (i == 0 || i == CTL_SPAN - 1)
      ctl_write_at(name, via, p + i, v, file, line);
    else
      ctl_write_u8(p + i, v);
  }
}
#define ctl_expect_span(name, via, p, want, code)                                                  \
  ctl_expect_span_at(name, via, (const volatile unsigned char *)(p), want, code, __FILE__, __LINE__)
#define ctl_fill_span(name, via, p, v)                                                             \
  ctl_fill_span_at(name, via, (volatile unsigned char *)(p), v, __FILE__, __LINE__)

/* The class and its geometry, from slabs.c (no page needed: slabs_init fixed both). */
__attribute__((unused)) static unsigned ctl_class(unsigned *size, unsigned *perslab) {
  unsigned id = slabs_clsid(CTL_REQUEST);
  CHECK(id, 801);
  *size = slabs_size((int)id);
  CHECK(slabs_available_chunks(id, NULL, perslab) == 0, 802); /* the class has no page yet */
  CHECK(*perslab >= 3 && *perslab <= CTL_MAX_PERSLAB, 803);
  printf("CLASS id=%u chunk_size=%u perslab=%u sizeof_item=%u\n", id, *size, *perslab,
         (unsigned)sizeof(item));
  return id;
}

/* Fill the class's first page and open a second, so that slabs_peek_page returns slab_list[0]: the
 * reference slabs.c keeps for the page that holds the first chunk handed out. `held` are the chunks
 * already taken from page 0; the filler chunks stay live. Returns page 0 and checks that every held
 * chunk lies inside it. */
__attribute__((unused)) static unsigned char *ctl_page_zero(unsigned id, unsigned size, unsigned perslab, void **held,
                                    unsigned nheld) {
  static void *filler[CTL_MAX_PERSLAB + 1];
  unsigned n = 0;
  while (slabs_available_chunks(id, NULL, NULL) != 0) {
    CHECK(n < CTL_MAX_PERSLAB, 804);
    filler[n] = slabs_alloc(id, 0);
    CHECK(filler[n], 805);
    ++n;
  }
  CHECK(n + nheld == perslab, 806);
  filler[n] = slabs_alloc(id, 0); /* the class's second page */
  CHECK(filler[n], 807);
  uint32_t psize = 0, pper = 0;
  unsigned char *page = slabs_peek_page(id, &psize, &pper);
  CHECK(page, 808);
  CHECK(psize == size && pper == perslab, 809);
  unsigned long lo = ctl_addr(page), hi = lo + (unsigned long)size * perslab;
  printf("PAGE id=%u slab_list0=%#lx span=[%#lx,%#lx) second_page_chunk=%#lx\n", id, lo, lo, hi,
         ctl_addr(filler[n]));
#if defined(__CAPSTONE__)
  printf("PAGE-CAP base=%#llx end=%#llx cursor=%#llx\n",
         (unsigned long long)__builtin_capstone_cap_get_base(page),
         (unsigned long long)__builtin_capstone_cap_get_end(page),
         (unsigned long long)__builtin_capstone_cap_get_cursor(page));
#endif
  for (unsigned i = 0; i < nheld; ++i)
    CHECK(ctl_addr(held[i]) >= lo && ctl_addr(held[i]) < hi &&
              (ctl_addr(held[i]) - lo) % size == 0, 810);
  CHECK(ctl_addr(filler[n]) < lo || ctl_addr(filler[n]) >= hi, 811);
  return page;
}

__attribute__((unused)) static void ctl_print_chunk(const char *name, const void *p) {
  printf("CHUNK name=%s addr=%#lx\n", name, ctl_addr(p));
#if defined(__CAPSTONE__)
  printf("CHUNK-CAP name=%s base=%#llx end=%#llx\n", name,
         (unsigned long long)__builtin_capstone_cap_get_base((void *)p),
         (unsigned long long)__builtin_capstone_cap_get_end((void *)p));
#endif
}

#endif
