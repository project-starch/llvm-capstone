/* Parent, sibling and successor controls at mruby's GC object slots (4.0.0-rc2).
 *
 * Pre-registration: capstone/docs/plans/2026-10-11-nested-lifetime-controls.md. The nested
 * allocator is mruby's GC: mrb_obj_alloc hands out an object slot of a heap page and the sweep
 * releases it. Under patch 0008 (MRB_CAPSTONE_SUBLET) each slot goes out as a CDERIVE child of
 * the page's lifetime (page->life) and the sweep revokes it (CREVOKE) once obj_free has run.
 * This driver embeds mruby and drives that allocator through its own API: mrb_str_new allocates
 * the object (one slot: a 32-byte string is embedded in its RVALUE), mrb_full_gc runs the sweep.
 *
 *   controls parent     A freed by the sweep, then its slot read and written through the page
 *   controls sibling    A and B on one page, A freed, B read and written through B
 *   controls successor  A freed, allocate until the slot comes back as C, C read and written
 *   controls stale      as successor, then A's old pointer is read (the positive control)
 *
 * Unreachability. mruby's collector is precise: root_scan_phase (gc.c) marks the global
 * variables, the GC arena, the built-in classes, top_self, mrb->exc, the VM stacks of mrb->c and
 * mrb->root_c, and the task queues. It never scans the C stack, C registers or C globals. A is
 * allocated between mrb_gc_arena_save and mrb_gc_arena_restore, so its arena entry is dropped;
 * it is never stored in a Ruby variable, an object, a global or the VM stack. The only copies of
 * A's pointer are this driver's C variables, which no root scan reads. That A really was freed is
 * then checked, not assumed: its slot's type, read through the page, is MRB_TT_FREE, and (patch
 * 0008) the page's issued[] entry for it is NULL.
 *
 * The page reference. The GC reaches a slot as page->objects[i], from the page pointers it keeps
 * in mrb->gc.heaps (what the sweep, gc_page_of and free_heap use). This driver walks the same
 * list. The heap page and RVALUE layouts are private to gc.c, so they are mirrored below,
 * verbatim from the pin; the mirror is checked at run time against the object mrb_obj_alloc
 * returned (same address at page->objects[i], same type, and under 0008 issued[i] equal to it),
 * and the run stops with exit 75 if it does not match. Built with -DMRB_CAPSTONE_SUBLET exactly
 * when libmruby was (controls/build.sh).
 *
 * Probes. Every access the controls make is one line in a ctl_probe_* function (noinline), and
 * that line prints "PROBE <name> <file>:<line>" right before the access, so a run records the
 * source location of each access whether it completes or faults.
 *
 * Exit: 0 every check held; 1 a check on the data failed; 75 the setup could not be made
 * (precondition, not a reading). A capability fault ends the process; the launcher reports it.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mruby.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/istruct.h>
#include <mruby/hash.h>
#include <mruby/proc.h>
#include <mruby/range.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <mruby/gc.h>
#include <mruby/error.h>

/* ---- mirror of gc.c's private layout at 4.0.0-rc2 (gc.c:114-186 with patch 0008) ---------- */
typedef struct RVALUE RVALUE;
struct free_obj {
  MRB_OBJECT_HEADER;
  RVALUE *next;
};
struct RVALUE_initializer {
  MRB_OBJECT_HEADER;
#if defined(MRB_WORD_BOXING) && defined(MRB_32BIT) && defined(MRB_USE_FLOAT32) && !defined(MRB_WORDBOX_NO_INLINE_FLOAT)
  char padding[sizeof(void*) * 4];
#else
  char padding[sizeof(void*) * 3];
#endif
};
struct RVALUE {
  union {
    struct RVALUE_initializer init;
    struct free_obj free;
    struct RBasic basic;
    struct RObject object;
    struct RClass klass;
#if defined(MRB_WORD_BOXING) || (defined(MRB_NAN_BOXING) && defined(MRB_INT64))
    struct RInteger integer;
#endif
#if defined(MRB_WORD_BOXING) && !defined(MRB_NO_FLOAT) && defined(MRB_WORDBOX_NO_INLINE_FLOAT)
    struct RFloat flt;
#endif
    struct RString string;
    struct RArray array;
    struct RHash hash;
    struct RRange range;
    struct RData data;
    struct RIStruct istruct;
    struct RProc proc;
    struct REnv env;
    struct RFiber fiber;
    struct RException exc;
    struct RBreak brk;
  } as;
};
#ifndef MRB_HEAP_PAGE_SIZE
#define MRB_HEAP_PAGE_SIZE 1024
#endif
struct mrb_heap_page {
  RVALUE *freelist;
  struct mrb_heap_page *next;
  struct mrb_heap_page *free_next;
  mrb_bool old:1;
  mrb_bool region:1;
#ifdef MRB_CAPSTONE_SUBLET
  void *life;
  struct RBasic *issued[MRB_HEAP_PAGE_SIZE];
#endif
  RVALUE objects[MRB_HEAP_PAGE_SIZE];
};
typedef struct mrb_heap_page page_t;

/* ---- addresses, notes ----------------------------------------------------------------------- */
#define ADDR(p) ((unsigned long)__builtin_capstone_cap_get_cursor((void *)(p)))
#define BASE(p) ((unsigned long)__builtin_capstone_cap_get_base((void *)(p)))
#define SRC "capstone/bug-corpora/mruby/gc-slot-repros/controls/" __FILE_NAME__
#define NOTE(name) note(name, __LINE__)

/* Every output line goes through say(): formatted here and handed to write(2) whole, one call per
 * line (stdio's writev of buffer plus tail lost whole lines on this platform in the pilot runs).
 * say() counts the lines and write failures; the last line reports both, so the runner can tell a
 * complete transcript from one that lost lines on the way out. */
static int said, write_errors;
__attribute__((format(printf, 1, 2))) static void say(const char *fmt, ...) {
  char line[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(line, sizeof line, fmt, ap);
  va_end(ap);
  if (n < 0) { write_errors++; return; }
  if (n >= (int)sizeof line) n = sizeof line - 1;
  for (int off = 0; off < n;) {
    ssize_t w = write(1, line + off, n - off);
    if (w <= 0) { write_errors++; break; }
    off += w;
  }
  said++;
}
static void note(const char *name, int line) { say("PROBE %s %s:%d\n", name, SRC, line); }

static const char A_PAYLOAD[] = "A-SLOT-PAYLOAD-0123456789abcdef!";   /* 32 bytes, embedded */
static const char B_PAYLOAD[] = "B-SIBLING-PAYLOAD-0123456789abcd";
static const char B_WRITTEN[] = "B-WRITTEN-AFTER-A-WAS-SWEPT-0001";
static const char C_PAYLOAD[] = "C-SUCCESSOR-PAYLOAD-0123456789ab";
static const char C_WRITTEN[] = "C-WRITTEN-THROUGH-C-0123456789ab";
#define PAYLOAD_LEN 32
#define TAIL_OFF 64            /* bytes into an 80-byte RVALUE: past free_obj's next (32..48) */
#define TAIL_PATTERN 0x5a5a1234c0ffee11ull

/* Copies of the stale pointer live in C only (see the head comment). */
static struct RString *volatile stale_A;

/* ---- probes: every access of the controls, one line each ------------------------------------- */
__attribute__((noinline)) unsigned ctl_probe_page_read_tt(volatile RVALUE *slot) {
  NOTE("page-read-tt"); return slot->as.free.tt; }
__attribute__((noinline)) unsigned long ctl_probe_page_read_next(volatile RVALUE *slot) {
  NOTE("page-read-next"); return ADDR(slot->as.free.next); }
__attribute__((noinline)) void ctl_probe_page_write_tail(volatile uint64_t *w, uint64_t v) {
  NOTE("page-write-tail"); *w = v; }
__attribute__((noinline)) uint64_t ctl_probe_page_read_tail(volatile uint64_t *w) {
  NOTE("page-read-tail"); return *w; }
#ifdef MRB_CAPSTONE_SUBLET
__attribute__((noinline)) unsigned long ctl_probe_page_read_issued(page_t *page, size_t i) {
  NOTE("page-read-issued"); return page->issued[i] ? ADDR(page->issued[i]) : 0; }
__attribute__((noinline)) unsigned ctl_probe_life_read_tt(volatile RVALUE *slot) {
  NOTE("life-read-tt"); return slot->as.free.tt; }
__attribute__((noinline)) void ctl_probe_life_write_tail(volatile uint64_t *w, uint64_t v) {
  NOTE("life-write-tail"); *w = v; }
__attribute__((noinline)) uint64_t ctl_probe_life_read_tail(volatile uint64_t *w) {
  NOTE("life-read-tail"); return *w; }
#endif
__attribute__((noinline)) void ctl_probe_obj_read(struct RString *s, char *out, unsigned *tt) {
  NOTE("obj-read"); *tt = s->tt; for (int k = 0; k < PAYLOAD_LEN; k++) out[k] = ((volatile char *)RSTR_EMBED_PTR(s))[k]; }
__attribute__((noinline)) void ctl_probe_obj_write(struct RString *s, const char *in) {
  NOTE("obj-write"); for (int k = 0; k < PAYLOAD_LEN; k++) ((volatile char *)RSTR_EMBED_PTR(s))[k] = in[k]; }
__attribute__((noinline)) void ctl_probe_stale_read_A(struct RString *a, char *out) {
  NOTE("stale-read-A"); for (int k = 0; k < PAYLOAD_LEN; k++) out[k] = ((volatile char *)RSTR_EMBED_PTR(a))[k]; }

/* ---- the allocator, driven through its API ----------------------------------------------------- */
static int fail(const char *mode, int code, const char *why) {
  say("CTL %s %s %s\n", mode, code == 75 ? "PRECONDITION" : "CHECK-FAILED", why);
  return code;
}

/* A 32-byte string: one slot from mrb_obj_alloc, embedded. keep=0 drops its arena entry. */
static struct RString *new_string(mrb_state *mrb, const char *payload, int keep) {
  int ai = mrb_gc_arena_save(mrb);
  struct RString *s = mrb_str_ptr(mrb_str_new(mrb, payload, PAYLOAD_LEN));
  if (!keep) mrb_gc_arena_restore(mrb, ai);
  return s;
}

/* The page holding address a, from the GC's own page list, and the slot index; NULL if none. */
static page_t *page_of(mrb_state *mrb, unsigned long a, size_t *index) {
  for (page_t *page = mrb->gc.heaps; page; page = page->next) {
    unsigned long d = a - ADDR(page->objects);
    if (d < MRB_HEAP_PAGE_SIZE * sizeof(RVALUE)) {
      if (d % sizeof(RVALUE)) return NULL;
      *index = d / sizeof(RVALUE);
      return page;
    }
  }
  return NULL;
}

static int count_pages(mrb_state *mrb) {
  int n = 0;
  for (page_t *page = mrb->gc.heaps; page; page = page->next) n++;
  return n;
}

/* The mirror matches the live object: page->objects[i] is it, its type is a string, and under
 * patch 0008 the page issued exactly this pointer for that slot. */
static int layout_ok(page_t *page, size_t i, struct RString *s) {
  if (ADDR(&page->objects[i]) != ADDR(s)) return 0;
  if (page->objects[i].as.basic.tt != MRB_TT_STRING) return 0;
#ifdef MRB_CAPSTONE_SUBLET
  if (!page->issued[i] || ADDR(page->issued[i]) != ADDR(s)) return 0;
#endif
  return RSTR_EMBED_P(s) && RSTRING_LEN(mrb_obj_value(s)) == PAYLOAD_LEN;
}

static void describe(const char *name, struct RString *s, page_t *page, size_t i) {
  say("ADDR %s=0x%lx base=0x%lx page=0x%lx objects=0x%lx index=%zu slot=0x%lx rvalue=%zu\n",
         name, ADDR(s), BASE(s), ADDR(page), ADDR(page->objects), i, ADDR(&page->objects[i]),
         sizeof(RVALUE));
}

/* Sweep until the slot reads MRB_TT_FREE through the page (at most 3 full collections). */
static int sweep_until_free(mrb_state *mrb, page_t *page, size_t i, int *gcs) {
  for (*gcs = 1; *gcs <= 3; (*gcs)++) {
    mrb_full_gc(mrb);
    if (ctl_probe_page_read_tt(&page->objects[i]) == MRB_TT_FREE) {
#ifdef MRB_CAPSTONE_SUBLET
      if (ctl_probe_page_read_issued(page, i) != 0) return 0;
#endif
      return 1;
    }
  }
  return 0;
}

static mrb_state *setup(const char *mode) {
  mrb_state *mrb = mrb_open();
  if (!mrb) return NULL;
  mrb_full_gc(mrb);
  mrb_full_gc(mrb);
  say("CTL %s start layout=%s rvalue=%zu page_slots=%d pages=%d live=%zu generational=%d\n",
         mode,
#ifdef MRB_CAPSTONE_SUBLET
         "0008",
#else
         "release",
#endif
         sizeof(RVALUE), MRB_HEAP_PAGE_SIZE, count_pages(mrb), mrb->gc.live, (int)mrb->gc.generational);
  return mrb;
}

/* A allocated and dropped; its page must also hold Object (a root), so the sweep cannot free
 * the page itself and the page reference stays the allocator's. */
static struct RString *alloc_A(mrb_state *mrb, const char *mode, page_t **page, size_t *i) {
  struct RString *a = new_string(mrb, A_PAYLOAD, 0);
  size_t ci;
  *page = page_of(mrb, ADDR(a), i);
  if (!*page || !layout_ok(*page, *i, a)) { fail(mode, 75, "layout-mirror-mismatch"); return NULL; }
  if (page_of(mrb, ADDR(mrb->object_class), &ci) != *page) { fail(mode, 75, "A-not-on-the-root-page"); return NULL; }
  describe("A", a, *page, *i);
  return a;
}

static int control_parent(mrb_state *mrb) {
  page_t *page; size_t i; int gcs;
  struct RString *a = alloc_A(mrb, "parent", &page, &i);
  if (!a) return 75;
  a = NULL;
  if (!sweep_until_free(mrb, page, i, &gcs)) return fail("parent", 75, "A-not-freed");
  say("FREED A after %d full gc: tt=MRB_TT_FREE read through the page\n", gcs);
  RVALUE *slot = &page->objects[i];
  unsigned long next = ctl_probe_page_read_next(slot);
  int at_head = ADDR(page->freelist) == ADDR(slot);
  say("READ page: tt=MRB_TT_FREE next=0x%lx; A's slot is the page's freelist head=%s\n", next, at_head ? "yes" : "no");
  volatile uint64_t *tail = (volatile uint64_t *)((char *)slot + TAIL_OFF);
  uint64_t old = ctl_probe_page_read_tail(tail);
  ctl_probe_page_write_tail(tail, TAIL_PATTERN);
  uint64_t back = ctl_probe_page_read_tail(tail);
  ctl_probe_page_write_tail(tail, old);
  say("WRITE page: wrote 0x%llx at slot+%d, read back 0x%llx\n", (unsigned long long)TAIL_PATTERN,
         TAIL_OFF, (unsigned long long)back);
  if (back != TAIL_PATTERN) return fail("parent", 1, "page-readback");
#ifdef MRB_CAPSTONE_SUBLET
  /* The same slot through page->life, the CDERIVE parent the slot's child was derived from. */
  RVALUE *lslot = (RVALUE *)((char *)page->life + i * sizeof(RVALUE));
  say("ADDR life=0x%lx base=0x%lx slot-through-life=0x%lx\n", ADDR(page->life), BASE(page->life), ADDR(lslot));
  unsigned ltt = ctl_probe_life_read_tt(lslot);
  volatile uint64_t *ltail = (volatile uint64_t *)((char *)lslot + TAIL_OFF);
  uint64_t lold = ctl_probe_life_read_tail(ltail);
  ctl_probe_life_write_tail(ltail, TAIL_PATTERN);
  uint64_t lback = ctl_probe_life_read_tail(ltail);
  ctl_probe_life_write_tail(ltail, lold);
  say("READ life: tt=%u (MRB_TT_FREE=%d); WRITE life: read back 0x%llx\n", ltt, MRB_TT_FREE,
         (unsigned long long)lback);
  if (ltt != MRB_TT_FREE || lback != TAIL_PATTERN) return fail("parent", 1, "life-readback");
#endif
  say("CTL parent completed\n");
  return 0;
}

static int control_sibling(mrb_state *mrb) {
  page_t *page, *bpage; size_t i, bi; int gcs;
  struct RString *b = new_string(mrb, B_PAYLOAD, 1);             /* stays in the arena: reachable */
  struct RString *a = alloc_A(mrb, "sibling", &page, &i);
  if (!a) return 75;
  bpage = page_of(mrb, ADDR(b), &bi);
  if (bpage != page || !layout_ok(bpage, bi, b)) return fail("sibling", 75, "B-not-on-A's-page");
  describe("B", b, bpage, bi);
  a = NULL;
  if (!sweep_until_free(mrb, page, i, &gcs)) return fail("sibling", 75, "A-not-freed");
  say("FREED A after %d full gc: tt=MRB_TT_FREE read through the page\n", gcs);
  char got[PAYLOAD_LEN + 1] = {0}; unsigned tt;
  ctl_probe_obj_read(b, got, &tt);
  int intact = tt == MRB_TT_STRING && b->c == mrb->string_class && !memcmp(got, B_PAYLOAD, PAYLOAD_LEN);
  say("READ B: tt=%u payload=\"%s\" intact=%s\n", tt, got, intact ? "yes" : "no");
  ctl_probe_obj_write(b, B_WRITTEN);
  ctl_probe_obj_read(b, got, &tt);
  int written = !memcmp(got, B_WRITTEN, PAYLOAD_LEN);
  say("WRITE B: read back \"%s\" match=%s\n", got, written ? "yes" : "no");
  if (!intact || !written) return fail("sibling", 1, "B-contents");
  say("CTL sibling completed\n");
  return 0;
}

/* successor and stale: A freed, then strings allocated until mrb_obj_alloc returns A's slot. */
static int control_successor(mrb_state *mrb, int stale) {
  const char *mode = stale ? "stale" : "successor";
  page_t *page; size_t i; int gcs;
  struct RString *a = alloc_A(mrb, mode, &page, &i);
  if (!a) return 75;
  unsigned long a_addr = ADDR(a);
  stale_A = a;
  a = NULL;
  if (!sweep_until_free(mrb, page, i, &gcs)) return fail(mode, 75, "A-not-freed");
  say("FREED A after %d full gc: tt=MRB_TT_FREE read through the page\n", gcs);
  mrb->gc.disabled = TRUE;               /* no collection while searching: every candidate stays */
  struct RString *c = NULL; int n;
  for (n = 1; n <= 4 * MRB_HEAP_PAGE_SIZE; n++) {
    int ai = mrb_gc_arena_save(mrb);
    struct RString *s = mrb_str_ptr(mrb_str_new(mrb, C_PAYLOAD, PAYLOAD_LEN));
    if (ADDR(s) == a_addr) { c = s; break; }                      /* C keeps its arena entry */
    mrb_gc_arena_restore(mrb, ai);
  }
  if (!c) { mrb->gc.disabled = FALSE; return fail(mode, 75, "A's-slot-not-reissued"); }
  size_t ci; page_t *cpage = page_of(mrb, ADDR(c), &ci);
  if (cpage != page || ci != i || !layout_ok(cpage, ci, c)) return fail(mode, 75, "layout-mirror-mismatch-C");
  describe("C", c, cpage, ci);
  say("REUSE C after %d allocations: C=0x%lx A=0x%lx equal=%s\n", n, ADDR(c), a_addr,
         ADDR(c) == a_addr ? "yes" : "no");
  char got[PAYLOAD_LEN + 1] = {0}; unsigned tt;
  if (!stale) {
    ctl_probe_obj_read(c, got, &tt);
    int ok = tt == MRB_TT_STRING && !memcmp(got, C_PAYLOAD, PAYLOAD_LEN);
    say("READ C: tt=%u payload=\"%s\" match=%s\n", tt, got, ok ? "yes" : "no");
    ctl_probe_obj_write(c, C_WRITTEN);
    ctl_probe_obj_read(c, got, &tt);
    int written = !memcmp(got, C_WRITTEN, PAYLOAD_LEN);
    say("WRITE C: read back \"%s\" match=%s\n", got, written ? "yes" : "no");
    mrb->gc.disabled = FALSE;
    if (!ok || !written) return fail(mode, 1, "C-contents");
    say("CTL successor completed\n");
    return 0;
  }
  say("STALE reading through A's old pointer 0x%lx\n", a_addr);
  ctl_probe_stale_read_A(stale_A, got);
  int is_c = !memcmp(got, C_PAYLOAD, PAYLOAD_LEN);
  say("STALE read through A returned \"%s\" equals-C's-data=%s\n", got, is_c ? "yes" : "no");
  mrb->gc.disabled = FALSE;
  say("CTL stale completed\n");
  return 0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "";
  int which = !strcmp(mode, "parent") ? 1 : !strcmp(mode, "sibling") ? 2 :
              !strcmp(mode, "successor") ? 3 : !strcmp(mode, "stale") ? 4 : 0;
  if (!which) { fprintf(stderr, "usage: controls parent|sibling|successor|stale\n"); return 2; }
  mrb_state *mrb = setup(mode);
  if (!mrb) return 75;
  int rc = which == 1 ? control_parent(mrb) : which == 2 ? control_sibling(mrb)
         : control_successor(mrb, which == 4);
  mrb_close(mrb);
  int ends = said;
  say("CTL %s exit %d lines=%d write_errors=%d\n", mode, rc, ends, write_errors);
  return rc;
}
