/* Perl SV-head lifetime adapter: the platform-independent core.
 *
 * Patch 0001 (patches/5.36.3) makes Perl take every SV head from
 * perl_svh_new() and return it with perl_svh_del(), instead of carving heads
 * from 4080-byte arenas and chaining free ones through their own bytes. This
 * core keeps upstream's allocation policy exactly -- a LIFO free list, and a
 * new "arena" of per_page heads only when that list is empty -- so a spatial
 * build reissues the same slot identities in the same order as upstream Perl.
 * What differs is where the free-list link lives: in this sidecar, never in a
 * released head, so a protected backend may revoke or poison the head.
 *
 * A backend (capstone.c, cheribsd.c, native.c) defines SVH_BACKEND_SLOT (its
 * per-head fields) and SVH_PLATFORM, includes this file, and then defines the
 * svh_backend_* functions declared below. It calls svh_publish() when a
 * released head may be issued again: at once, or after its revocation sweep.
 *
 * Every successful issue advances the common reuse observer
 * (experiments/study/reuse-gap-observer.h) and is keyed by the head's
 * address, so the report is the same PERL_REUSE_GAP histogram on every
 * platform. The observer and this sidecar hold integers and current leases
 * only; a released head's pointer is never retained. */
#ifndef PERL_SV_HEADS_H
#define PERL_SV_HEADS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../../../experiments/study/reuse-gap-observer.h"

#ifndef SVH_PLATFORM
#error "a backend defines SVH_PLATFORM and SVH_BACKEND_SLOT before this file"
#endif
#ifndef SVH_HEAD_ALIGN
#define SVH_HEAD_ALIGN 16 /* a capability; revocation and poison work in these */
#endif

/* PENDING: released, not yet issuable (for a protected backend, until its
 * revocation sweep has completed). RETIRED: released, never issued again. */
enum { SVH_FREE = 1, SVH_LIVE, SVH_PENDING, SVH_RETIRED };

struct svh_slot {
  void *client; /* the current lease while LIVE, else NULL */
  SVH_BACKEND_SLOT
  unsigned char state;
};

#define SVH_REUSE_SLOTS (1u << 17)
static struct reuse_gap_slot svh_reuse_slots[SVH_REUSE_SLOTS];
static struct reuse_gap_observer svh_reuse;

static struct {
  int ready, quiet;
  unsigned mode;
  size_t bytes, per_page, max_pages, pages, peak_pages;
  uint64_t base;
  struct svh_slot **page;
  uint32_t *free_stack;
  size_t free_top, free_cap;
  uint64_t issues, releases, live, peak_live, retired;
  size_t sidecar_bytes;
} svh;

/* The backend's side. */
static void svh_backend_init(size_t bytes, uint64_t *base, size_t *capacity);
static int svh_backend_carve(size_t g, struct svh_slot *s);
static void *svh_backend_issue(size_t g, struct svh_slot *s);
static void svh_backend_release(size_t g, struct svh_slot *s);
static int svh_backend_current(struct svh_slot *s, const void *head);
static uint64_t svh_backend_address(const void *head);
static void svh_backend_teardown(void);
struct svh_line;
static void svh_backend_report(struct svh_line *line);

static void svh_report(void);

_Noreturn static void svh_fail(unsigned code, const char *why) {
  char line[160];
  int n = snprintf(line, sizeof line, "PERL-SVH-FAIL code=%u %s\n", code, why);
  if (n > 0 && (size_t)n < sizeof line && write(2, line, (size_t)n) < 0)
    n = 0; /* nothing else can report it */
  _exit((int)(code & 255) ? (int)(code & 255) : 1);
}

static struct svh_slot *svh_slot_of(size_t g) {
  return &svh.page[g / svh.per_page][g % svh.per_page];
}

static void svh_init(size_t bytes, size_t per_page) {
  size_t capacity = 0;
  if (!bytes || bytes % SVH_HEAD_ALIGN || !per_page || per_page > 4096)
    svh_fail(801, "head geometry");
  svh.bytes = bytes;
  svh.per_page = per_page;
  svh_backend_init(bytes, &svh.base, &capacity);
  svh.max_pages = capacity / per_page;
  if (!svh.max_pages)
    svh_fail(802, "region smaller than one page of heads");
  svh.page = calloc(svh.max_pages, sizeof *svh.page);
  if (!svh.page)
    svh_fail(803, "page table");
  svh.sidecar_bytes += svh.max_pages * sizeof *svh.page;
  reuse_gap_init(&svh_reuse, svh_reuse_slots, SVH_REUSE_SLOTS);
#ifdef SVH_REPORT_OPTIONAL
  /* Test backends only: upstream tests compare a child's stderr, sometimes
     of a child started with a cleared environment, so they report only when
     PERL_SVH_REPORT=1. Read now: assigning $0 may overwrite the environment
     strings before exit. The study backends always report. */
  const char *report = getenv("PERL_SVH_REPORT");
  svh.quiet = !(report && report[0] == '1' && !report[1]);
#endif
  if (atexit(svh_report))
    svh_fail(804, "atexit");
  svh.ready = 1;
}

/* Upstream plants a free head at the front of PL_sv_root; this stack is
 * that list. */
static void svh_push(size_t g, struct svh_slot *s) {
  if (svh.free_top == svh.free_cap) {
    size_t cap = svh.free_cap ? 2 * svh.free_cap : 1024;
    uint32_t *stack = realloc(svh.free_stack, cap * sizeof *stack);
    if (!stack)
      svh_fail(806, "free stack");
    svh.sidecar_bytes += (cap - svh.free_cap) * sizeof *stack;
    svh.free_stack = stack;
    svh.free_cap = cap;
  }
  s->state = SVH_FREE;
  svh.free_stack[svh.free_top++] = (uint32_t)g;
}

/* A released head becomes issuable. */
static void svh_publish(size_t g) {
  struct svh_slot *s = svh_slot_of(g);
  if (s->state != SVH_PENDING)
    svh_fail(805, "publish state");
  svh_push(g, s);
}

/* A new arena's heads are issued in ascending order, as sv_add_arena chains
 * them; so they are pushed in reverse. */
static int svh_add_page(void) {
  if (svh.pages == svh.max_pages)
    return 0;
  struct svh_slot *slots = calloc(svh.per_page, sizeof *slots);
  if (!slots)
    svh_fail(807, "page sidecar");
  svh.sidecar_bytes += svh.per_page * sizeof *slots;
  svh.page[svh.pages] = slots;
  size_t first = svh.pages * svh.per_page;
  for (size_t i = 0; i < svh.per_page; ++i)
    if (!svh_backend_carve(first + i, &slots[i]))
      svh_fail(808, "carve");
  ++svh.pages;
  if (svh.pages > svh.peak_pages)
    svh.peak_pages = svh.pages;
  for (size_t i = svh.per_page; i-- > 0;)
    svh_push(first + i, &slots[i]);
  return 1;
}

/* The slot a pointer names, from its address alone, or NULL when it is not
 * a head this adapter carved. Nothing is read through the pointer. */
static struct svh_slot *svh_lookup(const void *head, size_t *index) {
  if (!svh.ready)
    return NULL;
  uint64_t a = svh_backend_address(head);
  uint64_t span = (uint64_t)svh.pages * svh.per_page * svh.bytes;
  if (a < svh.base || a - svh.base >= span || (a - svh.base) % svh.bytes)
    return NULL;
  *index = (size_t)((a - svh.base) / svh.bytes);
  return svh_slot_of(*index);
}

void *perl_svh_new(size_t bytes, size_t per_page) {
  if (!svh.ready)
    svh_init(bytes, per_page);
  else if (bytes != svh.bytes || per_page != svh.per_page)
    svh_fail(809, "head geometry changed");
  if (!svh.free_top && !svh_add_page())
    return NULL; /* Perl croaks "Out of memory!" */
  size_t g = svh.free_stack[--svh.free_top];
  struct svh_slot *s = svh_slot_of(g);
  if (s->state != SVH_FREE)
    svh_fail(810, "issued a head that is not free");
  void *head = svh_backend_issue(g, s);
  s->client = head;
  s->state = SVH_LIVE;
  ++svh.issues;
  if (++svh.live > svh.peak_live)
    svh.peak_live = svh.live;
  reuse_gap_attempt(&svh_reuse);
  reuse_gap_issue(&svh_reuse, svh_backend_address(head), svh.bytes);
  return head;
}

void perl_svh_del(void *head, int reusable) {
  size_t g;
  struct svh_slot *s = svh_lookup(head, &g);
  if (!s || s->state != SVH_LIVE || !svh_backend_current(s, head))
    svh_fail(811, "release of a head that is not a current lease");
  reuse_gap_release(&svh_reuse, svh_backend_address(head));
  ++svh.releases;
  --svh.live;
  s->client = NULL;
  if (!reusable) {
    /* SVf_BREAK: upstream keeps it off the free list for good, because
       dangling pointers to it may still be read. Neither reissued nor revoked. */
    s->state = SVH_RETIRED;
    ++svh.retired;
    return;
  }
  s->state = SVH_PENDING;
  svh_backend_release(g, s);
}

int perl_svh_released(const void *head) {
  size_t g;
  struct svh_slot *s = svh_lookup(head, &g);
  if (!s)
    return 0; /* not an adapter head (an immortal, say): upstream's test applies */
  return s->state != SVH_LIVE || !svh_backend_current(s, head);
}

size_t perl_svh_extent(void) { return svh.ready ? svh.pages : 0; }

/* Upstream S_visit's order: newest arena first, heads ascending within one. */
void *perl_svh_next(size_t *cursor, size_t limit) {
  if (limit > svh.pages)
    svh_fail(812, "visit extent");
  size_t total = limit * svh.per_page;
  while (*cursor < total) {
    size_t k = (*cursor)++;
    struct svh_slot *s = &svh.page[limit - 1 - k / svh.per_page][k % svh.per_page];
    if (s->state == SVH_LIVE)
      return s->client;
  }
  return NULL;
}

/* One field per snprintf call. A Capstone compiler without the C-48 fix
 * (ISSUES.md) misplaces every variadic argument after the first one spilled
 * past the argument registers: a first report printed the issue count as
 * max_pages and garbage as error. No call here passes more than two values,
 * so the report is right with either compiler. */
struct svh_line {
  char text[1024];
  int n;
};

static void svh_put_text(struct svh_line *l, const char *text) {
  size_t room = sizeof l->text - (size_t)l->n;
  int k = snprintf(l->text + l->n, room, "%s", text);
  if (k < 0 || (size_t)k >= room)
    svh_fail(813, "report line");
  l->n += k;
}

static void svh_put(struct svh_line *l, const char *name, unsigned long long value) {
  size_t room = sizeof l->text - (size_t)l->n;
  int k = snprintf(l->text + l->n, room, " %s=%llu", name, value);
  if (k < 0 || (size_t)k >= room)
    svh_fail(813, "report line");
  l->n += k;
}

static void svh_put_line(struct svh_line *l) {
  svh_put_text(l, "\n");
  if (write(2, l->text, (size_t)l->n) != l->n)
    svh_fail(814, "report write");
  l->n = 0;
}

static void svh_report(void) {
  svh_backend_teardown();
  if (svh.quiet)
    return;
  struct svh_line l = {{0}, 0};
  svh_put_text(&l, "PERL_REUSE_GAP");
  svh_put(&l, "attempts", svh_reuse.attempts);
  svh_put(&l, "issues", svh_reuse.issues);
  svh_put(&l, "releases", svh_reuse.releases);
  svh_put(&l, "reuses", svh_reuse.reuses);
  svh_put(&l, "distinct", svh_reuse.distinct_starts);
  svh_put(&l, "capacity", SVH_REUSE_SLOTS);
  svh_put(&l, "error", svh_reuse.error);
  svh_put_text(&l, " bins=");
  for (unsigned i = 0; i < 32; ++i) {
    char value[32];
    int k = snprintf(value, sizeof value, "%s%llu", i ? "," : "",
                     (unsigned long long)svh_reuse.bins[i]);
    if (k < 0 || (size_t)k >= sizeof value)
      svh_fail(813, "report line");
    svh_put_text(&l, value);
  }
  svh_put_line(&l);
  svh_put_text(&l, "PERL_SV_HEADS platform=");
  svh_put_text(&l, SVH_PLATFORM);
  svh_put(&l, "mode", svh.mode);
  svh_put(&l, "head_bytes", svh.bytes);
  svh_put(&l, "per_page", svh.per_page);
  svh_put(&l, "pages", svh.pages);
  svh_put(&l, "peak_pages", svh.peak_pages);
  svh_put(&l, "max_pages", svh.max_pages);
  svh_put(&l, "issues", svh.issues);
  svh_put(&l, "releases", svh.releases);
  svh_put(&l, "live", svh.live);
  svh_put(&l, "peak_live", svh.peak_live);
  svh_put(&l, "retired", svh.retired);
  svh_put(&l, "sidecar_bytes", svh.sidecar_bytes);
  svh_put(&l, "observer_bytes", sizeof svh_reuse_slots);
  svh_backend_report(&l);
  svh_put_line(&l);
}

#endif
