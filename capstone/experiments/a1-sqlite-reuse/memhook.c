/* A1: the measurement, recorded from inside the process, on any platform.
 *
 * One instrument for the three passes of the nested-allocators paper: x86 and
 * CheriBSD, where it is linked into a hosted speedtest1, and a Capstone domain,
 * where it is linked in beside speedtest1_domain.c (-DMEMHOOK_FREESTANDING).
 * The counting, the names and the output are the same on all three; what a
 * platform lacks is replaced under the one macro, and nowhere else.
 *
 * Levels, one instrument, no tracer:
 *
 *   level 0, "libc"      hosted only. sqlite3_config(SQLITE_CONFIG_MALLOC): a
 *                        constructor installs a sqlite3_mem_methods table that
 *                        records every call and forwards it to the methods that
 *                        were there (memsys1 over the system malloc).  With
 *                        --heap speedtest1 replaces that table by memsys5 and
 *                        this level sees nothing, which is the truth: nothing
 *                        below lookaside reaches the system allocator then.  A
 *                        domain has no level below memsys5, so no libc line.
 *   "lookaside"          a1_lookaside_alloc / a1_lookaside_free, called from
 *                        the eight slot paths of sqlite3DbMallocRawNN,
 *                        sqlite3DbFreeNN and sqlite3DbNNFreeNN, and
 *                        a1_lookaside_escape from the two paths that hand a
 *                        request down; hook-<version>.patch puts them there.
 *   "memsys5"            a1_memsys5_alloc / a1_memsys5_free / a1_memsys5_realloc
 *                        at memsys5MallocUnsafe's return, memsys5FreeUnsafe's
 *                        entry and memsys5Realloc's entry, the pair every
 *                        memsys5 method funnels through; the same patch.
 *
 * Tests: a1_begin_test / a1_end_test from speedtest1_begin_test and
 * speedtest1_end_test, the same patch, so every counter is keyed by test as
 * the probes keyed theirs.
 *
 * The patch is inert: a patched binary was held against bpftrace probes on
 * the same execution and against probes on an unpatched build, and all three
 * agreed to the event and to the bucket (experiments/a1/README, "Validated,
 * once").  What the patch cannot change is what SQLite allocates, and every
 * run checks that against the unhooked build's own counters.
 *
 * Counting: a realloc is a release of the old block and an allocation of the
 * new one, the ones that come back at the same address counted separately.
 * The free-to-reuse gap is measured on the level's own clocks: its running
 * allocation count (the reusing allocation included) and its running count
 * of released bytes (the object's own included, stamped before they are
 * added).  Sizes: at level 0 what memsys1 recorded for the block (xSize, the
 * request rounded to 8), at lookaside and memsys5 the request.  Beyond the
 * gap, the inputs of whatif.py: @prior_<level> and @beyond254 (objects the
 * address carried before; past 254, an 8-bit generation is spent), @poison16
 * (16-byte words over all objects), @addresses_<level> (objects per address
 * over the run), @size_<level>.
 *
 * Storage.  The hook must not allocate.  Hosted, its tables are mmaps taken
 * before SQLite initialises, one hashed table per level, and never grow.
 * Freestanding there is no mmap: the lookaside's addresses are the slots of one
 * pool, a small hashed table in the image holds them; memsys5's addresses are
 * its arena at 64-byte granularity, a direct-mapped table that the domain
 * carves next to the arena and hands over (speedtest1_hook_install).  An entry
 * is sixteen bytes there, because that table is one entry per atom and shares a
 * region with the arena; the running counts fit 32 bits at the sizes a domain
 * runs.  Hosted the entry keeps 64-bit words.
 *
 * Output is bpftrace's text format, in the names analyze.py, plot.py and
 * whatif.py read: per (test, level) @alloc, @free, @bytes, @realloc, @peak;
 * @kind per (test, call) at level 0; @escaped and @tnum, @tname per test; per
 * level the totals as @hook_*, @peakrun, @maxreuse, @distinct, @beyond254,
 * @poison16 and the histograms.  Hosted, a destructor writes it to A1_OUT,
 * default stderr; freestanding, the domain calls speedtest1_hook_report and
 * routes fprintf to the hostcall payload.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sqlite3.h"

#ifdef MEMHOOK_FREESTANDING
/* the domain's stdio and exit, defined in speedtest1_domain.c; the stub headers are empty */
extern FILE *stderr;
int fprintf(FILE *stream, const char *format, ...);
int snprintf(char *buffer, size_t size, const char *format, ...);
int sprintf(char *buffer, const char *format, ...);
__attribute__((noreturn)) void exit(int code);
#  define DIE() exit(3)
typedef uint32_t mh_word;        /* an entry is sixteen bytes, see Storage above */
#else
#  include <sys/mman.h>
#  define DIE() abort()
typedef uint64_t mh_word;
#endif

#ifdef __CHERI_PURE_CAPABILITY__
#  define ADDR(p) ((uint64_t)__builtin_cheri_address_get((void *)(p)))
#else
#  define ADDR(p) ((uint64_t)(uintptr_t)(p))
#endif

/* per address: when it was last released, on both clocks, how often it was
   handed out, and the size the level will count as released.  lastfree 0:
   not released since it was handed out, which cannot be a real value because a
   free comes after at least one allocation; reuses 0: never handed out. */
struct ent { mh_word lastfree, lastfree_b, reuses, size; };

#define MAXT 256                 /* tests per run; speedtest1's largest set has 85 */
struct pertest { uint64_t alloc, free, bytes, realloc, peak; };

/* A level's table is either hashed -- open addressing over `keys`, `bits`
   wide, entries never removed -- or direct-mapped: address -> (addr - base) /
   gran, when gran is set. */
struct level {
  const char *name;
  struct ent *tab;
  uint64_t *keys;                /* the hashed table's keys; 0 is empty */
  uint64_t nslots;               /* entries, a power of two when hashed */
  int bits;                      /* log2 nslots, for the hash */
  uint64_t nkeys;                /* keys in use, against the half-full limit */
  uint64_t base, gran;           /* direct-mapped when gran != 0 */
  uint64_t nent;                 /* @distinct: addresses handed out */
  uint64_t alloc, free, bytes, realloc, inplace, live, peak, maxreuse;
  uint64_t nalloc;               /* the clock */
  uint64_t fbytes;               /* the byte clock: bytes released so far */
  uint64_t beyond254, poison16;
  uint64_t hist[64];             /* hist[k] is [2^k, 2^(k+1)), hist[0] is [1] */
  uint64_t histb[64];            /* the same, in bytes released in between */
  uint64_t prior[64], prior0;    /* allocations by objects the address carried before; [0] apart */
  uint64_t sizeh[64];            /* object sizes */
  struct pertest t[MAXT];        /* per test; t[0] is outside any test */
};

#ifndef MEMHOOK_FREESTANDING
#define TBITS 23
#define TSIZE ((uint64_t)1 << TBITS)
static struct level L0 = { .name = "libc" };
static struct level L1 = { .name = "lookaside" };
static struct level L2 = { .name = "memsys5" };
#else
#define L1BITS 12
static struct ent l1tab[1 << L1BITS];
static uint64_t l1key[1 << L1BITS];
static struct level L1 = { .name = "lookaside", .tab = l1tab, .keys = l1key,
                           .nslots = 1 << L1BITS, .bits = L1BITS };
static struct level L2 = { .name = "memsys5" };
#endif

/* tests, in the order speedtest1 runs them */
static int cur, nseq;
static int tnum[MAXT];
static const char *tname[MAXT];
static uint64_t escaped[MAXT];
static uint64_t kind[MAXT][3];   /* level 0 calls per test: malloc, realloc, free */
static const char *KIND[3] = { "malloc", "realloc", "free" };

static struct ent *ent(struct level *L, uint64_t a) {
  if (L->gran) {
    if (a < L->base || (a - L->base) / L->gran >= L->nslots) {
      fprintf(stderr, "memhook: %s address outside the arena\n", L->name);
      DIE();
    }
    return &L->tab[(a - L->base) / L->gran];
  }
  uint64_t h = (a * 0x9E3779B97F4A7C15ull) >> (64 - L->bits);
  for (;;) {
    if (L->keys[h] == a) return &L->tab[h];
    if (L->keys[h] == 0) {
      if (++L->nkeys > L->nslots / 2) {
        fprintf(stderr, "memhook: %s address table full\n", L->name);
        DIE();
      }
      L->keys[h] = a;
      return &L->tab[h];
    }
    h = (h + 1) & (L->nslots - 1);
  }
}

static int lg(uint64_t x) { return 63 - __builtin_clzll(x); }

/* the allocation side, counted on the call, like the uprobe */
static void allocated(struct level *L, size_t n) {
  L->nalloc++; L->alloc++; L->bytes += n;
  if (n) L->sizeh[lg(n)]++;
  if (++L->live > L->peak) L->peak = L->live;
  struct pertest *t = &L->t[cur];
  t->alloc++; t->bytes += n;
  if (L->live > t->peak) t->peak = L->live;
}
/* the address side, on the result, like the uretprobe; size is what the
   level will count as released when this object goes */
static void returned(struct level *L, void *p, size_t size) {
  if (!p) return;
  struct ent *e = ent(L, ADDR(p));
  if (e->lastfree) {
    L->hist[lg(L->nalloc - e->lastfree)]++;
    L->histb[lg(L->fbytes - e->lastfree_b)]++;
    e->lastfree = 0;
  }
  if (e->reuses) L->prior[lg(e->reuses)]++; else { L->prior0++; L->nent++; }
  if (e->reuses >= 254) L->beyond254++;
  L->poison16 += (size + 15) / 16;
  e->size = (mh_word)size;
  if (++e->reuses > L->maxreuse) L->maxreuse = e->reuses;
}
static void released(struct level *L, void *p) {
  L->free++; L->live--;
  L->t[cur].free++;
  struct ent *e = ent(L, ADDR(p));
  /* stamped before the object's own bytes are added, so the gap in bytes
     includes them -- as the gap in allocations includes the reusing
     allocation -- and is never zero */
  e->lastfree = (mh_word)L->nalloc;
  e->lastfree_b = (mh_word)L->fbytes;
  L->fbytes += e->size;
}

/* ---- lookaside and memsys5 through the patch ---------------------------- */

/* -DMEMHOOK_NO_LOOKASIDE / -DMEMHOOK_NO_MEMSYS5: one level's entry points become
   no-ops, to bisect a failure of the instrumented program to one level's recording. */
#ifdef MEMHOOK_NO_LOOKASIDE
void *a1_lookaside_alloc(void *p, size_t n) { (void)n; return p; }
void a1_lookaside_free(void *p) { (void)p; }
void a1_lookaside_escape(void) {}
#else
void *a1_lookaside_alloc(void *p, size_t n) { allocated(&L1, n); returned(&L1, p, n); return p; }
void a1_lookaside_free(void *p) { released(&L1, p); }
void a1_lookaside_escape(void) { escaped[cur]++; }
#endif

#ifdef MEMHOOK_NO_MEMSYS5
void *a1_memsys5_alloc(void *p, size_t n) { (void)n; return p; }
void a1_memsys5_free(void *p) { (void)p; }
void a1_memsys5_realloc(void) {}
#else
void *a1_memsys5_alloc(void *p, size_t n) { allocated(&L2, n); returned(&L2, p, n); return p; }
void a1_memsys5_free(void *p) { released(&L2, p); }
void a1_memsys5_realloc(void) { L2.realloc++; L2.t[cur].realloc++; }
#endif

/* ---- tests -------------------------------------------------------------- */

void a1_begin_test(int num, const char *name) {
  if (nseq + 1 >= MAXT) { fprintf(stderr, "memhook: too many tests\n"); DIE(); }
  cur = ++nseq;
  tnum[cur] = num;
  tname[cur] = name;
}
void a1_end_test(void) { cur = 0; }

/* ---- report ------------------------------------------------------------- */

static const char *edge(uint64_t x, char *buf) {
  unsigned long long v = x;
  if (x >= (1u << 30) && x % (1u << 30) == 0) sprintf(buf, "%lluG", v >> 30);
  else if (x >= (1u << 20) && x % (1u << 20) == 0) sprintf(buf, "%lluM", v >> 20);
  else if (x >= 1024 && x % 1024 == 0) sprintf(buf, "%lluK", v >> 10);
  else sprintf(buf, "%llu", v);
  return buf;
}

static void hist_out(FILE *f, const char *name, const uint64_t *h, uint64_t zero, int with_zero) {
  fprintf(f, "@%s: \n", name);
  int top = 63;
  while (top > 0 && h[top] == 0) top--;
  if (with_zero) fprintf(f, "%-22s %10llu |\n", "[0]", (unsigned long long)zero);
  for (int k = 0; k <= top; k++) {
    char lo[24], hi[24], label[64];
    if (k == 0) snprintf(label, sizeof label, "[1]");
    else snprintf(label, sizeof label, "[%s, %s)",
                  edge((uint64_t)1 << k, lo), edge((uint64_t)1 << (k + 1), hi));
    fprintf(f, "%-22s %10llu |\n", label, (unsigned long long)h[k]);
  }
  fprintf(f, "\n");
}

#define U(x) ((unsigned long long)(x))

static void report_tests(FILE *f, struct level **levels, int n) {
  /* per (test, level), only what was touched, as bpftrace prints its maps */
  for (int s = 0; s <= nseq; s++) {
    for (int l = 0; l < n; l++) {
      struct level *L = levels[l];
      struct pertest *t = &L->t[s];
      if (t->alloc) fprintf(f, "@alloc[%d, %s]: %llu\n", s, L->name, U(t->alloc));
      if (t->free) fprintf(f, "@free[%d, %s]: %llu\n", s, L->name, U(t->free));
      if (t->bytes) fprintf(f, "@bytes[%d, %s]: %llu\n", s, L->name, U(t->bytes));
      if (t->realloc) fprintf(f, "@realloc[%d, %s]: %llu\n", s, L->name, U(t->realloc));
      if (t->peak) fprintf(f, "@peak[%d, %s]: %llu\n", s, L->name, U(t->peak));
    }
    for (int k = 0; k < 3; k++)
      if (kind[s][k]) fprintf(f, "@kind[%d, %s]: %llu\n", s, KIND[k], U(kind[s][k]));
    if (escaped[s]) fprintf(f, "@escaped[%d]: %llu\n", s, U(escaped[s]));
    if (s) {
      fprintf(f, "@tnum[%d]: %d\n", s, tnum[s]);
      fprintf(f, "@tname[%d]: %s\n", s, tname[s]);
    }
  }
  fprintf(f, "\n");
}

static void report_level(FILE *f, struct level *L) {
  if (L->alloc == 0) return;                 /* not seen: unpatched, or --heap */
  const char *n = L->name;
  char name[64];
  fprintf(f, "@hook_alloc[%s]: %llu\n", n, U(L->alloc));
  fprintf(f, "@hook_free[%s]: %llu\n", n, U(L->free));
  fprintf(f, "@hook_bytes[%s]: %llu\n", n, U(L->bytes));
  fprintf(f, "@hook_realloc[%s]: %llu\n", n, U(L->realloc));
  fprintf(f, "@hook_inplace[%s]: %llu\n", n, U(L->inplace));
  fprintf(f, "@hook_peak[%s]: %llu\n", n, U(L->peak));
  fprintf(f, "@peakrun[%s]: %llu\n", n, U(L->peak));
  fprintf(f, "@maxreuse[%s]: %llu\n", n, U(L->maxreuse));
  fprintf(f, "@distinct[%s]: %llu\n", n, U(L->nent));
  fprintf(f, "@hook_fbytes[%s]: %llu\n", n, U(L->fbytes));
  fprintf(f, "@beyond254[%s]: %llu\n", n, U(L->beyond254));
  fprintf(f, "@poison16[%s]: %llu\n", n, U(L->poison16));
  snprintf(name, sizeof name, "size_%s", n);       hist_out(f, name, L->sizeh, 0, 0);
  snprintf(name, sizeof name, "reuse_%s", n);      hist_out(f, name, L->hist, 0, 0);
  snprintf(name, sizeof name, "reusebytes_%s", n); hist_out(f, name, L->histb, 0, 0);
  snprintf(name, sizeof name, "prior_%s", n);      hist_out(f, name, L->prior, L->prior0, 1);
  /* objects per address, over the addresses the level handed out */
  uint64_t opa[64] = {0};
  for (uint64_t i = 0; i < L->nslots; i++)
    if (L->tab[i].reuses) opa[lg(L->tab[i].reuses)]++;
  snprintf(name, sizeof name, "addresses_%s", n);  hist_out(f, name, opa, 0, 0);
}
#undef U

#ifndef MEMHOOK_FREESTANDING

/* ---- hosted: level 0 through the memory methods, mmap'd tables, a destructor ---- */

static sqlite3_mem_methods orig;

static void *hk_malloc(int n) {
  kind[cur][0]++;
  allocated(&L0, (size_t)n);
  void *p = orig.xMalloc(n);
  returned(&L0, p, p ? (size_t)orig.xSize(p) : 0);
  return p;
}
static void hk_free(void *p) {
  if (p) { kind[cur][2]++; released(&L0, p); }
  orig.xFree(p);
}
static void *hk_realloc(void *p, int n) {
  kind[cur][1]++;
  L0.realloc++; L0.t[cur].realloc++;
  if (p) released(&L0, p);
  if (n) allocated(&L0, (size_t)n);
  void *q = orig.xRealloc(p, n);
  if (q && q == p) L0.inplace++;
  returned(&L0, q, q ? (size_t)orig.xSize(q) : 0);
  return q;
}

static void *map(size_t bytes) {
  void *t = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (t == MAP_FAILED) { perror("memhook: mmap"); abort(); }
  return t;
}
static void table(struct level *L) {
  L->tab = map(TSIZE * sizeof(struct ent));
  L->keys = map(TSIZE * sizeof(uint64_t));
  L->nslots = TSIZE;
  L->bits = TBITS;
}

__attribute__((destructor)) static void report(void) {
  const char *path = getenv("A1_OUT");
  FILE *f = path ? fopen(path, "w") : stderr;
  if (!f) { perror("memhook: A1_OUT"); return; }
  struct level *levels[3] = { &L0, &L1, &L2 };
  fprintf(f, "memhook: three levels recorded from inside the process\n\n");
  report_tests(f, levels, 3);
  for (int l = 0; l < 3; l++) report_level(f, levels[l]);
  if (f != stderr) fclose(f);
}

__attribute__((constructor)) static void install(void) {
  table(&L0); table(&L1); table(&L2);
  if (sqlite3_config(SQLITE_CONFIG_GETMALLOC, &orig) != SQLITE_OK) {
    fprintf(stderr, "memhook: GETMALLOC failed\n");
    abort();
  }
  sqlite3_mem_methods m = orig;
  m.xMalloc = hk_malloc;
  m.xFree = hk_free;
  m.xRealloc = hk_realloc;
  if (sqlite3_config(SQLITE_CONFIG_MALLOC, &m) != SQLITE_OK) {
    fprintf(stderr, "memhook: CONFIG_MALLOC failed\n");
    abort();
  }
}

#else

/* ---- freestanding: the domain hands over memsys5's table and asks for the report ---- */

/* memsys5's table for an arena of `arena_size` bytes at `arena`, in `table`, which
   must hold speedtest1_hook_table_bytes(arena_size). Called before sqlite3_initialize. */
size_t speedtest1_hook_table_bytes(size_t arena_size) { return arena_size / 64 * sizeof(struct ent); }
void speedtest1_hook_install(void *arena, size_t arena_size, void *table) {
  memset(table, 0, speedtest1_hook_table_bytes(arena_size));
  L2.tab = table;
  L2.nslots = arena_size / 64;
  L2.base = ADDR(arena);
  L2.gran = 64;
}

void speedtest1_hook_report(void) {
  struct level *levels[2] = { &L1, &L2 };
  fprintf(stderr, "memhook: two levels recorded from inside the domain\n\n");
  report_tests(stderr, levels, 2);
  for (int l = 0; l < 2; l++) report_level(stderr, levels[l]);
}

#endif
