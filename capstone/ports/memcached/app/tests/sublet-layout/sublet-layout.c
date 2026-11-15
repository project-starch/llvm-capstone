/* The unit test of patch 0006's layout fix (docs/plans/2026-11-15-memcached-port-layout-fix.md).
 *
 * One program, linked against slabs.c and cache.c of the allocators port's Sublet build
 * (MCP_SUBLET=ON: the shims and this application's patch 0006), run as `sublet-layout MODE`:
 *
 *   scan           hand out a slab chunk and a cache object, load every 16-byte slot inside each
 *                  one's bounds, and print LEAK for every tagged reference whose bounds reach past
 *                  the object's own. Exit 1 if there is one, else 0.
 *   stale          free a chunk with slabs_free, then read through the caller's reference. The read
 *                  must fault (cause 25).
 *   bounds         read one byte past a small chunk. The read must fault (cause 28).
 *   move           fill one page of a large class, move it to the global page pool with
 *                  slabs_finalize_page_move, use a chunk of the class's other page and a chunk of
 *                  another class that takes the moved page, then read a chunk of the moved page. The
 *                  last read must fault (cause 25).
 *   move-prealloc  as move, with every page cut from one block passed to slabs_init, as -L cuts
 *                  them. A chunk of an earlier page and one of the class's other page must survive
 *                  the move; the read of the moved page's chunk must fault (cause 25).
 *
 * Every access that may fault is announced on stdout before it runs (ACCESS-BEGIN), so the last
 * announcement names the access the launcher's fault line belongs to. */
#include "mc_slabs_shim.h"
#include "port.h"
#include "slabs.h"
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LARGE_REQUEST (96u * 1024u) /* a class with about 8 chunks per 1 MiB page */
#define SMALL_REQUEST 100u          /* class 1, a chunk below 4 KiB, so its bounds are exact */
#define BLOCK_BYTES ((size_t)64 << 20)

static int scan_mode(void);
static int stale_mode(void);
static int bounds_mode(void);
static int move_mode(int prealloc);
static int scan_object(const char *what, void *object);
static void fill(void *p, unsigned char value);
static unsigned char read_byte(const char *what, const void *p);

/* port.h's failure hook, as the corpus drivers define it: an infrastructure failure is never a
 * verdict. */
_Noreturn void mcp_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75);
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc != 2) {
    fprintf(stderr, "usage: sublet-layout scan|stale|bounds|move|move-prealloc\n");
    return 75;
  }
  printf("LAYOUT sizeof(item)=%zu offsetof(item,data)=%zu sizeof(item_chunk)=%zu\n",
         sizeof(item), offsetof(item, data), sizeof(item_chunk));

  const char *mode = argv[1];
  if (!strcmp(mode, "scan")) {
    return scan_mode();
  } else if (!strcmp(mode, "stale")) {
    return stale_mode();
  } else if (!strcmp(mode, "bounds")) {
    return bounds_mode();
  } else if (!strcmp(mode, "move")) {
    return move_mode(0);
  } else if (!strcmp(mode, "move-prealloc")) {
    return move_mode(1);
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 75;
  }
}

/* A chunk fresh off a class's free list, and a cache object fresh off its cache's free list. */
static int scan_mode(void) {
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  unsigned id = slabs_clsid(SMALL_REQUEST);
  item *chunk = slabs_alloc(id, 0);
  if (chunk == NULL) mcp_fail(1);
  printf("CHUNK class=%u size=%u\n", id, slabs_size(id));

  /* Two objects back on the free list, so the one handed out next held a link to the other. */
  cache_t *cache = cache_create("sublet-layout", 64, sizeof(void *));
  if (cache == NULL) mcp_fail(2);
  void *a = cache_alloc(cache), *b = cache_alloc(cache);
  if (a == NULL || b == NULL) mcp_fail(3);
  cache_free(cache, b);
  cache_free(cache, a);
  void *object = cache_alloc(cache);
  if (object == NULL) mcp_fail(4);

  int leaks = scan_object("slab-chunk", chunk) + scan_object("cache-object", object);
  printf("SCAN leaks=%d\n", leaks);
  return leaks ? 1 : 0;
}

static int stale_mode(void) {
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  unsigned id = slabs_clsid(SMALL_REQUEST);
  item *chunk = slabs_alloc(id, 0);
  if (chunk == NULL) mcp_fail(10);
  fill((char *)chunk + sizeof(item), 0xA1);
  slabs_free(chunk, id);
  printf("FREED chunk by slabs_free\n");
  read_byte("stale-chunk", (char *)chunk + sizeof(item));
  printf("UNEXPECTED the stale read completed\n");
  return 3;
}

static int bounds_mode(void) {
  slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  unsigned id = slabs_clsid(SMALL_REQUEST);
  item *chunk = slabs_alloc(id, 0);
  if (chunk == NULL) mcp_fail(20);
  read_byte("chunk-last-byte", (char *)chunk + slabs_size(id) - 1);
  read_byte("one-past-chunk", (char *)chunk + slabs_size(id));
  printf("UNEXPECTED the read past the chunk completed\n");
  return 3;
}

/* The page mover's last step, slabs_finalize_page_move, on a page whose chunks a caller still holds:
 * the mover would have rescued them, so every reference to them is stale. */
static int move_mode(int prealloc) {
  if (prealloc) {
    void *block = aligned_alloc((size_t)1 << 20, BLOCK_BYTES);
    if (block == NULL) mcp_fail(30);
    slabs_init(BLOCK_BYTES, settings.factor, true, NULL, block, false);
  } else {
    slabs_init(settings.maxbytes, settings.factor, false, NULL, NULL, false);
  }
  item *early = slabs_alloc(slabs_clsid(SMALL_REQUEST), 0); /* a page cut before the moved one */
  if (early == NULL) mcp_fail(31);
  fill((char *)early + sizeof(item), 0xE1);

  unsigned id = slabs_clsid(LARGE_REQUEST);
  unsigned free_chunks = 0, perslab = 0;
  free_chunks = slabs_available_chunks(id, NULL, &perslab);
  printf("CLASS large=%u size=%u perslab=%u free=%u\n", id, slabs_size(id), perslab, free_chunks);
  /* Hand out every chunk of the class's first page, then one chunk of a second page. */
  item *held[64];
  unsigned on_first = free_chunks ? free_chunks : perslab;
  if (on_first == 0 || on_first > 64) mcp_fail(32);
  for (unsigned i = 0; i < on_first; i++) {
    held[i] = slabs_alloc(id, 0);
    if (held[i] == NULL) mcp_fail(33);
    fill((char *)held[i] + sizeof(item), 0xA1);
  }
  item *other = slabs_alloc(id, 0);
  if (other == NULL) mcp_fail(34);
  fill((char *)other + sizeof(item), 0xB1);
  uint32_t size = 0, per = 0;
  char *page = slabs_peek_page(id, &size, &per);
  if (page == NULL) mcp_fail(35);
  unsigned long page_addr = (unsigned long)(uintptr_t)page;
  printf("PAGE first=%#lx held0=%#lx other=%#lx\n", page_addr, (unsigned long)(uintptr_t)held[0],
         (unsigned long)(uintptr_t)other);

  slabs_finalize_page_move(id, SLAB_GLOBAL_PAGE_POOL, page);
  printf("MOVED the first page of class %u to the global page pool\n", id);

  if (read_byte("other-page-chunk", (char *)other + sizeof(item)) != 0xB1) mcp_fail(36);
  if (read_byte("earlier-page-chunk", (char *)early + sizeof(item)) != 0xE1) mcp_fail(37);
  slabs_free(other, id);
  printf("FREED the other page's chunk by slabs_free\n");
  if (!prealloc) {
    /* A class with no page yet takes the moved page from the global pool. */
    unsigned id2 = slabs_clsid(5000);
    item *taker = slabs_alloc(id2, 0);
    if (taker == NULL) mcp_fail(38);
    unsigned long at = (unsigned long)(uintptr_t)taker;
    printf("REUSED class=%u chunk=%#lx in_moved_page=%d\n", id2, at, at - page_addr < (1ul << 20));
    fill((char *)taker + sizeof(item), 0xC1);
    if (read_byte("taker-chunk", (char *)taker + sizeof(item)) != 0xC1) mcp_fail(39);
    slabs_free(taker, id2);
    printf("FREED the taker's chunk by slabs_free\n");
  }
  read_byte("moved-page-chunk", (char *)held[0] + sizeof(item));
  printf("UNEXPECTED the read of the moved page's chunk completed\n");
  return 3;
}

/* Every tagged reference inside object's bounds; LEAK for one whose bounds reach past them. */
static int scan_object(const char *what, void *object) {
  unsigned long base = __builtin_capstone_cap_get_base(object);
  unsigned long end = __builtin_capstone_cap_get_end(object);
  unsigned long cursor = (unsigned long)(uintptr_t)object;
  printf("OBJECT %s base=%#lx end=%#lx\n", what, base, end);
  int leaks = 0;
  for (unsigned long at = (base + 15) & ~15ul; at + 16 <= end; at += 16) {
    void *slot = *(void *volatile *)((char *)object + (long)(at - cursor));
    if (!__builtin_capstone_cap_get_tag(slot)) {
      continue;
    }
    unsigned long sb = __builtin_capstone_cap_get_base(slot);
    unsigned long se = __builtin_capstone_cap_get_end(slot);
    int beyond = sb < base || se > end;
    printf("%s %s offset=%ld base=%#lx end=%#lx\n", beyond ? "LEAK" : "REF", what,
           (long)(at - cursor), sb, se);
    leaks += beyond;
  }
  return leaks;
}

static void fill(void *p, unsigned char value) {
  memset(p, value, 16);
}

/* The one load each announced access makes. */
__attribute__((noinline)) static unsigned char read_probe(const volatile unsigned char *p) {
  return *p;
}

static unsigned char read_byte(const char *what, const void *p) {
  printf("ACCESS-BEGIN %s addr=%#lx\n", what, (unsigned long)(uintptr_t)p);
  unsigned char v = read_probe(p);
  printf("ACCESS-DONE %s value=%#x\n", what, v);
  return v;
}
