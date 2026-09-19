#ifndef PG_CONTEXTS_WORKLOAD_H
#define PG_CONTEXTS_WORKLOAD_H
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/memutils_internal.h"

static MemoryContext make_context(unsigned kind, MemoryContext parent) {
  if (kind == 0)
    return GenerationContextCreate(parent, "generation", 0, 2048, 32768);
  if (kind == 1)
    return SlabContextCreate(parent, "slab", 4096, 64);
  return BumpContextCreate(parent, "bump", 0, 2048, 32768);
}

static unsigned long policy_hash;
static void record_stats(MemoryContext c) {
  MemoryContextCounters v = {0};
  c->methods->stats(c, NULL, NULL, &v, false);
  policy_hash = policy_hash * 31 + v.nblocks;
  policy_hash = policy_hash * 31 + v.freechunks;
  policy_hash = policy_hash * 31 + v.totalspace;
  policy_hash = policy_hash * 31 + v.freespace;
}

#define REQUIRE(x)                                                             \
  do {                                                                         \
    if (!(x))                                                                  \
      return __LINE__;                                                         \
  } while (0)
static unsigned char *objects[2048];

/* The same workload runs against upstream native, upstream spatial and the
 * protected managers. Only the two capability-ABI geometry hashes are compared.
 */
static unsigned contexts_workload(unsigned kind) {
  MemoryContext root =
      AllocSetContextCreateInternal(NULL, "root", 0, 2048, 8192);
  TopMemoryContext = CurrentMemoryContext = root;
  MemoryContext c = make_context(kind, root);
  MemoryContext sibling = make_context(kind, root);
  unsigned char *survivor = MemoryContextAlloc(sibling, 64);
  survivor[0] = 91;
  policy_hash = 0;
  record_stats(c);
  for (unsigned i = 0; i < 2048; ++i) {
    objects[i] = MemoryContextAlloc(c, 64);
    REQUIRE(objects[i] != NULL);
    memset(objects[i], (i % 251) + 1, 64);
  }
  record_stats(c);
  for (unsigned i = 0; i < 2048; ++i) {
    REQUIRE(objects[i][0] == (i % 251) + 1);
    REQUIRE(objects[i][63] == (i % 251) + 1);
    if (kind != 2) {
      REQUIRE(GetMemoryChunkContext(objects[i]) == c);
      REQUIRE(GetMemoryChunkSpace(objects[i]) >= 64);
    }
  }
  if (kind != 2) {
    for (unsigned i = 0; i < 2048; i += 2)
      pfree(objects[i]);
    record_stats(c);
    for (unsigned i = 1; i < 2048; i += 2) {
      REQUIRE(objects[i][63] == (i % 251) + 1);
      pfree(objects[i]);
    }
    record_stats(c);
    /* Recycled Slab slots and Generation blocks must work repeatedly. */
    for (unsigned i = 0; i < 512; ++i) {
      unsigned char *p = MemoryContextAlloc(c, 64);
      memset(p, 17, 64);
      REQUIRE(repalloc(p, 64) == p);
      pfree(p);
    }
  }
  MemoryContextReset(c);
  record_stats(c);
  REQUIRE(survivor[0] == 91);
  if (kind != 1) {
    unsigned char *large = MemoryContextAlloc(c, 40000);
    memset(large, 27, 40000);
    REQUIRE(large[39999] == 27);
    if (kind == 0) {
      unsigned char *grown = repalloc(large, 60000);
      REQUIRE(grown && grown[39999] == 27);
      REQUIRE(repalloc(grown, 128) == grown);
      pfree(grown);
      unsigned char *small = MemoryContextAlloc(c, 32);
      memset(small, 19, 32);
      grown = repalloc(small, 256);
      REQUIRE(grown[31] == 19);
      pfree(grown);
    }
  }
  for (unsigned round = 0; round < 64; ++round) {
    MemoryContextReset(c);
    unsigned char *p = MemoryContextAlloc(c, 64);
    p[63] = 47;
    REQUIRE(p[63] == 47 && survivor[0] == 91);
  }
  /* All three types participate in an actual mixed context tree. */
  MemoryContext child = make_context((kind + 1) % 3, c);
  MemoryContext grandchild = make_context((kind + 2) % 3, child);
  ((unsigned char *)MemoryContextAlloc(grandchild, 64))[0] = 51;
  MemoryContextReset(c); /* upstream mcxt deletes descendants first */
  REQUIRE(c->firstchild == NULL && survivor[0] == 91);
  ((unsigned char *)MemoryContextAlloc(c, 64))[0] = 39;
  MemoryContextDelete(c);
  REQUIRE(survivor[0] == 91);
  MemoryContextDelete(sibling);
  MemoryContextDelete(root);
  TopMemoryContext = CurrentMemoryContext = NULL;
  return 0;
}
#undef REQUIRE
#endif
