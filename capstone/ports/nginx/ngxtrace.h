/* What one call to nginx's pool interface looks like on disk, so a replay in a domain can make
 * the same call.
 *
 * This file is the interface between two repositories and it lives here, where the replay is,
 * because a domain build has to be self-contained. The recorder is in the paper's
 * experiments/a11/nginx, and its Makefile fetches this file at a pinned commit and checks its
 * hash, the way a11trace.h and the A1 instrument are fetched. One definition, two readers.
 *
 * WHY A TRACE AT ALL. The driver in port/ngx_domain.c makes calls I chose. It proves the port
 * answers its interface, and it cannot say what nginx asks for: how large the pools are, how many
 * objects each holds before it dies, how often a reset comes instead of a destroy. A cost measured
 * against calls I invented would be a cost of my invention. A1 already runs real nginx under wrk,
 * but it COUNTS, and a count cannot be replayed.
 *
 * IDENTITIES, NEVER ADDRESSES. A replay allocates its own memory, so a record names an object by
 * an id the recorder assigns at the allocation and retires at the free. An address that comes
 * back from the allocator carries a NEW id, because it is a new object. A pool is named the same
 * way, by an id assigned at its creation.
 *
 * WHAT IS NOT RECORDED. What the pool does inside itself: the block it chains when one is full,
 * the ngx_pool_large_t it carves for its own list, the cleanup record's own allocation. Those are
 * the allocator's answer to the call, and a replay that read them back would be replaying the
 * answer rather than asking the question. They are counted instead, in NGXT_BLOCKS, which is what
 * a replay can be held against.
 *
 * Little-endian, packed, fixed 40 bytes per record, the same shape as a11trace.h so that the two
 * readers stay recognisably one design. `endian` in the header is written as the number below and
 * read back to prove the reader agrees.
 */
#ifndef NGXTRACE_H
#define NGXTRACE_H
#include <stdint.h>

#define NGXT_MAGIC   "NGXTRACE"
#define NGXT_VERSION 1u
#define NGXT_ENDIAN  0x0102030405060708ull

enum {
  NGXT_CREATE = 1,   /* pool = the id assigned, s1 = the size asked for        */
  NGXT_DESTROY,      /* pool                                                   */
  NGXT_RESET,        /* pool: every object alive in it dies, no record each    */
  NGXT_PALLOC,       /* pool, obj = the id assigned, s1 = bytes                */
  NGXT_PNALLOC,      /* the same, and unaligned by contract                    */
  NGXT_PCALLOC,      /* the same, and zeroed                                   */
  NGXT_PMEMALIGN,    /* the same, aux = the alignment asked for                */
  NGXT_PFREE,        /* pool, obj: only a large allocation can be freed alone  */
  NGXT_CLEANUP,      /* pool, obj = the id assigned, s1 = the data bytes       */
  NGXT_BLOCKS,       /* not a call: what the pool took from level 0 over this
                        file, aux = most held at once, s1 taken, s2 given back,
                        s3 asked for with an alignment of its own              */
  NGXT_END,          /* s1 = records, s2 = pools, s3 = objects                 */
};

/* A worker is forked from the master without exec, so its history begins in another process's
 * file. A child writes its own file and says, in the header, whose file its history begins in and
 * how many of that file's records belong to it: the ones written before the fork, and not the
 * ones the parent wrote after. A replay of one worker therefore reads `prefix` records of the
 * file of `ppid`, recursively, and then this one.
 *
 * This is not a detail for nginx. The master reads the configuration in pools of its own and then
 * forks, so every worker's history starts inside the master's, and a replay that began at the
 * worker's own first record would create pools that already existed. */
struct ngxt_head {
  char     magic[8];     /* NGXT_MAGIC, not terminated */
  uint32_t version;
  uint32_t recsize;      /* sizeof(struct ngxt_rec), so a reader can refuse */
  uint64_t endian;       /* NGXT_ENDIAN */
  uint32_t pid;
  uint32_t ppid;         /* whose file this one continues, 0 if none */
  uint64_t prefix;       /* records of that file that belong to this history */
  uint32_t nphase;       /* bytes of `phase` used */
  uint32_t _pad;         /* written, so the size is the same on both sides: a compiler would round
                            76 up to 80 and a reader that computed 76 would then be one field out */
  char     phase[32];    /* which rung the run was, from NGXT_PHASE */
};                       /* 80 bytes */

struct ngxt_rec {
  uint32_t op;
  uint32_t pool;         /* pool id, 0 for none */
  uint32_t obj;          /* object id, or 0 */
  uint32_t aux;          /* an alignment, or a peak, depending on op */
  uint64_t s1, s2, s3;
};                       /* 40 bytes */

/* Both sizes are asserted, not commented. Three readers have to agree about them: the recorder
   that writes, the replay that reads, and the python that inspects a file by hand. A compiler
   that padded either one differently would be found here rather than at a record boundary a
   thousand records in. The typedef form, because a _Static_assert expression cannot stand at file
   scope in C99. */
typedef char ngxt_head_is_80[(sizeof(struct ngxt_head) == 80) ? 1 : -1];
typedef char ngxt_rec_is_40[(sizeof(struct ngxt_rec) == 40) ? 1 : -1];

#endif
