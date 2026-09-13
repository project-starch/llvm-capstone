/* What one call to PostgreSQL's memory-context interface looks like on disk,
 * so a replay in a domain can make the same call.
 *
 * This file is the interface between two repositories and it lives here,
 * where the replay is, because a domain build has to be self-contained. The
 * recorder is in the paper's experiments/a11/postgres, and its Makefile
 * fetches this file at a pinned commit and checks its hash, the way the A1
 * instrument is fetched. One definition, two readers.
 *
 * The recorder runs on x86 inside a real backend; the replay links the same
 * mmgr sources freestanding and drives them from this file.  Both read this
 * header, so the format has one definition.
 *
 * Identities, never addresses.  A replay allocates its own memory, so a record
 * names an object by an id the recorder assigns at the allocation and retires
 * at the free.  An address that comes back from the allocator carries a NEW
 * id, because it is a new object.  A context is named the same way, by an id
 * assigned at its creation.
 *
 * Little-endian, packed, fixed 40 bytes per record.  `endian` in the header is
 * written as the number below and read back to prove the reader agrees.
 */
#ifndef A11TRACE_H
#define A11TRACE_H
#include <stdint.h>

#define A11_MAGIC   "A11TRACE"
#define A11_VERSION 1u
#define A11_ENDIAN  0x0102030405060708ull

enum {
  A11_ALLOC = 1,     /* ctx, ptr = the id assigned, s1 = bytes asked for      */
  A11_FREE,          /* ctx, ptr                                              */
  A11_REALLOC,       /* ctx, ptr = the old id, aux = the new id, s1 = bytes   */
  A11_RESET,         /* ctx: every object alive in it dies, no record each    */
  A11_DELETE,        /* ctx: the same, and the children below it              */
  A11_CREATE_ASET,   /* ctx = the id assigned, aux = parent (0 = none),       */
  A11_CREATE_GEN,    /*   ptr = a hash of the name, s1..s3 = the type's size  */
  A11_CREATE_SLAB,   /*   parameters, in the order of its Create function     */
  A11_CREATE_BUMP,
  A11_BLOCKS,        /* not a call: what the manager took from the level below
                        over this file, aux = most held at once, s1 taken,
                        s2 given back, s3 grown or moved                      */
};

/* A backend is forked from the postmaster, so its history begins in another
 * process's file.  A child writes its own file and says, in the header, whose
 * file its history begins in and how many of that file's records belong to it:
 * the ones written before the fork, and not the ones the parent wrote after.
 * A replay of one process therefore reads `prefix` records of the file of
 * `ppid`, recursively, and then this one.  Identities are inherited across the
 * fork, so the two files speak of the same contexts and the same objects. */
struct a11_head {
  char     magic[8];     /* A11_MAGIC, not terminated */
  uint32_t version;
  uint32_t recsize;      /* sizeof(struct a11_rec), so a reader can refuse */
  uint64_t endian;       /* A11_ENDIAN */
  uint32_t pid;
  uint32_t ppid;         /* whose file this one continues, 0 if none */
  uint64_t prefix;       /* records of that file that belong to this history */
  uint32_t nphase;       /* bytes of `phase` used */
  uint32_t _pad;         /* written, so the size is the same on both sides: a
                            compiler would round 76 up to 80 and a reader that
                            computed 76 would then be one field out */
  char     phase[32];    /* what the run was, from A11_PHASE */
};                       /* 80 bytes */

struct a11_rec {
  uint32_t op;
  uint32_t ctx;
  uint32_t ptr;
  uint32_t aux;
  uint64_t s1, s2, s3;
};

/* A trace ends with a footer record, so a truncated file is detectable: op is
 * zero, and the three size fields carry what the recorder counted. */
#define A11_END 0u

#endif
