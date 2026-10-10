/* Why 6 of 25 corpus domains FAULT on base Capstone, measured rather than inferred.
 *
 * memsys5 is a buddy allocator with an IN-BAND freelist. On free it writes
 *     struct Mem5Link { int next; int prev; };      /_ 8 bytes _/
 * at the START of the (possibly coalesced) block, via
 *     #define MEM5LINK(idx) ((Mem5Link *)(&mem5.zPool[(idx)*mem5.szAtom]))
 * and nothing else: aCtrl[] is a SEPARATE out-of-band array, and the
 * memset(0x55, size) poison is compiled out unless SQLITE_DEBUG (we do not set it).
 * So a freed block keeps all of its bytes and all of its capability tags EXCEPT
 * whatever those 8 bytes destroy.
 *
 * On Capstone a capability is 16 bytes with one tag bit per 16-byte granule, so an
 * 8-byte integer store into bytes 0..7 clears the tag of granule 0 (bytes 0..15) and
 * leaves every other granule intact. THAT is the whole incidental-catch mechanism:
 * a freed object whose granule 0 held a POINTER yields an untagged capability, and
 * the next dereference of it faults with cause 24. A freed object read only through
 * scalars, or through pointers living at offset >= 16, stays silent.
 *
 * This domain measures it directly instead of arguing it. It uses
 * __builtin_capstone_cap_get_tag(), which QUERIES a tag without dereferencing, so the
 * whole measurement runs to completion and prints a granule map rather than dying at
 * the first bad load.
 *
 * It also prints ABI ground truth. The offsets the analysis depends on cannot be read
 * from the amalgamation's internal structs here, so the layouts are replicated with
 * identical declarations; the compiler lays them out by the same rules.
 * NOTE: __builtin_offsetof, never the &((T*)0)->m trick -- pointer arithmetic on a NULL
 * capability is itself a cause-24 fault on Capstone (that is the fts5 azArg bug).
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

/* --- replicas, for ABI offsets only --- */
struct R_RtreeNode {            /* ext/rtree/rtree.c  -- rows new-5 / new-16 */
  struct R_RtreeNode *pParent;
  long long iNode;
  int nRef;
  int isDirty;
  unsigned char *zData;
  struct R_RtreeNode *pNext;
};
struct R_EditDist3Config {      /* ext/misc/spellfix.c -- row 24 */
  int nLang;
  void *a;
};
struct R_Hash {                 /* src/hash.h */
  unsigned int htsize;
  unsigned int count;
  void *first;
  void *ht;
};
struct R_SchemaHead {           /* src/sqliteInt.h -- row 4 */
  int schema_cookie;
  int iGeneration;
  struct R_Hash tblHash;
};
struct R_BtreeHead {            /* src/btreeInt.h -- row 15 */
  void *db;
  void *pBt;
  unsigned char inTrans, sharable, locked, hasIncrblobCur;
  int wantToLock;
  int nBackup;
  unsigned int iDataVersion;
  void *pNext;
  void *pPrev;
};

static void kv(const char *k, unsigned long v){
  out_text("mem5tag "); out_text(k); out_text("="); out_uint(v); out_text("\n");
}

/* A live allocation whose capability we copy into the block under test. */
static void *g_target;

/* Fill every 16-byte granule of a block with a valid capability, free the block,
 * then report each granule's tag. sizeof(void*)==16 here, so p[i] IS granule i. */
static void granule_map(unsigned nbytes){
  unsigned i, ngran, nlost = 0;
  void **p = (void **)sqlite3_malloc((int)nbytes);
  out_text("mem5tag -- block of "); out_uint(nbytes); out_text(" bytes: ");
  if(!p){ out_text("ALLOC FAILED\n"); return; }
  ngran = nbytes / (unsigned)sizeof(void *);

  for(i = 0; i < ngran; i++) p[i] = g_target;

  /* all tagged before the free? */
  for(i = 0; i < ngran; i++) if(!__builtin_capstone_cap_get_tag(p[i])) nlost++;
  out_text("granules="); out_uint(ngran);
  out_text(" untagged_before_free="); out_uint(nlost);
  out_text("\n");

  sqlite3_free(p);

  /* The deliberate use-after-free: read each granule back and QUERY its tag.
   * Loading an untagged capability is legal; only dereferencing it faults. */
  out_text("mem5tag    tag map after free (granule:tag) ");
  nlost = 0;
  for(i = 0; i < ngran; i++){
    int t = __builtin_capstone_cap_get_tag(p[i]) ? 1 : 0;
    if(!t) nlost++;
    if(i < 8){ out_uint(i); out_text(":"); out_uint((unsigned)t); out_text(" "); }
  }
  if(ngran > 8) out_text("...");
  out_text("\n");
  out_text("mem5tag    untagged_after_free="); out_uint(nlost);
  out_text(" of "); out_uint(ngran); out_text("\n");

  /* Show the Mem5Link ints that did it: the first two 32-bit words. */
  { unsigned int *w = (unsigned int *)p;
    out_text("mem5tag    in-band Mem5Link at offset 0: next=");
    out_uint((unsigned long)w[0]);
    out_text(" prev="); out_uint((unsigned long)w[1]);
    out_text("  (bytes 8..15 untouched: ");
    out_uint((unsigned long)w[2]); out_text(" ");
    out_uint((unsigned long)w[3]); out_text(")\n");
  }
}

static int run_case(void){
  if (repro_init()) return 1;

  kv("sizeof_pointer", (unsigned long)sizeof(void *));
  kv("sizeof_Mem5Link_bytes", 8);

  out_text("mem5tag -- ABI offsets of the structs freed by the FAULT cases this probe explains\n"
          "mem5tag    (RtreeNode: rtreecursor, rtreeinode0; EditDist3Config: spellfixoom;\n"
          "mem5tag     Schema: detachtrig; Btree: backupattach -- 5 of the 7 faults.\n"
          "mem5tag     fts5inplace and fts3snipor fault by other sub-mechanisms.)\n");
  kv("RtreeNode.pParent", __builtin_offsetof(struct R_RtreeNode, pParent));
  kv("RtreeNode.iNode",   __builtin_offsetof(struct R_RtreeNode, iNode));
  kv("RtreeNode.nRef",    __builtin_offsetof(struct R_RtreeNode, nRef));
  kv("RtreeNode.zData",   __builtin_offsetof(struct R_RtreeNode, zData));
  kv("RtreeNode.pNext",   __builtin_offsetof(struct R_RtreeNode, pNext));
  kv("EditDist3Config.nLang", __builtin_offsetof(struct R_EditDist3Config, nLang));
  kv("EditDist3Config.a",     __builtin_offsetof(struct R_EditDist3Config, a));
  kv("Schema.schema_cookie",  __builtin_offsetof(struct R_SchemaHead, schema_cookie));
  kv("Schema.tblHash.first",  __builtin_offsetof(struct R_SchemaHead, tblHash.first));
  kv("Schema.tblHash.ht",     __builtin_offsetof(struct R_SchemaHead, tblHash.ht));
  kv("Btree.db",              __builtin_offsetof(struct R_BtreeHead, db));
  kv("Btree.pBt",             __builtin_offsetof(struct R_BtreeHead, pBt));

  g_target = sqlite3_malloc(64);
  kv("target_alloc_ok", (unsigned long)(g_target != 0));
  if(!g_target) return 1;
  kv("target_tagged", (unsigned long)(__builtin_capstone_cap_get_tag(g_target) ? 1 : 0));

  granule_map(64);
  granule_map(128);
  granule_map(256);
  granule_map(1024);

  sqlite3_free(g_target);
  out_text("mem5tag NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("mem5tag")
