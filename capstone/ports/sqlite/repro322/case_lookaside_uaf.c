/* A constructed use-after-free at the LOOKASIDE layer, and the lookaside counterpart of
 * case_mem5design.c (row 25). Both are demonstrators, not collected upstream defects, and
 * both exist for the same reason: to show what SQLite's allocator design does to a stale
 * pointer on a machine with capabilities but no temporal enforcement.
 *
 * case_lookaside_tagmap.c established the weaker half -- a freed slot's granule 0 still
 * holds a VALID capability, so the stale load does not fault. This case establishes the
 * half that actually hurts: the slot gets HANDED BACK OUT, so a stale pointer reads a
 * DIFFERENT, LIVE value. Not a crash, not garbage: somebody else's data, silently.
 *
 * sqlite3DbFreeNN pushes a freed slot onto the head of the free list,
 *     pBuf->pNext = db->lookaside.pFree;  db->lookaside.pFree = pBuf;
 * and sqlite3DbMallocRawNN pops that same head,
 *     if( (pBuf = db->lookaside.pFree)!=0 ){ db->lookaside.pFree = pBuf->pNext; return pBuf; }
 * so the list is strict LIFO: the next allocation of a fitting size gets the slot just
 * freed. This case does not assume that -- it LOCATES both buffers in the pool and reports
 * whether the indices match.
 *
 * Everything here runs through the public API. The only thing a real bug would add is
 * holding the pointer, which is exactly what the saved `victim` below stands in for:
 *   sqlite3_bind_text(..., SQLITE_TRANSIENT)  -> sqlite3VdbeMemGrow -> sqlite3DbMallocRaw
 *                                             -> a lookaside slot
 *   sqlite3_clear_bindings()                  -> sqlite3VdbeMemRelease -> sqlite3DbFree
 *                                             -> the slot goes back on the free list
 *
 * The pool is ours (SQLITE_DBCONFIG_LOOKASIDE takes a caller-supplied buffer), so every
 * slot address is known as pool + i*sz and the buffers can be found by their contents.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

#define LA_SZ   128            /* ROUNDDOWN8, and must exceed sizeof(LookasideSlot*) */
#define LA_CNT  40
#define PATLEN  32
static unsigned char la_pool[LA_SZ * LA_CNT] __attribute__((aligned(16)));

static void kv(const char *k, unsigned long v){
  out_text("lauaf "); out_text(k); out_text("="); out_uint(v); out_text("\n");
}

/* Which slot, if any, currently holds PATLEN repeats of c? Returns -1 if none.
 * Scans every slot at every 8-byte offset, because the Mem buffer starts at the slot's
 * base but nothing in the API promises that. */
static int find_pattern(unsigned char c){
  int i, off, k;
  for(i = 0; i < LA_CNT; i++){
    unsigned char *base = la_pool + (unsigned)i * LA_SZ;
    for(off = 0; off + PATLEN <= LA_SZ; off += 8){
      for(k = 0; k < PATLEN; k++) if(base[off + k] != c) break;
      if(k == PATLEN) return i;
    }
  }
  return -1;
}

static void fill(unsigned char *buf, unsigned char c){
  int k; for(k = 0; k < PATLEN; k++) buf[k] = c; buf[PATLEN] = 0;
}

static int run_case(void){
  unsigned char patA[PATLEN + 1], patZ[PATLEN + 1];
  sqlite3 *db = 0; sqlite3_stmt *st = 0; char *e = 0;
  int rc, iA, iZ, k;
  unsigned char *victim = 0;

  if (repro_init()) return 1;
  kv("slot_size", LA_SZ);
  kv("slot_count", LA_CNT);

  fill(patA, 'A');
  fill(patZ, 'Z');

  rc = sqlite3_open(":memory:", &db);
  kv("open_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc != SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_db_config(db, SQLITE_DBCONFIG_LOOKASIDE, la_pool, LA_SZ, LA_CNT);
  kv("db_config_lookaside_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc != SQLITE_OK){
    out_text("lauaf WARNING no pool installed; this case establishes nothing\n");
    sqlite3_close(db); out_text("lauaf NOTRAP done\n"); return 0;
  }

  rc = sqlite3_exec(db, "CREATE TABLE t(a);", 0, 0, &e);
  kv("setup_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(e){ sqlite3_free(e); e = 0; }

  rc = sqlite3_prepare_v2(db, "INSERT INTO t VALUES(?1)", -1, &st, 0);
  kv("prepare_rc", (unsigned long)(rc < 0 ? -rc : rc));
  if(rc != SQLITE_OK){ sqlite3_close(db); out_text("lauaf NOTRAP done\n"); return 0; }

  /* STEP 1: put 'A' x 32 into a lookaside slot through the public API. */
  rc = sqlite3_bind_text(st, 1, (const char *)patA, PATLEN, SQLITE_TRANSIENT);
  kv("bindA_rc", (unsigned long)(rc < 0 ? -rc : rc));
  iA = find_pattern('A');
  kv("slot_holding_A", (unsigned long)(iA < 0 ? 0xffffffffUL : (unsigned long)iA));
  if(iA < 0){
    out_text("lauaf the bound value did NOT land in a lookaside slot"
             " -- it went to memsys5 instead, so this case establishes nothing\n");
    sqlite3_finalize(st); sqlite3_close(db);
    out_text("lauaf NOTRAP done\n"); return 0;
  }
  /* The stale pointer a buggy SQLite would have kept. */
  victim = la_pool + (unsigned)iA * LA_SZ;

  /* STEP 2: free it. clear_bindings releases the Mem buffer back into the pool, and
   * nothing else is called in between, so the slot sits at the head of the free list. */
  rc = sqlite3_clear_bindings(st);
  kv("clear_bindings_rc", (unsigned long)(rc < 0 ? -rc : rc));

  /* Sample the tag HERE, while the slot is actually on the free list. This is the moment
   * case_lookaside_tagmap.c measures (39 of 40 tagged); sampling it any later is sampling a
   * different state, because the slot gets reallocated and overwritten below. */
  {
    void *g0 = ((void **)victim)[0];
    kv("freed_state_granule0_tagged",
       (unsigned long)(__builtin_capstone_cap_get_tag(g0) ? 1 : 0));
  }

  /* STEP 3: reallocate. A fitting request pops the free-list head -- the same slot. */
  rc = sqlite3_bind_text(st, 1, (const char *)patZ, PATLEN, SQLITE_TRANSIENT);
  kv("bindZ_rc", (unsigned long)(rc < 0 ? -rc : rc));
  iZ = find_pattern('Z');
  kv("slot_holding_Z", (unsigned long)(iZ < 0 ? 0xffffffffUL : (unsigned long)iZ));
  out_text(iZ == iA ? "lauaf REUSED the same slot (LIFO free list, as predicted)\n"
                    : "lauaf different slot -- the aliasing below is weaker than claimed\n");

  /* STEP 4: THE DANGLING READ. victim still points at the slot we were handed in step 1
   * and freed in step 2. On a machine with temporal enforcement this load is the one that
   * must trap. Report what it actually returns. */
  {
    int sawA = 0, sawZ = 0, off;
    for(off = 0; off + PATLEN <= LA_SZ; off += 8){
      for(k = 0; k < PATLEN; k++) if(victim[off + k] != 'A') break;
      if(k == PATLEN) sawA = 1;
      for(k = 0; k < PATLEN; k++) if(victim[off + k] != 'Z') break;
      if(k == PATLEN) sawZ = 1;
    }
    out_text("lauaf stale read did NOT fault; first 8 bytes: ");
    for(k = 0; k < 8; k++){ out_uint((unsigned)victim[k]); out_text(" "); }
    out_text("\n");
    kv("stale_still_sees_A", (unsigned long)sawA);
    kv("stale_now_sees_Z", (unsigned long)sawZ);
    if(sawZ && !sawA)
      out_text("lauaf  <== CONFIRMED: the stale pointer reads a DIFFERENT, LIVE value."
               " A use-after-free at the lookaside layer is silent AND wrong-object.\n");
    else if(sawA)
      out_text("lauaf  the old bytes are still there; the slot was not yet overwritten\n");
    else
      out_text("lauaf  neither pattern found at the slot base\n");
  }

  /* For contrast, the SAME word AFTER the slot was reallocated and filled with 32 bytes of
   * text. It is scalar now, so tag 0 -- which is NOT a contradiction of the freed-state
   * reading above, it is a different moment. An earlier version of this case reported only
   * this one and it read as though it contradicted case_lookaside_tagmap.c. */
  {
    void *g0 = ((void **)victim)[0];
    kv("after_reuse_granule0_tagged(text, so 0 expected)",
       (unsigned long)(__builtin_capstone_cap_get_tag(g0) ? 1 : 0));
  }

  /* PHASE 3, and the sharpest form of the result. Above, the slot was reused for another
   * bound VALUE. Let SQLite's OWN internals reallocate it instead: preparing a statement
   * builds a parse tree, and parse-tree nodes come from lookaside. The host oracle on 3.22.0
   * shows the slot is handed to the engine and its first 16 bytes become two pointer-sized
   * words followed by zeros -- the shape of a sqlite3DbMallocZero'd internal object with two
   * pointer fields, not of a free-list entry (LookasideSlot holds a single pNext).
   *
   * So the question this phase answers is the one that matters for the control arm: are those
   * words TAGGED? If they are, a dangling read at the lookaside layer does not merely go
   * unnoticed -- it hands the reader usable capabilities to live engine state. */
  {
    sqlite3_stmt *st3 = 0;
    int rc3;
    sqlite3_clear_bindings(st);          /* give the slot back one more time */
    rc3 = sqlite3_prepare_v2(db,
            "SELECT a FROM t WHERE a <> 'x' ORDER BY a", -1, &st3, 0);
    kv("phase3_internal_prepare_rc", (unsigned long)(rc3 < 0 ? -rc3 : rc3));

    out_text("lauaf phase3 stale first 32 bytes: ");
    for(k = 0; k < PATLEN; k++){ out_uint((unsigned)victim[k]); out_text(" "); }
    out_text("\n");

    /* Query the tags WITHOUT dereferencing, so this phase reports either way. */
    {
      void *w0 = ((void **)victim)[0];
      void *w1 = ((void **)victim)[1];
      int t0 = __builtin_capstone_cap_get_tag(w0) ? 1 : 0;
      int t1 = __builtin_capstone_cap_get_tag(w1) ? 1 : 0;
      int in0 = ((unsigned char *)w0 >= la_pool &&
                 (unsigned char *)w0 <  la_pool + sizeof(la_pool)) ? 1 : 0;
      kv("phase3_word0_tagged", (unsigned long)t0);
      kv("phase3_word1_tagged", (unsigned long)t1);
      kv("phase3_word0_points_into_our_pool", (unsigned long)in0);
      if(t0){
        /* Safe BECAUSE it is tagged. Follow it one hop and read a few bytes: this is the
         * dangling read yielding a usable capability to something still live. */
        unsigned char *q = (unsigned char *)w0;
        unsigned sum = 0, i;
        for(i = 0; i < 16; i++) sum += q[i];
        out_text("lauaf phase3 FOLLOWED the stale capability one hop without faulting,"
                 " byte-sum=");
        out_uint(sum);
        out_text("\n  <== a freed lookaside slot handed out a USABLE capability to live"
                 " engine state\n");
      } else {
        out_text("lauaf phase3 word0 is UNTAGGED, so following it would fault\n");
      }
    }
    if(st3) sqlite3_finalize(st3);
  }

  sqlite3_finalize(st);
  sqlite3_close(db);
  out_text("lauaf NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("lauaf")
