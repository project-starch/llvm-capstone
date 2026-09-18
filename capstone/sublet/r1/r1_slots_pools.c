/* r1_slots_pools.c — the R1 release-to-reuse cost harness of the Sublet paper
 * (paper-nested-allocators/experiments/R1-release-cost.md), as a Capstone domain.
 *
 * One root region (the host's `--arena` grant, linear), one pool of B bytes carved from it, n
 * leaves partitioning the pool, an unrelated region of U bytes beside it, and three lifetime
 * patterns: `shared` (one ancestor withdrawal ends every leaf), `combined` (every even-indexed
 * leaf is freed and reissued first, then the ancestor withdrawal), `individual` (one object of S
 * bytes among sixteen live 64-byte objects is freed and reissued). Two arms: `S` (custom-sublet:
 * the primitives of sublet.h) and `P` (custom-spatial: the same fixture as plain pointers into one
 * alias, bookkeeping only). Every release is timed in EXCLUSIVE intervals — bookkeeping, the
 * REVOKE, the fill the RTL's uninitialised-region rule forces, INIT, reissue up to a checked load
 * and store in the returned region — plus the outer bracket, with mcycle/minstret read in the
 * domain as ladder_perf_domain.h and speedtest1_measure.c do.
 *
 * Host protocol: sqlite_host.user's `--speedtest1` path (metadata region 0, payload region 1, the
 * arena as region 2; the `--speedtest1 '<args>'` text at SQLITE_HC_SPEED_ARGS_OFF of the payload,
 * its length in metadata->offset). This harness reads its arguments once, then owns the whole
 * payload as its output sink (the host prints payload[0..length)). It returns 0x4EB1xxxx so the
 * host's speedtest1 branch prints `SQ: speedtest1-ran=` and does not classify the run as a fault.
 *
 * Arguments (one line, space separated):
 *   --arm S|P  --series nodes|bytes|heap|depth|object  --pattern shared|combined|individual
 *   --reps N (default 5)  --touch 0|1 (touch the unrelated region before each measurement, 1)
 *   --budget N (refuse a point that would push the run's minted nodes past N, default 50000)
 *   --calib (print the empty-bracket cost and return)  --stale (dereference one stale alias last)
 *
 * Output: one `R1 ...` line per point and repetition, `R1 plan ...` before each point, `R1 end ...`.
 * The node figure `nd=` is the number of revocation nodes minted for the fixture whose death the
 * timed release causes (splits + mrevs since the fixture was issued), which is what the RTL's
 * table pays for it; the cycle figures are exclusive, `tt=` the outer bracket.
 */
#include "sublet.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U   /* the monitor's share selector, as sqlite_capstone_domain.c */

typedef unsigned long ulong;

/* ------------------------------------------------------------------ freestanding runtime -- */
void *memset(void *d, int c, ulong n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, ulong n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
static int streq(const char *a, const char *b) { while (*a && *a == *b) { a++; b++; } return *a == *b; }
static ulong atou(const char *s) { ulong v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (ulong)(*s++ - '0'); return v; }

/* ------------------------------------------------------------------ the host's regions ---- */
static volatile struct sqlite_hostcall_v0 *meta;
static volatile char *payload;
static unsigned nshare;
static sublet_cap root;                 /* the arena grant, linear; carved front to back */
static ulong out_used, out_limit;     /* the host declares its region size in meta->result (4096 for
                                         sqlite_host.user, 65536 for the readback host); the args sit
                                         at half of it, and this sink owns the whole region once the
                                         args are copied out */
static ulong out_lines;               /* newlines written: `R1 end lines=N` lets a chunked transcript prove itself complete */
static void out(const char *s) {
  char *p = (char *)payload;
  if (!meta || !payload || !out_limit) return;
  while (*s && out_used + 1 < out_limit) { if (*s == '\n') out_lines++; p[out_used++] = *s++; }
  meta->length = out_used;
}
static void outu(ulong v) {
  char d[24], t[24]; unsigned n = 0, i = 0;
  do { d[n++] = (char)('0' + v % 10UL); v /= 10UL; } while (v);
  while (n) t[i++] = d[--n];
  t[i] = 0; out(t);
}
static void kv(const char *k, ulong v) { out(" "); out(k); out("="); outu(v); }

/* ------------------------------------------------------------------ counters -------------- */
static inline ulong cyc(void) { ulong v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v; }
static inline ulong ret(void) { ulong v; __asm__ volatile("csrr %0, minstret" : "=r"(v)); return v; }

/* ------------------------------------------------------------------ the fixture ----------- */
#define MAXN 256
#define MAXDELEG 16
static sublet_cap pool;                 /* the released region, linear between repetitions */
static sublet_cap H;                    /* the handle senior to the pool: the shared death */
static sublet_cap leaf[MAXN + 1];
static sublet_cap deleg[MAXDELEG];      /* unary delegations (depth series), one chain per branch */
static void *alias[MAXN + 1];           /* what the fixture's users hold */
static sublet_cap unrel;                /* the unrelated region */
static void *unrel_alias;
static ulong pool_base, pool_bytes, unrel_bytes;
/* the spatial arm: the same regions as plain pointers into one alias each */
static char *pool_ptr, *unrel_ptr;
static ulong minted_at_issue;           /* splits + mrevs when the fixture was issued */
static void *stale_alias;               /* one leaf alias kept past its withdrawal, for --stale */

static ulong minted(void) { return sublet_stats.split + sublet_stats.mrev; }

static ulong root_short;                /* set when the arena could not supply a carve: the run stops */
static void carve_root(ulong bytes, sublet_cap *to) {
  ulong base, end;
  if (root_short) return;
  if (sublet_type(&root) == SUBLET_TYPE_NONE) { root_short = bytes; return; }   /* the root was spent */
  base = sublet_base(&root); end = sublet_end(&root);
  if (end - base < bytes + 16UL) { root_short = bytes; return; }               /* keep the root non-empty */
  sublet_carve(&root, base + bytes, to);
}

/* Touch every 64th byte of a region through an alias, in order (protocol step 2). */
static ulong touch(volatile char *p, ulong bytes, unsigned char pat) {
  ulong i, sum = 0;
  for (i = 0; i < bytes; i += 64) { p[i] = (char)(pat + (unsigned char)(i >> 6)); sum += (unsigned char)p[i]; }
  return sum;
}
static ulong verify(volatile char *p, ulong bytes, unsigned char pat) {
  ulong i, bad = 0;
  for (i = 0; i < bytes; i += 64) if ((unsigned char)p[i] != (unsigned char)(pat + (unsigned char)(i >> 6))) bad++;
  return bad;
}

/* The timed withdrawal, sublet_give_to's instruction sequence with mcycle read at every phase
 * boundary. c1 before REVOKE (after the handle's load), c2 after it, c3 after the fill the
 * uninitialised region asks for (or at once when the region came back initialised), c4 after INIT.
 * fb = the bytes the fill wrote (end - cursor after the revoke), ini = whether INIT ran, ty = the
 * type the region came back as (3 = UNINIT is the fill path). Same selector encodings as sublet.h:
 * rs2 = x1 type, x2 cursor, x4 end. */
static inline void timed_give_to(sublet_cap *handle, sublet_cap *slot, ulong *c1, ulong *c2, ulong *c3,
                                 ulong *c4, ulong *fb, ulong *ini, ulong *ty) {
  ulong a, b, c, d, f, i, t;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%[h])\n"
                   "csrr %[a], mcycle\n"
                   ".insn r 0x5b, 0x1, 0x00, x0, t0, x0\n"
                   "csrr %[b], mcycle\n"
                   ".insn r 0x5b, 0x1, 0x04, %[t], t0, x1\n"
                   "addi t3, %[t], -3\n"
                   "bnez t3, 3f\n"
                   ".insn r 0x5b, 0x1, 0x04, t1, t0, x2\n"
                   ".insn r 0x5b, 0x1, 0x04, t2, t0, x4\n"
                   "sub %[f], t2, t1\n"
                   "1: addi t3, t1, 16\n"
                   "bgtu t3, t2, 2f\n"
                   ".insn s 0x5b, 0x4, x0, 0(t0)\n"
                   "mv t1, t3\n"
                   "j 1b\n"
                   "2: csrr %[c], mcycle\n"
                   ".insn r 0x5b, 0x1, 0x09, t0, t0, x0\n"
                   "li %[i], 1\n"
                   "j 4f\n"
                   "3: li %[i], 0\n"
                   "li %[f], 0\n"
                   "csrr %[c], mcycle\n"
                   "4: csrr %[d], mcycle\n"
                   ".insn s 0x5b, 0x4, x0, 0(%[h])\n"
                   ".insn s 0x5b, 0x4, t0, 0(%[s])\n"
                   : [a] "=&r"(a), [b] "=&r"(b), [c] "=&r"(c), [d] "=&r"(d), [f] "=&r"(f), [i] "=&r"(i), [t] "=&r"(t)
                   : [h] "r"(handle), [s] "r"(slot)
                   : "t0", "t1", "t2", "t3", "memory");
  sublet_stats.revoke++;
  sublet_stats.init += i;
  *c1 = a; *c2 = b; *c3 = c; *c4 = d; *fb = f; *ini = i; *ty = t;
}

/* ------------------------------------------------------------------ one point ------------- */
struct point {
  const char *series, *pattern; int arm_sublet;
  unsigned n;            /* leaves partitioning the pool (individual: 16 unrelated objects + 1) */
  ulong B;               /* pool bytes (individual: 16*64 + S) */
  ulong U;               /* unrelated region bytes */
  ulong S;               /* the individually released object's bytes */
  unsigned chain;        /* longest unary-delegation chain (depth series) */
  unsigned ndeleg;       /* delegations in total (15 in the depth series) */
  unsigned touch_unrel;
};

/* Issue the fixture from a linear pool: the shared handle, the partition, the delegations, the
 * takes. Returns the nodes minted for it. */
static ulong issue(const struct point *pt) {
  ulong m0 = minted();
  unsigned i, k, d = 0;
  ulong sz = pt->B / pt->n;
  if (pt->arm_sublet) {
    sublet_handle(&pool, &H);
    for (i = 0; i + 1 < pt->n; i++) sublet_carve(&pool, pool_base + (ulong)(i + 1) * sz, &leaf[i]);
    sublet_move(&pool, &leaf[pt->n - 1]);
    if (pt->ndeleg) {
      /* chains of `chain` delegations on as many branches as 15 allow, the remainder on one more */
      unsigned left = pt->ndeleg;
      for (i = 0; i < pt->n && left; i++) {
        unsigned c = left < pt->chain ? left : pt->chain;
        for (k = 0; k < c; k++) { sublet_handle(&leaf[i], &deleg[d]); d++; left--; }
      }
    }
    for (i = 0; i < pt->n; i++) alias[i] = sublet_take(&leaf[i]);
  } else {
    for (i = 0; i < pt->n; i++) alias[i] = pool_ptr + (ulong)i * sz;
  }
  minted_at_issue = m0;
  return minted() - m0;
}

/* The individual pattern's fixture: 16 objects of 64 bytes and one of S, all taken. */
static ulong issue_objects(const struct point *pt) {
  ulong m0 = minted(); unsigned i;
  if (pt->arm_sublet) {
    for (i = 0; i < 16; i++) sublet_carve(&pool, pool_base + (ulong)(i + 1) * 64UL, &leaf[i]);
    sublet_move(&pool, &leaf[16]);                       /* the S-byte object */
    for (i = 0; i < 17; i++) alias[i] = sublet_take(&leaf[i]);
  } else {
    for (i = 0; i < 17; i++) alias[i] = pool_ptr + (ulong)i * 64UL;
  }
  minted_at_issue = m0;
  return minted() - m0;
}

static ulong nodes_minted_total;
static unsigned objects_live;           /* the individual pattern's seventeen objects are issued */

static int run_point(const struct point *pt, unsigned rep, ulong budget) {
  ulong c0, c1, c2, c3, c4, c5, i0, i1, fb = 0, ini = 0, ty = 7, tdn = 0;
  ulong inner_free = 0, inner_reissue = 0, nd, bad = 0, t;
  unsigned i, ok = 1;
  volatile ulong *q;
  ulong est = pt->arm_sublet ? (ulong)(2 * pt->n + pt->ndeleg + 4) : 0;
  if (nodes_minted_total + est > budget) {
    out("R1 refused"); out(" s="); out(pt->series); kv("rep", rep); kv("need", est); kv("minted", nodes_minted_total); out("\n");
    return 0;
  }
  /* 1. construct: the pool is linear in its slot (fresh from the root, or back from the last
     withdrawal); issue leaves, fill payloads, keep the aliases. */
  if (streq(pt->pattern, "individual")) {
    /* the sixteen live objects and the released one are issued ONCE per point; every repetition
       releases object 16 and reissues it, so it is taken again when the next repetition starts */
    if (rep == 1) { nd = issue_objects(pt); objects_live = 1; }
    else nd = 1;                                          /* the reissued object's own take handle */
  } else nd = issue(pt);
  for (i = 0; i < (streq(pt->pattern, "individual") ? 17u : pt->n); i++)
    if (alias[i]) touch((volatile char *)alias[i], (streq(pt->pattern, "individual") ? (i == 16 ? pt->S : 64UL) : pt->B / pt->n), 0x11);
  /* 2. touch the unrelated region once, in order (or leave it reserved and untouched: the control) */
  if (pt->U && pt->touch_unrel) touch((volatile char *)unrel_alias, pt->U, 0x33);
  /* combined: free and reissue every even-indexed leaf before the timed ancestor withdrawal */
  if (streq(pt->pattern, "combined")) {
    for (i = 0; i < pt->n; i += 2) {
      if (pt->arm_sublet) {
        t = cyc(); alias[i] = 0; sublet_give(&leaf[i]); inner_free += cyc() - t;
        t = cyc(); alias[i] = sublet_take(&leaf[i]); inner_reissue += cyc() - t;
      } else {
        t = cyc(); alias[i] = 0; inner_free += cyc() - t;
        t = cyc(); alias[i] = pool_ptr + (ulong)i * (pt->B / pt->n); inner_reissue += cyc() - t;
      }
    }
  }
  /* 3. the timed release: bookkeeping | revoke | fill | init | reissue to a checked load+store */
  stale_alias = alias[pt->n > 1 ? 1 : 0];                /* kept for the companion probe */
  i0 = ret();
  c0 = cyc();
  if (streq(pt->pattern, "individual")) {
    alias[16] = 0;                                       /* bookkeeping: the user's pointer */
    if (pt->arm_sublet) {
      timed_give_to(&leaf[16], &leaf[16], &c1, &c2, &c3, &c4, &fb, &ini, &ty);
      alias[16] = sublet_take(&leaf[16]);                /* reissue the same object */
    } else {
      c1 = c2 = c3 = c4 = cyc();
      alias[16] = pool_ptr + 16UL * 64UL;
    }
    q = (volatile ulong *)alias[16];
  } else {
    for (i = 0; i < pt->n; i++) { alias[i] = 0; if (pt->arm_sublet) sublet_clear(&leaf[i]); }
    if (pt->arm_sublet) {
      for (i = 0; i < pt->ndeleg; i++) sublet_clear(&deleg[i]);
      timed_give_to(&H, &pool, &c1, &c2, &c3, &c4, &fb, &ini, &ty);
      alias[0] = sublet_take(&pool);                     /* reissue: the returned region, whole */
    } else {
      c1 = c2 = c3 = c4 = cyc();
      alias[0] = pool_ptr;
    }
    q = (volatile ulong *)alias[0];
  }
  *q = 0x5A5A5A5AUL;                                     /* the first use: a checked store and load */
  if (*q != 0x5A5A5A5AUL) ok = 0;
  c5 = cyc();
  i1 = ret();
  /* 4. survivors, and the drain that hands the region back linear for the next repetition
     (outside the bracket, reported as td=) */
  if (pt->U) bad = pt->touch_unrel ? verify((volatile char *)unrel_alias, pt->U, 0x33) : 0;
  t = cyc();
  if (pt->arm_sublet && !streq(pt->pattern, "individual")) { alias[0] = 0; sublet_give(&pool); }
  tdn = cyc() - t;                                        /* individual: the reissued object stays taken */
  if (pt->arm_sublet && !streq(pt->pattern, "individual") && sublet_type(&pool) != SUBLET_TYPE_LIN) ok = 0;
  nodes_minted_total += minted() - minted_at_issue;
  out("R1"); out(" s="); out(pt->series); out(" p="); out(pt->pattern); out(pt->arm_sublet ? " a=S" : " a=P");
  kv("n", pt->n); kv("B", pt->B); kv("U", pt->U); kv("S", pt->S); kv("chain", pt->chain); kv("rep", rep);
  kv("nd", nd); kv("bk", c1 - c0); kv("rv", c2 - c1); kv("fl", c3 - c2); kv("in", c4 - c3); kv("re", c5 - c4);
  kv("tt", c5 - c0); kv("ir", i1 - i0); kv("ty", ty); kv("ini", ini); kv("fb", fb); kv("inf", inner_free);
  kv("inr", inner_reissue); kv("td", tdn); kv("bad", bad); kv("ok", ok); out("\n");
  return 1;
}

/* ------------------------------------------------------------------ the series ------------ */
static const ulong KIB = 1024UL;

static void run_series(const char *series, const char *pattern, int arm_sublet, unsigned reps, unsigned touch_unrel, ulong budget) {
  struct point pt; unsigned r, k;
  static const unsigned n_pts[5] = {1, 4, 16, 64, 256};
  static const ulong b_pts[5] = {4 * 1024UL, 16 * 1024UL, 64 * 1024UL, 256 * 1024UL, 1024 * 1024UL};
  static const ulong u_pts[5] = {0, 64 * 1024UL, 256 * 1024UL, 1024 * 1024UL, 4096 * 1024UL};
  static const unsigned d_chain[4] = {1, 2, 4, 8};       /* 15 delegations as 15x1, 7x2+1, 3x4+3, 1x8+7 */
  static const ulong s_pts[4] = {16, 64, 256, 4096};
  unsigned npts = streq(series, "depth") || streq(series, "object") ? 4 : 5;
  for (k = 0; k < npts; k++) {
    memset(&pt, 0, sizeof pt);
    pt.series = series; pt.pattern = pattern; pt.arm_sublet = arm_sublet; pt.touch_unrel = touch_unrel;
    pt.n = 16; pt.B = 256 * KIB; pt.U = 256 * KIB;
    if (streq(series, "nodes")) pt.n = n_pts[k];
    else if (streq(series, "bytes")) pt.B = b_pts[k];
    else if (streq(series, "heap")) pt.U = u_pts[k];
    else if (streq(series, "depth")) { pt.chain = d_chain[k]; pt.ndeleg = 15; }
    else if (streq(series, "object")) { pt.S = s_pts[k]; pt.n = 17; pt.B = 16 * 64UL + pt.S; pt.U = 0; pt.pattern = "individual"; }
    else { out("R1 unknown series\n"); return; }
    /* fresh regions for the point: the pool and the unrelated region, carved from the root */
    if (arm_sublet) {
      carve_root(pt.B, &pool); pool_base = sublet_base(&pool);
      if (pt.U) { carve_root(pt.U, &unrel); unrel_alias = sublet_take(&unrel); }
    } else {
      carve_root(pt.B, &pool); pool_base = sublet_base(&pool); pool_ptr = sublet_take(&pool);
      if (pt.U) { carve_root(pt.U, &unrel); unrel_alias = sublet_take(&unrel); }
    }
    pool_bytes = pt.B; unrel_bytes = pt.U;
    if (root_short) {                                    /* a reading, not a fault: the arena ran out */
      out("R1 refused: arena"); out(" s="); out(pt.series); kv("n", pt.n); kv("B", pt.B); kv("U", pt.U); kv("need", root_short);
      kv("left", sublet_type(&root) == SUBLET_TYPE_NONE ? 0 : sublet_end(&root) - sublet_base(&root)); out("\n");
      break;
    }
    out("R1 plan"); out(" s="); out(pt.series); out(" p="); out(pt.pattern); out(arm_sublet ? " a=S" : " a=P");
    kv("n", pt.n); kv("B", pt.B); kv("U", pt.U); kv("S", pt.S); kv("chain", pt.chain); kv("reps", reps);
    kv("minted", nodes_minted_total); out("\n");
    for (r = 1; r <= reps; r++) if (!run_point(&pt, r, budget)) break;
    /* the individual pattern leaves its 17 objects issued: hand them back so the next point's
       fixture starts from a clean table (their region is not reused) */
    if (arm_sublet && streq(pt.pattern, "individual") && objects_live) { unsigned i; for (i = 0; i < 17; i++) { alias[i] = 0; sublet_give(&leaf[i]); } objects_live = 0; }
  }
}

/* ------------------------------------------------------------------ latency (E4/H1) ------- */
/* A dependent-load chase over a B-byte region taken as one plain alias: every 64-byte line holds the
   index of the next line in a single random cycle (Sattolo), and the loop's load address depends on
   the previous load, so the cycles per load are the load-to-use latency at that working set — L1 for
   4 and 16 KiB (the D$ is 32 KiB), DRAM beyond. The chain is built in the region itself (no other
   memory), the first traversal warms, the second is timed. Plain 8-byte loads: the figure `tab:hardware`
   asks for and M2's calibration, not a capability-load figure. */
static ulong rng_state = 0x9E3779B97F4A7C15UL;
static ulong rng(void) { ulong x = rng_state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; return rng_state = x; }
static void run_latency(unsigned reps) {
  static const ulong b_pts[5] = {4 * 1024UL, 16 * 1024UL, 64 * 1024UL, 256 * 1024UL, 1024 * 1024UL};
  unsigned k, r;
  for (k = 0; k < 5; k++) {
    ulong B = b_pts[k], lines = B / 64UL, i, cur, steps, t0, t1, sum = 0;
    volatile ulong *m;
    carve_root(B, &pool); pool_base = sublet_base(&pool); pool_ptr = sublet_take(&pool);
    if (root_short) { out("R1 refused: arena"); out(" s=latency"); kv("B", B); kv("need", root_short); out("\n"); return; }
    m = (volatile ulong *)pool_ptr;
    for (i = 0; i < lines; i++) m[i * 8] = i;                       /* identity, one index per line */
    for (i = lines - 1; i >= 1; i--) {                                /* Sattolo: one cycle through every line */
      ulong j = rng() % i, t = m[i * 8]; m[i * 8] = m[j * 8]; m[j * 8] = t;
    }
    steps = lines * 4UL; if (steps < 65536UL) steps = 65536UL;        /* every line at least four times */
    for (r = 0; r <= reps; r++) {                                     /* r = 0 warms; r >= 1 is timed */
      cur = 0; t0 = cyc();
      for (i = 0; i < steps; i++) cur = m[cur * 8];
      t1 = cyc(); sum += cur;
      if (r) { out("R1 lat"); kv("B", B); kv("lines", lines); kv("loads", steps); kv("rep", r); kv("cyc", t1 - t0); kv("per", (t1 - t0) / steps); kv("sink", sum & 0xFF); out("\n"); }
    }
  }
}

/* ------------------------------------------------------------------ linear (R-21 / R-22) ----- */
/* Does this hardware clear the LINEAR source of a capability move? The spec says cincoffset/scc/... and
   stc write cnull to a linear source (they are "MOVC rd, rs1" plus a step); registry R-21/R-22 say the RTL
   does not, from a 12-second simulation repro (linear-clear-audit.S), never read on silicon. Every arm
   returns a type read (7 = not a capability, i.e. cleared; 0 = LINEAR; 1 = NONLIN) and none can fault:
   arm 0 MOVC of a LINEAR source (rd != rs)   -- the INSTRUMENT control: must read 7, else no arm below can see a clear
   arm 1 CINCOFFSET, NONLIN source            -- the CONFORMANCE control: must read 1 (a copyable source survives)
   arm 2 CINCOFFSET, LINEAR source (rd != rs) -- spec 7; R-21 says 0
   arm 3 SCC, LINEAR source (rd != rs)        -- spec 7; R-21 says 0
   arm 4 LDC of a LINEAR capability           -- the memory slot after the load: spec 7 (moved out); R-21-class 0
   arm 5 LDC of a NONLIN capability           -- control: the slot must still read 1
   arm 6 STC of a LINEAR register             -- the register after the store: spec 7; R-22 says 0
   arm 7 STC of a NONLIN register             -- control: must read 1
   Operands come from the arena: fresh linear carves, and NONLIN aliases taken from them. */
static ulong lcc_type_after_op(int op, sublet_cap *src, sublet_cap *dst, ulong arg) {
  ulong ty = 99;
  switch (op) {
  case 0: __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n .insn r 0x5b, 0x1, 0x0a, t1, t0, x0\n .insn r 0x5b, 0x1, 0x04, %0, t0, x1\n .insn s 0x5b, 0x4, t1, 0(%2)\n" : "=&r"(ty) : "r"(src), "r"(dst) : "t0", "t1", "memory"); break;        /* movc t1, t0 */
  case 1: case 2: __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n .insn r 0x5b, 0x1, 0x0c, t1, t0, %3\n .insn r 0x5b, 0x1, 0x04, %0, t0, x1\n .insn s 0x5b, 0x4, t1, 0(%2)\n" : "=&r"(ty) : "r"(src), "r"(dst), "r"(arg) : "t0", "t1", "memory"); break;   /* cincoffset t1, t0, arg */
  case 3: __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n .insn r 0x5b, 0x1, 0x05, t1, t0, %3\n .insn r 0x5b, 0x1, 0x04, %0, t0, x1\n .insn s 0x5b, 0x4, t1, 0(%2)\n" : "=&r"(ty) : "r"(src), "r"(dst), "r"(arg) : "t0", "t1", "memory"); break;           /* scc t1, t0, arg */
  case 4: case 5: __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n .insn s 0x5b, 0x4, t0, 0(%2)\n .insn i 0x5b, 0x3, t1, 0(%1)\n .insn r 0x5b, 0x1, 0x04, %0, t1, x1\n" : "=&r"(ty) : "r"(src), "r"(dst) : "t0", "t1", "memory"); break;                          /* ldc t0 <- src; park t0 in dst; ldc t1 <- src again: the slot's type after the first load */
  case 6: case 7: __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n .insn s 0x5b, 0x4, t0, 0(%2)\n .insn r 0x5b, 0x1, 0x04, %0, t0, x1\n" : "=&r"(ty) : "r"(src), "r"(dst) : "t0", "memory"); break;                                                              /* stc t0 -> dst; the register's type after the store */
  }
  return ty;
}
static void run_linear(unsigned reps) {
  static const char *what[8] = {"movc-LIN(instrument-control)", "cincoffset-NONLIN(conformance-control)", "cincoffset-LIN", "scc-LIN", "ldc-LIN(slot-after)", "ldc-NONLIN(slot-after,control)", "stc-LIN(reg-after)", "stc-NONLIN(reg-after,control)"};
  static const ulong expect_spec[8] = {7, 1, 7, 7, 7, 1, 7, 1};
  static const ulong expect_r21r22[8] = {7, 1, 0, 0, 0, 1, 0, 1};
  unsigned r, arm;
  for (r = 1; r <= reps; r++) {
    for (arm = 0; arm < 8; arm++) {
      sublet_cap src, dst, park; ulong ty, before;
      sublet_clear(&src); sublet_clear(&dst); sublet_clear(&park);
      carve_root(4096, &src);                                   /* a fresh LINEAR region in src */
      if (root_short) { out("R1 refused: arena s=linear"); kv("arm", arm); kv("need", root_short); out("\n"); return; }
      if (arm == 1 || arm == 5 || arm == 7) {                    /* the NONLIN controls: an alias of it in src */
        void *a = sublet_take(&src); sublet_move(&src, &park); sublet_store(&src, a);
      }
      before = sublet_type(&src);
      ty = lcc_type_after_op(arm, &src, &dst, 64);
      out("R1 lin"); kv("arm", arm); out(" what="); out(what[arm]); kv("rep", r); kv("before", before); kv("type", ty);
      kv("spec", expect_spec[arm]); kv("r21r22", expect_r21r22[arm]); out(ty == expect_spec[arm] ? " reads=spec" : (ty == expect_r21r22[arm] ? " reads=R21R22" : " reads=OTHER")); out("\n");
    }
  }
}

/* ------------------------------------------------------------------ chase (M2) --------------- */
/* M2's access path: N 64-byte records, each holding the INDEX of the next (never a capability) in a
   single random cycle (Sattolo, seeded); the capabilities live in a separately counted lookup array.
   One access = read the record's index (a plain ld through the record's capability), then fetch the next
   record's capability from the lookup array (an ldc: the DYN unit's node-validity query, at every
   privilege level), then dereference it. Arms: P (custom-spatial) = every lookup entry is one alias of the
   whole region offset to its record, so every access touches ONE node; S (custom-sublet) = every record
   is its own object with its own alias, so an access touches ITS node — N touched nodes, the node table's
   footprint growing with the working set; D (data-only) = the same chase by integer arithmetic on one
   base, no lookup ldc at all (the labelled non-protecting ablation). Two warm-up traversals, then the
   timed accesses; the cold first traversal is reported apart; checksum and visit count verified. */
/* ------------------------------------------------------------------ M1: churn (prepared) ------ */
/* Safe node reclamation under lifetime churn (experiments/M1-node-reclamation.md): sixteen live
 * 64-byte objects replaced in round-robin order, each replacement one release (revoke + init) and one
 * take (one node minted), to 10C cumulative allocations where C is the usable node capacity passed as
 * --cap. Arms (--arm): drop = clear every obsolete reference; ring = retain the 16 most recent obsolete
 * aliases in tagged memory, replacing the oldest; pressure = retain every obsolete alias in a
 * preallocated buffer (M1_MAXRET entries, accounted; a full buffer stops the run as "buffer", not as
 * exhaustion); release = the pressure arm, then every retained reference cleared, then 2C more
 * allocations. A snapshot every C/16 allocations prints the cumulative counters and non-faulting type
 * reads of the oldest retained alias and of a live slot. On the deployed (non-reclaiming) table the run
 * stops at --budget with stop=budget: that is the exhaustion classification, not a failure of the arm.
 * What the harness cannot read and a reclaiming build must expose: occupancy and free-node counters,
 * the reclamation count, and a node id/generation read for the identifier-turnover witness. The
 * faulting probe (--stale-take: a capability operation through the oldest retained alias, which must
 * fault INVALID_CAPABILITY and, in a domain, wedges -- M-1) runs LAST and only when asked. Subordinate
 * handles retained across a release are not modelled in this version. */
/* M1_LIVE is the live-object count AND the fixture's geometry: the carve loop below takes one 64-byte
 * leaf per live object, so -DM1_LIVE=4 and =64 carve a 256-byte and a 4 KiB pool respectively. A
 * geometry that changes with this knob is the intent, not a defect; a reader comparing arenas across
 * arms must compare M1_LIVE first. Overridable because the release walk's per-node conversion assumes
 * each revoke walks alloc/M1_LIVE dead nodes, and sweeping this knob is the only way to test that
 * assumption: if the measured slope scales as 1/M1_LIVE the conversion holds, and if it does not the
 * per-node figure is wrong by exactly that factor (apollo, 2026-09-16). */
#ifndef M1_LIVE
#define M1_LIVE 16
#endif
/* M1_LEAF is the bytes per live object. It exists because M1_LIVE alone cannot separate slot count from
 * memory: the pool is M1_LIVE * M1_LEAF, so sweeping M1_LIVE moves both at once. What it is FOR changed
 * once the existing captures were re-read per snapshot instead of per invocation (apollo, 2026-09-17).
 * Measured there, across M1_LIVE 2/4/8/16/64: take_cyc/n is a function of ALLOC and not of the preceding
 * revoke's walk length -- at a walk of 254 dead nodes it reads 67 cycles at M1_LIVE=2 and 130 at 64, while
 * at alloc 512 every geometry reads 67-68. The floor is 66.7-67.5 everywhere and the rise begins at alloc
 * 1792-2048, which is 28-32 KiB of 16-byte node table against a 32 KiB D-cache; only the post-knee plateau
 * (218/163/137/131/130) depends on M1_LIVE. So the knee looks like node-table capacity, and M1_LEAF is the
 * probe for it: touch() writes one byte per 64, so the pool occupies M1_LIVE*M1_LEAF/64 lines of the same
 * cache, and if the two share it the knee must move earlier by one allocation per 16 bytes of pool. A knee
 * that does NOT move refutes the sharing. What this knob does not hold constant, stated because the sweep
 * cannot avoid it: per-allocation touch volume scales with M1_LEAF, so the probe adds data traffic as well
 * as data residency, and a knee that moves shows they share a cache without saying which of the two did it. */
#ifndef M1_LEAF
#define M1_LEAF 64
#endif
/* M1_TOUCH is the bytes of each leaf the workload actually writes, and it exists ONLY to break the tie
 * M1_LEAF cannot break on its own: sweeping M1_LEAF scales the pool's RESIDENT lines and the per-allocation
 * touch VOLUME by the same factor, so a knee that moves is consistent with either (the RTL lane, 2026-09-17).
 * Setting M1_TOUCH below M1_LEAF carves a large pool and leaves all but M1_TOUCH bytes of each leaf never
 * written, hence never resident: the arm with a 24 KiB carve and a 64-byte touch has the address spread of
 * the largest arm and the cache residency of the smallest. If the knee tracks residency it must sit with the
 * smallest arm; if it tracks the carve it must sit with the largest. Defaults to M1_LEAF so every build that
 * does not set it is byte-identical to one from before this knob existed. */
#ifndef M1_TOUCH
#define M1_TOUCH M1_LEAF
#endif
/* The retained-reference buffer is an INSTRUMENT limit, not a property of the system, and a measurement
 * must not be bounded by its own instrument. At C = 256 the run's target is 10*C = 2560, so a 2048-entry
 * buffer stopped the pressure and release arms at stop=buffer before either reached its target and left
 * the release arm's phase 2 (gated on alloc >= target) unreached, so three of the four patterns were
 * really two (apollo, 2026-09-15). 4096 covers 10*C at C = 256 with margin; a larger C needs a larger
 * buffer again, and the run says which it hit. */
#ifndef M1_MAXRET
#define M1_MAXRET 4096
#endif
/* Overridable as of M1's approved run (apollo, 2026-09-18). At the protocol's production capacity the
 * interpretable size is not "as large as fits" but ONE RETAINED REFERENCE PER DISTINCT INDEX EVER
 * CONSUMED -- 65,532, the measured pool -- because that is what makes the retain-pressure arm answer
 * the strongest form of its question: does holding a stale reference to every index ever used prevent
 * or corrupt reuse? At 4096 the arm stops at 0.63 % of a 10C run and measures this buffer rather than
 * the reclaimer. If the data budget will not take the full size, report the largest that does TOGETHER
 * WITH THE FRACTION OF DISTINCT INDICES IT COVERS, since that fraction is what makes the number
 * interpretable at all.
 *
 * What a null in that arm means, recorded before the run rather than after: retention CANNOT prevent
 * reuse in this design, because no old reference is consulted at reclaim time -- the reclaim event is
 * the walk finding a node valid, and safety is carried by the generation rather than by reference
 * accounting (the implementation owner's specification, approved 2026-09-18). So the arm tests that
 * the generation check holds under a large retained set. A CLEAN NULL IS THE EXPECTED AND CORRECT
 * RESULT and is only interesting if it fails. */
/* M1_RELEASE_AT_BUFFER moves the release arm's phase-2 trigger from "10C allocations" to "the retained
 * buffer is full", and exists because the arm is otherwise UNMEASURABLE at the protocol's production
 * capacity. Release retains one alias per allocation for the whole of phase 1 -- it takes the same
 * branch as the pressure arm until phase 2 fires -- and a `void *` here is a 16-byte capability, so
 * 10C at C = 65,532 needs 10,485,120 bytes of buffer against a ~2 MB domain region. Measured, not
 * assumed: on 2026-09-18 the arm stopped at `stop=buffer alloc=4097` with M1_MAXRET = 4096 while drop
 * and ring beside it ran to 655,320 in the same boot.
 *
 * The two settings answer different questions and the lead asked for both: 0 keeps the approved
 * algorithm and the arm is run at a smaller C with the reduction stated, while 1 keeps C at the
 * production capacity and changes when phase 2 begins. NEITHER SUBSTITUTES FOR THE OTHER, and a
 * transcript produced with 1 must never be reported as the protocol's release arm without saying so --
 * which is why it lowers `target`, so the START LINE states the trigger that was actually used.
 *
 * It is applied at SETUP and not inside the loop, so the default build is byte-identical rather than
 * merely intended to be. That is not a style preference: the first version of this change wrote the
 * extra term into the loop's phase-2 condition, where it folds away at compile time, and the rebuilt
 * default image was NOT byte-identical -- reordering the short-circuit moved the codegen (598dce77
 * against the board's 1b7a04fe). */
#ifndef M1_RELEASE_AT_BUFFER
#define M1_RELEASE_AT_BUFFER 0
#endif
static void *m1_ring_alias[M1_LIVE];
static void *m1_ret_alias[M1_MAXRET];
static sublet_cap m1_tmp;

static ulong m1_alias_type(void *a) {
  if (!a) return 99;
  sublet_store(&m1_tmp, a);
  return sublet_type(&m1_tmp);
}

/* per snapshot interval: the raw cycles spent in the takes (mrev: minting) and in the gives (revoke + fill +
 * init: release), summed over the interval's allocations, so take_cyc/n and give_cyc/n against cumulative
 * allocations are the two cost curves as the table fills. Each bracket carries the timer's own read-to-read
 * floor (2 cycles on silicon, E4); the analysis subtracts it, the harness prints raw sums. */
static void m1_snap(const char *arm, ulong alloc, ulong C, ulong m0, unsigned nret, void *oldest,
                    ulong n, ulong tk, ulong tg, ulong ini) {
  out("R1 m1 snap arm="); out(arm); kv("alloc", alloc); kv("C", C); kv("minted", minted() - m0);
  kv("revoked", sublet_stats.revoke); kv("init", sublet_stats.init); kv("live", M1_LIVE); kv("leaf", M1_LEAF); kv("retained", nret);
  kv("n", n); kv("take_cyc", tk); kv("give_cyc", tg); kv("init_n", ini);
  kv("stale_alias_type", m1_alias_type(oldest)); kv("live_slot_type", sublet_type(&leaf[0])); out("\n");
}

static void run_m1(const char *arm, ulong C, ulong budget, unsigned stale_take) {
  ulong target, alloc = 0, snap_every, next_snap, m0, phase2_end = 0, t, tk = 0, tg = 0, n = 0, ini0;
  unsigned i, k, nret = 0, releasing = 0;
  void *old, *oldest = 0;
  const char *stop = "target";
  if (C < 16) C = 16;
  target = 10UL * C; snap_every = (C + 15UL) / 16UL; next_snap = snap_every;
#if M1_RELEASE_AT_BUFFER
  /* phase 2 at buffer-full instead of at 10C: lower the target so the LOOP is untouched and the
     start line reports the trigger that was actually used */
  if (streq(arm, "release") && target > (ulong)M1_MAXRET - 1UL) target = (ulong)M1_MAXRET - 1UL;
#endif
  carve_root(M1_LIVE * (ulong)M1_LEAF, &pool);
  if (root_short) { out("R1 m1 refused: arena\n"); return; }
  pool_base = sublet_base(&pool);
  m0 = minted();
  for (i = 0; i + 1 < M1_LIVE; i++) sublet_carve(&pool, pool_base + (ulong)(i + 1) * (ulong)M1_LEAF, &leaf[i]);
  sublet_move(&pool, &leaf[M1_LIVE - 1]);
  for (i = 0; i < M1_LIVE; i++) { alias[i] = sublet_take(&leaf[i]); touch((volatile char *)alias[i], (ulong)M1_TOUCH, 0x11); }
  out("R1 m1 start arm="); out(arm); kv("C", C); kv("target", target); kv("fixture_nodes", minted() - m0);
  kv("budget", budget); kv("maxret", M1_MAXRET); kv("leaf", M1_LEAF); kv("touch", M1_TOUCH); kv("pool", M1_LIVE * (ulong)M1_LEAF); kv("resident", M1_LIVE * (ulong)M1_TOUCH); kv("snap_every", snap_every);
#if M1_RELEASE_AT_BUFFER
  kv("rel_at_buffer", 1);   /* the release arm's phase 2 fires at buffer-full, NOT at 10C */
#endif
  out("\n");
  i = 0; ini0 = sublet_stats.init;
  for (;;) {
    if (alloc >= target && !releasing) {
      if (streq(arm, "release")) {
        /* phase 2: clear every retained reference, then 2C more allocations */
        for (k = 0; k < nret; k++) m1_ret_alias[k] = 0;
        nret = 0; oldest = 0; releasing = 1; phase2_end = alloc + 2UL * C;
        out("R1 m1 released arm="); out(arm); kv("alloc", alloc); kv("minted", minted() - m0); out("\n");
      } else break;
    }
    if (releasing && alloc >= phase2_end) break;
    if (minted() + 1UL > budget) { stop = "budget"; break; }
    old = alias[i];
    t = cyc(); sublet_give(&leaf[i]); tg += cyc() - t;     /* the object's lifetime ends: revoke, fill, init */
    if (streq(arm, "drop") || releasing) { alias[i] = 0; }
    else if (streq(arm, "ring")) { k = (unsigned)(alloc % M1_LIVE); m1_ring_alias[k] = old; if (nret < M1_LIVE) nret++; oldest = m1_ring_alias[(unsigned)((alloc + 1UL) % M1_LIVE)]; if (!oldest) oldest = m1_ring_alias[0]; }
    else { if (nret >= M1_MAXRET) { alias[i] = sublet_take(&leaf[i]); alloc++; stop = "buffer"; break;   /* the instrument, not the table: nret == M1_MAXRET before alloc == target */ } m1_ret_alias[nret++] = old; oldest = m1_ret_alias[0]; }
    t = cyc(); alias[i] = sublet_take(&leaf[i]); tk += cyc() - t;   /* a new object in its place: one node minted */
    touch((volatile char *)alias[i], (ulong)M1_TOUCH, (unsigned char)alloc);
    alloc++; n++;
    if (alloc >= next_snap) {
      m1_snap(arm, alloc, C, m0, nret, oldest, n, tk, tg, sublet_stats.init - ini0);
      next_snap += snap_every; n = 0; tk = 0; tg = 0; ini0 = sublet_stats.init;
    }
    i = (i + 1) % M1_LIVE;
  }
  out("R1 m1 end arm="); out(arm); out(" stop="); out(stop); kv("alloc", alloc); kv("minted", minted() - m0);
  kv("revoked", sublet_stats.revoke); kv("retained", nret); kv("released", releasing); out("\n");
  nodes_minted_total += minted() - m0;
  if (stale_take && oldest) {
    /* LAST, and expected to fault: a capability operation through the oldest retained alias */
    out("R1 m1 stale-take go\n");
    sublet_store(&m1_tmp, oldest);
    old = sublet_take(&m1_tmp);
    out("R1 m1 stale-take returned"); kv("nonzero", old != 0); out("\n");
  }
}

static void run_chase(unsigned reps, ulong seed, const char *arm) {
  static const ulong n_pts[5] = {16, 64, 256, 1024, 4096};
  unsigned k, r;
  for (k = 0; k < 5; k++) {
    ulong N = n_pts[k], i, steps = 100000UL, cur, t0, t1, i0, i1, sum, m0, chk = 0, visits;
    sublet_cap region, lreg, sreg; void **lookup; sublet_cap *leafslot; char *base = 0;
    m0 = minted();
    carve_root(N * 64UL, &region); carve_root(N * 16UL + 64UL, &lreg); carve_root(N * 16UL + 64UL, &sreg);
    if (root_short) { out("R1 refused: arena s=chase"); kv("N", N); kv("need", root_short); out("\n"); return; }
    lookup = (void **)sublet_take(&lreg); leafslot = (sublet_cap *)sublet_take(&sreg);
    if (streq(arm, "S")) {                                   /* every record its own object: N leaves, N aliases */
      for (i = 0; i < N; i++) { sublet_carve(&region, sublet_base(&region) + 64UL, &leafslot[i]); lookup[i] = sublet_take(&leafslot[i]); }
    } else {                                                 /* one alias of the region, offset per record */
      base = (char *)sublet_take(&region);
      for (i = 0; i < N; i++) lookup[i] = base + i * 64UL;
    }
    for (i = 0; i < N; i++) *(volatile ulong *)lookup[i] = i;                 /* identity chain */
    rng_state = 0x9E3779B97F4A7C15UL ^ (seed * 0x2545F4914F6CDD1DUL);
    for (i = N - 1; i >= 1; i--) {                                             /* Sattolo: one cycle */
      ulong j = rng() % i, a = *(volatile ulong *)lookup[i], b = *(volatile ulong *)lookup[j];
      *(volatile ulong *)lookup[i] = b; *(volatile ulong *)lookup[j] = a;
    }
    /* the cold first traversal, timed and reported apart; then two warm-ups; then the timed accesses */
    for (r = 0; r < 3; r++) {
      cur = 0; visits = 0; t0 = cyc();
      if (streq(arm, "D")) { for (i = 0; i < N; i++) { cur = *(volatile ulong *)(base + cur * 64UL); visits++; } }
      else                 { for (i = 0; i < N; i++) { cur = *(volatile ulong *)lookup[cur]; visits++; } }
      t1 = cyc();
      if (r == 0) { out("R1 chase-cold arm="); out(arm); kv("N", N); kv("seed", seed); kv("cyc", t1 - t0); kv("per", (t1 - t0) / N); kv("cycle_ok", cur == 0 && visits == N); out("\n"); }
    }
    for (r = 1; r <= reps; r++) {
      cur = 0; sum = 0; t0 = cyc(); i0 = ret();
      if (streq(arm, "D")) { for (i = 0; i < steps; i++) { cur = *(volatile ulong *)(base + cur * 64UL); sum += cur; } }
      else                 { for (i = 0; i < steps; i++) { cur = *(volatile ulong *)lookup[cur]; sum += cur; } }
      t1 = cyc(); i1 = ret();
      if (r == 1) chk = sum;
      out("R1 chase arm="); out(arm); kv("N", N); kv("seed", seed); kv("rep", r); kv("loads", steps); kv("cyc", t1 - t0); kv("per100", (t1 - t0) * 100UL / steps);
      kv("instret", i1 - i0); kv("touched", streq(arm, "S") ? N : (streq(arm, "P") ? 1UL : 0UL)); kv("minted", minted() - m0); kv("chk", sum); kv("chk_ok", sum == chk); out("\n");
    }
  }
}

/* ------------------------------------------------------------------ entry ----------------- */
void domain_main(unsigned *res, unsigned func) {
  static char args[256];
  ulong len, i;
  const char *arm = "S", *series = "nodes", *pattern = "shared";
  unsigned reps = 5, touch_unrel = 1, calib = 0, stale = 0, stale_take = 0; ulong cap = 65532UL;
  ulong seed = 1;
  ulong budget = 50000UL;
  int arm_sublet;
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (nshare == 0) meta = (volatile struct sqlite_hostcall_v0 *)res;
    else if (nshare == 1) payload = (volatile char *)res;
    else if (nshare == 2) sublet_store(&root, (void *)res);     /* the arena, linear, into its slot */
    nshare++;
    return;
  }
  if (!meta || !payload) { *res = 0x4EB1FFFCu; return; }
  /* the host declares its region size; a sane declaration (a power of two, 4 KiB .. 1 MiB) sizes the
     sink and locates the args at its half, as sqlite_hostcall.h derives SQLITE_HC_SPEED_ARGS_OFF */
  {
    ulong region = (ulong)meta->result;
    if (region < 4096UL || region > (1UL << 20) || (region & (region - 1))) { *res = 0x4EB1FFFDu; return; }
    out_limit = region - 64UL;
    len = (ulong)meta->offset;
    for (i = 0; i < len && i + 1 < sizeof args; i++) args[i] = payload[region / 2UL + i];
    args[i] = 0;
    out_used = 0; meta->length = 0;
    out("R1 entry"); kv("shares", nshare); kv("region", region); kv("opcode", (ulong)meta->opcode);
    kv("offset", len); kv("phase", (ulong)meta->phase); out("\n");
    if ((ulong)meta->opcode != SQLITE_HC_OP_SPEED) { out("R1 refused: opcode\n"); *res = 0x4EB1FFFBu; return; }
    if (len == 0 || len + 1 >= sizeof args) { out("R1 refused: args length\n"); *res = 0x4EB1FFFAu; return; }
  }
  /* split the args in place */
  {
    char *p = args; char *tok[32]; unsigned nt = 0;
    while (*p && nt < 32) {
      while (*p == ' ') p++;
      if (!*p) break;
      tok[nt++] = p;
      while (*p && *p != ' ') p++;
      if (*p) *p++ = 0;
    }
    for (i = 0; i < nt; i++) {
      if (streq(tok[i], "--arm") && i + 1 < nt) arm = tok[++i];
      else if (streq(tok[i], "--series") && i + 1 < nt) series = tok[++i];
      else if (streq(tok[i], "--seed") && i + 1 < nt) seed = atou(tok[++i]);
      else if (streq(tok[i], "--pattern") && i + 1 < nt) pattern = tok[++i];
      else if (streq(tok[i], "--reps") && i + 1 < nt) reps = (unsigned)atou(tok[++i]);
      else if (streq(tok[i], "--touch") && i + 1 < nt) touch_unrel = (unsigned)atou(tok[++i]);
      else if (streq(tok[i], "--budget") && i + 1 < nt) budget = atou(tok[++i]);
      else if (streq(tok[i], "--calib")) calib = 1;
      else if (streq(tok[i], "--stale")) stale = 1;
      else if (streq(tok[i], "--cap") && i + 1 < nt) cap = atou(tok[++i]);
      else if (streq(tok[i], "--stale-take")) stale_take = 1;
    }
  }
  arm_sublet = streq(arm, "S");
  if (reps > 20) reps = 20;
  out("R1 start arm="); out(arm); out(" series="); out(series); out(" pattern="); out(pattern);
  kv("reps", reps); kv("touch", touch_unrel); kv("budget", budget);
  if (sublet_type(&root) == SUBLET_TYPE_NONE) { out(" NO-ARENA\n"); *res = 0x4EB1FFFEu; return; }
  kv("arena", sublet_end(&root) - sublet_base(&root)); kv("root_type", sublet_type(&root)); out("\n");
  if (calib) {
    ulong a = cyc(), b = cyc(), c = cyc(), d = ret(), e = ret();
    out("R1 calib"); kv("cyc_cyc", b - a); kv("cyc_cyc2", c - b); kv("ret_ret", e - d); out("\n");
    *res = 0x4EB10000u; return;
  }
  if (streq(series, "latency")) run_latency(reps); else
  if (streq(series, "linear")) run_linear(reps); else
  if (streq(series, "chase")) run_chase(reps, seed, arm); else
  if (streq(series, "m1")) run_m1(arm, cap, budget, stale_take); else
  run_series(series, pattern, arm_sublet, reps, touch_unrel, budget);
  out("R1 end"); kv("minted", nodes_minted_total); kv("split", sublet_stats.split); kv("mrev", sublet_stats.mrev);
  kv("revoke", sublet_stats.revoke); kv("init", sublet_stats.init); kv("out", out_used); kv("lines", out_lines + 1); out("\n");
  if (stale && arm_sublet && stale_alias) {
    /* the companion probe, LAST: a load through a leaf alias whose ancestor was withdrawn. The
       emulator faults here (the alias sat in a C variable, reloaded through a slot the same way
       the S1 probes' pointers are); E1 measured that the RTL's LSU lets such a load retire. */
    volatile unsigned char *p = (volatile unsigned char *)stale_alias;
    out("R1 stale"); kv("byte", *p); out("\n");
  }
  *res = 0x4EB10000u | (unsigned)(nodes_minted_total & 0xFFFFu);
}
