// rcm-lb-dc.c -- the nested lane's copy of the rtl lane's rcm-lb.c (llvm-capstone dev 6d87efe8d50a), 2026-11-26, for the
// D$-residency pass on format v3 traces. Changes, all marked "v3:":
//   - word 1 bits 40..43 (D$ residency of the record's line before the probe, per LFSR seed) and bit 44 (the tracer's
//     online 64x4 LB missed) are read and stripped before every v2 check; a v2 trace reads exactly as before;
//   - a second LB, LB0, fixed at b42c8c644's 64x4, replays beside --lb's; on a v3 trace each check run start compares
//     LB0's miss with bit 44 (v3 mismatches must be 0);
//   - at each check run start where LB0 misses: tallies, per seed, whether the line was resident, over all LB0 misses,
//     over the misses --lb's LB does not take (removed_lb), and over those RC does not take (removed_rc);
//   - --walk dedup is refused with a v3 trace (the tracer's LB is asis).
// The original header follows.
//
// rcm-lb.c -- the record-cache model for Line B (the Sublet format, B5b-2), 2026-11-26, rtl lane.
//
// The Line B port of rcm.c (same directory). It replays a Sublet record-access trace through four organisations in one
// pass and prints EVENT counts only: the semantics owner's rule (llvm-capstone capstone/docs/plans/
// 2026-11-26-spatial-only-lineb.md's sibling, the record-cache decision) takes the time side from a MEASURED per-miss
// cost, not from a fitted cycle model (rcm.c's two-parameter cycle model was refuted out of sample, 1c7617108984).
//
//   LB  b42c8c644's verdict cache (capstone-ariane b5b2-prefix, load_store_unit.sv :302-305, :1331-1484): 64 sets x 4
//       ways, set = slot[5:0], tag = the FULL id {generation, slot}. Installs: every live record write (fill,
//       nodeval 1) and every live record read the unit makes (the probe's and the operations', rdfill). Placement: a
//       way holding the same INDEX (slot) is reused, else the lowest free way, else the set's round-robin victim, the
//       pointer advancing only on a displacement. A release write (nodeval 0) clears liveness in place and KEEPS the
//       entry, so a later check on that id hits dead (a deny without a probe). A check miss is a probe: one record read
//       (memory), which installs, then the access replays. A namespace edge empties it.
//   RC  the paper's 5.2 record cache, as rcm.c: 8 KiB, 2 ways, 64 sets, 64-byte lines of 4 records, LRU, write-back,
//       shared by checks and operations.
//   GC  the generation-only cache, as rcm.c: 512 entries, 4 ways, 128 sets, set = slot[6:0], LRU, checks only; a record
//       write updates a present entry in place.
//   NC  no cache: every record access is a memory access.
// Each has its own 16-entry fully associative LRU record TLB over 256-record pages, as rcm.c.
//
// TRACE FORMAT v2 (the Sublet tracer's, to be produced by the twin under x-capstone-ids=63): TWO little-endian u64 per
// event.
//   word 0: code[63:56] = kind << 2 | op, count[55:32], slot[31:0]   (rcm.c's word, kinds as rcm.c: c k p a r v o T I PI E)
//   word 1: live[32], generation[31:0]                               (the id's generation; live = the record, as read or
//                                                                    as written, is live for exactly that id)
//   op: 0 record read, 1 record write, 2/3 header read/write (not records). kind c = an access verdict (a check), a =
//   allocation (CDERIVE, CRENEW, CSMINT's create), r / v = release (CREVOKE: the 4(b) walk's reads and its writes), o =
//   any other operation's record access. T = a table switch, E = a namespace edge (srevroot change, context entry or
//   exit), I / PI = instruction counters (word 0 carries the value; word 1 is 0).
//   E: op 0 = trap out of protected U, 1 = srevroot write, 2 / 3 = SRET / MRET into protected U (the nested lane's
//   v2 tracer); word 1 bit 0 = 1 exactly when the RTL's namespace key changed, so LB flushes ONLY then. The key is
//   {vctx_table, srevroot}: b42c8c644's revnode_ns_flush_o (ex_stage.sv:1891) fires on a change of either (or the
//   csupctl bit-2 strobe), and under CAPSTONE_SUBLET_FMT vctx_table is 1 in S and U whenever srevroot != 0
//   (csr_regfile.sv:3008-3011), so a U <-> S switch under one srevroot does not flush. Every E is counted by op.
// Runs: a check run of N is N checks of one id; an operation run is ONE access, as rcm.c.
// Exit 2 on an empty, malformed or unreadable trace: no data is an error.
//
// --walk asis|dedup (default asis): how each op stretch is replayed. asis = the trace's own sequence. dedup = each slot
// ONCE per stretch, at the position of its first live access (or its first touch if it has none), carrying its final
// state: an install at the generation of its last live access, then an invalidate if a dead write follows that access.
// Both keep the twin's first-install order; they differ only where a stretch displaces a slot it installed and a later
// re-read re-installs it (self_evict), which is where the twin's re-reads (5n + 12 per CREVOKE) and the unit's
// (3n + 6) could disagree. The two LB check rows are a SENSITIVITY BRACKET over that, not a bound over all orders.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

enum { K_C, K_K, K_P, K_A, K_R, K_V, K_O, K_T, K_I, K_PI, K_E };
enum { CL_CHECK, CL_OP, NCL };
typedef struct { uint64_t acc, hit, miss, mrd, mwr, tlbmiss, wb; } st_t;

typedef struct { uint64_t page[16]; uint64_t age[16]; int n; uint64_t clock; } tlb_t;
static int tlb_access(tlb_t *t, uint64_t page) {
  t->clock++;
  for (int i = 0; i < t->n; i++) if (t->page[i] == page) { t->age[i] = t->clock; return 1; }
  int v;
  if (t->n < 16) v = t->n++;
  else { v = 0; for (int i = 1; i < 16; i++) if (t->age[i] < t->age[v]) v = i; }
  t->page[v] = page; t->age[v] = t->clock; return 0;
}
static void tlb_clear(tlb_t *t) { t->n = 0; }

// ---------------- LB: b42c8c644's verdict cache (64 x 4); --lb SETSxWAYS models another size of it, with the same
// placement rules: set = slot[log2(SETS)-1:0], same-index reuse, then the lowest free way, then the round-robin victim.
#define LB_MAXS 4096
#define LB_MAXW 32
static unsigned lb_sets = 64, lb_ways = 4;
typedef struct {
  unsigned sets, ways;   // v3: each LB carries its geometry
  uint64_t slot[LB_MAXS][LB_MAXW]; uint32_t gen[LB_MAXS][LB_MAXW]; uint8_t occ[LB_MAXS][LB_MAXW], live[LB_MAXS][LB_MAXW], rr[LB_MAXS];
  tlb_t tlb; st_t s[NCL];
  uint64_t probes, deny_hit_dead, dead_record, installs, inval, edges_flushed;
  uint64_t stretches, self_evict, untracked, se_kind[16];
} lb_t;
// An op STRETCH is a maximal run of operation events (no check, edge or table switch between them). A re-access of a
// slot LB already holds reuses its way and displaces nothing, so LB's state after a stretch depends only on the order
// of each slot's first install and on each slot's last (generation, live) -- NOT on how many times the operation
// reads or writes it -- EXCEPT when the stretch displaces a slot it installed itself. self_evict counts those. When
// it is 0, LB's check counts are the same for any access pattern with the same first-install order and final states
// (the twin's model walks differently from b42c8c644's unit). A real trace never reads 0: a stretch that installs
// more than 4 slots of one set displaces its own on the RTL too. So the LB rows are read under both --walk modes.
#define SH 16384
static uint64_t sh_key[SH]; static uint64_t sh_ep[SH]; static uint64_t sh_epoch = 1, sh_n; static int in_op, cur_kind;
static int sh_has(uint64_t s) { for (uint64_t h = (s * 0x9e3779b97f4a7c15ULL) >> 50, k = 0; k < SH; k++, h = (h + 1) & (SH - 1)) {
    if (sh_ep[h] != sh_epoch) return 0;
    if (sh_key[h] == s) return 1;
  }
  return 0; }
static void sh_add(uint64_t s, lb_t *c) { if (sh_n >= SH / 2) { c->untracked++; return; }
  for (uint64_t h = (s * 0x9e3779b97f4a7c15ULL) >> 50, k = 0; k < SH; k++, h = (h + 1) & (SH - 1)) {
    if (sh_ep[h] != sh_epoch) { sh_ep[h] = sh_epoch; sh_key[h] = s; sh_n++; return; }
    if (sh_key[h] == s) return;
  } }
static void sh_end(void) { if (in_op) { sh_epoch++; sh_n = 0; in_op = 0; } }
static void lb_install(lb_t *c, uint64_t slot, uint32_t gen) {
  unsigned set = slot & (c->sets - 1); int way = c->rr[set], reuse = 0;
  for (int w = 0; w < (int)c->ways; w++) if (c->occ[set][w] && c->slot[set][w] == slot) { way = w; reuse = 1; }
  if (!reuse) for (int w = c->ways - 1; w >= 0; w--) if (!c->occ[set][w]) { way = w; reuse = 1; }
  if (!reuse) { if (in_op && sh_has(c->slot[set][way])) { c->self_evict++; c->se_kind[cur_kind]++; } c->rr[set] = (c->rr[set] + 1) % c->ways; }
  if (in_op) sh_add(slot, c);
  c->occ[set][way] = 1; c->slot[set][way] = slot; c->gen[set][way] = gen; c->live[set][way] = 1; c->installs++;
}
static void lb_invalidate(lb_t *c, uint64_t slot) {
  unsigned set = slot & (c->sets - 1);
  for (int w = 0; w < (int)c->ways; w++) if (c->occ[set][w] && c->slot[set][w] == slot && c->live[set][w]) { c->live[set][w] = 0; c->inval++; }
}
// a check: hit live (allow), hit dead (deny, no probe), or a miss (a probe: one record read, an install if live)
static int lb_check(lb_t *c, uint64_t slot, uint32_t gen, int rec_live, uint64_t n) {   // v3: returns 1 on a miss
  unsigned set = slot & (c->sets - 1); st_t *s = &c->s[CL_CHECK]; s->acc += n;
  for (int w = 0; w < (int)c->ways; w++) if (c->occ[set][w] && c->slot[set][w] == slot && c->gen[set][w] == gen) {
    s->hit += n; if (!c->live[set][w]) c->deny_hit_dead += n; return 0;
  }
  s->miss++; s->mrd++; c->probes++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  if (rec_live) { lb_install(c, slot, gen); s->hit += n - 1; }   // the run's later checks hit what the probe installed
  else c->dead_record += n - 1;   // a dead id: the probe resolves DEAD, installs nothing, and the run's later checks are
                                  // denied by the one DEAD record (load_store_unit.sv rvc_dead_q): neither hit nor miss
  return 1;
}
static void lb_op(lb_t *c, uint64_t slot, uint32_t gen, int write, int live) {
  st_t *s = &c->s[CL_OP]; s->acc++; s->miss++;
  if (write) s->mwr++; else s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  if (write) { if (live) lb_install(c, slot, gen); else lb_invalidate(c, slot); }
  else if (live) lb_install(c, slot, gen);
}
static void lb_flush(lb_t *c) { memset(c->occ, 0, sizeof c->occ[0] * c->sets); memset(c->live, 0, sizeof c->live[0] * c->sets); tlb_clear(&c->tlb); c->edges_flushed++; }

// ---------------- RC, GC, NC: rcm.c's organisations, unchanged
typedef struct { uint64_t tag[64][2]; uint8_t valid[64][2], dirty[64][2], lru[64]; tlb_t tlb; st_t s[NCL]; uint64_t flush_wb; } rc_t;
static void rc_access(rc_t *c, uint64_t slot, int write, int cl) {
  uint64_t line = slot >> 2; unsigned set = line & 63; uint64_t tag = line >> 6;
  st_t *s = &c->s[cl]; s->acc++;
  for (int w = 0; w < 2; w++) if (c->valid[set][w] && c->tag[set][w] == tag) {
    s->hit++; c->lru[set] = 1 - w; if (write) c->dirty[set][w] = 1; return;
  }
  s->miss++;
  int w = c->valid[set][0] ? (c->valid[set][1] ? c->lru[set] : 1) : 0;
  if (c->valid[set][w] && c->dirty[set][w]) {
    s->wb++; s->mwr++;
    uint64_t vslot = ((c->tag[set][w] << 6) | set) << 2;
    if (!tlb_access(&c->tlb, vslot >> 8)) s->tlbmiss++;
  }
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  s->mrd++;
  c->valid[set][w] = 1; c->tag[set][w] = tag; c->dirty[set][w] = write; c->lru[set] = 1 - w;
}
static void rc_flush(rc_t *c) {
  for (int set = 0; set < 64; set++) for (int w = 0; w < 2; w++) {
    if (c->valid[set][w] && c->dirty[set][w]) c->flush_wb++;
    c->valid[set][w] = 0; c->dirty[set][w] = 0;
  }
  tlb_clear(&c->tlb);
}
typedef struct { uint64_t tag[128][4]; uint8_t valid[128][4]; uint64_t age[128][4]; uint64_t clock; tlb_t tlb; st_t s[NCL]; } gc_t;
static void gc_check(gc_t *c, uint64_t slot, int cl) {
  unsigned set = slot & 127; st_t *s = &c->s[cl]; s->acc++; c->clock++;
  for (int w = 0; w < 4; w++) if (c->valid[set][w] && c->tag[set][w] == (slot >> 7)) { s->hit++; c->age[set][w] = c->clock; return; }
  s->miss++; s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  int v = -1;
  for (int i = 0; i < 4; i++) if (!c->valid[set][i]) { v = i; break; }
  if (v < 0) { v = 0; for (int i = 1; i < 4; i++) if (c->age[set][i] < c->age[set][v]) v = i; }
  c->valid[set][v] = 1; c->tag[set][v] = slot >> 7; c->age[set][v] = c->clock;
}
static void gc_op(gc_t *c, uint64_t slot, int write, int cl) {
  st_t *s = &c->s[cl]; s->acc++; s->miss++;
  if (write) s->mwr++; else s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
}
static void gc_flush(gc_t *c) { memset(c->valid, 0, sizeof c->valid); tlb_clear(&c->tlb); }
typedef struct { tlb_t tlb; st_t s[NCL]; } nc_t;
static void nc_access(nc_t *c, uint64_t slot, int write, int cl, uint64_t n) {
  st_t *s = &c->s[cl]; s->acc += n; s->miss += n;
  if (write) s->mwr += n; else s->mrd += n;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
}

static lb_t LB; static rc_t RC; static gc_t GC; static nc_t NC;
static lb_t LB0;   // v3: b42c8c644's 64x4, the baseline the populations are removed from
#define NSEED 4
static uint64_t v3_checks, v3_bit44, v3_mismatch, pop_n[3], pop_res[3][NSEED];   // pops: 0 all LB0 misses, 1 removed_lb, 2 removed_rc

// ---------------- the op stretch buffer (--walk)
typedef struct { uint64_t slot, pos; uint32_t gen; uint8_t write, live, kind, cl; } opev_t;
static opev_t *buf, *obuf; static size_t nbuf, cbuf, cobuf; static int walk_dedup;
static void push_op(opev_t e) {
  if (nbuf == cbuf) { cbuf = cbuf ? 2 * cbuf : 4096; buf = realloc(buf, cbuf * sizeof *buf); if (!buf) { perror("realloc"); exit(2); } }
  e.pos = nbuf; buf[nbuf++] = e;
}
static void dispatch_op(const opev_t *e) {
  if (!in_op) { in_op = 1; LB.stretches++; }
  cur_kind = e->kind;
  lb_op(&LB, e->slot, e->gen, e->write, e->live);
  lb_op(&LB0, e->slot, e->gen, e->write, e->live);
  rc_access(&RC, e->slot, e->write, e->cl);
  gc_op(&GC, e->slot, e->write, e->cl);
  nc_access(&NC, e->slot, e->write, e->cl, 1);
}
static int by_slot_pos(const void *a, const void *b) {
  const opev_t *x = a, *y = b;
  if (x->slot != y->slot) return x->slot < y->slot ? -1 : 1;
  return x->pos < y->pos ? -1 : x->pos > y->pos;
}
static int by_pos(const void *a, const void *b) { const opev_t *x = a, *y = b; return x->pos < y->pos ? -1 : x->pos > y->pos; }
static size_t nob;
static void out_ev(opev_t e, uint64_t pos) {
  if (nob == cobuf) { cobuf = cobuf ? 2 * cobuf : 4096; obuf = realloc(obuf, cobuf * sizeof *obuf); if (!obuf) { perror("realloc"); exit(2); } }
  e.pos = pos; obuf[nob++] = e;
}
// end of a stretch: replay the buffered op events (asis: as traced; dedup: one canonical event pair per slot), then end it
static void stretch_end(void) {
  if (!walk_dedup || nbuf < 2) { for (size_t i = 0; i < nbuf; i++) dispatch_op(&buf[i]); }
  else {
    qsort(buf, nbuf, sizeof *buf, by_slot_pos); nob = 0;
    for (size_t i = 0, j; i < nbuf; i = j) {
      for (j = i; j < nbuf && buf[j].slot == buf[i].slot; j++) ;
      size_t fl = (size_t)-1, ll = 0, dw = (size_t)-1; int lwr = 0;
      for (size_t k = i; k < j; k++) {
        if (buf[k].live) { if (fl == (size_t)-1) fl = k; ll = k; if (buf[k].write) lwr = 1; }
      }
      for (size_t k = (fl == (size_t)-1 ? i : ll + 1); k < j; k++) if (buf[k].write && !buf[k].live) { dw = k; break; }
      if (fl != (size_t)-1) {
        opev_t e = buf[fl]; e.gen = buf[ll].gen; e.live = 1; e.write = lwr;
        out_ev(e, 2 * buf[fl].pos);
        if (dw != (size_t)-1) out_ev(buf[dw], 2 * buf[fl].pos + 1);   // the invalidate, right after the install
      } else if (dw != (size_t)-1) out_ev(buf[dw], 2 * buf[dw].pos);
      else out_ev(buf[i], 2 * buf[i].pos);                            // dead reads only: no LB effect
    }
    qsort(obuf, nob, sizeof *obuf, by_pos);
    for (size_t i = 0; i < nob; i++) dispatch_op(&obuf[i]);
  }
  nbuf = 0; sh_end();
}

// THIN (nested lane, 2026-11-27): --thin K:PH keeps tick t (1-based, counted at each E op 0 cause 0x80000007, the
// MTI) only when t % K == PH; a dropped tick has the flush bits cleared on every E from its MTI edge through the first
// MRET after an E op 0 cause 9 (the kernel's sbi_set_timer ecall), inclusive. An independent check of the rtl
// lane's thinned-trace readings; the replay is otherwise rcm-lb-dc unchanged.
static unsigned thin_k = 0, thin_ph = 0;
static uint64_t th_ticks, th_drop, th_closed, th_reopen, th_cleared, th_root_in, th_trap_in;
int main(int argc, char **argv) {
  int th_sup = 0, th_seen9 = 0;
  if (argc < 2) { fprintf(stderr, "usage: rcm-lb TRACE_V2 [--with-pcc] [--walk asis|dedup] [--lb SETSxWAYS]\n"); return 2; }
  int with_pcc = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--with-pcc")) with_pcc = 1;
    else if (!strcmp(argv[i], "--walk") && i + 1 < argc && (!strcmp(argv[i + 1], "asis") || !strcmp(argv[i + 1], "dedup"))) walk_dedup = !strcmp(argv[++i], "dedup");
    else if (!strcmp(argv[i], "--lb") && i + 1 < argc) {
      unsigned s = 0, w = 0; char x = 0, tail = 0;
      if (sscanf(argv[++i], "%u%c%u%c", &s, &x, &w, &tail) != 3 || x != 'x' || !s || (s & (s - 1)) || s > LB_MAXS || !w || w > LB_MAXW) {
        fprintf(stderr, "--lb wants SETSxWAYS, SETS a power of 2 <= %d, 1 <= WAYS <= %d: %s\n", LB_MAXS, LB_MAXW, argv[i]); return 2; }
      lb_sets = s; lb_ways = w;
    }
    else if (!strcmp(argv[i], "--thin") && i + 1 < argc) {
      if (sscanf(argv[++i], "%u:%u", &thin_k, &thin_ph) != 2 || thin_k < 1 || thin_ph >= thin_k) { fprintf(stderr, "--thin wants K:PH, PH < K\n"); return 2; }
    }
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  LB.sets = lb_sets; LB.ways = lb_ways; LB0.sets = 64; LB0.ways = 4;
  int fd = open(argv[1], O_RDONLY); if (fd < 0) { perror(argv[1]); return 2; }
  struct stat sb;
  if (fstat(fd, &sb) || sb.st_size == 0 || sb.st_size % 16) { fprintf(stderr, "%s: not a non-empty multiple of 16 bytes\n", argv[1]); return 2; }
  uint64_t ne = sb.st_size / 16;
  const uint64_t *w = mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (w == MAP_FAILED) { perror("mmap"); return 2; }
  uint64_t ev[16][4] = {{0}}, edges = 0, tables = 0, lastI = 0, lastP = 0, recs = 0;
  for (uint64_t i = 0; i < ne; i++) {
    uint64_t x = w[2 * i], y = w[2 * i + 1]; unsigned code = x >> 56, kind = code >> 2, op = code & 3;
    uint64_t v3 = 0;                                                   // v3: the residency bits, stripped
    if (kind < K_T) { v3 = (y >> 40) & 0x1f; y &= ~(0x1fULL << 40); }
    // word 1: a record event carries live[32] and generation[31:0] only; an E carries its flush bit only; T, I and
    // PI carry nothing. Checked BEFORE the marker branches below, which continue (f82523965196 checked after them,
    // so an E with word 1 = 2 was accepted; found by the nested lane).
    if (kind > K_E || (kind == K_E ? (y >> 1) != 0 : (kind >= K_T ? y != 0 : (y >> 33) != 0))) {
      fprintf(stderr, "malformed event at %llu\n", (unsigned long long)i); return 2;
    }
    if (kind == K_T || kind == K_E || kind <= K_P) stretch_end();   // a check, an edge or a table switch ends a stretch
    if (kind == K_T) { tables++; lb_flush(&LB); lb_flush(&LB0); rc_flush(&RC); gc_flush(&GC); tlb_clear(&NC.tlb); continue; }
    if (kind == K_I) { lastI = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_PI) { lastP = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_E) {
      unsigned cause = (unsigned)(x & 0xffffffffu);
      if (op == 0 && cause == 0x80000007u) {
        th_ticks++; if (th_sup) th_reopen++;
        th_sup = thin_k && (th_ticks % thin_k) != thin_ph; th_seen9 = 0; th_drop += th_sup;
      } else if (th_sup && op == 0 && cause == 9) th_seen9 = 1;
      else if (th_sup && op == 0) th_trap_in++;
      if (th_sup && op == 1) th_root_in++;
      if (th_sup && (y & 1)) { th_cleared++; y &= ~1ULL; }
      if (th_sup && op == 3 && th_seen9) { th_sup = 0; th_closed++; }
    }
    if (kind == K_E) { edges++; ev[K_E][op]++; if (y & 1) { lb_flush(&LB); lb_flush(&LB0); } continue; }   // RC and GC flush only on T
    uint64_t cnt = (x >> 32) & 0xffffff, slot = x & 0xffffffff; uint32_t gen = (uint32_t)y; int live = (y >> 32) & 1;
    if (cnt == 0) { fprintf(stderr, "zero-count run at %llu\n", (unsigned long long)i); return 2; }
    ev[kind][op] += cnt;
    if (op >= 2) continue;                                      // header words are not records
    if (kind == K_P && !with_pcc) continue;
    recs++;
    int write = (op == 1), cl = (kind <= K_P) ? CL_CHECK : CL_OP;
    if (cl == CL_CHECK && !write) {
      int mt = lb_check(&LB, slot, gen, live, cnt);
      int m0 = lb_check(&LB0, slot, gen, live, cnt);
      uint64_t rcm0 = RC.s[cl].miss;
      rc_access(&RC, slot, 0, cl); if (cnt > 1) { RC.s[cl].acc += cnt - 1; RC.s[cl].hit += cnt - 1; }
      int mr = RC.s[cl].miss != rcm0;
      v3_checks++; if (v3 >> 4) v3_bit44++;
      if (((v3 >> 4) & 1) != (unsigned)m0) v3_mismatch++;
      if (m0) {
        int in[3] = { 1, !mt, !mr };
        for (int p = 0; p < 3; p++) if (in[p]) { pop_n[p]++; for (int sd = 0; sd < NSEED; sd++) pop_res[p][sd] += (v3 >> sd) & 1; }
      }
      gc_check(&GC, slot, cl); if (cnt > 1) { GC.s[cl].acc += cnt - 1; GC.s[cl].hit += cnt - 1; }
      nc_access(&NC, slot, 0, cl, cnt);
    } else {
      push_op((opev_t){ .slot = slot, .gen = gen, .write = write, .live = live, .kind = kind, .cl = cl });
    }
  }
  stretch_end();
  if (!recs) { fprintf(stderr, "no record events\n"); return 2; }
  if (v3_bit44 && walk_dedup) { fprintf(stderr, "v3: the tracer's LB is asis; --walk dedup is refused on a v3 trace\n"); return 2; }
  printf("trace %s\nevents %llu tables %llu edges %llu instructions_all %llu instructions_protected %llu with_pcc %d\n",
         argv[1], (unsigned long long)ne, (unsigned long long)tables, (unsigned long long)edges,
         (unsigned long long)lastI, (unsigned long long)lastP, with_pcc);
  printf("%-3s %-5s %14s %14s %14s %14s %14s %12s %12s\n", "org", "class", "accesses", "hits", "misses", "mem_reads", "mem_writes", "tlb_misses", "writebacks");
  st_t *S[4] = {LB.s, RC.s, GC.s, NC.s}; const char *ON[4] = {"LB", "RC", "GC", "NC"};
  for (int m = 0; m < 4; m++) for (int cl = 0; cl < NCL; cl++) {
    st_t *s = &S[m][cl];
    printf("%-3s %-5s %14llu %14llu %14llu %14llu %14llu %12llu %12llu\n", ON[m], cl ? "op" : "check",
           (unsigned long long)s->acc, (unsigned long long)s->hit, (unsigned long long)s->miss, (unsigned long long)s->mrd,
           (unsigned long long)s->mwr, (unsigned long long)s->tlbmiss, (unsigned long long)s->wb);
  }
  printf("LB probes %llu deny_hit_dead %llu dead_record %llu installs %llu invalidations %llu flushes %llu\n",
         (unsigned long long)LB.probes, (unsigned long long)LB.deny_hit_dead, (unsigned long long)LB.dead_record, (unsigned long long)LB.installs,
         (unsigned long long)LB.inval, (unsigned long long)LB.edges_flushed);
  printf("LB op_stretches %llu self_evict %llu untracked %llu\n", (unsigned long long)LB.stretches,
         (unsigned long long)LB.self_evict, (unsigned long long)LB.untracked);
  printf("LB self_evict by kind a %llu r %llu v %llu o %llu walk %s\n", (unsigned long long)LB.se_kind[K_A],
         (unsigned long long)LB.se_kind[K_R], (unsigned long long)LB.se_kind[K_V], (unsigned long long)LB.se_kind[K_O],
         walk_dedup ? "dedup" : "asis");
  printf("LB geometry %u sets x %u ways\n", lb_sets, lb_ways);
  printf("E by op %llu %llu %llu %llu\n", (unsigned long long)ev[K_E][0], (unsigned long long)ev[K_E][1],
         (unsigned long long)ev[K_E][2], (unsigned long long)ev[K_E][3]);
  printf("RC flush_writebacks %llu\n", (unsigned long long)RC.flush_wb);
  printf("v3 check_run_starts %llu bit44 %llu mismatch_vs_LB0 %llu LB0_misses %llu\n", (unsigned long long)v3_checks,
         (unsigned long long)v3_bit44, (unsigned long long)v3_mismatch, (unsigned long long)LB0.s[CL_CHECK].miss);
  static const char *const PN[3] = { "all_LB0_misses", "removed_lb", "removed_rc" };
  for (int p = 0; p < 3; p++) {
    printf("v3 pop %-14s n %14llu resident", PN[p], (unsigned long long)pop_n[p]);
    for (int sd = 0; sd < NSEED; sd++) printf(" %llu", (unsigned long long)pop_res[p][sd]);
    printf(" cold_fraction");
    for (int sd = 0; sd < NSEED; sd++) printf(" %.6f", pop_n[p] ? 1.0 - (double)pop_res[p][sd] / pop_n[p] : -1.0);
    printf("\n");
  }
  printf("thin k %u ph %u ticks %llu dropped %llu closed %llu reopened %llu open_at_end %d flush_bits_cleared %llu root_writes_inside %llu other_traps_inside %llu\n",
         thin_k, thin_ph, (unsigned long long)th_ticks, (unsigned long long)th_drop, (unsigned long long)th_closed, (unsigned long long)th_reopen, th_sup,
         (unsigned long long)th_cleared, (unsigned long long)th_root_in, (unsigned long long)th_trap_in);
  printf("END\n");
  return 0;
}
