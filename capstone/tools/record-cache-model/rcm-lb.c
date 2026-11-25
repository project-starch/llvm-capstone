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
// Runs: a check run of N is N checks of one id; an operation run is ONE access, as rcm.c.
// Exit 2 on an empty, malformed or unreadable trace: no data is an error.
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

// ---------------- LB: b42c8c644's verdict cache
typedef struct {
  uint64_t slot[64][4]; uint32_t gen[64][4]; uint8_t occ[64][4], live[64][4], rr[64];
  tlb_t tlb; st_t s[NCL];
  uint64_t probes, deny_hit_dead, dead_record, installs, inval, edges_flushed;
} lb_t;
static void lb_install(lb_t *c, uint64_t slot, uint32_t gen) {
  unsigned set = slot & 63; int way = c->rr[set], reuse = 0;
  for (int w = 0; w < 4; w++) if (c->occ[set][w] && c->slot[set][w] == slot) { way = w; reuse = 1; }
  if (!reuse) for (int w = 3; w >= 0; w--) if (!c->occ[set][w]) { way = w; reuse = 1; }
  if (!reuse) c->rr[set] = (c->rr[set] + 1) & 3;
  c->occ[set][way] = 1; c->slot[set][way] = slot; c->gen[set][way] = gen; c->live[set][way] = 1; c->installs++;
}
static void lb_invalidate(lb_t *c, uint64_t slot) {
  unsigned set = slot & 63;
  for (int w = 0; w < 4; w++) if (c->occ[set][w] && c->slot[set][w] == slot && c->live[set][w]) { c->live[set][w] = 0; c->inval++; }
}
// a check: hit live (allow), hit dead (deny, no probe), or a miss (a probe: one record read, an install if live)
static void lb_check(lb_t *c, uint64_t slot, uint32_t gen, int rec_live, uint64_t n) {
  unsigned set = slot & 63; st_t *s = &c->s[CL_CHECK]; s->acc += n;
  for (int w = 0; w < 4; w++) if (c->occ[set][w] && c->slot[set][w] == slot && c->gen[set][w] == gen) {
    s->hit += n; if (!c->live[set][w]) c->deny_hit_dead += n; return;
  }
  s->miss++; s->mrd++; c->probes++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  if (rec_live) { lb_install(c, slot, gen); s->hit += n - 1; }   // the run's later checks hit what the probe installed
  else c->dead_record += n - 1;   // a dead id: the probe resolves DEAD, installs nothing, and the run's later checks are
                                  // denied by the one DEAD record (load_store_unit.sv rvc_dead_q): neither hit nor miss
}
static void lb_op(lb_t *c, uint64_t slot, uint32_t gen, int write, int live) {
  st_t *s = &c->s[CL_OP]; s->acc++; s->miss++;
  if (write) s->mwr++; else s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  if (write) { if (live) lb_install(c, slot, gen); else lb_invalidate(c, slot); }
  else if (live) lb_install(c, slot, gen);
}
static void lb_flush(lb_t *c) { memset(c->occ, 0, sizeof c->occ); memset(c->live, 0, sizeof c->live); tlb_clear(&c->tlb); c->edges_flushed++; }

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

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: rcm-lb TRACE_V2 [--with-pcc]\n"); return 2; }
  int with_pcc = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--with-pcc")) with_pcc = 1;
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  int fd = open(argv[1], O_RDONLY); if (fd < 0) { perror(argv[1]); return 2; }
  struct stat sb;
  if (fstat(fd, &sb) || sb.st_size == 0 || sb.st_size % 16) { fprintf(stderr, "%s: not a non-empty multiple of 16 bytes\n", argv[1]); return 2; }
  uint64_t ne = sb.st_size / 16;
  const uint64_t *w = mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (w == MAP_FAILED) { perror("mmap"); return 2; }
  uint64_t ev[8][4] = {{0}}, edges = 0, tables = 0, lastI = 0, lastP = 0, recs = 0;
  for (uint64_t i = 0; i < ne; i++) {
    uint64_t x = w[2 * i], y = w[2 * i + 1]; unsigned code = x >> 56, kind = code >> 2, op = code & 3;
    if (kind == K_T) { tables++; lb_flush(&LB); rc_flush(&RC); gc_flush(&GC); tlb_clear(&NC.tlb); continue; }
    if (kind == K_I) { lastI = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_PI) { lastP = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_E) { edges++; lb_flush(&LB); continue; }    // the RTL's namespace flush; RC and GC flush only on T
    if (kind > K_E || (y >> 33)) { fprintf(stderr, "malformed event at %llu\n", (unsigned long long)i); return 2; }
    uint64_t cnt = (x >> 32) & 0xffffff, slot = x & 0xffffffff; uint32_t gen = (uint32_t)y; int live = (y >> 32) & 1;
    if (cnt == 0) { fprintf(stderr, "zero-count run at %llu\n", (unsigned long long)i); return 2; }
    ev[kind][op] += cnt;
    if (op >= 2) continue;                                      // header words are not records
    if (kind == K_P && !with_pcc) continue;
    recs++;
    int write = (op == 1), cl = (kind <= K_P) ? CL_CHECK : CL_OP;
    if (cl == CL_CHECK && !write) {
      lb_check(&LB, slot, gen, live, cnt);
      rc_access(&RC, slot, 0, cl); if (cnt > 1) { RC.s[cl].acc += cnt - 1; RC.s[cl].hit += cnt - 1; }
      gc_check(&GC, slot, cl); if (cnt > 1) { GC.s[cl].acc += cnt - 1; GC.s[cl].hit += cnt - 1; }
      nc_access(&NC, slot, 0, cl, cnt);
    } else {
      lb_op(&LB, slot, gen, write, live);
      rc_access(&RC, slot, write, cl);
      gc_op(&GC, slot, write, cl);
      nc_access(&NC, slot, write, cl, 1);
    }
  }
  if (!recs) { fprintf(stderr, "no record events\n"); return 2; }
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
  printf("RC flush_writebacks %llu\nEND\n", (unsigned long long)RC.flush_wb);
  return 0;
}
