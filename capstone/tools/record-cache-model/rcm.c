// rcm.c -- record-cache model: replay a record-access trace (nested lane's tracing twin, rectrace-read.py format)
// through five organisations (RC, GC, NC, P1, B5 below) in one pass, and print counts. 2026-11-20, rtl lane, item 4 (i).
//
//   RC  the record cache of Section 5.2 (paper notes record-cache-design-2026-11-13.txt): 8 KiB, 2 ways, 64 sets,
//       64-byte lines of 4 records (16 B), set = s[7:2], tag = s[30:8], LRU, write-allocate, write-back. Used by
//       every record read and write (checks, allocation, release, release's read-only pass, other operations).
//   GC  the generation-only cache of 12 November (generation-cache-design-2026-11-12.txt): 512 entries, 4 ways,
//       128 sets, set = s[6:0], one generation per entry, LRU. Used by checks only; every other record read goes
//       to memory; every record write goes to memory and updates a present entry in place.
//   NC  no cache: every record access is a memory access.
//   P1  P1's verdict cache (capstone-ariane d83d3a633 load_store_unit.sv:277-278, :1251, :1311, :1375-1377): 64 sets x
//       4 ways of ids, set = slot[5:0], round-robin replacement per set; a check miss reads the record and installs;
//       every record write invalidates the slot's entry (the index broadcast); operations go to memory as in GC.
//   B5  B5a's stack (capstone-ariane 6e82d7bd2 / 7180a83b6, ex_stage.sv:2203-2232): P1's verdict cache in front; its
//       misses and every operation read go to a record cache of 64 sets x 2 ways of 4-record lines, LRU, per-record
//       valid bits, slots 0 and 1 never cached; a read miss installs the line with the demanded record and FILLS the
//       other three (counted as fill reads; the silicon yields a fill to the next request, so this is the upper bound);
//       write-through: every write goes out; a write hit updates the record, a write miss allocates nothing.
//       Events as B5a's HPM: 34 read hits, 35 read misses, 37 fill reads, and 31 = every walker record access =
//       35 + 37 + writes (+ slots 0/1 reads, which bypass). NOT MODELLED: B5a also flushes at every namespace edge
//       (each context entry/escape), which the trace does not mark; only T lines and fresh tables flush here.
// Each has its own 16-entry fully associative LRU record TLB over record pages of 256 records (page = s >> 8);
// a TLB miss is a directory walk.
// Run semantics: a run is N identical events on one slot. In RC/GC its first event is a real access and the
// other N-1 hit. In NC a run of checks (c, k, p) is N accesses; a run of a, r, v or o is ONE access (the model
// reads the same record 2-3 times for one operation; the nested lane's instruction).
// Table boundaries: a T line, or `ahw 8` followed by `aw 1024` (a fresh table at the same root, i.e. a new
// process). There RC writes back its dirty lines, and every cache and TLB is cleared.
// p (PCC checks, all slot 1024 on these traces) is excluded unless --with-pcc.
// Context edges (kind 10; the nested lane's tracer, QEMU e3f02b0202c9: op 1 = a context entered or resumed, op 0 = it
// left, slot = kind << 16 | cause; op 2 / 3 = sret / mret into protected U, unused on these traces): B5a and P1 empty
// their verdict and record caches at every namespace edge, so P1 and B5 flush there; RC and GC (the paper's design)
// flush only at a table switch. Counted as edges either way (the "trap" column prints op 0, the leaves).
// HARDWARE PATTERN (default; --raw turns it off), applied to EVERY organisation alike:
//  * model-only writes (the nested lane's reading of the twin: an allocation writes the new record twice, a release
//    writes each released record before its push): a record write to the same slot as the previous record write by the
//    SAME KIND of operation (header words ignored) is a dup_write (a release followed by a reallocation of the slot is a
//    real write);
//  * model-only reads (the RTL unit reads each record an operation touches once, capstone_rev_node.anvil derive_req /
//    kill_req / REVOKE_NODE): within one operation -- a stretch of allocation (a) or release (r, v) events not
//    interrupted by a check or by the other kind -- only the first read of each slot counts; repeats are dup_reads.
//  Neither reaches any organisation.
// Exit 2 on an empty, malformed or unreadable trace: no data is an error.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

enum { K_C, K_K, K_P, K_A, K_R, K_V, K_O, K_T, K_I, K_PI, K_E };
static const char *KN = "ckparvoT";
enum { CL_CHECK, CL_OP, NCL };          // class: checks (c, k, p) vs operations (a, r, v, o)

typedef struct { uint64_t acc, hit, miss, mrd, mwr, tlbmiss, wb; } st_t;

// ---------------- record TLB
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

// ---------------- RC: 64 sets x 2 ways of lines
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

// ---------------- GC: 128 sets x 4 ways of generations
typedef struct { uint64_t tag[128][4]; uint8_t valid[128][4]; uint64_t age[128][4]; uint64_t clock; tlb_t tlb; st_t s[NCL]; } gc_t;
static int gc_find(gc_t *c, uint64_t slot) {
  unsigned set = slot & 127; uint64_t tag = slot >> 7;
  for (int w = 0; w < 4; w++) if (c->valid[set][w] && c->tag[set][w] == tag) return w;
  return -1;
}
static void gc_check(gc_t *c, uint64_t slot, int cl) {
  unsigned set = slot & 127; st_t *s = &c->s[cl]; s->acc++; c->clock++;
  int w = gc_find(c, slot);
  if (w >= 0) { s->hit++; c->age[set][w] = c->clock; return; }
  s->miss++; s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  int v = -1;
  for (int i = 0; i < 4; i++) if (!c->valid[set][i]) { v = i; break; }
  if (v < 0) { v = 0; for (int i = 1; i < 4; i++) if (c->age[set][i] < c->age[set][v]) v = i; }
  c->valid[set][v] = 1; c->tag[set][v] = slot >> 7; c->age[set][v] = c->clock;
}
static void gc_op(gc_t *c, uint64_t slot, int write, int cl) {
  st_t *s = &c->s[cl]; s->acc++; s->miss++;          // an operation's record access always goes to memory
  if (write) s->mwr++; else s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  // a write updates a present entry in place (write-through with update): no invalidation, no extra traffic
}
static void gc_flush(gc_t *c) { memset(c->valid, 0, sizeof c->valid); tlb_clear(&c->tlb); }

// ---------------- NC
typedef struct { tlb_t tlb; st_t s[NCL]; } nc_t;
static void nc_access(nc_t *c, uint64_t slot, int write, int cl, uint64_t n) {
  st_t *s = &c->s[cl]; s->acc += n; s->miss += n;
  if (write) s->mwr += n; else s->mrd += n;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;  // repeats of one slot hit the TLB
}

// ---------------- P1: 64 sets x 4 ways, round-robin, invalidate on record write
typedef struct { uint64_t tag[64][4]; uint8_t valid[64][4], rr[64]; tlb_t tlb; st_t s[NCL]; } p1_t;
static void p1_check(p1_t *c, uint64_t slot, int cl) {
  unsigned set = slot & 63; uint64_t tag = slot >> 6; st_t *s = &c->s[cl]; s->acc++;
  for (int w = 0; w < 4; w++) if (c->valid[set][w] && c->tag[set][w] == tag) { s->hit++; return; }
  s->miss++; s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
  int v = c->rr[set]; c->rr[set] = (c->rr[set] + 1) & 3;
  c->valid[set][v] = 1; c->tag[set][v] = tag;
}
static void p1_op(p1_t *c, uint64_t slot, int write, int cl) {
  st_t *s = &c->s[cl]; s->acc++; s->miss++;
  if (write) { s->mwr++; unsigned set = slot & 63; for (int w = 0; w < 4; w++) if (c->valid[set][w] && c->tag[set][w] == (slot >> 6)) c->valid[set][w] = 0; }
  else s->mrd++;
  if (!tlb_access(&c->tlb, slot >> 8)) s->tlbmiss++;
}
static void p1_flush(p1_t *c) { memset(c->valid, 0, sizeof c->valid); tlb_clear(&c->tlb); }

typedef struct { p1_t vc; uint64_t tag[64][2]; uint8_t lv[64][2], valid[64][2], lru[64];
                 uint64_t e34, e35, e37, wr, bypass, vcmiss, opread; } b5_t;
static void b5_read(b5_t *c, uint64_t slot) {
  if (slot < 2) { c->bypass++; return; }
  uint64_t line = slot >> 2; unsigned set = line & 63; uint64_t tag = line >> 6; unsigned r = slot & 3;
  for (int w = 0; w < 2; w++) if (c->valid[set][w] && c->tag[set][w] == tag) {
    c->lru[set] = 1 - w;
    if (c->lv[set][w] >> r & 1) { c->e34++; return; }
    c->e35++; c->lv[set][w] |= 1 << r; return;           // a present line with this record invalid: demand read only
  }
  c->e35++;
  int w = c->valid[set][0] ? (c->valid[set][1] ? c->lru[set] : 1) : 0;
  c->valid[set][w] = 1; c->tag[set][w] = tag; c->lru[set] = 1 - w;
  uint8_t mask = 0xf; if (line == 0) mask &= ~3;           // slots 0 and 1 are never cached
  c->lv[set][w] = mask; c->e37 += __builtin_popcount(mask & ~(1u << r));
}
static void b5_write(b5_t *c, uint64_t slot) {
  c->wr++;
  if (slot < 2) return;
  uint64_t line = slot >> 2; unsigned set = line & 63; uint64_t tag = line >> 6;
  for (int w = 0; w < 2; w++) if (c->valid[set][w] && c->tag[set][w] == tag) { c->lru[set] = 1 - w; return; }  // update in place
}
static void b5_check(b5_t *c, uint64_t slot) {
  uint64_t before = c->vc.s[CL_CHECK].miss;
  p1_check(&c->vc, slot, CL_CHECK);
  if (c->vc.s[CL_CHECK].miss != before) { c->vcmiss++; b5_read(c, slot); }
}
static void b5_op(b5_t *c, uint64_t slot, int write) {
  p1_op(&c->vc, slot, write, CL_OP);                        // invalidates the verdict on a write
  if (write) b5_write(c, slot); else { c->opread++; b5_read(c, slot); }
}
static void b5_flush(b5_t *c) { p1_flush(&c->vc); memset(c->valid, 0, sizeof c->valid); }

static rc_t RC; static gc_t GC; static nc_t NC; static p1_t PV; static b5_t B5;

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: rcm TRACE [--with-pcc] [--max-words N]\n"); return 2; }
  int with_pcc = 0, raw = 0; uint64_t maxw = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--with-pcc")) with_pcc = 1;
    else if (!strcmp(argv[i], "--raw")) raw = 1;
    else if (!strcmp(argv[i], "--max-words") && i + 1 < argc) maxw = strtoull(argv[++i], 0, 0);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  int fd = open(argv[1], O_RDONLY); if (fd < 0) { perror(argv[1]); return 2; }
  struct stat sb; if (fstat(fd, &sb) || sb.st_size == 0 || sb.st_size % 8) { fprintf(stderr, "%s: not a non-empty multiple of 8\n", argv[1]); return 2; }
  uint64_t nw = sb.st_size / 8; if (maxw && maxw < nw) nw = maxw;
  const uint64_t *w = mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (w == MAP_FAILED) { perror("mmap"); return 2; }
  madvise((void *)w, sb.st_size, MADV_SEQUENTIAL);
  uint64_t ev[8][4] = {{0}}, runs[8][4] = {{0}}, roots = 0, fresh = 0, lastP = 0, lastI = 0;
  int prev_ahw8 = 0; uint64_t edges[4] = {0}, dupw = 0, lastw = ~0ULL; unsigned lastwk = 99; uint64_t opset[64]; int opn = 0; unsigned opk = 99; uint64_t dupr = 0;
  for (uint64_t i = 0; i < nw; i++) {
    uint64_t x = w[i]; unsigned code = x >> 56, kind = code >> 2, op = code & 3;
    if (kind == K_T) { roots++; rc_flush(&RC); gc_flush(&GC); p1_flush(&PV); b5_flush(&B5); tlb_clear(&NC.tlb); prev_ahw8 = 0; continue; }
    if (kind == K_I) { lastI = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_PI) { lastP = x & ((1ULL << 56) - 1); continue; }
    if (kind == K_E) { edges[op]++; p1_flush(&PV); b5_flush(&B5); continue; }
    if (kind > K_E) { fprintf(stderr, "malformed code %u at word %llu\n", code, (unsigned long long)i); return 2; }
    uint64_t cnt = (x >> 32) & 0xffffff, slot = x & 0xffffffff;
    ev[kind][op] += cnt; runs[kind][op]++;
    if (op >= 2) { prev_ahw8 = (kind == K_A && op == 3 && slot == 8); continue; }   // header word: not a record
    if (prev_ahw8 && kind == K_A && op == 1 && slot == 1024) { fresh++; rc_flush(&RC); gc_flush(&GC); p1_flush(&PV); b5_flush(&B5); tlb_clear(&NC.tlb); }
    prev_ahw8 = 0;
    if (kind == K_P && !with_pcc) continue;
    int write = (op == 1);
    int cl = (kind <= K_P) ? CL_CHECK : CL_OP;
    unsigned okind = (kind == K_V) ? K_R : kind;
    if (cl == CL_CHECK || okind != opk) { opn = 0; opk = (cl == CL_CHECK) ? 99 : okind; }
    int b5_dup_read = 0;
    if (cl == CL_OP && !write) {
      for (int q = 0; q < opn; q++) if (opset[q] == slot) { b5_dup_read = 1; break; }
      if (!b5_dup_read && opn < 64) opset[opn++] = slot;
      if (b5_dup_read && !raw) dupr++;
    }
    int dupw_now = write && slot == lastw && kind == lastwk;
    if (write) { lastw = slot; lastwk = kind; }
    if (cl == CL_OP && !raw && (dupw_now || b5_dup_read)) {
      /* hardware pattern (default): a model-only rewrite or re-read reaches no organisation; a duplicate write still
         invalidates P1's verdict (both copies do) -- it is the same slot, so it is already invalid after the first */
      if (dupw_now) dupw++;
      continue;
    }
    rc_access(&RC, slot, write, cl);
    if (cnt > 1 && cl == CL_CHECK) { RC.s[cl].acc += cnt - 1; RC.s[cl].hit += cnt - 1; }   // an op run is one access
    if (cl == CL_CHECK && !write) {
      gc_check(&GC, slot, cl);
      if (cnt > 1) { GC.s[cl].acc += cnt - 1; GC.s[cl].hit += cnt - 1; }
      p1_check(&PV, slot, cl);
      b5_check(&B5, slot);
      if (cnt > 1) { PV.s[cl].acc += cnt - 1; PV.s[cl].hit += cnt - 1; }
      nc_access(&NC, slot, write, cl, cnt);
    } else {
      gc_op(&GC, slot, write, cl);
      p1_op(&PV, slot, write, cl);
      b5_op(&B5, slot, write);
      nc_access(&NC, slot, write, cl, 1);
    }
  }
  printf("trace %s\nwords %llu of %llu\nroots %llu fresh_tables %llu\ninstructions_all %llu instructions_protected %llu\n",
         argv[1], (unsigned long long)nw, (unsigned long long)(sb.st_size / 8), (unsigned long long)roots,
         (unsigned long long)fresh, (unsigned long long)lastI, (unsigned long long)lastP);
  static const char *OPN[4] = {"read", "write", "hread", "hwrite"};
  for (int k = 0; k < 7; k++) for (int o = 0; o < 4; o++) if (runs[k][o])
    printf("events %c%-6s %llu runs %llu\n", KN[k], OPN[o], (unsigned long long)ev[k][o], (unsigned long long)runs[k][o]);
  printf("with_pcc %d\nedges trap %llu sret %llu mret %llu dup_writes %llu dup_reads %llu\n", with_pcc, (unsigned long long)edges[0], (unsigned long long)edges[2], (unsigned long long)edges[3], (unsigned long long)dupw, (unsigned long long)dupr);
  printf("%-3s %-5s %14s %14s %14s %14s %14s %12s %12s\n", "org", "class", "accesses", "hits", "misses", "mem_reads", "mem_writes", "tlb_misses", "writebacks");
  st_t *S[4] = {RC.s, GC.s, NC.s, PV.s}; const char *ON[4] = {"RC", "GC", "NC", "P1"};
  for (int m = 0; m < 4; m++) for (int cl = 0; cl < NCL; cl++) {
    st_t *s = &S[m][cl];
    printf("%-3s %-5s %14llu %14llu %14llu %14llu %14llu %12llu %12llu\n", ON[m], cl ? "op" : "check",
           (unsigned long long)s->acc, (unsigned long long)s->hit, (unsigned long long)s->miss, (unsigned long long)s->mrd,
           (unsigned long long)s->mwr, (unsigned long long)s->tlbmiss, (unsigned long long)s->wb);
  }
  printf("RC flush_writebacks %llu\n", (unsigned long long)RC.flush_wb);
  { uint64_t e31 = B5.e35 + B5.e37 + B5.wr;
    printf("B5 vc_misses %llu op_reads %llu ev34 %llu ev35 %llu ev37 %llu writes %llu bypass %llu ev31 %llu hit_rate %.4f fill_share %.4f ev31-35-37 %llu\n",
      (unsigned long long)B5.vcmiss, (unsigned long long)B5.opread, (unsigned long long)B5.e34, (unsigned long long)B5.e35,
      (unsigned long long)B5.e37, (unsigned long long)B5.wr, (unsigned long long)B5.bypass, (unsigned long long)e31,
      (double)B5.e34 / (B5.e34 + B5.e35), (double)B5.e37 / e31, (unsigned long long)(e31 - B5.e35 - B5.e37)); }
  uint64_t tot = 0; for (int k = 0; k < 7; k++) for (int o = 0; o < 4; o++) tot += runs[k][o];
  if (!tot) { fprintf(stderr, "no events\n"); return 2; }
  printf("END\n");
  return 0;
}
