/* The Capstone virtual-profile twin of probe.c: is the OLD capability after realloc() still
 * usable? One scenario per invocation (the virtual VM runs one domain per launch, no fork), the
 * scenario named by argv[1]. Each prints the two capabilities, whether the address stayed the
 * same, then reads one byte THROUGH THE OLD CAPABILITY at the labelled lbu `realloc_probe_read`.
 * A completed read prints ACCESS; a denied one shows in the serial log as a domain fault whose pc
 * the harness resolves to that label.
 *
 * Two controls: `fixed` reads through the NEW capability after a moving realloc (must complete),
 * and `freed` reads through a freed capability (must fault), so a fault at the label is shown to
 * be producible by this binary before any realloc verdict is read. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct scenario {
  const char *name;
  size_t first, second, index;
};
static const struct scenario SCENARIOS[] = {
    {"shrink-small-same-class", 256, 250, 255},
    {"shrink-small-other-class", 256, 200, 0},
    {"grow-small-within-class", 100, 112, 0},
    {"shrink-large-in-place", 1u << 20, 600u * 1024, (1u << 20) - 1},
    {"grow-moved", 64, 1u << 16, 0},
    /* mruby Array#sort! use-after-realloc, upstream fix eb398971b (#6649): a block shrinks the
     * array mid-sort, ary_shrink_capa() reallocs the backing store down, and the sort keeps the
     * captured base and reads the stale tail. The READ oracle below reads old+504 after a
     * 512 -> 256 B realloc. publish() in runtime/virtual/heap.c revokes the old child on every
     * successful realloc, so the stale read faults at the label whether or not the block moved. */
    {"mruby-sort-bang-shrink", 512, 256, 504},
};

/* The FFmpeg diracdec two-owners shape (upstream fix 8a4ea9644833): the realloc's result is kept
 * in a second field, the first keeps the old pointer and is still written through. The store is
 * the labelled sb `realloc_probe_write`; if it completes, the byte is read back through the NEW
 * capability to see whether the stale write landed in the live object. */
static const struct scenario TWO_OWNERS[] = {
    {"two-owners-in-place", 256, 250, 0},
    {"two-owners-moved", 64, 1u << 16, 0},
    /* Perl Gv_AMupdate stale HvAUX, upstream fix 4547997058 (RT #121362): aux = HvAUX(stash) is
     * an interior pointer into the bucket array's tail; S_hsplit's Renew() grows the array and
     * aux->xhv_aux_flags |= ... then writes through the stale interior pointer into live buckets.
     * The store below goes through old+192 (the interior aux slot) after a 200 -> 224 B grow, then
     * reads new+192 to see whether the stale write landed in the live object. */
    {"perl-hsplit-aux-grow", 200, 224, 192},
};

__attribute__((noinline)) static void write_old(volatile unsigned char *p, unsigned char v) {
  __asm__ volatile(".globl realloc_probe_write\nrealloc_probe_write:\nsb %0, 0(%1)\n"
                   :
                   : "r"(v), "r"(p)
                   : "memory");
}

__attribute__((noinline)) static unsigned read_old(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl realloc_probe_read\nrealloc_probe_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return (unsigned)value;
}

static void describe(const char *tag, const void *p) {
  printf(" %s=%#lx[%#lx,%#lx)", tag, (unsigned long)__builtin_capstone_cap_get_cursor((void *)p),
         (unsigned long)__builtin_capstone_cap_get_base((void *)p),
         (unsigned long)__builtin_capstone_cap_get_end((void *)p));
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "";
  if (!strcmp(mode, "fixed") || !strcmp(mode, "freed")) {
    unsigned char *p = malloc(64);
    if (!p) return 2;
    memset(p, 0x5a, 64);
    unsigned char *volatile keep = p;
    if (!strcmp(mode, "fixed")) {
      unsigned char *q = realloc(p, 1u << 16);
      if (!q) return 2;
      printf("SCEN fixed"); describe("old", keep); describe("new", q); printf("\n"); fflush(stdout);
      unsigned v = read_old(q);
      printf("ACCESS fixed completed value=0x%02x\n", v);
      fflush(stdout);
      free(q);
      return 0;
    }
    free(p);
    printf("SCEN freed"); describe("old", keep); printf("\n"); fflush(stdout);
    unsigned v = read_old(keep);
    printf("ACCESS freed completed value=0x%02x\n", v);
    fflush(stdout);
    return 0;
  }
  for (size_t i = 0; i < sizeof TWO_OWNERS / sizeof TWO_OWNERS[0]; i++) {
    const struct scenario *s = &TWO_OWNERS[i];
    if (strcmp(mode, s->name)) continue;
    unsigned char *thread_buf = malloc(s->first);
    if (!thread_buf) return 2;
    memset(thread_buf, 0x5a, s->first);
    unsigned char *volatile first_owner = thread_buf;
    unsigned char *slice_params_buf = realloc(thread_buf, s->second);
    if (!slice_params_buf) return 2;
    printf("SCEN %s", s->name);
    describe("old", first_owner); describe("new", slice_params_buf);
    printf(" same_addr=%d index=%zu\n",
           __builtin_capstone_cap_get_cursor((void *)first_owner) ==
               __builtin_capstone_cap_get_cursor(slice_params_buf),
           s->index);
    fflush(stdout);
    write_old(first_owner + s->index, 0xa5);
    unsigned seen = slice_params_buf[s->index];
    printf("ACCESS %s completed stale_write_seen_through_new=%d value=0x%02x\n", s->name,
           seen == 0xa5, seen);
    fflush(stdout);
    free(slice_params_buf);
    return 0;
  }
  for (size_t i = 0; i < sizeof SCENARIOS / sizeof SCENARIOS[0]; i++) {
    const struct scenario *s = &SCENARIOS[i];
    if (strcmp(mode, s->name)) continue;
    unsigned char *old = malloc(s->first);
    if (!old) return 2;
    memset(old, 0x5a, s->first);
    unsigned char *volatile keep = old;
    unsigned char *new = realloc(old, s->second);
    if (!new) return 2;
    printf("SCEN %s", s->name);
    describe("old", keep); describe("new", new);
    printf(" same_addr=%d index=%zu\n",
           __builtin_capstone_cap_get_cursor((void *)keep) == __builtin_capstone_cap_get_cursor(new),
           s->index);
    fflush(stdout);
    unsigned v = read_old(keep + s->index);
    printf("ACCESS %s completed value=0x%02x\n", s->name, v);
    fflush(stdout);
    free(new);
    return 0;
  }
  fprintf(stderr, "unknown mode %s\n", mode);
  return 3;
}
