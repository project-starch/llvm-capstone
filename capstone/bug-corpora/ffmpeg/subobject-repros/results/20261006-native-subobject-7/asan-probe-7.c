/* ASan's blindness to the SIX crossing shapes cases 3-9 add, measured two-sided
 * with a positive control that fires.
 *
 * Built at -O0 with volatile and read-backs. -O0 is LOAD-BEARING and the reason
 * is recorded in the 2026-10-05 bundle: at -O1 both arms of the original probe
 * were silent because the dead stores were optimised away, so the control could
 * not fire and "ASan is blind" would have been an artefact of the compiler
 * rather than a fact about ASan.
 *
 * Allocation here is plain malloc, NOT the port's arena. That is deliberate: the
 * corpus's own driver hands av_malloc one arena, so inside a case binary ASan
 * sees a single allocation and could not see an allocation bound even in
 * principle. A silence there would prove nothing. With malloc, ASan has a real
 * redzone at the allocation's edge -- which the `past` arm demonstrates -- so the
 * other arms' silence is about intra-object granularity and nothing else.
 *
 *   ./asan-probe-7 <arm>
 *
 * arms: member member-read underflow walk copy slice past
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One struct carrying every adjacency the seven cases need, so each arm crosses
 * a bound inside THIS allocation. */
struct shapes {
  uint8_t last[64];   /* case 3: read one past, into next member      */
  int last_len;       /* case 3's neighbour                           */
  uint32_t tile[16];  /* cases 5, 6, 8: write past, into next member  */
  uint32_t tail[8];   /* their neighbour                              */
  char dst[32];       /* case 7: copy sized by the source length      */
  char dst_next[32];  /* its neighbour                                */
  int hist[32];       /* case 9: carved sub-slice, 8 entries per slice */
};

/* The allocation is FREED before the report. Without that, LeakSanitizer fires
 * at exit and its summary line also contains the string "AddressSanitizer" --
 * so a detector grepping for that string calls every arm a detection, including
 * the six that are silent on bounds. That happened on the first run of this
 * probe. Key any check on `heap-buffer-overflow`, never on "AddressSanitizer". */
static struct shapes *g_s;
static int report(const char *arm, const char *what, unsigned long v) {
  free(g_s);
  g_s = NULL;
  printf("ARM %s %s=%lu\n", arm, what, v);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <arm>\n", argv[0]);
    return 75;
  }
  const char *arm = argv[1];
  volatile struct shapes *s = calloc(1, sizeof *s);
  if (!s)
    return 75;
  g_s = (struct shapes *)s;

  if (!strcmp(arm, "member")) {
    /* case 5/6/8 shape: WRITE one element past an array member. */
    s->tail[0] = 0xA5A5A5A5u;
    s->tile[16] = 0x41414141u;
    return report(arm, "neighbour", s->tail[0]);
  }
  if (!strcmp(arm, "member-read")) {
    /* case 3 shape: READ one element past an array member. */
    s->last_len = 0x5A5A5A5A;
    unsigned v = s->last[64];
    return report(arm, "read", v);
  }
  if (!strcmp(arm, "underflow")) {
    /* case 4 shape: index UNDERFLOWS out of the start of a member. `dst` is not
     * the first member, so dst[-1] is the last byte of tail[7] and is interior
     * to the allocation -- which is the whole point of case 4's prediction. The
     * sentinel goes on tail[7] so the byte read is attributable to it. */
    s->tail[7] = 0x5A5A5A5Au;
    size_t len = 0; /* strlen("") */
    volatile char *p = (volatile char *)s->dst;
    unsigned v = (unsigned char)p[len - 1]; /* dst + SIZE_MAX == dst - 1 */
    return report(arm, "read", v);
  }
  if (!strcmp(arm, "walk")) {
    /* case 6/8 shape: an UNBOUNDED loop walks off the member. */
    for (unsigned i = 0; i < 8; i++)
      s->tail[i] = 0xA5A5A5A5u;
    unsigned past = 0;
    for (unsigned j = 0; j < 16 + 8; j++) {
      s->tile[j] = j;
      if (j >= 16)
        past++;
    }
    return report(arm, "past", past);
  }
  if (!strcmp(arm, "copy")) {
    /* case 7 shape: a copy sized by its SOURCE length overruns the member. */
    char src[48];
    memset(src, 'A', sizeof src - 1);
    src[sizeof src - 1] = '\0';
    memset((void *)s->dst_next, 0x5A, sizeof s->dst_next);
    size_t span = strlen(src); /* 47, larger than dst's 32 */
    memcpy((void *)s->dst, src, span);
    unsigned past = 0;
    for (unsigned i = 0; i < sizeof s->dst_next; i++)
      if (s->dst_next[i] == 'A')
        past++;
    return report(arm, "past", past);
  }
  if (!strcmp(arm, "slice")) {
    /* case 9 shape: a CARVED sub-slice crossed by a data-controlled index. The
     * slice boundary is arithmetic -- 8 entries per slice inside hist[32] -- so
     * nothing declares it. */
    volatile int *slice = &s->hist[0];
    unsigned idx = 20; /* past the 8-entry slice, inside hist[32] */
    slice[idx] = 1;
    return report(arm, "bumped", (unsigned long)slice[idx]);
  }
  if (!strcmp(arm, "past")) {
    /* THE POSITIVE CONTROL. One uint32 past the whole allocation. If this arm is
     * silent, the probe proves nothing and every other arm's silence is void. */
    volatile uint32_t *p = (volatile uint32_t *)((char *)s + sizeof *s);
    *p = 0x41414141u;
    return report(arm, "wrote_past_allocation", (unsigned long)*p);
  }
  fprintf(stderr, "unknown arm %s\n", arm);
  return 75;
}
