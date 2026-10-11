#include "corpus.h"
#include <stdint.h>

/* AACArithState exactly as aacdec_ac.h:27-32 declares it at the pin, and as the
 * fix 89de2f0de1 changes it. The whole struct is ONE allocation -- it is a member
 * of AACDecContext (aacdec.h:156), which is avctx->priv_data -- so the bound
 * between `last` and `last_len` is not a bound any allocator knows about.
 *
 * The fix grows the array by one element rather than clamping the index, so the
 * two arms differ by exactly that term and the consumer below is byte-identical
 * between them. */
#define LAST 512 /* 2048 / 4 */

struct arith_state_buggy {
  uint8_t last[LAST];
  int last_len;
  uint8_t cur[4];
  uint16_t state_pre;
};
struct arith_state_fixed {
  uint8_t last[LAST + 1]; /* the fix: + 1 */
  int last_len;
  uint8_t cur[4];
  uint16_t state_pre;
};

/* ff_aac_ac_get_context, reduced to the one term that crosses (aacdec_ac.c:60):
 *
 *     c = c + (state->last[i + 1] << 8);
 *
 * The caller walks i over the window, so i reaches LAST-1 and the index reaches
 * LAST. Everything else in that function is arithmetic on values already read,
 * and is left out because it cannot move the access. */
#define CONTEXT_TERM(st, i) ((unsigned)(st)->last[(i) + 1] << 8)

FF2_CASE(3) {
  /* Case 3 -- AAC arithmetic-coding context, fix 89de2f0de1. SUB-OBJECT: a
   * one-byte read from the end of one struct member into the next, inside a
   * single allocation.
   *
   * aacdec_ac.c:60 reads state->last[i + 1]. The window length N is 2048/4 =
   * 512 = LAST, so the last iteration has i == LAST-1 and reads last[LAST].
   * Index 512 of a uint8_t[512] sits at byte offset 512, which is 4-aligned and
   * is exactly where `last_len` begins -- so the read takes that member's
   * lowest byte. The fix declares last[512 + 1], which makes index 512 a real
   * array element whose value is the memset-zero the decoder already does.
   *
   * NOTHING WE HAVE CAN CATCH THIS, which is the row's purpose: the crossing is
   * inside one allocation, so a per-allocation bound is in bounds for it. The
   * oracle is the upstream fix, not a protection. */
  const int sentinel_byte = 0x5A;
  const int last_len_sentinel = 0x5A5A5A5A; /* every byte distinct from 0 */

  unsigned read_at_LAST;
  unsigned long cap;

  if (fixed) {
    struct arith_state_fixed *st = av_refstruct_allocz(sizeof *st);
    CHECK(st, 611);
    /* The offsets are asserted, not assumed: under the FIX index LAST must be
     * inside `last` and must NOT alias last_len. */
    CHECK((char *)&st->last[LAST] < (char *)&st->last_len, 612);
    CHECK(sizeof st->last == LAST + 1, 613);
    st->last_len = last_len_sentinel;
    read_at_LAST = CONTEXT_TERM(st, LAST - 1);
    cap = sizeof st->last;
    av_refstruct_unref(&st);
  } else {
    struct arith_state_buggy *st = av_refstruct_allocz(sizeof *st);
    CHECK(st, 614);
    /* The whole claim of this case: index LAST of the first member IS the first
     * byte of the second. */
    CHECK((char *)&st->last[LAST] == (char *)&st->last_len, 615);
    CHECK(sizeof st->last == LAST, 616);
    st->last_len = last_len_sentinel;
    read_at_LAST = CONTEXT_TERM(st, LAST - 1);
    cap = sizeof st->last;
    av_refstruct_unref(&st);
  }

  /* The buggy arm must have read last_len's low byte; the fixed arm must have
   * read a zeroed element of its own array. Both are checked, so neither arm
   * can pass by the access simply not happening. */
  const unsigned from_neighbour = (unsigned)sentinel_byte << 8;
  int took_neighbour = read_at_LAST == from_neighbour;
  int took_own_element = read_at_LAST == 0;

  printf("cap=%lu touched=%d term=0x%04x\n", cap, LAST, read_at_LAST);

  FF2_VERDICT(!fixed && took_neighbour && cap == LAST,
              fixed && took_own_element && cap == LAST + 1,
              "index 512 of last[512] read the low byte of last_len, inside one "
              "allocation, and fed it into the arithmetic-coding context",
              "the fix's last[512 + 1] makes index 512 an element of its own member");
  return !fixed ? !took_neighbour : !took_own_element;
}
