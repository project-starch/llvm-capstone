#ifndef XGK_H
#define XGK_H
/* Follow-ups to the bit-27 finding (xgn = 9 + 2^27). Three questions:
 *   xga  is the poisoned stack slot INSIDE the domain's carved stack capability?
 *        If the frame runs past the carve, accesses land outside it -- and because the LSU
 *        capability check is inert on this bitstream, that would corrupt SILENTLY. That
 *        would make this OUR bug (frame vs carve), not the hardware's.
 *   xgm  WHEN is the counter poisoned -- during the walk, or only at the final read?
 *   xgk  does anything ADJACENT to the capability spill slot get damaged too?
 * Every arm returns a distinct band so no arm can be mistaken for another. */
#define LCCF(rd, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, sp, x" #f : "=r"(rd))
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgk_arr[9];
static struct xg_s *xgk_head;
static unsigned xgk_compute(void)
{
  volatile unsigned c0 = 0x11111111u, c1 = 0x22222222u, c2 = 0x33333333u;
  int i, n = 0; struct xg_s *p;
  unsigned bad = 0;
  xgk_head = 0;
  for (i = 0; i < 9; i++) { xgk_arr[i].v = i; xgk_arr[i].next = xgk_head; xgk_head = &xgk_arr[i]; }
  for (p = xgk_head; p; p = p->next) n++;
  if (c0 != 0x11111111u) bad |= 1u;      /* canaries around the capability spill slot */
  if (c1 != 0x22222222u) bad |= 2u;
  if (c2 != 0x33333333u) bad |= 4u;
  if (((unsigned)n & 0x08000000u) != 0u) bad |= 8u;
  return 700u + bad;                     /* 700 = nothing damaged; 708 = only the counter */
}
#endif
