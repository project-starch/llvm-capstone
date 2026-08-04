#ifndef XGM_H
#define XGM_H
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

static struct xg_s xgm_arr[9];
static struct xg_s *xgm_head;
static unsigned xgm_compute(void)
{
  int i, k = 0, n = 0; struct xg_s *p;
  unsigned long snap[10];
  xgm_head = 0;
  for (i = 0; i < 9; i++) { xgm_arr[i].v = i; xgm_arr[i].next = xgm_head; xgm_head = &xgm_arr[i]; }
  for (p = xgm_head; p; p = p->next) { n++; snap[k++] = (unsigned long)(unsigned)n; }
  for (i = 0; i < k; i++)
    if (snap[i] & 0x08000000UL) return (unsigned)(500 + i);   /* poisoned DURING iteration i */
  return (unsigned)(600 + (((unsigned)n & 0x08000000u) ? 1u : 0u));
  /* 609 impossible; 600 = never poisoned at all, 601 = clean during the walk but poisoned
     at the FINAL read -- which would put the fault in the post-loop reload, not the loop. */
}
#endif
