#ifndef XGA_H
#define XGA_H
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

static struct xg_s xga_arr[9];
static struct xg_s *xga_head;
static unsigned xga_compute(void)
{
  int i, n = 0; struct xg_s *p;
  unsigned long st = 0, en = 0, addr;
  LCCF(st, 3);                       /* stack capability start */
  LCCF(en, 4);                       /* stack capability end   */
  xga_head = 0;
  for (i = 0; i < 9; i++) { xga_arr[i].v = i; xga_arr[i].next = xga_head; xga_head = &xga_arr[i]; }
  for (p = xga_head; p; p = p->next) n++;
  addr = (unsigned long)(void *)&n;
  /* 41x = counter slot BELOW the carve, 42x = ABOVE it, 43x = inside.
     x = 1 if the counter read back poisoned, 0 if clean. */
  { unsigned band = (addr < st) ? 410u : (addr >= en) ? 420u : 430u;
    return band + (((unsigned)n & 0x08000000u) ? 1u : 0u); }
}
#endif
