#ifndef XGF_H
#define XGF_H
/* Auditor-driven controls for the bit-27 finding. Three of my claims were unsupported:
 *   - every xg* rung ran at LADDER_INSTR_MODE=4, and ladder_perf_domain.h:55-58 records that
 *     this very instrumentation once flipped beebs_prime from correct to a DETERMINISTIC
 *     silicon miscompute (mode 4 wrong, mode 0 correct) -- the exact signature here, and no
 *     mode-0 control was ever run;
 *   - xgw was offered as "same construct, different slot" but s0-0x28 there holds `i`, which
 *     is never read, so it never tested the accused slot;
 *   - nothing separates "the memory word is poisoned" from "the lw delivers a wrong value".
 * struct/globals identical to xgn in every arm. */
struct xg_s { struct xg_s *next; int v; };

static struct xg_s xgf_arr[9];
static struct xg_s *xgf_head;
static unsigned xgf_compute(void)
{
  volatile unsigned long pad = 0x5a5a5a5aUL; (void)pad;
  int i, n = 0; struct xg_s *p;
  xgf_head = 0;
  for (i = 0; i < 9; i++) { xgf_arr[i].v = i; xgf_arr[i].next = xgf_head; xgf_head = &xgf_arr[i]; }
  for (p = xgf_head; p; p = p->next) n++;
  return (unsigned)n;
}
#endif
