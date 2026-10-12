/* LIFETIME CONTROL 3b, the stale reference does not stay usable: the
 * positive control. Program 02's sequence exactly (lifetime.h,
 * wm_ctl_successor: A released with wmem_free, C handed out at A's address,
 * C written and read back), then ONE thing more: A's whole former range is
 * read through the old A pointer at the corpus's labelled read probe
 * (wm_probe, shared/driver.c, label wm_defect_probe), after the ready mark.
 *
 * Predicted: on the protected build the first read faults at the probe with
 * cause 25 (A's lifetime was revoked by wmem_free); on the release build every
 * read completes and returns C's bytes, so the protected fault is the
 * patch's. */
#include "../lifetime.h"

WM_CASE(3) {
  unsigned char *stale;
  unsigned char *c = wm_ctl_successor(&stale);
  unsigned char got[WM_CTL_N];
  printf("WM_CTL access read stale-A via=A addr=%#lx len=%lu "
         "probe=wm_defect_probe site=case.c:%d\n",
         WM_CTL_CURSOR(stale), WM_CTL_N, __LINE__ + 3);
  wm_held = stale;
  wm_mark();
  got[0] = (unsigned char)wm_probe(wm_held); /* the stale read */
  for (unsigned long i = 1; i < WM_CTL_N; i++)
    got[i] = (unsigned char)wm_probe(wm_held + i);
  printf("WM_CTL done read stale-A\n");
  wm_ctl_expect("stale-A-reads-C", got, 0, WM_CTL_N, wm_ctl_pattern_c);
  wm_ctl_expect("stale-A-reads-A", got, 0, WM_CTL_N, wm_ctl_pattern_a);
  printf("WM_CTL stale-A first_byte=%#x C_first_byte=%#x\n", got[0],
         (unsigned)c[0]);
}
