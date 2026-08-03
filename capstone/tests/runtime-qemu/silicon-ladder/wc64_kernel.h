#ifndef WC64_H
#define WC64_H
/* R-16 scale probe for __capstone_cap_init. wcap (2 capability-bearing initialised
 * globals) RETURNS -62 instead of 4 on silicon -- i.e. both pointers come back NULL --
 * while QEMU returns 4 from the same binary. Every other axis is ruled out by a
 * one-variable pair (image size, carve count, their conjunction, dom_data geometry, blob
 * size, and the loader). This scales the SAME construct toward SQLite's population to see
 * whether 'returns NULL' becomes 'hangs' -- which is what R-16 looks like.
 * Returns the number of non-NULL pointers; expect 64. A silicon count BELOW 64 localises
 * how many survive; a hang means the stall reproduces at rung scale. */
static char wc64_data[64] = { 'A', 0 };
static char *wc64_p0 = wc64_data + 0;
static char *wc64_p1 = wc64_data + 1;
static char *wc64_p2 = wc64_data + 2;
static char *wc64_p3 = wc64_data + 3;
static char *wc64_p4 = wc64_data + 4;
static char *wc64_p5 = wc64_data + 5;
static char *wc64_p6 = wc64_data + 6;
static char *wc64_p7 = wc64_data + 7;
static char *wc64_p8 = wc64_data + 8;
static char *wc64_p9 = wc64_data + 9;
static char *wc64_p10 = wc64_data + 10;
static char *wc64_p11 = wc64_data + 11;
static char *wc64_p12 = wc64_data + 12;
static char *wc64_p13 = wc64_data + 13;
static char *wc64_p14 = wc64_data + 14;
static char *wc64_p15 = wc64_data + 15;
static char *wc64_p16 = wc64_data + 16;
static char *wc64_p17 = wc64_data + 17;
static char *wc64_p18 = wc64_data + 18;
static char *wc64_p19 = wc64_data + 19;
static char *wc64_p20 = wc64_data + 20;
static char *wc64_p21 = wc64_data + 21;
static char *wc64_p22 = wc64_data + 22;
static char *wc64_p23 = wc64_data + 23;
static char *wc64_p24 = wc64_data + 24;
static char *wc64_p25 = wc64_data + 25;
static char *wc64_p26 = wc64_data + 26;
static char *wc64_p27 = wc64_data + 27;
static char *wc64_p28 = wc64_data + 28;
static char *wc64_p29 = wc64_data + 29;
static char *wc64_p30 = wc64_data + 30;
static char *wc64_p31 = wc64_data + 31;
static char *wc64_p32 = wc64_data + 32;
static char *wc64_p33 = wc64_data + 33;
static char *wc64_p34 = wc64_data + 34;
static char *wc64_p35 = wc64_data + 35;
static char *wc64_p36 = wc64_data + 36;
static char *wc64_p37 = wc64_data + 37;
static char *wc64_p38 = wc64_data + 38;
static char *wc64_p39 = wc64_data + 39;
static char *wc64_p40 = wc64_data + 40;
static char *wc64_p41 = wc64_data + 41;
static char *wc64_p42 = wc64_data + 42;
static char *wc64_p43 = wc64_data + 43;
static char *wc64_p44 = wc64_data + 44;
static char *wc64_p45 = wc64_data + 45;
static char *wc64_p46 = wc64_data + 46;
static char *wc64_p47 = wc64_data + 47;
static char *wc64_p48 = wc64_data + 0;
static char *wc64_p49 = wc64_data + 1;
static char *wc64_p50 = wc64_data + 2;
static char *wc64_p51 = wc64_data + 3;
static char *wc64_p52 = wc64_data + 4;
static char *wc64_p53 = wc64_data + 5;
static char *wc64_p54 = wc64_data + 6;
static char *wc64_p55 = wc64_data + 7;
static char *wc64_p56 = wc64_data + 8;
static char *wc64_p57 = wc64_data + 9;
static char *wc64_p58 = wc64_data + 10;
static char *wc64_p59 = wc64_data + 11;
static char *wc64_p60 = wc64_data + 12;
static char *wc64_p61 = wc64_data + 13;
static char *wc64_p62 = wc64_data + 14;
static char *wc64_p63 = wc64_data + 15;
static unsigned wc64_compute(void)
{
  unsigned ok = 0;
  if (wc64_p0) ok++;
  if (wc64_p1) ok++;
  if (wc64_p2) ok++;
  if (wc64_p3) ok++;
  if (wc64_p4) ok++;
  if (wc64_p5) ok++;
  if (wc64_p6) ok++;
  if (wc64_p7) ok++;
  if (wc64_p8) ok++;
  if (wc64_p9) ok++;
  if (wc64_p10) ok++;
  if (wc64_p11) ok++;
  if (wc64_p12) ok++;
  if (wc64_p13) ok++;
  if (wc64_p14) ok++;
  if (wc64_p15) ok++;
  if (wc64_p16) ok++;
  if (wc64_p17) ok++;
  if (wc64_p18) ok++;
  if (wc64_p19) ok++;
  if (wc64_p20) ok++;
  if (wc64_p21) ok++;
  if (wc64_p22) ok++;
  if (wc64_p23) ok++;
  if (wc64_p24) ok++;
  if (wc64_p25) ok++;
  if (wc64_p26) ok++;
  if (wc64_p27) ok++;
  if (wc64_p28) ok++;
  if (wc64_p29) ok++;
  if (wc64_p30) ok++;
  if (wc64_p31) ok++;
  if (wc64_p32) ok++;
  if (wc64_p33) ok++;
  if (wc64_p34) ok++;
  if (wc64_p35) ok++;
  if (wc64_p36) ok++;
  if (wc64_p37) ok++;
  if (wc64_p38) ok++;
  if (wc64_p39) ok++;
  if (wc64_p40) ok++;
  if (wc64_p41) ok++;
  if (wc64_p42) ok++;
  if (wc64_p43) ok++;
  if (wc64_p44) ok++;
  if (wc64_p45) ok++;
  if (wc64_p46) ok++;
  if (wc64_p47) ok++;
  if (wc64_p48) ok++;
  if (wc64_p49) ok++;
  if (wc64_p50) ok++;
  if (wc64_p51) ok++;
  if (wc64_p52) ok++;
  if (wc64_p53) ok++;
  if (wc64_p54) ok++;
  if (wc64_p55) ok++;
  if (wc64_p56) ok++;
  if (wc64_p57) ok++;
  if (wc64_p58) ok++;
  if (wc64_p59) ok++;
  if (wc64_p60) ok++;
  if (wc64_p61) ok++;
  if (wc64_p62) ok++;
  if (wc64_p63) ok++;
  return ok;
}
#endif
