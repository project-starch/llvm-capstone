#ifndef CAPINIT_H
#define CAPINIT_H
/* MINIMAL REPRO for CAPABLITY_OUT_OF_BOUND raised inside __capstone_cap_init.
 *
 * TWO INGREDIENTS. An earlier version had only the first, PASSED on silicon, and proved
 * nothing -- recorded here so it is not retried:
 *
 *   1. MORE THAN 32 pointer-valued global initialisers. That is the threshold at which the
 *      initializer function is now split into separate basic blocks.
 *   2. STORE OFFSETS BEYOND THE 12-BIT IMMEDIATE. The pointers are 2064 bytes apart, so an
 *      offset cannot sit in the `stc` immediate: the compiler materialises it in a register
 *      and emits `cincoffset rd, holder, rN`. That is the exact form of the faulting
 *      instruction in the SQLite domain, whose store was at holder+0x850.
 *
 * The first attempt used 64 TIGHTLY PACKED pointers: offsets stayed under 1024, every store
 * used a plain immediate, and both arms returned the right answer on the board.
 *
 * Only FOUR pointers are verified rather than all 40: the checks are unrolled and the point
 * is to keep .text inside the rung's 4 KiB window, which an all-40 check overflowed. The
 * read-back goes THROUGH the initialised pointer, so a run that skipped initialisation
 * cannot be mistaken for one that did it right. */

#define CAPINIT_N 40u

static long capinit_target[CAPINIT_N];

struct capinit_h {
  char pad0[2048]; void *p0;
  char pad1[2048]; void *p1;
  char pad2[2048]; void *p2;
  char pad3[2048]; void *p3;
  char pad4[2048]; void *p4;
  char pad5[2048]; void *p5;
  char pad6[2048]; void *p6;
  char pad7[2048]; void *p7;
  char pad8[2048]; void *p8;
  char pad9[2048]; void *p9;
  char pad10[2048]; void *p10;
  char pad11[2048]; void *p11;
  char pad12[2048]; void *p12;
  char pad13[2048]; void *p13;
  char pad14[2048]; void *p14;
  char pad15[2048]; void *p15;
  char pad16[2048]; void *p16;
  char pad17[2048]; void *p17;
  char pad18[2048]; void *p18;
  char pad19[2048]; void *p19;
  char pad20[2048]; void *p20;
  char pad21[2048]; void *p21;
  char pad22[2048]; void *p22;
  char pad23[2048]; void *p23;
  char pad24[2048]; void *p24;
  char pad25[2048]; void *p25;
  char pad26[2048]; void *p26;
  char pad27[2048]; void *p27;
  char pad28[2048]; void *p28;
  char pad29[2048]; void *p29;
  char pad30[2048]; void *p30;
  char pad31[2048]; void *p31;
  char pad32[2048]; void *p32;
  char pad33[2048]; void *p33;
  char pad34[2048]; void *p34;
  char pad35[2048]; void *p35;
  char pad36[2048]; void *p36;
  char pad37[2048]; void *p37;
  char pad38[2048]; void *p38;
  char pad39[2048]; void *p39;
};

static struct capinit_h capinit_holder = {
  {0}, &capinit_target[0],
  {0}, &capinit_target[1],
  {0}, &capinit_target[2],
  {0}, &capinit_target[3],
  {0}, &capinit_target[4],
  {0}, &capinit_target[5],
  {0}, &capinit_target[6],
  {0}, &capinit_target[7],
  {0}, &capinit_target[8],
  {0}, &capinit_target[9],
  {0}, &capinit_target[10],
  {0}, &capinit_target[11],
  {0}, &capinit_target[12],
  {0}, &capinit_target[13],
  {0}, &capinit_target[14],
  {0}, &capinit_target[15],
  {0}, &capinit_target[16],
  {0}, &capinit_target[17],
  {0}, &capinit_target[18],
  {0}, &capinit_target[19],
  {0}, &capinit_target[20],
  {0}, &capinit_target[21],
  {0}, &capinit_target[22],
  {0}, &capinit_target[23],
  {0}, &capinit_target[24],
  {0}, &capinit_target[25],
  {0}, &capinit_target[26],
  {0}, &capinit_target[27],
  {0}, &capinit_target[28],
  {0}, &capinit_target[29],
  {0}, &capinit_target[30],
  {0}, &capinit_target[31],
  {0}, &capinit_target[32],
  {0}, &capinit_target[33],
  {0}, &capinit_target[34],
  {0}, &capinit_target[35],
  {0}, &capinit_target[36],
  {0}, &capinit_target[37],
  {0}, &capinit_target[38],
  {0}, &capinit_target[39]
};

static unsigned capinit_compute(void)
{
  capinit_target[0] = 0+1;
  if (capinit_holder.p0 != (void *)&capinit_target[0]) return 0xBADu;
  if (*(long *)capinit_holder.p0 != (long)(0+1)) return 0xBADu;
  capinit_target[13] = 13+1;
  if (capinit_holder.p13 != (void *)&capinit_target[13]) return 0xBADu;
  if (*(long *)capinit_holder.p13 != (long)(13+1)) return 0xBADu;
  capinit_target[27] = 27+1;
  if (capinit_holder.p27 != (void *)&capinit_target[27]) return 0xBADu;
  if (*(long *)capinit_holder.p27 != (long)(27+1)) return 0xBADu;
  capinit_target[39] = 39+1;
  if (capinit_holder.p39 != (void *)&capinit_target[39]) return 0xBADu;
  if (*(long *)capinit_holder.p39 != (long)(39+1)) return 0xBADu;
  return CAPINIT_N;
}
#endif
