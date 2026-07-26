#ifndef GP_DIAG_KERNEL_H
#define GP_DIAG_KERNEL_H
/* Silicon-ladder DIAGNOSTIC rung: localize the gp-captable silicon miscompute.
 *
 * WHY THIS EXISTS. Every perf rung folds its work into ONE FNV-1a checksum, so a
 * silicon miscompute yields a single opaque wrong number that says nothing about
 * WHICH mechanism broke. Checksum inversion cannot recover it either: the fold is
 * not injective, so a wide fold (e.g. matmult's 64 words) admits many preimages,
 * and for beebs_recursion (only 3 folded words) NO plausible single-word
 * corruption reproduces the observed silicon value -- i.e. the corruption is
 * broad, not one bad value.
 *
 * So this rung returns RAW values, one per hypothesis, in separate result slots.
 * One board run (one power-cycle) then tells us exactly which mechanisms are
 * broken and which are fine. Each probe is deliberately independent: a probe must
 * not depend on an earlier probe's correctness.
 *
 * Slot map (see gp_diag_fpga_app.c; controller prints res[3..11] as dbg0..dbg8):
 *   dbg0 P1 scalar global write->read via `ldc gp[i]`      expect 0x5A5A
 *   dbg1 P2 global array store->readback (single element)  expect 1234
 *   dbg2 P3 global read from inside a NOINLINE callee      expect 0x5A5A
 *   dbg3 P4 deep self-recursion (fib(10), no globals)      expect 89
 *   dbg4 P5 mutual recursion (anka(10)), no globals        expect 1
 *   dbg5 P6 array store in a loop with a LIVE accumulator  expect 28
 *   dbg6 P7 INITIALIZED global (init-template materialize) expect 36
 *   dbg7 P8 function-local static, read-before-write       expect 3 (0+1+2)
 *   dbg8 P10 counted loop, NO call in body                 expect 9
 *   dbg9 P11 counted loop, CALL in body                    expect 9
 *   dbg10 P9 canary constant (slot plumbing sanity)        expect 0xC0FFEE
 *
 * A wrong canary means the plumbing, not the compiler, is at fault -- check that
 * first. Keep this kernel SMALL: all domain code must fit the monitor's 4 KiB PCC
 * window, and initialized globals materialize as li/sd immediates in .text.
 *
 * BOARD RESULT 2026-07-25 (v1, 9 probes, driven by a LOOP):
 *   dbg0=23130 (=0x5A5A, CORRECT), dbg1..dbg8 = 0, retval = 542820029.
 * 542820029 == FNV(H0, [0x5A5A]) EXACTLY -- i.e. the driving loop executed
 * exactly ONE iteration and the value it computed was correct (mcycle 1325
 * corroborates). Meanwhile the INNER 4-byte fold loop of that same iteration ran
 * all 4 iterations correctly. The two differ in one respect: the outer loop's
 * body contains a CALL. Hence P10/P11 above, and hence v2 writes each slot with
 * STRAIGHT-LINE code (no driving loop) so a probe's value is never confounded by
 * the loop bug being measured. */

#define GPD_CANARY 0xC0FFEEu

static int gpd_scalar;          /* P1, P3 */
static int gpd_arr[8];          /* P2, P6 */
static int gpd_stat_probe;      /* P8 driver keeps its own static inside the fn */

/* P3: a real (non-inlined) callee reading a global via gp. */
__attribute__((noinline)) static int gpd_read_scalar(void) { return gpd_scalar; }

/* P11 helper: the cheapest possible non-inlined callee (touches no global, no
   array, no recursion) -- so a loop calling it differs from P10's loop ONLY by
   the presence of a call in the body. */
__attribute__((noinline)) static int gpd_one(void) { return 1; }

/* P4: deep self-recursion, touches no global. */
static int gpd_fib(int i) { return (i < 2) ? 1 : gpd_fib(i - 1) + gpd_fib(i - 2); }

/* P5: mutual recursion, touches no global. */
static int gpd_anka(int);
static int gpd_kalle(int i) { return (i <= 0) ? 0 : gpd_anka(i - 1); }
static int gpd_anka(int i) { return (i <= 0) ? 1 : gpd_kalle(i - 1); }

/* P8: function-local static read BEFORE it is ever written -- depends on the
 * cap-table builder having zeroed the carved storage. Returns 0+1+2 == 3. */
__attribute__((noinline)) static int gpd_static_acc(void) {
  static int acc;               /* zero-init expected */
  int r = acc;                  /* read-before-write on the first call */
  acc += 1;
  return r;
}

static unsigned gpd_probe(int which) {
  switch (which) {
  case 0:                                    /* P1 scalar global RW */
    gpd_scalar = 0x5A5A;
    return (unsigned)gpd_scalar;
  case 1:                                    /* P2 array store -> readback */
    gpd_arr[3] = 1234;
    return (unsigned)gpd_arr[3];
  case 2:                                    /* P3 global read in a callee */
    gpd_scalar = 0x5A5A;
    return (unsigned)gpd_read_scalar();
  case 3: return (unsigned)gpd_fib(10);      /* P4 == 89 */
  case 4: return (unsigned)gpd_anka(10);     /* P5 == 1 */
  case 5: {                                  /* P6 store + live accumulator */
    int s = 0;
    for (int i = 0; i < 8; i++) { gpd_arr[i] = i; s += gpd_arr[i]; }
    return (unsigned)s;                      /* 0+1+..+7 == 28 */
  }
  case 6: {                                  /* P7 initialized global */
    /* A function-local const array is promoted to an initialized global
       (.L__const.*), the same construct beebs_insertsort's expected[] uses. */
    const int tmpl[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int s = 0;
    for (int i = 0; i < 8; i++) s += tmpl[i];
    return (unsigned)s;                      /* 36 */
  }
  case 7: {                                  /* P8 local static, zero-init */
    int s = 0;
    for (int i = 0; i < 3; i++) s += gpd_static_acc();
    gpd_stat_probe = s;
    return (unsigned)gpd_stat_probe;         /* 0+1+2 == 3 */
  }
  case 8:                                    /* P10 counted loop, NO call in body */
    { int n = 0; for (int i = 0; i < 9; i++) n++; return (unsigned)n; }   /* 9 */
  case 9:                                    /* P11 counted loop, CALL in body */
    { int n = 0; for (int i = 0; i < 9; i++) n += gpd_one(); return (unsigned)n; } /* 9 */
  default: return GPD_CANARY;                /* P9 canary */
  }
}

/* Probe count. Slots res[3 .. 3+GPD_NPROBE) -- keep LADDER_DBG_SLOTS in
   rtl-smoke/ladder_perf_ctl.c >= this. */
#define GPD_NPROBE 11

/* Checksum form, so this rung also works with the plain ladder harness/oracle. */
static unsigned gpd_compute(void) {
  unsigned h = 2166136261u;
  for (int p = 0; p < GPD_NPROBE; p++) {
    unsigned v = gpd_probe(p);
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  return h;
}
#endif
