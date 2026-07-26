#ifndef LADDER_PERF_DOMAIN_H
#define LADDER_PERF_DOMAIN_H
/* Board-perf variant of a silicon-ladder rung's domain_main.
 *
 * The QEMU ladder's <rung>_app.c does `*res = <rung>_compute()`. For the FPGA
 * perf run we additionally record the workload's cycle cost, so this domain_main
 * brackets the compute with mcycle reads and writes both values into the shared
 * region the controller reads back:
 *   res[0] = retval  (the checksum; controller checks it == the native oracle)
 *   res[1] = cycles  (mcycle delta across <rung>_compute() only)
 *   res[2] = 0xD09E  (ran-marker, so the controller can tell a real run from
 *                     an all-zero region)
 *
 * The entry glue (start-gp-captable-generic.S) delivers the shared region cap as
 * domain_main's first argument, exactly as the QEMU path does, so the same
 * gp-captable silicon build works on both. `res` is that argument cap (reached
 * directly, not via gp); the rung's own globals are reached via gp[i] as usual.
 *
 * mcycle vs rdcycle: the board GATES the unprivileged `cycle` counter
 * (counteren.CY off for the domain), so we read the M-mode `mcycle` CSR (0xB00),
 * which the on-board setup leaves domain-readable (confirmed by the borrow-cost
 * board runs). See tests/rtl-smoke/fpga_instrument.h for the full rationale.
 *
 * Usage (per rung): `#define LADDER_COMPUTE <rung>_compute` then
 * `#include "ladder_perf_domain.h"`, after the rung's kernel header. */

#ifndef LADDER_COMPUTE
#error "define LADDER_COMPUTE to the rung's compute function before including"
#endif

static inline unsigned long ladder_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

/* minstret (0xB02), read for the same reason mcycle is: the board gates the
   unprivileged `instret` for the domain. Pairing it with mcycle is what splits the
   measured capability overhead into "the ABI retires MORE instructions" (every
   global goes through `ldc rd, i*16(gp)`) versus "the same instructions cost MORE
   cycles". A cycle count alone conflates the two, and the plain-RISC-V baseline
   half already reports instret -- only this side was missing. */
static inline unsigned long ladder_rd_minstret(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, minstret" : "=r"(v));
  return v;
}

/* Slots clear of the debug window: the controller prints res[3 .. 3+LADDER_DBG_SLOTS)
   as dbg0.., currently reaching res[47], and gp_diag3's seeded data windows occupy
   res[32..48). So instret cannot simply take res[3]. */
#define LADDER_INSTRET_SLOT 64
#define LADDER_PHASE_SLOT   65

/* LADDER_INSTR_MODE -- the bisection knob.
 *
 * Adding the minstret instrumentation flipped beebs_prime from returning the
 * correct oracle to miscomputing on silicon, confirmed by a controlled A/B (mode 4
 * wrong and deterministic, mode 0 correct). The instrumentation is three separable
 * constructs, so building each ALONE says which one is the trigger -- something no
 * gp_diag variant could do, because gp_diag3's fault vanishes when perturbed while
 * both of these variants are tiny and freely modifiable.
 *
 *   0  none                  -- the pre-26-07 body, mcycle only          (PASSES)
 *   1  phase stores only     -- two stores of a constant to res[65]
 *   2  minstret reads only   -- two `csrr minstret`, no region store
 *   3  instret store only    -- one store of a constant to res[64]
 *   4  full (default)        -- all three, i.e. the shipping instrumentation (FAILS)
 *   5  one store to res[3]   -- LOW offset 0x18, otherwise identical to mode 3
 *   6  one store to res[32]  -- MID offset 0x100, otherwise identical to mode 3
 *   7  ENTRY PROBE           -- everything EXCEPT the compute actually runs
 *
 * Mode 7 answers a different question from 1-6: `matmult_int` and `coremark_matrix`
 * produce no END marker at all, and that was written up as "a domain-entry fault".
 * But a domain that enters correctly and then spins forever inside the compute is
 * externally IDENTICAL -- no output either way -- so entry was never actually
 * localized, only assumed. The distinction decides whether the domain-boundary
 * `fence.i` (an entry-side fix) is even the right patch.
 *
 * The obvious probe -- store a marker and read it back -- does NOT work here: the
 * controller only reports res[] after the cscall returns, so a hang hides the
 * marker too. Mode 7 instead makes the ANSWER be "does it return at all":
 * everything on the entry path runs unchanged (glue, gp-captable build, region
 * cap delivery, cscall/csreturn), but the compute is placed behind a branch that
 * is never taken.
 *
 *   returns (END marker appears) => entry path is FINE for this exact binary
 *                                   => the hang is INSIDE the compute
 *   still hangs                   => the hang is at/before entry
 *
 * This localizes the LAYER only, never the culprit. "Inside the compute" does not
 * mean "a compiler bug": the compute runs correctly under QEMU with the identical
 * binary, so whatever it is, it is a silicon divergence -- either codegen that hits
 * an RTL corner QEMU does not model, or an RTL bug outright. Do not collapse the two.
 *
 * `func` is the gate because it is genuinely opaque: nothing passes it (the
 * controller sets no func, domain_main is called from assembly glue), so the
 * optimizer cannot fold the branch and delete the call. That matters -- the
 * compute must stay LINKED so its globals keep their gp-captable slots and the
 * image layout stays as close to the hanging build as possible. Gating on `res`
 * instead would be unsafe: after the stores above, -O1 may assume res != NULL and
 * eliminate the call outright.
 *
 * retval comes back 0 in this mode, so the controller will flag an oracle
 * mismatch. That is expected and irrelevant -- the only signal is END vs no END.
 *
 * Modes 5-6 exist because the mode 1/2/3 board results narrowed the trigger to "an
 * extra STORE through the shared-region capability" (mode 2 adds 104 B of CSR reads
 * and PASSES; mode 3 adds 16 B containing one store and FAILS). Two hypotheses
 * survive: the store's OFFSET matters (res[0..2] at 0x0/0x8/0x10 are present in the
 * passing control, while the failing stores are at 0x200/0x208), or merely having
 * one MORE region store matters regardless of where. A low-offset store separates
 * them in one boot.
 *
 * Modes 1-3 deliberately store CONSTANTS, not real counter values: the question is
 * which construct perturbs codegen, not what the counters read. Keeping the stored
 * value constant across modes means the only variable is the construct itself.
 *
 * -DLADDER_NO_MINSTRET is kept as a spelling of mode 0 so the existing
 * beebs_prime_noins control rung builds unchanged. */
#ifdef LADDER_NO_MINSTRET
#  define LADDER_INSTR_MODE 0
#endif
#ifndef LADDER_INSTR_MODE
#  define LADDER_INSTR_MODE 4
#endif
void domain_main(unsigned long *res, unsigned func) {
  (void)func;
#if LADDER_INSTR_MODE == 7
  /* Entry probe: run the whole entry path, skip only the compute. See above. */
  res[LADDER_PHASE_SLOT] = 0xE117UL;
  unsigned v7 = 0u;
  if (func == 0xDEADBEEFu) v7 = LADDER_COMPUTE();   /* never taken; keeps it linked */
  res[0] = (unsigned long)v7;
  res[1] = 0UL;
  res[2] = 0xD09EUL;
}
#else
#if LADDER_INSTR_MODE == 1 || LADDER_INSTR_MODE == 4
  /* Whether minstret is domain-readable here is NOT established -- only mcycle is
     (confirmed by the borrow-cost board runs). A gated CSR faults, halting the
     domain, which looks exactly like every other silent failure. So record the
     phase first: if the host reads back PHASE=1 with the rest zero, minstret is
     the culprit and a failed boot still says something. */
  res[LADDER_PHASE_SLOT] = 1UL;
#endif
#if LADDER_INSTR_MODE == 2 || LADDER_INSTR_MODE == 4
  unsigned long i0 = ladder_rd_minstret();
#endif
#if LADDER_INSTR_MODE == 1 || LADDER_INSTR_MODE == 4
  res[LADDER_PHASE_SLOT] = 2UL;
#endif

  unsigned long c0 = ladder_rd_mcycle();
  unsigned v = LADDER_COMPUTE();
  unsigned long c1 = ladder_rd_mcycle();
#if LADDER_INSTR_MODE == 2 || LADDER_INSTR_MODE == 4
  unsigned long i1 = ladder_rd_minstret();
#endif

  res[0] = (unsigned long)v;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
#if LADDER_INSTR_MODE == 4
  res[LADDER_INSTRET_SLOT] = i1 - i0;
#elif LADDER_INSTR_MODE == 3
  res[LADDER_INSTRET_SLOT] = 0x1640UL;      /* constant: isolate the store itself */
#elif LADDER_INSTR_MODE == 2
  (void)i0; (void)i1;                       /* reads only; nothing stored */
#elif LADDER_INSTR_MODE == 5
  res[3] = 0x1640UL;                        /* LOW offset 0x18 */
#elif LADDER_INSTR_MODE == 6
  res[32] = 0x1640UL;                       /* MID offset 0x100 */
#endif
}
#endif /* LADDER_INSTR_MODE == 7 */
#endif
