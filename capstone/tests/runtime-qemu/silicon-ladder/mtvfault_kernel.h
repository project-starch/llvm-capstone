#ifndef MTVFAULT_KERNEL_H
#define MTVFAULT_KERNEL_H
/* Fault control for the INTERP_DOMAIN_MTVEC trap handler.
 *
 * This rung is an INSTRUMENT VALIDATOR, not a test of the silicon. It exists because
 * "stage 10 never returns" is uninterpretable on its own: a domain that faults with a
 * broken handler and a domain that genuinely wedges look identical from the console.
 * Until a handler is shown to FIRE on a real fault, "still hangs" means nothing.
 *
 * The domain writes this value, then takes a deliberate capability fault. Three outcomes,
 * all distinguishable:
 *
 *   17     -> the fault was taken AND the handler returned. Instrument works.
 *   2989   -> no fault occurred at all (the faulting op is legal here). Instrument is
 *             invalid for a different reason, and says so instead of lying.
 *   hang   -> the handler did not recover. Any stage verdict from that build is void.
 *
 * The host runs the same header without faulting, so the oracle is just the value.
 */
/* `volatile`, and a global rather than a literal, for two independent reasons:
   the build gate requires a real `ldc gp[i]` cap-table access (a rung with no global
   access is not exercising the silicon config at all, and the builder rejects it), and
   at -O1 a non-volatile static read here folds to a constant and the access disappears. */
static volatile unsigned mtvfault_val = 17u;
static inline unsigned mtvfault_expect(void) { return mtvfault_val; }
#endif
