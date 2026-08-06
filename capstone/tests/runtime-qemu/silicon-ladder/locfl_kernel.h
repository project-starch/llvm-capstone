#ifndef LOCFL_KERNEL_H
#define LOCFL_KERNEL_H
/* MINIMAL REPRODUCER for the SQLite silicon hang, off the SQLite path entirely.
 *
 * Root-caused 2026-08-06: sqlite3RegisterBuiltinFunctions() wedges on silicon while every
 * other step of sqlite3_initialize() returns rc=0. What that function does, and nothing else
 * in initialize() does, is build a LARGE LOCAL (stack) array of structs each holding a string
 * pointer and a function pointer -- `FuncDef capstoneBuiltinFunc[]`, ~100+ elements,
 * deliberately non-const.
 *
 * Measured NOT to be the cause, so the reproducer must exclude them:
 *   - a static const aggregate of 13 function pointers copies fine (PCacheSetDefault, rc=2/3)
 *   - an indirect call through one of those pointers returns (PcacheInitialize, rc=0)
 *   - a large structured write/link pass over a 256 KB global heap returns (memsys5Init, rc=0)
 *
 * So this rung isolates exactly the remaining variable: element count of a LOCAL
 * pointer-bearing aggregate. LOCFL_N is the only thing that changes between variants; grow it
 * until the rung stops returning. Returns a checksum so a WRONG value is as informative as a
 * hang -- a wedge yields one bit and costs a slot.
 */
#ifndef LOCFL_N
#define LOCFL_N 8
#endif

static volatile unsigned locfl_gate = 1u;  /* satisfies the ldc gp[i] build gate */
static int locfl_f0(void) { return 3; }
static int locfl_f1(void) { return 5; }
static int locfl_f2(void) { return 7; }
static int locfl_f3(void) { return 11; }

/* V3: FLAT array of function pointers -- no struct element type. */

static unsigned locfl_compute(void) {
  /* Non-const on purpose: FuncDef's own comment says the array "cannot be constant since
     changes are made to the pHash elements at start-time". A const local would let the
     compiler place it in .rodata and stop reproducing the shape. */
  int (*a[LOCFL_N])(void);
  unsigned i, s = 0;
  for (i = 0; i < LOCFL_N; i++)
    a[i] = (i & 2) ? ((i & 1) ? locfl_f0 : locfl_f1)
                   : ((i & 1) ? locfl_f2 : locfl_f3);
  for (i = 0; i < LOCFL_N; i++)
    s += (unsigned)a[i]() + i;
  return s + locfl_gate - 1u;
}
#endif
