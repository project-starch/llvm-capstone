#ifndef LOCNP_KERNEL_H
#define LOCNP_KERNEL_H
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
 * pointer-bearing aggregate. LOCNP_N is the only thing that changes between variants; grow it
 * until the rung stops returning. Returns a checksum so a WRONG value is as informative as a
 * hang -- a wedge yields one bit and costs a slot.
 */
#ifndef LOCNP_N
#define LOCNP_N 8
#endif

static volatile unsigned locnp_gate = 1u;  /* satisfies the ldc gp[i] build gate */
static int locnp_f0(void) { return 3; }
static int locnp_f1(void) { return 5; }
static int locnp_f2(void) { return 7; }
static int locnp_f3(void) { return 11; }

struct locnp_ent { int a; int b; int tag; };   /* V2: NO pointers at all */

static unsigned locnp_compute(void) {
  /* Non-const on purpose: FuncDef's own comment says the array "cannot be constant since
     changes are made to the pHash elements at start-time". A const local would let the
     compiler place it in .rodata and stop reproducing the shape. */
  struct locnp_ent a[LOCNP_N];
  unsigned i, s = 0;
  for (i = 0; i < LOCNP_N; i++) {
    a[i].a = (int)(i & 1) ? 5 : 4;
    a[i].b = (int)(i & 2) ? 7 : 3;
    a[i].tag  = (int)(i * 3u + 1u);
  }
  for (i = 0; i < LOCNP_N; i++) {
    s += (unsigned)a[i].a * 100u + (unsigned)a[i].b + (unsigned)a[i].tag;
  }
  return s + locnp_gate - 1u;
}
#endif
