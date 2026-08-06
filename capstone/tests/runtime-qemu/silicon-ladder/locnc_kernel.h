#ifndef LOCNC_KERNEL_H
#define LOCNC_KERNEL_H
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
 * pointer-bearing aggregate. LOCNC_N is the only thing that changes between variants; grow it
 * until the rung stops returning. Returns a checksum so a WRONG value is as informative as a
 * hang -- a wedge yields one bit and costs a slot.
 */
#ifndef LOCNC_N
#define LOCNC_N 8
#endif

static int locnc_f0(void) { return 3; }
static int locnc_f1(void) { return 5; }
static int locnc_f2(void) { return 7; }
static int locnc_f3(void) { return 11; }

struct locnc_ent { const char *name; int (*fn)(void); int tag; };

static unsigned locnc_compute(void) {
  /* Non-const on purpose: FuncDef's own comment says the array "cannot be constant since
     changes are made to the pHash elements at start-time". A const local would let the
     compiler place it in .rodata and stop reproducing the shape. */
  struct locnc_ent a[LOCNC_N];
  unsigned i, s = 0;
  for (i = 0; i < LOCNC_N; i++) {
    a[i].name = (i & 1) ? "alpha" : "beta";
    a[i].fn   = (i & 2) ? ((i & 1) ? locnc_f0 : locnc_f1)
                        : ((i & 1) ? locnc_f2 : locnc_f3);
    a[i].tag  = (int)(i * 3u + 1u);
  }
  for (i = 0; i < LOCNC_N; i++) {
    const char *p = a[i].name;
    unsigned len = 0;
    while (p[len]) len++;                 /* touch the string pointer  */
    /* V1: the fn pointer is STORED but never CALLED. If this returns, the fault is the
       indirect call through a locally-stored capability, not the store itself. */
    s += len * 100u + (unsigned)a[i].tag;
  }
  return s;
}
#endif
