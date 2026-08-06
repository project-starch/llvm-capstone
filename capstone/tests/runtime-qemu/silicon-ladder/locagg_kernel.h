#ifndef LOCAGG_KERNEL_H
#define LOCAGG_KERNEL_H
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
 * pointer-bearing aggregate. LOCAGG_N is the only thing that changes between variants; grow it
 * until the rung stops returning. Returns a checksum so a WRONG value is as informative as a
 * hang -- a wedge yields one bit and costs a slot.
 */
#ifndef LOCAGG_N
#define LOCAGG_N 8
#endif

static volatile unsigned locagg_gate = 1u;  /* keeps a gp[i] access alive */
static int locagg_f0(void) { return 3; }
static int locagg_f1(void) { return 5; }
static int locagg_f2(void) { return 7; }
static int locagg_f3(void) { return 11; }

struct locagg_ent { const char *name; int (*fn)(void); int tag; };

static unsigned locagg_compute(void) {
#ifdef LOCAGG_SENTINEL
  /* ENTRY-STALL DISCRIMINATOR. Returns before touching the array at all.
     The lpc controller emits NO post-entry marker, so `SHA5` last on this path cannot tell
     "the body wedged" from "the domain never ran" -- and the tree records every lpc-hosted
     domain dying in share #1 on 2026-08-06 regardless of content. Without this, locagg8's
     silence attributes nothing.
       returns 0x5E17 -> the glue, cap-init, entry and return all work; the silence in the
                         unguarded build is attributable to the array code
       wedges         -> the rung is an entry stall and says NOTHING about local aggregates
     Gated so the real path is byte-identical; the prefix is cmp-verified before shipping. */
  return 0x5E17u + locagg_gate - 1u;
#endif
  /* Non-const on purpose: FuncDef's own comment says the array "cannot be constant since
     changes are made to the pHash elements at start-time". A const local would let the
     compiler place it in .rodata and stop reproducing the shape. */
  struct locagg_ent a[LOCAGG_N];
  unsigned i, s = 0;
  for (i = 0; i < LOCAGG_N; i++) {
    a[i].name = (i & 1) ? "alpha" : "beta";
    a[i].fn   = (i & 2) ? ((i & 1) ? locagg_f0 : locagg_f1)
                        : ((i & 1) ? locagg_f2 : locagg_f3);
    a[i].tag  = (int)(i * 3u + 1u);
  }
  for (i = 0; i < LOCAGG_N; i++) {
    const char *p = a[i].name;
    unsigned len = 0;
    while (p[len]) len++;                 /* touch the string pointer  */
    s += len * 100u + (unsigned)a[i].fn() /* call through the fn ptr   */
       + (unsigned)a[i].tag;
  }
  return s;
}
#endif
