/* One function per (operation, width). Compile with -DW=8|16|32|64 and one of -DOP_cas -DOP_add -DOP_xchg. */
#include <stdint.h>
#define T_(w) uint##w##_t
#define T(w) T_(w)
#if OP_cas
int f(T(W) *p) { T(W) e = 0; return __atomic_compare_exchange_n(p, &e, 1, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); }
#elif OP_add
T(W) f(T(W) *p) { return __atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST); }
#elif OP_xchg
T(W) f(T(W) *p) { return __atomic_exchange_n(p, 1, __ATOMIC_SEQ_CST); }
#endif
