/* Control: the same four operations on a long must need no library call. */
long load(long *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
void store(long *p, long v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
int cas(long *p, long e, long d) { return __atomic_compare_exchange_n(p, &e, d, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); }
long xchg(long *p, long v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
