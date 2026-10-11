/* One atomic operation on a pointer per function; -DOP_load|store|cas|xchg. */
#if OP_load
void *f(void **p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
#elif OP_store
void f(void **p, void *v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
#elif OP_cas
int f(void **p, void *e, void *d) { return __atomic_compare_exchange_n(p, &e, d, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); }
#elif OP_xchg
void *f(void **p, void *v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
#endif
