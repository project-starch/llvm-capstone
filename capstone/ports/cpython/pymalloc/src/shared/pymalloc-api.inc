/* Included inside upstream obmalloc.c, where the actual allocator is visible.
 */
void pym_allocator_init(void) {
  OMState *state = get_state();
#ifdef PYMALLOC_REFERENCE
  /* Same initializer as CPython's init_obmalloc_pools(). */
#include "pycore_obmalloc_init.h"
  poolp temp[OBMALLOC_USED_POOLS_SIZE] = _obmalloc_pools_INIT(state->pools);
  memcpy(&state->pools.used, temp, sizeof temp);
#else
  for (unsigned i = 0; i < NB_SMALL_SIZE_CLASSES; ++i) {
    poolp head = &state->pools.heads[i];
    head->nextpool = head->prevpool = head;
  }
#endif
}
void *pym_malloc(size_t n) {
  void *p = n > PY_SSIZE_T_MAX ? NULL : _PyObject_Malloc(NULL, n);
  pym_observe(PYM_ALLOC, p, n);
  return p;
}
void *pym_calloc(size_t k, size_t n) {
  if (n && k > (size_t)PY_SSIZE_T_MAX / n)
    return NULL;
  void *p = _PyObject_Calloc(NULL, k, n);
  pym_observe(PYM_CALLOC, p, k * n);
  return p;
}
void *pym_realloc(void *p, size_t n) {
  void *q = n > PY_SSIZE_T_MAX ? NULL : _PyObject_Realloc(NULL, p, n);
  pym_observe(PYM_REALLOC, q, n);
  return q;
}
void pym_free(void *p) {
  pym_observe(PYM_FREE, p, 0);
  _PyObject_Free(NULL, p);
}
