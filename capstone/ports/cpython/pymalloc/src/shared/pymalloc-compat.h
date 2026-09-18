#ifndef PYMALLOC_COMPAT_H
#define PYMALLOC_COMPAT_H
#include "port.h"
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef intptr_t Py_ssize_t;
typedef struct PyInterpreterState PyInterpreterState;
typedef struct _PyRuntimeState _PyRuntimeState;
#define PyAPI_FUNC(t) t
#define PY_SSIZE_T_MAX INTPTR_MAX
#define Py_ALWAYS_INLINE __attribute__((always_inline))
#define Py_UNUSED(x) x __attribute__((unused))
#define _Py_SIZE_ROUND_UP(n, a) (((n) + (a) - 1) & ~((a) - 1))
#define _Py_ALIGN_DOWN(p, a) ((uintptr_t)(p) & ~((uintptr_t)(a) - 1))
#define _Py_NO_SANITIZE_ADDRESS
#define _Py_NO_SANITIZE_THREAD
#define _Py_NO_SANITIZE_MEMORY
#define assert(x) ((x) ? (void)0 : pym_fail(1000 + __LINE__))
#define PyMem_RawMalloc pym_raw_malloc
#define PyMem_RawCalloc pym_raw_calloc
#define PyMem_RawRealloc pym_raw_realloc
#define PyMem_RawFree pym_raw_free
#include "pycore_obmalloc.h"
#define uint pymem_uint
static struct {
  void *ctx;
  void *(*alloc)(void *, size_t);
  void (*free)(void *, void *, size_t);
} _PyObject_Arena = {NULL, pym_arena_alloc, pym_arena_free};
void *_PyObject_Malloc(void *, size_t);
void *_PyObject_Calloc(void *, size_t, size_t);
void *_PyObject_Realloc(void *, void *, size_t);
void _PyObject_Free(void *, void *);
#endif
