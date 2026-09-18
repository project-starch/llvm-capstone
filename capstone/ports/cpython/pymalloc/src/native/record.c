/* Public PyMem wrappers record successful MEM/OBJ requests in a single GIL
 * interpreter. Bootstrap allocations predating the window are not invented:
 * unknown frees are ignored and a successful unknown realloc is a new birth.
 * Fixed storage and libc FILE I/O avoid recursion through the wrapped APIs.
 */
#define PY_SSIZE_T_CLEAN
#include "port.h"
#include <Python.h>
#include <stdio.h>
#include <string.h>
#define MAP_SIZE (2 * PYM_MAX_OBJECTS)
static PyMemAllocatorEx original[2];
static PyInterpreterState *interpreter;
static FILE *stream;
static uint32_t map[MAP_SIZE], spare[PYM_MAX_OBJECTS];
static void *pointers[PYM_MAX_OBJECTS];
static uint32_t next_id, spare_count, live;
static uint64_t count;
static int active, failed, installed;
static size_t position(void *p) {
  size_t index =
      ((uintptr_t)p >> 4) * UINT64_C(11400714819323198485) & (MAP_SIZE - 1);
  size_t tomb = MAP_SIZE;
  for (size_t n = 0; n < MAP_SIZE; ++n) {
    uint32_t v = map[index];
    if (!v)
      return tomb != MAP_SIZE ? tomb : index;
    if (v == UINT32_MAX) {
      if (tomb == MAP_SIZE)
        tomb = index;
    } else if (pointers[v - 1] == p)
      return index;
    index = (index + 1) & (MAP_SIZE - 1);
  }
  return tomb;
}
static void emit(uint64_t op, uint64_t id, uint64_t size, uint64_t value) {
  struct pym_event e = {op, id, size, value};
  if (count >= (PYM_FILE_BYTES - sizeof(struct pym_header)) / sizeof e - 1 ||
      fwrite(&e, sizeof e, 1, stream) != 1)
    failed = 1;
  ++count;
}
static void birth(void *p, size_t n, int op) {
  if (!p || failed)
    return;
  if (!spare_count && next_id == PYM_MAX_OBJECTS) {
    failed = 1;
    return;
  }
  uint32_t id = spare_count ? spare[--spare_count] : next_id++;
  size_t slot = position(p);
  if (slot == MAP_SIZE || (map[slot] && map[slot] != UINT32_MAX)) {
    failed = 1;
    return;
  }
  pointers[id] = p;
  map[slot] = id + 1;
  ++live;
  emit(op, id, n, count & 255);
}
static int recording(void) {
  if (!active || failed)
    return 0;
  /* Multiple interpreters or a GIL-free callback invalidate this capture. */
  if (!PyGILState_Check() || PyInterpreterState_Get() != interpreter) {
    failed = 1;
    return 0;
  }
  return 1;
}
static void *record_malloc(void *ctx, size_t n) {
  PyMemAllocatorEx *a = ctx;
  void *p = a->malloc(a->ctx, n);
  if (recording())
    birth(p, n, PYM_ALLOC);
  return p;
}
static void *record_calloc(void *ctx, size_t k, size_t n) {
  PyMemAllocatorEx *a = ctx;
  void *p = a->calloc(a->ctx, k, n);
  if (recording())
    birth(p, k * n, PYM_CALLOC);
  return p;
}
static void record_free(void *ctx, void *p) {
  PyMemAllocatorEx *a = ctx;
  if (p && recording()) {
    size_t slot = position(p);
    uint32_t v = slot < MAP_SIZE ? map[slot] : 0;
    if (v && v != UINT32_MAX) {
      uint32_t id = v - 1;
      emit(PYM_FREE, id, 0, 0);
      map[slot] = UINT32_MAX;
      pointers[id] = NULL;
      spare[spare_count++] = id;
      --live;
    }
  }
  a->free(a->ctx, p);
}
static void *record_realloc(void *ctx, void *p, size_t n) {
  PyMemAllocatorEx *a = ctx;
  int capture = recording();
  size_t slot = p && capture ? position(p) : MAP_SIZE;
  uint32_t v = slot < MAP_SIZE ? map[slot] : 0;
  void *q = a->realloc(a->ctx, p, n);
  if (!capture || !q)
    return q;
  if (!v || v == UINT32_MAX) {
    birth(q, n, PYM_ALLOC);
    return q;
  }
  uint32_t id = v - 1;
  map[slot] = UINT32_MAX;
  pointers[id] = NULL;
  slot = position(q);
  if (slot == MAP_SIZE) {
    failed = 1;
    return q;
  }
  pointers[id] = q;
  map[slot] = id + 1;
  emit(PYM_REALLOC, id, n, count & 255);
  return q;
}
static PyObject *start(PyObject *self, PyObject *args) {
  (void)self;
  const char *path;
  if (!PyArg_ParseTuple(args, "s", &path))
    return NULL;
  if (installed)
    return PyErr_Format(PyExc_RuntimeError, "capture already active");
  stream = fopen(path, "wb+");
  if (!stream)
    return PyErr_SetFromErrno(PyExc_OSError);
  struct pym_header h = {.magic = PYM_MAGIC};
  if (fwrite(&h, sizeof h, 1, stream) != 1) {
    fclose(stream);
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  memset(map, 0, sizeof map);
  memset(pointers, 0, sizeof pointers);
  next_id = spare_count = live = 0;
  count = 0;
  failed = 0;
  interpreter = PyInterpreterState_Get();
  for (int i = 0; i < 2; ++i) {
    PyMemAllocatorDomain domain = i ? PYMEM_DOMAIN_OBJ : PYMEM_DOMAIN_MEM;
    PyMem_GetAllocator(domain, &original[i]);
    PyMemAllocatorEx a = {&original[i], record_malloc, record_calloc,
                          record_realloc, record_free};
    PyMem_SetAllocator(domain, &a);
  }
  active = installed = 1;
  Py_RETURN_NONE;
}
static PyObject *stop(PyObject *self, PyObject *ignored) {
  (void)self;
  (void)ignored;
  if (!installed)
    return PyErr_Format(PyExc_RuntimeError, "no active capture");
  active = 0;
  for (int i = 0; i < 2; ++i)
    PyMem_SetAllocator(i ? PYMEM_DOMAIN_OBJ : PYMEM_DOMAIN_MEM, &original[i]);
  installed = 0;
  if (!failed)
    emit(PYM_END, live, 0, 0);
  struct pym_header h = {.magic = PYM_MAGIC, .count = count};
  if (fseek(stream, 0, SEEK_SET) || fwrite(&h, sizeof h, 1, stream) != 1)
    failed = 1;
  if (fclose(stream))
    failed = 1;
  stream = NULL;
  if (failed)
    return PyErr_Format(PyExc_RuntimeError,
                        "capture exceeded limits, crossed interpreter "
                        "boundaries, or failed I/O; discard incomplete trace");
  return Py_BuildValue("(KK)", (unsigned long long)count,
                       (unsigned long long)live);
}
static PyMethodDef methods[] = {{"start", start, METH_VARARGS, NULL},
                                {"stop", stop, METH_NOARGS, NULL},
                                {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,
                                    "_pymrecord",
                                    NULL,
                                    -1,
                                    methods,
                                    NULL,
                                    NULL,
                                    NULL,
                                    NULL};
PyMODINIT_FUNC PyInit__pymrecord(void) { return PyModule_Create(&module); }
