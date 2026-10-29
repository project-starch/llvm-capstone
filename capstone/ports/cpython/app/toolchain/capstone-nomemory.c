/* A failure-injecting allocator hook, provided as a small static module so the
 * allocator-boundary corpus's MemoryError-path cases can run in the interpreter.
 *
 * This is HARNESS SCAFFOLDING, not a program under test and not a defect. It is
 * CPython's own `_testcapi.set_nomemory` / `remove_mem_hooks` mechanism (the
 * upstream pyfailmalloc hook), copied verbatim from Modules/_testcapi/mem.c of
 * the pinned 3.13.7 so the trigger is the one upstream's regression test runs. The capstone build does not include the full
 * _testcapi (32 files of C-API exercise code), so these two functions are given
 * their own module, `_capstone_nomemory`, which uses only the public
 * PyMemAllocatorEx API and therefore compiles for the target unchanged.
 */
#include <Python.h>

/* ---- verbatim from CPython 3.13.7 Modules/_testcapi/mem.c ---------------- */
static struct {
    int installed;
    PyMemAllocatorEx raw;
    PyMemAllocatorEx mem;
    PyMemAllocatorEx obj;
} FmHook;

static struct {
    int start;
    int stop;
    Py_ssize_t count;
} FmData;

static int
fm_nomemory(void)
{
    FmData.count++;
    if (FmData.count > FmData.start &&
        (FmData.stop <= 0 || FmData.count <= FmData.stop))
    {
        return 1;
    }
    return 0;
}

static void *
hook_fmalloc(void *ctx, size_t size)
{
    PyMemAllocatorEx *alloc = (PyMemAllocatorEx *)ctx;
    if (fm_nomemory()) {
        return NULL;
    }
    return alloc->malloc(alloc->ctx, size);
}

static void *
hook_fcalloc(void *ctx, size_t nelem, size_t elsize)
{
    PyMemAllocatorEx *alloc = (PyMemAllocatorEx *)ctx;
    if (fm_nomemory()) {
        return NULL;
    }
    return alloc->calloc(alloc->ctx, nelem, elsize);
}

static void *
hook_frealloc(void *ctx, void *ptr, size_t new_size)
{
    PyMemAllocatorEx *alloc = (PyMemAllocatorEx *)ctx;
    if (fm_nomemory()) {
        return NULL;
    }
    return alloc->realloc(alloc->ctx, ptr, new_size);
}

static void
hook_ffree(void *ctx, void *ptr)
{
    PyMemAllocatorEx *alloc = (PyMemAllocatorEx *)ctx;
    alloc->free(alloc->ctx, ptr);
}

static void
fm_setup_hooks(void)
{
    if (FmHook.installed) {
        return;
    }
    FmHook.installed = 1;

    PyMemAllocatorEx alloc;
    alloc.malloc = hook_fmalloc;
    alloc.calloc = hook_fcalloc;
    alloc.realloc = hook_frealloc;
    alloc.free = hook_ffree;
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &FmHook.raw);
    PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &FmHook.mem);
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &FmHook.obj);

    alloc.ctx = &FmHook.raw;
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &alloc);

    alloc.ctx = &FmHook.mem;
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);

    alloc.ctx = &FmHook.obj;
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);
}

static void
fm_remove_hooks(void)
{
    if (FmHook.installed) {
        FmHook.installed = 0;
        PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &FmHook.raw);
        PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &FmHook.mem);
        PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &FmHook.obj);
    }
}

static PyObject *
set_nomemory(PyObject *self, PyObject *args)
{
    /* Memory allocation fails after 'start' allocation requests, and until
     * 'stop' allocation requests except when 'stop' is negative or equal
     * to 0 (default) in which case allocation failures never stop. */
    FmData.count = 0;
    FmData.stop = 0;
    if (!PyArg_ParseTuple(args, "i|i", &FmData.start, &FmData.stop)) {
        return NULL;
    }
    fm_setup_hooks();
    Py_RETURN_NONE;
}

static PyObject *
remove_mem_hooks(PyObject *self, PyObject *Py_UNUSED(ignored))
{
    fm_remove_hooks();
    Py_RETURN_NONE;
}
/* ---- end verbatim -------------------------------------------------------- */

static PyMethodDef methods[] = {
    {"set_nomemory", set_nomemory, METH_VARARGS,
     "set_nomemory(start, stop): fail allocations in [start, stop)."},
    {"remove_mem_hooks", remove_mem_hooks, METH_NOARGS,
     "remove_mem_hooks(): restore the allocators set_nomemory replaced."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_capstone_nomemory",
    "Upstream pyfailmalloc hook (CPython's _testcapi.set_nomemory), as a module.",
    -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC
PyInit__capstone_nomemory(void)
{
    return PyModule_Create(&module);
}
