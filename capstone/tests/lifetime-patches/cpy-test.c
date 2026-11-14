/* cpy-test.c: unit test of CPython's pymalloc lifetime patch (0014), linked into the interpreter in
 * place of Programs/python.o, with inject.h forced into obmalloc.c.
 *   cpy-test stale    a pymalloc block read after PyObject_Free: must fault (cause 25)
 *   cpy-test alloc    a failed CDERIVE in PyObject_Malloc: the request is served, and the block that
 *                     got no lifetime goes back to its pool (the next request of the size gets it)
 *   cpy-test realloc  a failed CDERIVE in an in-place PyObject_Realloc: NULL, old block intact */
#include <Python.h>
#include "pycore_pystate.h"
#include "pycore_interp.h"
#include <stdio.h>
#include <string.h>

long lifec_fail_left, lifec_fail_size, lifec_failed;
unsigned long lifec_failed_at;

static int stale(void)
{
    volatile char *p = PyObject_Malloc(64);
    p[0] = 7;
    PyObject_Free((void *)p);
    printf("CPYTEST stale: read after free\n");
    fflush(stdout);
    printf("CPYTEST stale: read %d, no fault\n", p[0]);
    return 1;
}

static int alloc(void)
{
    void *holder = PyObject_Malloc(488);   /* keeps the size class's pool in use */
    lifec_fail_size = 488;
    lifec_fail_left = 1;
    void *p = PyObject_Malloc(488);
    lifec_fail_left = 0;
    void *q = PyObject_Malloc(488);
    unsigned long qa = __builtin_capstone_cap_get_cursor(q);
    printf("CPYTEST alloc: failed=%ld served=%s next_block_is_the_failed_one=%s\n", lifec_failed,
           p ? "yes" : "no", qa == lifec_failed_at ? "yes" : "no");
    if (p == NULL)
        return 1;
    memset(p, 0x5a, 488);
    PyObject_Free(p);
    PyObject_Free(q);
    PyObject_Free(holder);
    return lifec_failed == 1 && qa == lifec_failed_at ? 0 : 1;
}

static int realloc_inplace(void)
{
    unsigned char *p = PyObject_Malloc(400);
    memset(p, 0x5a, 400);
    lifec_fail_size = 390;
    lifec_fail_left = 1;
    void *q = PyObject_Realloc(p, 390);
    lifec_fail_left = 0;
    printf("CPYTEST realloc: failed=%ld result=%s\n", lifec_failed, q ? "pointer" : "NULL");
    fflush(stdout);
    if (q != NULL)
        return 1;
    for (int i = 0; i < 400; i++)
        if (p[i] != 0x5a)
            return 1;
    PyObject_Free(p);
    printf("CPYTEST realloc: old block intact and freed\n");
    return 0;
}

static int tagged(void *p)
{
    unsigned long t;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(t) : "r"(p));
    return t != 7;
}

static int arenas(void)
{
    struct _obmalloc_state *st = _PyInterpreterState_GET()->obmalloc;
    struct arena_object *a = st->mgmt.arenas;
    printf("CPYTEST arenas: max=%u array_tagged=%d\n", st->mgmt.maxarenas, tagged(a));
    for (unsigned i = 0; i < st->mgmt.maxarenas; i++)
        if (a[i].address)
            printf("CPYTEST arenas: %u address=%lx arena_ptr_tagged=%d pool_address_tagged=%d\n", i,
                   (unsigned long)a[i].address, tagged(a[i].arena_ptr), tagged(a[i].pool_address));
    return 0;
}

int main(int argc, char **argv)
{
    Py_NoSiteFlag = 1;
    Py_InitializeEx(0);
    const char *mode = argc > 1 ? argv[1] : "";
    int rc = !strcmp(mode, "stale") ? stale() : !strcmp(mode, "alloc") ? alloc()
           : !strcmp(mode, "realloc") ? realloc_inplace() : !strcmp(mode, "arenas") ? arenas() : 2;
    printf("CPYTEST %s: rc=%d\n", mode, rc);
    fflush(stdout);
    return rc;
}
