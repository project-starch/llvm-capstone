/* _pymctl: the parent, sibling and successor controls of
 * capstone/docs/plans/2026-10-11-nested-lifetime-controls.md, for CPython's
 * pymalloc (PyObject_Malloc / PyObject_Free) inside the real interpreter.
 *
 * A built-in module, compiled into BOTH arms from this one file (build-arm.sh
 * appends it to Modules/Setup.local's *static* block). Nothing in it depends on
 * the arm: the protected arm differs only in the interpreter's patch 0014.
 *
 *   _pymctl.parent()     control 1: free A, then read and write A's former range
 *                        through the pool header pointer obmalloc itself uses
 *   _pymctl.sibling()    control 2: A and B in one pool, free A, B read/written
 *                        and its contents checked
 *   _pymctl.successor()  control 3: free A, allocate C until C's address is A's,
 *                        read and write through C
 *   _pymctl.stale()      the stale run: control 3's sequence, then ONE read
 *                        through the old A at the labelled stale probe
 *
 * THE PARENT REFERENCE. obmalloc keeps, per arena, the capability the system
 * allocator (mallocng) returned for the 1 MiB arena: allarenas[i].arena_ptr,
 * patch 0009, read through _Py_ARENA_PTR. In the protected build patch 0014
 * derives arena_life from it (CDERIVE) and every client block from arena_life
 * (CDERIVE again), and pymalloc_free revokes the block's child through
 * arena_life (CREVOKE), then continues with arena_inside(ao, p) =
 * _Py_ARENA_PTR(ao) + (p - ao->address) and pool = POOL_ADDR(that). Control 1
 * takes exactly that path: the arena found by address, _Py_ARENA_PTR plus the
 * offset, POOL_ADDR, then the pool plus A's offset in it. In the release build
 * the same field exists (0009) and the same path is taken; there the client's
 * own pointer would reach the pool too.
 *
 * Every access to the memory under test goes through one of the labelled probes
 * below, so a fault's pc names the probe. Output is written with write(2) from
 * a stack buffer, so printing allocates nothing from pymalloc in between.
 */
#include "Python.h"
#include "pycore_interp.h"    // PyInterpreterState.obmalloc
#include "pycore_obmalloc.h"  // struct arena_object, POOL_ADDR, ARENA_SIZE
#include "pycore_pystate.h"   // _PyInterpreterState_GET

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define CTL_N 48          /* one size class, as pymalloc-repros' OBJ */
#define CTL_MAX_TRIES 4096

/* ---- labelled probes ---------------------------------------------------- */

__attribute__((noinline)) static unsigned
pymctl_read_probe(const volatile unsigned char *p)
{
    unsigned long value;
    __asm__ volatile(".globl pymctl_read\npymctl_read:\nlbu %0, 0(%1)\n"
                     : "=r"(value) : "r"(p) : "memory");
    return (unsigned)value;
}

__attribute__((noinline)) static void
pymctl_write_probe(volatile unsigned char *p, unsigned v)
{
    unsigned long value = v;
    __asm__ volatile(".globl pymctl_write\npymctl_write:\nsb %0, 0(%1)\n"
                     :: "r"(value), "r"(p) : "memory");
}

/* The stale access, alone in its function so a fault here is unambiguous. */
__attribute__((noinline)) static unsigned
pymctl_stale_probe(const volatile unsigned char *p)
{
    unsigned long value;
    __asm__ volatile(".globl pymctl_stale_read\npymctl_stale_read:\nlbu %0, 0(%1)\n"
                     : "=r"(value) : "r"(p) : "memory");
    return (unsigned)value;
}

/* Pointer-sized reads through the parent: the pool's freeblock head and the
 * free-list link pymalloc_free writes into a freed block. */
__attribute__((noinline)) static uintptr_t
pymctl_link_probe(void *const volatile *p)
{
    return (uintptr_t)*p;
}

/* ---- output ------------------------------------------------------------- */

static void
say(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf - 1, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return;
    }
    if (n > (int)sizeof buf - 2) {
        n = (int)sizeof buf - 2;
    }
    buf[n++] = '\n';
    (void)!write(1, buf, (size_t)n);
}

static PyObject *
fail(const char *control, const char *why)
{
    say("CTL %s FAILED %s", control, why);
    PyErr_Format(PyExc_RuntimeError, "%s: %s", control, why);
    return NULL;
}

#define ADDR(p) ((unsigned long)(uintptr_t)(p))

/* The arena holding ADDR, found by address over obmalloc's own table. */
static struct arena_object *
arena_of(uintptr_t addr, unsigned *index)
{
    struct _obmalloc_state *state = _PyInterpreterState_GET()->obmalloc;
    for (unsigned i = 0; i < state->mgmt.maxarenas; i++) {
        struct arena_object *ao = &state->mgmt.arenas[i];
        if (ao->address != 0 && addr >= ao->address
                && addr < ao->address + ARENA_SIZE) {
            *index = i;
            return ao;
        }
    }
    return NULL;
}

/* obmalloc's own reference to the pool holding ADDR, as pymalloc_free
 * reaches it: _Py_ARENA_PTR(ao) + offset, then POOL_ADDR. */
static poolp
parent_pool(uintptr_t addr, unsigned *index)
{
    struct arena_object *ao = arena_of(addr, index);
    if (ao == NULL) {
        return NULL;
    }
    pymem_block *inside = _Py_ARENA_PTR(ao) + (addr - ao->address);
    return POOL_ADDR(inside);
}

/* ---- control 1: parent -------------------------------------------------- */

static PyObject *
ctl_parent(PyObject *self, PyObject *unused)
{
    unsigned char *a = PyObject_Malloc(CTL_N);
    if (a == NULL) {
        return fail("parent", "PyObject_Malloc returned NULL");
    }
    memset(a, 0xA1, CTL_N);
    uintptr_t addr = (uintptr_t)a;
    unsigned arena;
    poolp pool = parent_pool(addr, &arena);
    if (pool == NULL) {
        return fail("parent", "A is not in any pymalloc arena");
    }
    uintptr_t off = addr - (uintptr_t)pool;
    say("CTL parent A=0x%lx n=%d arena=%u arena_address=0x%lx pool=0x%lx offset=%lu szidx=%u",
        ADDR(a), CTL_N, arena, (unsigned long)_PyInterpreterState_GET()->obmalloc
            ->mgmt.arenas[arena].address, ADDR(pool), (unsigned long)off, pool->szidx);
    uintptr_t lastfree = pymctl_link_probe((void *const volatile *)&pool->freeblock);

    PyObject_Free(a);
    say("CTL parent freed A; pool.count=%u", pool->ref.count);

    /* The pool header, through the parent: A is now the free list's head. */
    uintptr_t head = pymctl_link_probe((void *const volatile *)&pool->freeblock);
    /* A's former range, through the parent: pool + A's offset. */
    unsigned char *former = (unsigned char *)pool + off;
    uintptr_t link = pymctl_link_probe((void *const volatile *)former);
    unsigned last = pymctl_read_probe(former + CTL_N - 1);
    say("CTL parent read pool.freeblock=0x%lx (A 0x%lx) link=0x%lx (old head 0x%lx) "
        "byte[%d]=0x%02x (A wrote 0xa1)", (unsigned long)head, ADDR(a),
        (unsigned long)link, (unsigned long)lastfree, CTL_N - 1, last);
    pymctl_write_probe(former + CTL_N - 1, 0x5A);
    unsigned back = pymctl_read_probe(former + CTL_N - 1);
    pymctl_write_probe(former + CTL_N - 1, last);   /* leave pymalloc's block as it was */
    say("CTL parent wrote 0x5a read back 0x%02x, restored 0x%02x", back,
        pymctl_read_probe(former + CTL_N - 1));
    if (head != addr || link != lastfree || last != 0xA1 || back != 0x5A) {
        return fail("parent", "the parent read something pymalloc did not leave there");
    }
    say("CTL parent completed=1");
    Py_RETURN_NONE;
}

/* ---- control 2: sibling ------------------------------------------------- */

static PyObject *
ctl_sibling(PyObject *self, PyObject *unused)
{
    static unsigned char *extra[CTL_MAX_TRIES];
    unsigned nextra = 0;
    unsigned char *a = PyObject_Malloc(CTL_N), *b = NULL;
    if (a == NULL) {
        return fail("sibling", "PyObject_Malloc returned NULL");
    }
    uintptr_t pa = (uintptr_t)a >> POOL_BITS;
    for (unsigned t = 0; t < CTL_MAX_TRIES; t++) {
        unsigned char *c = PyObject_Malloc(CTL_N);
        if (c == NULL) {
            return fail("sibling", "PyObject_Malloc returned NULL");
        }
        if (((uintptr_t)c >> POOL_BITS) == pa) {
            b = c;
            break;
        }
        extra[nextra++] = c;
    }
    if (b == NULL) {
        return fail("sibling", "no second block in A's pool");
    }
    for (int i = 0; i < CTL_N; i++) {
        b[i] = (unsigned char)(i * 7 + 3);
    }
    say("CTL sibling A=0x%lx B=0x%lx n=%d pool=0x%lx skipped=%u",
        ADDR(a), ADDR(b), CTL_N, (unsigned long)(pa << POOL_BITS), nextra);

    PyObject_Free(a);
    say("CTL sibling freed A");

    int bad = 0;
    for (int i = 0; i < CTL_N; i++) {
        if (pymctl_read_probe(b + i) != (unsigned char)(i * 7 + 3)) {
            bad++;
        }
    }
    say("CTL sibling read B: %d of %d bytes differ from what B was given", bad, CTL_N);
    pymctl_write_probe(b + CTL_N - 1, 0x5A);
    unsigned back = pymctl_read_probe(b + CTL_N - 1);
    say("CTL sibling wrote B[%d]=0x5a read back 0x%02x", CTL_N - 1, back);
    PyObject_Free(b);
    for (unsigned i = 0; i < nextra; i++) {
        PyObject_Free(extra[i]);
    }
    if (bad || back != 0x5A) {
        return fail("sibling", "B's contents were not intact");
    }
    say("CTL sibling completed=1 intact=1");
    Py_RETURN_NONE;
}

/* ---- control 3: successor, and the stale run ---------------------------- */

/* Held through memory, as pymalloc-repros' `held`, across the free. */
static unsigned char *volatile pymctl_held;
static unsigned char *successor_extra[CTL_MAX_TRIES];
static unsigned successor_nextra;

/* Free A, then allocate until pymalloc hands out A's address again. */
static unsigned char *
successor(const char *control)
{
    unsigned char *a = PyObject_Malloc(CTL_N), *c = NULL;
    if (a == NULL) {
        return NULL;
    }
    memset(a, 0xA1, CTL_N);
    pymctl_held = a;
    PyObject_Free(a);
    unsigned t;
    successor_nextra = 0;
    for (t = 1; t <= CTL_MAX_TRIES; t++) {
        unsigned char *x = PyObject_Malloc(CTL_N);
        if (x == NULL) {
            return NULL;
        }
        if ((uintptr_t)x == (uintptr_t)pymctl_held) {
            c = x;
            break;
        }
        successor_extra[successor_nextra++] = x;
    }
    say("CTL %s A=0x%lx C=0x%lx equal=%d tries=%u n=%d", control, ADDR(pymctl_held),
        ADDR(c), c != NULL, c ? t : CTL_MAX_TRIES, CTL_N);
    if (c == NULL) {
        return NULL;
    }
    /* Through C: write every byte, read them back. */
    for (int i = 0; i < CTL_N; i++) {
        pymctl_write_probe(c + i, 0xC3);
    }
    int bad = 0;
    for (int i = 0; i < CTL_N; i++) {
        if (pymctl_read_probe(c + i) != 0xC3) {
            bad++;
        }
    }
    say("CTL %s wrote 0xc3 through C, read back: %d of %d bytes differ", control, bad, CTL_N);
    return bad ? NULL : c;
}

static void
successor_release(unsigned char *c)
{
    PyObject_Free(c);
    for (unsigned i = 0; i < successor_nextra; i++) {
        PyObject_Free(successor_extra[i]);
    }
}

static PyObject *
ctl_successor(PyObject *self, PyObject *unused)
{
    unsigned char *c = successor("successor");
    if (c == NULL) {
        return fail("successor", "C did not reach A's address or did not read back");
    }
    successor_release(c);
    say("CTL successor completed=1");
    Py_RETURN_NONE;
}

static PyObject *
ctl_stale(PyObject *self, PyObject *unused)
{
    unsigned char *c = successor("stale");
    if (c == NULL) {
        return fail("stale", "C did not reach A's address or did not read back");
    }
    say("CTL stale reading through the old A at pymctl_stale_read");
    unsigned v = pymctl_stale_probe(pymctl_held);
    say("CTL stale read through A returned 0x%02x (C wrote 0xc3) reads_c=%d", v, v == 0xC3);
    successor_release(c);
    say("CTL stale completed=1");
    Py_RETURN_NONE;
}

static PyMethodDef pymctl_methods[] = {
    {"parent", ctl_parent, METH_NOARGS, "control 1: the parent stays usable"},
    {"sibling", ctl_sibling, METH_NOARGS, "control 2: the sibling stays usable"},
    {"successor", ctl_successor, METH_NOARGS, "control 3: the successor stays usable"},
    {"stale", ctl_stale, METH_NOARGS, "control 3, then a read through the stale A"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef pymctl_module = {
    PyModuleDef_HEAD_INIT, "_pymctl",
    "Parent, sibling and successor controls for pymalloc.", -1, pymctl_methods,
};

PyMODINIT_FUNC
PyInit__pymctl(void)
{
    return PyModule_Create(&pymctl_module);
}
