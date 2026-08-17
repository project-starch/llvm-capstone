#include <stdint.h>
#include <string.h>
#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/cstack.h"
#include "py/mperrno.h"
#include "py/mphal.h"
#include "py/stackctrl.h"

/* ---- the GC heap. One global; under -capstone-gp-captable its storage is CARVED
   from dom_data at entry, so it is charged against the domain's stack budget.
   32-byte aligned and a multiple of 32 so gc_init's align-down is a no-op.

   384 KiB is the knee, measured on chunk 800_916 which holds both the heap-bound failures and
   the one test a bigger heap costs. 96K -> 41 PASS, 192K -> 43, 384K -> 46, 512K -> 46 with not
   one status different from 384K, so past this point the only thing growing is the heap's bite
   out of the C stack (STACK reserve 678,832 -> 383,920 -> 252,848). The cost is
   micropython/viper_large_jump.py going UNSCORED -> FAIL: on a small heap it hits MemoryError
   and prints upstream's SKIP-TOO-LARGE, on a large one it gets as far as the deliberately
   disabled Viper decorator. Both statements are true; five real passes are worth the swap. */
#ifndef MPY_HEAP_SIZE
#define MPY_HEAP_SIZE (384U * 1024U)
#endif
static unsigned char mpy_heap[MPY_HEAP_SIZE] __attribute__((aligned(32)));

/* ---- output: the hostcall shared region, same shape as benchmarks/sqlite */
#define MPY_DPI_REGION_SHARE 1U
#define MPY_TEST_START_MAGIC 0x4d50595354415254ULL /* "MPYSTART" */
struct mpy_hostcall_v0 { unsigned long long phase, opcode, offset, length; long long result, error; };
#define MPY_HC_REGION_SIZE 4096UL
static volatile struct mpy_hostcall_v0 *hc_meta;
static volatile char *hc_payload;
static unsigned hc_share_count;

/* Capture what the interpreter prints, so a run can be checked against the EXPECTED OUTPUT and
   not merely against "returned without raising". Without this the only evidence that
   print(1+1) worked is an rc of 0, which is exactly the kind of clean result this project has
   learned not to trust: the hostcall region is only wired up when the host shares it, and when
   it does not, tx_strn writes nowhere and still reports success. */
#define MPY_CAP_MAX 192
static char mpy_cap_buf[MPY_CAP_MAX];
static unsigned mpy_cap_len;
#ifdef MPY_TEST_RUNNER
static unsigned mpy_out_hash;
static unsigned mpy_out_len;
#endif

mp_uint_t mp_hal_stdout_tx_strn(const char *str, size_t len) {
    for (size_t i = 0; i < len && mpy_cap_len < MPY_CAP_MAX; ++i) {
        mpy_cap_buf[mpy_cap_len++] = str[i];
    }
#ifdef MPY_TEST_RUNNER
    /* Hash EVERY byte, not only the ones that fit in mpy_cap_buf. Scoring a test on a truncated
       prefix would turn "output too long for the capture buffer" into a silent wrong verdict,
       and MicroPython's own tests routinely print more than 192 bytes. FNV-1a costs the same as
       the position-weighted sum it replaces and mixes far better.
       Placed before the hostcall block below, which consumes `len` and `str`. */
    for (size_t i = 0; i < len; ++i) {
        mpy_out_hash = (mpy_out_hash ^ (unsigned char)str[i]) * 16777619u;
    }
    mpy_out_len += len;
#endif
    if (!hc_meta || !hc_payload) return len;
    char *payload = (char *)hc_payload;
    unsigned long off = hc_meta->length;
    while (len-- && off + 1 < MPY_HC_REGION_SIZE) payload[off++] = *str++;
    hc_meta->length = off;
    return off;
}

/* ---- what py/ demands and the port must supply ---- */
#if !MICROPY_VFS && MICROPY_ENABLE_EXTERNAL_IMPORT
mp_lexer_t *mp_lexer_new_from_file(qstr filename) {
    (void)filename;
    mp_raise_OSError(MP_ENOENT);
}

mp_import_stat_t mp_import_stat(const char *path) {
    (void)path;
    return MP_IMPORT_STAT_NO_EXIST;
}
#endif

#if !MICROPY_VFS && MICROPY_PY_IO
mp_obj_t mp_builtin_open(size_t n_args, const mp_obj_t *args, mp_map_t *kwargs) {
    (void)n_args;
    (void)args;
    (void)kwargs;
    mp_raise_OSError(MP_ENOENT);
}
MP_DEFINE_CONST_FUN_OBJ_KW(mp_builtin_open_obj, 1, mp_builtin_open);
#endif

void nlr_jump_fail(void *val) {
    (void)val;
    mp_hal_stdout_tx_strn("\nNLRFAIL\n", 9);
    for (;;) { }
}

void gc_collect(void) {
    jmp_buf regs;              /* spills ra,s0-s11,sp as tagged 16-byte slots */
    setjmp(regs);
    gc_collect_start();
    void **sp_now = (void **)&regs;
    char *top = MP_STATE_THREAD(stack_top);
    gc_collect_root(sp_now, ((size_t)(top - (char *)sp_now)) / sizeof(void *));
    gc_collect_end();
}

/* ---- one line of Python ---- */
static int do_str(const char *src, mp_parse_input_kind_t kind) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, strlen(src), 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t pt = mp_parse(lex, kind);
        mp_obj_t module_fun = mp_compile(&pt, source_name, false);
        mp_call_function_0(module_fun);
        nlr_pop();
        return 0;
    }
    mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(nlr.ret_val));
    return 1;
}

/* how much stack the entry glue actually left: lcc(sp,2)-lcc(sp,3), the stkhr probe */
static size_t cap_stack_headroom(void) {
    unsigned long cur, base;
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x2" : "=r"(cur));
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
    return (size_t)(cur - base);
}

static size_t mpy_cstack_size(void) {
    size_t size = cap_stack_headroom();
#ifdef MPY_CSTACK_MAX
    if (size > MPY_CSTACK_MAX) {
        size = MPY_CSTACK_MAX;
    }
#endif
    return size;
}

/* MPY_STAGE bisects startup. Each stage RETURNS a marker instead of running on, so a stage that
   does not return is the bisection point -- the rule this project learned the expensive way:
   a wedged domain emits nothing, so a run that only ever fails tells you one bit per boot.
   Stages are cumulative and ascending; build one .dom per stage and run them all in ONE boot,
   ascending, with a known-good control first.

   Stage 0 is the control for the domain ITSELF: it proves entry, cap-init and the carve loop
   completed, without touching MicroPython at all. If stage 0 does not return, nothing above it
   says anything about the interpreter. */
#ifndef MPY_STAGE
#define MPY_STAGE 99
#endif
#define MPY_MARK(n) do { *res = 0x4D500000u | (unsigned)(n); return; } while (0)

#ifdef MPY_TEST_RUNNER
/* One image, many tests. The loader (`capstone-test.user <dom> <times>`) calls a domain
   repeatedly and prints each call's retval, and the entry glue's reentry path deliberately does
   NOT rebuild the cap table or re-run the initialisers, so a global survives between calls.
   That turns N tests into N domain switches inside ONE boot instead of N boots.
 *
 * Two properties fall out of the loader's own output rather than needing a harness:
 *   - the bisection is free. A fault kills the rest of the boot, so the output simply stops --
 *     and "Called dom (7-th time)" absent names test 7 without halving anything.
 *   - each test's verdict is its own retval, so a wrong ANSWER is distinguishable from a fault.
 *
 * The tests are generated into mpy_tests.h by tools/gen-test-table.py from MicroPython's own
 * tests/ directory, together with the expected output each one must produce.
 */
#include "mpy_tests.h"

static unsigned mpy_test_idx;

static void mpy_run_one_test(unsigned *res) {
    if (mpy_test_idx >= MPY_TEST_COUNT) {
        *res = 0xFFFFFFFFu;              /* past the end: the driver asked for too many calls */
        return;
    }
    unsigned idx = mpy_test_idx++;
    mpy_cap_len = 0;
    mpy_out_len = 0;
    mpy_out_hash = 2166136261u;         /* FNV-1a offset basis */
    /* Full reset per test, in the same order a fresh boot uses, so test N cannot pass because
       test N-1 left something behind. gc_init is included deliberately: after mp_deinit nothing
       reachable lives in the heap, so re-initialising it is safe and it is the only thing that
       makes an OOM in test N attributable to test N. sp is re-recorded because the stack limit
       must be measured from THIS call's frame, not the first call's. */
    mp_cstack_init_with_sp_here(mpy_cstack_size());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    int rc = do_str(mpy_tests[idx], MP_PARSE_FILE_INPUT);
    mp_deinit();
    /* 32 bits back to the loader, which prints them as an unsigned long:
         31     raised an uncaught exception -- informational, since the traceback text goes
                through tx_strn and is therefore already inside the hash
         20-30  test index, so a result stays attributable if the driver's call count slips
         16-19  low nibble of the output length, a cheap sanity aid when a hash mismatches
         0-15   FNV-1a over the whole output
       tools/gen-test-table.py::expected_retval builds the same word from the expected output. */
    *res = ((rc ? 1u : 0u) << 31)
         | ((idx & 0x7ff) << 20)
         | ((mpy_out_len & 0xf) << 16)
         | (mpy_out_hash & 0xffffu);
}
#endif

void domain_main(unsigned *res, unsigned func) {
    if (func == MPY_DPI_REGION_SHARE) {
        if (hc_share_count == 0) {
            hc_meta = (volatile struct mpy_hostcall_v0 *)res;
            #ifdef MPY_TEST_RUNNER
            /* A resumable suite runner puts the requested global test index in offset. This is
               deliberately opt-in by magic, so ordinary hostcall metadata keeps its meaning. */
            if (hc_meta->phase == MPY_TEST_START_MAGIC && hc_meta->offset <= MPY_TEST_COUNT) {
                mpy_test_idx = (unsigned)hc_meta->offset;
            }
            #endif
        }
        else if (hc_share_count == 1) hc_payload = (volatile char *)res;
        ++hc_share_count;
        return;
    }
    if (hc_meta) hc_meta->length = 0;

#ifdef MPY_TEST_RUNNER
    mpy_run_one_test(res);              /* each call runs the next test in the table */
    return;
#endif

#if MPY_STAGE == 0
    MPY_MARK(0xA0);            /* entered, nothing else -- NOTE it touches no global at all,
                                  so it does NOT prove the carve loop or the blob copy worked */
#endif

#if MPY_STAGE == 15
    /* Publish sp.BASE and sp.END from domain_main with the SAME instruction the glue's
       diagnostic uses, so the two sets of numbers are directly comparable. */
    {
        unsigned long b, e;
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(b));
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x4" : "=r"(e));
        /* Return the value rather than printing it: csdebugprint (funct7 0x43) issued from C
           took QEMU down, while the retval path is proven. MPY_SP_FIELD picks which one, so
           one boot covers both. The low 32 bits are enough -- the domain sits at 0x1016xxxxx
           and only the offset within it is in question. */
#ifndef MPY_SP_FIELD
#define MPY_SP_FIELD 3
#endif
        *res = (unsigned)((MPY_SP_FIELD == 3) ? b : e);
        return;
    }
#endif

#if MPY_STAGE == 13
    /* Read the cap-init table entry out of the BLOB, the way the glue does, and publish it.
       The offset is computed at RUNTIME from the same two linker symbols the glue uses, so
       this is layout-correct for whatever image it is built into -- a fixed offset would
       measure a different word in every build, which is how the first attempt went wrong.
       Taking the DIFFERENCE of two link-time addresses is arithmetic, not a load, so it is
       legal here even though neither symbol is reachable as data. */
    {
        extern char __capstone_cap_init_start[];
        extern char __gpfree_globals_base[];
        unsigned long blob_off =
            (unsigned long)((const char *)__capstone_cap_init_start
                            - (const char *)__gpfree_globals_base);
        unsigned long base;
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
        {
            volatile unsigned long anchor = 0;
            const unsigned char *here = (const unsigned char *)&anchor;
            const unsigned long *p =
                (const unsigned long *)(here - ((unsigned long)&anchor - base) + blob_off);
            /* The entry is a signed 8-byte delta; publish the HIGH half, because the low half
               was already shown to survive and the high half is where the corruption is. */
            *res = (unsigned)((*p) >> 32);
            return;
        }
    }
#endif

#if MPY_STAGE == 14
    /* Same read, publishing the LOW half, so one boot covers both halves of the word. */
    {
        extern char __capstone_cap_init_start[];
        extern char __gpfree_globals_base[];
        unsigned long blob_off =
            (unsigned long)((const char *)__capstone_cap_init_start
                            - (const char *)__gpfree_globals_base);
        unsigned long base;
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
        {
            volatile unsigned long anchor = 0;
            const unsigned char *here = (const unsigned char *)&anchor;
            const unsigned long *p =
                (const unsigned long *)(here - ((unsigned long)&anchor - base) + blob_off);
            *res = (unsigned)(*p);
            return;
        }
    }
#endif

#if MPY_STAGE == 12
    /* Read the blob DIRECTLY and publish what the glue would read as the cap-init delta.
       sp.BASE is dom_data.base, which IS blob offset 0 (the glue writes its "built" flag
       there with sd 0(s1)). sp is a capability, so offsetting it down to its own base
       yields a legal capability over the blob -- no new authority is created.
       MPY_BLOB_OFF selects the word; the image value for that offset is known statically,
       so a mismatch localises the copy rather than merely reporting a fault. */
    {
        unsigned long cur, base, val;
#ifndef MPY_BLOB_OFF
#define MPY_BLOB_OFF 0x8130
#endif
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x2" : "=r"(cur));
        __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
        {
            /* Plain C pointer arithmetic on a stack address: `&anchor` is a capability over
               the stack region, and offsetting it keeps the tag (an integer round trip would
               not). No inline asm, so no register-class guessing. */
            volatile unsigned long anchor = 0;
            const unsigned char *sp_here = (const unsigned char *)&anchor;
            const unsigned long *p =
                (const unsigned long *)(sp_here - ((unsigned long)&anchor - base) + MPY_BLOB_OFF);
            (void)cur;
            val = *p;
        }
        *res = (unsigned)val;   /* the raw word, not a marker: the value IS the result */
        return;
    }
#endif

#if MPY_STAGE == 10
    /* Does a .bss global work? Its storage is CARVED from dom_data by the entry glue and its
       size comes from the descriptor, so this fails if the descriptor or the carve is wrong,
       and it needs no initializer bytes. */
    mpy_heap[0] = 0x5A;
    mpy_heap[sizeof(mpy_heap) - 1] = 0xA5;
    MPY_MARK((mpy_heap[0] == 0x5A && mpy_heap[sizeof(mpy_heap) - 1] == 0xA5) ? 0xB0 : 0xE0);
#endif

#if MPY_STAGE == 11
    /* Does an INITIALIZED global work? mp_const_none_obj and the type tables live in the
       globals template, so this additionally needs the blob copy to carry real bytes, not
       just a correctly sized carve. Reading its type pointer needs cap-init to have tagged
       it, so with cap-init skipped this reads the untagged image value -- the point is only
       that the READ itself works and returns something recognisable. */
    {
        extern const mp_obj_type_t mp_type_int;
        unsigned long v = (unsigned long)mp_type_int.name;
        MPY_MARK(v ? 0xB1 : 0xE1);
    }
#endif

    mp_cstack_init_with_sp_here(mpy_cstack_size());
#if MPY_STAGE == 1
    MPY_MARK(0xA1);            /* the stack limit is set */
#endif

    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
#if MPY_STAGE == 2
    MPY_MARK(0xA2);            /* the heap exists */
#endif

#if MPY_STAGE == 3
    {   /* one allocation, before mp_init touches anything else */
        void *p = gc_alloc(64, 0);
        MPY_MARK(p ? 0xA3 : 0xE3);
    }
#endif

    mp_init();
#if MPY_STAGE == 4
    MPY_MARK(0xA4);            /* the VM state is built: qstr pool, module dict, root pointers */
#endif

#if MPY_STAGE == 5
    {   /* the lexer alone: it allocates and it walks a string literal from the cap table */
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, "1+1", 3, 0);
            nlr_pop();
            MPY_MARK(lex ? 0xA5 : 0xE5);
        }
        MPY_MARK(0xE5);        /* raised inside the lexer */
    }
#endif

#if MPY_STAGE == 6
    {   /* lexer + parser: builds a parse tree on the heap */
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, "1+1", 3, 0);
            mp_parse_tree_t pt = mp_parse(lex, MP_PARSE_FILE_INPUT);
            (void)pt;
            nlr_pop();
            MPY_MARK(0xA6);
        }
        MPY_MARK(0xE6);
    }
#endif

#if MPY_STAGE == 7
    {   /* + the compiler: emits bytecode and builds the function object */
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, "1+1", 3, 0);
            qstr src = lex->source_name;
            mp_parse_tree_t pt = mp_parse(lex, MP_PARSE_FILE_INPUT);
            mp_obj_t fun = mp_compile(&pt, src, false);
            nlr_pop();
            MPY_MARK(fun ? 0xA7 : 0xE7);
        }
        MPY_MARK(0xE7);
    }
#endif

    /* Stage 8 and the default: execute. 8 runs `1+1` with no print, so it exercises the VM
       without the output path; the default runs the real thing. */
#if MPY_STAGE == 8
    int rc = do_str("1+1", MP_PARSE_FILE_INPUT);
#elif defined(MPY_PROG)
    /* One Python construct per variant, so a hang names the construct instead of the program.
       Ordered by how much of the runtime each needs; build them all and run them in ONE boot,
       ascending, because a hang takes the rest of the boot with it. */
    static const char *const progs[] = {
        /*0*/ "print(2+3)\n",
        /*1*/ "x = [1,2,3]\nprint(len(x))\n",
        /*2*/ "def f():\n    return 7\nprint(f())\n",
        /*3*/ "s = 0\nfor i in range(10):\n    s += i\nprint(s)\n",
        /*4*/ "print('ab' * 3)\n",
        /*5*/ "try:\n    1//0\nexcept ZeroDivisionError:\n    print(9)\n",
        /*6*/ "d = {}\nd[1] = 2\nprint(d[1])\n",
        /*7*/ "import gc\ngc.collect()\nprint(8)\n",
        /*8*/ /* Force the collector to run by ALLOCATION PRESSURE rather than by calling it:
                 2000 short-lived lists against a 96 KiB heap cannot fit without a collection,
                 so this exercises gc_mark/gc_sweep on the real path. An explicit gc.collect()
                 needs the gc module, which ROM_LEVEL_MINIMUM does not enable. */
              "n = 0\nfor i in range(2000):\n    x = [i, i+1, i+2]\n    n += x[2]\nprint(n)\n",
    };
    int rc = do_str(progs[MPY_PROG], MP_PARSE_FILE_INPUT);
#elif defined(MPY_HARD_PROGRAM)
    /* A program that is not a single expression: a defined function called in a loop, a list
       comprehension, string multiplication, an exception raised and caught, and an explicit
       collection. The collection matters most -- print(1+1) allocates too little to ever run
       the GC, so the two gc.c patches in this tree are unproven at runtime without it. */
    int rc = do_str(
        "def f(n):\n"
        "    s = 0\n"
        "    for i in range(n):\n"
        "        s += i\n"
        "    return s\n"
        "a = [i*i for i in range(50)]\n"
        "b = 'x' * 100\n"
        "try:\n"
        "    1//0\n"
        "except ZeroDivisionError:\n"
        "    c = 7\n"
        "import gc\n"
        "gc.collect()\n"
        "d = {}\n"
        "for i in range(20):\n"
        "    d[i] = str(i)\n"
        "print(f(100), len(a), len(b), c, len(d), d[19], a[7])\n",
        MP_PARSE_FILE_INPUT);
#else
    int rc = do_str("print(1+1)", MP_PARSE_FILE_INPUT);
#endif
    mp_deinit();
#ifdef MPY_RETURN_OUTPUT
    /* Return the captured bytes instead of a status: byte 0 in bits 0-7, byte 1 in 8-15, and
       the length in bits 16-23. For print(1+1) that is '2' (0x32), '\n' (0x0a), length 2. */
    {
        /* length in bits 16-31, a 16-bit sum over every captured byte below it. The sum covers
           the WHOLE output, so a run that prints the right number of wrong characters still
           fails -- length alone would not catch that. */
        unsigned sum = 0;
        for (unsigned i = 0; i < mpy_cap_len; ++i) {
            sum = (sum + (unsigned char)mpy_cap_buf[i] * (i + 1)) & 0xffff;
        }
        *res = ((unsigned)mpy_cap_len << 16) | sum;
    }
    (void)rc;
#else
    *res = 0x4D500000u | (unsigned)rc;
#endif
}
