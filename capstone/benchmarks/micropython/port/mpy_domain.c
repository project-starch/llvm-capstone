#include <stdint.h>
#include <string.h>
#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/cstack.h"
#include "py/mphal.h"
#include "py/stackctrl.h"

/* ---- the GC heap. One global; under -capstone-gp-captable its storage is CARVED
   from dom_data at entry, so it is charged against the domain's stack budget.
   32-byte aligned and a multiple of 32 so gc_init's align-down is a no-op. */
#ifndef MPY_HEAP_SIZE
#define MPY_HEAP_SIZE (96U * 1024U)
#endif
static unsigned char mpy_heap[MPY_HEAP_SIZE] __attribute__((aligned(32)));

/* ---- output: the hostcall shared region, same shape as benchmarks/sqlite */
#define MPY_DPI_REGION_SHARE 1U
struct mpy_hostcall_v0 { unsigned long long phase, opcode, offset, length; long long result, error; };
#define MPY_HC_REGION_SIZE 4096UL
static volatile struct mpy_hostcall_v0 *hc_meta;
static volatile char *hc_payload;
static unsigned hc_share_count;

mp_uint_t mp_hal_stdout_tx_strn(const char *str, size_t len) {
    if (!hc_meta || !hc_payload) return 0;
    char *payload = (char *)hc_payload;
    unsigned long off = hc_meta->length;
    while (len-- && off + 1 < MPY_HC_REGION_SIZE) payload[off++] = *str++;
    hc_meta->length = off;
    return off;
}

/* ---- what py/ demands and the port must supply ---- */
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

void domain_main(unsigned *res, unsigned func) {
    if (func == MPY_DPI_REGION_SHARE) {
        if (hc_share_count == 0) hc_meta = (volatile struct mpy_hostcall_v0 *)res;
        else if (hc_share_count == 1) hc_payload = (volatile char *)res;
        ++hc_share_count;
        return;
    }
    if (hc_meta) hc_meta->length = 0;

#if MPY_STAGE == 0
    MPY_MARK(0xA0);            /* entered, nothing else -- NOTE it touches no global at all,
                                  so it does NOT prove the carve loop or the blob copy worked */
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

    mp_cstack_init_with_sp_here(cap_stack_headroom());
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
#else
    int rc = do_str("print(1+1)", MP_PARSE_FILE_INPUT);
#endif
    mp_deinit();
    *res = 0x4D500000u | (unsigned)rc;
}
