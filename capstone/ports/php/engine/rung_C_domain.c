/* Phase 2 rung C: does the SCANNER, PARSER and COMPILER run inside a domain?
 *
 * Rung B proved zend_startup completes. This rung is the next distinct subsystem: it hands
 * the engine PHP source text and asks for an op_array back. That exercises
 *
 *   - zend_language_scanner.c  (flex-generated, and the scanner state SCNG/lex save-restore)
 *   - zend_language_parser.c   (bison-generated; zendparse's frame is 16,028 bytes, the
 *                               largest in the image, so this is also the first real test
 *                               of the 512 KB stack)
 *   - zend_compile.c           (the compiler proper, and pass_two, which rewrites every
 *                               opcode operand -- the pass the plan flagged as baking
 *                               absolute pointers)
 *
 * and it is the first rung to exercise setjmp/longjmp IN ANGER: a compile error reaches
 * zend_bailout, which longjmps through EG(bailout). Without zend_try that is exit(-1)
 * (zend.c: "Bailed out without a bailout address!"), so the bailout path is not optional
 * here -- it is how PHP's own eval calls this function.
 *
 * NO "<?php" PREFIX. compile_string does BEGIN(ST_IN_SCRIPTING), i.e. the scanner starts
 * already inside script mode, exactly as eval() does. Feeding it "<?php 1;" would scan
 * "<?php" as code and fail to parse. The plan's wording says "<?php 1;"; the correct input
 * for this entry point is "1;".
 *
 * Progress is reported through *res so a fault says HOW FAR it got:
 *   bits 0..7   stage bits, below
 *   bits 8..15  op_array->last (opcode count), saturated at 255
 *   bits 16..23 number of zend_error calls seen, saturated at 255
 */
#include <zend.h>
#include <zend_API.h>
#include <zend_compile.h>

#define ST_ENTERED   0x01u   /* domain_main reached */
#define ST_SINKED    0x02u   /* output sink installed */
#define ST_STARTED   0x04u   /* zend_startup returned SUCCESS */
#define ST_ACTIVATED 0x08u   /* init_compiler returned -- the per-REQUEST tables exist */
#define ST_COMPILED  0x10u   /* compile_string returned non-NULL */
#define ST_OPS       0x20u   /* op_array has opcodes and a non-zero count */
#define ST_RETURN    0x40u   /* a ZEND_RETURN is where zend_do_return put it: at last-2 */
#define ST_EVALTYPE  0x80u   /* op_array->type == ZEND_EVAL_CODE */
#define ST_BAILED    0x100u  /* zend_catch was taken -- a compile error, NOT a pass */
#define ST_ERRORS    0x200u  /* zend_error was called at least once -- NOT a pass */

/* -DRUNG_C_BAD selects a SYNTAX ERROR instead, as the negative control: it must reach
 * zend_error and unwind through EG(bailout), which is the only path in this port that
 * actually takes a longjmp. Chosen without whitespace because EXTRA_CF is word-split. */
#ifndef RUNG_C_SOURCE
# ifdef RUNG_C_BAD
#  define RUNG_C_SOURCE "1+"
# else
#  define RUNG_C_SOURCE "1;"
# endif
#endif

extern void (*php_capstone_sink)(const char *, unsigned long);

static void sink(const char *s, unsigned long n);

/* --- the utility_functions PHP needs, same shapes as rung B --- */
static unsigned err_count;
static int      first_err_type;

static void ub_error(int type, const char *fname, const uint lineno,
                     const char *format, va_list args)
{
    (void)fname; (void)lineno; (void)format; (void)args;
    if (!err_count) { first_err_type = type; }
    ++err_count;
}
static int ub_printf(const char *fmt, ...) { (void)fmt; return 0; }
static int ub_write(const char *s, uint n)  { sink(s, n); return (int)n; }
static FILE *ub_fopen(const char *f, char **opened) { (void)f; (void)opened; return (FILE *)0; }
static void ub_message(long msg, void *data) { (void)msg; (void)data; }
static void ub_block(void)   { }
static void ub_unblock(void) { }
static int  ub_getcfg(char *name, uint nlen, zval *contents)
{ (void)name; (void)nlen; (void)contents; return FAILURE; }
static void ub_ticks(int t) { (void)t; }
static void ub_on_timeout(int sec TSRMLS_DC) { (void)sec; }
static int  ub_stream_open(const char *fn, zend_file_handle *h TSRMLS_DC)
{ (void)fn; (void)h; return FAILURE; }
static int ub_vspprintf(char **pbuf, size_t max_len, const char *fmt, va_list ap)
{
    int n = (max_len && max_len < 4096u) ? (int)max_len : 4095;
    *pbuf = (char *) malloc((unsigned long)n + 1);
    if (!*pbuf) { return 0; }
    return vsnprintf(*pbuf, (unsigned long)n + 1, fmt, ap);
}

static void sink(const char *s, unsigned long n) { (void)s; (void)n; }

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned st = 0;
    unsigned ops = 0, op_first = 0, op_last = 0;
    *res = st;

    st |= ST_ENTERED;              *res = st;
    php_capstone_sink = sink;
    st |= ST_SINKED;               *res = st;

    {
        zend_utility_functions uf;
        uf.error_function              = ub_error;
        uf.printf_function             = ub_printf;
        uf.write_function              = ub_write;
        uf.fopen_function              = ub_fopen;
        uf.message_handler             = ub_message;
        uf.block_interruptions         = ub_block;
        uf.unblock_interruptions       = ub_unblock;
        uf.get_configuration_directive = ub_getcfg;
        uf.ticks_function              = ub_ticks;
        uf.on_timeout                  = ub_on_timeout;
        uf.stream_open_function        = ub_stream_open;
        uf.vspprintf_function          = ub_vspprintf;

        if (zend_startup(&uf, (char **)0, 1) != SUCCESS) { *res = st | 0x8000u; return; }
        st |= ST_STARTED;          *res = st;
    }

    /* PER-REQUEST ACTIVATION, and it is NOT optional.
     *
     * zend_startup builds the PERSISTENT tables. CG(filenames_table) is not one of them:
     * it is created by init_compiler (zend_compile.c:153), which a SAPI reaches through
     * zend_activate() once per request. compile_string -> zend_prepare_string_for_scanning
     * -> zend_set_compiled_filename does zend_hash_find(&CG(filenames_table), ...) on its
     * filename argument, so without this the very first hash lookup runs against a zeroed
     * HashTable: ht->arBuckets is untagged and `p = ht->arBuckets[nIndex]` raises cause 24
     * (observed, at zend_hash.c:852, with the filename literal still live in a register).
     *
     * init_compiler rather than the full zend_activate: this rung only COMPILES, and
     * init_executor's per-request executor state belongs to the rung that runs opcodes. */
    init_compiler(TSRMLS_C);
    st |= ST_ACTIVATED;            *res = st;

    /* compile_string COPIES its argument: it does `tmp = *source_string; zval_copy_ctor(&tmp)`
     * and only then lets zend_prepare_string_for_scanning erealloc the COPY to len+2 for
     * flex's doubled NUL. So a .rodata literal is safe to hand it -- the read is exactly len
     * bytes, and nothing writes through our pointer. */
    {
        zend_op_array *oa = (zend_op_array *)0;
        zval src;
        static const char source[] = RUNG_C_SOURCE;

        src.type = IS_STRING;
        src.value.str.val = (char *)source;
        src.value.str.len = (int)(sizeof(source) - 1);   /* no NUL, as strlen would give */
        src.refcount = 1;
        src.is_ref = 0;

        zend_try {
            oa = compile_string(&src, "rung-C" TSRMLS_CC);
        } zend_catch {
            st |= ST_BAILED;
        } zend_end_try();

        if (oa) {
            st |= ST_COMPILED;
            if (oa->opcodes && oa->last > 0) {
                st |= ST_OPS;
                ops = (oa->last > 63u) ? 63u : (unsigned)oa->last;
                op_first = (unsigned)oa->opcodes[0].opcode;
                op_last  = (unsigned)oa->opcodes[oa->last - 1].opcode;
                /* zend_do_return appends ZEND_RETURN and zend_do_handle_exception then
                 * appends ZEND_HANDLE_EXCEPTION after it, so RETURN sits at last-2. This
                 * is the cheapest check that the COMPILER ran and not just the scanner. */
                if (oa->last >= 2u
                    && oa->opcodes[oa->last - 2].opcode == ZEND_RETURN) { st |= ST_RETURN; }
            }
            if (oa->type == ZEND_EVAL_CODE) { st |= ST_EVALTYPE; }
        }
    }

    if (err_count) { st |= ST_ERRORS; }
    *res = st
         | ((ops      & 0x3Fu) << 10)
         | ((op_first & 0xFFu) << 16)
         | ((op_last  & 0xFFu) << 24);
}
