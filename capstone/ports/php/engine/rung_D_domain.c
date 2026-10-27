/* Phase 2 rung D: does PHP EXECUTE? This is the milestone the plan calls "PHP executes".
 *
 * Rung C proved the scanner, parser and compiler produce an op_array. This rung runs it,
 * through zend_eval_string -- PHP's own compile-and-execute entry point (zend_execute_API.c
 * :948), so the executor is driven exactly as the engine drives it rather than by a
 * hand-rolled imitation of the EG(return_value_ptr_ptr) / EG(active_op_array) dance.
 *
 * With a non-NULL retval_ptr, zend_eval_string wraps the source as `return <src> ;`. So the
 * source here is an EXPRESSION, "6*7", and a pass means the VM came back with the long 42:
 * the opcode dispatch table, the temp-variable ABI (EX_T), ZEND_MUL on two constants, and
 * ZEND_RETURN copying a zval out all worked.
 *
 * THREE THINGS THIS RUNG NEEDS THAT THE EARLIER ONES DID NOT:
 *
 *  1. init_executor() (zend_execute_API.c:117) alongside init_compiler(). zend_startup builds
 *     the PERSISTENT tables; the per-request executor state -- EG(symbol_table), the argument
 *     and arg_types stacks, the symtable cache, EG(function_table)/EG(class_table) aliases --
 *     comes from here. A SAPI reaches both through zend_activate().
 *
 *  2. ub_error MUST call zend_bailout() for the fatal types. zend_error NEVER bails on its own
 *     (grep count: 0); php_error_cb does, at main/main.c:773-781. Our rung-C callback only
 *     counted errors, which means a genuine E_ERROR did not stop the engine -- it reported and
 *     returned, and execution continued past a fatal error. Harmless while only compiling,
 *     wrong the moment opcodes run. E_PARSE is deliberately EXCLUDED, exactly as php_error_cb
 *     excludes it, because the parser reports failure by return value instead.
 *
 *  3. Consequently this is the first rung where setjmp/longjmp is load-bearing: the bailout
 *     unwinds through EG(bailout), which zend_try installs. -DRUNG_D_FATAL is the negative
 *     control that forces that path.
 *
 * Reported through *res:
 *   bits 0..9    stage bits
 *   bits 10..13  retval zval type   (IS_NULL 0, IS_LONG 1, IS_DOUBLE 2, IS_STRING 3)
 *   bits 16..31  retval long value, low 16 bits
 */
#include <zend.h>
#include <zend_API.h>
#include <zend_compile.h>
#include <zend_execute.h>

#define ST_ENTERED   0x001u
#define ST_SINKED    0x002u
#define ST_STARTED   0x004u  /* zend_startup returned SUCCESS */
#define ST_COMPILER  0x008u  /* init_compiler returned */
#define ST_EXECUTOR  0x010u  /* init_executor returned */
#define ST_EVALED    0x020u  /* zend_eval_string returned SUCCESS */
#define ST_ISLONG    0x040u  /* the returned zval is IS_LONG */
#define ST_IS42      0x080u  /* ... and its value is 42 */
#define ST_BAILED    0x100u  /* zend_catch taken -- expected ONLY under RUNG_D_FATAL */
#define ST_ERRORS    0x200u  /* zend_error seen -- expected ONLY under RUNG_D_FATAL */

/* "6*7" evaluates to 42. -DRUNG_D_FATAL substitutes a COMPILE-time fatal (redeclaring a
 * function is E_COMPILE_ERROR, not E_PARSE), which must reach zend_bailout and unwind.
 * No whitespace: EXTRA_CF is word-split. */
#ifndef RUNG_D_SOURCE
# ifdef RUNG_D_FATAL
#  define RUNG_D_SOURCE "0;function f(){}function f(){}"
# else
#  define RUNG_D_SOURCE "6*7"
# endif
#endif

extern void (*php_capstone_sink)(const char *, unsigned long);

static void sink(const char *s, unsigned long n) { (void)s; (void)n; }

static unsigned err_count;
static int      first_err_type;

/* Mirrors php_error_cb (main/main.c:773-781): bail on the fatal types, and NOT on E_PARSE. */
static void ub_error(int type, const char *fname, const uint lineno,
                     const char *format, va_list args)
{
    (void)fname; (void)lineno; (void)format; (void)args;
    if (!err_count) { first_err_type = type; }
    ++err_count;
    switch (type) {
        case E_CORE_ERROR:
        case E_ERROR:
        case E_COMPILE_ERROR:
        case E_USER_ERROR:
            EG(exit_status) = 255;
            zend_bailout();
            /* not reached */
        default:
            break;
    }
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

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    unsigned st = 0;
    unsigned rtype = 0, rlow = 0;
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

    init_compiler(TSRMLS_C);
    st |= ST_COMPILER;             *res = st;
    init_executor(TSRMLS_C);
    st |= ST_EXECUTOR;             *res = st;

    {
        zval ret;
        int rc = FAILURE;
        static char source[] = RUNG_D_SOURCE;   /* not const: zend_eval_string takes char* */

        INIT_ZVAL(ret);
        zend_try {
            rc = zend_eval_string(source, &ret, "rung-D" TSRMLS_CC);
        } zend_catch {
            st |= ST_BAILED;
        } zend_end_try();

        if (rc == SUCCESS) {
            st |= ST_EVALED;
            rtype = (unsigned)ret.type;
            if (ret.type == IS_LONG) {
                st |= ST_ISLONG;
                rlow = (unsigned)(ret.value.lval & 0xFFFF);
                if (ret.value.lval == 42) { st |= ST_IS42; }
            }
        }
    }

    if (err_count) { st |= ST_ERRORS; }
    *res = st | ((rtype & 0xFu) << 10) | ((rlow & 0xFFFFu) << 16);
}
