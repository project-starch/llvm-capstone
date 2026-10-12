/* The PHP engine as an ordinary Linux process: a real main(), a trigger on stdin.
 *
 * WHAT THIS REPLACES. engine/rung_E_domain.c is the physical port's SAPI: a domain entry
 * point `domain_main(unsigned *res, unsigned func)` that reports through a pointer the monitor
 * supplies, with the trigger source compiled in by -DRUNG_E_CRASHnnn and output discarded into
 * an accumulator because the domain has no stdio. None of that survives contact with a profile
 * that has an operating system under it, so this is a new file rather than an #ifdef on that
 * one -- and engine/ stays exactly as it is, because it is the comparison column.
 *
 * TWO DELIBERATE DIFFERENCES FROM THE PHYSICAL SAPI, both of which reduce what an arm can
 * differ by:
 *
 *   1. THE TRIGGER ARRIVES ON STDIN. The physical port selects it with a -D, so a case's two
 *      arms are two different images: they differ in the allocator flag under test AND in the
 *      compiled-in source text, the string table and every address downstream of it. Here one
 *      image serves all 27 cases and the arms differ in the allocator flag alone. The sources
 *      themselves are lifted from that same #ifdef ladder by extract-triggers.py, with the C
 *      preprocessor doing the unescaping, so they are the same bytes and not a transcription.
 *
 *   2. THE SINK WRITES TO STDOUT. The physical sink sums the bytes into a volatile accumulator
 *      because there is nowhere to put them; that line was once `{ (void)s; (void)n; }` and
 *      hid a real use-after-free, since var_dump hands its output function a pointer it never
 *      dereferences itself. Writing the bytes is what a real SAPI does and reads them for
 *      real, so this is strictly the more faithful of the two.
 *
 * EVERYTHING ELSE IS KEPT, including things that look like leftovers:
 *   - the stage-bit retval, verbatim, because `reached` is derived from it and a case whose
 *     trigger never ran must not be scored;
 *   - zend_register_default_classes, which php_module_startup does (main/main.c:1438) and
 *     which this port must do by hand until Phase B links main/ -- without it CRASH-047 dies
 *     with "Class 'X' not found" before the opcodes under test run;
 *   - zend_deactivate, the request teardown, because CRASH-047's use-after-free fires inside
 *     it and a port that never tears a request down is not running the same program;
 *   - the zend_utility_functions table, unchanged in behaviour, so Phase A's only variable is
 *     the profile. Phase B replaces the whole table with PHP's real main/ and sapi/cli.
 *
 * THE STAGE LINES ARE PRINTED AND FLUSHED AS THEY HAPPEN, not summarised at the end. A
 * capability fault ends the process, so a summary that is only printed on the way out is
 * exactly the evidence that a faulting run cannot produce.
 */
#include <zend.h>
#include "zend_alloc.h"
#include <zend_API.h>
#include <zend_compile.h>
#include <zend_execute.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "php_virtual_port.h"

/* THE GEOMETRY THE PORT DERIVES FROM, checked against PHP's own header rather than assumed.
 * zend_cap_widen steps back ZEND_CAP_HEADER_BYTES to find the block the heap handed out; if
 * that is not exactly sizeof(zend_mem_header) it looks up the wrong block and _efree reads a
 * neighbour's header. A differently configured tree must therefore fail to BUILD. */
PHP_VIRTUAL_ASSERT(header_is_48, sizeof(zend_mem_header) == ZEND_CAP_HEADER_BYTES);
PHP_VIRTUAL_ASSERT(header_padding_is_0, MEM_HEADER_PADDING == 0);

/* Same bits and the same values as engine/rung_E_domain.c, so a verdict derived from one is
 * comparable with a verdict derived from the other. */
#define ST_ENTERED    0x001u
#define ST_SINKED     0x002u
#define ST_STARTED    0x004u
#define ST_COMPILER   0x008u
#define ST_EXECUTOR   0x010u
#define ST_REGISTERED 0x020u
#define ST_EVALED     0x040u
#define ST_ISARRAY    0x080u
#define ST_BAILED     0x100u
#define ST_ERRORS     0x200u

/* The arm, named out loud. Verification item 5 of the retarget plan requires that the two runs
 * of a pair report DIFFERENT mode lines -- a pair that silently built the same way is the
 * cheapest way to manufacture a result, and the physical port's own gate exists because of it.
 * In hier mode the arm is still a compile-time flag, exactly as it is physically; the runtime
 * mode bits arrive with the allocator seam. */
#ifdef ZEND_CAP_BOUNDS_REAL_SIZE
#  define PHP_VIRTUAL_ARM "bounds=real-size (CONTROL: stock PHP's rounding is reachable)"
#else
#  define PHP_VIRTUAL_ARM "bounds=exact (FAULT: every capability ends at the caller's request)"
#endif

static unsigned err_count;
static unsigned long sink_bytes;

/* THE SINK MUST READ ITS BYTES -- see the header. fwrite does, for real. */
static void sink(const char *s, unsigned long n)
{
    if (n) { fwrite(s, 1, (size_t)n, stdout); }
    sink_bytes += n;
}

static void ub_error(int type, const char *fname, const uint lineno,
                     const char *format, va_list args)
{
    /* THE MESSAGE IS FORMATTED, NOT DISCARDED. Every real SAPI formats it (php_error_cb ->
     * vspprintf), and at least one corpus defect lives inside that formatting: CRASH-068 frees
     * a constant's name and hands the same pointer to an E_NOTICE as %s, so the use-after-free
     * IS the formatter's read. A handler that ignored `format` would report that case MISSED
     * for a reason with nothing to do with capabilities. */
    char msgbuf[256];
    int mlen;
    (void)fname; (void)lineno;
    mlen = vsnprintf(msgbuf, sizeof msgbuf, format, args);
    if (mlen > 0) { sink(msgbuf, (unsigned long)(mlen < 255 ? mlen : 255)); }
    ++err_count;
    switch (type) {            /* mirrors php_error_cb (main/main.c:773-781) */
        case E_CORE_ERROR:
        case E_ERROR:
        case E_COMPILE_ERROR:
        case E_USER_ERROR:
            EG(exit_status) = 255;
            zend_bailout();
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

/* ext/standard has no module to register here, so the functions the triggers name are
 * introduced directly. Identical to the physical table, including the no-op var_dump for the
 * builds where the real one is not linked: an UNDEFINED call bails out of the eval before the
 * opcodes under test run, which would score as not-reached. */
extern void zif_parse_url(INTERNAL_FUNCTION_PARAMETERS);

static void zif_rung_nop(INTERNAL_FUNCTION_PARAMETERS)
{
    (void)ht; (void)return_value; (void)this_ptr; (void)return_value_used;
}

#if defined(PHP_CAP_EXT_PURE)
extern void zif_var_dump(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_str_repeat(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_strtolower(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_chr(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_strspn(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_stristr(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_count(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_usort(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_array_walk_recursive(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_array_reduce(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_extract(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_unserialize(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_htmlspecialchars(INTERNAL_FUNCTION_PARAMETERS);
extern void zif_user_sprintf(INTERNAL_FUNCTION_PARAMETERS);
#endif

static zend_function_entry php_virtual_functions[] = {
    { "parse_url", zif_parse_url,  (struct _zend_arg_info *)0, 0, 0 },
#if defined(PHP_CAP_EXT_PURE)
    { "var_dump",  zif_var_dump,   (struct _zend_arg_info *)0, 0, 0 },
    { "str_repeat",           zif_str_repeat,           (struct _zend_arg_info *)0, 0, 0 },
    { "strtolower",           zif_strtolower,           (struct _zend_arg_info *)0, 0, 0 },
    { "chr",                  zif_chr,                  (struct _zend_arg_info *)0, 0, 0 },
    { "strspn",               zif_strspn,               (struct _zend_arg_info *)0, 0, 0 },
    { "stristr",              zif_stristr,              (struct _zend_arg_info *)0, 0, 0 },
    { "count",                zif_count,                (struct _zend_arg_info *)0, 0, 0 },
    { "usort",                zif_usort,                (struct _zend_arg_info *)0, 0, 0 },
    { "array_walk_recursive", zif_array_walk_recursive, (struct _zend_arg_info *)0, 0, 0 },
    { "array_reduce",         zif_array_reduce,         (struct _zend_arg_info *)0, 0, 0 },
    { "extract",              zif_extract,              (struct _zend_arg_info *)0, 0, 0 },
    { "unserialize",          zif_unserialize,          (struct _zend_arg_info *)0, 0, 0 },
    { "htmlspecialchars",     zif_htmlspecialchars,     (struct _zend_arg_info *)0, 0, 0 },
    { "sprintf",              zif_user_sprintf,         (struct _zend_arg_info *)0, 0, 0 },
#else
    { "var_dump",  zif_rung_nop,   (struct _zend_arg_info *)0, 0, 0 },
#endif
    { (char *)0,   (void (*)(INTERNAL_FUNCTION_PARAMETERS))0, (struct _zend_arg_info *)0, 0, 0 }
};

/* THE ARM'S OWN POSITIVE CONTROL, run through the same engine in the same state as a case.
 *
 * A bound that never fires and a program with no defect are the same clean run, and this
 * project has published that confusion before. So each arm carries a probe whose expected
 * outcome is OPPOSITE on the two arms, and the judge is told which to expect
 * (bug-corpora/tools/arms.json `controls`):
 *
 *   bounds=real-size (control)  -> COMPLETES. REAL_SIZE(9) is 16, so index 9 is inside the
 *                                  rounding slack -- which is precisely the mechanism that
 *                                  hides these cases from ASAN on a stock allocator.
 *   bounds=exact     (fault)    -> FAULTS. The capability ends at the 9 bytes requested.
 *
 * The geometry is CRASH-110's, deliberately: a 9-byte string read at index 9. So the control
 * does not merely show that some bound somewhere works, it shows that THE bound this case
 * depends on is reachable in this image, on this arm, in this run.
 *
 * Returning 0 here is not a pass by itself -- what it means depends on the arm, which is the
 * judge's business and not this file's. */
static int run_bound_control(void)
{
    volatile unsigned char seen;
    char *s = (char *) emalloc(9);
    if (!s) { printf("CONTROL FATAL emalloc(9) returned NULL\n"); fflush(stdout); return 3; }
    printf("CONTROL emalloc(9) bounds=[0x%lx,0x%lx) cursor=0x%lx\n",
           __builtin_capstone_cap_get_base(s), __builtin_capstone_cap_get_end(s),
           __builtin_capstone_cap_get_cursor(s));
    /* In-bounds first: a fault HERE would not be the control firing, it would be the port
     * broken, and the two must not be confusable. */
    memset(s, 'a', 9);
    printf("CONTROL wrote 9 bytes in bounds; now reading index 9\n");
    fflush(stdout);
    seen = ((volatile unsigned char *)s)[9];
    printf("CONTROL read s[9]=0x%02x -- NO FAULT: the rounding slack is reachable\n",
           (unsigned)seen);
    efree(s);
    printf("CONTROL completed\n");
    fflush(stdout);
    return 0;
}

static unsigned stage;

static void stage_line(const char *what)
{
    printf("PHP_VIRTUAL stage=0x%03x %s\n", stage, what);
    fflush(stdout);
}

/* The trigger. Read with plain malloc, OUTSIDE PHP's allocator, so nothing this file does
 * perturbs the allocation history the case under test depends on. zend_eval_string copies the
 * text into an emalloc'd buffer before compiling it (zend_execute_API.c:957-966, both
 * branches), so PHP's scanner never reads this buffer and its bounds cannot mask or
 * manufacture a fault in the trigger text itself. */
static char *read_trigger(unsigned long *len_out)
{
    unsigned long cap = 4096, n = 0;
    char *buf = (char *) malloc(cap);
    if (!buf) { return (char *)0; }
    for (;;) {
        size_t got = fread(buf + n, 1, (size_t)(cap - n - 1), stdin);
        n += (unsigned long)got;
        if (n + 1 < cap) { break; }                 /* short read: end of input */
        cap *= 2;
        { char *bigger = (char *) realloc(buf, cap);
          if (!bigger) { free(buf); return (char *)0; }
          buf = bigger; }
    }
    buf[n] = '\0';
    *len_out = n;
    return buf;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "expr";
    const char *label = (argc > 2) ? argv[2] : "trigger";
    int is_stmt, is_control;
    unsigned rtype = 0;
    unsigned long trigger_len = 0;
    char *trigger;

    printf("PHP_VIRTUAL arm %s\n", PHP_VIRTUAL_ARM);
    printf("PHP_VIRTUAL case=%s mode=%s\n", label, mode);
    fflush(stdout);

    if (strcmp(mode, "stmt") != 0 && strcmp(mode, "expr") != 0
        && strcmp(mode, "control") != 0) {
        fprintf(stderr, "usage: %s <expr|stmt|control> [case-label] < trigger.php\n", argv[0]);
        return 2;
    }
    is_stmt = (strcmp(mode, "stmt") == 0);
    is_control = (strcmp(mode, "control") == 0);

    if (is_control) {
        /* No trigger on stdin for the control, and it must not read one: a control that
         * consumed a case's input would make the two indistinguishable in the transcript. */
        trigger = (char *)0;
    } else {
        trigger = read_trigger(&trigger_len);
    }
    if (!is_control && (!trigger || !trigger_len)) {
        /* An empty trigger must be an ERROR and not a quiet pass: a run that evaluated
         * nothing would otherwise complete cleanly and read as a control that held. */
        fprintf(stderr, "PHP_VIRTUAL FATAL: no trigger on stdin\n");
        return 2;
    }
    if (!is_control) {
        printf("PHP_VIRTUAL trigger bytes=%lu\n", trigger_len);
        fflush(stdout);
    }

    stage |= ST_ENTERED;  stage_line("entered");
    /* ST_SINKED is kept so the stage word stays comparable with the physical arm's, but there
     * is nothing to install here. Physically the bit records pointing php_capstone_sink at the
     * SAPI, because libc/php_capstone_stdio.c routes the domain's printf through it; on this
     * profile stdio is musl's and PHP's output reaches stdout through write_function below. An
     * undefined WEAK reference to php_capstone_sink would also be refused outright by the
     * C-56 preflight in run-domain-smoke.py, which is the right call -- a weak symbol nobody
     * defines is a silently dead code path. */
    stage |= ST_SINKED;   stage_line("sink is the SAPI write_function (no stdio shim here)");

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
        if (zend_startup(&uf, (char **)0, 1) != SUCCESS) {
            printf("PHP_VIRTUAL FATAL zend_startup failed\n");
            fflush(stdout);
            return 3;
        }
        zend_register_default_classes(TSRMLS_C);
        stage |= ST_STARTED; stage_line("zend_startup + default classes");
    }

    init_compiler(TSRMLS_C);
    stage |= ST_COMPILER;  stage_line("init_compiler");
    init_executor(TSRMLS_C);
    stage |= ST_EXECUTOR;  stage_line("init_executor");

    if (zend_register_functions((zend_class_entry *)0, php_virtual_functions,
                                CG(function_table), MODULE_PERSISTENT TSRMLS_CC) != SUCCESS) {
        printf("PHP_VIRTUAL FATAL zend_register_functions failed\n");
        fflush(stdout);
        return 4;
    }
    stage |= ST_REGISTERED; stage_line("functions registered");

    if (is_control) {
        int crc = run_bound_control();
        printf("PHP_VIRTUAL CONTROL_DONE rc=%d\n", crc);
        printf("PHP_VIRTUAL_DONE\n");
        fflush(stdout);
        return crc;
    }

    {
        zval ret;
        int rc = FAILURE;

        INIT_ZVAL(ret);
        stage_line("evaluating the trigger");
        zend_try {
            if (is_stmt) {
                /* Statements get no retval, so zend_eval_string does not wrap them in
                 * `return ... ;` -- a statement list does not parse inside that wrapper. */
                rc = zend_eval_string(trigger, (zval *)0, (char *)"trigger" TSRMLS_CC);
            } else {
                rc = zend_eval_string(trigger, &ret, (char *)"trigger" TSRMLS_CC);
            }
        } zend_catch {
            stage |= ST_BAILED;
        } zend_end_try();
        stage_line("trigger returned");

#if !defined(PHP_CAP_NO_REQUEST_SHUTDOWN)
        /* THE REQUEST TEARDOWN, on by default. main/main.c:1155 is the only caller of
         * zend_deactivate, and main/ is not linked in Phase A, so without this line nothing
         * ever tears a request down -- and CRASH-047's use-after-free fires during exactly
         * that teardown (shutdown_executor -> zend_objects_store_free_object_storage). */
        zend_deactivate(TSRMLS_C);
        stage_line("request torn down");
#endif

        if (rc == SUCCESS) {
            stage |= ST_EVALED;
            if (is_stmt) {
                stage |= ST_ISARRAY;    /* statement cases: completion IS the pass condition */
            } else {
                rtype = (unsigned)ret.type;
                if (ret.type == IS_ARRAY) { stage |= ST_ISARRAY; }
            }
        }
    }

    if (err_count) { stage |= ST_ERRORS; }

    printf("PHP_VIRTUAL RESULT stage=0x%03x rtype=%u errors=%u output_bytes=%lu\n",
           stage, rtype, err_count, sink_bytes);
    printf("PHP_VIRTUAL retval=0x%08x\n",
           stage | ((rtype & 0xFu) << 10)
                 | (((err_count > 255u) ? 255u : err_count) << 16));
    printf("PHP_VIRTUAL_DONE\n");
    fflush(stdout);
    return 0;
}
