/* Stage bisect of zend_startup's body.
 *
 * WHY THIS EXISTS. rung B exhausts ~2.6 MB of stack inside zend_startup. Everything cheaper
 * has been eliminated by experiment (see FINDINGS.md): not the image ceiling, not a small
 * stack, not the support layer, not longjmp, not a static function-pointer table, not one
 * huge frame, not a bogus dynamic sp adjustment, and the excursion calls neither malloc nor
 * alloca -- so no watchdog sited in an allocator can see it. The remaining reliable move is
 * to run zend_startup's steps one at a time and see which one blows.
 *
 * The sequence below mirrors Zend/zend.c:538-655 (the non-ZTS path) in order.
 * -DRUNG_S_STAGE=N runs stages 1..N and returns 0xS000|N on success.
 *
 * TWO STEPS CANNOT BE REPLICATED: register_standard_class() and
 * zend_set_default_compile_time_values() are static in zend.c. Skipping the former leaves
 * zend_standard_class_def unset, so a failure in stage 9 (zend_register_default_classes)
 * could be that rather than the bug under test -- an untagged/NULL class-def dereference
 * looks nothing like a stack overflow, so the two are still distinguishable, but do not
 * read stage 9 as conclusive on its own.
 */
#include <zend.h>
#include <zend_API.h>
#include <zend_constants.h>
#include <zend_list.h>
#include <zend_builtin_functions.h>
#include <zend_modules.h>
#include <zend_extensions.h>
#include <zend_ini.h>

#ifndef RUNG_S_STAGE
#define RUNG_S_STAGE 1
#endif

extern void (*php_capstone_sink)(const char *, unsigned long);
static unsigned long sink_bytes;
static void sink(const char *s, unsigned long n) { (void)s; sink_bytes += n; }

/* --- the utility functions, same as rung B --- */
static void ub_error(int t, const char *f, const uint l, const char *fmt, va_list ap)
{ (void)t;(void)f;(void)l;(void)fmt;(void)ap; }
static int  ub_printf(const char *f, ...) { (void)f; return 0; }
static int  ub_write(const char *s, uint n) { sink(s, n); return (int)n; }
static FILE *ub_fopen(const char *f, char **o) { (void)f;(void)o; return (FILE *)0; }
static void ub_message(long m, void *d) { (void)m;(void)d; }
static void ub_block(void) { }
static void ub_unblock(void) { }
static int  ub_getcfg(char *n, uint l, zval *c) { (void)n;(void)l;(void)c; return FAILURE; }
static void ub_ticks(int t) { (void)t; }
static void ub_on_timeout(int s TSRMLS_DC) { (void)s; }
static int  ub_stream_open(const char *f, zend_file_handle *h TSRMLS_DC)
{ (void)f;(void)h; return FAILURE; }
int vsnprintf(char *, unsigned long, const char *, va_list);
void *malloc(unsigned long);
static int ub_vspprintf(char **pbuf, size_t max_len, const char *fmt, va_list ap)
{
    char probe[1];
    int n = vsnprintf(probe, 1, fmt, ap);
    if (n < 0) { n = 0; }
    if (max_len && (size_t)n > max_len) { n = (int)max_len; }
    *pbuf = (char *) malloc((unsigned long)n + 1);
    if (!*pbuf) { return 0; }
    return vsnprintf(*pbuf, (unsigned long)n + 1, fmt, ap);
}

/* zend.c's own externs for the two scanner globals it constructs. */
extern zend_scanner_globals ini_scanner_globals;
extern zend_scanner_globals language_scanner_globals;

/* --- depth watchdog, reporting through the fault channel --- */
extern unsigned long php_depth_floor, php_depth_low, php_depth_frames;
extern unsigned long php_depth_mode_ra, php_depth_mode_count;
extern int           php_depth_armed;
extern void        (*php_depth_trip_fn)(void);
extern void          php_fault_report(unsigned long);
extern int           zend_startup(zend_utility_functions *, char **, int);

static void trip(void)
{
    /* No longjmp: the fault channel does not depend on unwinding.
     *   [63:56] 0xD1  [55:40] frames  [39:24] repeat count  [23:2] offset from zend_startup */
    unsigned long refv = __builtin_capstone_cap_get_cursor((void *)&zend_startup);
    unsigned long off  = (php_depth_mode_ra >= refv) ? (php_depth_mode_ra - refv)
                                                     : (refv - php_depth_mode_ra);
    unsigned long neg  = (php_depth_mode_ra >= refv) ? 0UL : 1UL;
    php_fault_report((0xD1UL << 56)
                   | ((php_depth_frames     & 0xFFFFUL) << 40)
                   | ((php_depth_mode_count & 0xFFFFUL) << 24)
                   | ((off & 0x3FFFFFUL) << 2) | (neg << 1));
}

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    php_capstone_sink = sink;
#ifdef RUNG_S_WATCH
    {
        void *fp = __builtin_frame_address(0);
        php_depth_low     = __builtin_capstone_cap_get_cursor(fp);
        php_depth_floor   = php_depth_low - (64UL * 1024UL);
        php_depth_trip_fn = trip;
        php_depth_armed   = 1;
    }
#endif

#if RUNG_S_STAGE >= 1
    zend_startup_extensions_mechanism();
#endif

#if RUNG_S_STAGE >= 2
    /* zend.c:563-585 -- the utility wiring. Assigned before anything that might call
     * zend_error, so an error inside a later stage reaches ub_error rather than a NULL. */
    zend_error_cb                      = ub_error;
    zend_printf                        = ub_printf;
    zend_write                         = (zend_write_func_t) ub_write;
    zend_fopen                         = ub_fopen;
    zend_block_interruptions           = ub_block;
    zend_unblock_interruptions         = ub_unblock;
    zend_ticks_function                = ub_ticks;
    zend_on_timeout                    = ub_on_timeout;
    zend_vspprintf                     = ub_vspprintf;
    zend_stream_open_function          = ub_stream_open;
    /* zend_message_dispatcher_p and zend_get_configuration_directive_p are static in
     * zend.c and cannot be set from here. Nothing in stages 1-9 is expected to dispatch a
     * message or read an ini directive; if a stage fails through one of those it will be a
     * NULL call, not a stack overflow, so it stays distinguishable. */
    (void)ub_message; (void)ub_getcfg;
    zend_init_opcodes_handlers();
#endif

#if RUNG_S_STAGE >= 3
    /* zend.c:592-602. Sub-bisected with RUNG_S_SUB because stage 3 as a whole fails and it
     * is eight lines; SUB=0 is the allocations alone, each higher value adds one step. */
#ifndef RUNG_S_SUB
#define RUNG_S_SUB 99
#endif
    CG(function_table)    = (HashTable *) malloc(sizeof(HashTable));
    CG(class_table)       = (HashTable *) malloc(sizeof(HashTable));
    CG(auto_globals)      = (HashTable *) malloc(sizeof(HashTable));
    if (!CG(function_table) || !CG(class_table) || !CG(auto_globals)) { *res = 0x5E01u; return; }
#if RUNG_S_SUB >= 1
#ifdef RUNG_S_NULL_DTOR
    /* Same call with a NULL destructor: isolates whether materialising the cast function
     * pointer ZEND_FUNCTION_DTOR ((void(*)(void*)) zend_function_dtor) is the trigger. */
    zend_hash_init_ex(CG(function_table), 100, NULL, NULL, 1, 0);
#else
    zend_hash_init_ex(CG(function_table), 100, NULL, ZEND_FUNCTION_DTOR, 1, 0);
#endif
#endif
#if RUNG_S_SUB >= 2
    zend_hash_init_ex(CG(class_table), 10, NULL, ZEND_CLASS_DTOR, 1, 0);
#endif
#if RUNG_S_SUB >= 3
    zend_hash_init_ex(&module_registry, 50, NULL, ZEND_MODULE_DTOR, 1, 0);
#endif
#if RUNG_S_SUB >= 4
    zend_init_rsrc_list_dtors();
#endif
#if RUNG_S_SUB >= 5
    zval_used_for_init.is_ref   = 0;
    zval_used_for_init.refcount = 1;
    zval_used_for_init.type     = IS_NULL;
#endif
#endif

#if RUNG_S_STAGE >= 4
    /* zend.c:632-636 */
    zend_hash_init_ex(CG(auto_globals), 8, NULL, (dtor_func_t) zend_auto_global_dtor, 1, 0);
    /* scanner_globals_ctor is static in zend.c:521; replicate its body (it only clears
     * six fields) rather than skipping it, since the scanner reads them. */
    ini_scanner_globals.c_buf_p = (char *) 0;
    ini_scanner_globals.init = 1;
    ini_scanner_globals.start = 0;
    ini_scanner_globals.current_buffer = NULL;
    ini_scanner_globals.yy_in = NULL;
    ini_scanner_globals.yy_out = NULL;
    language_scanner_globals.c_buf_p = (char *) 0;
    language_scanner_globals.init = 1;
    language_scanner_globals.start = 0;
    language_scanner_globals.current_buffer = NULL;
    language_scanner_globals.yy_in = NULL;
    language_scanner_globals.yy_out = NULL;
    zend_startup_constants();
#endif

#if RUNG_S_STAGE >= 5
    zend_register_standard_constants(TSRMLS_C);
#endif

#if RUNG_S_STAGE >= 6
    zend_register_auto_global("GLOBALS", sizeof("GLOBALS") - 1, NULL TSRMLS_CC);
#endif

#if RUNG_S_STAGE >= 7
    zend_init_rsrc_plist(TSRMLS_C);
#endif

#if RUNG_S_STAGE >= 8
    zend_startup_builtin_functions(TSRMLS_C);
#endif

#if RUNG_S_STAGE >= 9
    /* zend.c:653 -- the LAST call in zend_startup. Corrected: zend_register_default_classes
     * is NOT part of zend_startup (php_module_startup calls it later), so it moved to 10. */
    zend_ini_startup(TSRMLS_C);
#endif

#if RUNG_S_STAGE >= 10
    /* Beyond zend_startup, and NOT a fair test on its own: register_standard_class() is
     * static in zend.c and was skipped, so zend_standard_class_def is unset here. */
    zend_register_default_classes(TSRMLS_C);
#endif

    *res = 0x5000u | (unsigned)RUNG_S_STAGE;
}
