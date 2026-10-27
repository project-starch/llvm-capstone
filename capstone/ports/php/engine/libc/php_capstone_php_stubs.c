/* Stand-ins for the three PHP translation units deliberately left out of the build.
 *
 * THIS FILE INCLUDES PHP'S OWN HEADERS ON PURPOSE. An earlier version declared
 * `ini_scanner_globals` as an opaque `unsigned long[16]` -- 128 bytes -- which was WRONG
 * and cost a debugging cycle:
 *
 *   zend_startup() calls scanner_globals_ctor(&ini_scanner_globals) at Zend/zend.c:634,
 *   and the real struct _zend_scanner_globals (Zend/zend_globals.h:274) holds EIGHT
 *   pointers plus a dozen ints. On this target a pointer is 16 bytes, so the pointers
 *   alone are 128 bytes and the constructor wrote well past the guess into whatever .bss
 *   followed it. The resulting fault was a stack-bounds violation in an unrelated
 *   function, which looked like runaway recursion and was not.
 *
 * Rule for any future stub of a PHP global: take the TYPE from PHP's header. Never size a
 * struct by eye on a target where every pointer is twice as wide as the author assumed.
 */
#include <zend.h>
#include <zend_globals.h>
#include <zend_highlight.h>

/* Zend/zend_ini_scanner.c -- php.ini TEXT parsing. The domain hardcodes its ini defaults
 * so no ini file is ever scanned, but zend_startup constructs this unconditionally. */
ZEND_API zend_scanner_globals ini_scanner_globals;

/* Zend/zend_reflection_api.c -- 3426 lines, the single largest object (57 KB .text),
 * not on the execution path. Referenced only by its PHP_MINIT call. */
ZEND_API void zend_register_reflection_api(TSRMLS_D) { }

/* Zend/zend_highlight.c -- reachable only via highlight_file()/highlight_string(). */
ZEND_API void zend_highlight(zend_syntax_highlighter_ini *syntax_highlighter_ini TSRMLS_DC)
{ (void)syntax_highlighter_ini; }

#if defined(PHP_CAP_ALLOC_SEAM)
/* The two symbols the engine still imports from Zend/zend_alloc.c, which PHP_CAP_ALLOC_SEAM
 * drops. They live here rather than beside the allocator because they need PHP's headers, and
 * libc/php_capstone_malloc.c deliberately does not include them (it defines its own size_t).
 *
 * alloc_globals: exported by zend_alloc.c and linked against, though AG() itself is referenced
 * only from inside that TU. Zeroed and never read here -- notably its `cache` is unused, which
 * is the point of the re-seam: PHP's allocator cache is gone, so an emalloc/efree pair is now
 * visible to the capability allocator instead of being absorbed by a recycling cache.
 *
 * start_memory_manager: a no-op. zend_arena_init() runs lazily on the first carve, so there is
 * nothing to start. TSRMLS_D is `void` in this non-ZTS build. */
ZEND_API zend_alloc_globals alloc_globals;

ZEND_API void start_memory_manager(TSRMLS_D) { }
#endif

#if defined(PHP_CAP_EXT_PURE)
/* ---------------------------------------------------------------------------
 * Stand-ins for the globals that ext/standard's pure TUs reference through PG(), BG(), SG().
 *
 * These are DEFINITIONS AT PHP'S OWN TYPE, per the rule at the top of this file: the structs are
 * large and full of pointers, and guessing a size on a 16-byte-pointer target corrupts whatever
 * .bss follows. Zeroed is correct for every one of them here -- the trigger paths read ini
 * settings and request state that a domain does not have, and a zeroed struct is what "unset"
 * looks like. Nothing writes them, because no module init runs.
 * ------------------------------------------------------------------------- */
#include <php.h>
#include <SAPI.h>
#include <php_globals.h>
#include <ext/standard/basic_functions.h>

PHPAPI php_core_globals   core_globals;     /* PG()  -- main/php_globals.h */
PHPAPI sapi_globals_struct sapi_globals;    /* SG()  -- main/SAPI.h */
PHPAPI sapi_module_struct  sapi_module;     /* the SAPI itself; rung E drives the engine directly */
PHPAPI php_basic_globals   basic_globals;   /* BG()  -- ext/standard/basic_functions.h */

/* ext/standard/php_rand.h. Only shuffle()/str_shuffle()/array_rand() reach it and no trigger does,
 * but a stub returning a constant would make those silently degenerate. A tiny deterministic LCG
 * keeps them behaving like a shuffle, and determinism is what a differential harness wants. */
PHPAPI long php_rand(TSRMLS_D)
{
    static unsigned long seed = 123459876UL;
    seed = seed * 1103515245UL + 12345UL;
    return (long)((seed >> 16) & 0x7FFFFFFFUL);
}

/* ext/standard/scanf.c, reached only from sscanf()/fscanf(). Not linked: scanf.c pulls in the
 * file/stream layer. Returning FAILURE is honest -- the function is absent, not broken. */
PHPAPI int php_sscanf_internal(char *string, char *format, int argc, zval ***args,
                               int varStart, zval **return_value TSRMLS_DC)
{
    (void)string; (void)format; (void)argc; (void)args; (void)varStart; (void)return_value;
    return FAILURE;
}

/* main/main.c's error reporting, the 0-parameter form. php_error_docref1 already lives in
 * php_capstone_ext_stubs.c; this routes to the same place, zend_error, so a diagnostic raised by
 * ext/standard is not silently swallowed. */
PHPAPI void php_error_docref0(const char *docref TSRMLS_DC, int type, const char *format, ...)
{
    (void)docref;
    zend_error(type, "%s", format);
}
#endif

#if defined(PHP_CAP_EXT_PURE)
/* ---------------------------------------------------------------------------
 * main/output.c, main/main.c and main/snprintf.c, none of which this port links.
 *
 * Output goes where the domain's output goes -- through zend_write, the pointer the SAPI already
 * owns -- so an ext/standard function that prints is not silently muted. Output BUFFERING is a
 * different matter: it exists so print_r($x, true) can capture into a string, and faking it would
 * make that return an empty string that looks like a result. These report failure instead, which
 * the callers check.
 * ------------------------------------------------------------------------- */
PHPAPI int php_body_write(const char *str, uint str_length TSRMLS_DC)
{
    return zend_write(str, str_length);
}

PHPAPI int php_printf(const char *format, ...)
{
    char buf[512];
    va_list ap;
    int n;
    va_start(ap, format);
    n = vsnprintf(buf, sizeof buf, format, ap);
    va_end(ap);
    if (n < 0) { return 0; }
    if ((unsigned)n >= sizeof buf) { n = (int)sizeof buf - 1; }
    return zend_write(buf, (uint)n);
}

/* spprintf allocates; its callers efree the result, so this must emalloc, not use a static. */
PHPAPI int spprintf(char **pbuf, size_t max_len, const char *format, ...)
{
    va_list ap;
    int n;
    size_t cap = (max_len && max_len < 4096u) ? max_len : 4095u;
    *pbuf = (char *) emalloc(cap + 1);
    if (!*pbuf) { return 0; }
    va_start(ap, format);
    n = vsnprintf(*pbuf, cap + 1, format, ap);
    va_end(ap);
    return n < 0 ? 0 : n;
}

/* No output buffering in a domain. Returning FAILURE is what every caller tests for. */
PHPAPI int php_start_ob_buffer(zval *output_handler, uint chunk_size, zend_bool erase TSRMLS_DC)
{ (void)output_handler; (void)chunk_size; (void)erase; return FAILURE; }

PHPAPI void php_end_ob_buffer(zend_bool send_buffer, zend_bool just_flush TSRMLS_DC)
{ (void)send_buffer; (void)just_flush; }

PHPAPI int php_ob_get_buffer(zval *p TSRMLS_DC)
{ (void)p; return FAILURE; }

/* The __PHP_Incomplete_Class machinery, reached when unserialize() meets a class that is not
 * defined. Nothing stores a name, so nothing can look one up. */
char *php_lookup_class_name(zval *object, size_t *nlen)
{ (void)object; if (nlen) { *nlen = 0; } return (char *)0; }

void php_store_class_name(zval *object, const char *name, size_t len)
{ (void)object; (void)name; (void)len; }
#endif
