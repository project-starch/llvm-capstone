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
