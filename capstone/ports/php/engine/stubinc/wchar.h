/* Wide characters are not supported in the domain, and nothing on the trigger path needs them.
 *
 * ext/standard/basic_functions.h includes <wchar.h> under HAVE_WCHAR_H and uses mbstate_t only
 * under _REENTRANT && HAVE_MBRLEN && HAVE_MBSTATE_T, none of which this build defines. So the
 * header only has to exist. mbstate_t is declared anyway, so that a future TU referencing it fails
 * at the USE with a clear message rather than here with a missing include. */
#ifndef PHP_CAPSTONE_WCHAR_H
#define PHP_CAPSTONE_WCHAR_H

typedef struct { int __count; unsigned int __value; } mbstate_t;

#endif
