/* Freestanding stand-in for the GLib surface that wmem's core and its four
 * allocators use. Anything wider than this is deliberately absent. */
#ifndef WM_SHIM_GLIB_H
#define WM_SHIM_GLIB_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "port.h"
#define _U_ __attribute__((unused))
#define G_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define G_GNUC_MALLOC
#define G_GNUC_ALLOC_SIZE(n)
#define G_MAXSSIZE ((size_t)(SIZE_MAX / 2))
#define g_malloc(n) wm_sys_alloc(n)
#define g_free(p) wm_sys_free(p)
#define g_realloc(p, n) wm_sys_realloc((p), (n))
#define g_assert_true(x) ((x) ? (void)0 : wm_fail(101))
#define g_assert_not_reached() wm_fail(102)
#define g_warning(...) ((void)0)
/* Each pool's allocator is chosen explicitly by the port; there is no
 * environment override, and the domain has no environment at all. */
#define getenv(name) wm_getenv(name)
#endif
