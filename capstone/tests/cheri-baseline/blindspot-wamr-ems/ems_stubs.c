/* The seven platform symbols EMS needs, so the allocator can be measured without
 * the whole runtime around it.
 *
 * Deliberately minimal, and none of them is on the path under test: the mutexes
 * are no-ops because the reproducer is single-threaded, and logging is discarded.
 * EMS's own sources are compiled UNMODIFIED apart from the one knob that selects
 * the vulnerable or the fixed arm -- b_memcpy_s is the only one with real
 * behaviour, and it is a bounded copy exactly as its name says.
 */
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

int
b_memcpy_s(void *dst, unsigned dlen, const void *src, unsigned slen)
{
    if (slen == 0)
        return 0;
    if (!dst || !src || dlen < slen)
        return -1;
    memcpy(dst, src, slen);
    return 0;
}

void
bh_log(unsigned log_level, const char *file, int line, const char *fmt, ...)
{
    (void)log_level; (void)file; (void)line; (void)fmt;
}

int os_printf(const char *fmt, ...) { (void)fmt; return 0; }
int os_vprintf(const char *fmt, va_list ap) { (void)fmt; (void)ap; return 0; }

/* Single-threaded by construction; a real mutex would only add a dependency. */
int  os_mutex_init(void *m)    { (void)m; return 0; }
int  os_mutex_destroy(void *m) { (void)m; return 0; }
int  os_mutex_lock(void *m)    { (void)m; return 0; }
int  os_mutex_unlock(void *m)  { (void)m; return 0; }
