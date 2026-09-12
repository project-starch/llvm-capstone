/* The printf helpers, on the host.
 *
 * PostgreSQL routes its own formatting through these, for the sake of
 * platforms whose printf differs. On a host they are the platform's, by
 * definition. A domain has no stdout, so it has a file of its own.
 */
#include "postgres.h"
#include <stdarg.h>
#include <stdio.h>

/* port.h redirects the whole family to PostgreSQL's own, this file included.
   Each is undefined above every call, or a stub calls itself through the macro
   and the linker asks for the rest of the family. */
#undef printf
#undef vprintf
#undef snprintf
#undef fprintf
#undef vsnprintf
#undef vfprintf

int pg_vsnprintf(char *str, size_t count, const char *fmt, va_list ap)
{
    return vsnprintf(str, count, fmt, ap);
}
int pg_vfprintf(FILE *stream, const char *fmt, va_list ap)
{
    return vfprintf(stream, fmt, ap);
}
int pg_vprintf(const char *fmt, va_list ap)
{
    return vprintf(fmt, ap);
}
int pg_snprintf(char *str, size_t count, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vsnprintf(str, count, fmt, ap); va_end(ap); return r;
}
int pg_fprintf(FILE *stream, const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vfprintf(stream, fmt, ap); va_end(ap); return r;
}
/* Not reached from the manager, but port.h redirects every caller that
   includes it, and a program built on this port is such a caller. */
int pg_printf(const char *fmt,...)
{
    va_list ap; va_start(ap, fmt); int r = vprintf(fmt, ap); va_end(ap); return r;
}
