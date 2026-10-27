/* Minimal stub: reached through php.h / TSRM. A domain has no filesystem or clock, so
 * nothing declared here is ever called; the declarations exist so php.h parses. */
#ifndef _UTIME_H
#define _UTIME_H 1
#include <sys/types.h>
struct utimbuf { time_t actime; time_t modtime; };
int utime(const char *, const struct utimbuf *);
#endif
