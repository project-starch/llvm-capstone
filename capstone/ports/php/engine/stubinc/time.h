/* Minimal stub: reached through php.h / TSRM. A domain has no filesystem or clock, so
 * nothing declared here is ever called; the declarations exist so php.h parses. */
#ifndef _TIME_H
#define _TIME_H 1
#include <sys/types.h>
struct tm {
    int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
    long tm_gmtoff; const char *tm_zone;
};
struct timespec { time_t tv_sec; long tv_nsec; };
#define CLOCKS_PER_SEC 1000000L
typedef long clock_t;
time_t     time(time_t *);
clock_t    clock(void);
double     difftime(time_t, time_t);
time_t     mktime(struct tm *);
struct tm *localtime(const time_t *);
struct tm *gmtime(const time_t *);
char      *asctime(const struct tm *);
char      *ctime(const time_t *);
size_t     strftime(char *, size_t, const char *, const struct tm *);
#endif
