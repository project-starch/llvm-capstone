#ifndef _SYS_TIME_H
#define _SYS_TIME_H 1
#include <sys/types.h>
struct timeval  { long tv_sec; long tv_usec; };
struct timezone { int tz_minuteswest; int tz_dsttime; };
struct itimerval { struct timeval it_interval; struct timeval it_value; };
#define ITIMER_REAL    0
#define ITIMER_VIRTUAL 1
#define ITIMER_PROF    2
int setitimer(int, const struct itimerval *, struct itimerval *);
int getitimer(int, struct itimerval *);
int gettimeofday(struct timeval *, struct timezone *);
#endif
