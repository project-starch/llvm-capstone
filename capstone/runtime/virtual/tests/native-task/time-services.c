#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/select.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
static int failures,checks;
#define CHECK(x) do { ++checks; if (!(x)) { printf("FAIL line %d: %s errno=%d\n",__LINE__,#x,errno); ++failures; }} while(0)
static int64_t ns(struct timespec t) { return (int64_t)t.tv_sec*1000000000+t.tv_nsec; }
int main(void) {
 struct timespec before,after,delay={0,2000000},bad={0,1000000000},remain={123,456};
 CHECK(clock_gettime(CLOCK_MONOTONIC,&before)==0);
 CHECK(nanosleep(&delay,&remain)==0);
 CHECK(clock_gettime(CLOCK_MONOTONIC,&after)==0);
 CHECK(ns(after)-ns(before)>=ns(delay));
 CHECK(remain.tv_sec==123 && remain.tv_nsec==456);
 errno=0;CHECK(nanosleep(&bad,0)==-1 && errno==EINVAL);
 CHECK(clock_nanosleep(CLOCK_MONOTONIC,0,&delay,0)==0);
 CHECK(clock_nanosleep(CLOCK_REALTIME,0,&delay,0)==0);
 CHECK(clock_nanosleep(CLOCK_MONOTONIC,2,&delay,0)==EINVAL);
 CHECK(clock_gettime(CLOCK_MONOTONIC,&before)==0);
 before.tv_nsec+=2000000;if(before.tv_nsec>=1000000000){before.tv_nsec-=1000000000;++before.tv_sec;}
 CHECK(clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&before,&remain)==0);
 CHECK(clock_gettime(CLOCK_MONOTONIC,&after)==0);CHECK(ns(after)>=ns(before));
 CHECK(remain.tv_sec==123 && remain.tv_nsec==456);
 CHECK(clock_gettime(CLOCK_MONOTONIC,&before)==0);
 CHECK(pselect(0,0,0,0,&delay,0)==0);
 CHECK(clock_gettime(CLOCK_MONOTONIC,&after)==0);CHECK(ns(after)-ns(before)>=ns(delay));
 errno=0;CHECK(pselect(0,0,0,0,&bad,0)==-1 && errno==EINVAL);
 int p[2];CHECK(pipe(p)==0);
 fd_set readset,writeset;FD_ZERO(&readset);FD_SET(p[0],&readset);
 struct timespec zero={0};CHECK(pselect(p[0]+1,&readset,0,0,&zero,0)==0);CHECK(!FD_ISSET(p[0],&readset));
 CHECK(write(p[1],"x",1)==1);FD_SET(p[0],&readset);FD_ZERO(&writeset);FD_SET(p[1],&writeset);
 CHECK(pselect(p[1]+1,&readset,&writeset,0,&zero,0)==2);CHECK(FD_ISSET(p[0],&readset));CHECK(FD_ISSET(p[1],&writeset));
 CHECK(close(p[0])==0);CHECK(close(p[1])==0);
 FD_ZERO(&readset);FD_SET(p[0],&readset);errno=0;CHECK(pselect(p[0]+1,&readset,0,0,&zero,0)==-1 && errno==EBADF);
 sigset_t mask,prior,observed;sigemptyset(&mask);sigaddset(&mask,SIGUSR1);
 CHECK(sigprocmask(SIG_SETMASK,0,&prior)==0);
 CHECK(pselect(0,0,0,0,&delay,&mask)==0);
 CHECK(sigprocmask(SIG_SETMASK,0,&observed)==0);
 CHECK(sigismember(&observed,SIGUSR1)==sigismember(&prior,SIGUSR1));
 printf("TIME checks=%d failures=%d\n",checks,failures);return !!failures;
}
