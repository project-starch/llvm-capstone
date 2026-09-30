/* Only what Zend/zend_execute_API.c:128,1220-1242 references for the max_execution_time
 * watchdog and the debug SIGSEGV handler. Both are compiled out for the domain; these
 * exist so the TUs parse. */
#ifndef _SIGNAL_H
#define _SIGNAL_H 1
typedef unsigned long sigset_t;
typedef void (*__sighandler_t)(int);
#define SIG_DFL ((__sighandler_t)0)
#define SIG_IGN ((__sighandler_t)1)
#define SIG_ERR ((__sighandler_t)-1)
#define SIGSEGV 11
#define SIGFPE  8
#define SIGPROF 27
#define SIGALRM 14
#define SIGCHLD 17
#define SIG_BLOCK   0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2
__sighandler_t signal(int, __sighandler_t);
__sighandler_t sigset(int, __sighandler_t);
int sigemptyset(sigset_t *);
int sigaddset(sigset_t *, int);
int sigprocmask(int, const sigset_t *, sigset_t *);
#endif
