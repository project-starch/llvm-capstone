#ifndef A11_SIGNAL_H
#define A11_SIGNAL_H
typedef int sig_atomic_t;
typedef void (*sighandler_t)(int);
#define SIGINT 2
#endif
