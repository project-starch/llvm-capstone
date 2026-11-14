#include <signal.h>
#include <unistd.h>

static void handler(int sig) { (void)sig; }
static char alternate[SIGSTKSZ] __attribute__((aligned(16)));

int main(void)
{
    struct sigaction sa = {0}, old, current;
    sa.sa_handler = handler;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGUSR1);
    if (sigaction(SIGINT, &sa, &old) || sigaction(SIGINT, 0, &current)) return 1;
    if (current.sa_handler != handler || !(current.sa_flags & SA_RESTART) ||
        sigismember(&current.sa_mask, SIGUSR1) != 1) return 2;
    if (signal(SIGINT, SIG_IGN) != handler || sigaction(SIGINT, &old, 0)) return 3;
    sigset_t mask, before, after;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    if (sigprocmask(SIG_BLOCK, &mask, &before) ||
        sigprocmask(SIG_SETMASK, 0, &after)) return 4;
    if (sigismember(&after, SIGINT) != 1 ||
        sigprocmask(SIG_SETMASK, &before, 0)) return 5;
    stack_t ss = { .ss_sp = alternate, .ss_size = sizeof alternate }, previous, query;
    if (sigaltstack(&ss, &previous) || sigaltstack(0, &query)) return 6;
    if (query.ss_sp != alternate || query.ss_size != sizeof alternate ||
        query.ss_flags & SS_DISABLE) return 7;
    ((char *)query.ss_sp)[0] = 42;
    if (sigaltstack(&previous, 0)) return 8;
    return write(1, "signal ok\n", 10) != 10;
}
