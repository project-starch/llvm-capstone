#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <unistd.h>

_Static_assert(offsetof(struct __jmp_buf_tag, __fl) == 224, "mask flag offset");
_Static_assert(offsetof(struct __jmp_buf_tag, __ss) == 232, "mask offset");
extern int jump_register(jmp_buf, void *, int);
static sigjmp_buf env;
static int value = 42;

__attribute__((noinline)) static void mask_jump(void)
{
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    if (sigprocmask(SIG_UNBLOCK, &mask, 0)) _exit(10);
    siglongjmp(env, 7);
}

int main(void)
{
    if (jump_register(env, &value, 0)) return 1;
    if (jump_register(env, &value, 1)) return 2;
    sigset_t mask, old, current;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &mask, &old)) return 3;
    int result = sigsetjmp(env, 1);
    if (!result) mask_jump();
    if (result != 7 || sigprocmask(SIG_SETMASK, 0, &current) ||
        sigismember(&current, SIGUSR1) != 1) return 4;
    result = sigsetjmp(env, 0);
    if (!result) mask_jump();
    if (result != 7 || sigprocmask(SIG_SETMASK, 0, &current) ||
        sigismember(&current, SIGUSR1) != 0) return 5;
    if (sigprocmask(SIG_SETMASK, &old, 0)) return 6;
    return write(1, "setjmp ok\n", 10) != 10;
}
