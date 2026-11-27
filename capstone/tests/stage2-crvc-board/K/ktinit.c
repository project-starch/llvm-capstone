/* ktinit: /init of K's twin test boots. Runs each test program as a child, prints its wait status, then powers off.
 * Every line it prints starts with KTEST. */
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/reboot.h>
#include <sched.h>
static void run(const char *path) {
    printf("KTEST begin %s\n", path); fflush(stdout);
    pid_t pid = fork();
    if (!pid) { execl(path, path, (char *)0); perror("KTEST exec"); _exit(127); }
    /* Poll rather than block: init stays runnable, so each wakeup of the child is a switch between two mms and
     * switch_mm runs for it (against only the idle thread, which keeps the child's mm, it would return early). */
    int st = 0; while (waitpid(pid, &st, WNOHANG) == 0) sched_yield();
    if (WIFEXITED(st)) printf("KTEST end %s exit=%d\n", path, WEXITSTATUS(st));
    else if (WIFSIGNALED(st)) printf("KTEST end %s signal=%d\n", path, WTERMSIG(st));
    else printf("KTEST end %s status=0x%x\n", path, st);
    fflush(stdout); sleep(1);
}
int main(void) {
    printf("KTEST up\n");
    run("/crvcleak");
    run("/crvcmis.dom");
    printf("KTEST done\n"); fflush(stdout); sync();
    reboot(RB_POWER_OFF);
    return 0;
}
