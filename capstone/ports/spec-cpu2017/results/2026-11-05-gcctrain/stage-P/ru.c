/* ru <cmd...>: run cmd, then print its peak RSS and fault counts to stderr as one RUSAGE line */
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char **argv) {
    pid_t p = fork();
    if (p == 0) { execv(argv[1], argv + 1); _exit(127); }
    int st; struct rusage ru; wait4(p, &st, 0, &ru);
    FILE *f = fopen("/tmp/ru.log", "a"); if (!f) f = stderr;
    fprintf(f, "RUSAGE %s maxrss_kb=%ld minflt=%ld majflt=%ld utime=%ld.%06ld stime=%ld.%06ld\n", argv[1], ru.ru_maxrss, ru.ru_minflt, ru.ru_majflt,
            (long)ru.ru_utime.tv_sec, (long)ru.ru_utime.tv_usec, (long)ru.ru_stime.tv_sec, (long)ru.ru_stime.tv_usec); if (f != stderr) fclose(f);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
}
