#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int run(const char *loader, const char *domain, int expect_signal) {
  fflush(NULL);
  pid_t child = fork();
  if (child < 0)
    return 1;
  if (!child) {
    struct rlimit no_core = {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    execl(loader, loader, domain, (char *)NULL);
    _exit(127);
  }
  int status;
  pid_t result;
  do {
    result = waitpid(child, &status, 0);
  } while (result < 0 && errno == EINTR);
  if (result != child)
    return 1;
  if (expect_signal) {
    if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV)
      return 1;
    puts("RUNTIME_SUPERVISOR SIGSEGV");
  } else {
    if (!WIFEXITED(status) || WEXITSTATUS(status))
      return 1;
    puts("RUNTIME_SUPERVISOR HEALTHY");
  }
  return 0;
}

int main(int argc, char **argv) {
  /* launcher, healthy domain, one or more fault domains */
  if (argc < 4 || run(argv[1], argv[2], 0))
    return 1;
  for (int i = 3; i < argc; ++i)
    if (run(argv[1], argv[i], 1))
      return 1;
  if (run(argv[1], argv[2], 0))
    return 1;
  puts("__CAPSTONE_RUNTIME_ISOLATION_DONE__");
  return 0;
}
