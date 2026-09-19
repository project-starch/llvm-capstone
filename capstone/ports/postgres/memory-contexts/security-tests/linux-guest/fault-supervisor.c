/* An independent guest process verifies real signal death, not shell status. */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int run(const char *loader, const char *domain, const char *input,
               int expect_signal) {
  fflush(NULL);
  pid_t child = fork();
  if (child < 0)
    return 1;
  if (!child) {
    struct rlimit no_core = {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    execl(loader, loader, domain, input, "--linear-arena", (char *)NULL);
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
    puts("FAULT_ISOLATION CHILD SIGSEGV");
  } else {
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
      return 1;
    puts("FAULT_ISOLATION HEALTHY");
  }
  return 0;
}

int main(int argc, char **argv) {
  /* loader, healthy domain, empty input, fault domain, one or more inputs */
  if (argc < 6)
    return 2;
  if (run(argv[1], argv[2], argv[3], 0))
    return 1;
  for (int i = 5; i < argc; ++i)
    if (run(argv[1], argv[4], argv[i], 1))
      return 1;
  if (run(argv[1], argv[2], argv[3], 0))
    return 1;
  puts("__CAPSTONE_FAULT_ISOLATION_DONE__");
  return 0;
}
