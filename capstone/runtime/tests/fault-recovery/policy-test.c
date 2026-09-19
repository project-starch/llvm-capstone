#define _POSIX_C_SOURCE 200809L
#include "capstone/linux-domain-fault.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static void cleanup(void *context) {
  int fd = *(int *)context;
  if (write(fd, "C", 1) != 1)
    _exit(2);
}

static int check(int fault, int ignore, int block, int with_cleanup,
                 int output) {
  int pipefd[2];
  if (pipe(pipefd))
    return 1;
  pid_t child = fork();
  if (child < 0)
    return 1;
  if (!child) {
    close(pipefd[0]);
    struct rlimit no_core = {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    signal(SIGSEGV, ignore ? SIG_IGN : SIG_DFL);
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGSEGV);
    if (pthread_sigmask(block ? SIG_BLOCK : SIG_UNBLOCK, &signals, NULL))
      _exit(3);
    if (output) {
      int stream[2];
      if (pipe(stream))
        _exit(4);
      signal(SIGPIPE, SIG_DFL);
      signal(SIGALRM, SIG_DFL);
      sigemptyset(&signals);
      sigaddset(&signals, SIGPIPE);
      sigaddset(&signals, SIGALRM);
      pthread_sigmask(SIG_UNBLOCK, &signals, NULL);
      if (output == 1) {
        close(stream[0]); /* Any write would deliver SIGPIPE, not SIGSEGV. */
      } else {
        int flags = fcntl(stream[1], F_GETFL);
        if (flags < 0 || fcntl(stream[1], F_SETFL, flags | O_NONBLOCK))
          _exit(5);
        char fill[4096] = {0};
        while (write(stream[1], fill, sizeof fill) > 0)
          ;
        if (errno != EAGAIN || fcntl(stream[1], F_SETFL, flags))
          _exit(6);
        /* Keep the read end open, but never drain it: a blocking write hangs.
         */
      }
      if (dup2(stream[1], STDOUT_FILENO) < 0)
        _exit(7);
      close(stream[1]);
      alarm(2); /* A broken policy must fail promptly, not hang the test. */
    }
    capstone_domain_exit_on_fault(fault ? CAPSTONE_DOMAIN_FAULT_RETVAL : 42,
                                  with_cleanup ? cleanup : NULL, &pipefd[1]);
    _exit(0);
  }
  close(pipefd[1]);
  int status;
  pid_t result;
  do {
    result = waitpid(child, &status, 0);
  } while (result < 0 && errno == EINTR);
  char bytes[2];
  ssize_t count = read(pipefd[0], bytes, sizeof bytes);
  close(pipefd[0]);
  if (result != child || count != (fault && with_cleanup ? 1 : 0))
    return 1;
  if (count == 1 && bytes[0] != 'C')
    return 1;
  return fault ? !(WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV)
               : !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

int main(void) {
  for (int fault = 0; fault < 2; ++fault)
    for (int ignore = 0; ignore < 2; ++ignore)
      for (int block = 0; block < 2; ++block)
        for (int cleanup_enabled = 0; cleanup_enabled < 2; ++cleanup_enabled)
          if (check(fault, ignore, block, cleanup_enabled, 0))
            return 1;
  int failed = 0;
  for (int output = 1; output <= 2; ++output)
    failed |= check(1, 1, 1, 0, output);
  return failed;
}
