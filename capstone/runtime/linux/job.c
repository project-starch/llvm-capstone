/* SSH has no portable numeric representation for a signal-killed command.
 * This short-lived Linux parent records waitpid status independently of stdio.
 * It is also usable by other remote harnesses; there is no resident job service.
 */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile sig_atomic_t child_pid;

static void forward(int signal_number) {
  if (child_pid > 0) kill(-child_pid, signal_number);
}

int main(int argc, char **argv) {
  int command = 3, change_user = 0;
  uid_t uid = 0;
  gid_t gid = 0;
  if (argc >= 6 && !strcmp(argv[2], "--user")) {
    char *end;
    if (argv[3][0] < '0' || argv[3][0] > '9') return 125;
    errno = 0;
    unsigned long u = strtoul(argv[3], &end, 10);
    if (errno || end == argv[3] || *end != ':' || u >= UINT32_MAX) return 125;
    const char *group = end + 1;
    if (*group < '0' || *group > '9') return 125;
    unsigned long g = strtoul(group, &end, 10);
    if (errno || end == group || *end || g >= UINT32_MAX) return 125;
    uid = (uid_t)u;
    gid = (gid_t)g;
    change_user = 1;
    command = 5;
  }
  if (argc <= command || strcmp(argv[command - 1], "--")) {
    fputs("usage: capstone-job RESULT.json [--user UID:GID] -- COMMAND [ARG...]\n", stderr);
    return 125;
  }
  sigset_t blocked, previous;
  sigemptyset(&blocked);
  sigaddset(&blocked, SIGINT);
  sigaddset(&blocked, SIGTERM);
  sigaddset(&blocked, SIGHUP);
  if (sigprocmask(SIG_BLOCK, &blocked, &previous)) return 125;
  struct sigaction action = {0};
  action.sa_handler = forward;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL) ||
      sigaction(SIGHUP, &action, NULL)) return 125;
  pid_t child = fork();
  if (child < 0) { perror("capstone-job: fork"); return 125; }
  if (!child) {
    if (setpgid(0, 0)) _exit(125);
    action.sa_handler = SIG_DFL;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGHUP, &action, NULL);
    sigprocmask(SIG_SETMASK, &previous, NULL);
    if (change_user && ((geteuid() == 0 && setgroups(0, NULL)) || setgid(gid) || setuid(uid))) {
      perror("capstone-job: user");
      _exit(125);
    }
    execvp(argv[command], argv + command);
    _exit(errno == ENOENT ? 127 : 126);
  }
  child_pid = child;
  /* The child also sets its group, so signalling cannot race with exec. */
  if (setpgid(child, child) && errno != EACCES && errno != ESRCH) {
    kill(child, SIGKILL);
    return 125;
  }
  sigprocmask(SIG_SETMASK, &previous, NULL);
  int status;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) { perror("capstone-job: waitpid"); return 125; }
  }
  sigprocmask(SIG_BLOCK, &blocked, NULL);
  child_pid = 0;
  const char *kind = WIFSIGNALED(status) ? "signal" : "exit";
  int value = WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status);
  size_t bytes = strlen(argv[1]) + sizeof ".tmp";
  char *temporary = malloc(bytes);
  if (!temporary) return 125;
  snprintf(temporary, bytes, "%s.tmp", argv[1]);
  int fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) { perror("capstone-job: result"); free(temporary); return 125; }
  char record[96];
  int length = snprintf(record, sizeof record,
                        "{\"version\":1,\"kind\":\"%s\",\"value\":%d}\n", kind, value);
  int error = write(fd, record, (size_t)length) != length;
  error |= close(fd) != 0;
  if (!error) error = rename(temporary, argv[1]) != 0;
  if (error) { perror("capstone-job: result"); unlink(temporary); }
  free(temporary);
  return error ? 125 : WIFSIGNALED(status) ? 128 + value : value;
}
