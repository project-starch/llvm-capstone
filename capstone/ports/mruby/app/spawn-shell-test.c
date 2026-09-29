#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int capstone_mruby_spawn(const char *, int, int, int, int *);

int main(void) {
  int input[2], output[2], pid, status;
  assert(capstone_mruby_spawn(" \t\n", -1, -1, -1, &pid) == -1 && errno == ENOENT);
  assert(!pipe(input) && !pipe(output));
  int extra = open("/dev/null", O_RDONLY);
  assert(extra >= 0 && dup2(extra, 42) == 42);
  close(extra);
  assert(write(input[1], "input\n", 6) == 6);
  close(input[1]);
  assert(!capstone_mruby_spawn("test ! -e /proc/self/fd/42 || exit 9; cat; printf err >&2",
                              input[0], output[1], output[1], &pid));
  close(input[0]);
  close(output[1]);
  close(42);
  char buffer[32] = {0};
  size_t used = 0;
  ssize_t count;
  while ((count = read(output[0], buffer + used, sizeof buffer - used)) > 0) used += count;
  assert(count == 0 && used == 9 && !memcmp(buffer, "input\nerr", 9));
  close(output[0]);
  assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  return 0;
}
