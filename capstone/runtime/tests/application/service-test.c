#include "capstone/application-service.h"
#include "capstone/hostcall.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static unsigned mask = 7;
static char payload[4096];
static struct hostcall_v0 response;
static long request(unsigned op, unsigned fd, const char *text, unsigned size) {
  struct capstone_app_fd_request arg = {fd, 0};
  memcpy(payload, &arg, sizeof arg);
  if (text) memcpy(payload + sizeof arg, text, size);
  struct hostcall_v0 req = {.opcode = op, .offset = sizeof arg, .length = size};
  assert(!capstone_application_service(&mask, &req, &response, payload));
  return response.error ? response.error : response.result;
}

int main(void) {
  int input[2], output[2], errors[2];
  assert(!pipe(input) && !pipe(output) && !pipe(errors));
  pid_t child = fork();
  assert(child >= 0);
  if (!child) {
    signal(SIGPIPE, SIG_DFL);
    assert(dup2(input[0], 0) == 0 && dup2(output[1], 1) == 1 && dup2(errors[1], 2) == 2);
    close(input[0]); close(input[1]); close(output[0]); close(output[1]);
    close(errors[0]); close(errors[1]);
    assert(request(CAPSTONE_APP_READ, 0, NULL, 100) == 3);
    assert(!memcmp(payload + 16, "in\n", 3));
    assert(request(CAPSTONE_APP_READ, 0, NULL, 100) == 0);
    assert(request(CAPSTONE_APP_WRITE, 1, "out\n", 4) == 4);
    assert(request(CAPSTONE_APP_WRITE, 2, "err\n", 4) == 4);
    assert(request(CAPSTONE_APP_STAT, 1, NULL, 16) == 0);
    struct capstone_app_stat st;
    memcpy(&st, payload + 16, sizeof st);
    assert(S_ISFIFO(st.mode));
    assert(request(CAPSTONE_APP_CLOSE, 1, NULL, 0) == 0);
    assert(request(CAPSTONE_APP_WRITE, 1, "bad", 3) == -EBADF);
    assert(request(CAPSTONE_APP_READ, 100, NULL, 1) == -EBADF);
    struct hostcall_v0 bad = {.opcode = CAPSTONE_APP_READ, .offset = 4090, .length = 10};
    assert(!capstone_application_service(&mask, &bad, &response, payload));
    assert(response.error == -EINVAL);
    _exit(0);
  }
  close(input[0]); close(output[1]); close(errors[1]);
  assert(write(input[1], "in\n", 3) == 3); close(input[1]);
  char buffer[20];
  assert(read(output[0], buffer, sizeof buffer) == 4 && !memcmp(buffer, "out\n", 4));
  assert(read(output[0], buffer, sizeof buffer) == 0);
  assert(read(errors[0], buffer, sizeof buffer) == 4 && !memcmp(buffer, "err\n", 4));
  close(output[0]); close(errors[0]);
  int status;
  assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status));
  puts("application stream contract passed");
}
