#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int constructed;
__attribute__((constructor)) static void initialize(void) {
  const char *value = getenv("CAPSTONE_CONTRACT");
  constructed = value && !strcmp(value, "environment with spaces");
}

int main(int argc, char **argv) {
  if (!constructed || argc != 4 || strcmp(argv[2], "") ||
      strcmp(argv[3], "argument with spaces\nand newline"))
    return 41;
  if (write(1, "stdout\n", 7) != 7 || write(2, "stderr\n", 7) != 7)
    return 42;
  /* These two controls require the same application built with HEAP=sublet. */
  if (!strcmp(argv[1], "fault-stale")) {
    volatile unsigned char *stale = malloc(64);
    if (!stale) return 46;
    *stale = 42;
    free((void *)stale);
    return *stale;
  }
  if (!strcmp(argv[1], "fault-exhaust")) {
    for (;;) {
      volatile unsigned char *p = malloc(64);
      if (!p) return 46;
      *p = 42;
      free((void *)p);
    }
  }
  if (!strcmp(argv[1], "write-loop")) {
    char output[8192];
    memset(output, 'x', sizeof output);
    while (write(1, output, sizeof output) > 0) {}
    return 43;
  }
  if (!strcmp(argv[1], "loop")) {
    for (;;)
      __asm__ volatile ("" ::: "memory");
  }
  if (!strcmp(argv[1], "fault-mrev")) {
    __asm__ volatile (".insn r 0x5b, 1, 8, t0, x0, x0" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-privcsr")) {
    __asm__ volatile ("csrw mie, zero" ::: "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-debugcap")) {
    __asm__ volatile (".insn r 0x5b, 1, 64, t0, x0, x0" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-capenter")) {
    __asm__ volatile (".insn r 0x5b, 1, 13, x0, x0, x0" ::: "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-vector")) {
    __asm__ volatile (".insn i 0x5b, 7, x0, x0, 0\n"
                      "li sp, 0\nli gp, 0\nld t0, 0(sp)" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-stack")) {
    __asm__ volatile("li sp, 0\nli gp, 0\nld t0, 0(sp)" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault")) {
    volatile unsigned long *invalid = (void *)1;
    *invalid = 1;
    return 43;
  }
  if (!strcmp(argv[1], "exit139"))
    return 139;
  char input[16];
  ssize_t n = read(0, input, sizeof input);
  if (n != 6 || memcmp(input, "input\n", 6))
    return 44;
  char cwd[1024];
  if (!getcwd(cwd, sizeof cwd) || strcmp(cwd, "/tmp"))
    return 45;
  puts("application: ok");
  return 0;
}
