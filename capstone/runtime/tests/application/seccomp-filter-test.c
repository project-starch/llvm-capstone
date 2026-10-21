/* The launcher's seccomp filter: the allowlist it collects, the search program
 * it builds (run here on every number below 1200, and on random lists of 1 to
 * 510 numbers), and, on RISC-V, the filter installed in a child. */
#include "../../linux/delegate-service.h"
#include <assert.h>
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#define NUMBERS 1200

/* Classic BPF on (arch, nr): 1 allow, 0 deny, -1 for an instruction or a
 * jump target the program must not contain. */
static int run(const struct sock_filter *p, unsigned len, uint32_t arch, uint32_t nr) {
  uint32_t a = 0;
  for (unsigned pc = 0; pc < len; ++pc) {
    const struct sock_filter *f = &p[pc];
    unsigned off;
    switch (f->code) {
    case BPF_LD | BPF_W | BPF_ABS:
      if (f->k == offsetof(struct seccomp_data, arch)) a = arch;
      else if (f->k == offsetof(struct seccomp_data, nr)) a = nr;
      else return -1;
      break;
    case BPF_RET | BPF_K:
      if (f->k == SECCOMP_RET_ALLOW) return 1;
      return f->k == (SECCOMP_RET_ERRNO | ENOSYS) ? 0 : -1;
    case BPF_JMP | BPF_JA:
      if (f->k >= len - pc - 1) return -1;
      pc += f->k;
      break;
    case BPF_JMP | BPF_JEQ | BPF_K:
    case BPF_JMP | BPF_JGE | BPF_K:
      off = (BPF_OP(f->code) == BPF_JEQ ? a == f->k : a >= f->k) ? f->jt : f->jf;
      if (off >= len - pc - 1) return -1;
      pc += off;
      break;
    default:
      return -1;
    }
  }
  return -1;
}

/* The program for allow[0..n) allows exactly those numbers, for RISC-V only. */
static void check(const uint16_t *allow, unsigned n) {
  struct sock_filter program[CAPSTONE_SECCOMP_PROGRAM_MAX(512)];
  unsigned len = capstone_delegate_seccomp_program(allow, n, program), i = 0;
  assert(len <= CAPSTONE_SECCOMP_PROGRAM_MAX(n));
  for (uint32_t nr = 0; nr < NUMBERS; ++nr) {
    int listed = i < n && allow[i] == nr;
    i += listed;
    assert(run(program, len, AUDIT_ARCH_RISCV64, nr) == listed);
    assert(run(program, len, AUDIT_ARCH_X86_64, nr) == 0);
  }
}

int main(void) {
  uint16_t allow[512];
  unsigned n = capstone_delegate_seccomp_allowlist(allow, 512);
  assert(n > 0);
  for (unsigned i = 1; i < n; ++i)
    assert(allow[i - 1] < allow[i]);
  const struct capstone_delegate_shape *s;
  for (size_t i = 0; (s = capstone_delegate_shape_at(i)); ++i)
    if (s->group == CAPSTONE_GROUP_DELEGATED && s->nr < 1100) {
      unsigned j = 0;
      while (j < n && allow[j] != s->nr) ++j;
      assert(j < n);
    }
  check(allow, n);

  srand(1);
  for (unsigned size = 1; size <= 510; ++size)
    for (int rep = 0; rep < 3; ++rep) {
      unsigned char used[NUMBERS] = {0};
      for (unsigned m = 0; m < size;) {
        unsigned v = (unsigned)rand() % 1100;
        if (!used[v]) { used[v] = 1; ++m; }
      }
      uint16_t list[512];
      unsigned m = 0;
      for (unsigned v = 0; v < NUMBERS; ++v)
        if (used[v]) list[m++] = (uint16_t)v;
      check(list, m);
    }

#if defined(__riscv) && __riscv_xlen == 64
  /* Installed: a listed call runs, and an unlisted call the kernel does
   * implement, so that only the filter can refuse it, answers ENOSYS. The
   * candidates read and change nothing: sched_getscheduler(0), and
   * personality(0xffffffff), its query form. */
  static const long harmless[][2] = {{SYS_sched_getscheduler, 0}, {SYS_personality, 0xffffffff}};
  long unlisted = -1, arg = 0;
  for (unsigned i = 0; i < sizeof harmless / sizeof harmless[0] && unlisted < 0; ++i) {
    unsigned j = 0;
    while (j < n && allow[j] != harmless[i][0]) ++j;
    if (j == n) { unlisted = harmless[i][0]; arg = harmless[i][1]; }
  }
  assert(unlisted >= 0);
  assert(syscall(unlisted, arg) >= 0);
  pid_t child = fork();
  assert(child >= 0);
  if (!child) {
    if (capstone_delegate_seccomp()) _exit(2);
    if (syscall(SYS_getpid) != getpid()) _exit(3);
    errno = 0;
    _exit(syscall(unlisted, arg) == -1 && errno == ENOSYS ? 0 : 4);
  }
  int status;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  printf("seccomp filter: %u numbers; installed, unlisted %ld refused\n", n, unlisted);
#else
  printf("seccomp filter: %u numbers; not installed on this architecture\n", n);
#endif
  return 0;
}
