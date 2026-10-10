/* What a jump through a pointer is checked against on the virtual profile.
 *
 *   call-rights perm | control | noexec | heap
 *
 * perm     print the rights of a function pointer and of a malloc'd buffer
 *          (X=1, W=2, R=4, MANAGE=8)
 * control  call a function through its pointer; must return 42
 * noexec   the same pointer with only READ kept (CSTIGHTEN): same address,
 *          executable pages, no X right in the capability
 * heap     call a malloc'd buffer holding one RISC-V `ret`
 *
 * The virtual compiler profile calls by address within PCC
 * (-capstone-gp-free) and returns by a scalar address, so the rights of the
 * capability a call goes through are not what authorizes the fetch: PCC and
 * the page permissions are. Measured 2026-10-11 (README, "Indirect calls"):
 * noexec returns 42; heap stops with cause 12, an instruction page fault at
 * the buffer. A profile that jumps through capabilities would fault both at
 * the jump instead; this program is the check for that change. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) static int real(void) { return 42; }

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  setvbuf(stdout, NULL, _IONBF, 0);
  int (*volatile code)(void) = real;
  uint32_t *buf = malloc(64);
  if (!buf)
    return 3;
  if (!strcmp(argv[1], "perm")) {
    printf("CALL-RIGHTS perm code=%lu buf=%lu\n",
           (unsigned long)__builtin_capstone_cap_get_perm((void *)code),
           (unsigned long)__builtin_capstone_cap_get_perm(buf));
    return 0;
  }
  if (!strcmp(argv[1], "control")) {
    printf("CALL-RIGHTS control returned %d\n", code());
    return 0;
  }
  if (!strcmp(argv[1], "noexec")) {
    void *ro;
    /* CSTIGHTEN keeps the rights named by rs2's register number: x4 is READ. */
    __asm__ volatile(".insn r 0x5b, 0x1, 0x02, %0, %1, x4\n"
                     : "=r"(ro) : "r"((void *)code) : "memory");
    printf("CALL-RIGHTS noexec perm=%lu calling\n",
           (unsigned long)__builtin_capstone_cap_get_perm(ro));
    int (*volatile f)(void) = (int (*)(void))ro;
    printf("CALL-RIGHTS noexec RETURNED %d\n", f());
    return 1;
  }
  if (!strcmp(argv[1], "heap")) {
    memset(buf, 0, 64);
    buf[0] = 0x00008067u; /* jalr x0, 0(ra) */
    int (*volatile f)(void) = (int (*)(void))(void *)buf;
    printf("CALL-RIGHTS heap calling\n");
    printf("CALL-RIGHTS heap RETURNED %d\n", f());
    return 1;
  }
  return 2;
}
