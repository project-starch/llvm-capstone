#include <sys/mman.h>
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
static void *volatile roots[4];
int main(void) {
  size_t page = (size_t)getpagesize();
  char *region = mmap(NULL, page * 4, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (region == MAP_FAILED) return 2;
  void *manager = cheri_setboundsexact(region, 16);
  void *hidden = cheri_setboundsexact(region + page, 16);
  roots[0] = cheri_clearperm(manager, CHERI_PERM_SW_VMEM | CHERI_PERM_POISON);
  roots[1] = cheri_clearperm(hidden, CHERI_PERM_SW_VMEM | CHERI_PERM_POISON);
  roots[2] = cheri_clearperm(cheri_setboundsexact(region + 2*page, 16), CHERI_PERM_SW_VMEM | CHERI_PERM_POISON);
  roots[3] = cheri_clearperm(cheri_setboundsexact(region + 3*page, 16), CHERI_PERM_SW_VMEM | CHERI_PERM_POISON);
  __asm__ volatile("cpoison %0, 0(%0)" : : "C"(manager) : "memory");
  __asm__ volatile("cpoison %0, 0(%0)" : : "C"(hidden) : "memory");
  if (mprotect(region + page, page, PROT_NONE)) return 3;
  if (munmap(region + 3*page, page)) return 4;
  struct cheri_revoke_syscall_info info = {0};
  if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START, 0, &info)) return 5;
  fprintf(stderr, "tags=%u,%u,%u perms=%lx,%lx,%lx\n", cheri_gettag(roots[0]), cheri_gettag(roots[1]), cheri_gettag(roots[2]), cheri_getperm(roots[0]), cheri_getperm(roots[1]), cheri_getperm(roots[2]));
  if ((cheri_gettag(roots[0]) && (cheri_getperm(roots[0]) & CHERI_PERM_LOAD)) ||
      (cheri_gettag(roots[1]) && (cheri_getperm(roots[1]) & CHERI_PERM_LOAD))) return 6;
  if (!cheri_gettag(roots[2]) || !(cheri_getperm(roots[2]) & CHERI_PERM_LOAD)) return 7;
  puts("PROBE-OK resident hidden zero-fill unmapped");
  return 0;
}
