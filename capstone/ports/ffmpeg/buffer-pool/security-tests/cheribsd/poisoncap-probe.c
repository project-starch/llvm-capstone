/* Paired live/poison/sweep controls; one requested case per guest process. */
#include <cheri/cheric.h>
#include <cheri/revoke.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void *volatile old_alias;

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  if (!feature_present("cheri_caprevoke_poison")) {
    fputs("POISONCAP missing kernel feature\n", stderr);
    return 2;
  }
  unsigned char *root = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON, -1, 0);
  if (root == MAP_FAILED)
    return 2;
  unsigned char *child = cheri_setboundsexact(root, 64);
  child = cheri_clearperm(child, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
  unsigned char *sibling = cheri_setboundsexact(root + 128, 64);
  sibling = cheri_clearperm(sibling, CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
  volatile unsigned char *live = child;
  live[0] = 17;
  sibling[0] = 41;
  if (live[0] != 17 || sibling[0] != 41)
    return 1;
  old_alias = child;
  if (!strcmp(argv[1], "live")) {
    puts("POISONCAP live PASS");
    return 0;
  }
  if (strcmp(argv[1], "read") && strcmp(argv[1], "write") &&
      strcmp(argv[1], "reuse") && strcmp(argv[1], "reused-read"))
    return 2;
  unsigned char *poison = cheri_setboundsexact(root, 64);
  for (unsigned i = 0; i < 64; i += 16) {
    void *word = poison + i;
    __asm__ volatile("cpoison %0, 0(%0)" : : "C"(word) : "memory");
  }
  if (sibling[0] != 41)
    return 1;
  if (!strcmp(argv[1], "reuse") || !strcmp(argv[1], "reused-read")) {
    struct cheri_revoke_syscall_info info = {0};
    puts("POISONCAP sweep BEGIN");
    fflush(stdout);
    if (cheri_revoke(CHERI_REVOKE_LAST_PASS | CHERI_REVOKE_IGNORE_START |
                     CHERI_REVOKE_TAKE_STATS, 0, &info) != 0) {
      perror("POISONCAP cheri_revoke");
      return 1;
    }
    printf("POISONCAP sweep END old_tag=%u root_tag=%u\n",
           (unsigned)cheri_gettag(old_alias), (unsigned)cheri_gettag(root));
    fflush(stdout);
    if (cheri_gettag(old_alias) || !cheri_gettag(root) || sibling[0] != 41)
      return 1;
    for (unsigned i = 0; i < 64; i += 16) {
      void *word = root + i;
      __asm__ volatile("cclearpoison %0, 0(%0)" : : "C"(word) : "memory");
    }
    live = cheri_clearperm(cheri_setboundsexact(root, 64),
                           CHERI_PERM_POISON | CHERI_PERM_SW_VMEM);
    live[0] = 59;
    if (live[0] != 59 || sibling[0] != 41)
      return 1;
    if (!strcmp(argv[1], "reuse")) {
      puts("POISONCAP reuse PASS");
      return 0;
    }
  }
  printf("POISONCAP %s READY\n", argv[1]);
  fflush(stdout);
  volatile unsigned char *stale = old_alias;
  if (!strcmp(argv[1], "write"))
    *stale = 93;
  else
    (void)*stale;
  puts("POISONCAP unexpected access");
  return 1;
}
