#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include "capstone.h"

#define CHECK(x) do { if (!(x)) { perror(#x); return 1; } } while (0)

static int owner(void) {
  int fd = open(CAPSTONE_DEV_PATH, O_RDWR | O_CLOEXEC);
  if (fd >= 0 && ioctl(fd, IOCTL_PROCESS_ENABLE)) { close(fd); return -1; }
  return fd;
}

static unsigned long live(int fd) {
  struct ioctl_process_stats s;
  if (ioctl(fd, IOCTL_PROCESS_STATS, &s)) return ~0UL;
  return s.live_regions;
}

int main(void) {
  int a = owner(), b = owner(), legacy = open(CAPSTONE_DEV_PATH, O_RDWR);
  CHECK(a >= 0 && b >= 0 && legacy >= 0);
  unsigned long baseline = live(b);
  struct ioctl_region_create_args ar = {.len = 4096}, br = {.len = 4096};
  CHECK(!ioctl(a, IOCTL_REGION_CREATE, &ar));
  CHECK(!ioctl(b, IOCTL_REGION_CREATE, &br));
  CHECK(live(b) == baseline + 2);
  struct ioctl_region_query_args q = {.region_id = ar.region_id};
  CHECK(!ioctl(b, IOCTL_REGION_QUERY, &q) && !q.len);
  CHECK(ioctl(legacy, IOCTL_REGION_QUERY, &q) == -1 && errno == EBUSY);
  CHECK(mmap(NULL, 4096, PROT_READ, MAP_SHARED, b, ar.mmap_offset) == MAP_FAILED);
  CHECK(mmap(NULL, 4096, PROT_READ, MAP_SHARED, legacy, ar.mmap_offset) == MAP_FAILED);
  close(legacy);
  char *am = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, a, ar.mmap_offset);
  char *bm = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, b, br.mmap_offset);
  CHECK(am != MAP_FAILED && bm != MAP_FAILED);
  memset(am, 0x5a, 4096);
  memset(bm, 0xa5, 4096);
  int duplicate = dup(a);
  CHECK(duplicate >= 0);
  close(a);
  CHECK(live(b) == baseline + 2);
  close(duplicate); /* VMA retains the last file reference. */
  CHECK(live(b) == baseline + 2);
  int ready[2], release[2];
  CHECK(!pipe(ready) && !pipe(release));
  pid_t child = fork();
  CHECK(child >= 0);
  if (!child) {
    close(ready[0]); close(release[1]);
    munmap(bm, 4096); close(b);
    if (am[0] != 0x5a || write(ready[1], "x", 1) != 1) _exit(2);
    char c;
    if (read(release[0], &c, 1) != 1 || am[4095] != 0x5a) _exit(2);
    munmap(am, 4096);
    _exit(0);
  }
  close(ready[1]); close(release[0]);
  char c;
  CHECK(read(ready[0], &c, 1) == 1);
  CHECK(!munmap(am, 4096));
  CHECK(live(b) == baseline + 2); /* Forked VMA is still alive. */
  CHECK(write(release[1], "x", 1) == 1);
  int status;
  CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status));
  CHECK(live(b) == baseline + 1);
  a = owner();
  struct ioctl_region_create_args reused = {.len = 4096};
  CHECK(a >= 0 && !ioctl(a, IOCTL_REGION_CREATE, &reused));
  CHECK(reused.region_id == ar.region_id);
  am = mmap(NULL, 4096, PROT_READ, MAP_SHARED, a, reused.mmap_offset);
  CHECK(am != MAP_FAILED);
  for (unsigned i = 0; i < 4096; ++i) CHECK(am[i] == 0 && (unsigned char)bm[i] == 0xa5);
  CHECK(ioctl(a, IOCTL_REGION_CREATE, (void *)1) == -1 && errno == EFAULT);
  struct ioctl_region_create_args huge = {.len = 1UL << 40};
  CHECK(ioctl(a, IOCTL_REGION_CREATE, &huge) == -1 && errno == EINVAL);
  munmap(am, 4096); close(a);
  /* Failed copy_to_user must leave the accepted extent owned until close. */
  a = owner();
  struct ioctl_region_create_args *ro = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(a >= 0 && ro != MAP_FAILED);
  ro->len = 4096;
  CHECK(!mprotect(ro, 4096, PROT_READ));
  CHECK(ioctl(a, IOCTL_REGION_CREATE, ro) == -1 && errno == EFAULT);
  close(a); munmap(ro, 4096);
  CHECK(live(b) == baseline + 1);
  munmap(bm, 4096);
  int stats = owner();
  close(b);
  CHECK(stats >= 0 && live(stats) == baseline);
  close(stats);
  puts("process ownership, dup/fork/mmap lifetime, scrub, rollback: PASS");
  return 0;
}
