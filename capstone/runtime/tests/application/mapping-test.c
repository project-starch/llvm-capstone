/* Rename the overrides so this native test does not interpose the sanitizer's
 * own mappings. Exercise the same allocation and size checks as applications. */
#define mmap port_mmap
#define __mmap port_internal_mmap
#define munmap port_munmap
#define __munmap port_internal_munmap
#define shmget port_shmget
#define shmat port_shmat
#define shmdt port_shmdt
#define shmctl port_shmctl
#define __vm_wait port_vm_wait
#include "../../../ports/musl-capstone/runtime/mmap_shm_level0.c"
#include <assert.h>

int main(void) {
  for (size_t n = SIZE_MAX - 8191; n != 0; ++n) {
    errno = 0;
    assert(mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0) == MAP_FAILED && errno == ENOMEM);
    errno = 0;
    assert(shmget(IPC_PRIVATE, n, IPC_CREAT | 0600) == -1 && errno == ENOMEM);
  }
  char *p = mmap(NULL, 4097, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(p != MAP_FAILED && ((uintptr_t)p & 4095) == 0);
  for (size_t i = 0; i < 4097; ++i) assert(p[i] == 0);
  p[4096] = 42;
  assert(munmap(p, 4096) == -1 && errno == EINVAL);
  assert(p[4096] == 42 && munmap(p, 4097) == 0);
  int id = shmget(IPC_PRIVATE, 4097, IPC_CREAT | 0600);
  assert(id > 0);
  p = shmat(id, NULL, 0);
  assert(p != (void *)-1);
  p[4096] = 17;
  assert(shmctl(id, IPC_RMID, NULL) == 0 && p[4096] == 17);
  assert(shmdt(p) == 0);
  return 0;
}
