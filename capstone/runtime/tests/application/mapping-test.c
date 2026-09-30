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
#define mprotect port_mprotect
#define __mprotect port_internal_mprotect
/* The runtime's locks (capstone/lock.h) are the domain's; one native thread
   needs none. */
#include <capstone/lock.h>
void capstone_lock(volatile int *word) { (void)word; }
void capstone_unlock(volatile int *word) { (void)word; }
/* hostcall.c's unserved report: count what mprotect refuses. */
static int unserved;
void __capstone_hc_note_unserved(long n) { (void)n; ++unserved; }
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
  /* mprotect answers 0 only for read-write pages inside one mapping (a
     thread stack past its guard page), and reports everything else. */
  p = mmap(NULL, 3 * 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(p != MAP_FAILED);
  assert(mprotect(p + 4096, 2 * 4096, PROT_READ | PROT_WRITE) == 0 && unserved == 0);
  assert(mprotect(p, 3 * 4096, PROT_READ | PROT_WRITE) == 0 && unserved == 0);
  errno = 0;
  assert(mprotect(p + 4096, 3 * 4096, PROT_READ | PROT_WRITE) == -1 && errno == ENOSYS);
  assert(mprotect(p, 4096, PROT_NONE) == -1 && errno == ENOSYS);
  assert(mprotect(p, 4096, PROT_READ) == -1 && errno == ENOSYS && unserved == 3);
  assert(munmap(p, 3 * 4096) == 0);
  return 0;
}
