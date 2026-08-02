/* Row 12 — mruby#4001, UAF in File#initialize_copy via a dangling DATA_PTR.
 * asan.txt: heap-use-after-free READ of size 4, io_get_open_fptr io.c:78,
 *           0 bytes inside a 16-byte region.
 * Freed by: mrb_io_initialize_copy -> mrb_free -> free. An explicit gem-code
 *           free, not GC and not a realloc move.
 *
 * Mechanism: initialize_copy frees the receiver's DATA_PTR first, then
 * io_get_open_fptr raises TypeError on the invalid source and longjmps out
 * before a replacement struct is assigned back. The receiver keeps the
 * dangling pointer; File#close then reads through it.
 */
#include "../mock-mruby/mock_mruby.h"
#include <stdint.h>

struct mrb_io { /* 16 bytes, as ASan reports the region */
  int fd;
  int fd2;
  int readable;
  int writable;
};

static volatile int sink;

int main(void) {
  mrb_state *mrb = mrb_open(1024);

  struct mrb_io *fptr = (struct mrb_io *)mrb_malloc(mrb, sizeof *fptr);
  fptr->fd = 3;

  /* File#initialize_copy(0): frees DATA_PTR, then bails out on TypeError. */
  mrb_free(mrb, fptr);
  /* ... longjmp; the receiver's DATA_PTR is never replaced ... */

  /* File#close -> io_get_open_fptr reads fptr->fd through the dangling ptr. */
  sink = ((volatile struct mrb_io *)fptr)->fd; /* READ of size 4 at +0 */

  mock_report("mruby_io_dataptr_uaf", "use-after-free-survived");
  mrb_close(mrb);
  return 0;
}
