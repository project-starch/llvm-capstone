/* Row 13 — mruby#4927, UAF in hash_slice (mrb_get_args copying inversion).
 * asan.txt: heap-use-after-free READ of size 16, hash_slice hash-ext.c:59,
 *           128 bytes inside a 2048-byte VM stack region.
 * Freed by: mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during an #eql? callback that re-enters the VM. The stale argv is
 *           held in an explicit C local across a loop (one of the corpus's
 *           more robust rows).
 */
#define ROW_ID "mruby_hash_slice_uaf"
#define STACK_BYTES 2048
#define ACCESS_OFF 128
#define ACCESS_SIZE 16
#define ROW_IS_WRITE 0
#include "vm_stack_uaf.h"
