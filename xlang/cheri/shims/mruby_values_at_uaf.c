/* Row 8 — mruby#4926, UAF in hash_values_at (mrb_get_args copying inversion).
 * asan.txt: heap-use-after-free READ of size 8, hash_values_at hash-ext.c:33,
 *           128 bytes inside a 2048-byte VM stack region.
 * Freed by: mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during lookup #2 of values_at; the UAF read is lookup #3.
 */
#define ROW_ID "mruby_values_at_uaf"
#define STACK_BYTES 2048
#define ACCESS_OFF 128
#define ACCESS_SIZE 8
#define ROW_IS_WRITE 0
#include "vm_stack_uaf.h"
