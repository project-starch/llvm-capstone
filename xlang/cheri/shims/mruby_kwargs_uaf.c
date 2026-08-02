/* Row 5 — CVE-2022-1934, mruby UAF in hash_new_from_values.
 * asan.txt: heap-use-after-free READ of size 8, hash_new_from_values
 *           vm.c:1167, 72 bytes inside a 1024-byte VM stack region.
 * Freed by: mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during a Bad#eql? callback; 3 keyword pairs mean a later loop
 *           iteration reads through the stale regs.
 */
#define ROW_ID "mruby_kwargs_uaf"
#define STACK_BYTES 1024
#define ACCESS_OFF 72
#define ACCESS_SIZE 8
#define ROW_IS_WRITE 0
#include "vm_stack_uaf.h"
