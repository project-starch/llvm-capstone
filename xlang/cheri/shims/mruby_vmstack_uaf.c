/* Row 4 — CVE-2022-1071, mruby UAF in OP_GETCONST handling.
 * asan.txt: heap-use-after-free WRITE of size 8, mrb_vm_exec vm.c:1426,
 *           16 bytes inside a 1024-byte VM stack region.
 * Freed by: mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during the const_missing callback (recurse(150), 100 locals/frame).
 */
#define ROW_ID "mruby_vmstack_uaf"
#define STACK_BYTES 1024
#define ACCESS_OFF 16
#define ACCESS_SIZE 8
#define ROW_IS_WRITE 1
#include "vm_stack_uaf.h"
