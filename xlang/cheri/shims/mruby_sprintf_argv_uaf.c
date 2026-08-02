/* Row 15 — mruby#3722, UAF in mrb_str_format (sprintf argv).
 * asan.txt: heap-use-after-free READ of size 16, mrb_str_format sprintf.c:735,
 *           80 bytes inside a 2048-byte VM stack region.
 * Freed by: stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during arg 1's to_s; the stale argv read is for arg 2.
 *
 * The row's QEMU leg is heap-layout sensitive (absolute vs relative paths).
 * That sensitivity belongs to the interpreter's layout, not to this shim,
 * which pins the geometry directly.
 */
#define ROW_ID "mruby_sprintf_argv_uaf"
#define STACK_BYTES 2048
#define ACCESS_OFF 80
#define ACCESS_SIZE 16
#define ROW_IS_WRITE 0
#include "vm_stack_uaf.h"
