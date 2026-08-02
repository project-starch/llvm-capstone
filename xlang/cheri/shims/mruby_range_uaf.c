/* Row 10 — CVE-2022-1106, mruby OP_RANGE_INC stale regs[a] write.
 * The corpus's template row (task spec §5).
 * asan.txt: heap-use-after-free WRITE of size 8, mrb_vm_exec vm.c:2822,
 *           16 bytes inside a 1024-byte VM stack region.
 * Freed by: mrb_stack_extend -> stack_extend_alloc -> mrb_realloc -> realloc,
 *           during Bad#<=> on its FIRST invocation only.
 *
 * Row caveat carried over: the native reproduction depends on the compiler
 * caching &regs[a] across the call. Here that is not left to chance — the
 * shim holds the pointer in a named local, which is why this row is as
 * robust as rows 8/13/15 rather than compiler-sensitive.
 */
#define ROW_ID "mruby_range_uaf"
#define STACK_BYTES 1024
#define ACCESS_OFF 16
#define ACCESS_SIZE 8
#define ROW_IS_WRITE 1
#include "vm_stack_uaf.h"
