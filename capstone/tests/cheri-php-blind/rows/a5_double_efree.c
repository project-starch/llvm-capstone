/* Double efree. Stock allocator: the SAME block is pushed onto the cache twice,
 * so no double free() happens at the efree sites at all. It surfaces only at
 * shutdown, where the cache drain free()s the block twice -- which is why the
 * pristine PHP build crashes inside shutdown_memory_manager at
 * REMOVE_POINTER_FROM_LIST (Zend/zend_alloc.c:490), not at the defect. A fault
 * in the SHUT phase is therefore a LATE catch, not a catch of the defect. */
#include "row.h"
#include <string.h>
ROW_MAIN("a5_double_efree", {
    char *s = (char *) emalloc(24);
    memset(s, 'd', 24);
    efree(s);
    efree(s);                          /* second efree of the same block */
    printf("DOUBLE-EFREE-RETURNED\n");
})
