/* Double efree. Stock allocator: the SAME block is pushed onto the cache twice,
 * so no double free() happens at the efree sites at all. It surfaces only at
 * shutdown, where the cache drain free()s the block twice -- which is why the
 * pristine PHP build crashes inside shutdown_memory_manager at
 * REMOVE_POINTER_FROM_LIST (Zend/zend_alloc.c:490), not at the defect. A fault
 * in the SHUT phase is therefore a LATE catch, not a catch of the defect. */
#include "row.h"
#include <string.h>
/* The header the allocator will read back on the second efree. Printing the TAG
 * of its list links turns "this must be a tag fault" into a measurement: if the
 * links come back untagged after the block is freed, the fault on the second
 * efree is CHERI's tag check on a clobbered pointer field, not an allocator
 * double-free abort -- and it needs no revocation, which is why it fires even in
 * the spatial config. */
typedef struct { void *pNext; void *pLast; unsigned int sz; } hdr_view;

ROW_MAIN("a5_double_efree", {
    char *s = (char *) emalloc(24);
    hdr_view *h = (hdr_view *) (void *) (s - php_alloc_header_bytes());
    memset(s, 'd', 24);
    printf("LINK-TAGS-BEFORE pNext=%d pLast=%d\n",
           (int) __builtin_cheri_tag_get(h->pNext),
           (int) __builtin_cheri_tag_get(h->pLast));
    efree(s);
    printf("LINK-TAGS-AFTER-1ST pNext=%d pLast=%d\n",
           (int) __builtin_cheri_tag_get(h->pNext),
           (int) __builtin_cheri_tag_get(h->pLast));
    efree(s);                          /* second efree of the same block */
    printf("DOUBLE-EFREE-RETURNED\n");
})
