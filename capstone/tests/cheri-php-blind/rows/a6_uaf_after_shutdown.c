/* POSITIVE CONTROL FOR REVOCATION, INSIDE THE STOCK ARM.
 *
 * Same stock allocator, but the use happens AFTER shutdown_memory_manager has
 * drained the cache and genuinely free()d the block. If the vehicle's
 * revocation works at all, eager (and possibly async) must fault here. A MISS
 * on a1-a4 combined with a CATCH here is the precise evidence that CHERI's
 * blindness is caused by free() never being reached -- not by a misconfigured
 * image. Note this row deliberately uses the phases out of order. */
#include "row.h"
#include <string.h>

int main(void)
{
    char *s;
    setvbuf(stdout, (char *) NULL, _IONBF, 0);
    printf("ROWNAME a6_uaf_after_shutdown\n");
    zend_alloc_startup();
    printf("LAYOUT hdr=%zu cache=%d rs(26)=%zu\n",
           php_alloc_header_bytes(), php_alloc_cache_is_on(),
           php_alloc_real_size(26));
    s = (char *) emalloc(32);
    memset(s, 'r', 32);
    efree(s);                          /* cached, still malloc-live */
    ROW_MARK("SHUT-BEGIN");
    shutdown_memory_manager();         /* NOW the block is really free()d */
    ROW_MARK("SHUT-END");
    ROW_MARK("USE-BEGIN");
    printf("POST-FREE-READ byte0=%c\n", s[0]);   /* genuine post-free use */
    row_consume(s, 32);
    ROW_MARK("USE-END");
    ROW_MARK("DONE");
    return 0;
}
