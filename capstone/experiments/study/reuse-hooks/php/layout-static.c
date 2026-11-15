/* The static layout check of docs/plans/2026-11-15-php-port-layout-fix.md: PHP's block header
 * as the compiler lays it out for this target, built with the realphp flags of one arm
 * (build-layout-trace.sh). Both arms must print the same line. */
#include "zend.h"
#include "zend_alloc.h"
#include <stddef.h>
#include <stdio.h>

int main(void)
{
    printf("LAYOUT-STATIC sizeof(zend_mem_header)=%lu MEM_HEADER_PADDING=%lu offsetof(pNext)=%lu "
           "offsetof(pLast)=%lu sizeof(void*)=%lu PLATFORM_ALIGNMENT=%lu MAX_CACHED_MEMORY=%d "
           "MAX_CACHED_ENTRIES=%d\n",
           (unsigned long)sizeof(zend_mem_header), (unsigned long)MEM_HEADER_PADDING,
           (unsigned long)offsetof(zend_mem_header, pNext),
           (unsigned long)offsetof(zend_mem_header, pLast), (unsigned long)sizeof(void *),
           (unsigned long)PLATFORM_ALIGNMENT, MAX_CACHED_MEMORY, MAX_CACHED_ENTRIES);
    return 0;
}
