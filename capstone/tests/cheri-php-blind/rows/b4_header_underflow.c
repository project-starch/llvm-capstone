/* Supplementary, not one of the 31: a read BELOW the object, into the
 * zend_mem_header. malloc's capability covers the header too, and emalloc
 * returns (char*)p + hdr with those same bounds, so an underflow is in bounds
 * as well. Shows the blindness is two-sided, not just a tail-slack artefact.
 * Unlike the slack rows this one does NOT flip under -DPHP_PATCHED_ALLOC: the
 * header is present in both arms. */
#include "row.h"
#include <string.h>
ROW_MAIN("b4_header_underflow", {
    char *s = (char *) emalloc(32);
    memset(s, 'u', 32);
    printf("UNDERFLOW reading s[-1] and s[-8] (inside zend_mem_header)\n");
    printf("  s[-1]=%02x s[-8]=%02x\n",
           (unsigned char) s[-1], (unsigned char) s[-8]);   /* under-read */
    row_consume(s - 8, 8);
    efree(s);
})
