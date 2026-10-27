/* The zval/refcount shape that dominates the corpus's CWE-416 set: a freed
 * struct's POINTER member is loaded and then dereferenced. On purecap that
 * member is a capability living in freed memory -- the measurement question is
 * whether its tag survives. Under the stock allocator the block is only cached,
 * so nothing has overwritten it and the inner capability is intact. */
#include "row.h"
#include <string.h>

typedef struct { int refcount; int len; char *val; } fake_zval;

ROW_MAIN("a4_uaf_zval_member", {
    fake_zval *z = (fake_zval *) emalloc(sizeof(fake_zval));
    z->refcount = 1;
    z->len = 11;
    z->val = (char *) emalloc(12);
    memcpy(z->val, "hello world", 12);

    efree(z);                          /* object gone, dangling handle kept */
    printf("STALE-REFCOUNT=%d\n", z->refcount);
    printf("STALE-MEMBER-DEREF [%s]\n", z->val);   /* load cap from freed mem, deref */
    row_consume(z->val, 12);
    efree(z->val);
})
