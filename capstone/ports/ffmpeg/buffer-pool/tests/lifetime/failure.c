#include <assert.h>
#include <stdio.h>
#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
extern int life_renew_fail;
static int freed;
static void entry_free(AVRefStructOpaque opaque, void *p){
    (void)opaque; (void)p; freed++;
}
int main(int argc, char **argv){
    AVBufferPool *pool = av_buffer_pool_init(64, NULL);
    AVBufferRef *b;
    AVRefStructPool *rp;
    void *obj;
    unsigned char *stale;
    assert(pool);
    b = av_buffer_pool_get(pool); assert(b);
    stale = b->data; *stale = 42;
    if (argc > 1 && argv[1][0]!='f') {
        av_buffer_unref(&b);
        if (argv[1][0]=='b') return *(volatile unsigned char *)stale;
        av_buffer_pool_uninit(&pool);
        rp = av_refstruct_pool_alloc_ext(64, 0, NULL, NULL, NULL, entry_free, NULL);
        obj = av_refstruct_pool_get(rp); assert(obj);
        stale = obj; *stale = 42;
        av_refstruct_unref(&obj);
        return *(volatile unsigned char *)stale;
    }
    life_renew_fail = argc == 1;
    av_buffer_unref(&b);
    life_renew_fail = 0;
    b = av_buffer_pool_get(pool); assert(b && b->data);
    av_buffer_unref(&b); av_buffer_pool_uninit(&pool);
    rp = av_refstruct_pool_alloc_ext(64, 0, NULL, NULL, NULL, entry_free, NULL);
    assert(rp);
    obj = av_refstruct_pool_get(rp); assert(obj);
    life_renew_fail = 1;
    av_refstruct_unref(&obj);
    assert(freed==1);
    life_renew_fail = 0;
    obj = av_refstruct_pool_get(rp); assert(obj);
    av_refstruct_unref(&obj); av_refstruct_pool_uninit(&rp);
    assert(freed==2);
    pool = av_buffer_pool_init(64, NULL); assert(pool);
    b = av_buffer_pool_get(pool); assert(b);
    av_buffer_pool_uninit(&pool);
    b->data[0] = 7;
    av_buffer_unref(&b);
    rp = av_refstruct_pool_alloc(64, 0); assert(rp);
    obj = av_refstruct_pool_get(rp); assert(obj);
    av_refstruct_pool_uninit(&rp);
    *(unsigned char *)obj = 7;
    av_refstruct_unref(&obj);
    puts("LIFETIME ffmpeg PASS");
    return 0;
}
