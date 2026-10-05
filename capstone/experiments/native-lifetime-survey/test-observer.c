#include "observer.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
static unsigned char arena[512];
static void *worker(void *unused) {
    (void)unused;
    unsigned char *root=malloc(128);
    for(unsigned i=0;i<1000;i++) {
        ns_alloc(NS_MC_CACHE,root,root+16,32);
        ns_free(NS_MC_CACHE,root,root+16);
    }
    ns_destroy(NS_MC_CACHE,root);
    free(root);
    return NULL;
}
int main(int argc, char **argv) {
    if(argc>1 && !strcmp(argv[1],"threads")) {
        pthread_t threads[4];
        for(unsigned i=0;i<4;i++) if(pthread_create(&threads[i],NULL,worker,NULL)) return 2;
        for(unsigned i=0;i<4;i++) if(pthread_join(threads[i],NULL)) return 3;
        return 0;
    }
    unsigned f=NS_SQLITE_LOOKASIDE;
    const void *c=(void *)1;
    ns_backing_acquire(arena,sizeof arena);
    ns_alloc(f,c,arena,16);
    if(argc>1 && !strcmp(argv[1],"overlap")) {
        ns_alloc(f,c,arena,16); return 1;
    }
    ns_free(f,c,arena);
    ns_alloc(f,c,arena,16); /* same live backing, gap 1 */
    ns_free(f,c,arena);
    ns_backing_release(arena);
    ns_backing_acquire(arena,sizeof arena);
    ns_alloc(f,c,arena,16); /* same numeric backing address, new generation */
    ns_alloc(f,c,arena+32,16);
    ns_free(f,c,arena+32);
    ns_bulk(f,c); /* only arena remains alive */
    ns_alloc(f,c,arena,16); /* gap 1 after reset */
    ns_resize(f,c,arena,arena,24,1); /* not a new object */
    ns_resize(f,c,arena,NULL,48,1); /* failed resize preserves object */
    ns_free(f,c,arena);
    ns_destroy(f,c);
    ns_alloc(f,c,arena,16); /* cross-instance reuse has no local gap */
    ns_free(f,c,arena);
    ns_backing_release(arena);
    ns_alloc(f,c,arena+256,16); /* deliberately unknown backing */
    ns_free(f,c,arena+256);
    ns_alloc(f,c,arena+256,16);
    ns_free(f,c,arena+256);
    void *root=malloc(128);
    ns_alloc(NS_WMEM_SIMPLE,c,root,128);
    ns_free(NS_WMEM_SIMPLE,c,root);
    free(root);
    /* A failed real libc realloc must retain the original backing generation. */
    root=malloc(128);
    ns_alloc(NS_WMEM_STRICT,c,root,128);
    ns_free(NS_WMEM_STRICT,c,root);
    volatile size_t impossible=(size_t)-1/2;
    void *failed=realloc(root,impossible);
    if(failed) return 4;
    ns_alloc(NS_WMEM_STRICT,c,root,128);
    ns_free(NS_WMEM_STRICT,c,root);
    free(root);
    return 0;
}
