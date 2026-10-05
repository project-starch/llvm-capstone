/* Linux/glibc observation only. No allocator policy or returned pointer changes. */
#define _GNU_SOURCE
#include "observer.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>

#define ROOTS (1u << 19)
#define OBJECTS (1u << 21)
#define SCOPES (1u << 18)
#define BINS 64
typedef uint64_t U;
typedef struct { uintptr_t p; size_t n; U generation, priority; unsigned l, r, next; } Root;
typedef struct {
    uintptr_t p; U backing, scope, retired;
    size_t size; unsigned family, live, prev, next;
} Object;
typedef struct {
    uintptr_t p; U generation, clock; unsigned family, head;
} Scope;
typedef struct {
    U alloc, free, bulk_free, bulk_calls, destroy, noop_free;
    U reuse, inside, outside, unknown_reuse, unknown_alloc;
    U cross_instance, unknown_free, resize_inplace, resize_failed;
    U live, peak_live, live_bytes, peak_bytes, gap_inside[BINS], gap_other[BINS];
} Counts;
static Root roots[ROOTS];
static Object objects[OBJECTS];
static Scope scopes[SCOPES];
static Counts counts[NS_FAMILY_COUNT];
static const char *names[NS_FAMILY_COUNT] = {
    "invalid", "sqlite-lookaside", "sqlite-memsys5", "postgres-aset",
    "postgres-generation", "postgres-slab", "postgres-bump", "cpython-pymalloc",
    "mruby-gc", "perl-sv", "ffmpeg-buffer-pool", "ffmpeg-refstruct-pool",
    "wmem-simple", "wmem-strict", "wmem-block", "wmem-block-fast",
    "memcached-slab", "memcached-object-cache"
};
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local unsigned internal;
static unsigned tree, root_used, root_free, initialized;
static U next_root, next_scope, root_acquires, root_releases;
static const char *output;
static pid_t owner;

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
extern void __libc_free(void *);

static void fail(const char *message) {
    ssize_t written=write(2, "native-survey: ", 15);
    written=write(2, message, strlen(message));
    written=write(2, "\n", 1);
    (void)written;
    _exit(86);
}
static U mix(U x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static int enter(void) {
    if (internal) return 0;
    internal = 1;
    pthread_mutex_lock(&mutex);
    if (!initialized) {
        output = getenv("NS_OUT"); owner = getpid(); initialized = 1;
    }
    if (!output) {
        pthread_mutex_unlock(&mutex); internal = 0; return 0;
    }
    if (owner != getpid()) fail("forked measured process requires a separate execution");
    return 1;
}
static void leave(void) { pthread_mutex_unlock(&mutex); internal = 0; }
static unsigned merge(unsigned a, unsigned b) {
    if (!a) return b;
    if (!b) return a;
    if (roots[a].priority < roots[b].priority) {
        roots[a].r = merge(roots[a].r, b); return a;
    }
    roots[b].l = merge(a, roots[b].l); return b;
}
static unsigned insert(unsigned t, unsigned x) {
    if (!t) return x;
    if (roots[x].p == roots[t].p) fail("duplicate live backing");
    if (roots[x].p < roots[t].p) {
        roots[t].l = insert(roots[t].l, x);
        unsigned l = roots[t].l;
        if (roots[l].priority < roots[t].priority) {
            roots[t].l = roots[l].r; roots[l].r = t; return l;
        }
    } else {
        roots[t].r = insert(roots[t].r, x);
        unsigned r = roots[t].r;
        if (roots[r].priority < roots[t].priority) {
            roots[t].r = roots[r].l; roots[r].l = t; return r;
        }
    }
    return t;
}
static unsigned remove_root(unsigned t, uintptr_t p) {
    if (!t) return 0;
    if (roots[t].p == p) {
        unsigned next = merge(roots[t].l, roots[t].r);
        roots[t].next = root_free; root_free = t; root_releases++;
        return next;
    }
    if (p < roots[t].p) roots[t].l = remove_root(roots[t].l, p);
    else roots[t].r = remove_root(roots[t].r, p);
    return t;
}
static unsigned containing(uintptr_t p, size_t size) {
    unsigned t = tree, found = 0;
    while (t) {
        if (roots[t].p <= p) { found = t; t = roots[t].r; }
        else t = roots[t].l;
    }
    if (found && p - roots[found].p < roots[found].n &&
        size <= roots[found].n - (p - roots[found].p)) return found;
    return 0;
}
static void acquire(const void *ptr, size_t n) {
    if (!ptr) return;
    uintptr_t p = (uintptr_t)ptr;
    if (!n) n = 1;
    if (containing(p, 1)) fail("overlapping backing registration");
    unsigned t = tree;
    while (t) {
        if (roots[t].p > p && roots[t].p - p < n) fail("overlapping backing extent");
        t = p < roots[t].p ? roots[t].l : roots[t].r;
    }
    unsigned x;
    if (root_free) { x = root_free; root_free = roots[x].next; }
    else { x = ++root_used; if (x == ROOTS) fail("backing capacity exhausted"); }
    next_root++;
    roots[x] = (Root){.p=p, .n=n, .generation=next_root, .priority=mix(next_root)};
    tree = insert(tree, x); root_acquires++;
}
static void family_check(unsigned f) {
    if (!f || f >= NS_FAMILY_COUNT) fail("invalid allocator family");
}
static unsigned scope_find(unsigned f, const void *p) {
    unsigned i = (unsigned)mix((uintptr_t)p ^ ((U)f << 48)) & (SCOPES - 1);
    for (unsigned n=0; n<SCOPES; n++, i=(i+1)&(SCOPES-1)) {
        if (!scopes[i].family) {
            scopes[i] = (Scope){.p=(uintptr_t)p,.family=f,.generation=++next_scope};
            return i;
        }
        if (scopes[i].p == (uintptr_t)p && scopes[i].family == f) return i;
    }
    fail("instance capacity exhausted"); return 0;
}
static unsigned object_find(unsigned f, const void *p) {
    /* Slot zero is reserved as the linked-list terminator. */
    unsigned i = ((unsigned)mix((uintptr_t)p ^ ((U)f << 48)) & (OBJECTS-1));
    for (unsigned n=0; n<OBJECTS; n++, i=(i+1)&(OBJECTS-1)) {
        if (!i) continue;
        if (!objects[i].family) {
            objects[i].p=(uintptr_t)p; objects[i].family=f; return i;
        }
        if (objects[i].p == (uintptr_t)p && objects[i].family == f) return i;
    }
    fail("object history capacity exhausted"); return 0;
}
static void retire(unsigned s, unsigned x, int bulk) {
    Object *o=&objects[x]; Scope *c=&scopes[s]; Counts *v=&counts[o->family];
    if (!o->live || o->scope != c->generation) fail("invalid object retirement");
    if (o->prev) objects[o->prev].next=o->next;
    else c->head=o->next;
    if (o->next) objects[o->next].prev=o->prev;
    o->live=0; o->retired=c->clock; o->next=o->prev=0;
    v->free++; v->bulk_free+=bulk; v->live--; v->live_bytes-=o->size;
}
static void free_object(unsigned f, const void *instance, const void *p) {
    if (!p) return;
    unsigned s=scope_find(f,instance), x=object_find(f,p);
    if (!objects[x].live) { counts[f].unknown_free++; return; }
    retire(s,x,0);
}
static void issue(unsigned f, const void *instance, const void *p, size_t size) {
    if (!p) return;
    unsigned s=scope_find(f,instance), x=object_find(f,p);
    Scope *c=&scopes[s]; Object *o=&objects[x]; Counts *v=&counts[f];
    unsigned r=containing((uintptr_t)p, size ? size : 1);
    U backing=r ? roots[r].generation : 0;
    if (o->live) fail("allocation overlaps a recorded live object");
    c->clock++; v->alloc++;
    if (!backing) v->unknown_alloc++;
    if (o->scope) {
        v->reuse++;
        int inside=backing && backing==o->backing;
        if (!backing || !o->backing) v->unknown_reuse++;
        else if (inside) v->inside++;
        else v->outside++;
        if (o->scope == c->generation) {
            U gap=c->clock-o->retired;
            unsigned bin=63u-(unsigned)__builtin_clzll(gap);
            if (inside) v->gap_inside[bin]++;
            else v->gap_other[bin]++;
        } else v->cross_instance++;
    }
    o->scope=c->generation; o->backing=backing; o->size=size; o->live=1;
    o->next=c->head; o->prev=0;
    if (c->head) objects[c->head].prev=x;
    c->head=x; v->live++; v->live_bytes+=size;
    if (v->live>v->peak_live) v->peak_live=v->live;
    if (v->live_bytes>v->peak_bytes) v->peak_bytes=v->live_bytes;
}
void ns_alloc(unsigned f, const void *c, const void *p, size_t n) {
    if (!enter()) return;
    family_check(f); issue(f,c,p,n); leave();
}
void ns_free(unsigned f, const void *c, const void *p) {
    if (!enter()) return;
    family_check(f); free_object(f,c,p); leave();
}
void ns_noop_free(unsigned f) {
    if (!enter()) return;
    family_check(f); counts[f].noop_free++; leave();
}
void ns_resize(unsigned f, const void *c, const void *p, const void *q, size_t n, int dies) {
    if (!enter()) return;
    family_check(f);
    if (!q && n) counts[f].resize_failed++;
    else if (p && p==q) {
        unsigned x=object_find(f,p);
        if (!objects[x].live) fail("in-place resize of unknown object");
        Counts *v=&counts[f]; Object *o=&objects[x];
        v->resize_inplace++; v->live_bytes-=o->size; v->live_bytes+=n; o->size=n;
        if (v->live_bytes>v->peak_bytes) v->peak_bytes=v->live_bytes;
        unsigned r=containing((uintptr_t)q,n?n:1);
        o->backing=r ? roots[r].generation : 0;
    } else {
        if (dies) free_object(f,c,p);
        issue(f,c,q,n);
    }
    leave();
}
static void bulk(unsigned f, const void *p, int destroy) {
    unsigned s=scope_find(f,p); Scope *c=&scopes[s];
    counts[f].bulk_calls++;
    while(c->head) retire(s,c->head,1);
    if (destroy) { counts[f].destroy++; c->generation=++next_scope; c->clock=0; }
}
void ns_bulk(unsigned f, const void *p) {
    if (!enter()) return;
    family_check(f); bulk(f,p,0); leave();
}
void ns_destroy(unsigned f, const void *p) {
    if (!enter()) return;
    family_check(f); bulk(f,p,1); leave();
}
void ns_backing_acquire(const void *p, size_t n) {
    if (!enter()) return;
    acquire(p,n); leave();
}
void ns_backing_release(const void *p) {
    if (!enter()) return;
    tree=remove_root(tree,(uintptr_t)p); leave();
}
void *malloc(size_t n) {
    if (!enter()) return __libc_malloc(n);
    void *p=__libc_malloc(n); acquire(p,n); leave(); return p;
}
void *calloc(size_t n, size_t size) {
    if (!enter()) return __libc_calloc(n,size);
    void *p=__libc_calloc(n,size); acquire(p,n*size); leave(); return p;
}
void free(void *p) {
    if (!enter()) { __libc_free(p); return; }
    tree=remove_root(tree,(uintptr_t)p); __libc_free(p); leave();
}
void *realloc(void *p, size_t n) {
    if (!enter()) return __libc_realloc(p,n);
    void *q=__libc_realloc(p,n);
    if (q || !n) {
        tree=remove_root(tree,(uintptr_t)p); acquire(q,n);
    }
    leave(); return q;
}
void *memalign(size_t alignment, size_t n) {
    if (!enter()) return __libc_memalign(alignment,n);
    void *p=__libc_memalign(alignment,n); acquire(p,n); leave(); return p;
}
void *aligned_alloc(size_t alignment, size_t n) { return memalign(alignment,n); }
int posix_memalign(void **p, size_t alignment, size_t n) {
    if (alignment<sizeof(void *) || (alignment & (alignment-1))) return EINVAL;
    int saved=errno;
    void *q=memalign(alignment,n);
    errno=saved;
    if (!q) return ENOMEM;
    *p=q; return 0;
}
static void put(int fd, const char *s) {
    size_t left=strlen(s);
    while(left) {
        ssize_t n=write(fd,s,left);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) fail("report write failed");
        s+=n; left-=(size_t)n;
    }
}
static void number(int fd, const char *key, U value) {
    char buf[128];
    snprintf(buf,sizeof buf,"\"%s\":%llu,",key,(unsigned long long)value); put(fd,buf);
}
void ns_report(void) {
    if (!enter()) return;
    char path[4096], buf[256];
    int len=snprintf(path,sizeof path,"%s.%d.json",output,(int)getpid());
    if(len<0 || len>=(int)sizeof path) fail("output path too long");
    int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600);
    if(fd<0) fail("cannot open report");
    put(fd,"{\"schema\":1,"); number(fd,"pid",getpid());
    number(fd,"backing_acquires",root_acquires); number(fd,"backing_releases",root_releases);
    put(fd,"\"allocators\":[");
    int comma=0;
    for(unsigned f=1;f<NS_FAMILY_COUNT;f++) {
        Counts *v=&counts[f];
        if(!v->alloc && !v->unknown_free && !v->bulk_calls) continue;
        if(comma++) put(fd,",");
        snprintf(buf,sizeof buf,"{\"family\":\"%s\",",names[f]); put(fd,buf);
#define FIELD(x) number(fd,#x,v->x)
        FIELD(alloc); FIELD(free); FIELD(bulk_free); FIELD(bulk_calls); FIELD(destroy);
        FIELD(noop_free); FIELD(reuse); FIELD(inside); FIELD(outside); FIELD(unknown_reuse);
        FIELD(unknown_alloc); FIELD(cross_instance); FIELD(unknown_free);
        FIELD(resize_inplace); FIELD(resize_failed); FIELD(live); FIELD(peak_live);
        FIELD(live_bytes); FIELD(peak_bytes);
#undef FIELD
        for(unsigned h=0;h<2;h++) {
            put(fd,h?",\"gap_other\":[":"\"gap_inside\":[");
            for(unsigned b=0;b<BINS;b++) {
                snprintf(buf,sizeof buf,"%s%llu",b?",":"",(unsigned long long)(h?v->gap_other[b]:v->gap_inside[b]));
                put(fd,buf);
            }
            put(fd,"]");
        }
        put(fd,"}");
    }
    put(fd,"]}\n");
    if(close(fd)) fail("report close failed");
    leave();
}
__attribute__((destructor)) static void report_at_exit(void) { ns_report(); }
