#!/usr/bin/env python3
"""Add observation-only hooks to the exact source versions in sources.json.

Edits are checked in memory before any file is written. Missing or duplicate
anchors fail the build. No Capstone adapter or allocation policy is imported.
"""
from pathlib import Path
import sys

class Edits:
    def __init__(self, root):
        self.root, self.files = root, {}

    def text(self, name):
        if name not in self.files:
            self.files[name] = (self.root / name).read_text()
        return self.files[name]

    def replace(self, name, old, new, count=1):
        text = self.text(name)
        if text.count(old) != count:
            raise ValueError(f'{name}: expected {count} anchors, found {text.count(old)}: {old[:100]!r}')
        self.files[name] = text.replace(old, new)

    def after(self, name, anchor, addition, count=1):
        self.replace(name, anchor, anchor + addition, count)

    def include(self, name):
        text = self.text(name)
        if '"observer.h"' in text:
            raise ValueError(f'{name}: already instrumented')
        # Place after existing includes through an explicit anchor at each site.
        self.files[name] = '#include "observer.h"\n' + text

    def save(self):
        for name, text in self.files.items():
            (self.root / name).write_text(text)
            print(name)

def sqlite(e):
    f='sqlite3.c'
    e.include(f)
    e.after(f, '      db->lookaside.anStat[0]++;\n',
            '      ns_alloc(NS_SQLITE_LOOKASIDE, db, pBuf, n);\n', 2)
    e.after(f, '      LookasideSlot *pBuf = (LookasideSlot*)p;\n',
            '      ns_free(NS_SQLITE_LOOKASIDE, db, p);\n')
    e.replace(f, '  if( db->lookaside.bMalloced ){\n',
              '  ns_destroy(NS_SQLITE_LOOKASIDE, db);\n  if( db->lookaside.bMalloced ){\n', 2)
    e.replace(f, '  return (void*)&mem5.zPool[i*mem5.szAtom];',
              '  ns_alloc(NS_SQLITE_MEMSYS5, &mem5, &mem5.zPool[i*mem5.szAtom], nByte);\n'
              '  return (void*)&mem5.zPool[i*mem5.szAtom];')
    e.after(f, 'static void memsys5FreeUnsafe(void *pOld){\n',
            '  ns_free(NS_SQLITE_MEMSYS5, &mem5, pOld);\n')

def cpython(e):
    f='Objects/obmalloc.c'
    e.after(f, '#include <stdbool.h>\n', '#include "observer.h"\n')
    e.replace(f, 'pymalloc_alloc(OMState *state, void *Py_UNUSED(ctx), size_t nbytes)\n{',
              'ns_real_pymalloc_alloc(OMState *state, void *Py_UNUSED(ctx), size_t nbytes)\n{')
    e.replace(f, 'pymalloc_free(OMState *state, void *Py_UNUSED(ctx), void *p)\n{',
              'ns_real_pymalloc_free(OMState *state, void *Py_UNUSED(ctx), void *p)\n{')
    wrapper='''
static inline int ns_real_pymalloc_free(OMState *, void *, void *);
static inline void *pymalloc_alloc(OMState *state, void *ctx, size_t n) {
    void *p=ns_real_pymalloc_alloc(state,ctx,n);
    if(p) ns_alloc(NS_PYMALLOC,state,p,n);
    return p;
}
static inline int pymalloc_free(OMState *state, void *ctx, void *p) {
    int mine=ns_real_pymalloc_free(state,ctx,p);
    if(mine) ns_free(NS_PYMALLOC,state,p);
    return mine;
}
'''
    anchor='\nvoid *\n_PyObject_Malloc(void *ctx, size_t nbytes)'
    e.replace(f,anchor,wrapper+anchor)
    # Default Linux arena allocation uses mmap, outside malloc interposition.
    e.after(f,'    if (ptr == MAP_FAILED)\n        return NULL;\n',
            '    ns_backing_acquire(ptr, size);\n')
    e.replace(f,'    munmap(ptr, size);',
              '    ns_backing_release(ptr);\n    munmap(ptr, size);')

def mruby(e):
    f='src/gc.c'
    e.include(f)
    e.after(f,'  paint_partial_white(gc, &p->as.basic);\n',
            '  ns_alloc(NS_MRUBY_GC,mrb,&p->as.basic,sizeof(RVALUE));\n')
    e.after(f,'obj_free(mrb_state *mrb, struct RBasic *obj, mrb_bool end)\n{\n',
            '  ns_free(NS_MRUBY_GC,mrb,obj);\n')

def perl(e):
    e.include('sv_inline.h')
    e.after('sv_inline.h','        ++PL_sv_count;                                  \\\n',
            '        ns_alloc(NS_PERL_SV, &PL_sv_count, (p), sizeof(SV)); \\\n')
    e.after('sv.c','#define plant_SV(p) \\\n    STMT_START {\t\t\t\t\t\\\n',
            '        ns_free(NS_PERL_SV, &PL_sv_count, (p)); \\\n')

def postgresql(e):
    f='src/backend/utils/mmgr/mcxt.c'
    e.after(f,'#include "utils/memutils_memorychunk.h"', '\n#include "observer.h"')
    e.replace(f,'static const MemoryContextMethods mcxt_methods[] = {',
              'static MemoryContextMethods mcxt_methods[] = {')
    text=e.text(f)
    begin=text.index('static MemoryContextMethods mcxt_methods[] = {')
    end=text.index('\n};\n',begin)+4
    wrappers='\nstatic MemoryContextMethods ns_original[lengthof(mcxt_methods)];\n'
    kinds=[('MCTX_ASET_ID','aset','NS_PG_ASET'),('MCTX_GENERATION_ID','generation','NS_PG_GENERATION'),
           ('MCTX_SLAB_ID','slab','NS_PG_SLAB'),('MCTX_BUMP_ID','bump','NS_PG_BUMP')]
    for ident,tag,family in kinds:
        wrappers+=f'''
static void *ns_alloc_{tag}(MemoryContext c, Size n, int flags) {{
    void *p=ns_original[{ident}].alloc(c,n,flags);
    ns_alloc({family},c,p,n); return p;
}}
static void ns_free_{tag}(void *p) {{
    MemoryContext c=p ? GetMemoryChunkContext(p) : NULL;
    ns_free({family},c,p); ns_original[{ident}].free_p(p);
}}
static void *ns_realloc_{tag}(void *p, Size n, int flags) {{
    MemoryContext c=p ? GetMemoryChunkContext(p) : NULL;
    void *q=ns_original[{ident}].realloc(p,n,flags);
    ns_resize({family},c,p,q,n,1); return q;
}}
static void ns_reset_{tag}(MemoryContext c) {{
    ns_bulk({family},c); ns_original[{ident}].reset(c);
}}
static void ns_delete_{tag}(MemoryContext c) {{
    ns_destroy({family},c); ns_original[{ident}].delete_context(c);
}}
'''
    wrappers+='\nstatic void ns_install(void) {\n'
    for ident,tag,_ in kinds:
        wrappers+=f'    ns_original[{ident}]=mcxt_methods[{ident}];\n'
        for method,hook in [('alloc','alloc'),('free_p','free'),('realloc','realloc'),
                            ('reset','reset'),('delete_context','delete')]:
            wrappers+=f'    mcxt_methods[{ident}].{method}=ns_{hook}_{tag};\n'
    wrappers+='}\n'
    e.files[f]=text[:end]+wrappers+text[end:]
    e.after(f,'MemoryContextInit(void)\n{\n\tAssert(TopMemoryContext == NULL);', '\n\tns_install();')

def ffmpeg(e):
    f='libavutil/buffer.c'
    e.include(f)
    e.after(f,'    BufferPoolEntry *buf = opaque;\n    AVBufferPool *pool = buf->pool;\n',
            '    ns_free(NS_AVBUFFER,pool,data);\n')
    e.replace(f,'    if (ret)\n        atomic_fetch_add_explicit(&pool->refcount, 1, memory_order_relaxed);\n\n    return ret;\n}',
              '    if (ret) {\n        atomic_fetch_add_explicit(&pool->refcount, 1, memory_order_relaxed);\n'
              '        ns_alloc(NS_AVBUFFER,pool,ret->data,pool->size);\n    }\n\n    return ret;\n}')
    e.after(f,'static void buffer_pool_free(AVBufferPool *pool)\n{\n',
            '    ns_destroy(NS_AVBUFFER,pool);\n')
    f='libavutil/refstruct.c'
    e.include(f)
    e.after(f,'    AVRefStructPool *pool = ref->opaque.nc;\n',
            '    ns_free(NS_AVREFSTRUCT,pool,get_userdata(ref));\n')
    e.after(f,'    memcpy(datap, &ret, sizeof(ret));\n',
            '    ns_alloc(NS_AVREFSTRUCT,pool,ret,pool->size);\n')
    e.after(f,'static void pool_free(AVRefStructPool *pool)\n{\n',
            '    ns_destroy(NS_AVREFSTRUCT,pool);\n')

def wireshark(e):
    f='wsutil/wmem/wmem_core.c'
    e.include(f)
    mapping='''
static unsigned ns_wmem_family(wmem_allocator_t *allocator) {
    switch(allocator->type) {
    case WMEM_ALLOCATOR_SIMPLE: return NS_WMEM_SIMPLE;
    case WMEM_ALLOCATOR_STRICT: return NS_WMEM_STRICT;
    case WMEM_ALLOCATOR_BLOCK: return NS_WMEM_BLOCK;
    case WMEM_ALLOCATOR_BLOCK_FAST: return NS_WMEM_BLOCK_FAST;
    default: abort();
    }
}
'''
    e.replace(f,'\nvoid *\nwmem_alloc(wmem_allocator_t *allocator, const size_t size)',
              mapping+'\nvoid *\nwmem_alloc(wmem_allocator_t *allocator, const size_t size)')
    e.replace(f,'    return allocator->walloc(allocator->private_data, size);',
              '    void *p=allocator->walloc(allocator->private_data,size);\n'
              '    ns_alloc(ns_wmem_family(allocator),allocator,p,size);\n    return p;')
    e.replace(f,'    allocator->wfree(allocator->private_data, ptr);',
              '    if(allocator->type==WMEM_ALLOCATOR_BLOCK_FAST)\n'
              '        ns_noop_free(ns_wmem_family(allocator));\n'
              '    else ns_free(ns_wmem_family(allocator),allocator,ptr);\n'
              '    allocator->wfree(allocator->private_data, ptr);')
    e.replace(f,'    return allocator->wrealloc(allocator->private_data, ptr, size);',
              '    void *q=allocator->wrealloc(allocator->private_data,ptr,size);\n'
              '    ns_resize(ns_wmem_family(allocator),allocator,ptr,q,size,\n'
              '              allocator->type!=WMEM_ALLOCATOR_BLOCK_FAST);\n    return q;')
    e.replace(f,'    allocator->free_all(allocator->private_data);',
              '    if(final) ns_destroy(ns_wmem_family(allocator),allocator);\n'
              '    else ns_bulk(ns_wmem_family(allocator),allocator);\n'
              '    allocator->free_all(allocator->private_data);')

def memcached(e):
    f='slabs.c'
    e.include(f)
    e.after(f,'#include "memcached.h"\n','static int ns_carving;\n')
    e.after(f,'    if (ret) {\n        MEMCACHED_SLABS_ALLOCATE(id, p->size, ret);\n',
            '        ns_alloc(NS_MC_SLAB,&slabclass[id],ret,p->size);\n')
    e.after(f,'    if ((it->it_flags & ITEM_CHUNKED) == 0) {\n',
            '        if(!ns_carving) ns_free(NS_MC_SLAB,p,it);\n')
    e.replace(f,'    // return the header object.\n',
              '    ns_free(NS_MC_SLAB,p,it);\n    // return the header object.\n')
    e.after(f,'        p = &slabclass[chunk->slabs_clsid];\n        next_chunk = chunk->next;\n',
            '        ns_free(NS_MC_SLAB,p,chunk);\n')
    e.replace(f,'    for (x = 0; x < p->perslab; x++) {\n        do_slabs_free(ptr, id);\n        ptr += p->size;\n    }\n',
              '    ns_carving=1;\n    for (x = 0; x < p->perslab; x++) {\n'
              '        do_slabs_free(ptr, id);\n        ptr += p->size;\n    }\n    ns_carving=0;\n')
    f='cache.c'
    e.include(f)
    e.replace(f,'    return object;\n}\n\nvoid cache_free(cache_t *cache, void *ptr) {',
              '    if(object) {\n        size_t size=cache->bufsize;\n#ifndef NDEBUG\n'
              '        size-=2*sizeof(redzone_pattern);\n#endif\n'
              '        ns_alloc(NS_MC_CACHE,cache,object,size);\n    }\n'
              '    return object;\n}\n\nvoid cache_free(cache_t *cache, void *ptr) {')
    e.after(f,'void do_cache_free(cache_t *cache, void *ptr) {\n',
            '    ns_free(NS_MC_CACHE,cache,ptr);\n')
    e.after(f,'void cache_destroy(cache_t *cache) {\n',
            '    ns_destroy(NS_MC_CACHE,cache);\n')

if __name__ == '__main__':
    if sys.argv[1] == '--check':
        _, name, actual, original, variant = sys.argv[1:]
        edits = Edits(Path(original))
        globals()[name](edits)
        for filename, observed in edits.files.items():
            wanted = observed if variant == 'observed' else (Path(original)/filename).read_text()
            if (Path(actual)/filename).read_text() != wanted:
                raise ValueError(f'{filename}: stale or missing hooks in {variant} build')
        print('PASS exact source hooks',name,variant)
    else:
        name, root = sys.argv[1:]
        edits = Edits(Path(root))
        globals()[name](edits)
        edits.save()
