#!/usr/bin/env python3
"""Instrument pinned FFmpeg in a scratch source tree; fail on unknown context."""

import pathlib
import shutil
import sys

here = pathlib.Path(__file__).resolve().parent
src = pathlib.Path(sys.argv[1]) / "libavutil"


def edit(name, replacements):
    p = src / name
    s = p.read_text()
    for old, new, count in replacements:
        if s.count(old) != count:
            raise SystemExit(
                f"{name}: expected {count} occurrences of {old!r}, got {s.count(old)}"
            )
        s = s.replace(old, new)
    p.write_text(s)


edit(
    "buffer.c",
    [
        ('#include "thread.h"', '#include "thread.h"\n#include "trace.h"', 1),
        (
            "    AVBufferPool *pool = av_mallocz(sizeof(*pool));",
            "    FF2_GUARD;\n    AVBufferPool *pool = av_mallocz(sizeof(*pool));",
            2,
        ),
        (
            "    pool->alloc2    = alloc;",
            "    if (alloc || pool_free) ff2_fail(125);\n    pool->alloc2    = alloc;",
            1,
        ),
        (
            "    pool->alloc    = alloc ? alloc : av_buffer_alloc;",
            "    if (alloc && alloc != av_buffer_alloc && alloc != av_buffer_allocz) ff2_fail(126);\n    pool->alloc    = alloc ? alloc : av_buffer_alloc;",
            1,
        ),
        (
            "    return pool;\n}",
            "    uint64_t call = ff2_begin(FF2_CREATE, FF2_BUFFER, NULL, size, pool->alloc == av_buffer_allocz, NULL);\n    ff2_end(call, pool);\n    return pool;\n}",
            2,
        ),
        (
            "        buf->free(buf->opaque, buf->data);",
            "        ff2_drop(FF2_BUFFER, pool, buf->data);\n        buf->free(buf->opaque, buf->data);",
            1,
        ),
        (
            "void av_buffer_pool_uninit(AVBufferPool **ppool)\n{",
            "void av_buffer_pool_uninit(AVBufferPool **ppool)\n{\n    FF2_GUARD;",
            1,
        ),
        (
            "    pool   = *ppool;",
            "    pool   = *ppool;\n    uint64_t call = ff2_begin(FF2_CLOSE, FF2_BUFFER, pool, 0, 0, NULL);",
            1,
        ),
        (
            "static void pool_release_buffer(void *opaque, uint8_t *data)\n{",
            "static void pool_release_buffer(void *opaque, uint8_t *data)\n{\n    FF2_GUARD;",
            1,
        ),
        (
            "    AVBufferPool *pool = buf->pool;",
            "    AVBufferPool *pool = buf->pool;\n    uint64_t call = ff2_begin(FF2_RETURN, FF2_BUFFER, pool, 0, 0, data);",
            1,
        ),
        (
            "        buffer_pool_free(pool);\n}",
            "        buffer_pool_free(pool);\n    ff2_end(call, NULL);\n}",
            2,
        ),
        (
            "    buf->data   = ret->buffer->data;",
            "    ff2_new(FF2_BUFFER, pool, ret->data);\n    buf->data   = ret->buffer->data;",
            1,
        ),
        (
            "AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)\n{",
            "AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)\n{\n    FF2_GUARD;\n    uint64_t call = ff2_begin(FF2_GET, FF2_BUFFER, pool, 0, 0, NULL);",
            1,
        ),
        (
            "    return ret;\n}\n\nvoid *av_buffer_pool_buffer_get_opaque",
            "    ff2_end(call, ret ? ret->data : NULL);\n    return ret;\n}\n\nvoid *av_buffer_pool_buffer_get_opaque",
            1,
        ),
    ],
)

edit(
    "refstruct.c",
    [
        ('#include "thread.h"', '#include "thread.h"\n#include "trace.h"', 1),
        (
            "void av_refstruct_unref(void *objp)",
            "static uint64_t ff2_unref_begin(RefCount *ref, void *obj);\n\nvoid av_refstruct_unref(void *objp)",
            1,
        ),
        (
            "void av_refstruct_unref(void *objp)\n{",
            "void av_refstruct_unref(void *objp)\n{\n    FF2_GUARD;",
            1,
        ),
        (
            "        if (ref->free_cb)\n",
            "        uint64_t call = ff2_unref_begin(ref, obj);\n        if (ref->free_cb)\n",
            1,
        ),
        (
            "        ref->free(ref);",
            "        ref->free(ref);\n        if (call) ff2_end(call, NULL);",
            1,
        ),
        (
            "static void pool_free(AVRefStructPool *pool)",
            """static void pool_return_entry(void *ref);
static void pool_unref(void *ref);
static uint64_t ff2_unref_begin(RefCount *ref, void *obj)
{
    if (ref->free == pool_return_entry)
        return ff2_begin(FF2_RETURN, FF2_REFSTRUCT, ref->opaque.nc, 0, 0, obj);
    if (ref->free == pool_unref)
        return ff2_begin(FF2_CLOSE, FF2_REFSTRUCT, obj, 0, 0, NULL);
    return 0;
}

static void pool_free(AVRefStructPool *pool)""",
            1,
        ),
        (
            "    if (pool->free_cb)\n        pool->free_cb(pool->opaque);",
            """    if (pool->free_cb) {
        uint64_t call = ff2_callback(FF2_REFSTRUCT, pool, NULL, FF2_FREE_POOL_CB);
        pool->free_cb(pool->opaque);
        ff2_end(call, NULL);
    }""",
            1,
        ),
        (
            "    if (pool->free_entry_cb)\n        pool->free_entry_cb(pool->opaque, get_userdata(ref));\n    av_free(ref);",
            """    if (pool->free_entry_cb) {
        uint64_t call = ff2_callback(FF2_REFSTRUCT, pool, get_userdata(ref), FF2_FREE_ENTRY_CB);
        pool->free_entry_cb(pool->opaque, get_userdata(ref));
        ff2_end(call, NULL);
    }
    ff2_drop(FF2_REFSTRUCT, pool, get_userdata(ref));
    av_free(ref);""",
            1,
        ),
        (
            "    pool->reset_cb(pool->opaque, entry);",
            "    uint64_t call = ff2_callback(FF2_REFSTRUCT, pool, entry, FF2_RESET_CB);\n    pool->reset_cb(pool->opaque, entry);\n    ff2_end(call, NULL);",
            1,
        ),
        (
            "static int refstruct_pool_get_ext(void *datap, AVRefStructPool *pool)\n{",
            "static int refstruct_pool_get_ext(void *datap, AVRefStructPool *pool)\n{\n    FF2_GUARD;\n    uint64_t call = ff2_begin(FF2_GET, FF2_REFSTRUCT, pool, 0, 0, NULL);",
            1,
        ),
        (
            "        ref->free = pool_return_entry;",
            "        ref->free = pool_return_entry;\n        ff2_new(FF2_REFSTRUCT, pool, ret);",
            1,
        ),
        (
            "            int err = pool->init_cb(pool->opaque, ret);",
            "            uint64_t cb = ff2_callback(FF2_REFSTRUCT, pool, ret, FF2_INIT_CB);\n            int err = pool->init_cb(pool->opaque, ret);\n            ff2_end(cb, NULL);\n            if (err < 0) ff2_fail(127);",
            1,
        ),
        (
            "    memcpy(datap, &ret, sizeof(ret));",
            "    memcpy(datap, &ret, sizeof(ret));\n    ff2_end(call, ret);",
            1,
        ),
        (
            "    AVRefStructPool *pool = av_refstruct_alloc_ext(sizeof(*pool), 0, NULL,",
            "    FF2_GUARD;\n    AVRefStructPool *pool = av_refstruct_alloc_ext(sizeof(*pool), 0, NULL,",
            1,
        ),
        (
            "    return pool;\n}",
            """    uint64_t trace_flags = flags;
    if (init_cb) trace_flags |= FF2_HAS_INIT;
    if (reset_cb) trace_flags |= FF2_HAS_RESET;
    if (free_entry_cb) trace_flags |= FF2_HAS_FREE_ENTRY;
    if (free_cb) trace_flags |= FF2_HAS_FREE_POOL;
    uint64_t call = ff2_begin(FF2_CREATE, FF2_REFSTRUCT, NULL, size, trace_flags, NULL);
    ff2_end(call, pool);
    return pool;
}""",
            1,
        ),
    ],
)
edit(
    "Makefile", [("OBJS = adler32.o", "OBJS = ff2_record.o ff2_observe.o adler32.o", 1)]
)
for original, name in [
    ("../../shared/trace.h", "trace.h"),
    ("record-pool-events.c", "ff2_record.c"),
    ("../../shared/observe-pool-events.c", "ff2_observe.c"),
]:
    shutil.copy2(here / original, src / name)
