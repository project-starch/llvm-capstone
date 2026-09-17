#!/usr/bin/env python3
"""Patch a scratch FFmpeg source tree, refusing any unexpected context."""
import pathlib
import shutil
import sys

here = pathlib.Path(__file__).resolve().parent
src = pathlib.Path(sys.argv[1])
path = src / "libavutil/buffer.c"
text = path.read_text()

def replace(old, new, count=1):
    global text
    if text.count(old) != count:
        raise SystemExit(f"instrumentation context mismatch: {old!r}")
    text = text.replace(old, new)

replace('#include "thread.h"', '#include "thread.h"\n#include "record.inc"')
replace('    AVBufferPool *pool = av_mallocz(sizeof(*pool));',
        '    FF_RECORD_GUARD;\n    AVBufferPool *pool = av_mallocz(sizeof(*pool));', 2)
replace('    return pool;\n}', '    ff_record_create(pool);\n    return pool;\n}', 2)
replace('        buf->free(buf->opaque, buf->data);',
        '        ff_record_drop(pool, buf->data);\n        buf->free(buf->opaque, buf->data);')
replace('void av_buffer_pool_uninit(AVBufferPool **ppool)\n{',
        'void av_buffer_pool_uninit(AVBufferPool **ppool)\n{\n    FF_RECORD_GUARD;')
replace('    pool   = *ppool;',
        '    pool   = *ppool;\n    struct ff_event recorded = ff_record_close(pool);')
replace('static void pool_release_buffer(void *opaque, uint8_t *data)\n{',
        'static void pool_release_buffer(void *opaque, uint8_t *data)\n{\n    FF_RECORD_GUARD;')
replace('    AVBufferPool *pool = buf->pool;',
        '    AVBufferPool *pool = buf->pool;\n    struct ff_event recorded = ff_record_return(pool, data);')
replace('        buffer_pool_free(pool);\n}',
        '        buffer_pool_free(pool);\n    ff_record_emit(recorded);\n}', 2)
replace('AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)\n{',
        'AVBufferRef *av_buffer_pool_get(AVBufferPool *pool)\n{\n    FF_RECORD_GUARD;\n    int recorded_reuse = pool->pool != NULL;')
replace('    return ret;\n}\n\nvoid *av_buffer_pool_buffer_get_opaque',
        '    ff_record_get(pool, ret, recorded_reuse);\n    return ret;\n}\n\nvoid *av_buffer_pool_buffer_get_opaque')
path.write_text(text)
for name in ('record.inc', 'replay-format.h'):
    shutil.copy2(here / name, src / 'libavutil' / name)
