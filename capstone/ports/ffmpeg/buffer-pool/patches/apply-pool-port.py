#!/usr/bin/env python3
"""Apply the lifetime port to a previously instrumented scratch FFmpeg tree."""

import pathlib
import sys

src = pathlib.Path(sys.argv[1]) / "libavutil"


def edit(name, changes):
    p = src / name
    s = p.read_text()
    for old, new, count in changes:
        if s.count(old) != count:
            raise SystemExit(
                f"{name}: expected {count} occurrences of {old!r}, got {s.count(old)}"
            )
        s = s.replace(old, new)
    p.write_text(s)


edit(
    "buffer_internal.h",
    [
        (
            "typedef struct BufferPoolEntry {",
            "typedef struct BufferPoolEntry {\n    uintptr_t ff2_address;",
            1,
        )
    ],
)
edit(
    "buffer.c",
    [
        ("    av_free(data);", "    ff2_payload_free(data);", 1),
        ("    data = av_malloc(size);", "    data = ff2_payload_alloc(size);", 1),
        (
            "    ret = av_buffer_create(data, size, av_buffer_default_free, NULL, 0);\n    if (!ret)\n        av_freep(&data);",
            "    ret = av_buffer_create(data, size, av_buffer_default_free, NULL, 0);\n    if (!ret)\n        ff2_payload_free(data);",
            1,
        ),
        (
            "        ff2_drop(FF2_BUFFER, pool, buf->data);",
            "        buf->data = ff2_payload_issue(buf->ff2_address);\n        ff2_drop(FF2_BUFFER, pool, buf->data);",
            1,
        ),
        (
            "    uint64_t call = ff2_begin(FF2_RETURN, FF2_BUFFER, pool, 0, 0, data);",
            "    uint64_t call = ff2_begin(FF2_RETURN, FF2_BUFFER, pool, 0, 0, data);\n    ff2_payload_return(data);\n    buf->data = NULL;",
            1,
        ),
        (
            "    buf->data   = ret->buffer->data;",
            "    buf->ff2_address = (uintptr_t)ret->buffer->data;\n    buf->data   = ret->buffer->data;",
            1,
        ),
        (
            "    if (buf) {\n        memset(&buf->buffer",
            "    if (buf) {\n        buf->data = ff2_payload_issue(buf->ff2_address);\n        memset(&buf->buffer",
            1,
        ),
        (
            "            buf->buffer.flags_internal |= BUFFER_FLAG_NO_FREE;\n        }",
            "            buf->buffer.flags_internal |= BUFFER_FLAG_NO_FREE;\n        } else {\n            ff2_payload_return(buf->data);\n            buf->data = NULL;\n        }",
            1,
        ),
    ],
)
edit(
    "refstruct.c",
    [
        (
            "(RefCount*)((char*)obj - REFCOUNT_OFFSET)",
            "(RefCount*)ff2_ref_meta(obj)",
            1,
        ),
        (
            "(const RefCount*)((const char*)obj - REFCOUNT_OFFSET)",
            "(const RefCount*)ff2_ref_meta(obj)",
            1,
        ),
        (
            "    return (char*)buf + REFCOUNT_OFFSET;",
            "    return ff2_ref_data(buf);",
            1,
        ),
        ("    ref->free    = av_free;", "    ref->free    = ff2_ref_free;", 1),
        (
            "    buf = av_malloc(size + REFCOUNT_OFFSET);",
            "    buf = ff2_ref_alloc(size, sizeof(RefCount));",
            1,
        ),
        ("av_free(get_refcount(pool));", "ff2_ref_free(get_refcount(pool));", 2),
        ("av_free(ref);", "ff2_ref_free(ref);", 2),
        (
            "    AVRefStructPool *pool = ref->opaque.nc;",
            "    AVRefStructPool *pool = ref->opaque.nc;\n    ff2_ref_return(ref);",
            1,
        ),
        ("        ret = get_userdata(ref);", "        ret = ff2_ref_issue(ref);", 1),
    ],
)
