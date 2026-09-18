# Whisper / ggml defects — a negative result at the 1.9.4 pin

*Whether the ggml context allocator has consumer-side temporal defects usable as
corpus specimens, searched the same way as PostgreSQL and CPython. Assembled
2026-09-18 against `ports/whisper/ggml-context/upstream.json` = whisper.cpp
1.9.4. **The answer is no, and the reason is structural.***

## What the port covers

All three patches touch exactly one file:

```
$ grep -h '^+++ ' ports/whisper/ggml-context/patches/*.patch | sed 's#.*[ab]/##' | sort -u
ggml/src/ggml.c
```

So the surface under test is `ggml_context` — a bump arena carved out of one
buffer, handed out by `ggml_new_object` / `ggml_new_tensor`, and released
wholesale by `ggml_free`. That is the same shape as CPython's `PyArena` and
PostgreSQL's Bump context.

## Two histories, one answer

`ggml.c` is vendored: it is developed in llama.cpp and synced into whisper.cpp.
Both histories were searched, on commit subjects, for
`use-after-free|use after free|double free|double-free|dangling|uaf`:

| repo | commits | temporal hits | of those, touching `ggml/src/ggml.c` |
|---|---|---|---|
| llama.cpp | 11,041 | 33 | **1** |
| whisper.cpp | 5,213 | 19 | **1** |

Both point at the same commit — `threadpool : skip polling for unused threads`
(llama/9461), a scheduling change that happens to use the word, not a temporal
defect.

A wider net over `ggml.c`'s own history (`free|leak|memory corrupt|invalid
read|heap`, any subject) returns **11** commits. None is a consumer-side
temporal defect. The nearest misses:

- `b9e02e818 ggml : fix memory leaks when loading invalid gguf files` — a leak,
  the opposite failure: memory not freed, rather than used after being freed.
- `24ad19d0 ggml : fix possible buffer use after free in sched reserve` — a real
  use-after-free, but in `ggml-backend.cpp`, the backend scheduler, which the
  port does not cover.
- `76684141 ruby : fix dangling pointers … on parallel transcription` and
  `454b91de main : fix dangling pointer when using stdin` — real temporal
  defects in whisper.cpp, but on ordinary `malloc`/`new` memory in the Ruby
  binding and the CLI. A malloc-level tool sees those, so they are not blindspot
  cases even though they are consumer-side.

## Why this is structural, not a gap in the search

A bump arena has almost no per-object free to get wrong. Checked against the
pinned tag rather than assumed: `ggml_free` is the **only** `ggml_free*` symbol
the public header declares (`ggml/include/ggml.h:814` at `v1.9.4`, and nothing
else in that file matches `ggml_free[a-z_]*`), and its body frees the arena
buffer and the context struct, nothing smaller:

```c
void ggml_free(struct ggml_context * ctx) {          /* ggml.c:1663 */
    if (ctx == NULL) { return; }
    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }
    GGML_FREE(ctx);
}
```

There is no per-tensor free, so a consumer cannot end one object's lifetime
early and keep using it. The defect class the corpus is built
from — *the consumer frees a chunk back to the nested allocator and then reads
through a pointer it kept* — requires an allocator that takes individual chunks
back. ggml does not have that operation.

This matches the port's own framing: the ggml arm demonstrates **whole-arena
revocation**, not per-chunk temporal safety. It is the right arm for the
Sublet-covers-bump-arenas claim and the wrong place to look for specimens.

## Consequence for the corpus

No `bug-corpora/whisper/` directory. The corpus's temporal specimens come from
PostgreSQL's AllocSet/Generation/Slab (`bug-corpora/postgres/mmgr-repros/`, 8
cases) and CPython's pymalloc (`bug-corpora/cpython/pymalloc-repros/`, 21
reachable — see `cpython-pymalloc-defects.md`), both of which free per chunk.

Recorded rather than dropped, because the absence is itself the finding: it
predicts that any future port of an arena-only allocator — CPython's `PyArena`,
PostgreSQL's Bump — will yield the same empty result, and that effort is better
spent on the layers that hand chunks back.
