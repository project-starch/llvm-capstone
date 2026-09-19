# Whisper / ggml defects — a negative result at the 1.9.4 pin

*Whether ggml's allocators have consumer-side temporal defects usable as corpus
specimens, surveyed the way the CPython one was. Assembled 2026-09-18, deepened
2026-09-19 after the first pass was found to rest on a weak instrument.
**The answer is no, and three separate lines of evidence agree.***

## Why this was redone

The first pass grepped commit **subjects** for temporal words and found one hit,
a threadpool change. That is a fair instrument for CPython, whose fixes are
titled *"Fix use-after-free in X"* by convention. It is a weak one here: ggml's
commits are titled `ggml : fix ...` with the detail in the body or nowhere. **A
clean zero from a weak instrument is the failure mode this project keeps paying
for**, so the question was re-asked from the files a commit touches rather than
from how its author described it.

The answer did not change. The confidence behind it did.

The three scripts are in `whisper-ggml-survey/`, committed so a negative
result can be re-derived rather than trusted.

## ggml stacks three allocators; the port covers one

| file | allocator | commits | ported |
|---|---|---|---|
| `ggml/src/ggml.c` | `ggml_context`, a bump arena | 622 | **yes** |
| `ggml/src/ggml-alloc.c` | `ggml_gallocr`, the graph allocator | 69 | no |
| `ggml/src/ggml-backend.cpp` | the backend buffer scheduler | 128 | no |

All three histories were read, not sampled — they are small enough.

## 1. The allocator files: every lifetime-word commit, and what each is

Widening the net from "use-after-free" to any word touching memory lifetime
(`freed|dangling|stale|leak|lifetime|premature|realloc|reuse|invalid read`):

| file | hits / commits | what they are |
|---|---|---|
| `ggml.c` | 4 / 622 | a gguf **leak** fix, a metal leak fix, a task-count refactor, a graph-reuse feature. No temporal defect. |
| `ggml-alloc.c` | 5 / 69 | all **allocator-internal**: free-block bookkeeping, chunk preference, a leak when reusing a larger tensor. |
| `ggml-backend.cpp` | 5 / 128 | four scheduling/realloc features, and one real use-after-free — read below. |

The one real use-after-free, `eda7e1d4f`, *"fix possible buffer use after free in
sched reserve"*, is three lines:

```c
     ggml_backend_sched_split_graph(sched, measure_graph);
+    ggml_backend_sched_synchronize(sched);
     if (!ggml_gallocr_reserve_n(sched->galloc, ...)) {
         return false;
     }
     ggml_backend_sched_reset(sched);
-    ggml_backend_sched_synchronize(sched);
```

Pending async work had to finish before buffers were reallocated. It is
**inside the scheduler** and it is an **asynchrony** bug — excluded on both of
the axes the CPython triage uses, and in a layer the port does not cover anyway.

## 2. The consumer side, which is what a corpus case actually needs

A corpus case is a defect in a **user** of the allocator. For CPython that meant
`Modules/` code holding a borrowed pointer across a free. For ggml it means
whisper.cpp or llama.cpp code holding a `ggml_tensor *` or a `tensor->data`
across a `ggml_free`, a graph reset, or a buffer reallocation.

| consumer source | commits | mentioning any lifetime **or crash** word |
|---|---|---|
| `src/whisper.cpp` | 346 | **1** — "fixed crash in GPU device selection on multi-GPU systems" |
| `examples/` | 628 | 3 — a missing-argument segfault, a vocab-length crash, and a stdin dangling pointer on ordinary `malloc` memory |

Not one of those four is a temporal defect on ggml memory. In the entire history
of whisper.cpp's own sources, **there is no use-after-free of allocator memory to
reproduce.**

## 3. Why — read from the source, after a false positive

The structural argument is that a bump arena freed whole leaves no window for a
stale per-object pointer. Checking it against the pinned tag turned up something
that looked, in a thirteen-line window, like the opposite:

```c
cache.k = ggml_new_tensor_1d(ctx, wtype, n_elements);   // 1007
cache.v = ggml_new_tensor_1d(ctx, wtype, n_elements);
cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
ggml_backend_buffer_clear(cache.buffer, 0);
ggml_free(ctx);                                          // 1018
return true;                        // cache.k, cache.v still live!
```

Tensors built from `ctx`, stored in a long-lived struct, and `ggml_free(ctx)`
called before returning. That is precisely the corpus shape — if the window were
the whole story. It is not. Twenty lines earlier:

```c
cache.ctx_buf.resize(2*ggml_tensor_overhead());          //  986
struct ggml_init_params params = {
    /*.mem_size   =*/ cache.ctx_buf.size(),
    /*.mem_buffer =*/ cache.ctx_buf.data(),              // caller-owned
    /*.no_alloc   =*/ true,
};
```

The arena is a `std::vector` **owned by the long-lived `cache` struct**, not by
ggml. And `ggml_free` frees the buffer only when it owns it:

```c
void ggml_free(struct ggml_context * ctx) {          /* ggml.c:1663 */
    if (ctx->mem_buffer_owned) {
        ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }
    GGML_FREE(ctx);
}
```

So `ggml_free(ctx)` here releases a context *handle* and nothing else. The
tensors outlive it because the consumer owns the memory they live in. The same
idiom appears at the VAD state (`vctx->h_state`, `vctx->c_state`, line 4807).

**This is a sharper version of the structural claim than the first pass made.**
It is not merely that the arena is freed wholesale — it is that in the idiom
whisper.cpp actually uses, *the consumer owns the arena*, so ggml never ends a
lifetime the consumer is still relying on. There is no per-tensor free
(`ggml_free` is the only `ggml_free*` symbol the public header declares), and the
arena's lifetime is a C++ object's lifetime.

## The limitation this result has, stated plainly

**This method finds defects that were FIXED.** It measures upstream's fix
history, so it also measures how much security attention a project gets. CPython
is continuously fuzzed, has a security response team, and had 44 temporal fixes
land in one release series. whisper.cpp has neither, and a project nobody fuzzes
produces few use-after-free fixes whether or not it has use-after-frees.

So the honest claim is **not** "ggml has no temporal defects". It is:

- ggml's ported allocator has **no structural window** for the corpus's defect
  class, and that part is proven from the source rather than from history;
- and no consumer-side temporal defect on ggml memory has ever been **found and
  fixed** upstream, across 974 consumer commits and 819 allocator commits.

The first is strong. The second is an absence of evidence, and is reported as
one.

## Consequence for the corpus

No `bug-corpora/whisper/`. The corpus's temporal specimens come from PostgreSQL's
AllocSet/Generation/Slab and CPython's pymalloc — both of which take individual
chunks back, which is the operation the defect class requires.

Recorded rather than dropped, because the absence is itself a finding: it
predicts the same empty result for any arena-only allocator, including CPython's
own `PyArena` and PostgreSQL's Bump, and says that effort belongs on the layers
that free per chunk. If a ggml layer is ever worth porting for this purpose it is
`ggml-alloc.c`'s `ggml_gallocr`, which *does* free and reuse per tensor — but the
defects there are in the allocator, not in its consumers, so it would demonstrate
a different claim than this corpus makes.
