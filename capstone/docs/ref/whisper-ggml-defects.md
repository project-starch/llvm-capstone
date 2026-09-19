# Whisper / ggml defects — a negative result at the 1.9.4 pin

*Whether ggml's allocators have consumer-side temporal defects usable as corpus
specimens. Assembled 2026-09-18, deepened 2026-09-19 twice: once after the first
pass was found to rest on a weak instrument, and again after the second was found
**not** to have searched CVEs at all. **The answer is no, and four separate lines
of evidence agree** -- but see section 4 for what the first three could not see.*

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

Not one of those four is a temporal defect on ggml memory. But this table is a
**keyword** search of commit subjects, and section 4 shows it misses a real
whisper.cpp use-after-free CVE whose fix is titled "fix memory leak". Read it as
what it is -- a lower bound from one instrument -- and take the conclusion from
the structural argument in section 3 and the advisory sweep in section 4.

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

## 4. Published CVEs — the half the first two passes never did

**RETRACTED 2026-09-19.** Both earlier passes claimed this survey was run "the way
the CPython one was". That was false in a specific way: the CPython survey is
anchored on upstream's issue tracker, every case carrying a `gh-NNNNN`. The ggml
passes searched **commit history only** and never looked at a CVE database, a
GitHub Security Advisory, or either project's issue tracker. The conclusion
survives; the claim that it rested on equivalent evidence did not.

### Why the earlier net could not have found them

A commit that fixes a reported vulnerability is rarely described as one. Of every
commit touching `examples/common-whisper.cpp` — the file holding a real
use-after-free CVE — **not one carries a temporal word in its subject or body.**
The fix is titled:

    47b9eb37  examples : fix memory leak in read_audio_data (#3810)

*Leak*: the opposite word. A net built from `use-after-free|dangling|double free`
misses it by construction, and so would a net built from this doc's earlier,
wider list. The lesson generalises past ggml: **search the advisory stream, not
only the commit stream.**

### Every published CVE, and why none is a corpus case

| CVE | what | where | why not a case |
|---|---|---|---|
| CVE-2025-14569 | **use-after-free** in `read_audio_data` | whisper.cpp `examples/common-whisper.cpp`, ≤1.8.2 | **fixed in 1.8.3, before our 1.9.4 pin.** And the memory is a `std::vector<float>` — ordinary C++ heap that ASan sees, not ggml allocator memory |
| CVE-2026-17512 | out-of-bounds read in `log_mel_spectrogram` | whisper.cpp `src/whisper.cpp` 1.8.4-58, fix pending | **spatial**, not temporal |
| CVE-2026-43622 | **double free** — `new_1batch()` mallocs, `free_1batch()` deletes | llama.cpp Android JNI, b1886–b7445 | allocator *mismatch* on malloc/new memory; not ggml, not whisper |
| CVE-2026-43631 | **use-after-free** of the `vocab` pointer, RCE | llama.cpp `llama-server` `--sleep-idle-seconds` | **concurrency** — worker threads racing the sleep transition; `llama_vocab`, not ggml memory |
| CVE-2026-43632 | same dangling `vocab`, other endpoints | llama.cpp `llama-server` | same |
| CVE-2026-70640 | **race-condition use-after-free** on `llama_context` | llama.cpp Android JNI, b1886–b7445 | **concurrency**, and llama.cpp's own object |

Four of the six are genuine temporal defects. **None is a temporal defect on
memory handed out by a ggml allocator**, which is what this corpus is made of.
They divide cleanly into three exclusions the CPython triage uses as well:
spatial rather than temporal; concurrency rather than single-threaded; and
ordinary `malloc`/`new`/`std::vector` memory rather than nested-allocator memory.

The whisper.cpp one, CVE-2025-14569, is the closest miss and worth stating
precisely: a real use-after-free, in whisper.cpp itself, in a consumer path — and
still not a case, because the buffer is a `std::vector` and a malloc-level tool
reports it. That is the distinction the whole corpus rests on.

### Context this adds

Ten vulnerabilities were reported to llama.cpp by one research group between July
2025 and June 2026 through GitHub Security Advisories and MITRE. All ten
advisories were closed by the maintainer without fixes or CVE assignment;
VulnCheck allocated the CVE IDs afterwards, and five of the ten remained
unpatched as of June 2026.

That materially changes this document's earlier caveat. The first version said a
project nobody fuzzes produces few use-after-free *fixes* whether or not it has
use-after-frees. The stronger and now-evidenced form: **llama.cpp has had
reported, CVE-assigned memory-safety vulnerabilities that were never fixed at
all**, so counting fixes in its history under-reports its defects by an amount
this survey cannot bound. A history-based survey measures a project's *response*,
not its *code*.

For this corpus the practical consequence is unchanged — none of the reported
defects is in a ggml allocator's consumer — and the structural argument in
section 3 does not depend on the history at all, which is why it carries the
conclusion and the counting does not.

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
