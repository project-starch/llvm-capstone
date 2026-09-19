#!/usr/bin/env python3
"""The one remaining path to a non-zero whisper corpus.

ggml_context has no per-object free -- settled. But ggml-alloc.c's ggml_gallocr
DOES free and reuse per tensor (ggml_dyn_tallocr_free_tensor,
ggml_gallocr_free_node), and ggml_gallocr_reserve can MOVE a tensor's storage.

So a consumer-side case could still exist in the shape:
    take tensor->data (or a ggml_tensor*), call something that reserves or
    reallocates the graph, then use the old pointer.

The upstream history hints the hazard is real -- GGML_SCHED_NO_REALLOC exists as
a debug option, and there are fixes titled "fix graph reallocation with multiple
chunks" and "graph_reused". Whether WHISPER.CPP is exposed is a different
question, and it is answered by reading whisper.cpp, not ggml.
"""
import re
import subprocess

W = "/tmp/claude-1000/corpus-research/whispercpp.git"
TAG = "v1.9.4"


def git(*a):
    return subprocess.run(["git", "-C", W, *a], capture_output=True,
                          text=True).stdout


src = git("show", f"{TAG}:src/whisper.cpp").split("\n")

print("=" * 76)
print("Does whisper.cpp call the graph allocator / scheduler reserve at all?")
print("=" * 76)
API = ["ggml_gallocr_reserve", "ggml_gallocr_alloc_graph", "ggml_gallocr_new",
       "ggml_backend_sched_reserve", "ggml_backend_sched_alloc_graph",
       "ggml_backend_sched_graph_compute", "ggml_backend_alloc_ctx_tensors"]
for a in API:
    hits = [(i + 1, l.strip()) for i, l in enumerate(src) if a in l]
    print(f"\n  {a}: {len(hits)}")
    for ln, t in hits[:6]:
        print(f"     {ln:>5}  {t[:88]}")

print()
print("=" * 76)
print("Does whisper.cpp ever cache a tensor->data pointer?")
print("=" * 76)
DATA = re.compile(r"->data\b")
hits = [(i + 1, l.strip()) for i, l in enumerate(src) if DATA.search(l)]
print(f"  {len(hits)} lines mention ->data; the ones that ASSIGN it to something:")
for ln, t in hits:
    if re.search(r"=\s*[\w.\->\[\]]*->data\b", t) and not t.startswith("//"):
        print(f"     {ln:>5}  {t[:88]}")
