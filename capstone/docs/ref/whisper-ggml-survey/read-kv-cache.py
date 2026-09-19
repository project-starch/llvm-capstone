#!/usr/bin/env python3
"""Does whisper.cpp really keep ggml_tensor* past ggml_free of their context?

A 13-line window suggested it does. That is the shape a positive finding takes
when the window was chosen by the query, so read the WHOLE function: how ctx is
created, what no_alloc means here, and what ggml_backend_alloc_ctx_tensors does
with the tensors.
"""
import subprocess

REPO = "/tmp/claude-1000/corpus-research/whispercpp.git"


def show(ref, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"],
                          capture_output=True, text=True).stdout.split("\n")


src = show("v1.9.4", "src/whisper.cpp")

# Walk back from the free to the function's opening brace.
end = 1021
start = end
depth = 0
for i in range(end - 1, max(0, end - 120), -1):
    if src[i].startswith("static ") or src[i].startswith("bool ") or \
       (src[i] and not src[i][0].isspace() and "(" in src[i]):
        start = i + 1
        break
print("=" * 74)
print("the whole function that frees ctx at line 1018")
print("=" * 74)
for i in range(start - 1, end):
    print(f"{i + 1:>6}  {src[i][:96]}")

print()
print("=" * 74)
print("where cache.k / cache.v are READ afterwards")
print("=" * 74)
for i, line in enumerate(src):
    if ".k," in line or "->k," in line or "cache.k" in line or "kv.k" in line:
        print(f"{i + 1:>6}  {line[:96]}")
