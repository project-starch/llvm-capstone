# The three scripts behind the ggml negative result

Every count in `../whisper-ggml-defects.md` comes from one of these. They are
committed so a negative result can be re-derived rather than trusted — a "no
defects found" is exactly the kind of claim that should not rest on someone's
word.

    layers.py         the three allocator layers and each one's full history
    consumers.py      whisper.cpp's OWN sources, which is where a corpus case
                      would have to come from
    read-kv-cache.py  the source read that explains why there is no window
    security-net.py   a WIDE memory-safety net over both repos, classified
                      spatial vs temporal -- 369 hits in llama.cpp where the
                      temporal net found 33
    trace-cve-2025-14569.py
                      why a keyword net could not have found a real CVE: the
                      fix is titled "fix memory leak" 

## Inputs

Bare clones of both repositories; ggml is developed in llama.cpp and synced into
whisper.cpp, so both histories matter.

    git clone --bare https://github.com/ggml-org/llama.cpp   $GGML_REPO
    git clone --bare https://github.com/ggml-org/whisper.cpp $WHISPER_REPO
    GGML_REPO=... WHISPER_REPO=... python3 layers.py

The defaults are where the 2026-09-19 run had them.

## Why there are three, and not one grep

The first version of this survey grepped commit **subjects** for temporal words
and reported one hit. CPython's fixes are titled "Fix use-after-free in X" by
convention, so a subject grep is a fair instrument there. ggml's are titled
`ggml : fix ...`, so it is not.

`layers.py` therefore selects by the **files a commit touches** and then widens
the keyword net, so a fix can be found regardless of how it was described.
`consumers.py` asks the question that actually decides whether a corpus case
exists — is there a defect in a *user* of the allocator — which no amount of
searching `ggml/` can answer. `read-kv-cache.py` exists because a thirteen-line
window made the pinned source look like it held tensors past `ggml_free(ctx)`; it
prints the whole function, which shows the arena is caller-owned.

That last one is the pattern worth copying: **a positive finding from a narrowed
view needs the same suspicion as a clean zero.**

## The one that got away, and what it cost

`security-net.py` and `trace-cve-2025-14569.py` were added on 2026-09-19, after
the survey had twice been described as equivalent to the CPython one. It was not:
the CPython survey is anchored on upstream's issue tracker, and neither ggml pass
had looked at a CVE database, a security advisory, or an issue tracker at all.

What that missed: **CVE-2025-14569**, a use-after-free in whisper.cpp's own
`read_audio_data`. Every commit touching that file has a subject with no temporal
word in it; the fix says *"fix memory leak"*. No keyword net over commit subjects
finds that, however wide.

It turned out not to be a corpus case — the buffer is a `std::vector<float>`, so
a malloc-level tool reports it — but that was luck, not method. **Search the
advisory stream, not only the commit stream.**

## The version sweep (section 5)

    api-over-time.py          every public ggml free-like symbol, release by
                              release, across all 39 whisper.cpp tags
    free-tensor-v143.py       the one release with a real public per-tensor free:
                              what it did, and who called it
    free-tensor-backends.py   what every backend set that hook to (all NULL)
    patched-file-location.py  where the file our port patches lives per release,
                              which bounds how far back the port can reach

`api-over-time.py` carries a caveat worth repeating: it selects headers by
**basename**, which also matches the stale vendored copy under
`bindings/ruby/ext/`. That copy kept a declaration two releases after the real
header dropped it, and reading the result without checking which file each hit
came from over-stated the API's lifetime by two releases. Check the path, not the
filename.
