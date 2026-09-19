# The three scripts behind the ggml negative result

Every count in `../whisper-ggml-defects.md` comes from one of these. They are
committed so a negative result can be re-derived rather than trusted — a "no
defects found" is exactly the kind of claim that should not rest on someone's
word.

    layers.py         the three allocator layers and each one's full history
    consumers.py      whisper.cpp's OWN sources, which is where a corpus case
                      would have to come from
    read-kv-cache.py  the source read that explains why there is no window

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
