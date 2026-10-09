# mruby GC slots: planned, no cases yet

The boundary is mruby's GC slots, `mrb_heap_page`. The declaration is the
`gc-slot-repros` entry in [../corpus.json](../corpus.json), which carries the
pin, the status and why nothing here is live in the ported release.

**This file is why the directory exists.** The group used to hold its own
`corpus.json` and nothing else; with the declaration moved into the program's
file the directory would have been empty, and git does not track an empty
directory -- so a fresh checkout would lose it, the checker would report
`group 'gc-slot-repros' has no directory`, and `xlang/repro`'s `related` link
would dangle. Both happened, and both were caught by running the gate against a
checkout rather than against the working tree it was prepared in.

The intended first cases are the six xlang rows that share the realloc-vmstack
shape -- 4, 5, 8, 10, 13 and 15 -- re-run inside the complete interpreter
rather than as distilled C shims.
