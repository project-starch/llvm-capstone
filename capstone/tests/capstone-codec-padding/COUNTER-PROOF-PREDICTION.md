# -capstone-pad-globals: the counter proof, predicted before any image exists (2026-10-20 13:45)

Program: memcached plain-temporal case 00 + the corpus controls, the nine-arm runner (run-virtual-cases.py), on
qkit7-wc9-nr (window counter, 16384 sites per kind; nf2 module; norandmaps). Two images, one variable:
- OFF: SDK t3-freeze-nb-nomovc (clang a5be1a4d, no pad flag);
- ON:  the same SDK recipe rebuilt with clang llvm-padg (01b62884c0c7 + the pass) and CAPSTONE_VIRTUAL_PAD_GLOBALS=ON,
  whose OFF build must equal t3-freeze-nb-nomovc in every ALLOC section (the identity control).

Predicted readings (result-inexact = SHRINK/SHRINKTO/SPLIT results no cursor represents; classes by
shrink-source.py over the image's own disassembly, sites accepted only at a SHRINK):
1. OFF: result-inexact sites in heap.c that form `index_heads` (64 KiB; first value [X, X + 0x10000) with X not a
   multiple of 128): __capstone_malloc_find, __capstone_malloc_slot_key, _slot_take, _slot_release,
   _prepare_free, cap_malloc_validate and malloc's inlined find, each n > 0. Source class: global.
2. ON: none of those sites; index_heads's capability base is a multiple of 128; the result-inexact events whose
   source class is `global` total 0. Any global residue names a global the pass skipped (section, TLS, < 4 KiB is
   exact by construction), and is a finding about the pass, not noise.
3. ON vs OFF verdicts: identical cells (padding moves no object a case's defect touches out of its bounds; a
   different verdict is a finding).
4. Not predicted to change: escape-inexact and op-inexact firmware counts (32 op-inexact are the kit monitor's).
