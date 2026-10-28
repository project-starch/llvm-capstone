# CRASH-078: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-078",
 "set": "core-166",
 "trigger": "corpus/CRASH-078.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "array-diff-intersect-globals-symtable-alias-bucket-uaf",
 "c_file_line": "ext/standard/array.c:3273 (defect: *return_value = **args[0]; zval_copy_ctor no-ops for &EG(symbol_table)); crash frame ext/standard/array.c:171",
 "upstream_fix_commit": "96d755978cc8",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-stdext",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-078"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-078.php` from that zip, byte for byte (sha256
`26b692ecbf0ecd48aeabaaba8d44eab8ae6f53f64b365b806635c031e033356b`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `ext/standard/array.c:171` in
`array_key_compare`, stockalloc `ASAN:heap-use-after-free`, pristine `OK` (rc 0).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped (the stockalloc run): efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `96d755978cc8`, the directory prefix `96d755978cc8`.
