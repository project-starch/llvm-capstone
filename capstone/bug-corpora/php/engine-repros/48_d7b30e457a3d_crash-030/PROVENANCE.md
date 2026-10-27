# CRASH-030: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-030",
 "set": "core-166",
 "trigger": "corpus/CRASH-030.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "call-method-on-destroyed-object-store-bucket",
 "c_file_line": "Zend/zend_execute_API.c:617-625 (missing `!object_buckets[handle].valid` guard; faulting deref `calling_scope = Z_OBJCE_PP(fci->object_pp);` at :625, resolving to Zend/zend_object_handlers.c:898)",
 "upstream_fix_commit": "d7b30e457a3d",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5.x",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": "V5C-063"
 },
 "internal_id": "V5C-030"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-030.php` from that zip, byte for byte (sha256
`400f29ad266f4e2a419570cb196c0e733f27a00dbc97b6c3e638b6b3d4f46601`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_object_handlers.c:898` in
`zend_std_object_get_class`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:11` (rc -11).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `d7b30e457a3d`, the directory prefix `d7b30e457a3d`.
