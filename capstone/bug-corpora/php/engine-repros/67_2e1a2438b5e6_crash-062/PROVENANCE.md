# CRASH-062: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-062",
 "set": "core-166",
 "trigger": "corpus/CRASH-062.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "user-error-handler-recursive-compile-corrupts-CG-state-uaf",
 "c_file_line": "Zend/zend_execute_API.c:282 (premature zend_destroy_rsrc_list in shutdown_executor, run before Zend/zend.c:822 shutdown_compiler); claimed site Zend/zend.c:975 is real but is only the semantics bug",
 "upstream_fix_commit": "2e1a2438b5e6",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5.x",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": "V5C-146"
 },
 "internal_id": "V5C-062"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-062.php` from that zip, byte for byte (sha256
`8122d96ce062422e52dcc23cc6144568f98c553305ae2d2a35b1f30c551d9766`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `main/streams/streams.c:276` in
`_php_stream_free`, stockalloc `EXIT:255`, pristine `EXIT:255` (rc 255).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads EXIT:255, PHP's own fatal error, with no detector report. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'temporal-asan-visible', ours (gcc 15 ASan) in 'asan-blind' -- the stockalloc run decides, and that run's detector differs.

**Upstream fix.** php-src `2e1a2438b5e6`, the directory prefix `2e1a2438b5e6`.
