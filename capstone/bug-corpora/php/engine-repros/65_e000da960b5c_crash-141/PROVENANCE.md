# CRASH-141: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-141",
 "set": "core-166",
 "trigger": "corpus/CRASH-141.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "call-user-function-byref-param-refcount1-skips-separation-binds-userland-reference-into-engine-owned-oparray-literal",
 "c_file_line": "Zend/zend_object_handlers.c:75 (with Zend/zend_execute_API.c:748-753)",
 "upstream_fix_commit": "e000da960b5c",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-core3",
  "history_status": "fixed-by-rewrite",
  "confidence": "medium",
  "merged_members": ""
 },
 "internal_id": "V5C-141"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-141.php` from that zip, byte for byte (sha256
`0c74aeada9d4356f50b23462104edc8d2b36027aeab49b592311fc37d207f35b`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute.c:59` in
`zend_pzval_lock_func`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:6` (rc -6).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** pristine outcome: corpus recorded 'SIGNAL:11', ours 'SIGNAL:6'.

**Upstream fix.** php-src `e000da960b5c`, the directory prefix `e000da960b5c`.
