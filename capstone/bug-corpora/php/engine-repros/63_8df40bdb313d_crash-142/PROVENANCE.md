# CRASH-142: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-142",
 "set": "core-166",
 "trigger": "corpus/CRASH-142.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "arg-stack-slots-not-nulled-during-arg-free-loop-reentrant-backtrace-reads-freed-arg-zvals-uaf",
 "c_file_line": "Zend/zend_execute.h:117-127 (defective free loop at :124); observed read of the stale slot at Zend/zend_builtin_functions.c:1420",
 "upstream_fix_commit": "8df40bdb313d",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-core3",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-142"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-142.php` from that zip, byte for byte (sha256
`fbe3f74a41a7aeef2ed0f5e0c0e380ddee97690738354b997b59525f58b7c831`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_builtin_functions.c:1420` in
`debug_backtrace_get_args`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:6` (rc -6).
Hence **temporal-asan-visible**: a temporal defect (CWE-416) that ASan reports with zend_alloc.c as shipped (the stockalloc run, ASAN:SEGV) and that the patched build reports as ASAN:heap-use-after-free: the release reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** pristine outcome: corpus recorded 'SIGNAL:11', ours 'SIGNAL:6'.

**Upstream fix.** php-src `8df40bdb313d`, the directory prefix `8df40bdb313d`.
