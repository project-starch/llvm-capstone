# CRASH-153: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-153",
 "set": "core-166",
 "trigger": "corpus/CRASH-153.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-457",
 "category": "uninitialised-read",
 "engine_locus": "engine-core",
 "root_cause_id": "hash-del-key-or-index-decrements-nnumofelements-after-destructor-stale-count-drives-uninitialised-pointer-deref",
 "c_file_line": "Zend/zend_hash.c:489-497 (defective ordering: pDestructor at :490, ht->nNumOfElements-- only at :497); faulting consumer ext/standard/array.c:1893-1894",
 "upstream_fix_commit": "86434be9462c",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-roundE-nofault-reaudit",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-174"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-153.php` from that zip, byte for byte (sha256
`5565bf9b3c44e7f34f9bc72bd485d6707e5aed73a105ea102593a9c5b7767130`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:SEGV` at `ext/standard/array.c:1894` in
`_phpi_pop`, stockalloc `OK`, pristine `SIGNAL:11` (rc -11).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK, so emalloc's cache or rounding hides it from any detector at free().

**Where our run and the corpus's record disagree:** population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'neither', ours (gcc 15 ASan) in 'asan-blind' -- the stockalloc run decides, and that run's detector differs.

**Upstream fix.** php-src `86434be9462c`, the directory prefix `86434be9462c`.
