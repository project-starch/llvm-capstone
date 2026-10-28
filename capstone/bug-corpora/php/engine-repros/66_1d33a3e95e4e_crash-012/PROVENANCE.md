# CRASH-012: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-012",
 "set": "core-166",
 "trigger": "corpus/CRASH-012.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-590",
 "category": "invalid-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "globals-hashtable-destroy-efree-no-symtable-guard",
 "c_file_line": "ext/standard/array.c:2060",
 "upstream_fix_commit": "1d33a3e95e4e",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "carried-from-4.0.2",
  "origin_stream": "carried-from-4.0.2",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-012"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-012.php` from that zip, byte for byte (sha256
`5dc2ad7c2b55d9b29d38873287f51f2fcbb72f564b07e53963116614894f5e52`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:attempting free on address` at `Zend/zend_alloc.c:287` in
`_efree`, stockalloc `ASAN:attempting free on address`, pristine `SIGNAL:6` (rc -6).
Hence **temporal-asan-visible**: a temporal defect (CWE-590) that ASan reports with zend_alloc.c as shipped (the stockalloc run, ASAN:attempting free on address) and that the patched build reports as ASAN:attempting free on address: the release reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'neither', ours (gcc 15 ASan) in 'temporal-asan-visible' -- the stockalloc run decides, and that run's detector differs.

**Upstream fix.** php-src `1d33a3e95e4e`, the directory prefix `1d33a3e95e4e`.
