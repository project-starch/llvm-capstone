# CRASH-040: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-040",
 "set": "core-166",
 "trigger": "corpus/CRASH-040.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "hash-apply-deleter-runs-destructor-before-unlinking-bucket",
 "c_file_line": "Zend/zend_execute.c:69 (push) + Zend/zend_execute.c:76 (drain)  [claimed site Zend/zend_hash.c:573-575 is present verbatim but unexercised]",
 "upstream_fix_commit": "c42b3bf68938",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5.x",
  "history_status": "fixed-by-rewrite",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-040"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-040.php` from that zip, byte for byte (sha256
`a57183fbd3624308849237b87be5968e94a94ab4bad38eb8c70eadc53ff41f03`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute_API.c:389` in
`_zval_ptr_dtor`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:11` (rc -11).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `c42b3bf68938`, the directory prefix `c42b3bf68938`.
