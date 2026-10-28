# CRASH-031: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-031",
 "set": "core-166",
 "trigger": "corpus/CRASH-031.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "destructor-run-on-exception-failed-constructor-frees-live-object",
 "c_file_line": "Zend/zend_execute.c:2779",
 "upstream_fix_commit": "f5cf052225ff",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5.x",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-031"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-031.php` from that zip, byte for byte (sha256
`a506e22de5e9fefdab0781007872f0b21aa53fa968f74110ad5c8b1acb698dc4`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute.c:3853` in
`zend_fe_fetch_handler`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:11` (rc -11).
Hence **temporal-asan-visible**: a temporal defect (CWE-416) that ASan reports with zend_alloc.c as shipped (the stockalloc run, ASAN:SEGV) and that the patched build reports as ASAN:heap-use-after-free: the release reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `f5cf052225ff`, the directory prefix `f5cf052225ff`.
