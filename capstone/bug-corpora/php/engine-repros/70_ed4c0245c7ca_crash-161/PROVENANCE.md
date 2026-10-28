# CRASH-161: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-161",
 "set": "core-166",
 "trigger": "corpus/CRASH-161.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-415",
 "category": "double-free",
 "engine_locus": "engine-core",
 "root_cause_id": "zval-copy-ctor-array-hands-over-hashtable-ownership-only-after-zend-hash-copy-bailout-leaves-two-zvals-owning-one-hashtable-double-free",
 "c_file_line": "Zend/zend_variables.c:151-152 (zend_hash_copy runs before `zvalue->value.ht = tmp_ht;`)",
 "upstream_fix_commit": "ed4c0245c7ca",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-roundE-core5(5.4-5.6)",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-182"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-161.php` from that zip, byte for byte (sha256
`a70dc1f49bb235f6caf99ba2c5283e8cf445b2632a32ae4e67875356399548af`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_hash.c:514` in
`zend_hash_destroy`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:11` (rc -11).
Hence **temporal-asan-visible**: a temporal defect (CWE-415) that ASan reports with zend_alloc.c as shipped (the stockalloc run, ASAN:SEGV) and that the patched build reports as ASAN:heap-use-after-free: the release reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** pristine outcome: corpus recorded 'OK', ours 'SIGNAL:11'; population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'neither', ours (gcc 15 ASan) in 'temporal-asan-visible' -- the stockalloc run decides, and that run's detector differs; crashes_on_pristine_build: corpus says False, our pristine run read SIGNAL:11.

**Upstream fix.** php-src `ed4c0245c7ca`, the directory prefix `ed4c0245c7ca`.
