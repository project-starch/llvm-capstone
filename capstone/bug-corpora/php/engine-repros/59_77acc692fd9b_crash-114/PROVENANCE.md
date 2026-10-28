# CRASH-114: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-114",
 "set": "core-166",
 "trigger": "corpus/CRASH-114.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "user-filter-bucket-double-append-missing-refcount-inc-uaf",
 "c_file_line": "ext/standard/user_filters.c:420",
 "upstream_fix_commit": "77acc692fd9b",
 "crashes_on_pristine_build": "True",
 "needs_build": "asan",
 "requires": "writable /tmp",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-stdext",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-114"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-114.php` from that zip, byte for byte (sha256
`4124230f262adb343ca6828b0d69196bc1b5e00eb7655deea5c17814251bc678`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `main/streams/streams.c:953` in
`_php_stream_write_filtered`, stockalloc `ASAN:SEGV`, pristine `SIGNAL:11` (rc -11).
Hence **temporal-asan-visible**: a temporal defect (CWE-416) that ASan reports with zend_alloc.c as shipped (the stockalloc run, ASAN:SEGV) and that the patched build reports as ASAN:heap-use-after-free: the release reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `77acc692fd9b`, the directory prefix `77acc692fd9b`.
