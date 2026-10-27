# CRASH-120: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-120",
 "set": "core-166",
 "trigger": "corpus/CRASH-120.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-125",
 "category": "out-of-bounds-read",
 "engine_locus": "pure-ext",
 "root_cause_id": "unserialize-object_common1-cursor-advance-past-truncated-buffer-oob-read",
 "c_file_line": "ext/standard/var_unserializer.c:208-210 (defect, object_common1) == ext/standard/var_unserializer.re:213-215; OOB read surfaces at ext/standard/var_unserializer.c:195 (finish_nested_data)",
 "upstream_fix_commit": "16b3003ffc63",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-stdext",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-120"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-120.php` from that zip, byte for byte (sha256
`2bc5c92fd04f881a2d317778f870eb54c666aa7b6c0caa616c0383402b402f7c`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-buffer-overflow` at `ext/standard/var_unserializer.c:195` in
`finish_nested_data`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'neither', ours (gcc 15 ASan) in 'asan-blind' -- the stockalloc run decides, and that run's detector differs.

**Upstream fix.** php-src `16b3003ffc63`, the directory prefix `16b3003ffc63`.
