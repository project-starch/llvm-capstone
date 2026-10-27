# CRASH-102: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-102",
 "set": "core-166",
 "trigger": "corpus/CRASH-102.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-125",
 "category": "heap-buffer-overflow",
 "engine_locus": "pure-ext",
 "root_cause_id": "strspn-strcspn-common-handler-unsigned-cast-clamp-truncation-oob-read",
 "c_file_line": "ext/standard/string.c:240 (fault surfaces at ext/standard/string.c:1332 in php_strspn)",
 "upstream_fix_commit": "29b9d79c2ff6",
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
 "internal_id": "V5C-102"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-102.php` from that zip, byte for byte (sha256
`7c46cf4623a97e0cf373cf91f8c951d801848026bad581ba501f627e0ab647ed`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-buffer-overflow` at `ext/standard/string.c:1332` in
`php_strspn`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK, so emalloc's cache or rounding hides it from any detector at free().

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `29b9d79c2ff6`, the directory prefix `29b9d79c2ff6`.
