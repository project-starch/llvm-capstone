# CRASH-068: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-068",
 "set": "core-166",
 "trigger": "corpus/CRASH-068.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "register-constant-redefine-frees-name-before-error-message-uaf",
 "c_file_line": "Zend/zend_constants.c:320",
 "upstream_fix_commit": "1ea22c90046bc51543dd5a606ff83823056654a1",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "fuzzing",
  "origin_stream": "fuzz",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-068"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-068.php` from that zip, byte for byte (sha256
`c4159f6020bf5fa7a92f331977dbafaacc389d131c900a73e3c86af18a852409`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `main/spprintf.c:518` in
`xbuf_format_converter`, stockalloc `ASAN:heap-use-after-free`, pristine `OK` (rc 0).
Hence **temporal-asan-visible**: a CWE-416 that ASan reports with zend_alloc.c as shipped: efree reached free(), so a detector at the libc boundary sees it; the control population.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `1ea22c90046bc51543dd5a606ff83823056654a1`, the directory prefix `1ea22c90046b`.
