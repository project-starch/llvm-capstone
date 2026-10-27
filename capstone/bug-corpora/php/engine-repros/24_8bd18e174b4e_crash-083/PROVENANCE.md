# CRASH-083: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-083",
 "set": "core-166",
 "trigger": "corpus/CRASH-083.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "call-user-func-params-into-argument-stack-invalidated-by-autoload-realloc-uaf",
 "c_file_line": "ext/standard/basic_functions.c:1885 (defect: full params[] fetched before zend_is_callable); fault observed at ext/standard/basic_functions.c:1902",
 "upstream_fix_commit": "8bd18e174b4e",
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
 "internal_id": "V5C-083"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-083.php` from that zip, byte for byte (sha256
`533020a5832702f5b844593f29e654c77eaaa5ba760c0515fc71e22b15612bde`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `ext/standard/basic_functions.c:1902` in
`zif_call_user_func`, stockalloc `OK`, pristine `SIGNAL:11` (rc -11).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** pristine outcome: corpus recorded 'OK', ours 'SIGNAL:11'; population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'temporal-asan-visible', ours (gcc 15 ASan) in 'asan-blind' -- the stockalloc run decides, and that run's detector differs; crashes_on_pristine_build: corpus says False, our pristine run read SIGNAL:11.

**Upstream fix.** php-src `8bd18e174b4e`, the directory prefix `8bd18e174b4e`.
