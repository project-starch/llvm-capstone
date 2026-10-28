# CRASH-092: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-092",
 "set": "core-166",
 "trigger": "corpus/CRASH-092.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "pure-ext",
 "root_cause_id": "http-fopen-request-fulluri-separate-zval-then-ptr-dtor-frees-shared-context-option-uaf",
 "c_file_line": "ext/standard/http_fopen_wrapper.c:197-201 (free at :201). Claimed \":263\" is the line number in the 2005 fix commit's tree, NOT in php-5.0.0.",
 "upstream_fix_commit": "09881626498d",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "network: working loopback",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5x-stdext",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-092"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-092.php` from that zip, byte for byte (sha256
`0bb8dc6815347e53b384320d5fdbde23abb9f8236d99b4e234c2855abe6687ac`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute_API.c:389` in
`_zval_ptr_dtor`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `09881626498d`, the directory prefix `09881626498d`.
