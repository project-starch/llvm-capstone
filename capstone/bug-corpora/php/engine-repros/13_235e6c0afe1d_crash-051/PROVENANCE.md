# CRASH-051: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-051",
 "set": "core-166",
 "trigger": "corpus/CRASH-051.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-562",
 "category": "stack-use-after-return",
 "engine_locus": "engine-core",
 "root_cause_id": "std-call-user-call-stack-zval-args-zval_dtor-frees-retained-value-suar",
 "c_file_line": "Zend/zend_object_handlers.c:524 (+ :542, :587-588)",
 "upstream_fix_commit": "235e6c0afe1d",
 "crashes_on_pristine_build": "False",
 "needs_build": "asan",
 "requires": "",
 "how_collected": {
  "discovery_method": "commit-history-reconstruction",
  "origin_stream": "history-5.x",
  "history_status": "historical-known",
  "confidence": "high",
  "merged_members": ""
 },
 "internal_id": "V5C-051"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-051.php` from that zip, byte for byte (sha256
`922261f8d4a75556a1c5ebc382009dc71207280f6a64e2cee61ecee42ca56c4e`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:stack-use-after-return` at `Zend/zend_execute.c:59` in
`zend_pzval_lock_func`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** population: the collaborator's instrument (gcc 13 ASan, 2026-10-26) placed it in 'neither', ours (gcc 15 ASan) in 'asan-blind' -- the stockalloc run decides, and that run's detector differs.

**Upstream fix.** php-src `235e6c0afe1d`, the directory prefix `235e6c0afe1d`.
