# CRASH-024: provenance

**Corpus.** `php-5.0.0-bug-corpus` (zip sha256 `31d22a3b468e96e0ff9f991d8667e78a7d5311db8767d7067bf8c76cbdf92064`), the collaborator's corpus of
185 PHP 5.0.0 defects, each a distinct root cause audited against the 5.0.0 source. Its record
for this case, verbatim:

```json
{
 "id": "CRASH-024",
 "set": "core-166",
 "trigger": "corpus/CRASH-024.php",
 "vuln_class": "memory-corruption",
 "cwe": "CWE-416",
 "category": "use-after-free",
 "engine_locus": "engine-core",
 "root_cause_id": "assign-ref-of-function-or-method-call-return-drops-live-this-uaf",
 "c_file_line": "Zend/zend_compile.c:487-508 (zend_do_assign_ref - missing `if (zend_is_function_or_method_call(rvar)) opline->extended_value = ZEND_RETURNS_FUNCTION;`) + Zend/zend_execute.c:2241-2248 (zend_assign_ref_handler - missing the corresponding fcall_returned_reference guard). NOTE: the claimed 'Zend/zend_compile.c:582' is wrong (that line is inside zend_do_for_before_statement) and 'Zend/zend_vm_def.h' does not exist at 5.0.0 - the VM was still hand-written zend_execute.c; zend_vm_def.h arrives in 5.1.",
 "upstream_fix_commit": "79d6f11a4a3f",
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
 "internal_id": "V5C-024"
}
```

**Trigger.** `trigger.php` is `corpus/CRASH-024.php` from that zip, byte for byte (sha256
`a7fd961e7a5b4087a08079009d2e6d9721eed8e6cbbd1d2760d09116507b8c58`).

**Oracle, ours.** Measured on p12, gcc 15.2.0, libasan from gcc 15, 2026-10-27 with the corpus's own `scripts/run-corpus.py`
against three builds of the pinned museum tarball (sha256
`5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919`) made by the corpus's
recipes: `asan` (its two `zend_alloc.c` detector patches, `-fsanitize=address`),
`stockalloc` (the same without those two patches: PHP's allocator as shipped under ASan), and
the plain build. `oracle.json` carries the result: ASan `ASAN:heap-use-after-free` at `Zend/zend_execute.c:59` in
`zend_pzval_lock_func`, stockalloc `OK`, pristine `OK` (rc 0).
Hence **asan-blind**: ASan reports it only with PHP's allocator patched (cache off, no 8-byte rounding); with zend_alloc.c as shipped the run reads OK. That is the selection, by measurement; which mechanism hides it, and which arm sees it, is what the arms record.

**Where our run and the corpus's record disagree:** nowhere -- ASan class, pristine outcome and the crash flag all agree.

**Upstream fix.** php-src `79d6f11a4a3f`, the directory prefix `79d6f11a4a3f`.
